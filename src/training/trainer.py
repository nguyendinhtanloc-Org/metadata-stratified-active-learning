"""
YOLO Trainer cho Active Learning.
Quản lý việc train/re-train model với labeled data.
"""

from pathlib import Path
from typing import Optional, List, Dict, Union
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import shutil
import yaml
import torch
import gc

from data.loader import YOLO_CLASS_NAMES, labels_to_yolo_lines


class YOLOTrainer:
    """
    Trainer class cho YOLO model trong Active Learning loop.
    """

    def __init__ (
        self,
        model_name: str = "yolov8n",
        device = 0,
        project_dir: str = "experiments/runs",
        train_seed = None
    ):
        """
        Khởi tạo trainer.

        Args:
            model_name: Tên YOLO model (yolov8n, yolov8s, yolov8m, ...)
            device: Thiết bị train ("cuda" hoặc "cpu")
            project_dir: Thư mục lưu experiment results
            train_seed: Seed cho ultralytics training. None = để ultralytics
                tự dùng (mặc định 0 - tức mọi experiment giống nhau).
        """

        self.model_name = model_name
        self.device = device
        self.project_dir = Path(project_dir)
        self.model = None
        self.best_model_path = None
        self.pretrained_path = f"{self.model_name}.pt"
        self.train_seed = train_seed
        self._last_eval_yaml = None

        # Verify GPU availability
        if device != "cpu" and not self._check_cuda():
            raise RuntimeError(
                f"CUDA device requested but not available. "
                f"Either install CUDA drivers or set device='cpu' in config."
            )

    def load_model(self, model_path: Optional[str] = None):
        """
        Load YOLO model.
        
        Args:
            model_path: Đường dẫn checkpoint. Nếu None, load pretrained.
        """

        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path) 
        else:
            # Load pretrained nhưng KHÔNG download lại nếu đã có
            if Path(self.pretrained_path).exists():
                self.model = YOLO(self.pretrained_path)
            else:
                self.model = YOLO(f"{self.model_name}.pt")

    def _check_cuda(self) -> bool:
        """Kiểm tra CUDA có khả dụng không."""
        try:
            return torch.cuda.is_available() and torch.cuda.device_count() > 0
        except ImportError:
            return False

    def prepare_dataset_with_metadata(
        self,
        image_paths: List[str],
        metadata_list: List[Dict],
        output_dir: str = "data/processed/yolo_dataset",
        split_seed: int = 42,
        val_ratio: float = 0.2,
        eval_paths: List[str] = None,
        eval_metadata: List[Dict] = None,
        shared_eval_dir: str = None,
    ) -> str:
        """
        Chuẩn bị dataset từ metadata (labels nằm trong metadata, không phải file riêng).

        Chia làm 3 phần:
        - train: ảnh labeled dùng để train (trừ val nội bộ)
        - val:   20% ảnh labeled, dùng cho early stopping / chọn best weights
        - eval:  fixed held-out set (nếu có), dùng để BÁO CÁO mAP

        Vì sao cần eval set cố định:
        Nếu mAP đo trên val set rút ngẫu nhiên từ labeled pool mỗi round,
        thì mAP ở các round khác nhau đo trên ảnh KHÁC NHAU -> không so sánh được,
        và hơn nữa là rò rỉ (ảnh có thể nằm trong train của round sau).
        Eval set cố định giữ mAP comparable giữa các round, các seed, các pipeline.

        Args:
            val_ratio: Tỉ lệ ảnh labeled dành cho validation nội bộ.
            eval_paths: Đường dẫn ảnh eval set cố định (dùng làm `val` trong yaml).
            eval_metadata: Metadata tương ứng với eval_paths.
        """

        output_path = Path(output_dir)
        images_train = output_path / "images" / "train"
        images_val = output_path / "images" / "val"
        labels_train = output_path / "labels" / "train"
        labels_val = output_path / "labels" / "val"

        for d in [images_train, images_val, labels_train, labels_val]:
            d.mkdir(parents=True, exist_ok=True)

        # Sort theo tên file trước khi split để val set ỔN ĐỊNH giữa các round.
        # Nếu không sort, thứ tự image_paths thay đổi mỗi round (do set ordering)
        # -> train_test_split chọn ảnh val khác mỗi vòng -> mAP không comparable.
        paired = sorted(
            zip(image_paths, metadata_list),
            key=lambda p: str(p[0])
        )
        sorted_paths = [p[0] for p in paired]
        sorted_metadata = [p[1] for p in paired]

        train_indices, val_indices = train_test_split(
            range(len(sorted_paths)),
            test_size=val_ratio,
            random_state=split_seed
        )

        for idx in train_indices:
            image_path = sorted_paths[idx]
            metadata = sorted_metadata[idx]
            image_name = Path(image_path).name
            shutil.copy(image_path, images_train / image_name)
            yolo_lines = labels_to_yolo_lines(metadata.get("labels", []))
            if yolo_lines:
                label_file = labels_train / f"{Path(image_path).stem}.txt"
                with open(label_file, "w") as f:
                    f.write("\n".join(yolo_lines))

        for idx in val_indices:
            image_path = sorted_paths[idx]
            metadata = sorted_metadata[idx]
            image_name = Path(image_path).name
            shutil.copy(image_path, images_val / image_name)
            yolo_lines = labels_to_yolo_lines(metadata.get("labels", []))
            if yolo_lines:
                label_file = labels_val / f"{Path(image_path).stem}.txt"
                with open(label_file, "w") as f:
                    f.write("\n".join(yolo_lines))

        # Ghi eval set (nếu có) thành dataset RIÊNG.
        # QUAN TRỌNG: dataset.yaml vẫn trỏ `val` về val nội bộ (20% ảnh labeled)
        # để early stopping / chọn best weights KHÔNG bị rò rỉ. Sau khi train xong
        # sẽ gọi evaluate trên eval.yaml để lấy mAP báo cáo - mAP này đo trên tập
        # CỐ ĐỊNH nên comparable giữa các round/seed/pipeline.
        eval_yaml_path = None
        if eval_paths and eval_metadata:
            # Eval set được COPY 1 LẦN DUY NHẤT vào thư mục dùng chung,
            # dùng lại cho mọi round và mọi experiment -> tiết kiệm ~3.6GB disk
            # và rất nhiều thời gian I/O copy.
            eval_root = Path(shared_eval_dir or (output_path / "eval_set"))
            images_eval = eval_root / "images"
            labels_eval = eval_root / "labels"
            images_eval.mkdir(parents=True, exist_ok=True)
            labels_eval.mkdir(parents=True, exist_ok=True)

            manifest = eval_root / "manifest.txt"
            expected = len(eval_paths)
            already_done = False
            if manifest.exists() and images_eval.exists():
                try:
                    already_done = int(manifest.read_text().strip()) == expected
                except Exception:
                    already_done = False

            if already_done:
                print(f"[EVAL] Reuse eval set đã copy sẵn ({expected} ảnh) tại {eval_root}")
            else:
                # Xóa nội dung cũ nếu có (tránh ảnh sót lại từ lần chạy trước)
                for stale in images_eval.glob("*"):
                    if stale.is_file():
                        stale.unlink()
                for stale in labels_eval.glob("*"):
                    if stale.is_file():
                        stale.unlink()

                for image_path, metadata in zip(eval_paths, eval_metadata):
                    image_name = Path(image_path).name
                    shutil.copy(image_path, images_eval / image_name)
                    yolo_lines = labels_to_yolo_lines(metadata.get("labels", []))
                    if yolo_lines:
                        label_file = labels_eval / f"{Path(image_path).stem}.txt"
                        with open(label_file, "w") as f:
                            f.write("\n".join(yolo_lines))

                manifest.write_text(str(expected))
                print(f"[EVAL] Copied {expected} eval images to {eval_root}")

            # Dataset riêng cho eval: train và val cùng trỏ về eval images
            eval_config = {
                "path": str(eval_root.absolute()),
                "train": "images",
                "val": "images",
                "names": YOLO_CLASS_NAMES
            }
            eval_yaml_path = eval_root / "eval.yaml"
            with open(eval_yaml_path, "w") as f:
                yaml.dump(eval_config, f)

        dataset_config = {
            "path": str(output_path.absolute()),
            "train": "images/train",
            "val": "images/val",
            "names": YOLO_CLASS_NAMES
        }

        yaml_path = output_path / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(dataset_config, f)

        self._last_eval_yaml = eval_yaml_path

        return str(yaml_path)
    
    def train(
        self,
        dataset_yaml: str,
        epochs: int = 10,
        batch_size: int = 16,
        imgsz: int = 640,
        save_name: str = "train",
        patience: int = 0,
        ) -> dict:
        """
        Train YOLO model.

        Args:
            patience: Early stopping patience (số epoch không cải thiện thì dừng).
                0 = tắt early stopping (train đủ số epoch).
        """

        if self.model is None:
            if self.best_model_path and Path(self.best_model_path).exists():
                self.load_model(self.best_model_path)
            else:
                self.load_model()

        train_kwargs = dict(
            data = dataset_yaml,
            epochs = epochs,
            batch = batch_size,
            imgsz = imgsz,
            device = self.device,
            project = self.project_dir,
            name = save_name,
            exist_ok = True,
            verbose = False,
            # --- Tối ưu tốc độ (chất lượng gần như không đổi) ---
            amp = True,           # mixed precision FP16: nhanh ~2x trên T4
            workers = 8,          # dataloader workers
            cache = False,         # không cache vào RAM (tiết kiệm RAM Kaggle)
            plots = False,         # bỏ vẽ chart -> tiết kiệm thời gian + disk
            val = True,
        )

        # Seed cho ultralytics: nếu không set, ultralytics dùng seed=0 cho MỌI
        # experiment -> 3 "seed" khác nhau chỉ khác ở bước AL selection, phần
        # training giống hệt nhau. Set seed để mỗi replicate thực sự khác nhau.
        if self.train_seed is not None:
            train_kwargs["seed"] = self.train_seed

        # Early stopping: patience > 0 thì ultralytics tự dừng khi mAP không tăng
        if patience and patience > 0:
            train_kwargs["patience"] = patience

        results = self.model.train(**train_kwargs)

        self.best_model_path = str(Path(self.project_dir) / save_name / "weights" / "best.pt")

        # Đánh giá trên FIXED eval set (nếu có) - đây là mAP dùng để báo cáo.
        # Val nội bộ (20% labeled) chỉ dùng cho early stopping / chọn weights.
        eval_metrics = {}
        eval_yaml = getattr(self, "_last_eval_yaml", None)
        if eval_yaml and Path(eval_yaml).exists():
            try:
                eval_metrics = self.evaluate(eval_yaml)
                print(f"[EVAL] Fixed eval set mAP50={eval_metrics['map50']:.4f} "
                      f"mAP50-95={eval_metrics['map50_95']:.4f}")
            except Exception as e:
                print(f"[WARNING] Eval on fixed eval set failed: {e}")

        return {
            "best_model_path": self.best_model_path,
            "map50": results.results_dict.get("metrics/mAP50(B)", 0.0),
            "map50_95": results.results_dict.get("metrics/mAP50-95(B)", 0.0),
            "epochs_run": getattr(results, "epochs", epochs),
            # Metric chính để so sánh giữa các round/seed/pipeline
            "eval_map50": eval_metrics.get("map50", 0.0),
            "eval_map50_95": eval_metrics.get("map50_95", 0.0),
            "eval_map75": eval_metrics.get("map75", 0.0),
            "eval_precision": eval_metrics.get("precision", 0.0),
            "eval_recall": eval_metrics.get("recall", 0.0),
        }

    def evaluate(self, dataset_yaml: str, weights: str = None) -> dict:
        """Evaluate model trên một dataset yaml.

        Args:
            dataset_yaml: Đường dẫn dataset yaml
            weights: Đường dẫn weights cụ thể (mặc định dùng model hiện tại).
                Dùng best.pt để đánh giá đúng model được chọn.
        """
        if weights:
            from ultralytics import YOLO
            eval_model = YOLO(weights)
        else:
            if self.model is None:
                raise ValueError("Model chưa được load.")
            eval_model = self.model

        metrics = eval_model.val(
            data = dataset_yaml,
            device = self.device,
            verbose = False,
            plots = False,
        )

        return {
            "map50": metrics.box.map50,
            "map75": metrics.box.map75,
            "map50_95": metrics.box.map,
            "precision": metrics.box.mp,
            "recall": metrics.box.mr
        }

    def predict(self, image_path: str, **kwargs):
        """Predict trên một ảnh."""
        if self.model is None:
            raise ValueError("Model chưa được load.")
        return self.model.predict(source=image_path, **kwargs)

    def predict_batch(self, image_paths: List[str], batch_size: int = 8, show_progress: bool = True) -> List:
        """
        Predict trên nhiều ảnh cùng lúc với batch size nhỏ.
        
        Args:
            image_paths: Danh sách đường dẫn ảnh
            batch_size: Số ảnh xử lý mỗi batch (NHỎ để tránh OOM)
            show_progress: Hiển thị progress bar
        
        Returns:
            List các YOLO Results objects
        """
        if self.model is None:
            raise ValueError("Model chưa được load.")
        
        results = []
        total = len(image_paths)
        
        for i in range(0, total, batch_size):
            batch = image_paths[i:i + batch_size]
            
            # Batch predict với batch_size NHỎ
            batch_results = self.model.predict(
                source=batch,
                batch=batch_size,
                verbose=False
            )
            
            # Nếu batch chỉ có 1 ảnh, ultralytics trả về 1 object thay vì list
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
            
            # Dọn memory SAU MỖI BATCH
            if i % 500 == 0 and i > 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if show_progress and (i + batch_size) % 2000 == 0:
                print(f"      Predicted {min(i + batch_size, total)}/{total} images")
        
        return results
