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
    """Trainer class cho YOLO model trong Active Learning loop."""

    def __init__(
        self,
        model_name: str = "yolov8n",
        device=0,
        project_dir: str = "experiments/runs",
        train_seed=None,
        retrain_from_scratch: bool = True,
        warmup_epochs: float = 1.0,
    ):
        self.model_name = model_name
        self.device = device
        self.project_dir = Path(project_dir)
        self.model = None
        self.best_model_path = None
        self.pretrained_path = f"{self.model_name}.pt"
        self.train_seed = train_seed
        self.retrain_from_scratch = retrain_from_scratch
        self.warmup_epochs = warmup_epochs
        self._last_eval_yaml = None

        if device != "cpu" and not self._check_cuda():
            raise RuntimeError(
                "CUDA device requested but not available. "
                "Either install CUDA drivers or set device='cpu' in config."
            )

    def load_model(self, model_path: Optional[str] = None):
        """Load YOLO model."""
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            if Path(self.pretrained_path).exists():
                self.model = YOLO(self.pretrained_path)
            else:
                self.model = YOLO(f"{self.model_name}.pt")

    def _check_cuda(self) -> bool:
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
        output_path = Path(output_dir)
        images_train = output_path / "images" / "train"
        images_val = output_path / "images" / "val"
        labels_train = output_path / "labels" / "train"
        labels_val = output_path / "labels" / "val"

        # Xóa thư mục train/val cũ của round nếu chạy đè để tránh trộn ảnh rác
        for d in [images_train, images_val, labels_train, labels_val]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        paired = sorted(zip(image_paths, metadata_list), key=lambda p: str(p[0]))
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

        eval_yaml_path = None
        if eval_paths and eval_metadata:
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
        epochs: int = 12,
        batch_size: int = 16,
        imgsz: int = 640,
        save_name: str = "train",
        patience: int = 0,
    ) -> dict:
        """
        Train YOLO model.
        Khi retrain_from_scratch=True (mặc định chuẩn cho Active Learning), mỗi round
        khởi tạo lại từ pretrained yolov8n.pt gốc và train trên toàn bộ tập labeled
        tích lũy để tránh catastrophic forgetting và tránh overfit vào ảnh Round 0.
        """
        if self.retrain_from_scratch or self.model is None:
            self.load_model(None)  # Luôn nạp lại yolov8n.pt gốc
        elif self.best_model_path and Path(self.best_model_path).exists():
            self.load_model(self.best_model_path)
        else:
            self.load_model(None)

        train_kwargs = dict(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=self.device,
            project=self.project_dir,
            name=save_name,
            exist_ok=True,
            verbose=False,
            amp=True,
            workers=8,
            cache=False,
            plots=False,
            val=True,
            warmup_epochs=self.warmup_epochs,  # Giảm warmup xuống 1.0 epoch
            cos_lr=True,                       # Cosine LR scheduler giúp hội tụ mượt hơn
        )

        if self.train_seed is not None:
            train_kwargs["seed"] = self.train_seed

        if patience and patience > 0:
            train_kwargs["patience"] = patience

        results = self.model.train(**train_kwargs)

        best_pt_candidates = []
        trainer_obj = getattr(self.model, "trainer", None)
        if trainer_obj is not None and getattr(trainer_obj, "save_dir", None):
            best_pt_candidates.append(Path(trainer_obj.save_dir) / "weights" / "best.pt")
        try:
            from ultralytics.utils import SETTINGS
            runs_dir = Path(SETTINGS.get("runs_dir", "runs/detect")).expanduser()
            best_pt_candidates.append(
                runs_dir / self.project_dir / save_name / "weights" / "best.pt"
            )
        except Exception:
            pass
        best_pt_candidates.append(Path(self.project_dir) / save_name / "weights" / "best.pt")

        found = next((c for c in best_pt_candidates if c.exists()), None)
        self.best_model_path = str(found) if found else None

        # Load lại đúng best.pt vừa tìm được vào self.model để bước predict round sau chuẩn xác
        if self.best_model_path:
            self.load_model(self.best_model_path)

        # Đánh giá trên FIXED eval set (2000 ảnh)
        eval_metrics = {}
        eval_yaml = getattr(self, "_last_eval_yaml", None)
        if eval_yaml and Path(eval_yaml).exists():
            try:
                eval_metrics = self.evaluate(eval_yaml, weights=self.best_model_path)
                print(f"[EVAL] Fixed eval set mAP50={eval_metrics['map50']:.4f} "
                      f"mAP50-95={eval_metrics['map50_95']:.4f}")
            except Exception as e:
                print(f"[WARNING] Eval on fixed eval set failed: {e}")

        return {
            "best_model_path": self.best_model_path,
            "map50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
            "map50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
            "epochs_run": getattr(results, "epochs", epochs),
            "eval_map50": float(eval_metrics.get("map50", 0.0)),
            "eval_map50_95": float(eval_metrics.get("map50_95", 0.0)),
            "eval_map75": float(eval_metrics.get("map75", 0.0)),
            "eval_precision": float(eval_metrics.get("precision", 0.0)),
            "eval_recall": float(eval_metrics.get("recall", 0.0)),
            "per_class_ap50": eval_metrics.get("per_class_ap50", {}),
            "per_class_ap50_95": eval_metrics.get("per_class_ap50_95", {}),
        }

    def evaluate(self, dataset_yaml: str, weights: str = None) -> dict:
        """Evaluate model trên dataset yaml và trích xuất luôn Per-class AP cho RQ2."""
        if weights and Path(weights).exists():
            eval_model = YOLO(weights)
        else:
            if self.model is None:
                raise ValueError("Model chưa được load.")
            eval_model = self.model

        metrics = eval_model.val(
            data=dataset_yaml,
            device=self.device,
            verbose=False,
            plots=False,
        )

        # Trích xuất Per-class AP50 và AP50-95 để trả lời Câu hỏi phụ RQ2 trong Proposal
        per_class_ap50 = {}
        per_class_ap50_95 = {}
        try:
            ap50_arr = getattr(metrics.box, "ap50", None)
            maps_arr = getattr(metrics.box, "maps", None)
            ap_class_index = getattr(metrics.box, "ap_class_index", None)

            if isinstance(YOLO_CLASS_NAMES, dict):
                class_names = [YOLO_CLASS_NAMES[k] for k in sorted(YOLO_CLASS_NAMES.keys())]
            else:
                class_names = list(YOLO_CLASS_NAMES)

            if ap50_arr is not None and ap_class_index is not None:
                for idx_pos, cls_id in enumerate(ap_class_index):
                    cname = class_names[int(cls_id)] if int(cls_id) < len(class_names) else f"class_{cls_id}"
                    per_class_ap50[cname] = round(float(ap50_arr[idx_pos]), 4)

            if maps_arr is not None:
                for cls_id, val in enumerate(maps_arr):
                    cname = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                    per_class_ap50_95[cname] = round(float(val), 4)
        except Exception as e:
            print(f"[WARN] Không trích xuất được per-class AP: {e}")

        return {
            "map50": float(metrics.box.map50),
            "map75": float(metrics.box.map75),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "per_class_ap50": per_class_ap50,
            "per_class_ap50_95": per_class_ap50_95,
        }

    def predict(self, image_path: str, **kwargs):
        if self.model is None:
            raise ValueError("Model chưa được load.")
        return self.model.predict(source=image_path, **kwargs)