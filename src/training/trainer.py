"""
YOLO Trainer cho Active Learning.
Quản lý việc train/re-train model với labeled data.
"""

from pathlib import Path
from typing import Optional, List, Dict
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import shutil
import yaml
import torch

from data.loader import YOLO_CLASS_NAMES, labels_to_yolo_lines


class YOLOTrainer:
    """
    Trainer class cho YOLO model trong Active Learning loop.
    """

    def __init__ (
        self,
        model_name: str = "yolov8n",
        device = 0,
        project_dir: str = "experiments/runs"
    ):
        """
        Khởi tạo trainer.
        
        Args:
            model_name: Tên YOLO model (yolov8n, yolov8s, yolov8m, ...)
            device: Thiết bị train ("cuda" hoặc "cpu")
            project_dir: Thư mục lưu experiment results
        """

        self.model_name = model_name
        self.device = device
        self.project_dir = Path(project_dir)
        self.model = None
        self.best_model_path = None
        self.pretrained_path = f"{self.model_name}.pt"

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
            return torch.cuda.is_available()
        except ImportError:
            return False

    def prepare_dataset_with_metadata(
        self,
        image_paths: List[str],
        metadata_list: List[Dict],
        output_dir: str = "data/processed/yolo_dataset",
        split_seed: int = 42
    ) -> str:
        """
        Chuẩn bị dataset từ metadata (labels nằm trong metadata, không phải file riêng).

        Args:
            image_paths: Danh sách đường dẫn ảnh
            metadata_list: Danh sách metadata tương ứng
            output_dir: Thư mục output

        Returns:
            Đường dẫn đến dataset.yaml
        """

        output_path = Path(output_dir)
        images_train = output_path / "images" / "train"
        images_val = output_path / "images" / "val"
        labels_train = output_path / "labels" / "train"
        labels_val = output_path / "labels" / "val"

        # Tạo directories
        for d in [images_train, images_val, labels_train, labels_val]:
            d.mkdir(parents=True, exist_ok=True)

        # split train/test (80/20)
        train_indices, val_indices = train_test_split(
            range(len(image_paths)),
            test_size=0.2,
            random_state=split_seed
        )

        # Copy ảnh và tạo labels cho train
        for idx in train_indices:
            image_path = image_paths[idx]
            metadata = metadata_list[idx]
            image_name = Path(image_path).name

            # copy ảnh
            shutil.copy(image_path, images_train / image_name)

            # tạo label file từ metadata
            yolo_lines = labels_to_yolo_lines(metadata.get("labels", []))
            if yolo_lines:
                label_file = labels_train / f"{Path(image_path).stem}.txt"
                with open(label_file, "w") as f:
                    f.write("\n".join(yolo_lines))

        # Copy ảnh và tạo labels cho val
        for idx in val_indices:
            image_path = image_paths[idx]
            metadata = metadata_list[idx]
            image_name = Path(image_path).name

            shutil.copy(image_path, images_val / image_name)

            yolo_lines = labels_to_yolo_lines(metadata.get("labels", []))
            if yolo_lines:
                label_file = labels_val / f"{Path(image_path).stem}.txt"
                with open(label_file, "w") as f:
                    f.write("\n".join(yolo_lines))

        # tạo dataset.yaml
        dataset_config = {
            "path": str(output_path.absolute()),
            "train": "images/train",
            "val": "images/val",
            "names": YOLO_CLASS_NAMES
        }

        yaml_path = output_path / "dataset.yaml"

        with open(yaml_path, "w") as f:
            yaml.dump(dataset_config, f)

        return str(yaml_path)
    
    def train(
        self,
        dataset_yaml: str,
        epochs: int = 10,
        batch_size: int = 16,
        imgsz: int = 640,
        save_name: str = "train"
        ) -> dict:
        """
        Train YOLO model.
        
        Args:
            dataset_yaml: Đường dẫn đến dataset.yaml
            epochs: Số epochs
            batch_size: Batch size
            imgsz: Image size
            save_name: Tên experiment
        
        Returns:
            Dict chứa metrics kết quả
        """

        # Load pretrained model CHỈ MỘT LẦN
        if self.model is None:
            # Round 0: dùng pretrained
            # Round N (N>0): dùng best model từ round trước
            if self.best_model_path and Path(self.best_model_path).exists():
                self.load_model(self.best_model_path)
            else:
                self.load_model()

        # run training
        results = self.model.train(
            data = dataset_yaml,
            epochs = epochs,
            batch = batch_size,
            imgsz = imgsz,
            device = self.device,
            project = self.project_dir,
            name = save_name,
            exist_ok = True,
            verbose = False
        )

        # lưu best model path
        self.best_model_path = str(Path(self.project_dir) / save_name / "weights" / "best.pt")

        return {
            "best_model_path": self.best_model_path,
            "map50": results.results_dict.get("metrics/mAP50(B)", 0.0),
            "map50_95": results.results_dict.get("metrics/mAP50-95(B)", 0.0),
        }

    def evaluate(self, dataset_yaml: str) -> dict:
        """
        Evaluate model trên validation set.
        
        Args:
            dataset_yaml: Đường dẫn đến dataset.yaml
        
        Returns:
            Dict chứa evaluation metrics
        """

        if self.model is None:
            raise ValueError("Model chưa được load. Gọi load_model() trước.")

        metrics = self.model.val(
            data = dataset_yaml,
            device = self.device,
            verbose = False
        )

        return {
            "map50": metrics.box.map50,
            "map75": metrics.box.map75,
            "map50_95": metrics.box.map,
            "precision": metrics.box.mp,
            "recall": metrics.box.mr
        }

    def predict(self, image_path: str, **kwargs):
        """
        Predict trên một ảnh.
        
        Args:
            image_path: Đường dẫn ảnh
            **kwargs: Các tham số khác cho predict
        
        Returns:
            YOLO Results object
        """

        if self.model is None:
            raise ValueError("Model chưa được load. Gọi load_model() trước.")
        
        return self.model.predict(source=image_path, **kwargs)

        