"""
YOLO Trainer cho Active Learning.
Quản lý việc train/re-train model với labeled data.
"""

from pathlib import Path
from typing import Optional, List
from ultralytics import YOLO
import shutil
import yaml
import torch

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

    def load_model(self, model_path: Optional[str] = None):
        """
        Load YOLO model.
        
        Args:
            model_path: Đường dẫn checkpoint. Nếu None, load pretrained.
        """

        if model_path and Path(model_path).exists():
            # load từ checkpoint
            self.model = YOLO(model_path) 
        else:
            # Load pretrained model
            self.model = YOLO(f"{self.model_name}.pt")

    def _check_cuda(self) -> bool:
        """Kiểm tra CUDA có khả dụng không."""

        try:
            return torch.cuda.is_available()
        except ImportError:
            return False

    def prepare_dataset(
        self,
        image_paths: List[str],
        labels_dir: str,
        output_dir: str = "data/processed/yolo_dataset" 
    ) -> str:
        """
        Chuẩn bị dataset theo format YOLO.
        
        Args:
            image_paths: Danh sách đường dẫn ảnh đã label
            labels_dir: Thư mục chứa labels (YOLO format)
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
        n_train = int(len(image_paths) * 0.8)
        train_paths = image_paths[:n_train]
        val_paths= image_paths[n_train:]

        # Link/copy ảnh và labels cho train
        for image_path in train_paths:
            image_name = Path(image_path).name
            # copy ảnh
            shutil.copy(image_path, images_train / image_name)
            # copy label (nếu có)
            label_path = Path(labels_dir) / f"{Path(image_path).stem}.txt"

            if label_path.exists():
                shutil.copy(label_path, labels_train / f"{Path(image_path).stem}.txt")

        for image_path in val_paths:
            image_name = Path(image_path).name
            shutil.copy(image_path, images_val / image_name)
            label_path = Path(labels_dir) / f"{Path(image_path).stem}.txt"
            if label_path.exists():
                shutil.copy(label_path, labels_val / f"{Path(image_path).stem}.txt")

        # tạo dataset.yaml
        dataset_config = {
            "path": str(output_path.absolute()),
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "pedestrian",
                1: "rider",
                2: "other person",
                3: "bicycle",
                4: "car",
                5: "bus",
                6: "truck",
                7: "motorcycle",
                8: "traffic light",
                9: "traffic sign"
            }
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

        if self.model is None:
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

        