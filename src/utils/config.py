"""
Configuration loader cho pipeline.
Load settings từ YAML files.
"""

import yaml
from pathlib import Path
from typing import Any

def load_config(config_path: str) -> dict[str, Any]:
    """
    Load configuration từ YAML file.
    
    Args:
        config_path: Đường dẫn đến file .yaml
    
    Returns:
        Dict chứa configuration
    """

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config

def get_project_root() -> Path:
    """Trả về project root directory."""

    return Path(__file__).parent.parent.parent

def get_data_path() -> Path:
    """Trả về data directory."""

    return get_project_root() / "data"

def get_annotations_path(filename: str) -> Path:
    """Trả về đường dẫn đến annotation file."""

    return get_data_path() / "raw" / "annotations" / filename

DEFAULT_CONFIG = {
    "dataset": {
        "train_annotations": "bdd100k_labels_images_train.json",
        "val_annotations": "bdd100k_labels_images_val.json",
    },
    "active_learning": {
        "initial_budget": 1000, # Số ảnh ban đầu
        "batch_size": 500, # Số ảnh mỗi vòng
        "num_rounds": 10, # Số vòng
        "num_classes": 10, # Số class
    },
    "sampling": {
        "stratify_fields": ["weather", "scene", "timeofday"],
    },
    "model": {
        "name": "yolov8n",
        "device": "cuda"
    }
}