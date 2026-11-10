"""
Dataset module cho BDD100K trong Active Learning.
Cung cấp BDD100KDataset wrapper và tái xuất các tiện ích nạp dữ liệu từ data.loader.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
from src.data.loader import (
    BDD100K_TO_YOLO,
    YOLO_CLASS_NAMES,
    load_raw_annotations,
    convert_box2d_to_yolo,
    find_image_path,
)


class BDD100KDataset:
    """
    Quản lý tập dữ liệu BDD100K có gắn kèm metadata cho Active Learning.
    Hỗ trợ truy xuất ảnh, nhãn bounding box YOLO và thông tin môi trường.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        image_dir: Optional[str] = None
    ):
        """
        Args:
            samples: Danh sách metadata dictionaries của từng ảnh
            image_dir: Thư mục chứa ảnh gốc (tùy chọn)
        """
        self.samples = samples
        self.image_dir = Path(image_dir) if image_dir else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        image_name = item.get("name")
        image_path = None
        if self.image_dir and image_name:
            image_path = self.image_dir / image_name

        return {
            "name": image_name,
            "path": str(image_path) if image_path else item.get("image_path"),
            "attributes": item.get("attributes", {}),
            "weather": item.get("weather", "unknown"),
            "scene": item.get("scene", "unknown"),
            "timeofday": item.get("timeofday", "unknown"),
            "labels": item.get("labels", []),
        }

    def get_metadata_list(self) -> List[Dict[str, Any]]:
        """Trả về toàn bộ danh sách metadata phục vụ phân tầng AL."""
        return self.samples


__all__ = [
    "BDD100KDataset",
    "BDD100K_TO_YOLO",
    "YOLO_CLASS_NAMES",
    "load_raw_annotations",
    "convert_box2d_to_yolo",
    "find_image_path",
]

