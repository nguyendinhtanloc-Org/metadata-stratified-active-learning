"""
Data loader cho BDD100K dataset.
Load annotations và trích xuất metadata fields.
Convert box2d → YOLO format.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple

# BDD100K → YOLO class mapping
BDD100K_TO_YOLO = {
    'pedestrian': 0,
    'rider': 1,
    'person': 0,       # Map 'person' → pedestrian class
    'bike': 3,         # Map 'bike' → bicycle
    'motor': 7,        # Map 'motor' → motorcycle
    'bicycle': 3,
    'car': 4,
    'bus': 5,
    'truck': 6,
    'motorcycle': 7,
    'traffic light': 8,
    'traffic sign': 9,
}

# YOLO class names
YOLO_CLASS_NAMES = {
    0: "pedestrian",
    1: "rider",
    2: "other person",
    3: "bicycle",
    4: "car",
    5: "bus",
    6: "truck",
    7: "motorcycle",
    8: "traffic light",
    9: "traffic sign",
}

# BDD100K default image size
DEFAULT_IMAGE_WIDTH = 1280
DEFAULT_IMAGE_HEIGHT = 720


def convert_box2d_to_yolo(
    box2d: Dict[str, float],
    img_width: int = DEFAULT_IMAGE_WIDTH,
    img_height: int = DEFAULT_IMAGE_HEIGHT
) -> Tuple[float, float, float, float]:
    """
    Convert BDD100K box2d (Absolute) → YOLO format (normalized xywh).
    """
    x1, y1 = box2d['x1'], box2d['y1']
    x2, y2 = box2d['x2'], box2d['y2']

    box_width = x2 - x1
    box_height = y2 - y1
    x_center = x1 + box_width / 2
    y_center = y1 + box_height / 2

    # Normalize
    x_c = max(0.0, min(1.0, x_center / img_width))
    y_c = max(0.0, min(1.0, y_center / img_height))
    w = max(0.0, min(1.0, box_width / img_width))
    h = max(0.0, min(1.0, box_height / img_height))

    return x_c, y_c, w, h


def labels_to_yolo_lines(labels: List[Dict]) -> List[str]:
    """
    Convert BDD100K labels → YOLO format lines (chỉ labels có box2d).
    """
    yolo_lines = []

    for label in labels:
        category = label.get('category', '')
        box2d = label.get('box2d')

        if not box2d or category not in BDD100K_TO_YOLO:
            continue

        class_id = BDD100K_TO_YOLO[category]
        x_c, y_c, w, h = convert_box2d_to_yolo(box2d)
        yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    return yolo_lines


def load_bdd100k_annotations(json_path: str) -> list[dict]:
    """Load annotations từ file JSON của BDD100K."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def extract_metadata(annotation: dict) -> dict:
    """
    Trích xuất metadata fields từ 1 annotation.
    """
    return {
        "filename": annotation.get("name", ""),
        "weather": annotation.get("attributes", {}).get("weather", "unknown"),
        "scene": annotation.get("attributes", {}).get("scene", "unknown"),
        "timeofday": annotation.get("attributes", {}).get("timeofday", "unknown"),
        "labels": annotation.get("labels", [])
    }


def get_all_metadata(json_path: str) -> list[dict]:
    """Load BDD100K annotations và trích xuất metadata."""
    annotations = load_bdd100k_annotations(json_path)
    return [extract_metadata(annotation) for annotation in annotations]
