"""
Data loader cho BDD100K dataset.
Load annotations và trích xuất metadata fields.
"""

import json
from pathlib import Path

def load_bdd100k_annotations(json_path: str) -> list[dict]:
    """
    Load annotations từ file JSON của BDD100K.
    
    Args:
        json_path: Đường dẫn đến file bdd100k_labels_images_*.json
    
    Returns:
        List of dicts, mỗi dict chứa metadata của 1 ảnh
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    return data

def extract_metadata(annotation: dict) -> dict:
    """
    Trích xuất metadata fields từ 1 annotation.
    
    Args:
        annotation: 1 record từ BDD100K JSON
    
    Returns:
        Dict chứa: filename, weather, scene, timeofday, labels
    """

    return {
        "filename" : annotation.get("name", ""), # Tên file
        "weather" : annotation.get("attributes", {}).get("weather", "unknown"), # default là unknown
        "scene" : annotation.get("attributes", {}).get("scene", "unknown"), # default là unknown
        "timeofday" : annotation.get("attributes", {}).get("timeofday", "unknown"), # default là unknown
        "labels" : annotation.get("labels", [])
    }

def get_all_metadata(json_path: str) -> list[dict]:
    """
    Load BDD100K annotations và trích xuất metadata cho tất cả ảnh.
    
    Args:
        json_path: Đường dẫn đến file JSON
    
    Returns:
        List of metadata dicts
    """

    annotations = load_bdd100k_annotations(json_path)
    metadata_list = [extract_metadata(annotation) for annotation in annotations]

    return metadata_list
