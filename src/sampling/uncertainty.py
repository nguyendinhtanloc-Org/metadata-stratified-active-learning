"""
Uncertainty sampling cho Active Learning.
Tính uncertainty score cho mỗi ảnh dựa trên YOLO predictions.
"""

import math
from typing import List, Tuple
import numpy as np
from pathlib import Path


def calc_image_uncertainty(result, num_classes: int = 10) -> float:
    """
    Tính uncertainty cho 1 ảnh từ YOLO Results object.
    
    Logic:
    1. Lấy confidence + class index từ mỗi box
    2. Tạo pseudo class probabilities (sum conf per class)
    3. Tính Shannon entropy từ probabilities
    
    Args:
        result: ultralytics.engine.results.Results (1 ảnh)
        num_classes: số lượng class (BDD100K = 10)
    
    Returns:
        float: entropy score (cao = mô hình không chắc chắn)
    """
    
    # Nếu không detect được box nào → entropy cao nhất
    if result.boxes is None or len(result.boxes) == 0:
        return math.log2(num_classes)
    
    # Lấy confidence và class index cho mỗi box
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy()
    
    # Tạo pseudo class probabilities
    # Sum confidence cho mỗi class
    class_confs = np.zeros(num_classes)
    for conf, cls in zip(confs, clss):
        class_confs[int(cls)] += conf
    
    # Normalize thành probabilities
    total = class_confs.sum()
    if total == 0:
        return math.log2(num_classes)
    
    probs = class_confs / total
    
    # Tính Shannon entropy
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)
    
    return entropy


def rank_by_uncertainty(
    model, 
    image_paths: List[str], 
    num_classes: int = 10
) -> List[Tuple[str, float]]:
    """
    Predict trên tất cả ảnh và sắp xếp theo uncertainty giảm dần.
    
    Args:
        model: ultralytics YOLO model
        image_paths: list of str (đường dẫn ảnh)
        num_classes: số class
    
    Returns:
        list of (path, score): sorted giảm dần theo uncertainty
    """

    results_with_paths = []
    
    # Predict từng ảnh với stream=True (nhanh hơn)
    for image_path in image_paths:
        result = model.predict(
            source=str(image_path), 
            verbose=False,
            stream=True  # Streaming mode
        )[0]  # Lấy result đầu tiên
        
        uncertainty = calc_image_uncertainty(result, num_classes)
        results_with_paths.append((str(image_path), uncertainty))
    
    # Sắp xếp giảm dần theo uncertainty (uncertainty CAO nhất lên đầu)
    results_with_paths.sort(key=lambda x: x[1], reverse=True)
    
    return results_with_paths


def get_top_k_uncertain(
    model, 
    image_paths: List[str], 
    k: int,
    num_classes: int = 10
) -> List[str]:
    """
    Lấy top-K ảnh có uncertainty cao nhất.
    
    Args:
        model: YOLO model
        image_paths: list đường dẫn ảnh
        k: số ảnh cần lấy
        num_classes: số class
    
    Returns:
        List đường dẫn của top-K ảnh
    """

    ranked = rank_by_uncertainty(model, image_paths, num_classes)
    
    return [path for path, _ in ranked[:k]]
