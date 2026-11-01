"""
Uncertainty sampling cho Active Learning.
Tính uncertainty score cho mỗi ảnh dựa trên YOLO predictions.
"""

import math
from typing import List, Tuple
import numpy as np


def calc_image_uncertainty(result, num_classes: int = 10) -> float:
    """
    Tính uncertainty cho 1 ảnh từ YOLO Results object.
    
    Cải tiến chuẩn Object Detection Active Learning:
    1. Lọc bỏ các box nhiễu nền cực thấp (conf < 0.05) sinh ra do conf=0.001.
    2. Tính Box-level Uncertainty (Binary Entropy + 1 - conf) để bắt đúng các
       đối tượng mà mô hình đang phân vân (conf quanh 0.2 - 0.6).
    3. Kết hợp với Image-level Class Entropy và chuẩn hóa về thang [0, log2(num_classes)].
    """
    max_score = math.log2(num_classes)

    # Nếu không có box nào
    if result.boxes is None or len(result.boxes) == 0:
        return max_score * 0.5  # Mức trung bình để tránh lấy toàn ảnh trống (background)

    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy()

    # Lọc các box có ý nghĩa (loại bỏ nhiễu nền conf=0.001)
    valid_mask = confs >= 0.05
    if not np.any(valid_mask):
        # Ảnh có tín hiệu rất yếu (under-confident) -> độ không chắc chắn cao vừa phải
        return max_score * 0.75

    confs = confs[valid_mask]
    clss = clss[valid_mask]

    # 1. Box-level Uncertainty: Binary entropy H(p) + Least Confidence (1 - p)
    # p càng gần 0.3 - 0.5 thì mô hình càng phân vân; p gần 0.95 thì uncertainty -> 0
    eps = 1e-6
    p = np.clip(confs, eps, 1.0 - eps)
    binary_entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))  # trong [0, 1]
    least_conf = 1.0 - p                                                # trong [0, 1]
    box_uncertainty = float(np.mean(0.6 * binary_entropy + 0.4 * least_conf))

    # 2. Image-level Class Distribution Entropy (trọng số theo độ không chắc chắn của box)
    class_weights = np.zeros(num_classes)
    for c_val, cls_idx in zip(least_conf, clss):
        c_idx = int(cls_idx)
        if 0 <= c_idx < num_classes:
            class_weights[c_idx] += c_val

    total_w = class_weights.sum()
    class_entropy_norm = 0.0
    if total_w > 0:
        probs = class_weights / total_w
        ent = -sum(prob * math.log2(prob) for prob in probs if prob > 0)
        class_entropy_norm = ent / max_score  # chuẩn hóa về [0, 1]

    # 3. Điểm tổng hợp (70% độ khó của các bounding box + 30% độ phức tạp đa lớp)
    combined_norm = 0.75 * box_uncertainty + 0.25 * class_entropy_norm
    return float(combined_norm * max_score)


def rank_by_uncertainty(
    model, 
    image_paths: List[str], 
    num_classes: int = 10
) -> List[Tuple[str, float]]:
    """Predict trên tất cả ảnh và sắp xếp theo uncertainty giảm dần."""
    results_with_paths = []
    for image_path in image_paths:
        for result in model.predict(source=str(image_path), verbose=False, stream=True):
            uncertainty = calc_image_uncertainty(result, num_classes)
            results_with_paths.append((str(image_path), uncertainty))
    
    results_with_paths.sort(key=lambda x: x[1], reverse=True)
    return results_with_paths


def get_top_k_uncertain(
    model, 
    image_paths: List[str], 
    k: int,
    num_classes: int = 10
) -> List[str]:
    """Lấy top-K ảnh có uncertainty cao nhất."""
    ranked = rank_by_uncertainty(model, image_paths, num_classes)
    return [path for path, _ in ranked[:k]]