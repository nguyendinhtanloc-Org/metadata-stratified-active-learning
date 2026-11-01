"""
Metadata-Stratified Sampling cho Active Learning.
Kết hợp uncertainty sampling với metadata stratification.
"""

from typing import List, Dict, Tuple
from collections import defaultdict
import math

from metrics.entropy import calc_batch_entropy


def stratify_by_metadata(
    metadata_list: List[Dict],
    stratify_fields: List[str] = ["weather", "scene", "timeofday"],
) -> Dict[Tuple, List[int]]:
    """Phân tầng metadata theo các fields chỉ định."""
    strata = defaultdict(list)
    for idx, meta in enumerate(metadata_list):
        key = tuple(meta.get(field, "unknown") for field in stratify_fields)
        strata[key].append(idx)
    return dict(strata)


def sample_proportional_to_stratum_size(
    strata: Dict[Tuple, List[int]],
    total_samples: int
) -> Dict[Tuple, int]:
    """
    Phân bổ số lượng sample cho mỗi stratum theo tỷ lệ kích thước
    sử dụng thuật toán Largest Remainder Method (Hamilton method)
    đảm bảo tổng số mẫu luôn bằng ĐÚNG total_samples (không bị lố hoặc thiếu).
    """
    total_size = sum(len(indices) for indices in strata.values())
    if total_size == 0 or total_samples <= 0:
        return {}

    if total_samples >= total_size:
        return {key: len(indices) for key, indices in strata.items()}

    allocation = {}
    remainders = []
    allocated_sum = 0

    # Bước 1: Lấy phần nguyên (floor) trước để không bao giờ vượt quá total_samples
    for key, indices in strata.items():
        exact_quota = (len(indices) / total_size) * total_samples
        floor_quota = min(int(math.floor(exact_quota)), len(indices))
        allocation[key] = floor_quota
        allocated_sum += floor_quota
        remainder = exact_quota - floor_quota
        remainders.append((remainder, len(indices), key))

    # Bước 2: Phân bổ phần dư (remaining) cho các tầng có phần thập phân lớn nhất
    remaining = total_samples - allocated_sum
    remainders.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for _, _, key in remainders:
        if remaining <= 0:
            break
        if allocation[key] < len(strata[key]):
            allocation[key] += 1
            remaining -= 1

    # Bước 3: Nếu vẫn còn dư (do một số tầng bị chạm trần len(strata[key]))
    if remaining > 0:
        for _, _, key in remainders:
            while remaining > 0 and allocation[key] < len(strata[key]):
                allocation[key] += 1
                remaining -= 1

    return allocation


def stratified_sampling(
    metadata_list: List[Dict], 
    uncertainty_scores: Dict[int, float],
    batch_size: int,
    stratify_fields: List[str] = ["weather", "scene", "timeofday"]
) -> List[int]:
    """Chọn batch đa dạng về metadata, ưu tiên uncertainty cao."""
    strata = stratify_by_metadata(metadata_list, stratify_fields)
    allocation = sample_proportional_to_stratum_size(strata, batch_size)

    selected_indices = []
    for stratum_key, n_to_select in allocation.items():
        if n_to_select <= 0:
            continue
        stratum_indices = strata[stratum_key]
        sorted_indices = sorted(
            stratum_indices,
            key=lambda idx: uncertainty_scores.get(idx, 0.0),
            reverse=True
        )
        selected_indices.extend(sorted_indices[:n_to_select])

    return selected_indices


def calculate_batch_diversity(
    metadata_list: List[Dict],
    selected_indices: List[int],
    stratify_fields: List[str] = ["weather", "scene", "timeofday"]
) -> Dict[str, float]:
    """Tính entropy (diversity) của batch đã chọn."""
    selected_metadata = [metadata_list[i] for i in selected_indices]
    diversity = {}
    for field in stratify_fields:
        diversity[field] = calc_batch_entropy(selected_metadata, field)
    return diversity