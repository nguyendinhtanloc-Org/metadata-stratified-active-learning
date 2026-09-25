"""
Metadata-Stratified Sampling cho Active Learning.
Kết hợp uncertainty sampling với metadata stratification.
"""

from typing import List, Dict, Tuple
from collections import defaultdict

from metrics.entropy import calc_batch_entropy

def stratify_by_metadata(
    metadata_list: List[Dict],
    stratify_fields: List[str] = ["weather", "scene", "timeofday"],
) -> Dict[Tuple, List[int]]:
    """
    Phân tầng metadata theo các fields chỉ định.
    
    Args:
        metadata_list: List of metadata dicts từ loader
        stratify_fields: Fields để phân tầng
    
    Returns:
        Dict với key = tuple của field values, value = list indices
        Ví dụ: {("clear", "city", "daytime"): [0, 5, 12], ...}
    """

    strata = defaultdict(list)

    for idx, meta in enumerate(metadata_list):
        # tạo tuple key từ các fields values
        key = tuple(meta.get(field, "unknown") for field in stratify_fields)
        strata[key].append(idx)

    return dict(strata)

def sample_proportional_to_stratum_size(
    strata: Dict[Tuple, List[int]],
    total_samples: int
) -> Dict[Tuple, int]:
    """
    Phân bổ số lượng sample cho mỗi stratum theo tỷ lệ kích thước.
        
    Args:
        strata: Dict từ stratify_by_metadata
        total_samples: Tổng số samples cần chọn
        
    Returns:
        Dict với key = stratum, value = số samples cho stratum đó
    """
    
    total_size = sum(len(indices) for indices in strata.values())

    if total_size == 0:
        return {}

    allocation = {}
    remaining = total_samples

    for key, indices in strata.items():
        proportion = len(indices) / total_size
        # Dùng round() thay vì int()
        n_samples = round(proportion * total_samples)
        # Đảm bảo không vượt quá
        n_samples = min(n_samples, len(indices))
        allocation[key] = n_samples
        remaining -= n_samples

    # Phân bổ remaining một cách fair
    if remaining > 0:
        # Sort theo tỷ lệ để ưu tiên strata lớn
        sorted_strata = sorted(
            strata.items(),
            key=lambda x: len(x[1]) / total_size,
            reverse=True
        )
        
        for key, _ in sorted_strata:
            if remaining <= 0:
                break
            if allocation[key] < len(strata[key]):
                allocation[key] += 1
                remaining -= 1
    elif remaining < 0:
        # Over-allocation: Giảm từ strata nhỏ nhất
        sorted_strata = sorted(
            strata.items(),
            key=lambda x: len(x[1]) / total_size
        )
        
        for key, _ in sorted_strata:
            if remaining >= 0:
                break
            if allocation[key] > 0:
                allocation[key] -= 1
                remaining += 1

    # Verify tổng = total_samples
    assert sum(allocation.values()) == total_samples, \
        f"Allocation {sum(allocation.values())} != {total_samples}"

    return allocation

def stratified_sampling(
    metadata_list: List[Dict], 
    uncertainty_scores: Dict[int, float],
    batch_size: int,
    stratify_fields: List[str] = ["weather", "scene", "timeofday"]
    ) -> List[int]:
    """
    Chọn batch đa dạng về metadata, ưu tiên uncertainty cao.
    
    Args:
        metadata_list: List of metadata dicts
        uncertainty_scores: Dict {image_index: uncertainty_score}
        batch_size: Số lượng ảnh cần chọn
        stratify_fields: Fields để phân tầng
    
    Returns:
        List các indices của ảnh được chọn
    """

    # phân tầng metadata
    strata = stratify_by_metadata(metadata_list, stratify_fields)

    # tính allocation cho mỗi stratum
    allocation = sample_proportional_to_stratum_size(strata, batch_size)

    # chọn top k từ mỗi stratum
    selected_indices = []

    for stratum_key, n_to_select in allocation.items():
        stratum_indices = strata[stratum_key]

        # Sort theo uncertainty GIẢM DẦN trước (LUÔN LUÔN sort)
        sorted_indices = sorted(
            stratum_indices,
            key=lambda idx: uncertainty_scores.get(idx, 0.0),
            reverse=True  # Uncertainty CAO nhất lên đầu
        )

        # Sau đó lấy top N
        if n_to_select >= len(stratum_indices):
            selected_indices.extend(stratum_indices)  # Lấy tất cả
        else:
            selected_indices.extend(sorted_indices[:n_to_select])  # Lấy top N đã sort

    return selected_indices

def calculate_batch_diversity(
    metadata_list: List[Dict],
    selected_indices: List[int],
    stratify_fields: List[str] = ["weather", "scene", "timeofday"]
) -> Dict[str, float]:
    """
    Tính entropy (diversity) của batch đã chọn.
    
    Args:
        metadata_list: Full metadata list
        selected_indices: Indices đã chọn
        stratify_fields: Fields để đánh giá
    
    Returns:
        Dict với entropy của mỗi field
    """

    selected_metadata = [metadata_list[i] for i in selected_indices]
    diversity = {}

    for field in stratify_fields:
        entropy = calc_batch_entropy(selected_metadata, field)
        diversity[field] = entropy

    return diversity
