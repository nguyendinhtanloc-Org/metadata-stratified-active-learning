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

    # tổng số ảnh trong tất cả strata
    total_size = sum(len(indices) for indices in strata.values())

    if total_size == 0:
        return {}

    allocation = {}
    remaining = total_samples

    for key, indices in strata.items():
        # tính tỉ lệ
        proportion = len(indices) / total_size
        # round xuống để đảm bảo rằng không vượt quá total
        n_samples = int(proportion * total_samples)

        # Đảm bảo không lấy nhiều hơn số ảnh trong stratum
        n_samples = min(n_samples, len(indices))

        allocation[key] = n_samples
        remaining -= n_samples

    # Phân bổ remaining samples cho các stratum lớn nhất
    if remaining > 0:
        sorted_strata = sorted(
            strata.items(),
            key = lambda x: len(x[1]), # Sort theo số lượng ảnh giảm dần
            reverse=True
        )
        
        for key, _ in sorted_strata:
            if remaining <= 0:
                break

            # thêm 1 sample nếu stratum còn capacity
            if allocation[key] < len(strata[key]):
                allocation[key] += 1
                remaining -= 1

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
        stratum_indicies = strata[stratum_key]

        if n_to_select >= len(stratum_indicies):
            # nếu cần chọn nhiều hơn số ảnh trong stratum
            selected_indices.extend(stratum_indicies)
        else:
            # sort stratum indicies theo uncertainty scores theo chiều giảm dần
            sorted_indicies = sorted(
                stratum_indicies,
                key = lambda idx: uncertainty_scores.get(idx, 0.0),
                reverse=True # Uncertainty CAO nhất lên đầu
            )

            # lấy top N
            selected_indices.extend(sorted_indicies[:n_to_select])

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
