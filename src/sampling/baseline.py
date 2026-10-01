"""
Baseline Sampling: Uncertainty-only (không có metadata stratification).
Dùng để so sánh với stratified sampling.
"""

from typing import List, Dict, Any

def baseline_sampling(
    uncertainty_scores: Dict[int, float],
    batch_size: int,
    metadata_list: Any = None,
    stratify_fields: Any = None,
    **kwargs
) -> List[int]:
    """
    Chọn batch chỉ dựa trên uncertainty cao nhất.
    Không có metadata stratification.

    Args:
        uncertainty_scores: Dict {image_index: uncertainty_score}
        batch_size: Số lượng ảnh cần chọn
        metadata_list: Ignored (để compatibility với stratified_sampling API)
        stratify_fields: Ignored (để compatibility với stratified_sampling API)
        **kwargs: Ignored (để future-proof)

    Returns:
        List các indices được chọn (top-K theo uncertainty)
    """

    if not uncertainty_scores:
        return []

    # Sort theo uncertainty GIẢM DẦN
    sorted_indices = sorted(
        uncertainty_scores.keys(),
        key=lambda idx: uncertainty_scores.get(idx, 0.0),
        reverse=True  # Uncertainty CAO nhất lên đầu
    )

    # Lấy top K
    return sorted_indices[:batch_size]
