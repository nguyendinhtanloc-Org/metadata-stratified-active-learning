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
    stratify_fields: List[str] = ["weather", "scene", "timeofday"],
    hybrid_top_k_ratio: int = 0,
) -> List[int]:
    """Chọn batch đa dạng về metadata, ưu tiên uncertainty cao.

    hybrid_top_k_ratio > 0 bật chế độ HYBRID (Top-K filtering):
      1. Lọc Top-K ảnh khó nhất (uncertainty cao nhất) trên toàn bộ candidate pool,
         K = batch_size * hybrid_top_k_ratio.
      2. CHỈ trong Top-K đó mới phân tầng metadata + phân bổ quota.
      3. Trong mỗi tầng, lấy top theo uncertainty.

    Lý do (failure mode của bản cũ): bản cũ phân tầng trên TOÀN BỘ pool rồi
    mới lấy top-k uncertainty trong từng tầng. Tầng "dễ" (model tự tin) bị ép
    lấy ảnh uncertainty thấp chỉ vì tầng đó đông -> batch gồm nhiều ảnh dễ,
    lãng phí ngân sách annotation. Hybrid lấy quota theo phân phối *bên trong
    tập ảnh khó*, nên mọi ảnh được chọn đều khó.

    hybrid_top_k_ratio = 0 -> hành vi cũ (giữ để ablation).
    """
    if hybrid_top_k_ratio and hybrid_top_k_ratio > 0:
        k = min(len(metadata_list), batch_size * hybrid_top_k_ratio)
        top_k = sorted(
            range(len(metadata_list)),
            key=lambda i: uncertainty_scores.get(i, 0.0),
            reverse=True
        )[:k]
        # Đánh lại index 0..k-1 để hàm phân tầng dùng lại được
        remap = {old: new for new, old in enumerate(top_k)}
        sub_metadata = [metadata_list[old] for old in top_k]
        sub_scores = {remap[old]: uncertainty_scores.get(old, 0.0) for old in top_k}
        selected = stratified_sampling(
            sub_metadata, sub_scores, batch_size, stratify_fields,
            hybrid_top_k_ratio=0,
        )
        return [top_k[i] for i in selected]

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