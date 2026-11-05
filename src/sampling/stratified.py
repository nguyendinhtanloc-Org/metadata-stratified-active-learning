"""
Metadata-Stratified Sampling cho Active Learning.
Kết hợp uncertainty sampling với metadata stratification.
"""

from typing import List, Dict, Tuple, Any
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


def allocate_quota(
    strata: Dict[Tuple, List[int]],
    total_samples: int,
    mode: str = "proportional",
) -> Dict[Tuple, int]:
    """
    Phân bổ quota số ảnh cho mỗi stratum bằng Largest Remainder Method
    (Hamilton), đảm bảo tổng quota = min(total_samples, size_pool).

    mode:
      - "proportional": quota ∝ kích thước stratum. Giữ nguyên phân phối
        metadata của pool -> KHÔNG cải thiện coverage cho stratum hiếm
        (hành vi cũ, giữ để so sánh).
      - "equal": quota chia đều giữa các stratum, cap ở size stratum.
      - "inverse": quota ∝ 1/size -> ưu tiên stratum HIẾM (rare scene,
        adverse weather). Mục tiêu: phủ đều vùng khó thay vì theo phân
        phối gốc (P1).

    Quota luôn bị cap <= size stratum; phần dư sau cap được trả dần cho
    stratum còn capacity (theo phần dư lớn nhất, tie-break trọng số cao),
    nên tổng không bao giờ vượt total_samples.
    """
    total_size = sum(len(indices) for indices in strata.values())
    if total_size == 0 or total_samples <= 0:
        return {}

    target = min(total_samples, total_size)
    if target == total_size:
        return {key: len(indices) for key, indices in strata.items()}

    if mode == "proportional":
        weights = {key: float(len(v)) for key, v in strata.items()}
    elif mode == "equal":
        weights = {key: 1.0 for key in strata}
    elif mode == "inverse":
        weights = {key: 1.0 / len(v) for key, v in strata.items()}
    else:
        raise ValueError(
            f"quota mode chưa hỗ trợ: {mode!r} "
            f"(hỗ trợ: 'proportional', 'equal', 'inverse')"
        )

    weight_sum = sum(weights.values())

    allocation = {}
    remainders = []
    allocated_sum = 0
    for key, indices in strata.items():
        exact_quota = (weights[key] / weight_sum) * target
        floor_quota = min(int(math.floor(exact_quota)), len(indices))
        allocation[key] = floor_quota
        allocated_sum += floor_quota
        remainders.append((exact_quota - floor_quota, weights[key], key))

    remaining = target - allocated_sum
    remainders.sort(key=lambda x: (x[0], x[1]), reverse=True)
    for _, _, key in remainders:
        if remaining <= 0:
            break
        if allocation[key] < len(strata[key]):
            allocation[key] += 1
            remaining -= 1

    # Phần dư sau khi một số tầng bị cap (ví dụ mode="inverse" với tầng hiếm
    # chỉ còn 10 ảnh) được chia LẠI theo trọng số cho các tầng còn capacity,
    # thay vì đổ hết vào một tầng tuỳ thứ tự dict (thiên lệch ngẫu nhiên).
    while remaining > 0:
        cap_keys = [k for k in strata if allocation[k] < len(strata[k])]
        if not cap_keys:
            break
        w_cap = sum(weights[k] for k in cap_keys)
        added = 0
        for k in cap_keys:
            share = int((weights[k] / w_cap) * remaining)
            give = min(share, len(strata[k]) - allocation[k])
            allocation[k] += give
            added += give
        remaining -= added
        if added == 0:
            # Không chia được theo tỷ lệ (share < 1 cho tất cả) -> tra từng
            # đơn vị cho tầng có trọng số cao nhất còn capacity.
            for k in sorted(cap_keys, key=lambda x: weights[x], reverse=True):
                if remaining <= 0:
                    break
                allocation[k] += 1
                remaining -= 1

    return allocation


def sample_proportional_to_stratum_size(
    strata: Dict[Tuple, List[int]],
    total_samples: int
) -> Dict[Tuple, int]:
    """[Giữ tương thích] Quota proportional - dùng allocate_quota(mode=...) cho các mode khác."""
    return allocate_quota(strata, total_samples, mode="proportional")


def stratified_sampling(
    metadata_list: List[Dict],
    uncertainty_scores: Dict[int, float],
    batch_size: int,
    stratify_fields: List[str] = ["weather", "scene", "timeofday"],
    hybrid_top_k_ratio: int = 0,
    quota_mode: str = "proportional",
    quota_scope: str = "joint",
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

    quota_mode (P1):
      - "proportional" (default, hành vi cũ): quota theo kích thước stratum.
      - "inverse": quota ∝ 1/size, ưu tiên stratum hiếm.
      - "equal": quota chia đều.

    quota_scope:
      - "joint" (default): một tầng = tổ hợp (weather, scene, timeofday).
      - "per_field": quota tính RIÊNG cho từng field, batch chọn thỏa mãn
        đồng thời mọi quota field -> cân bằng được diversity đo theo từng field
        (đúng metric đang báo cáo). "per_field" chỉ dùng với quota_mode
        "inverse"/"equal".
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
            hybrid_top_k_ratio=0, quota_mode=quota_mode, quota_scope=quota_scope,
        )
        return [top_k[i] for i in selected]

    if quota_scope == "per_field":
        if quota_mode == "proportional":
            raise ValueError(
                "quota_scope='per_field' không dùng được với quota_mode='proportional' "
                "(proportional theo field = proportional theo joint về mặt quota). "
                "Dùng 'inverse' hoặc 'equal'."
            )
        quotas = allocate_quota_per_field(
            metadata_list, batch_size, stratify_fields, mode=quota_mode
        )
        return select_balanced_batch(
            metadata_list=metadata_list,
            uncertainty_scores=uncertainty_scores,
            batch_size=batch_size,
            stratify_fields=stratify_fields,
            quotas=quotas,
        )

    strata = stratify_by_metadata(metadata_list, stratify_fields)
    allocation = allocate_quota(strata, batch_size, mode=quota_mode)

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


def allocate_quota_per_field(
    metadata_list: List[Dict],
    total_samples: int,
    stratify_fields: List[str],
    mode: str = "inverse",
) -> Dict[str, Dict[Any, int]]:
    """
    Tính quota RIÊNG cho từng field (weather / scene / timeofday).

    Vì sao cần: joint strata (weather x scene x timeofday) tạo ra hàng trăm
    tầng trên BDD100K. Equal/inverse quota trên joint gần như rút ngẫu nhiên
    trong tầng và KHÔNG kiểm soát được phân bố của từng field đơn lẻ - tức
    metric diversity đang báo cáo (entropy theo field) không được cải thiện.

    Quay lại quota từng field độc lập, rồi chọn ảnh thỏa mãn đồng thời các
    quota đó (xem select_balanced_batch).

    mode: "inverse" (ưu tiên giá trị hiếm), "equal" (chia đều),
          "proportional" (theo tần suất gốc).
    """
    if mode not in ("inverse", "equal", "proportional"):
        raise ValueError(f"quota mode chưa hỗ trợ: {mode!r}")

    quotas = {}
    for field in stratify_fields:
        counts = defaultdict(int)
        for meta in metadata_list:
            counts[meta.get(field, "unknown")] += 1
        if not counts:
            continue

        if mode == "proportional":
            weights = {k: float(v) for k, v in counts.items()}
        elif mode == "equal":
            weights = {k: 1.0 for k in counts}
        else:  # inverse
            weights = {k: 1.0 / v for k, v in counts.items()}

        w_sum = sum(weights.values())
        target = min(total_samples, sum(counts.values()))

        alloc = {}
        remainders = []
        allocated = 0
        for key, w in weights.items():
            exact = (w / w_sum) * target
            floor_q = min(int(math.floor(exact)), counts[key])
            alloc[key] = floor_q
            allocated += floor_q
            remainders.append((exact - floor_q, w, key))

        remaining = target - allocated
        remainders.sort(key=lambda x: (x[0], x[1]), reverse=True)
        for _, _, key in remainders:
            if remaining <= 0:
                break
            if alloc[key] < counts[key]:
                alloc[key] += 1
                remaining -= 1
        if remaining > 0:
            for _, _, key in remainders:
                while remaining > 0 and alloc[key] < counts[key]:
                    alloc[key] += 1
                    remaining -= 1

        quotas[field] = alloc
    return quotas


def select_balanced_batch(
    metadata_list: List[Dict],
    uncertainty_scores: Dict[int, float],
    batch_size: int,
    stratify_fields: List[str],
    quotas: Dict[str, Dict[Any, int]],
) -> List[int]:
    """
    Chọn batch thỏa mãn đồng thời quota của từng field, ưu tiên uncertainty cao.

    Tham lam (greedy): duyệt candidate theo uncertainty giảm dần, nhận ảnh nếu
    mọi field của ảnh đó chưa đạt quota. Ảnh bị bỏ (vì đã đủ quota ở ít nhất
    một field) được giữ lại làm "dự phòng" cho các field còn thiếu.

    Sau vòng lượt chính, nếu batch chưa đủ (vì pool không đủ ảnh thỏa quota),
    lấp thêm theo uncertainty từ các ảnh còn lại.
    """
    if not quotas:
        return []

    remaining = {f: dict(q) for f, q in quotas.items()}
    order = sorted(
        range(len(metadata_list)),
        key=lambda i: uncertainty_scores.get(i, 0.0),
        reverse=True,
    )

    selected = []
    leftover = []
    for idx in order:
        if len(selected) >= batch_size:
            leftover.append(idx)
            continue
        vals = {f: metadata_list[idx].get(f, "unknown") for f in stratify_fields}
        if all(remaining[f].get(vals[f], 0) > 0 for f in stratify_fields):
            selected.append(idx)
            for f in stratify_fields:
                remaining[f][vals[f]] -= 1
        else:
            leftover.append(idx)

    # Lấp thêm nếu chưa đủ batch (pool không đủ ảnh thỏa quota)
    if len(selected) < batch_size:
        selected.extend(leftover[: batch_size - len(selected)])

    return selected[:batch_size]


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