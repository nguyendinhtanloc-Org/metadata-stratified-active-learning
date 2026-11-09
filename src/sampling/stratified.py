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


def _waterfill_hamilton_allocation(
    capacities: Dict[Any, int],
    weights: Dict[Any, float],
    total_samples: int,
) -> Dict[Any, int]:
    """
    Phân bổ quota có chặn trần (Constrained Proportional Allocation / Water-filling)
    kết hợp Largest Remainder Method (Hamilton).

    1. Lặp Water-filling: nhóm nào có exact_quota >= sức chứa còn lại thì khóa ở
       đúng sức chứa (counts[k]), rồi chia phần dư cho các nhóm còn sức chứa
       theo đúng tỷ lệ trọng số (tránh lỗi đổ dồn toàn bộ phần dư vào 1 nhóm).
    2. Làm tròn Hamilton trên các nhóm chưa chạm trần để tổng đúng bằng target.
    """
    total_size = sum(capacities.values())
    if total_size == 0 or total_samples <= 0:
        return {}

    target = min(total_samples, total_size)
    if target == total_size:
        return dict(capacities)

    alloc = {k: 0 for k in capacities}
    remaining = target
    uncapped = {k for k, cap in capacities.items() if cap > 0}

    while uncapped and remaining > 0:
        w_sum = sum(weights[k] for k in uncapped)
        if w_sum <= 0:
            break
        over = [
            k for k in uncapped
            if (weights[k] / w_sum) * remaining >= (capacities[k] - alloc[k])
        ]
        if not over:
            break
        for k in over:
            remaining -= (capacities[k] - alloc[k])
            alloc[k] = capacities[k]
            uncapped.remove(k)

    if uncapped and remaining > 0:
        w_sum = sum(weights[k] for k in uncapped)
        remainders = []
        for k in uncapped:
            exact = (weights[k] / w_sum) * remaining
            fl = min(int(math.floor(exact)), capacities[k] - alloc[k])
            alloc[k] += fl
            remainders.append((exact - fl, weights[k], k))

        rem = target - sum(alloc.values())
        remainders.sort(key=lambda x: (x[0], x[1]), reverse=True)
        for _, _, k in remainders:
            if rem <= 0:
                break
            if alloc[k] < capacities[k]:
                alloc[k] += 1
                rem -= 1

    return alloc


def allocate_quota(
    strata: Dict[Tuple, List[int]],
    total_samples: int,
    mode: str = "proportional",
) -> Dict[Tuple, int]:
    """
    Phân bổ quota số ảnh cho mỗi stratum bằng Water-filling + Largest Remainder
    Method (Hamilton), đảm bảo tổng quota = min(total_samples, size_pool).

    mode:
      - "proportional": quota ∝ kích thước stratum. Giữ nguyên phân phối
        metadata của pool -> KHÔNG cải thiện coverage cho stratum hiếm
        (hành vi cũ, giữ để so sánh).
      - "sqrt": quota ∝ sqrt(size) -> cân bằng Pareto giữa tần suất gốc và
        chia đều (nâng mạnh nhóm hiếm nhưng không làm cạn kiệt nhóm đa số).
      - "equal": quota chia đều giữa các stratum, cap ở size stratum.
      - "inverse": quota ∝ 1/size -> ưu tiên stratum hiếm (mục tiêu phủ rộng
        các điều kiện ít gặp như rare scene, adverse weather).
    """
    capacities = {key: len(indices) for key, indices in strata.items()}
    if sum(capacities.values()) == 0 or total_samples <= 0:
        return {}

    if mode == "proportional":
        weights = {key: float(len(v)) for key, v in strata.items()}
    elif mode == "sqrt":
        weights = {key: math.sqrt(len(v)) for key, v in strata.items() if len(v) > 0}
    elif mode == "equal":
        weights = {key: 1.0 for key in strata}
    elif mode == "inverse":
        weights = {key: 1.0 / len(v) for key, v in strata.items() if len(v) > 0}
    else:
        raise ValueError(
            f"quota mode chưa hỗ trợ: {mode!r} "
            f"(hỗ trợ: 'proportional', 'sqrt', 'equal', 'inverse')"
        )

    return _waterfill_hamilton_allocation(capacities, weights, total_samples)


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
    hybrid_top_k_ratio: float = 0,
    quota_mode: str = "proportional",
    quota_scope: str = "joint",
) -> List[int]:
    """Chọn batch đa dạng về metadata, ưu tiên uncertainty cao.

    hybrid_top_k_ratio > 0 bật chế độ HYBRID (Top-K filtering):
      1. Lọc Top-K ảnh khó nhất (uncertainty cao nhất) trên toàn bộ candidate pool,
         K = int(round(batch_size * hybrid_top_k_ratio)).
      2. CHỈ trong Top-K đó mới phân tầng metadata + phân bổ quota.
      3. Trong mỗi tầng, lấy top theo uncertainty.

    Lý do (failure mode của bản cũ): bản cũ phân tầng trên TOÀN BỘ pool rồi
    mới lấy top-k uncertainty trong từng tầng. Tầng "dễ" (model tự tin) bị ép
    lấy ảnh uncertainty thấp chỉ vì tầng đó đông -> batch gồm nhiều ảnh dễ,
    lãng phí ngân sách annotation. Hybrid lấy quota theo phân phối *bên trong
    tập ảnh khó*, nên mọi ảnh được chọn đều khó.

    hybrid_top_k_ratio = 0 -> hành vi cũ (giữ để ablation).

    quota_mode:
      - "proportional" (default): quota theo kích thước stratum gốc.
      - "sqrt": quota ∝ sqrt(size), cân bằng Pareto giữa tần suất gốc và chia đều.
      - "inverse": quota ∝ 1/size, ưu tiên stratum hiếm.
      - "equal": quota chia đều giữa các nhóm.

    quota_scope:
      - "joint" (default): một tầng = tổ hợp (weather, scene, timeofday).
      - "per_field": quota tính RIÊNG cho từng field, batch chọn thỏa mãn
        đồng thời mọi quota field -> cân bằng được diversity đo theo từng field
        (đúng metric đang báo cáo). "per_field" dùng với quota_mode
        "sqrt"/"inverse"/"equal".
    """
    if hybrid_top_k_ratio and hybrid_top_k_ratio > 0:
        k = min(len(metadata_list), int(round(batch_size * hybrid_top_k_ratio)))
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
                "Dùng 'sqrt', 'inverse' hoặc 'equal'."
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
    Tính quota RIÊNG cho từng field (weather / scene / timeofday) bằng
    Water-filling + Largest Remainder Method (Hamilton).

    Vì sao cần: joint strata (weather x scene x timeofday) tạo ra hàng trăm
    tầng trên BDD100K. Equal/inverse quota trên joint gần như rút ngẫu nhiên
    trong tầng và KHÔNG kiểm soát được phân bố của từng field đơn lẻ - tức
    metric diversity đang báo cáo (entropy theo field) không được cải thiện.

    Quay lại quota từng field độc lập, rồi chọn ảnh thỏa mãn đồng thời các
    quota đó (xem select_balanced_batch).

    mode: "inverse" (ưu tiên giá trị hiếm), "equal" (chia đều),
          "sqrt" (căn bậc hai tần suất), "proportional" (theo tần suất gốc).
    """
    if mode not in ("inverse", "equal", "sqrt", "proportional"):
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
        elif mode == "sqrt":
            weights = {k: math.sqrt(v) for k, v in counts.items() if v > 0}
        elif mode == "equal":
            weights = {k: 1.0 for k in counts}
        else:  # inverse
            weights = {k: 1.0 / v for k, v in counts.items() if v > 0}

        quotas[field] = _waterfill_hamilton_allocation(
            capacities=dict(counts),
            weights=weights,
            total_samples=total_samples,
        )
    return quotas


def quota_diagnostics(
    metadata_list: List[Dict],
    selected_indices: List[int],
    stratify_fields: List[str],
    quotas: Dict[str, Dict[Any, int]],
) -> Dict[str, Any]:
    """
    Đo mức độ QUOTA được tôn trọng thực tế (để ghi log, không dùng để chọn).

    Trả về:
      - l1_error[field] : tổng |quota - actual| (0 = quota đúng tuyệt đối)
      - unique[field]   : số giá trị khác nhau thực sự có trong batch
      - n_unique_values[field] : số giá trị có trong pool
    """
    diag: Dict[str, Any] = {}
    for field in stratify_fields:
        actual = defaultdict(int)
        for i in selected_indices:
            actual[metadata_list[i].get(field, "unknown")] += 1
        q = quotas.get(field, {})
        l1 = sum(abs(q.get(v, 0) - actual.get(v, 0))
                 for v in set(q) | set(actual))
        pool_values = {m.get(field, "unknown") for m in metadata_list}
        diag.setdefault("l1_error", {})[field] = l1
        diag.setdefault("unique", {})[field] = len(actual)
        diag.setdefault("n_unique_values", {})[field] = len(pool_values)
    return diag


def select_balanced_batch(
    metadata_list: List[Dict],
    uncertainty_scores: Dict[int, float],
    batch_size: int,
    stratify_fields: List[str],
    quotas: Dict[str, Dict[Any, int]],
    ipf_iterations: int = 60,
    uncertainty_weight: float = 0.3,
) -> List[int]:
    """
    Chọn batch thỏa mãn đồng thời quota của từng field, ưu tiên uncertainty cao.

    Thuật toán: Capacity-Constrained IPF (Iterative Proportional Fitting /
    Sinkhorn) kết hợp Marginal-Respecting Rounding trên các CELL
    (tổ hợp giá trị của stratify_fields).

    Cách làm:
      1. Gom pool thành cells theo (weather, scene, timeofday) và sắp xếp ảnh
         bên trong mỗi cell theo uncertainty giảm dần.
      2. Chạy Capacity-Constrained IPF (ipf_iterations vòng): ở mỗi field,
         cân chỉnh trọng số cell để tổng biên khớp với quota của field và
         chặn trần trọng số cell không vượt quá số ảnh thực có len(cells[c]).
      3. Làm tròn rời rạc có trừ lùi quota (Marginal-Respecting Rounding):
         tại mỗi bước chọn 1 ảnh từ cell tối ưu hóa theo thứ tự ưu tiên:
         (a) số trường vẫn còn thiếu quota (rem_quota > 0),
         (b) số giá trị (hiếm) chưa xuất hiện lần nào trong batch (coverage),
         (c) độ lệch trọng số IPF còn lại (weights[c] - cell_picked[c]),
         (d) điểm uncertainty của ảnh tiếp theo trong cell.
    """
    if not quotas or batch_size <= 0 or not metadata_list:
        return []

    target_batch = min(batch_size, len(metadata_list))

    # 1) Gom cells và sắp xếp ảnh trong từng cell theo uncertainty giảm dần
    cells = defaultdict(list)
    for i, meta in enumerate(metadata_list):
        key = tuple(meta.get(f, "unknown") for f in stratify_fields)
        cells[key].append(i)
    for c in cells:
        cells[c].sort(key=lambda i: uncertainty_scores.get(i, 0.0), reverse=True)
    cell_keys = list(cells.keys())

    # 2) Capacity-Constrained IPF / Sinkhorn
    weights = {}
    for c in cell_keys:
        mean_u = sum(uncertainty_scores.get(i, 0.0) for i in cells[c]) / len(cells[c])
        weights[c] = len(cells[c]) * (
            (1.0 - uncertainty_weight) + uncertainty_weight * mean_u
        )

    alive_count = len(cell_keys)
    for _ in range(ipf_iterations):
        for fi, field in enumerate(stratify_fields):
            field_quota = quotas.get(field, {})
            marginal = defaultdict(float)
            for c in cell_keys:
                marginal[c[fi]] += weights[c]
            for c in cell_keys:
                value = c[fi]
                target = field_quota.get(value, 0)
                if target <= 0:
                    weights[c] = 0.0
                elif marginal[value] > 0:
                    weights[c] *= target / marginal[value]
            # Chặn trần theo số ảnh thực có của từng cell để đẩy phần dư sang cell khác
            for c in cell_keys:
                if weights[c] > len(cells[c]):
                    weights[c] = float(len(cells[c]))
        alive_count = sum(1 for c in cell_keys if weights[c] > 0)
        if alive_count == 0:
            break

    # 3) Làm tròn rời rạc có trừ lùi quota (Marginal-Respecting Rounding)
    rem_quota = {f: dict(quotas.get(f, {})) for f in stratify_fields}
    actual_counts = {f: defaultdict(int) for f in stratify_fields}
    cell_picked = {c: 0 for c in cell_keys}
    n_filled = 0

    # 3a) Nhận phần nguyên floor(weights[c]) trong giới hạn sức chứa cell và rem_quota
    for c in sorted(cell_keys, key=lambda c: weights[c], reverse=True):
        want = int(math.floor(weights[c]))
        if want <= 0:
            continue
        avail_cell = len(cells[c]) - cell_picked[c]
        avail_q = min(rem_quota[f].get(c[fi], 0) for fi, f in enumerate(stratify_fields))
        take = min(want, avail_cell, avail_q, target_batch - sum(cell_picked.values()))
        if take > 0:
            cell_picked[c] += take
            for fi, f in enumerate(stratify_fields):
                actual_counts[f][c[fi]] += take
                rem_quota[f][c[fi]] -= take

    # 3b) Phân bổ phần lẻ: ưu tiên (số field thiếu quota, độ phủ nhóm mới, phần dư IPF, uncertainty)
    total_picked = sum(cell_picked.values())
    for _ in range(target_batch - total_picked):
        best_c = None
        best_key = None
        for c in cell_keys:
            idx_in_cell = cell_picked[c]
            if idx_in_cell >= len(cells[c]):
                continue
            sat_fields = sum(
                1 for fi, f in enumerate(stratify_fields)
                if rem_quota[f].get(c[fi], 0) > 0
            )
            new_cov = sum(
                1 for fi, f in enumerate(stratify_fields)
                if actual_counts[f][c[fi]] == 0 and quotas[f].get(c[fi], 0) > 0
            )
            ipf_rem = weights[c] - idx_in_cell
            next_u = uncertainty_scores.get(cells[c][idx_in_cell], 0.0)
            cand_key = (sat_fields, new_cov, ipf_rem, next_u)
            if best_key is None or cand_key > best_key:
                best_key = cand_key
                best_c = c

        if best_c is None:
            break
        if best_key[0] < len(stratify_fields):
            n_filled += 1

        cell_picked[best_c] += 1
        for fi, f in enumerate(stratify_fields):
            val = best_c[fi]
            actual_counts[f][val] += 1
            if rem_quota[f].get(val, 0) > 0:
                rem_quota[f][val] -= 1

    # 3c) Hoán đổi cục bộ 1-1 giữa cell thừa và cell thiếu để triệt tiêu sai số làm tròn 3 chiều
    for _ in range(20):
        cur_l1 = sum(
            abs(quotas[f].get(v, 0) - actual_counts[f].get(v, 0))
            for f in stratify_fields
            for v in set(quotas[f]) | set(actual_counts[f])
        )
        if cur_l1 == 0:
            break
        improved = False
        deficit_cells = [
            c for c in cell_keys
            if cell_picked[c] < len(cells[c])
            and any(actual_counts[f][c[fi]] < quotas[f].get(c[fi], 0)
                    for fi, f in enumerate(stratify_fields))
        ]
        excess_cells = [
            c for c in cell_keys
            if cell_picked[c] > 0
            and any(actual_counts[f][c[fi]] > quotas[f].get(c[fi], 0)
                    for fi, f in enumerate(stratify_fields))
        ]
        for c_add in deficit_cells:
            for c_rem in excess_cells:
                if c_add == c_rem:
                    continue
                delta = 0
                for fi, f in enumerate(stratify_fields):
                    v_add, v_rem = c_add[fi], c_rem[fi]
                    if v_add == v_rem:
                        continue
                    q_add = quotas[f].get(v_add, 0)
                    q_rem = quotas[f].get(v_rem, 0)
                    a_add = actual_counts[f][v_add]
                    a_rem = actual_counts[f][v_rem]
                    delta += (abs(q_add - (a_add + 1)) - abs(q_add - a_add))
                    delta += (abs(q_rem - (a_rem - 1)) - abs(q_rem - a_rem))
                if delta < 0:
                    cell_picked[c_add] += 1
                    cell_picked[c_rem] -= 1
                    for fi, f in enumerate(stratify_fields):
                        actual_counts[f][c_add[fi]] += 1
                        actual_counts[f][c_rem[fi]] -= 1
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    selected = []
    for c in cell_keys:
        if cell_picked[c] > 0:
            selected.extend(cells[c][:cell_picked[c]])

    # 4) Instrumentation: đo mức quota thực sự được tôn trọng.
    try:
        diag = quota_diagnostics(metadata_list, selected[:target_batch],
                                 stratify_fields, quotas)
        parts = ", ".join(
            f"{f}: L1={diag['l1_error'][f]} "
            f"uniq={diag['unique'][f]}/{diag['n_unique_values'][f]}"
            for f in stratify_fields
        )
        print(f"[QUOTA] {parts} | fill={n_filled} | cells_alive={alive_count}",
              flush=True)
    except Exception as e:
        print(f"[QUOTA] diagnostics failed: {e}", flush=True)

    return selected[:target_batch]


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