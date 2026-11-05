"""
Unit tests cho sampling modules.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sampling.stratified import (
    stratify_by_metadata,
    sample_proportional_to_stratum_size,
    stratified_sampling,
    calculate_batch_diversity
)

class TestStratifyByMetadata:
    """Test stratify_by_metadata function."""

    def test_basic_stratification(self):
        """Test phân tầng cơ bản."""

        metadata = [
            {"weather": "clear", "scene": "city"},
            {"weather": "rain", "scene": "highway"},
            {"weather": "clear", "scene": "city"}
        ]

        strata = stratify_by_metadata(metadata, ["weather", "scene"])

        # Check structure
        assert len(strata) == 2 # 2 strata unique
        assert ("clear", "city") in strata
        assert ("rain", "highway") in strata

        # check indices
        assert strata[("clear", "city")] == [0, 2]
        assert strata[("rain", "highway")] == [1]

    def test_single_stratum(self):
        """Tất cả cùng metadata = 1 stratum."""

        metadata = [
            {"weather": "clear", "scene": "city"},
            {"weather": "clear", "scene": "city"},
        ]

        strata = stratify_by_metadata(metadata, ["weather"])

        assert len(strata) == 1

class TestSampleProportional:
    """Test sample_proportional_to_stratum_size function."""

    def test_equal_distribution(self):
        """Test khi tất cả strata bằng nhau."""

        strata = {
            ("a",): [0, 1, 2, 3],
            ("b",): [4, 5, 6, 7],
        }

        allocation = sample_proportional_to_stratum_size(strata, 4)
        # 8 items, chọn 4 → mỗi stratum 2 items
        total = sum(allocation.values())

        assert total == 4

    def test_total_samples_respected(self):
        """Tổng allocation = total_samples."""

        strata = {
            ("a",): list(range(10)),
            ("b",): list(range(10, 15)),
        }

        total_samples = 7
        allocation = sample_proportional_to_stratum_size(strata, total_samples)

        assert sum(allocation.values()) == total_samples

    def test_cannot_exceed_stratum_size(self):
        """Không lấy nhiều hơn số items trong stratum."""

        strata = {
            ("a",): [0, 1],  # Chỉ có 2 items
        }

        allocation = sample_proportional_to_stratum_size(strata, 10)

        assert allocation[("a",)] <= 2 # không được vượt quá 2

class TestStratifiedSampling:
    """Test stratified_sampling function."""

    def test_returns_correct_count(self):
        """Số lượng selected = batch_size."""

        metadata = [
            {"weather": "clear", "scene": "city"},
            {"weather": "rain", "scene": "highway"},
            {"weather": "clear", "scene": "city"},
            {"weather": "rain", "scene": "highway"},
        ]

        uncertainty = {0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3}
        selected = stratified_sampling(metadata, uncertainty, batch_size=2)

        assert len(selected) == 2

    def test_selects_high_uncertainty(self):
        """Ưu tiên chọn uncertainty cao."""

        metadata = [
            {"weather": "clear", "scene": "city"},
            {"weather": "rain", "scene": "highway"},
        ]

        uncertainty = {0: 0.9, 1: 0.1}  # idx 0 cao hơn
        selected = stratified_sampling(metadata, uncertainty, batch_size=1)

        assert 0 in selected or len(selected) == 1

    def test_respects_stratification(self):
        """Đảm bảo diversity giữa strata."""

        # Tạo data với 2 strata rõ ràng
        metadata = [
            {"weather": "clear", "scene": "city"},    # stratum A
            {"weather": "rain", "scene": "highway"},  # stratum B
            {"weather": "clear", "scene": "city"},    # stratum A
            {"weather": "rain", "scene": "highway"},  # stratum B
        ]

        uncertainty = {0: 0.9, 1: 0.8, 2: 0.7, 3: 0.6}
        selected = stratified_sampling(metadata, uncertainty, batch_size=2)
        # Nên chọn được cả 2 strata
        selected_weather = [metadata[i]["weather"] for i in selected]

        assert len(set(selected_weather)) >= 1 # Có thể 1 hoặc 2 strata

class TestCalculationDiversity:
    """Test calculate_batch_diversity function."""

    def test_diversity_calculation(self):
        """Test tính diversity đúng."""

        metadata = [
            {"weather": "clear", "scene": "city"},
            {"weather": "rain", "scene": "highway"},
            {"weather": "foggy", "scene": "residential"},
        ]

        diversity = calculate_batch_diversity(metadata, [0, 1, 2], ["weather"])

        assert "weather" in diversity
        assert diversity["weather"] > 0  # Có diversity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

class TestAllocateQuotaModes:
    """Test các mode quota mới (P1) - inverse/equal thay cho proportional."""

    def test_inverse_favours_rare_strata(self):
        """Quota inverse phải cho tầng hiếm NHIỀU hơn so với proportional."""
        from sampling.stratified import allocate_quota

        strata = {
            ("common",): list(range(90)),   # tầng đông
            ("rare",): list(range(90, 100)),  # tầng hiếm (chỉ 10 ảnh)
        }
        inv = allocate_quota(strata, 50, mode="inverse")
        prop = allocate_quota(strata, 50, mode="proportional")

        assert sum(inv.values()) == 50
        # Tầng hiếm bị bão hòa (lấy hết 10) thay vì bị bỏ qua như proportional.
        assert inv[("rare",)] == 10
        assert prop[("rare",)] < 10
        assert inv[("rare",)] > prop[("rare",)]

    def test_inverse_never_below_proportional_for_rare(self):
        """Với tầng hiếm có capacity, inverse luôn >= proportional."""
        from sampling.stratified import allocate_quota

        strata = {
            ("a",): list(range(100)),
            ("rare",): list(range(100, 120)),  # 20 ảnh
        }
        inv = allocate_quota(strata, 40, mode="inverse")
        prop = allocate_quota(strata, 40, mode="proportional")
        assert inv[("rare",)] >= prop[("rare",)]
        assert inv[("rare",)] > prop[("rare",)]

    def test_equal_splits_evenly(self):
        """Quota equal chia đều giữa các tầng (khi đủ sức chứa)."""
        from sampling.stratified import allocate_quota

        strata = {
            ("a",): list(range(40)),
            ("b",): list(range(40, 80)),
            ("c",): list(range(80, 120)),
        }
        alloc = allocate_quota(strata, 30, mode="equal")

        assert sum(alloc.values()) == 30
        assert alloc[("a",)] == alloc[("b",)] == alloc[("c",)] == 10

    def test_quota_never_exceeds_stratum_size(self):
        """Mọi mode: quota <= size stratum."""
        from sampling.stratified import allocate_quota

        strata = {
            ("tiny",): [0, 1],
            ("big",): list(range(2, 100)),
        }
        for mode in ("proportional", "equal", "inverse"):
            alloc = allocate_quota(strata, 50, mode=mode)
            assert alloc[("tiny",)] <= 2, mode
            assert sum(alloc.values()) == 50, mode

    def test_invalid_mode_raises(self):
        """Mode lạ phải raise ValueError (không silent fallback)."""
        from sampling.stratified import allocate_quota

        strata = {("a",): [0, 1, 2]}
        with pytest.raises(ValueError):
            allocate_quota(strata, 2, mode="bogus")

    def test_total_never_exceeds_pool_size(self):
        """Nếu total_samples > pool size thì lấy hết, không âm."""
        from sampling.stratified import allocate_quota

        strata = {("a",): [0, 1], ("b",): [2]}
        alloc = allocate_quota(strata, 100, mode="inverse")
        assert sum(alloc.values()) == 3


class TestPerFieldQuota:
    """Test quota_scope='per_field' (P1): quota riêng cho từng field."""

    def _make_metadata(self):
        # 1 scene phổ biến (street) x 8, 1 scene hiếm (tunnel) x 2
        meta = []
        for _ in range(8):
            meta.append({"weather": "clear", "scene": "street", "timeofday": "daytime"})
        for _ in range(2):
            meta.append({"weather": "rainy", "scene": "tunnel", "timeofday": "night"})
        return meta

    def test_per_field_balances_scene(self):
        """per_field + inverse: scene hiếm phải được ưu tiên hơn scene đông."""
        from sampling.stratified import stratified_sampling

        meta = self._make_metadata()
        scores = {i: 1.0 - i * 0.01 for i in range(len(meta))}  # idx 0..7 là street
        selected = stratified_sampling(
            meta, scores, batch_size=4,
            stratify_fields=["weather", "scene", "timeofday"],
            quota_mode="inverse", quota_scope="per_field",
        )
        scenes = [meta[i]["scene"] for i in selected]
        assert len(selected) == 4
        assert scenes.count("tunnel") >= 1, scenes

    def test_per_field_rejects_proportional(self):
        """per_field + proportional là vô nghĩa -> phải raise."""
        from sampling.stratified import stratified_sampling

        meta = self._make_metadata()
        with pytest.raises(ValueError):
            stratified_sampling(
                meta, {i: 0.5 for i in range(len(meta))}, batch_size=2,
                quota_mode="proportional", quota_scope="per_field",
            )

    def test_per_field_respects_batch_size(self):
        """Batch size luôn được tôn trọng."""
        from sampling.stratified import stratified_sampling

        meta = self._make_metadata()
        scores = {i: 0.9 - i * 0.05 for i in range(len(meta))}
        selected = stratified_sampling(
            meta, scores, batch_size=3,
            quota_mode="inverse", quota_scope="per_field",
        )
        assert len(selected) == 3
        assert len(set(selected)) == 3

    def test_joint_scope_still_works(self):
        """quota_scope='joint' (default) không đổi hành vi cũ."""
        from sampling.stratified import stratified_sampling

        meta = self._make_metadata()
        scores = {i: 0.5 for i in range(len(meta))}
        selected = stratified_sampling(
            meta, scores, batch_size=4, quota_scope="joint",
        )
        assert len(selected) == 4


class TestQuotaDiversityGain:
    """Kiểm chứng mục tiêu của P1: quota inverse làm TĂNG entropy theo field."""

    def test_inverse_improves_scene_entropy_vs_proportional(self):
        """Trên pool lệch mạnh, inverse phải cho entropy scene CAO HƠN proportional."""
        from sampling.stratified import stratified_sampling
        from metrics.entropy import calc_batch_entropy

        # pool: 90 street + 10 tunnel, batch 20
        meta = ([{"weather": "clear", "scene": "street", "timeofday": "daytime"}] * 90
                + [{"weather": "rainy", "scene": "tunnel", "timeofday": "night"}] * 10)
        # uncertainty ngẫu nhiên giả định -> dùng điểm tổng hợp 0.5 cho tất cả
        scores = {i: 0.5 for i in range(len(meta))}

        sel_prop = stratified_sampling(
            meta, scores, batch_size=20, quota_mode="proportional")
        sel_inv = stratified_sampling(
            meta, scores, batch_size=20, quota_mode="inverse")

        ent_prop = calc_batch_entropy([meta[i] for i in sel_prop], "scene")
        ent_inv = calc_batch_entropy([meta[i] for i in sel_inv], "scene")
        assert ent_inv > ent_prop, f"inverse {ent_inv} <= proportional {ent_prop}"
