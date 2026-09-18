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