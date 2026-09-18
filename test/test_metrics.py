"""
Unit tests cho metrics module.
"""

import pytest
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics.entropy import calc_entropy, calc_batch_entropy

class TestCalcEntropy:
    """Test calc_entropy function."""

    def test_uniform_distribution(self):
        # 4 giá trị đều nhau: entropy = log2(4) = 2 bits
        counter = Counter({"a": 1, "b": 1, "c": 1, "d": 1})
        entropy = calc_entropy(counter)

        assert abs(entropy - 2.0) < 0.01

    def test_single_value(self):
        """Entropy = 0 khi chỉ có 1 giá trị."""

        counter = Counter({"a": 10})
        entropy = calc_entropy(counter)

        assert entropy == 0.0

    def test_binary_distribution(self):
        """Entropy của phân bố nhị phân (50/50)."""

        counter = Counter({"a": 5, "b": 5})
        entropy = calc_entropy(counter)

        assert abs(entropy - 1.0) < 0.01

    def test_skewed_distribution(self):
        """Entropy thấp khi phân bố lệch."""

        counter = Counter({"a": 9, "b": 1})
        entropy = calc_entropy(counter)

        assert 0 < entropy < 1.0

    def test_empty_counter(self):
        """Entropy = 0 với counter rỗng."""

        counter = Counter()
        entropy = calc_entropy(counter)

        assert entropy == 0.0

class TestCalcBatchEntropy:
    """Test calc_batch_entropy function."""

    def test_weather_entropy(self):
        """Test entropy trên weather field."""

        metadata = [
            {"filename": "a.jpg", "weather": "clear"},
            {"filename": "b.jpg", "weather": "rain"},
            {"filename": "c.jpg", "weather": "clear"},
            {"filename": "d.jpg", "weather": "foggy"}
        ]

        entropy = calc_batch_entropy(metadata, "weather")

        assert entropy > 0 # phải có diversity

    def test_uniform_weather(self):
        """Tất cả cùng weather = entropy thấp."""

        metadata = [
            {"filename": "a.jpg", "weather": "clear"},
            {"filename": "b.jpg", "weather": "clear"},
            {"filename": "c.jpg", "weather": "clear"},
        ]

        entropy = calc_batch_entropy(metadata, "weather")

        assert entropy == 0.0

    def test_missing_filed(self):
        """Test với field không tồn tại."""

        metadata = [{"filename": "a.jpg"}]
        # sẽ raise KeyError hoặc dùng default "unknown"
        entropy = calc_batch_entropy(metadata, "nonexistent")

        # nếu có default "unknown", entropy = 0
        assert entropy == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
