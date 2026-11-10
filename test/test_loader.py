"""
Unit tests cho data loader.
"""

from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import extract_metadata

class TestExtractMetadata:
    """Test extract_metadata function."""

    def test_extract_weather_scene_timeofday(self):
        """Test trích xuất đúng các fields."""

        annotation = {
            "name": "test.jpg",
            "attributes": {
                "weather": "rain",
                "scene": "city",
                "timeofday": "night"
            },
            "labels": [{"category": "car"}]
        }

        result = extract_metadata(annotation)

        assert result["filename"] == "test.jpg"
        assert result["weather"] == "rain"
        assert result["scene"] == "city"
        assert result["timeofday"] == "night"
        assert result["labels"] == [{"category": "car"}]

    def test_default_values_of_missing_fields(self):
        """Test default values khi field missing."""

        annotation = {
            "name": "test.jpg"
        }

        result = extract_metadata(annotation)

        assert result["weather"] == "unknown"
        assert result["scene"] == "unknown"
        assert result["timeofday"] == "unknown"
        assert result["labels"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])