"""
Unit tests cho Active Learning Loop.
"""

from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from active_learning.loop import ActiveLearningLoop

class TestActiveLearningLoop:
    """Test khởi tạo loop với config đúng."""

    def test_initialization(self):
        """Test khởi tạo loop với config đúng."""

        config = {
            "seed": 42,
            "initial_budget": 10,
            "batch_size": 5,
            "num_rounds": 3,
            "num_classes": 10,
            "stratify_fields": ["weather", "scene", "timeofday"],
            "model_name": "yolov8n",
            "device": "cuda"
        }

        loop = ActiveLearningLoop(config, experiment_name="test")

        assert loop.seed == 42
        assert loop.initial_budget == 10
        assert loop.batch_size == 5
        assert loop.num_rounds == 3

    def test_initialization_with_default(self):
        """Test khởi tạo với default values."""

        config = {}

        loop = ActiveLearningLoop(config)

        assert loop.seed == 42
        assert loop.initial_budget == 1000
        assert loop.batch_size == 500

    def test_checkout_point_save_load(self):
        """Test save và load checkpoint."""

        config = {"seed": 42, "initial_budget": 10}

        loop = ActiveLearningLoop(config, experiment_name="test_checkpoint")

        # mock state
        loop.labeled_indices = {1, 2, 3}
        loop.unlabeled_indices = {4, 5, 6, 7, 8, 9, 10}
        loop.history = [{"round": 0, "map50": 0.5}]

        # save checkpoint
        checkpoint_path = "test_checkpoint.pkl"
        loop.save_checkpoint(checkpoint_path)

        # tạo loop mới và load
        loop2 = ActiveLearningLoop(config, experiment_name="test_checkpoint")
        loop2.load_checkpoint(checkpoint_path)

        assert loop2.labeled_indices == {1, 2, 3}
        assert loop2.unlabeled_indices == {4, 5, 6, 7, 8, 9, 10}
        assert len(loop2.history) == 1

        # clean up
        Path(checkpoint_path).unlink(missing_ok=True)

    def test_seed_reproducibility(self):
        """Test seed tạo ra kết quả giống nhau."""

        config1 = {"seed": 123, "initial_budget": 5}
        config2 = {"seed": 123, "initial_budget": 5}

        loop1 = ActiveLearningLoop(config1, experiment_name="test1")
        loop2 = ActiveLearningLoop(config2, experiment_name="test2")

        # Cả hai nên có cùng seed
        assert loop1.seed == loop2.seed == 123


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
