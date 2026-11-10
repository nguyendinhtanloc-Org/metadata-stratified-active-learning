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
            "al_batch_size": 5,
            "num_rounds": 3,
            "num_classes": 10,
            "stratify_fields": ["weather", "scene", "timeofday"],
            "model_name": "yolov8n",
            "device": "cpu"
        }

        loop = ActiveLearningLoop(config, experiment_name="test")

        assert loop.seed == 42
        assert loop.initial_budget == 10
        assert loop.batch_size == 5
        assert loop.num_rounds == 3

    def test_initialization_with_default(self):
        """Test khởi tạo với default values."""

        # device="cpu": môi trường test không có GPU (default là 0/GPU).
        config = {"device": "cpu"}

        loop = ActiveLearningLoop(config)

        assert loop.seed == 42
        assert loop.initial_budget == 500
        assert loop.batch_size == 500

    def test_checkout_point_save_load(self, tmp_path):
        """Test save và load checkpoint."""

        config = {"seed": 42, "initial_budget": 10, "device": "cpu"}

        loop = ActiveLearningLoop(config, experiment_name="test_checkpoint")

        # mock state
        loop.labeled_indices = {1, 2, 3}
        loop.unlabeled_indices = {4, 5, 6, 7, 8, 9, 10}
        loop.results_history = [{"round": 0, "map50": 0.5}]

        # save checkpoint (tmp_path: không ghi vào repo)
        checkpoint_path = str(tmp_path / "test_checkpoint.pkl")
        loop.save_checkpoint(checkpoint_path)

        # tạo loop mới và load
        loop2 = ActiveLearningLoop(config, experiment_name="test_checkpoint")
        loop2.load_checkpoint(checkpoint_path)

        assert loop2.labeled_indices == {1, 2, 3}
        assert loop2.unlabeled_indices == {4, 5, 6, 7, 8, 9, 10}
        assert len(loop2.results_history) == 1

    def test_quota_config_defaults(self):
        """Quota mặc định = hành vi cũ (proportional/joint)."""
        loop = ActiveLearningLoop({"device": "cpu"}, experiment_name="test_quota")
        assert loop.quota_mode == "proportional"
        assert loop.quota_scope == "joint"

    def test_quota_config_overrides(self):
        """Config quota_mode/quota_scope được nạp đúng."""
        loop = ActiveLearningLoop(
            {"device": "cpu", "quota_mode": "inverse", "quota_scope": "per_field"},
            experiment_name="test_quota2",
        )
        assert loop.quota_mode == "inverse"
        assert loop.quota_scope == "per_field"

    def test_seed_reproducibility(self):
        """Test seed tạo ra kết quả giống nhau."""

        config1 = {"seed": 123, "initial_budget": 5, "device": "cpu"}
        config2 = {"seed": 123, "initial_budget": 5, "device": "cpu"}

        loop1 = ActiveLearningLoop(config1, experiment_name="test1")
        loop2 = ActiveLearningLoop(config2, experiment_name="test2")

        # Cả hai nên có cùng seed
        assert loop1.seed == loop2.seed == 123


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
