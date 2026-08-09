"""保护训练和 rollout 在重依赖加载前验证输出路径。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vla_project.training import rollout, train, vla_rollout, vla_train


class TrainingOutputBoundaryTests(unittest.TestCase):
    def test_bc_training_rejects_escape_before_loading_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "vla_project.training.dataset.BCDataset"
        ) as dataset:
            with self.assertRaisesRegex(ValueError, "parent traversal"):
                train.run_training(
                    output_dir="outputs/training/../../src",
                    project_root_override=temp_dir,
                )
            dataset.assert_not_called()

    def test_vla_training_rejects_escape_before_loading_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "vla_project.training.dataset.VLADataset"
        ) as dataset:
            with self.assertRaisesRegex(ValueError, "parent traversal"):
                vla_train.run_training(
                    output_dir="outputs/training/../../src",
                    project_root_override=temp_dir,
                )
            dataset.assert_not_called()

    def test_bc_rollout_rejects_escape_before_loading_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            rollout, "load_model_for_rollout"
        ) as load_model:
            with self.assertRaisesRegex(ValueError, "parent traversal"):
                rollout.run_rollout_evaluation(
                    "checkpoint.pt",
                    output_dir="outputs/rollout/../../src",
                    project_root_override=temp_dir,
                )
            load_model.assert_not_called()

    def test_vla_rollout_rejects_escape_before_loading_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            vla_rollout, "_load_vla_model"
        ) as load_model:
            with self.assertRaisesRegex(ValueError, "parent traversal"):
                vla_rollout.run_vla_rollout(
                    "checkpoint.pt",
                    "sim_config.yaml",
                    "dataset",
                    "outputs/rollout/../../src",
                    project_root_override=temp_dir,
                )
            load_model.assert_not_called()

    def test_training_rejects_nonempty_existing_run_before_data_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "outputs/training/existing"
            output.mkdir(parents=True)
            (output / "checkpoint_best.pt").write_text(
                "evidence", encoding="utf-8"
            )
            with patch("vla_project.training.dataset.BCDataset") as dataset:
                with self.assertRaisesRegex(FileExistsError, "not empty"):
                    train.run_training(
                        output_dir=output,
                        project_root_override=root,
                    )
                dataset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
