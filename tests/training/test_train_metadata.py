"""保护训练输出版本与 checkpoint 的动作语义证据。"""

import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from vla_project.training import train


def tokenizer_asset(representation="delta_q", num_bins=64):
    return {
        "representation": representation,
        "binning": "quantile",
        "requested_num_bins": num_bins,
        "edges": [[float(i) for i in range(num_bins + 1)] for _ in range(7)],
        "reconstruction_values": [
            [float(i) + 0.25 for i in range(num_bins)]
            for _ in range(7)
        ],
    }


class TrainingOutputVersionTests(unittest.TestCase):
    def test_delta_defaults_use_v2_for_full_and_overfit(self):
        self.assertEqual(
            train._default_training_output("delta_q_64", None),
            Path("outputs/training/bc_delta_q_64_full_v2"),
        )
        self.assertEqual(
            train._default_training_output("delta_q_64", 10),
            Path("outputs/training/bc_delta_q_64_overfit_10_v2"),
        )

    def test_existing_representations_keep_v1_names(self):
        self.assertEqual(
            train._default_training_output("regression", None),
            Path("outputs/training/bc_regression_v1"),
        )
        self.assertEqual(
            train._default_training_output("absolute_q_32", 10),
            Path("outputs/training/bc_absolute_q_32_overfit_10_v1"),
        )

    def test_rejects_explicit_delta_v1_output_name(self):
        with self.assertRaisesRegex(ValueError, "delta checkpoint.*v2"):
            train._validate_training_output_version(
                "delta_q_64",
                Path("outputs/training/custom_delta_v1"),
            )

        accepted = Path("outputs/training/custom_delta_v2")
        self.assertEqual(
            train._validate_training_output_version(
                "delta_q_64",
                accepted,
            ),
            accepted,
        )


class CheckpointMetadataTests(unittest.TestCase):
    def test_hashes_exact_training_split_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            split_path = Path(temp_dir) / "episode_split.json"
            split_path.write_bytes(b'{"train":[0],"val":[1]}\n')

            actual = train._training_split_sha256(split_path)

        self.assertEqual(
            actual,
            hashlib.sha256(b'{"train":[0],"val":[1]}\n').hexdigest(),
        )

    def test_builds_delta_metadata_with_frozen_tokenizer(self):
        tokenizer = tokenizer_asset()

        actual = train._build_checkpoint_metadata(
            "delta_q_64",
            "a" * 64,
            tokenizer=tokenizer,
        )

        self.assertEqual(
            actual["action_semantics"],
            "same_episode_saved_target_delta_v2",
        )
        self.assertEqual(actual["training_split_sha256"], "a" * 64)
        self.assertEqual(actual["tokenizer"], tokenizer)
        self.assertNotIn("action_stats", actual)

    def test_builds_regression_metadata_with_action_stats(self):
        actual = train._build_checkpoint_metadata(
            "regression",
            "b" * 64,
            action_stats=([1.0] * 7, [0.5] * 7),
        )

        self.assertEqual(
            actual["action_semantics"],
            "absolute_joint_target_v1",
        )
        self.assertEqual(
            actual["action_stats"],
            {"mean": [1.0] * 7, "std": [0.5] * 7},
        )
        self.assertNotIn("tokenizer", actual)

    def test_checkpoint_merges_semantics_and_epoch_metrics(self):
        model = torch.nn.Linear(2, 1)
        model.action_representation = "delta_q_64"
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        common = train._build_checkpoint_metadata(
            "delta_q_64",
            "c" * 64,
            tokenizer=tokenizer_asset(),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"

            train._save_checkpoint(
                model,
                optimizer,
                3,
                path,
                common_metadata=common,
                epoch_metrics={"val_loss": 1.25, "train_loss": 0.75},
            )
            saved = torch.load(path, map_location="cpu", weights_only=False)

        self.assertEqual(saved["metadata"]["training_split_sha256"], "c" * 64)
        self.assertEqual(saved["metadata"]["val_loss"], 1.25)
        self.assertEqual(saved["metadata"]["train_loss"], 0.75)
        self.assertEqual(
            saved["metadata"]["action_semantics"],
            "same_episode_saved_target_delta_v2",
        )


if __name__ == "__main__":
    unittest.main()
