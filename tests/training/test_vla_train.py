"""保护 VLA v2 训练目录、输入契约和 checkpoint 证据。"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from vla_project.training import vla_train


class VlaTrainingMetadataTests(unittest.TestCase):
    def test_default_outputs_are_new_v2_directories(self):
        self.assertEqual(
            vla_train._default_training_output(None),
            Path("outputs/training/vla_regression_full_v2"),
        )
        self.assertEqual(
            vla_train._default_training_output(10),
            Path("outputs/training/vla_regression_overfit_10_v2"),
        )

    def test_rejects_explicit_v1_output_name(self):
        with self.assertRaisesRegex(ValueError, "VLA v2"):
            vla_train._validate_training_output_version(
                Path("outputs/training/vla_custom_v1")
            )

    def test_checkpoint_metadata_hashes_exact_v2_inputs_and_stats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            contents = {
                "dataset_manifest.json": b'{"schema_version":"expert_multi_v2"}\n',
                "trajectory_expert.jsonl": b'{"episode_idx":0}\n',
                "episode_summary.jsonl": b'{"episode_idx":0}\n',
                "episode_split.json": b'{"schema_version":"episode_split_v2"}\n',
            }
            for name, content in contents.items():
                (root / name).write_bytes(content)

            actual = vla_train._build_checkpoint_metadata(
                root,
                train_episode_ids=[0, 2],
                action_stats=([1.0] * 7, [0.5] * 7),
                training_seed=42,
            )

        self.assertEqual(actual["dataset_schema_version"], "expert_multi_v2")
        self.assertEqual(actual["split_schema_version"], "episode_split_v2")
        self.assertEqual(actual["train_episode_ids"], [0, 2])
        self.assertEqual(actual["action_stats"]["mean"], [1.0] * 7)
        self.assertEqual(actual["action_stats"]["std"], [0.5] * 7)
        self.assertEqual(actual["training_seed"], 42)
        for name, content in contents.items():
            key = name.replace(".jsonl", "").replace(".json", "") + "_sha256"
            self.assertEqual(actual[key], hashlib.sha256(content).hexdigest())


class VlaInputContractTests(unittest.TestCase):
    def test_rejects_split_without_exact_balanced_v2_contract(self):
        split_doc = {
            "schema_version": "episode_split_v2",
            "train": list(range(250)),
            "val": list(range(250, 300)),
            "task_counts": {
                "train": {"red": 126, "blue": 124},
                "val": {"red": 24, "blue": 26},
            },
        }

        with self.assertRaisesRegex(ValueError, "task_counts"):
            vla_train._validate_v2_split(split_doc)

    def test_accepts_exact_balanced_non_overlapping_v2_split(self):
        split_doc = {
            "schema_version": "episode_split_v2",
            "train": list(range(250)),
            "val": list(range(250, 300)),
            "task_counts": {
                "train": {"red": 125, "blue": 125},
                "val": {"red": 25, "blue": 25},
            },
        }

        vla_train._validate_v2_split(split_doc)


if __name__ == "__main__":
    unittest.main()
