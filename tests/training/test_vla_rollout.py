"""保护 VLA rollout 的 checkpoint 绑定和保存场景确定性重放。"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from vla_project.training import vla_rollout


class CheckpointBindingTests(unittest.TestCase):
    def test_rejects_dataset_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = {
                "dataset_manifest.json": b"manifest\n",
                "trajectory_expert.jsonl": b"trajectory\n",
                "episode_summary.jsonl": b"summary\n",
                "episode_split.json": b"split\n",
            }
            for name, content in files.items():
                (root / name).write_bytes(content)
            metadata = {
                "dataset_schema_version": "expert_multi_v2",
                "split_schema_version": "episode_split_v2",
                "action_stats": {"mean": [0.0] * 7, "std": [1.0] * 7},
                **{
                    f"{name.replace('.jsonl', '').replace('.json', '')}_sha256":
                    hashlib.sha256(content).hexdigest()
                    for name, content in files.items()
                },
            }
            metadata["episode_split_sha256"] = "0" * 64

            with self.assertRaisesRegex(ValueError, "episode_split.*SHA-256"):
                vla_rollout.validate_checkpoint_dataset_binding(metadata, root)

    def test_accepts_exact_hashes_and_returns_action_stats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = {
                "dataset_manifest.json": b"manifest\n",
                "trajectory_expert.jsonl": b"trajectory\n",
                "episode_summary.jsonl": b"summary\n",
                "episode_split.json": b"split\n",
            }
            for name, content in files.items():
                (root / name).write_bytes(content)
            metadata = {
                "dataset_schema_version": "expert_multi_v2",
                "split_schema_version": "episode_split_v2",
                "action_stats": {"mean": [0.0] * 7, "std": [1.0] * 7},
                **{
                    f"{name.replace('.jsonl', '').replace('.json', '')}_sha256":
                    hashlib.sha256(content).hexdigest()
                    for name, content in files.items()
                },
            }

            mean, std = vla_rollout.validate_checkpoint_dataset_binding(metadata, root)

        self.assertEqual(mean.tolist(), [0.0] * 7)
        self.assertEqual(std.tolist(), [1.0] * 7)


class SavedSceneSpecTests(unittest.TestCase):
    def test_extracts_only_saved_scene_pose_robot_and_camera(self):
        summary = {
            "schema_version": "expert_multi_v2",
            "episode_idx": 7,
            "random_seed": 1007,
            "initial_scene_state": {
                "target_block": "blue",
                "blocks": {
                    "red": {
                        "position": [-0.1, 0.4, 0.05],
                        "orientation": [0.0, 0.0, 0.0, 1.0],
                    },
                    "blue": {
                        "position": [0.1, 0.45, 0.05],
                        "orientation": [0.0, 0.0, 0.1, 0.995],
                    },
                },
                "robot": {
                    "joint_positions": [0.1] * 7,
                    "joint_velocities": [0.0] * 7,
                    "ee_position": [0.0, 0.0, 1.0],
                },
                "camera_eye": [1.05, 0.4, 1.65],
            },
        }

        actual = vla_rollout.extract_saved_scene_spec(summary)

        self.assertEqual(actual["episode_idx"], 7)
        self.assertEqual(actual["random_seed"], 1007)
        self.assertEqual(actual["blocks"]["red"]["position"], [-0.1, 0.4, 0.05])
        self.assertEqual(actual["blocks"]["blue"]["position"], [0.1, 0.45, 0.05])
        self.assertEqual(actual["joint_positions"], [0.1] * 7)
        self.assertEqual(actual["camera_eye"], [1.05, 0.4, 1.65])

    def test_rejects_missing_saved_blue_pose(self):
        summary = {
            "schema_version": "expert_multi_v2",
            "episode_idx": 1,
            "random_seed": 1001,
            "initial_scene_state": {
                "blocks": {"red": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}},
                "robot": {"joint_positions": [0] * 7, "joint_velocities": [0] * 7},
                "camera_eye": [1, 0, 1],
            },
        }

        with self.assertRaisesRegex(ValueError, "saved scene"):
            vla_rollout.extract_saved_scene_spec(summary)


if __name__ == "__main__":
    unittest.main()
