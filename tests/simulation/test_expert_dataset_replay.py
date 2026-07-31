import json
from pathlib import Path
import tempfile
import unittest

import yaml

from vla_project.simulation.expert_dataset_replay import (
    ReplayFrame,
    load_replay_inputs,
)


class ReplayFrameTests(unittest.TestCase):
    def test_carries_the_state_from_the_saved_step(self):
        frame = ReplayFrame(
            source_row={"episode_idx": 2, "step_idx": 24},
            robot_id=7,
            block_id=8,
            source_camera_eye=[1.0, 0.4, 1.6],
            target_joint_angles=[0.1] * 7,
            target_pos=[0.1, 0.4, 0.2],
            ee_pos=[0.0, 0.3, 0.4],
            block_pos=[0.1, 0.4, 0.05],
            distance_to_target=0.2,
        )

        self.assertEqual(frame.source_row["step_idx"], 24)
        self.assertEqual(frame.target_joint_angles, [0.1] * 7)
        self.assertEqual(frame.distance_to_target, 0.2)


class ReplayInputTests(unittest.TestCase):
    def test_loads_contract_from_the_dataset_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            manifest = {
                "jsonl_name": "trajectory.jsonl",
                "summary_jsonl_name": "summary.jsonl",
            }
            (dataset_dir / "dataset_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (dataset_dir / "config_snapshot.yaml").write_text(
                yaml.safe_dump({"camera": {"image_width": 224}}),
                encoding="utf-8",
            )
            (dataset_dir / "trajectory.jsonl").write_text(
                json.dumps({"episode_idx": 0, "step_idx": 0}) + "\n",
                encoding="utf-8",
            )
            (dataset_dir / "summary.jsonl").write_text(
                json.dumps({"episode_idx": 0}) + "\n",
                encoding="utf-8",
            )

            loaded_manifest, config, frames, summaries = load_replay_inputs(
                dataset_dir
            )

        self.assertEqual(loaded_manifest, manifest)
        self.assertEqual(config["camera"]["image_width"], 224)
        self.assertEqual(frames, [{"episode_idx": 0, "step_idx": 0}])
        self.assertEqual(summaries, [{"episode_idx": 0}])


if __name__ == "__main__":
    unittest.main()
