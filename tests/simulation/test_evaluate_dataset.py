"""测试 expert_v1 数据集的只读质量扫描。"""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vla_project.simulation.evaluate_dataset import (
    evaluate_dataset,
    evaluate_pilot_gate,
    write_quality_report,
)


class EvaluateDatasetTests(unittest.TestCase):
    def write_json(self, path, value):
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_jsonl(self, path, rows):
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in rows
            ),
            encoding="utf-8",
        )

    def manifest(self):
        return {
            "schema_version": "expert_v1",
            "action_dim": 9,
            "random_seed": 1000,
            "pilot_num_episodes": 1,
            "image_width": 224,
            "image_height": 224,
            "jsonl_name": "trajectory_expert.jsonl",
            "summary_jsonl_name": "episode_summary.jsonl",
        }

    def frame(self, **overrides):
        row = {
            "schema_version": "expert_v1",
            "episode_idx": 0,
            "step_idx": 24,
            "random_seed": 1000,
            "image_path": "ep_0_step_24.jpg",
            "instruction": "悬停在红色积木上方",
            "action": [0.0] * 7 + [1.0, 1],
            "camera_eye": [1.0, 0.4, 1.6],
            "block_pos": [0.1, 0.4, 0.05],
            "target_pos": [0.1, 0.4, 0.2],
            "ee_pos": [0.1, 0.4, 0.19],
            "distance_to_target": 0.01,
            "termination_reason": "success",
        }
        row.update(overrides)
        return row

    def summary(self, **overrides):
        row = {
            "schema_version": "expert_v1",
            "episode_idx": 0,
            "random_seed": 1000,
            "initial_ee_pos": [0.0, 0.0, 1.261],
            "initial_block_pos": [0.1, 0.4, 0.05],
            "num_steps": 25,
            "num_frames": 1,
            "final_distance": 0.01,
            "termination_reason": "success",
            "camera_eye": [1.0, 0.4, 1.6],
            "final_block_pos": [0.1, 0.4, 0.05],
            "final_target_pos": [0.1, 0.4, 0.2],
            "final_ee_pos": [0.1, 0.4, 0.19],
        }
        row.update(overrides)
        return row

    def make_dataset(self, dataset_dir, frames, summaries):
        self.write_json(
            dataset_dir / "dataset_manifest.json",
            self.manifest(),
        )
        self.write_jsonl(
            dataset_dir / "trajectory_expert.jsonl",
            frames,
        )
        self.write_jsonl(
            dataset_dir / "episode_summary.jsonl",
            summaries,
        )

    def test_evaluates_valid_minimal_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            cv2.imwrite(
                str(dataset_dir / "ep_0_step_24.jpg"),
                np.zeros((224, 224, 3), dtype=np.uint8),
            )
            self.make_dataset(
                dataset_dir,
                [self.frame()],
                [self.summary()],
            )

            report = evaluate_dataset(dataset_dir)

        self.assertEqual(report["num_episodes"], 1)
        self.assertEqual(report["valid_episode_count"], 1)
        self.assertEqual(report["num_frames"], 1)
        self.assertEqual(report["success_count"], 1)
        self.assertEqual(report["success_rate"], 1.0)
        self.assertTrue(report["pilot_gate"]["passed"])
        for field in (
            "schema_error_count",
            "action_dim_error_count",
            "missing_image_count",
            "unreadable_image_count",
            "image_size_mismatch_count",
            "orphan_image_count",
            "duplicate_step_key_count",
            "seed_error_count",
            "frame_count_mismatch_count",
            "terminal_flag_error_count",
        ):
            self.assertEqual(report[field], 0, field)
        self.assertEqual(report["final_distance_stats"]["median"], 0.01)
        self.assertEqual(report["frames_per_episode_stats"]["max"], 1)
        self.assertEqual(report["block_position"]["x_min"], 0.1)
        self.assertEqual(
            report["camera_position"]["min"],
            [1.0, 0.4, 1.6],
        )

    def test_reports_all_corruptions_in_one_scan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            cv2.imwrite(
                str(dataset_dir / "ep_0_step_24.jpg"),
                np.zeros((224, 224, 3), dtype=np.uint8),
            )
            (dataset_dir / "unreadable.jpg").write_text(
                "not a jpeg",
                encoding="utf-8",
            )
            cv2.imwrite(
                str(dataset_dir / "wrong_size.jpg"),
                np.zeros((32, 32, 3), dtype=np.uint8),
            )
            cv2.imwrite(
                str(dataset_dir / "orphan.jpg"),
                np.zeros((224, 224, 3), dtype=np.uint8),
            )
            missing_field = self.frame(
                step_idx=25,
                image_path="missing.jpg",
                action=[0.0] * 8,
                random_seed=999,
                termination_reason="running",
            )
            missing_field.pop("instruction")
            frames = [
                self.frame(),
                self.frame(),
                missing_field,
                self.frame(
                    step_idx=26,
                    image_path="unreadable.jpg",
                    termination_reason="running",
                    action=[0.0] * 7 + [1.0, 0],
                ),
                self.frame(
                    step_idx=27,
                    image_path="wrong_size.jpg",
                    action=[0.0] * 7 + [1.0, 0],
                ),
            ]
            self.make_dataset(
                dataset_dir,
                frames,
                [self.summary(num_frames=99)],
            )

            report = evaluate_dataset(dataset_dir)

        self.assertEqual(report["schema_error_count"], 1)
        self.assertEqual(report["action_dim_error_count"], 1)
        self.assertEqual(report["duplicate_step_key_count"], 1)
        self.assertEqual(report["seed_error_count"], 1)
        self.assertEqual(report["missing_image_count"], 1)
        self.assertEqual(report["unreadable_image_count"], 1)
        self.assertEqual(report["image_size_mismatch_count"], 1)
        self.assertEqual(report["orphan_image_count"], 1)
        self.assertEqual(report["frame_count_mismatch_count"], 1)
        self.assertEqual(report["terminal_flag_error_count"], 1)
        self.assertGreaterEqual(len(report["errors"]), 10)

    def test_empty_numeric_series_return_none_stats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            self.make_dataset(dataset_dir, [], [])

            report = evaluate_dataset(dataset_dir)

        self.assertEqual(
            report["final_distance_stats"],
            {"min": None, "mean": None, "median": None, "max": None},
        )
        self.assertIsNone(report["block_position"]["x_min"])
        self.assertEqual(
            report["camera_position"],
            {"min": None, "max": None},
        )

    def test_writes_quality_report(self):
        report = {"num_episodes": 1, "success_rate": 1.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = write_quality_report(Path(temp_dir), report)
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(output_path.name, "dataset_quality_report.json")
        self.assertEqual(saved, report)

    def valid_pilot_report(self):
        report = {
            "num_episodes": 10,
            "success_count": 10,
        }
        for field in (
            "schema_error_count",
            "action_dim_error_count",
            "missing_image_count",
            "unreadable_image_count",
            "image_size_mismatch_count",
            "orphan_image_count",
            "duplicate_step_key_count",
            "seed_error_count",
            "frame_count_mismatch_count",
            "terminal_flag_error_count",
        ):
            report[field] = 0
        return report

    def test_pilot_gate_accepts_exactly_clean_ten_episode_run(self):
        gate = evaluate_pilot_gate(
            self.valid_pilot_report(),
            {"pilot_num_episodes": 10},
        )

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["failed_checks"], [])

    def test_pilot_gate_rejects_each_failed_contract(self):
        corruptions = {
            "num_episodes": 9,
            "success_count": 9,
            "schema_error_count": 1,
            "action_dim_error_count": 1,
            "missing_image_count": 1,
            "unreadable_image_count": 1,
            "image_size_mismatch_count": 1,
            "orphan_image_count": 1,
            "duplicate_step_key_count": 1,
            "seed_error_count": 1,
            "frame_count_mismatch_count": 1,
            "terminal_flag_error_count": 1,
        }
        for field, value in corruptions.items():
            with self.subTest(field=field):
                report = self.valid_pilot_report()
                report[field] = value
                gate = evaluate_pilot_gate(
                    report,
                    {"pilot_num_episodes": 10},
                )
                self.assertFalse(gate["passed"])
                self.assertTrue(gate["failed_checks"])


if __name__ == "__main__":
    unittest.main()
