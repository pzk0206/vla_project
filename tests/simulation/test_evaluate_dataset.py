"""测试 expert_v1 数据集的只读质量扫描。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from vla_project.simulation.evaluate_dataset import (
    evaluate_dataset,
    evaluate_pilot_gate,
    evaluate_scale_gate,
    main,
    select_active_gate,
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
            "target_num_episodes": 300,
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
        self.assertEqual(report["active_gate"], "pilot")
        self.assertTrue(report["passed"])
        self.assertFalse(report["scale_gate"]["passed"])
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

    def valid_scale_report(self):
        report = {
            "valid_episode_count": 300,
            "success_rate": 0.99,
            "block_position": {
                "x_bin_counts": [60, 60, 60, 60, 60],
                "y_bin_counts": [60, 60, 60, 60, 60],
            },
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

    def test_scale_gate_accepts_clean_covered_target_dataset(self):
        gate = evaluate_scale_gate(
            self.valid_scale_report(),
            {"target_num_episodes": 300},
        )

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["failed_checks"], [])

    def test_scale_gate_rejects_size_rate_integrity_and_coverage_failures(self):
        corruptions = {
            "valid_episode_count": 299,
            "success_rate": 0.989,
            "schema_error_count": 1,
            "x_bin_counts": [0, 75, 75, 75, 75],
            "y_bin_counts": [75, 75, 75, 75, 0],
        }
        for field, value in corruptions.items():
            with self.subTest(field=field):
                report = self.valid_scale_report()
                if field in ("x_bin_counts", "y_bin_counts"):
                    report["block_position"][field] = value
                else:
                    report[field] = value

                gate = evaluate_scale_gate(
                    report,
                    {"target_num_episodes": 300},
                )

                self.assertFalse(gate["passed"])
                self.assertTrue(gate["failed_checks"])


    def report(self):
        return {
            "num_episodes": 1,
            "success_rate": 1.0,
            "errors": [],
            "active_gate": "pilot",
            "passed": True,
        }

    @patch("vla_project.simulation.evaluate_dataset.write_quality_report")
    @patch("vla_project.simulation.evaluate_dataset.evaluate_dataset")
    def test_explicit_dataset_dir_bypasses_config(
        self, evaluate_mock, write_mock
    ):
        evaluate_mock.return_value = self.report()

        main(["--dataset-dir", "outputs/dataset/expert_topdown_v1"])

        evaluate_mock.assert_called_once_with(
            Path("outputs/dataset/expert_topdown_v1")
        )

    @patch("vla_project.simulation.evaluate_dataset.write_quality_report")
    @patch("vla_project.simulation.evaluate_dataset.evaluate_dataset")
    def test_default_still_reads_dataset_dir_from_config(
        self, evaluate_mock, write_mock
    ):
        evaluate_mock.return_value = self.report()
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                "dataset:\n  output_dir: outputs/dataset/original\n",
                encoding="utf-8",
            )
            with patch(
                "vla_project.simulation.evaluate_dataset.CONFIG_PATH",
                str(config_path),
            ):
                main([])

        evaluate_mock.assert_called_once_with(
            Path("outputs/dataset/original")
        )

    def test_selects_scale_gate_only_at_target_size(self):
        active_gate, passed = select_active_gate(
            {"num_episodes": 300},
            {"target_num_episodes": 300},
            {"passed": False},
            {"passed": True},
        )

        self.assertEqual(active_gate, "scale")
        self.assertTrue(passed)

    def test_intermediate_dataset_does_not_pass_pilot_gate(self):
        active_gate, passed = select_active_gate(
            {"num_episodes": 11},
            {"target_num_episodes": 300},
            {"passed": False},
            {"passed": False},
        )

        self.assertEqual(active_gate, "pilot")
        self.assertFalse(passed)

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


class ExpertMultiV2Tests(unittest.TestCase):
    def scene(self, target="red", red=None, blue=None, ee=None):
        return {
            "target_block": target,
            "blocks": {
                "red": {
                    "position": red or [-0.15, 0.40, 0.05],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
                "blue": {
                    "position": blue or [0.15, 0.40, 0.05],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
            },
            "robot": {
                "joint_positions": [0.0] * 7,
                "joint_velocities": [0.0] * 7,
                "ee_position": ee or [0.0, 0.0, 1.261],
            },
            "camera_eye": [1.05, 0.4, 1.65],
        }

    def manifest(self):
        return {
            "schema_version": "expert_multi_v2",
            "action_dim": 9,
            "random_seed": 1000,
            "pilot_num_episodes": 2,
            "target_num_episodes": 300,
            "image_width": 224,
            "image_height": 224,
            "jsonl_name": "trajectory_expert.jsonl",
            "summary_jsonl_name": "episode_summary.jsonl",
            "pair_sampling": {
                "min_axis_separation_xy": 0.12,
                "max_episode_drift_xy": 0.005,
            },
            "tasks": [
                {"instruction": "悬停在红色积木上方", "target_block": "red"},
                {"instruction": "悬停在蓝色积木上方", "target_block": "blue"},
            ],
        }

    def frame(self, episode_idx, target):
        block = [-0.15, 0.40, 0.05] if target == "red" else [0.15, 0.40, 0.05]
        instruction = f"悬停在{'红' if target == 'red' else '蓝'}色积木上方"
        return {
            "schema_version": "expert_multi_v2",
            "episode_idx": episode_idx,
            "step_idx": 0,
            "random_seed": 1000 + episode_idx,
            "image_path": f"ep_{episode_idx}_step_0.jpg",
            "instruction": instruction,
            "action": [0.0] * 7 + [1.0, 1],
            "camera_eye": [1.05, 0.4, 1.65],
            "block_pos": block,
            "target_pos": [block[0], block[1], 0.20],
            "ee_pos": [block[0], block[1], 0.19],
            "distance_to_target": 0.01,
            "termination_reason": "success",
            "scene_state": self.scene(target=target),
        }

    def summary(self, episode_idx, target):
        frame = self.frame(episode_idx, target)
        state = frame["scene_state"]
        return {
            "schema_version": "expert_multi_v2",
            "episode_idx": episode_idx,
            "random_seed": 1000 + episode_idx,
            "initial_ee_pos": [0.0, 0.0, 1.261],
            "initial_block_pos": frame["block_pos"],
            "num_steps": 1,
            "num_frames": 1,
            "final_distance": 0.01,
            "termination_reason": "success",
            "camera_eye": [1.05, 0.4, 1.65],
            "final_block_pos": frame["block_pos"],
            "final_target_pos": frame["target_pos"],
            "final_ee_pos": frame["ee_pos"],
            "task_instruction": frame["instruction"],
            "target_block": target,
            "initial_scene_state": state,
            "final_scene_state": state,
        }

    def make_dataset(self, root, frames=None, summaries=None):
        frames = frames or [self.frame(0, "red"), self.frame(1, "blue")]
        summaries = summaries or [self.summary(0, "red"), self.summary(1, "blue")]
        (root / "dataset_manifest.json").write_text(
            json.dumps(self.manifest(), ensure_ascii=False), encoding="utf-8"
        )
        (root / "trajectory_expert.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in frames),
            encoding="utf-8",
        )
        (root / "episode_summary.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in summaries),
            encoding="utf-8",
        )
        for row in frames:
            cv2.imwrite(
                str(root / row["image_path"]),
                np.zeros((224, 224, 3), dtype=np.uint8),
            )

    def test_valid_v2_dataset_reports_scene_metrics_and_balanced_pilot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_dataset(root)
            report = evaluate_dataset(root)

        self.assertTrue(report["passed"])
        self.assertEqual(report["task_counts"], {"red": 1, "blue": 1})
        self.assertEqual(report["scene_state_error_count"], 0)
        self.assertEqual(report["target_consistency_error_count"], 0)
        self.assertEqual(report["block_overlap_error_count"], 0)
        self.assertEqual(report["block_drift_error_count"], 0)
        self.assertGreaterEqual(report["pair_min_axis_separation_stats"]["min"], 0.12)
        self.assertLessEqual(report["block_xy_drift_stats"]["max"], 0.005)

    def test_v2_scan_counts_scene_target_overlap_and_drift_errors(self):
        frames = [self.frame(0, "red"), self.frame(1, "blue")]
        summaries = [self.summary(0, "red"), self.summary(1, "blue")]
        frames[0]["scene_state"]["blocks"].pop("blue")
        frames[1]["instruction"] = "悬停在红色积木上方"
        summaries[0]["initial_scene_state"]["blocks"]["blue"]["position"] = [-0.10, 0.40, 0.05]
        summaries[1]["final_scene_state"] = self.scene(
            target="blue", blue=[0.16, 0.40, 0.05]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_dataset(root, frames, summaries)
            report = evaluate_dataset(root)

        self.assertFalse(report["passed"])
        self.assertGreater(report["scene_state_error_count"], 0)
        self.assertGreater(report["target_consistency_error_count"], 0)
        self.assertGreater(report["block_overlap_error_count"], 0)
        self.assertGreater(report["block_drift_error_count"], 0)

    def test_v2_scene_rejects_nonfinite_pose_bad_quaternion_and_joint_shape(self):
        corruptions = {
            "nonfinite_position": lambda state: state["blocks"]["red"].update(
                {"position": [float("nan"), 0.40, 0.05]}
            ),
            "nonunit_quaternion": lambda state: state["blocks"]["red"].update(
                {"orientation": [0.0, 0.0, 0.0, 2.0]}
            ),
            "joint_shape": lambda state: state["robot"].update(
                {"joint_positions": [0.0] * 6}
            ),
        }
        for name, corrupt in corruptions.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                frames = [self.frame(0, "red"), self.frame(1, "blue")]
                corrupt(frames[0]["scene_state"])
                self.make_dataset(root, frames=frames)

                report = evaluate_dataset(root)

                self.assertGreater(report["scene_state_error_count"], 0)
                self.assertFalse(report["passed"])

    def gate_report(self, episodes, red, blue):
        report = {
            "num_episodes": episodes,
            "success_count": episodes,
            "valid_episode_count": episodes,
            "success_rate": 1.0,
            "task_counts": {"red": red, "blue": blue},
            "block_position": {
                "x_bin_counts": [60] * 5,
                "y_bin_counts": [60] * 5,
            },
        }
        for field in (
            "schema_error_count", "action_dim_error_count", "missing_image_count",
            "unreadable_image_count", "image_size_mismatch_count", "orphan_image_count",
            "duplicate_step_key_count", "seed_error_count", "frame_count_mismatch_count",
            "terminal_flag_error_count", "scene_state_error_count",
            "target_consistency_error_count", "block_overlap_error_count",
            "block_drift_error_count",
        ):
            report[field] = 0
        return report

    def test_v2_gates_require_exact_task_balance(self):
        pilot_manifest = {**self.manifest(), "pilot_num_episodes": 10}
        self.assertTrue(
            evaluate_pilot_gate(self.gate_report(10, 5, 5), pilot_manifest)["passed"]
        )
        self.assertFalse(
            evaluate_pilot_gate(self.gate_report(10, 6, 4), pilot_manifest)["passed"]
        )
        self.assertTrue(
            evaluate_scale_gate(self.gate_report(300, 150, 150), self.manifest())["passed"]
        )
        self.assertFalse(
            evaluate_scale_gate(self.gate_report(300, 151, 149), self.manifest())["passed"]
        )


if __name__ == "__main__":
    unittest.main()
