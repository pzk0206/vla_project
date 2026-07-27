"""测试 expert_v1 数据集的确定性重放可见性审计。"""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pybullet as p
import yaml

from vla_project.simulation.audit_dataset_visibility import (
    ReplayValidationError,
    build_visibility_rows,
    count_block_pixels,
    read_jsonl,
    replay_episode,
    run_visibility_audit,
    summarize_visibility,
    validate_replay_image,
    validate_episode_contract,
)
from vla_project.simulation.control_arm import (
    apply_joint_targets,
    calculate_target_joints,
    capture_rgb,
    connect_physics,
    get_hover_target,
    load_block,
    load_config,
    reset_robot_to_home,
    sample_camera_eye,
    settle_object,
    setup_world,
)


class VisibilityMathTests(unittest.TestCase):
    def test_counts_block_object_id_while_ignoring_link_bits_and_background(self):
        block_id = 2
        segmentation = np.array(
            [
                [-1, block_id],
                [block_id | (3 << 24), 7],
            ],
            dtype=np.int64,
        )

        self.assertEqual(count_block_pixels(segmentation, block_id), 2)


class ReplayImageTests(unittest.TestCase):
    def test_accepts_same_image_after_default_jpeg_roundtrip(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[8:24, 8:24] = [0, 0, 255]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "frame.jpg"
            self.assertTrue(cv2.imwrite(str(path), image))

            result = validate_replay_image(path, image)

        self.assertTrue(result["replay_exact_match"])
        self.assertEqual(result["replay_pixel_mae"], 0.0)

    def test_rejects_replay_with_changed_pixels(self):
        original = np.zeros((32, 32, 3), dtype=np.uint8)
        replay = np.full((32, 32, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "frame.jpg"
            self.assertTrue(cv2.imwrite(str(path), original))

            with self.assertRaisesRegex(
                ReplayValidationError,
                "replay_image_mismatch",
            ):
                validate_replay_image(path, replay)


def observation(step_idx, pixels):
    return {
        "schema_version": "visibility_audit_v1",
        "episode_idx": 3,
        "random_seed": 1003,
        "step_idx": step_idx,
        "image_path": f"ep_3_step_{step_idx}.jpg",
        "visible_block_pixels": pixels,
        "replay_pixel_mae": 0.0,
        "replay_exact_match": True,
    }


class VisibilityRowTests(unittest.TestCase):
    def test_builds_rows_with_existing_visibility_boundaries(self):
        rows = build_visibility_rows(
            [
                observation(0, 0),
                observation(1, 25),
                observation(2, 75),
                observation(3, 100),
            ],
            reference_pixels=100,
        )

        self.assertEqual(
            [row["visibility_group"] for row in rows],
            ["severe", "partial", "clear", "clear"],
        )
        self.assertEqual(rows[1]["block_visibility_ratio"], 0.25)

    def test_rejects_nonpositive_reference_or_visible_above_reference(self):
        with self.assertRaises(ValueError):
            build_visibility_rows([observation(0, 0)], reference_pixels=0)
        with self.assertRaises(ValueError):
            build_visibility_rows(
                [observation(0, 101)],
                reference_pixels=100,
            )


class VisibilitySummaryTests(unittest.TestCase):
    def test_summarizes_groups_endpoints_and_longest_nonclear_run(self):
        rows = [
            {
                **observation(0, 100),
                "block_visibility_ratio": 1.0,
                "visibility_group": "clear",
                "reference_block_pixels": 100,
            },
            {
                **observation(24, 70),
                "block_visibility_ratio": 0.7,
                "visibility_group": "partial",
                "reference_block_pixels": 100,
            },
            {
                **observation(48, 20),
                "block_visibility_ratio": 0.2,
                "visibility_group": "severe",
                "reference_block_pixels": 100,
            },
            {
                **observation(72, 80),
                "block_visibility_ratio": 0.8,
                "visibility_group": "clear",
                "reference_block_pixels": 100,
            },
        ]

        summary = summarize_visibility(rows)

        self.assertEqual(
            summary["visibility_group_counts"],
            {"clear": 2, "partial": 1, "severe": 1},
        )
        self.assertEqual(summary["episodes_with_nonclear"], 1)
        self.assertEqual(summary["episodes_with_severe"], 1)
        self.assertEqual(
            summary["initial_visibility_group_counts"],
            {"clear": 1},
        )
        self.assertEqual(
            summary["terminal_visibility_group_counts"],
            {"clear": 1},
        )
        self.assertEqual(summary["longest_nonclear_saved_frame_run"], 2)
        self.assertEqual(
            summary["longest_nonclear_run"],
            {
                "episode_idx": 3,
                "start_step": 24,
                "end_step": 48,
                "num_frames": 2,
            },
        )
        self.assertEqual(
            summary["per_episode"][0]["longest_nonclear_run"],
            summary["longest_nonclear_run"],
        )


class ReplayContractTests(unittest.TestCase):
    def test_rejects_seed_camera_and_duplicate_step_mismatches(self):
        manifest = {"random_seed": 1000}
        summary = {"episode_idx": 3, "random_seed": 1003}
        valid = [
            {
                "episode_idx": 3,
                "random_seed": 1003,
                "step_idx": 0,
                "camera_eye": [1.0, 0.4, 1.6],
                "image_path": "ep_3_step_0.jpg",
            }
        ]
        validate_episode_contract(manifest, summary, valid)

        with self.assertRaisesRegex(
            ReplayValidationError,
            "seed_mismatch",
        ):
            validate_episode_contract(
                manifest,
                {**summary, "random_seed": 999},
                valid,
            )
        with self.assertRaisesRegex(
            ReplayValidationError,
            "duplicate_step",
        ):
            validate_episode_contract(manifest, summary, valid + valid)
        with self.assertRaisesRegex(
            ReplayValidationError,
            "camera_mismatch",
        ):
            validate_episode_contract(
                manifest,
                summary,
                valid
                + [
                    {
                        **valid[0],
                        "step_idx": 24,
                        "camera_eye": [1.1, 0.4, 1.6],
                    }
                ],
            )


class ReplayIntegrationTests(unittest.TestCase):
    def test_replays_real_saved_frame_and_recovers_segmentation(self):
        config = load_config("sim_config.yaml")
        config["connection_mode"] = "DIRECT"
        episode_idx = 0
        seed = config["dataset"]["random_seed"]

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            random.seed(seed)
            connect_physics("DIRECT")
            try:
                _, robot_id = setup_world(config)
                reset_robot_to_home(
                    robot_id,
                    config["robot"],
                    config["dataset"],
                )
                for _ in range(config["task"]["initial_settle_steps"]):
                    p.stepSimulation()
                block_id = load_block(config["task"])
                settle_object(
                    config,
                    config["task"]["initial_settle_steps"],
                )
                camera_eye = sample_camera_eye(config["camera"])
                target = get_hover_target(
                    block_id,
                    config["task"]["hover_height"],
                )
                joints = calculate_target_joints(
                    robot_id,
                    config["robot"],
                    target,
                )
                apply_joint_targets(robot_id, config["robot"], joints)
                p.stepSimulation()
                image = capture_rgb(config["camera"], camera_eye)
                image_path = dataset_dir / "ep_0_step_0.jpg"
                self.assertTrue(cv2.imwrite(str(image_path), image))
            finally:
                p.disconnect()

            rows = replay_episode(
                config,
                {"random_seed": seed},
                {"episode_idx": episode_idx, "random_seed": seed},
                [
                    {
                        "episode_idx": episode_idx,
                        "random_seed": seed,
                        "step_idx": 0,
                        "camera_eye": camera_eye,
                        "image_path": str(image_path),
                    }
                ],
            )

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["replay_exact_match"])
        self.assertGreater(rows[0]["reference_block_pixels"], 0)
        self.assertGreater(rows[0]["visible_block_pixels"], 0)


def write_minimal_dataset_inputs(dataset_dir):
    manifest = {
        "schema_version": "expert_v1",
        "random_seed": 1000,
        "jsonl_name": "trajectory_expert.jsonl",
        "summary_jsonl_name": "episode_summary.jsonl",
    }
    frame = {
        "episode_idx": 0,
        "random_seed": 1000,
        "step_idx": 0,
        "camera_eye": [1.0, 0.4, 1.6],
        "image_path": str(dataset_dir / "ep_0_step_0.jpg"),
    }
    summary = {
        "episode_idx": 0,
        "random_seed": 1000,
    }
    (dataset_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (dataset_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump({"connection_mode": "DIRECT"}),
        encoding="utf-8",
    )
    (dataset_dir / "trajectory_expert.jsonl").write_text(
        json.dumps(frame) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "episode_summary.jsonl").write_text(
        json.dumps(summary) + "\n",
        encoding="utf-8",
    )
    cv2.imwrite(
        frame["image_path"],
        np.zeros((32, 32, 3), dtype=np.uint8),
    )


class AuditRunnerTests(unittest.TestCase):
    def test_failure_writes_only_failure_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            output_dir = dataset_dir / "visibility_audit_v1"
            dataset_dir.mkdir()
            write_minimal_dataset_inputs(dataset_dir)

            with patch(
                "vla_project.simulation.audit_dataset_visibility.replay_episode",
                side_effect=ReplayValidationError(
                    "replay_image_mismatch",
                    episode_idx=0,
                    step_idx=0,
                ),
            ):
                with self.assertRaises(ReplayValidationError):
                    run_visibility_audit(dataset_dir, output_dir)

            self.assertTrue(
                (output_dir / "visibility_audit_failure.json").is_file()
            )
            self.assertFalse(
                (output_dir / "frame_visibility.jsonl").exists()
            )
            self.assertFalse(
                (output_dir / "visibility_audit_summary.json").exists()
            )

    def test_success_publishes_complete_rows_and_summary(self):
        replayed = [
            {
                **observation(0, 100),
                "episode_idx": 0,
                "random_seed": 1000,
                "reference_block_pixels": 100,
                "block_visibility_ratio": 1.0,
                "visibility_group": "clear",
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            output_dir = dataset_dir / "visibility_audit_v1"
            dataset_dir.mkdir()
            write_minimal_dataset_inputs(dataset_dir)
            with patch(
                "vla_project.simulation.audit_dataset_visibility.replay_episode",
                return_value=replayed,
            ):
                summary = run_visibility_audit(dataset_dir, output_dir)

            saved_rows = read_jsonl(
                output_dir / "frame_visibility.jsonl"
            )
            saved_summary = json.loads(
                (
                    output_dir / "visibility_audit_summary.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(saved_rows, replayed)
        self.assertTrue(summary["replay_validation"]["passed"])
        self.assertEqual(saved_summary, summary)


if __name__ == "__main__":
    unittest.main()
