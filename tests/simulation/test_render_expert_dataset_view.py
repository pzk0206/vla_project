import copy
from pathlib import Path
import tempfile
import unittest

from vla_project.simulation.expert_dataset_replay import ReplayFrame
from vla_project.simulation.render_expert_dataset_view import (
    DerivationValidationError,
    build_topdown_config,
    derive_frame_row,
    derive_manifest,
    derive_summary_row,
    validate_episode_termination,
    validate_replay_values,
)


def source_config():
    return {
        "camera": {
            "workspace_center": [0.0, 0.4, 0.0],
            "eye_offset_base": [1.05, 0.0, 1.65],
            "eye_offset_random_range": [-0.1, 0.1],
            "up_vector": [0, 0, 1],
            "image_width": 224,
            "image_height": 224,
            "fov": 50,
            "near_val": 0.1,
            "far_val": 100.0,
        },
        "vlm_evaluation": {
            "camera_override": {
                "image_width": 448,
                "image_height": 448,
                "eye_offset_base": [0.0, 0.0, 3.0],
                "eye_offset_random_range": [0.0, 0.0],
                "up_vector": [0, 1, 0],
                "fov": 45,
            }
        },
        "dataset": {
            "output_dir": "outputs/dataset/expert_scaling_v1",
            "schema_version": "expert_v1",
            "random_seed": 1000,
            "capture_interval_steps": 24,
        },
        "robot": {"controlled_joints": 7},
        "task": {"hover_height": 0.15},
    }


def source_row():
    return {
        "schema_version": "expert_v1",
        "episode_idx": 2,
        "step_idx": 24,
        "random_seed": 1002,
        "image_path": "outputs/dataset/expert_scaling_v1/ep_2_step_24.jpg",
        "instruction": "悬停在红色积木上方",
        "action": [0.1] * 7 + [1.0, 0],
        "camera_eye": [1.0, 0.4, 1.6],
        "block_pos": [0.1, 0.4, 0.05],
        "target_pos": [0.1, 0.4, 0.2],
        "ee_pos": [0.0, 0.3, 0.4],
        "distance_to_target": 0.2,
        "termination_reason": "running",
    }


def replay_frame():
    row = source_row()
    return ReplayFrame(
        source_row=row,
        robot_id=7,
        block_id=8,
        source_camera_eye=list(row["camera_eye"]),
        target_joint_angles=[0.1] * 7,
        target_pos=list(row["target_pos"]),
        ee_pos=list(row["ee_pos"]),
        block_pos=list(row["block_pos"]),
        distance_to_target=row["distance_to_target"],
    )


class CameraContractTests(unittest.TestCase):
    def test_builds_frozen_vlm_topdown_camera_without_mutating_source(self):
        original = source_config()
        before = copy.deepcopy(original)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "expert_topdown_v1"
            derived, eye = build_topdown_config(original, output_dir)

        self.assertEqual(original, before)
        self.assertEqual(derived["camera"]["image_width"], 448)
        self.assertEqual(derived["camera"]["image_height"], 448)
        self.assertEqual(
            derived["camera"]["eye_offset_random_range"], [0.0, 0.0]
        )
        self.assertEqual(derived["camera"]["up_vector"], [0, 1, 0])
        self.assertEqual(derived["camera"]["fov"], 45)
        self.assertEqual(eye, [0.0, 0.4, 3.0])
        self.assertEqual(
            derived["dataset"]["schema_version"], "expert_view_v1"
        )
        self.assertEqual(
            derived["dataset"]["output_dir"], str(output_dir)
        )


class ReplayValueTests(unittest.TestCase):
    def test_accepts_exact_values_and_one_nanometer_absolute_tolerance(self):
        frame = replay_frame()
        row = copy.deepcopy(frame.source_row)
        row["ee_pos"][0] += 1e-9

        validate_replay_values(frame, row)

    def test_rejects_misaligned_joint_target(self):
        frame = replay_frame()
        row = copy.deepcopy(frame.source_row)
        row["action"][0] = 9.0

        with self.assertRaises(DerivationValidationError) as caught:
            validate_replay_values(frame, row)

        self.assertEqual(caught.exception.reason, "replay_action_mismatch")

    def test_rejects_nonfinite_state(self):
        frame = replay_frame()
        row = copy.deepcopy(frame.source_row)
        row["block_pos"][1] = float("nan")

        with self.assertRaises(DerivationValidationError) as caught:
            validate_replay_values(frame, row)

        self.assertEqual(caught.exception.reason, "non_finite_source_value")

    def test_episode_requires_only_the_last_frame_to_terminate(self):
        rows = [source_row(), source_row()]
        rows[0]["step_idx"] = 0
        rows[1]["step_idx"] = 24
        rows[1]["action"][-1] = 1
        rows[1]["termination_reason"] = "success"

        validate_episode_termination(rows, {"termination_reason": "success"})

        rows[0]["action"][-1] = 1
        with self.assertRaises(DerivationValidationError) as caught:
            validate_episode_termination(
                rows, {"termination_reason": "success"}
            )
        self.assertEqual(
            caught.exception.reason, "invalid_episode_termination"
        )


class DerivedRowTests(unittest.TestCase):
    def test_frame_changes_only_view_identity_and_adds_provenance(self):
        row = source_row()
        derived = derive_frame_row(
            row,
            "outputs/dataset/expert_topdown_v1/ep_2_step_24.jpg",
            [0.0, 0.4, 3.0],
        )

        for field in (
            "episode_idx",
            "step_idx",
            "random_seed",
            "instruction",
            "action",
            "block_pos",
            "target_pos",
            "ee_pos",
            "distance_to_target",
            "termination_reason",
        ):
            self.assertEqual(derived[field], row[field])
        self.assertEqual(derived["schema_version"], "expert_view_v1")
        self.assertEqual(derived["camera_eye"], [0.0, 0.4, 3.0])
        self.assertEqual(derived["source_image_path"], row["image_path"])
        self.assertEqual(derived["source_camera_eye"], row["camera_eye"])

    def test_summary_and_manifest_preserve_expert_results(self):
        summary = {
            "schema_version": "expert_v1",
            "episode_idx": 2,
            "random_seed": 1002,
            "num_steps": 25,
            "num_frames": 2,
            "final_distance": 0.02,
            "termination_reason": "success",
            "camera_eye": [1.0, 0.4, 1.6],
        }
        derived_summary = derive_summary_row(
            summary, [0.0, 0.4, 3.0]
        )
        self.assertEqual(derived_summary["num_steps"], 25)
        self.assertEqual(derived_summary["final_distance"], 0.02)
        self.assertEqual(
            derived_summary["source_camera_eye"], [1.0, 0.4, 1.6]
        )

        manifest = {
            "schema_version": "expert_v1",
            "dataset_name": "expert_scaling_v1",
            "image_width": 224,
            "image_height": 224,
            "action_dim": 9,
            "target_num_episodes": 300,
        }
        derived_manifest = derive_manifest(
            manifest,
            "outputs/dataset/expert_scaling_v1",
            {"source_manifest_sha256": "abc"},
        )
        self.assertEqual(derived_manifest["action_dim"], 9)
        self.assertEqual(
            derived_manifest["dataset_name"], "expert_topdown_v1"
        )
        self.assertTrue(derived_manifest["derived_read_only"])
        self.assertEqual(derived_manifest["image_width"], 448)


if __name__ == "__main__":
    unittest.main()
