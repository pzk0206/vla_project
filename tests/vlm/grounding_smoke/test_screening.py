"""验证动态 smoke 案例的资格判定、选择和 PyBullet 编排。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import vla_project.vlm.grounding_smoke.screening
from vla_project.vlm.grounding_smoke.screening import (
    _make_run_dir,
    classify_candidate,
    run_candidate,
    run_screening,
    select_qualified_cases,
)


SETTINGS = {
    "clear_visibility_threshold": 0.75,
    "max_pose_error": 0.005,
}


class RunDirectoryTests(unittest.TestCase):
    def test_rejects_unsafe_run_names_before_creating_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "outputs/vlm_evaluations/screening"
            for run_name in ("../escape", "a/b", "/tmp/escape"):
                with self.subTest(run_name=run_name), patch.object(
                    Path, "mkdir"
                ) as mkdir:
                    with self.assertRaises(ValueError):
                        _make_run_dir(
                            output_root,
                            run_name,
                            project_root_override=root,
                        )
                    mkdir.assert_not_called()

    def test_creates_safe_run_below_managed_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "outputs/vlm_evaluations/screening"

            created = _make_run_dir(
                output_root,
                "run_fixed",
                project_root_override=root,
            )

            self.assertEqual(created, output_root / "run_fixed")
            self.assertTrue(created.is_dir())


def valid_observation():
    return {
        "target_error_3d": 0.004,
        "block_visibility_ratio": 0.80,
    }


class CandidateClassificationTests(unittest.TestCase):
    def test_initial_observation_qualifies_at_inclusive_boundaries(self):
        observation = valid_observation()
        observation["target_error_3d"] = 0.005
        observation["block_visibility_ratio"] = 0.75

        result = classify_candidate(observation, SETTINGS)

        self.assertTrue(result["qualified"])
        self.assertEqual(result["reason"], "qualified")

    def test_start_pose_error_strictly_over_limit_is_rejected(self):
        observation = valid_observation()
        observation["target_error_3d"] = 0.005001

        result = classify_candidate(observation, SETTINGS)

        self.assertFalse(result["qualified"])
        self.assertEqual(result["reason"], "start_pose_error")

    def test_visibility_strictly_below_threshold_is_rejected(self):
        observation = valid_observation()
        observation["block_visibility_ratio"] = 0.749999

        result = classify_candidate(observation, SETTINGS)

        self.assertFalse(result["qualified"])
        self.assertEqual(result["reason"], "visibility_below_threshold")

    def test_start_pose_reason_precedes_visibility_reason(self):
        observation = {
            "target_error_3d": 0.006,
            "block_visibility_ratio": 0.50,
        }

        result = classify_candidate(observation, SETTINGS)

        self.assertEqual(result["reason"], "start_pose_error")

    def test_exception_is_rejected_with_evidence(self):
        result = classify_candidate(None, SETTINGS, error="boom")

        self.assertFalse(result["qualified"])
        self.assertEqual(result["reason"], "candidate_error")
        self.assertEqual(result["error"], "boom")

    def test_missing_observation_is_candidate_error(self):
        result = classify_candidate(None, SETTINGS)

        self.assertFalse(result["qualified"])
        self.assertEqual(result["reason"], "candidate_error")
        self.assertEqual(result["error"], "missing_initial_observation")


class SelectionTests(unittest.TestCase):
    def test_selects_smallest_distinct_seed_in_direction_order(self):
        candidates = [
            {"seed": 55, "direction": direction, "qualified": True}
            for direction in ("left", "right", "front")
        ] + [
            {"seed": 56, "direction": "right", "qualified": True},
            {"seed": 57, "direction": "front", "qualified": True},
        ]

        selected = select_qualified_cases(
            candidates, ["left", "right", "front"]
        )

        self.assertEqual(
            [(row["seed"], row["direction"]) for row in selected],
            [(55, "left"), (56, "right"), (57, "front")],
        )

    def test_unqualified_lower_seed_is_skipped(self):
        candidates = [
            {"seed": 55, "direction": "left", "qualified": False},
            {"seed": 56, "direction": "left", "qualified": True},
        ]

        selected = select_qualified_cases(candidates, ["left"])

        self.assertEqual(selected[0]["seed"], 56)

    def test_missing_direction_raises_without_partial_selection(self):
        with self.assertRaises(ValueError):
            select_qualified_cases(
                [{"seed": 55, "direction": "left", "qualified": True}],
                ["left", "right", "front"],
            )


def screening_config(output_dir):
    """构造 runner 测试所需的最小配置。"""
    return {
        "connection_mode": "DIRECT",
        "gravity": [0, 0, -9.8],
        "enable_time_sleep": False,
        "simulation_hz": 240,
        "robot": {"ee_link_index": 6, "controlled_joints": 7},
        "task": {"initial_settle_steps": 1},
        "probe": {"sim_steps_per_action": 1},
        "camera": {"workspace_center": [0.0, 0.4, 0.0]},
        "vlm_evaluation": {
            "balanced_pose_tolerance": 0.005,
            "balanced_pose_ik_iterations": 20,
            "camera_override": {
                "image_width": 448,
                "image_height": 448,
                "eye_offset_base": [0.0, 0.0, 3.0],
                "eye_offset_random_range": [0.0, 0.0],
                "up_vector": [0, 1, 0],
                "fov": 45,
            },
        },
        "grounding_smoke": {
            "start_offset_xy": 0.10,
            "hover_z": 0.20,
            "move_step_xy": 0.02,
            "visibility_reference_pixels": 378,
            "clear_visibility_threshold": 0.75,
            "screening": {
                "output_dir": str(
                    Path(output_dir)
                    / "outputs/vlm_evaluations/grounding_smoke_screening"
                ),
                "seed_range": [55, 100],
                "directions": ["left", "right", "front"],
                "max_pose_error": 0.005,
            },
        },
    }


class ScreeningRunnerTests(unittest.TestCase):
    def test_candidate_uses_only_the_initial_observation(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        segmentation = np.full((20, 20), 20, dtype=np.int64)
        with tempfile.TemporaryDirectory() as temp_dir:
            config = screening_config(temp_dir)
            candidate_dir = Path(temp_dir) / "seed_055_left"
            with (
                patch(
                    "vla_project.vlm.grounding_smoke.screening.connect_physics"
                ),
                patch(
                    "vla_project.vlm.grounding_smoke.screening.setup_world",
                    return_value=(1, 10),
                ),
                patch(
                    "vla_project.vlm.grounding_smoke.screening.load_block",
                    return_value=20,
                ),
                patch(
                    "vla_project.vlm.grounding_smoke.screening.settle_object"
                ),
                patch(
                    "vla_project.vlm.grounding_smoke.screening.get_object_position",
                    return_value=[0.0, 0.4, 0.05],
                ),
                patch(
                    "vla_project.vlm.grounding_smoke.screening.build_balanced_ee_positions",
                    return_value={"left": [0.1, 0.4, 0.2]},
                ),
                patch(
                    "vla_project.vlm.grounding_smoke.screening.reset_robot_to_target",
                    return_value=[0.1, 0.4, 0.2],
                ),
                patch(
                    "vla_project.vlm.grounding_smoke.screening.sample_camera_eye",
                    return_value=[0.0, 0.4, 3.0],
                ),
                patch(
                    "vla_project.vlm.grounding_smoke.screening.capture_rgb_and_segmentation",
                    return_value=(image, segmentation),
                ) as capture,
                patch(
                    "vla_project.vlm.grounding_smoke.screening.cv2.imwrite",
                    return_value=True,
                ),
                patch(
                    "vla_project.vlm.grounding_smoke.screening.p.disconnect"
                ) as disconnect,
            ):
                result = run_candidate(config, 55, "left", candidate_dir)

        self.assertTrue(result["qualified"])
        self.assertEqual(result["reason"], "qualified")
        self.assertEqual(result["seed"], 55)
        self.assertEqual(result["direction"], "left")
        self.assertEqual(result["target_error_3d"], 0.0)
        self.assertEqual(result["block_visibility_ratio"], 1.0)
        self.assertTrue(result["image_path"].endswith("step_00.jpg"))
        capture.assert_called_once()
        disconnect.assert_called_once()
        self.assertNotIn(
            "call_openai_compatible_api",
            vla_project.vlm.grounding_smoke.screening.__dict__,
        )

    def test_candidate_error_is_recorded_and_physics_is_released(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = screening_config(temp_dir)
            candidate_dir = Path(temp_dir) / "seed_055_left"
            with (
                patch(
                    "vla_project.vlm.grounding_smoke.screening.connect_physics"
                ),
                patch(
                    "vla_project.vlm.grounding_smoke.screening.setup_world",
                    side_effect=RuntimeError("setup failed"),
                ),
                patch(
                    "vla_project.vlm.grounding_smoke.screening.p.disconnect"
                ) as disconnect,
            ):
                result = run_candidate(config, 55, "left", candidate_dir)

        self.assertFalse(result["qualified"])
        self.assertEqual(result["reason"], "candidate_error")
        self.assertIn("setup failed", result["error"])
        disconnect.assert_called_once()

    def test_screening_evaluates_full_cartesian_product_and_writes_summary(self):
        def fake_candidate(config, seed, direction, candidate_dir):
            qualified = (
                (direction == "left" and seed == 55)
                or (direction == "right" and seed in {55, 56})
                or (direction == "front" and seed in {55, 56, 57})
            )
            candidate_dir.mkdir(parents=True, exist_ok=False)
            return {
                "seed": seed,
                "direction": direction,
                "steps": [],
                "qualified": qualified,
                "reason": "qualified" if qualified else "visibility_below_threshold",
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            config = screening_config(temp_dir)
            with patch(
                "vla_project.vlm.grounding_smoke.screening.run_candidate",
                side_effect=fake_candidate,
            ) as candidate:
                run_dir, summary = run_screening(
                    config,
                    run_name="run_test",
                    project_root_override=temp_dir,
                )

            trace_rows = [
                json.loads(line)
                for line in (run_dir / "candidate_trace.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            saved_summary = json.loads(
                (run_dir / "screening_summary.json").read_text(encoding="utf-8")
            )
            with self.assertRaises(FileExistsError):
                run_screening(
                    config,
                    run_name="run_test",
                    project_root_override=temp_dir,
                )

        self.assertEqual(candidate.call_count, 138)
        self.assertEqual(len(trace_rows), 138)
        self.assertEqual(summary["num_candidates"], 138)
        self.assertEqual(saved_summary, summary)
        self.assertEqual(
            [
                (row["seed"], row["direction"])
                for row in summary["selected_cases"]
            ],
            [(55, "left"), (56, "right"), (57, "front")],
        )
        self.assertEqual(
            summary["selection_rule"],
            {
                "seed_range": [55, 100],
                "directions": ["left", "right", "front"],
                "max_pose_error": 0.005,
                "clear_visibility_threshold": 0.75,
                "observation_scope": "initial_only",
                "distinct_seeds": True,
                "selection": "smallest_seed_in_direction_order",
            },
        )


if __name__ == "__main__":
    unittest.main()
