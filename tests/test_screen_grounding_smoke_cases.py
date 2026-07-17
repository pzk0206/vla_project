"""验证动态 smoke 案例的资格判定、选择和 PyBullet 编排。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import screen_grounding_smoke_cases
from screen_grounding_smoke_cases import (
    classify_candidate,
    run_candidate,
    run_screening,
    select_qualified_cases,
)


SETTINGS = {
    "num_actions": 4,
    "clear_visibility_threshold": 0.75,
    "max_pose_error": 0.005,
    "max_final_distance_xy": 0.03,
}


def valid_steps():
    """返回五个满足 clear、可达和最终距离要求的观察点。"""
    return [
        {
            "observation_step": step,
            "block_visibility_ratio": 0.80,
            "target_error_3d": 0.004,
            "true_distance_xy": distance,
        }
        for step, distance in enumerate((0.10, 0.08, 0.06, 0.04, 0.02))
    ]


class CandidateClassificationTests(unittest.TestCase):
    def test_five_valid_observations_qualify(self):
        result = classify_candidate(valid_steps(), SETTINGS)

        self.assertTrue(result["qualified"])
        self.assertEqual(result["reason"], "qualified")

    def test_visibility_equal_to_threshold_is_allowed(self):
        rows = valid_steps()
        rows[2]["block_visibility_ratio"] = 0.75

        self.assertTrue(classify_candidate(rows, SETTINGS)["qualified"])

    def test_visibility_below_threshold_is_rejected(self):
        rows = valid_steps()
        rows[2]["block_visibility_ratio"] = 0.749999

        self.assertEqual(
            classify_candidate(rows, SETTINGS)["reason"],
            "visibility_below_threshold",
        )

    def test_start_pose_error_has_specific_reason(self):
        rows = valid_steps()
        rows[0]["target_error_3d"] = 0.005001

        self.assertEqual(
            classify_candidate(rows, SETTINGS)["reason"],
            "start_pose_error",
        )

    def test_motion_pose_error_has_specific_reason(self):
        rows = valid_steps()
        rows[1]["target_error_3d"] = 0.005001

        self.assertEqual(
            classify_candidate(rows, SETTINGS)["reason"],
            "motion_target_error",
        )

    def test_pose_error_equal_to_limit_is_allowed(self):
        rows = valid_steps()
        rows[0]["target_error_3d"] = 0.005
        rows[3]["target_error_3d"] = 0.005

        self.assertTrue(classify_candidate(rows, SETTINGS)["qualified"])

    def test_final_distance_over_three_centimeters_is_rejected(self):
        rows = valid_steps()
        rows[-1]["true_distance_xy"] = 0.030001

        self.assertEqual(
            classify_candidate(rows, SETTINGS)["reason"],
            "final_distance_error",
        )

    def test_final_distance_equal_to_limit_is_allowed(self):
        rows = valid_steps()
        rows[-1]["true_distance_xy"] = 0.03

        self.assertTrue(classify_candidate(rows, SETTINGS)["qualified"])

    def test_exception_is_rejected(self):
        result = classify_candidate([], SETTINGS, error="boom")

        self.assertFalse(result["qualified"])
        self.assertEqual(result["reason"], "candidate_error")
        self.assertEqual(result["error"], "boom")

    def test_incomplete_trace_is_rejected(self):
        result = classify_candidate(valid_steps()[:-1], SETTINGS)

        self.assertFalse(result["qualified"])
        self.assertEqual(result["reason"], "incomplete_trace")

    def test_early_explicit_failure_precedes_incomplete_trace(self):
        rows = valid_steps()[:1]
        rows[0]["block_visibility_ratio"] = 0.70

        self.assertEqual(
            classify_candidate(rows, SETTINGS)["reason"],
            "visibility_below_threshold",
        )


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
                "output_dir": str(output_dir),
                "seed_range": [55, 100],
                "directions": ["left", "right", "front"],
                "num_actions": 4,
                "max_pose_error": 0.005,
                "max_final_distance_xy": 0.03,
            },
        },
    }


class ScreeningRunnerTests(unittest.TestCase):
    def test_candidate_uses_four_real_control_steps_and_five_observations(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        segmentation = np.full((20, 20), 20, dtype=np.int64)
        requested_targets = [
            [0.08, 0.4, 0.2],
            [0.06, 0.4, 0.2],
            [0.04, 0.4, 0.2],
            [0.02, 0.4, 0.2],
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            config = screening_config(temp_dir)
            candidate_dir = Path(temp_dir) / "seed_055_left"
            with (
                patch("screen_grounding_smoke_cases.connect_physics"),
                patch("screen_grounding_smoke_cases.setup_world", return_value=(1, 10)),
                patch("screen_grounding_smoke_cases.load_block", return_value=20),
                patch("screen_grounding_smoke_cases.settle_object"),
                patch("screen_grounding_smoke_cases.get_object_position", return_value=[0.0, 0.4, 0.05]),
                patch("screen_grounding_smoke_cases.build_balanced_ee_positions", return_value={"left": [0.1, 0.4, 0.2]}),
                patch("screen_grounding_smoke_cases.reset_robot_to_target", return_value=[0.1, 0.4, 0.2]),
                patch("screen_grounding_smoke_cases.sample_camera_eye", return_value=[0.0, 0.4, 3.0]),
                patch("screen_grounding_smoke_cases.capture_rgb_and_segmentation", return_value=(image, segmentation)),
                patch("screen_grounding_smoke_cases.cv2.imwrite", return_value=True),
                patch("screen_grounding_smoke_cases.direction_to_target", side_effect=requested_targets) as direction_to_target,
                patch("screen_grounding_smoke_cases.calculate_target_joints", return_value=[0.0] * 7),
                patch("screen_grounding_smoke_cases.apply_joint_targets"),
                patch("screen_grounding_smoke_cases.get_link_position", side_effect=requested_targets),
                patch("screen_grounding_smoke_cases.p.disconnect") as disconnect,
            ):
                result = run_candidate(config, 55, "left", candidate_dir)

        self.assertTrue(result["qualified"])
        self.assertEqual(len(result["steps"]), 5)
        self.assertEqual(direction_to_target.call_count, 4)
        self.assertTrue(
            all(call.args[0] == "left" for call in direction_to_target.call_args_list)
        )
        disconnect.assert_called_once()
        self.assertNotIn(
            "call_openai_compatible_api", screen_grounding_smoke_cases.__dict__
        )

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
                "screen_grounding_smoke_cases.run_candidate",
                side_effect=fake_candidate,
            ) as candidate:
                run_dir, summary = run_screening(config, run_name="run_test")

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
                run_screening(config, run_name="run_test")

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


if __name__ == "__main__":
    unittest.main()
