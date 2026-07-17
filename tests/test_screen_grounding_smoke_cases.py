"""验证动态 smoke 案例的资格判定和确定性选择规则。"""

import unittest

from screen_grounding_smoke_cases import (
    classify_candidate,
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


if __name__ == "__main__":
    unittest.main()
