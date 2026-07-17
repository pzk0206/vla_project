"""验证 grounding 世界坐标策略不读取红块真值，并保护安全边界。"""

import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from grounding_targeting import (
    SmokeSafetyAbort,
    compute_grounding_action,
    load_frozen_calibration,
)


CALIBRATION = {
    "num_clear_calibration_samples": 15,
    "correction_x": 0.02492227406480192,
    "correction_y": -0.019467343494422532,
}
ZERO_CALIBRATION = {
    "num_clear_calibration_samples": 15,
    "correction_x": 0.0,
    "correction_y": 0.0,
}
SETTINGS = {
    "plane_z": 0.0,
    "workspace_x": [-0.30, 0.30],
    "workspace_y": [0.30, 0.70],
    "stop_distance_xy": 0.02,
    "max_target_jump_xy": 0.03,
    "min_progress_xy": 0.005,
    "no_progress_limit": 2,
}


def empty_safety_state():
    """返回互不共享的初始安全状态。"""
    return {
        "previous_target_xy": None,
        "previous_predicted_distance": None,
        "no_progress_count": 0,
    }


class GroundingTargetingTests(unittest.TestCase):
    def compute_from_world(
        self,
        raw_world,
        *,
        ee_pos=(0.0, 0.4, 0.2),
        calibration=ZERO_CALIBRATION,
        safety_state=None,
    ):
        """把相机几何替换为已知交点，只测试策略层行为。"""
        with (
            patch(
                "grounding_targeting.normalized_box_center_to_pixel",
                return_value=(223.5, 223.5),
            ),
            patch(
                "grounding_targeting.pixel_to_world_on_plane",
                return_value=list(raw_world),
            ),
        ):
            return compute_grounding_action(
                red_block_box=[450, 450, 550, 550],
                image_size=(448, 448),
                view_matrix=[1.0] * 16,
                projection_matrix=[1.0] * 16,
                calibration=calibration,
                ee_pos=list(ee_pos),
                safety_state=safety_state or empty_safety_state(),
                settings=SETTINGS,
            )

    def test_decision_interface_cannot_receive_block_truth(self):
        parameters = inspect.signature(compute_grounding_action).parameters

        self.assertNotIn("block_pos", parameters)
        self.assertNotIn("true_block_pos", parameters)

    def test_loads_only_expected_frozen_calibration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.json"
            path.write_text(json.dumps(CALIBRATION), encoding="utf-8")
            loaded = load_frozen_calibration(
                path,
                expected_count=15,
                expected_x=0.02492227406480192,
                expected_y=-0.019467343494422532,
            )

        self.assertEqual(loaded, CALIBRATION)

    def test_applies_correction_and_selects_dominant_axis(self):
        action = self.compute_from_world(
            [0.075, 0.41, 0.0], calibration=CALIBRATION
        )

        self.assertEqual(action["direction"], "right")
        self.assertEqual(action["box_center_pixel"], [223.5, 223.5])
        self.assertAlmostEqual(
            action["corrected_target_world"][0], 0.09992227406480192
        )
        self.assertAlmostEqual(
            action["corrected_target_world"][1], 0.3905326565055775
        )

    def test_stop_requires_both_axes_within_two_centimeters(self):
        action = self.compute_from_world(
            [0.019, 0.38, 0.0], ee_pos=[0.0, 0.4, 0.2]
        )

        self.assertEqual(action["direction"], "stop")

    def test_equal_axis_error_uses_x_as_dominant_axis(self):
        action = self.compute_from_world(
            [0.03, 0.43, 0.0], ee_pos=[0.0, 0.4, 0.2]
        )

        self.assertEqual(action["direction"], "right")

    def test_workspace_boundary_is_inclusive(self):
        for target in (
            [-0.30, 0.50, 0.0],
            [0.30, 0.50, 0.0],
            [0.00, 0.30, 0.0],
            [0.00, 0.70, 0.0],
        ):
            with self.subTest(target=target):
                action = self.compute_from_world(target)
                self.assertEqual(action["corrected_target_world"], target)

    def test_target_outside_workspace_aborts(self):
        for target in (
            [-0.300001, 0.50, 0.0],
            [0.300001, 0.50, 0.0],
            [0.00, 0.299999, 0.0],
            [0.00, 0.700001, 0.0],
        ):
            with self.subTest(target=target):
                with self.assertRaises(SmokeSafetyAbort) as context:
                    self.compute_from_world(target)
                self.assertEqual(context.exception.reason, "target_out_of_workspace")

    def test_target_jump_equal_to_limit_is_allowed(self):
        state = empty_safety_state()
        state["previous_target_xy"] = [0.0, 0.4]

        action = self.compute_from_world([0.03, 0.4, 0.0], safety_state=state)

        self.assertTrue(math.isclose(action["target_jump_xy"], 0.03))

    def test_target_jump_above_limit_aborts(self):
        state = empty_safety_state()
        state["previous_target_xy"] = [0.0, 0.4]

        with self.assertRaises(SmokeSafetyAbort) as context:
            self.compute_from_world([0.030001, 0.4, 0.0], safety_state=state)

        self.assertEqual(context.exception.reason, "target_jump")

    def test_two_consecutive_subthreshold_improvements_abort(self):
        state = {
            "previous_target_xy": [0.1, 0.4],
            "previous_predicted_distance": 0.10,
            "no_progress_count": 0,
        }
        first = self.compute_from_world(
            [0.1, 0.4, 0.0], ee_pos=[0.004, 0.4, 0.2], safety_state=state
        )
        self.assertEqual(first["safety_state"]["no_progress_count"], 1)

        with self.assertRaises(SmokeSafetyAbort) as context:
            self.compute_from_world(
                [0.1, 0.4, 0.0],
                ee_pos=[0.008, 0.4, 0.2],
                safety_state=first["safety_state"],
            )

        self.assertEqual(context.exception.reason, "no_progress")

    def test_one_sufficient_improvement_resets_no_progress_counter(self):
        state = {
            "previous_target_xy": [0.1, 0.4],
            "previous_predicted_distance": 0.10,
            "no_progress_count": 1,
        }

        action = self.compute_from_world(
            [0.1, 0.4, 0.0], ee_pos=[0.006, 0.4, 0.2], safety_state=state
        )

        self.assertEqual(action["safety_state"]["no_progress_count"], 0)

    def test_calibration_rejects_boolean_and_non_finite_values(self):
        for invalid_value in (True, math.nan, math.inf, -math.inf):
            with self.subTest(invalid_value=invalid_value):
                payload = dict(CALIBRATION, correction_x=invalid_value)
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "calibration.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_frozen_calibration(
                            path,
                            expected_count=15,
                            expected_x=0.02492227406480192,
                            expected_y=-0.019467343494422532,
                        )


if __name__ == "__main__":
    unittest.main()
