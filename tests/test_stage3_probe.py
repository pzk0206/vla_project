import unittest

from stage3_probe import (
    decide_direction,
    determine_probe_termination,
    direction_to_target,
)


class DirectionToTargetTests(unittest.TestCase):
    def test_cardinal_directions_move_one_world_axis_step(self):
        ee_pos = [1.0, 2.0, 3.0]
        cases = {
            "left": [0.75, 2.0, 0.5],
            "right": [1.25, 2.0, 0.5],
            "front": [1.0, 2.25, 0.5],
            "back": [1.0, 1.75, 0.5],
        }
        for direction, expected in cases.items():
            with self.subTest(direction=direction):
                self.assertEqual(
                    direction_to_target(direction, ee_pos, 0.5, 0.25),
                    expected,
                )

    def test_stop_keeps_xy_and_uses_hover_height(self):
        self.assertEqual(
            direction_to_target("stop", [1.0, 2.0, 3.0], 0.5, 0.25),
            [1.0, 2.0, 0.5],
        )

    def test_unknown_direction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知方向"):
            direction_to_target("up", [1.0, 2.0, 3.0], 0.5, 0.25)


class DecideDirectionTests(unittest.TestCase):
    def test_unknown_probe_mode_is_rejected(self):
        probe_config = {
            "mode": "heuritsic",
            "stop_distance_xy": 0.02,
        }
        with self.assertRaisesRegex(ValueError, "probe.mode"):
            decide_direction(
                probe_config,
                image_bgr=None,
                ee_pos=[0.0, 0.0, 0.5],
                block_pos=[0.1, 0.1, 0.0],
            )


class DetermineProbeTerminationTests(unittest.TestCase):
    def test_success_precedes_control_step_limit(self):
        self.assertEqual(
            determine_probe_termination(0.02, 79, 80, 0.03),
            "success",
        )

    def test_last_control_step_reports_limit(self):
        self.assertEqual(
            determine_probe_termination(0.5, 79, 80, 0.03),
            "max_control_steps",
        )

    def test_intermediate_step_keeps_running(self):
        self.assertEqual(
            determine_probe_termination(0.5, 10, 80, 0.03),
            "running",
        )


if __name__ == "__main__":
    unittest.main()
