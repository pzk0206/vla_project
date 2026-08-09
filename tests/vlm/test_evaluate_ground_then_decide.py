"""测试结构化 grounding 后方向决策的解析和一致性判断。"""

import tempfile
import unittest
from unittest.mock import patch

from vla_project.vlm.evaluate_ground_then_decide import (
    build_ground_then_decide_prompt,
    direction_from_grounding_boxes,
    evaluate_ground_then_decide,
    parse_ground_then_decide,
)


class OutputBoundaryTests(unittest.TestCase):
    def test_rejects_unsafe_run_name_before_reading_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "vla_project.vlm.evaluate_ground_then_decide.read_jsonl"
        ) as read:
            config = {
                "vlm_evaluation": {
                    "sample_output_dir": "outputs/vlm_samples/samples",
                    "run_output_dir": "outputs/vlm_evaluations",
                    "ground_then_decide_run_name": "/tmp/escape",
                }
            }
            with self.assertRaisesRegex(ValueError, "one directory component"):
                evaluate_ground_then_decide(
                    config,
                    project_root_override=temp_dir,
                )
            read.assert_not_called()


class GroundThenDecidePromptTests(unittest.TestCase):
    def test_prompt_requires_boxes_before_direction_without_truth(self):
        prompt = build_ground_then_decide_prompt("悬停在红色积木上方")

        self.assertIn('"end_effector"', prompt)
        self.assertIn('"red_block"', prompt)
        self.assertIn('"direction"', prompt)
        self.assertIn("先完成两个框", prompt)
        self.assertNotIn("block_pos", prompt)
        self.assertNotIn("expected_direction", prompt)


class ParseGroundThenDecideTests(unittest.TestCase):
    def test_parses_valid_structured_response(self):
        parsed = parse_ground_then_decide(
            '{"end_effector":[600,400,700,500],'
            '"red_block":[400,400,500,500],'
            '"direction":"screen_left"}'
        )

        self.assertEqual(parsed["direction"], "screen_left")
        self.assertEqual(parsed["end_effector"], [600.0, 400.0, 700.0, 500.0])

    def test_rejects_invalid_direction(self):
        self.assertIsNone(
            parse_ground_then_decide(
                '{"end_effector":[600,400,700,500],'
                '"red_block":[400,400,500,500],"direction":"left"}'
            )
        )


class DirectionFromGroundingBoxesTests(unittest.TestCase):
    def test_uses_axis_with_larger_center_distance(self):
        self.assertEqual(
            direction_from_grounding_boxes(
                {
                    "end_effector": [600, 400, 700, 500],
                    "red_block": [400, 420, 500, 520],
                }
            ),
            "screen_left",
        )
        self.assertEqual(
            direction_from_grounding_boxes(
                {
                    "end_effector": [500, 300, 600, 400],
                    "red_block": [510, 600, 610, 700],
                }
            ),
            "screen_down",
        )


if __name__ == "__main__":
    unittest.main()
