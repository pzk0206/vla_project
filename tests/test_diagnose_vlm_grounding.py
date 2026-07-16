"""测试 VLM 末端与红块定位诊断的 prompt、解析和画框。"""

import unittest

import numpy as np

from diagnose_vlm_grounding import (
    build_grounding_prompt,
    draw_grounding_boxes,
    parse_grounding_boxes,
)


class GroundingPromptTests(unittest.TestCase):
    def test_prompt_requests_structure_based_boxes_without_truth_leakage(self):
        prompt = build_grounding_prompt("悬停在红色积木上方")

        self.assertIn("沿机械臂的连接关系逐节追踪", prompt)
        self.assertIn('"end_effector"', prompt)
        self.assertIn('"red_block"', prompt)
        self.assertIn("0 到 1000", prompt)
        self.assertNotIn("深灰色", prompt)
        self.assertNotIn("最下端", prompt)


class ParseGroundingBoxesTests(unittest.TestCase):
    def test_parses_json_inside_markdown_fence(self):
        boxes = parse_grounding_boxes(
            '```json\n{"end_effector":[100,200,300,400],'
            '"red_block":[500,600,700,800]}\n```'
        )

        self.assertEqual(boxes["end_effector"], [100.0, 200.0, 300.0, 400.0])
        self.assertEqual(boxes["red_block"], [500.0, 600.0, 700.0, 800.0])

    def test_rejects_missing_or_invalid_boxes(self):
        self.assertIsNone(parse_grounding_boxes('{"end_effector":[1,2,3,4]}'))
        self.assertIsNone(
            parse_grounding_boxes(
                '{"end_effector":[300,200,100,400],'
                '"red_block":[500,600,700,800]}'
            )
        )


class DrawGroundingBoxesTests(unittest.TestCase):
    def test_draws_both_boxes_without_modifying_source(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = {
            "end_effector": [100, 100, 300, 300],
            "red_block": [600, 600, 800, 800],
        }

        annotated = draw_grounding_boxes(image, boxes)

        self.assertFalse(np.any(image))
        self.assertTrue(np.any(annotated[10:31, 10:31]))
        self.assertTrue(np.any(annotated[60:81, 60:81]))


if __name__ == "__main__":
    unittest.main()
