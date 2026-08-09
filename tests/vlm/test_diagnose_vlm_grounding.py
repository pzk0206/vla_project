"""测试 VLM 末端与红块定位诊断的 prompt、解析、画框和断点续跑。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from vla_project.vlm.diagnose_vlm_grounding import (
    build_grounding_prompt,
    diagnose_grounding,
    draw_grounding_boxes,
    parse_grounding_boxes,
)


class OutputBoundaryTests(unittest.TestCase):
    def test_rejects_unsafe_run_name_before_reading_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "vla_project.vlm.diagnose_vlm_grounding.read_jsonl"
        ) as read:
            config = {
                "vlm_evaluation": {
                    "sample_output_dir": "outputs/vlm_samples/samples",
                    "run_output_dir": "outputs/vlm_evaluations",
                    "grounding_run_name": "../escape",
                }
            }
            with self.assertRaisesRegex(ValueError, "one directory component"):
                diagnose_grounding(
                    config,
                    project_root_override=temp_dir,
                )
            read.assert_not_called()


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


class GroundingResumeTests(unittest.TestCase):
    """重复运行时不得为已完成样本再次支付 API 调用成本。"""

    def test_completed_sample_is_skipped_and_seed_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_dir = root / "samples"
            image_path = sample_dir / "images" / "sample.jpg"
            image_path.parent.mkdir(parents=True)
            self.assertTrue(
                cv2.imwrite(
                    str(image_path),
                    np.zeros((16, 16, 3), dtype=np.uint8),
                )
            )
            sample = {
                "sample_id": "seed_42_d020_left",
                "image_path": str(image_path),
                "instruction": "悬停在红色积木上方",
                "expected_direction": "left",
                "random_seed": 42,
            }
            (sample_dir / "samples.jsonl").write_text(
                json.dumps(sample, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            config = {
                "vlm_evaluation": {
                    "sample_output_dir": str(sample_dir),
                    "run_output_dir": str(root / "outputs/vlm_evaluations"),
                    "grounding_run_name": "grounding_test",
                },
                "probe": {"api": {}},
            }
            boxes = {
                "end_effector": [100.0, 100.0, 200.0, 200.0],
                "red_block": [500.0, 500.0, 600.0, 600.0],
            }
            with patch(
                "vla_project.vlm.diagnose_vlm_grounding.call_openai_compatible_api",
                return_value=(boxes, "{}"),
            ) as api_mock:
                first_dir, first_results = diagnose_grounding(
                    config,
                    limit=1,
                    project_root_override=root,
                )
                second_dir, second_results = diagnose_grounding(
                    config,
                    limit=1,
                    project_root_override=root,
                )

            self.assertEqual(api_mock.call_count, 1)
            self.assertEqual(first_dir, second_dir)
            self.assertEqual(first_results[0]["random_seed"], 42)
            self.assertEqual(second_results[0]["random_seed"], 42)
            lines = (first_dir / "grounding_predictions.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(lines), 1)


if __name__ == "__main__":
    unittest.main()
