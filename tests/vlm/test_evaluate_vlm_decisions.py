"""测试离线 VLM 方向评估中的 prompt 安全边界和统计口径。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from vla_project.vlm.evaluate_vlm_decisions import (
    build_offline_prompt,
    evaluate_offline,
    map_screen_to_world,
    parse_screen_direction,
    summarize_predictions,
)


class OutputBoundaryTests(unittest.TestCase):
    def test_rejects_unsafe_run_name_before_reading_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "vla_project.vlm.evaluate_vlm_decisions.read_jsonl"
        ) as read:
            config = {
                "vlm_evaluation": {
                    "sample_output_dir": "outputs/vlm_samples/samples",
                    "run_output_dir": "outputs/vlm_evaluations",
                    "offline_run_name": "a/b",
                }
            }
            with self.assertRaisesRegex(ValueError, "one directory component"):
                evaluate_offline(
                    config,
                    project_root_override=temp_dir,
                )
            read.assert_not_called()


class BuildOfflinePromptTests(unittest.TestCase):
    """保证 prompt 只使用任务指令，不泄露仿真真值和标准答案。"""

    def test_prompt_excludes_diagnostic_ground_truth(self):
        sample = {
            "instruction": "悬停在红色积木上方",
            "expected_direction": "right",
            "block_pos": [0.123456, 0.456789, 0.1],
            "ee_pos": [0.654321, 0.111111, 0.5],
        }

        prompt = build_offline_prompt(sample)

        self.assertIn(sample["instruction"], prompt)
        self.assertIn("红块主要在末端右侧，输出 screen_right", prompt)
        self.assertIn("红块主要在末端左侧，输出 screen_left", prompt)
        self.assertIn("红块主要在末端下方，输出 screen_down", prompt)
        self.assertIn("红块主要在末端上方，输出 screen_up", prompt)
        self.assertIn("单张正俯视图", prompt)
        # 通过机械臂结构确定末端，不能依赖容易随视角变化的颜色和画面高低。
        self.assertIn("沿相互连接的机械臂连杆逐节追踪", prompt)
        self.assertIn("连接关系中最后一节的末梢", prompt)
        self.assertIn("不是画面中的直线距离", prompt)
        self.assertIn("像素距离绝对值", prompt)
        self.assertNotIn("深灰色", prompt)
        self.assertNotIn("最下端", prompt)
        self.assertNotIn("距离底座最远", prompt)
        # 方向均衡集只测四个移动方向，停止判断应由独立样本评估。
        self.assertNotIn("stop", prompt.lower())
        self.assertNotIn("绿色末端", prompt)
        self.assertNotIn("MAIN", prompt)
        self.assertNotIn("AUX", prompt)
        self.assertNotIn("0.123456", prompt)
        self.assertNotIn("0.654321", prompt)


class ScreenDirectionMappingTests(unittest.TestCase):
    """固定图像坐标标签到 PyBullet 世界动作的确定性转换。"""

    def test_parses_screen_direction_from_model_text(self):
        self.assertEqual(parse_screen_direction("screen_right"), "screen_right")
        self.assertEqual(
            parse_screen_direction("move screen_up"),
            "screen_up",
        )
        self.assertIsNone(parse_screen_direction("cannot decide"))

    def test_maps_screen_directions_to_world_actions(self):
        self.assertEqual(map_screen_to_world("screen_right"), "right")
        self.assertEqual(map_screen_to_world("screen_left"), "left")
        self.assertEqual(map_screen_to_world("screen_up"), "front")
        self.assertEqual(map_screen_to_world("screen_down"), "back")
        self.assertEqual(map_screen_to_world("stop"), "stop")


class SummarizePredictionsTests(unittest.TestCase):
    """固定合法率、准确率、分类指标、混淆矩阵和错误计数定义。"""

    def test_builds_required_offline_metrics(self):
        predictions = [
            {
                "expected_direction": "right",
                "predicted_direction": "right",
                "latency_seconds": 0.1,
                "error_type": None,
            },
            {
                "expected_direction": "right",
                "predicted_direction": "left",
                "latency_seconds": 0.2,
                "error_type": None,
            },
            {
                "expected_direction": "left",
                "predicted_direction": "left",
                "latency_seconds": 0.3,
                "error_type": None,
            },
            {
                "expected_direction": "front",
                "predicted_direction": None,
                "latency_seconds": 0.4,
                "error_type": "InvalidModelResponseError",
            },
        ]

        summary = summarize_predictions(predictions)

        self.assertEqual(summary["total_samples"], 4)
        self.assertEqual(summary["valid_output_count"], 3)
        self.assertEqual(summary["valid_output_rate"], 0.75)
        self.assertEqual(summary["exact_match_accuracy"], 0.5)
        self.assertAlmostEqual(summary["average_api_latency_seconds"], 0.25)
        self.assertEqual(
            summary["error_type_counts"],
            {"InvalidModelResponseError": 1},
        )
        self.assertEqual(summary["per_direction"]["right"]["precision"], 1.0)
        self.assertEqual(summary["per_direction"]["right"]["recall"], 0.5)
        self.assertEqual(summary["per_direction"]["left"]["precision"], 0.5)
        self.assertEqual(summary["per_direction"]["left"]["recall"], 1.0)
        self.assertEqual(
            summary["confusion_matrix"]["front"]["__invalid__"],
            1,
        )


class OfflineResumeTests(unittest.TestCase):
    """验证重复运行时不会为同一 sample_id 再次调用付费 API。"""

    def test_completed_sample_is_skipped_on_second_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_dir = root / "samples"
            run_root = root / "outputs/vlm_evaluations"
            image_path = sample_dir / "images" / "sample.jpg"
            image_path.parent.mkdir(parents=True)
            self.assertTrue(
                cv2.imwrite(
                    str(image_path),
                    np.zeros((16, 16, 3), dtype=np.uint8),
                )
            )
            sample = {
                "sample_id": "seed_42_step_000",
                "image_path": str(image_path),
                "instruction": "悬停在红色积木上方",
                "expected_direction": "right",
            }
            (sample_dir / "samples.jsonl").write_text(
                json.dumps(sample, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            config = {
                "vlm_evaluation": {
                    "sample_output_dir": str(sample_dir),
                    "run_output_dir": str(run_root),
                    "offline_run_name": "offline_test",
                },
                "probe": {"api": {}},
            }

            # 模拟 Qwen 返回 right；不会访问真实网络。
            with patch(
                "vla_project.vlm.evaluate_vlm_decisions.call_openai_compatible_api",
                return_value=("screen_right", "screen_right"),
            ) as api_mock:
                first_run_dir, first_summary = evaluate_offline(
                    config,
                    limit=1,
                    project_root_override=root,
                )
                second_run_dir, second_summary = evaluate_offline(
                    config,
                    limit=1,
                    project_root_override=root,
                )

            self.assertEqual(api_mock.call_count, 1)
            self.assertEqual(first_run_dir, second_run_dir)
            self.assertEqual(first_summary["total_samples"], 1)
            self.assertEqual(second_summary["exact_match_accuracy"], 1.0)
            prediction_lines = (
                first_run_dir / "predictions.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(prediction_lines), 1)


if __name__ == "__main__":
    unittest.main()
