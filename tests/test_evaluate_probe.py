"""测试 Stage 3 批量评估的统计口径和诊断文件保留策略。

这里不运行真实 episode，而是把少量人工构造的摘要交给评估函数。这样可以快速
确认成功率、中位数、失败编号等统计没有算错，并验证清理图片时不会误删 trace。
"""

import tempfile
import unittest
from pathlib import Path

from evaluate_probe import aggregate_probe_summaries, cleanup_success_images


class AggregateTests(unittest.TestCase):
    """保护批量汇总结果以及成功/失败 episode 的文件清理规则。"""

    def test_aggregates_success_failure_and_metrics(self):
        """成功、普通失败和异常三类 episode 应按统一口径汇总。"""
        rows = [
            # 正常成功：有最终距离、控制步数和方向计数。
            {
                "episode_idx": 0,
                "success": True,
                "termination_reason": "success",
                "final_distance": 0.02,
                "num_control_steps": 10,
                "direction_counts": {"front": 2},
            },
            # 正常失败：达到最大控制步数，但仍然有可用于分析的距离。
            {
                "episode_idx": 1,
                "success": False,
                "termination_reason": "max_control_steps",
                "final_distance": 0.5,
                "num_control_steps": 80,
                "direction_counts": {"left": 3},
            },
            # 程序异常：没有有效最终距离，统计距离时必须忽略 None。
            {
                "episode_idx": 2,
                "success": False,
                "termination_reason": "error",
                "final_distance": None,
                "num_control_steps": 0,
                "direction_counts": {},
            },
        ]
        result = aggregate_probe_summaries(rows, {"probe": {}})

        # 3 条中仅 1 条成功；异常也属于失败，并单独计入 error_count。
        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["failure_count"], 2)
        self.assertEqual(result["error_count"], 1)
        self.assertAlmostEqual(result["success_rate"], 1 / 3)
        # 有效距离只有 0.02 和 0.5，中位数是两者平均值 0.26。
        self.assertEqual(result["final_distance_median"], 0.26)
        # 所有 episode 的方向次数需要按方向名称累加。
        self.assertEqual(result["direction_counts"], {"front": 2, "left": 3})
        # 普通失败与异常 episode 都应进入后续失败案例分析清单。
        self.assertEqual(result["failed_episode_indices"], [1, 2])

    def test_success_cleanup_deletes_images_but_keeps_trace(self):
        """成功案例可删除大图片节省空间，但必须保留轻量 trace。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            # 用最小占位内容模拟真实 episode 输出，不依赖 OpenCV。
            (path / "probe_step_00.jpg").write_bytes(b"x")
            (path / "probe_trace.jsonl").write_text("{}\n")
            cleanup_success_images(path, success=True, enabled=True)
            self.assertFalse((path / "probe_step_00.jpg").exists())
            self.assertTrue((path / "probe_trace.jsonl").exists())

    def test_failure_cleanup_keeps_images(self):
        """失败案例的步骤图片必须保留，方便人工查看失败发生在哪一帧。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            image = path / "probe_step_00.jpg"
            image.write_bytes(b"x")
            cleanup_success_images(path, success=False, enabled=True)
            self.assertTrue(image.exists())
