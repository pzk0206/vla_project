"""测试 Stage 3 在线闭环中可以脱离真实仿真验证的关键规则。

覆盖世界坐标方向映射、模式校验、终止状态、trace 摘要和关节误差符号。把这些
小规则单独测试，可以在不启动 PyBullet episode 的情况下快速发现控制逻辑回归。
"""

import unittest

from stage3_probe import (
    decide_direction,
    determine_probe_termination,
    direction_to_target,
    joint_angle_errors,
    summarize_probe_trace,
)


class DirectionToTargetTests(unittest.TestCase):
    """固定方向词到 PyBullet 世界坐标目标点的映射。"""

    def test_cardinal_directions_move_one_world_axis_step(self):
        """四个方向每次只能改变一个平面轴，并统一使用悬停高度。

        这里的 left/right/front/back 属于世界坐标，不代表相机图片中的方向。
        """
        ee_pos = [1.0, 2.0, 3.0]
        cases = {
            "left": [0.75, 2.0, 0.5],
            "right": [1.25, 2.0, 0.5],
            "front": [1.0, 2.25, 0.5],
            "back": [1.0, 1.75, 0.5],
        }
        for direction, expected in cases.items():
            # subTest 让四个方向分别报告结果；一个方向失败时仍能看出具体是哪一个。
            with self.subTest(direction=direction):
                self.assertEqual(
                    direction_to_target(direction, ee_pos, 0.5, 0.25),
                    expected,
                )

    def test_stop_keeps_xy_and_uses_hover_height(self):
        """stop 不应继续横向移动，但 z 仍应回到规定的安全悬停高度。"""
        self.assertEqual(
            direction_to_target("stop", [1.0, 2.0, 3.0], 0.5, 0.25),
            [1.0, 2.0, 0.5],
        )

    def test_unknown_direction_is_rejected(self):
        """未实现的方向必须明确报错，不能静默执行不可预测动作。"""
        with self.assertRaisesRegex(ValueError, "未知方向"):
            direction_to_target("up", [1.0, 2.0, 3.0], 0.5, 0.25)


class DecideDirectionTests(unittest.TestCase):
    """检查 probe 决策模式配置错误时能否尽早失败。"""

    def test_unknown_probe_mode_is_rejected(self):
        """故意拼错 heuristic，验证错误信息能够指向 probe.mode。"""
        probe_config = {
            # 这是测试输入中的故意拼写错误，不是代码笔误。
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
    """固定 probe 的 success、max_control_steps、running 判定边界。"""

    def test_success_precedes_control_step_limit(self):
        """最后一个允许步骤如果到达目标，应优先记录 success。"""
        self.assertEqual(
            determine_probe_termination(0.02, 79, 80, 0.03),
            "success",
        )

    def test_last_control_step_reports_limit(self):
        """第 80 步仍未成功时，应记录达到控制步数上限。

        control_step 从 0 开始计数，所以 80 步中的最后一步下标是 79。
        """
        self.assertEqual(
            determine_probe_termination(0.5, 79, 80, 0.03),
            "max_control_steps",
        )

    def test_intermediate_step_keeps_running(self):
        """尚未成功且未到最后一步时，episode 应继续运行。"""
        self.assertEqual(
            determine_probe_termination(0.5, 10, 80, 0.03),
            "running",
        )


class SummarizeProbeTraceTests(unittest.TestCase):
    """检查逐步 trace 能否压缩成正确的 episode 级摘要。"""

    def test_builds_episode_summary(self):
        """摘要应保留首末距离、方向次数、退步次数和最终状态。"""
        rows = [
            {
                "direction": "front",
                "distance_before": 1.0,
                "distance_after": 0.8,
                "distance_delta": 0.2,
                "termination_reason": "running",
                "block_pos": [0.1, 0.4, 0.05],
            },
            {
                "direction": "left",
                "distance_before": 0.8,
                "distance_after": 0.9,
                # distance_delta < 0 表示动作后离目标更远，属于一次“退步”。
                "distance_delta": -0.1,
                "termination_reason": "max_control_steps",
                "block_pos": [0.1, 0.4, 0.05],
            },
        ]
        summary = summarize_probe_trace(rows, 3, 45, "run/trace.jsonl")
        # 两行 trace 就是两个闭环控制步。
        self.assertEqual(summary["num_control_steps"], 2)
        self.assertEqual(summary["initial_distance"], 1.0)
        self.assertEqual(summary["final_distance"], 0.9)
        self.assertEqual(summary["direction_counts"], {"front": 1, "left": 1})
        self.assertEqual(summary["distance_increase_steps"], 1)
        self.assertFalse(summary["success"])


class JointDiagnosticsTests(unittest.TestCase):
    """固定关节跟踪误差的符号定义，避免诊断日志正负颠倒。"""

    def test_joint_angle_errors_use_target_minus_actual(self):
        """每个关节误差都必须按 target - actual 独立计算。"""
        self.assertEqual(
            joint_angle_errors([1.0, -0.5, 0.25], [0.75, -0.25, 0.5]),
            [0.25, -0.25, -0.25],
        )


if __name__ == "__main__":
    # 支持直接执行本文件，也支持项目级 unittest discover。
    unittest.main()
