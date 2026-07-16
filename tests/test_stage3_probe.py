"""测试 Stage 3 在线闭环中可以脱离真实仿真验证的关键规则。

覆盖世界坐标方向映射、模式校验、终止状态、trace 摘要和关节误差符号。把这些
小规则单独测试，可以在不启动 PyBullet episode 的情况下快速发现控制逻辑回归。
"""

import io
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

import numpy as np

from stage3_probe import (
    InvalidModelResponseError,
    apply_end_effector_marker,
    build_vision_direction_prompt,
    call_openai_compatible_api,
    combine_camera_views,
    decide_direction,
    determine_probe_termination,
    direction_to_target,
    joint_angle_errors,
    map_screen_to_world,
    parse_direction,
    parse_screen_direction,
    sync_end_effector_marker,
    summarize_probe_trace,
)


class EndEffectorMarkerTests(unittest.TestCase):
    """绿色标记必须只修改配置指定的真实末端 link。"""

    @patch("stage3_probe.p.createMultiBody", return_value=12)
    @patch("stage3_probe.p.createVisualShape", return_value=11)
    @patch("stage3_probe.p.changeVisualShape")
    def test_marks_configured_end_effector_link_green(
        self,
        change_visual_shape,
        create_visual_shape,
        create_multi_body,
    ):
        marker_id = apply_end_effector_marker(
            robot_id=7,
            robot_config={"ee_link_index": 6},
            marker_config={
                "enabled": True,
                "color_rgba": [0, 1, 0, 1],
                "radius": 0.04,
            },
        )

        self.assertEqual(marker_id, 12)
        change_visual_shape.assert_called_once_with(
            7,
            6,
            rgbaColor=[0, 1, 0, 1],
        )
        create_visual_shape.assert_called_once_with(
            2,
            radius=0.04,
            rgbaColor=[0, 1, 0, 1],
        )
        create_multi_body.assert_called_once_with(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=11,
        )

    @patch("stage3_probe.p.resetBasePositionAndOrientation")
    @patch(
        "stage3_probe.p.getLinkState",
        return_value=(None, None, None, None, [1, 2, 3], [0, 0, 0, 1]),
    )
    def test_syncs_marker_to_current_link_pose(self, get_link_state, reset_marker_pose):
        sync_end_effector_marker(marker_id=12, robot_id=7, link_index=6)

        get_link_state.assert_called_once_with(7, 6)
        reset_marker_pose.assert_called_once_with(12, [1, 2, 3], [0, 0, 0, 1])


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


class CombineCameraViewsTests(unittest.TestCase):
    """固定左右双视图的尺寸和排列顺序。"""

    def test_concatenates_primary_left_and_secondary_right(self):
        primary = np.zeros((8, 100, 3), dtype=np.uint8)
        secondary = np.full((8, 100, 3), 255, dtype=np.uint8)

        combined = combine_camera_views(primary, secondary)

        # 顶部 24 像素用于 MAIN/AUX 标题，中间 4 像素黄色分隔线。
        self.assertEqual(combined.shape, (32, 204, 3))
        self.assertTrue(np.all(combined[24:, :100] == 0))
        self.assertTrue(np.all(combined[24:, 104:] == 255))
        self.assertTrue(np.all(combined[:, 100:104] == [0, 255, 255]))
        # 标题栏不是纯黑，证明标签文字确实被画入图片。
        self.assertTrue(np.any(combined[:24, :100] != 0))

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

    @patch(
        "stage3_probe.call_openai_compatible_api",
        return_value=("screen_up", "screen_up"),
    )
    def test_api_uses_screen_prompt_then_maps_to_world(self, api_call):
        """在线 API 也应使用图像标签，再由代码转换成世界坐标动作。"""
        direction, raw_response, source = decide_direction(
            {"mode": "api", "api": {}},
            image_bgr=None,
            ee_pos=[0.0, 0.0, 0.5],
            block_pos=[0.1, 0.1, 0.0],
            instruction="悬停在红色积木上方",
        )

        self.assertEqual(direction, "front")
        self.assertEqual(raw_response, "screen_up")
        self.assertEqual(source, "api")
        self.assertIs(api_call.call_args.kwargs["response_parser"], parse_screen_direction)
        self.assertIn("沿相互连接的机械臂连杆逐节追踪", api_call.call_args.kwargs["prompt_text"])


class VisionDirectionPromptTests(unittest.TestCase):
    """在线和离线共用同一套末端识别与图像方向规则。"""

    def test_online_prompt_has_separate_stop_branch(self):
        prompt = build_vision_direction_prompt(
            "悬停在红色积木上方",
            include_stop=True,
        )

        self.assertIn("连接关系中最后一节的末梢", prompt)
        self.assertIn("不是画面中的直线距离", prompt)
        self.assertIn("像素距离绝对值", prompt)
        self.assertIn("stop", prompt)
        self.assertNotIn("深灰色", prompt)
        self.assertNotIn("最下端", prompt)

    def test_screen_labels_map_deterministically_to_world(self):
        self.assertEqual(map_screen_to_world("screen_up"), "front")


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

class ParseDirectionTests(unittest.TestCase):
    """测试模型文本回复能否被正确解析为离散方向。"""

    def test_exact_direction_is_valid(self):
        """模型只返回合法方向时，应直接得到该方向。"""
        self.assertEqual(parse_direction("left"), "left")

    def test_direction_can_be_extracted_from_short_explanation(self):
        """模型附带简短说明时，也应提取其中的合法方向。"""
        self.assertEqual(parse_direction("move right"), "right")

    def test_invalid_response_returns_none_instead_of_false_stop(self):
        """回复中没有合法方向时应返回 None，不能误判为 stop。"""
        self.assertIsNone(parse_direction("I cannot see the block"))


class ApiResponseTests(unittest.TestCase):
    """测试 API 成功返回后，对模型文本内容的处理。"""

    def test_http_401_is_not_retried(self):
        """认证失败无法靠重试恢复，因此必须在第一次请求后停止。"""
        api_config = {
            "base_url_env": "VLA_API_BASE_URL",
            "api_key_env": "VLA_API_KEY",
            "model_env": "VLA_MODEL_NAME",
            "endpoint_path": "/chat/completions",
            "temperature": 0.0,
            "timeout_seconds": 30,
            "max_retries": 2,
        }
        unauthorized_error = urllib.error.HTTPError(
            url="https://example.invalid/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"invalid api key"}'),
        )
        environment = {
            "VLA_API_BASE_URL": "https://example.invalid/v1",
            "VLA_API_KEY": "invalid-test-key",
            "VLA_MODEL_NAME": "test-model",
        }

        with (
            patch.dict("stage3_probe.os.environ", environment),
            patch(
                "stage3_probe.encode_image_to_data_url",
                return_value="data:image/jpeg;base64,test",
            ),
            patch(
                "stage3_probe.urllib.request.urlopen",
                side_effect=unauthorized_error,
            ) as urlopen_mock,
        ):
            with self.assertRaisesRegex(RuntimeError, "401"):
                call_openai_compatible_api(None, api_config)

        # 即使 max_retries=2，认证错误也只能请求一次。
        self.assertEqual(urlopen_mock.call_count, 1)

    def test_http_429_is_retried_then_success_succeeds(self):
        """服务限流属于瞬时错误，第一次 429 后应再次请求。"""
        api_config = {
            "base_url_env": "VLA_API_BASE_URL",
            "api_key_env": "VLA_API_KEY",
            "model_env": "VLA_MODEL_NAME",
            "endpoint_path": "/chat/completions",
            "temperature": 0.0,
            "timeout_seconds": 30,
            "max_retries": 2,
        }
        rate_limit_error = urllib.error.HTTPError(
            url="https://example.invalid/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"rate limited"}'),
        )
        success_response = MagicMock()
        success_response.__enter__.return_value.read.return_value = (
            b'{"choices":[{"message":{"content":"right"}}]}'
        )
        environment = {
            "VLA_API_BASE_URL": "https://example.invalid/v1",
            "VLA_API_KEY": "test-key",
            "VLA_MODEL_NAME": "test-model",
        }

        # 第一次模拟 429，第二次返回合法方向；整个测试不会访问真实网络。
        with (
            patch.dict("stage3_probe.os.environ", environment),
            patch(
                "stage3_probe.encode_image_to_data_url",
                return_value="data:image/jpeg;base64,test",
            ),
            patch(
                "stage3_probe.urllib.request.urlopen",
                side_effect=[rate_limit_error, success_response],
            ) as urlopen_mock,
        ):
            direction, raw_response = call_openai_compatible_api(None, api_config)

        self.assertEqual(direction, "right")
        self.assertEqual(raw_response, "right")
        self.assertEqual(urlopen_mock.call_count, 2)

    def test_invalid_model_response_raises_specific_error(self):
        """无法解析的回复必须报错，不能作为正常方向继续执行。"""
        api_config = {
            "base_url_env": "VLA_API_BASE_URL",
            "api_key_env": "VLA_API_KEY",
            "model_env": "VLA_MODEL_NAME",
            "endpoint_path": "/chat/completions",
            "temperature": 0.0,
            "timeout_seconds": 30,
            "max_retries": 2,
        }
        response = MagicMock()
        response.__enter__.return_value.read.return_value = (
            b'{"choices":[{"message":{"content":"I cannot see the block"}}]}'
        )
        environment = {
            "VLA_API_BASE_URL": "https://example.invalid/v1",
            "VLA_API_KEY": "test-key",
            "VLA_MODEL_NAME": "test-model",
        }

        # 请求、图片编码和环境变量都由测试替代，不会访问真实 API，也不会产生费用。
        with (
            patch.dict("stage3_probe.os.environ", environment),
            patch(
                "stage3_probe.encode_image_to_data_url",
                return_value="data:image/jpeg;base64,test",
            ),
            patch(
                "stage3_probe.urllib.request.urlopen",
                return_value=response,
            ) as urlopen_mock,
        ):
            with self.assertRaisesRegex(
                InvalidModelResponseError,
                "I cannot see the block",
            ):
                call_openai_compatible_api(None, api_config)

        # 确认请求使用配置值，而不是退回代码中的固定超时时间。
        self.assertEqual(urlopen_mock.call_args.kwargs["timeout"], 30)


if __name__ == "__main__":
    # 支持直接执行本文件，也支持项目级 unittest discover。
    unittest.main()
