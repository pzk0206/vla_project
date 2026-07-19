"""测试数据采集主流程中最容易发生边界错误的纯逻辑。

覆盖三部分：IK 是否收到关节约束、episode 终止原因的优先级，以及追加采集时
episode 编号的计算。外部仿真接口使用 mock，文件测试使用临时目录，因此不会
打开 PyBullet 窗口，也不会改动真实 ``outputs/dataset/``。
"""

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from vla_project.simulation import control_arm

from vla_project.simulation.control_arm import (
    calculate_target_joints,
    capture_rgb,
    determine_termination,
    next_episode_index,
)


class CaptureRgbCameraMatrixTests(unittest.TestCase):
    @patch("vla_project.simulation.control_arm.p.getCameraImage")
    @patch("vla_project.simulation.control_arm.compute_camera_matrices")
    def test_returns_rgb_and_segmentation_with_matching_shapes(
        self, compute_matrices, get_camera_image
    ):
        compute_matrices.return_value = ([0.0] * 16, [1.0] * 16)
        segmentation = np.array([[3, 3], [7, -1]], dtype=np.int32)
        get_camera_image.return_value = (
            2,
            2,
            np.zeros((2, 2, 4), dtype=np.uint8),
            None,
            segmentation,
        )

        image, mask = control_arm.capture_rgb_and_segmentation(
            {"image_width": 2, "image_height": 2},
            [0.0, 0.4, 3.0],
        )

        self.assertEqual(image.shape, (2, 2, 3))
        self.assertEqual(mask.shape, (2, 2))
        np.testing.assert_array_equal(mask, segmentation)

    @patch("vla_project.simulation.control_arm.p.getCameraImage")
    @patch("vla_project.simulation.control_arm.compute_camera_matrices")
    def test_passes_shared_matrices_to_pybullet_renderer(
        self, compute_matrices, get_camera_image
    ):
        view_matrix = [float(index) for index in range(16)]
        projection_matrix = [float(index + 16) for index in range(16)]
        compute_matrices.return_value = (view_matrix, projection_matrix)
        get_camera_image.return_value = (
            2,
            2,
            np.zeros((2, 2, 4), dtype=np.uint8),
            None,
            np.zeros((2, 2), dtype=np.int32),
        )
        camera_config = {
            "image_width": 2,
            "image_height": 2,
        }

        image = capture_rgb(camera_config, [0.0, 0.4, 3.0])

        compute_matrices.assert_called_once_with(
            camera_config, [0.0, 0.4, 3.0]
        )
        self.assertEqual(
            get_camera_image.call_args.kwargs["viewMatrix"], view_matrix
        )
        self.assertEqual(
            get_camera_image.call_args.kwargs["projectionMatrix"], projection_matrix
        )
        self.assertEqual(image.shape, (2, 2, 3))


class CalculateTargetJointsTests(unittest.TestCase):
    """确保 IK 使用真实关节限位和当前姿态，避免再次撞关节上限。"""

    # patch 把真实 PyBullet 函数替换成可预测的假函数：
    # - 不需要真正加载 KUKA；
    # - 可以直接检查 calculateInverseKinematics 最终收到了哪些参数。
    # 装饰器从下往上应用，所以 mock 参数按从下往上的顺序传入测试方法。
    @patch(
        "vla_project.simulation.control_arm.p.calculateInverseKinematics",
        return_value=[0.0] * 7,
    )
    @patch(
        "vla_project.simulation.control_arm.p.getJointState",
        side_effect=lambda _robot, joint: (0.1 * joint,),
    )
    @patch("vla_project.simulation.control_arm.p.getJointInfo")
    @patch(
        "vla_project.simulation.control_arm.p.getQuaternionFromEuler",
        return_value=[0.0, 0.0, 0.0, 1.0],
    )
    def test_ik_receives_joint_limits_and_current_pose_as_rest_pose(
        self,
        _quaternion,
        get_joint_info,
        _get_joint_state,
        calculate_ik,
    ):
        """7 个关节的上下限、范围和当前角度必须全部传给 IK。"""

        def joint_info(_robot, joint):
            """构造与 PyBullet getJointInfo 相同位置的限位字段。

            PyBullet 返回的元组中，下标 8 是下限，下标 9 是上限。这里让每个
            关节的数值不同，防止生产代码即使写死一个限位也错误通过测试。
            """
            info = [None] * 10
            info[8] = -1.0 - joint
            info[9] = 1.0 + joint
            return tuple(info)

        get_joint_info.side_effect = joint_info
        # 只放 calculate_target_joints 真正会读取的最小配置，避免测试依赖整份 YAML。
        robot_config = {
            "ee_link_index": 6,
            "controlled_joints": 7,
            "target_orientation_euler": [0.0, 3.141592653589793, 0.0],
            "ik_residual_threshold": 0.001,
        }

        # robot_id=3 和目标坐标都是测试占位值；PyBullet 已被 mock，不会访问真机器人。
        calculate_target_joints(3, robot_config, [0.1, 0.2, 0.3])

        # call_args.kwargs 让测试检查“调用边界”，而不是只看 IK 的假返回值。
        kwargs = calculate_ik.call_args.kwargs
        self.assertEqual(kwargs["lowerLimits"], [-1.0 - joint for joint in range(7)])
        self.assertEqual(kwargs["upperLimits"], [1.0 + joint for joint in range(7)])
        self.assertEqual(kwargs["jointRanges"], [2.0 + 2.0 * joint for joint in range(7)])
        self.assertEqual(kwargs["restPoses"], [0.1 * joint for joint in range(7)])


class DetermineTerminationTests(unittest.TestCase):
    """固定 success、stuck、max_steps 三种终止原因的边界与优先级。"""

    def test_success_includes_exact_distance_threshold(self):
        """距离恰好等于成功阈值时也应算成功（使用 <= 而不是 <）。"""
        self.assertEqual(
            determine_termination(
                distance_to_target=0.03,
                recent_distances=[0.04, 0.03],
                step_idx=10,
                success_distance=0.03,
                stuck_window_steps=2,
                stuck_min_improvement=0.0005,
                force_terminal_after_step=976,
            ),
            "success",
        )

    def test_max_steps_is_not_masked_by_full_improving_stuck_window(self):
        """窗口已满但仍明显改善时，不应误报 stuck，最终应报 max_steps。"""
        self.assertEqual(
            determine_termination(
                distance_to_target=0.5,
                recent_distances=[0.6, 0.5],
                step_idx=976,
                success_distance=0.03,
                stuck_window_steps=2,
                stuck_min_improvement=0.0005,
                force_terminal_after_step=976,
            ),
            "max_steps",
        )

    def test_stuck_precedes_max_steps_when_both_apply(self):
        """同一步同时满足卡住和步数上限时，优先保留更具体的 stuck 原因。

        0.5001 -> 0.5 只改善 0.0001，小于要求的 0.0005，因此确实属于卡住。
        """
        self.assertEqual(
            determine_termination(
                distance_to_target=0.5,
                recent_distances=[0.5001, 0.5],
                step_idx=976,
                success_distance=0.03,
                stuck_window_steps=2,
                stuck_min_improvement=0.0005,
                force_terminal_after_step=976,
            ),
            "stuck",
        )


class NextEpisodeIndexTests(unittest.TestCase):
    """确保覆盖采集和追加采集都能得到正确的下一个 episode 编号。"""

    def test_missing_summary_starts_from_zero(self):
        """摘要文件尚不存在代表首次采集，应从 episode 0 开始。"""
        # TemporaryDirectory 会在测试结束后自动清理，不污染真实 outputs/dataset/。
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "episode_summary.jsonl"
            self.assertEqual(next_episode_index(summary_path), 0)

    def test_existing_summary_continues_after_largest_episode(self):
        """追加采集应使用历史最大编号加一，而不是简单使用行数。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "episode_summary.jsonl"
            rows = [{"episode_idx": 2}, {"episode_idx": 7}, {"episode_idx": 4}]
            summary_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            # 历史顺序可以不连续或不排序；最大编号是 7，所以下一个必须是 8。
            self.assertEqual(next_episode_index(summary_path), 8)


if __name__ == "__main__":
    # 允许把本文件当脚本单独运行，同时兼容 unittest discover。
    unittest.main()
