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


class BlockPairSamplingTests(unittest.TestCase):
    """阻止双积木重叠和随机任务失衡重新进入采集。"""

    def make_task_config(self):
        return {
            "pair_sampling": {
                "x_range": [-0.2, 0.2],
                "y_range": [0.38, 0.5],
                "z": 0.1,
                "min_axis_separation_xy": 0.12,
                "max_attempts": 2,
            }
        }

    def test_rejects_overlapping_draw_before_returning_separated_pair(self):
        class SequenceRng:
            def __init__(self):
                self.values = iter(
                    [
                        0.00, 0.40, 0.05, 0.45,
                        -0.20, 0.40, 0.20, 0.40,
                    ]
                )

            def uniform(self, _low, _high):
                return next(self.values)

        positions = control_arm.sample_block_pair_positions(
            self.make_task_config(),
            rng=SequenceRng(),
        )

        self.assertEqual(positions["red"], [-0.2, 0.4, 0.1])
        self.assertEqual(positions["blue"], [0.2, 0.4, 0.1])

    def test_impossible_pair_range_fails_after_finite_attempts(self):
        class MidpointRng:
            def uniform(self, low, high):
                return (low + high) / 2.0

        with self.assertRaisesRegex(
            ValueError,
            "unable to sample non-overlapping block pair",
        ):
            control_arm.sample_block_pair_positions(
                self.make_task_config(),
                rng=MidpointRng(),
            )

    def test_balanced_task_schedule_is_reproducible_and_exact(self):
        config = {
            "dataset": {
                "instruction": "悬停在红色积木上方",
                "random_seed": 1000,
                "task_selection": "balanced_alternating",
            },
            "task": {
                "tasks": [
                    {
                        "instruction": "悬停在红色积木上方",
                        "target_block": "red",
                    },
                    {
                        "instruction": "悬停在蓝色积木上方",
                        "target_block": "blue",
                    },
                ]
            },
        }

        first = [control_arm.select_task(config, i) for i in range(300)]
        second = [control_arm.select_task(config, i) for i in range(300)]

        self.assertEqual(first, second)
        self.assertEqual(
            sum(target == "red" for _, target in first),
            150,
        )
        self.assertEqual(
            sum(target == "blue" for _, target in first),
            150,
        )


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


class DatasetRunContractTests(unittest.TestCase):
    """保护版本化目录、manifest 和追加兼容性。"""

    def make_config(self, clean_before_run=True):
        return {
            "robot": {"controlled_joints": 7},
            "dataset": {
                "output_dir": "outputs/dataset/expert_scaling_v1",
                "jsonl_name": "trajectory_expert.jsonl",
                "summary_jsonl_name": "episode_summary.jsonl",
                "schema_version": "expert_v1",
                "instruction": "悬停在红色积木上方",
                "random_seed": 1000,
                "pilot_num_episodes": 10,
                "target_num_episodes": 300,
                "clean_before_run": clean_before_run,
            },
            "task": {
                "block_position": {
                    "x_range": [-0.2, 0.2],
                    "y_range": [0.38, 0.5],
                    "z": 0.1,
                }
            },
            "camera": {
                "image_width": 224,
                "image_height": 224,
                "eye_offset_base": [1.05, 0.0, 1.65],
                "eye_offset_random_range": [-0.1, 0.1],
            },
        }

    def test_rejects_dataset_root_as_clean_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                control_arm.validate_versioned_dataset_dir(
                    "outputs/dataset",
                    project_root_override=temp_dir,
                )

    def test_accepts_versioned_child_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = control_arm.validate_versioned_dataset_dir(
                "outputs/dataset/expert_scaling_v1",
                project_root_override=temp_dir,
            )
            self.assertEqual(
                path,
                Path(temp_dir).resolve()
                / "outputs/dataset/expert_scaling_v1",
            )

    def test_accepts_absolute_versioned_child_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            requested = (
                Path(temp_dir).resolve()
                / "outputs/dataset/expert_scaling_v1"
            )

            path = control_arm.validate_versioned_dataset_dir(
                requested,
                project_root_override=temp_dir,
            )

            self.assertEqual(path, requested)

    def test_rejects_parent_traversal_before_it_reaches_source_tree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "parent traversal"):
                control_arm.validate_versioned_dataset_dir(
                    "outputs/dataset/../../src",
                    project_root_override=temp_dir,
                )

    def test_builds_expert_v1_manifest(self):
        manifest = control_arm.build_dataset_manifest(self.make_config())

        self.assertEqual(manifest["schema_version"], "expert_v1")
        self.assertEqual(manifest["action_dim"], 9)
        self.assertEqual(
            manifest["seed_rule"],
            "random_seed + episode_idx",
        )
        self.assertEqual(manifest["pilot_num_episodes"], 10)
        self.assertEqual(manifest["target_num_episodes"], 300)
        self.assertTrue(manifest["created_at"])

    def test_prepare_dataset_writes_manifest_and_config_snapshot(self):
        config = self.make_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir, _, _ = control_arm.prepare_dataset(
                config["dataset"],
                full_config=config,
                project_root_override=temp_dir,
            )
            run_dir = Path(dataset_dir)
            self.assertTrue((run_dir / "dataset_manifest.json").is_file())
            self.assertTrue((run_dir / "config_snapshot.yaml").is_file())

    def test_clean_before_run_refuses_nonempty_version_directory(self):
        config = self.make_config(clean_before_run=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = (
                Path(temp_dir)
                / "outputs/dataset/expert_scaling_v1"
            )
            dataset.mkdir(parents=True)
            evidence = dataset / "trajectory_expert.jsonl"
            evidence.write_text("evidence\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "new dataset version"):
                control_arm.prepare_dataset(
                    config["dataset"],
                    full_config=config,
                    project_root_override=temp_dir,
                )

            self.assertEqual(
                evidence.read_text(encoding="utf-8"),
                "evidence\n",
            )

    def test_clean_before_run_allows_empty_version_directory(self):
        config = self.make_config(clean_before_run=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = (
                Path(temp_dir)
                / "outputs/dataset/expert_scaling_v1"
            )
            dataset.mkdir(parents=True)

            dataset_dir, _, _ = control_arm.prepare_dataset(
                config["dataset"],
                full_config=config,
                project_root_override=temp_dir,
            )

            self.assertEqual(Path(dataset_dir), dataset)
            self.assertTrue((dataset / "dataset_manifest.json").is_file())

    def test_append_rejects_incompatible_manifest(self):
        config = self.make_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            control_arm.prepare_dataset(
                config["dataset"],
                full_config=config,
                project_root_override=temp_dir,
            )
            incompatible = self.make_config(clean_before_run=False)
            incompatible["dataset"]["schema_version"] = "expert_v2"
            with self.assertRaises(ValueError):
                control_arm.prepare_dataset(
                    incompatible["dataset"],
                    full_config=incompatible,
                    project_root_override=temp_dir,
                )


class SceneStateTests(unittest.TestCase):
    """保护确定性重放需要的完整双积木动态状态。"""

    def make_scene_state(self, target_block="blue"):
        return {
            "target_block": target_block,
            "blocks": {
                "red": {
                    "position": [-0.15, 0.40, 0.05],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
                "blue": {
                    "position": [0.15, 0.40, 0.05],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
            },
            "robot": {
                "joint_positions": [0.1 * index for index in range(7)],
                "joint_velocities": [0.01 * index for index in range(7)],
                "ee_position": [0.0, 0.4, 0.2],
            },
            "camera_eye": [1.05, 0.4, 1.65],
        }

    @patch(
        "vla_project.simulation.control_arm.get_link_position",
        return_value=(0.0, 0.4, 0.2),
    )
    @patch("vla_project.simulation.control_arm.p.getJointState")
    @patch("vla_project.simulation.control_arm.p.getBasePositionAndOrientation")
    def test_capture_reads_both_blocks_robot_and_camera(
        self,
        get_base_pose,
        get_joint_state,
        _get_link_position,
    ):
        get_base_pose.side_effect = {
            10: ((-0.15, 0.40, 0.05), (0.0, 0.0, 0.0, 1.0)),
            11: ((0.15, 0.40, 0.05), (0.0, 0.0, 0.0, 1.0)),
        }.get
        get_joint_state.side_effect = lambda _robot, joint: (
            0.1 * joint,
            0.01 * joint,
            0.0,
            0.0,
        )

        state = control_arm.capture_scene_state(
            robot_id=3,
            robot_config={"controlled_joints": 7, "ee_link_index": 6},
            block_ids={"red": 10, "blue": 11},
            camera_eye=[1.05, 0.4, 1.65],
            target_block="blue",
        )

        self.assertEqual(state, self.make_scene_state())

    def test_v2_frame_rejects_missing_scene_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "scene_state"):
                control_arm.write_dataset_step(
                    jsonl_path=Path(temp_dir) / "trajectory.jsonl",
                    schema_version="expert_multi_v2",
                    episode_idx=0,
                    step_idx=0,
                    random_seed=1000,
                    image_path="ep_0_step_0.jpg",
                    instruction="悬停在蓝色积木上方",
                    action=[0.0] * 7 + [1.0, 0],
                    camera_eye=[1.05, 0.4, 1.65],
                    block_pos=[0.15, 0.40, 0.05],
                    target_pos=[0.15, 0.40, 0.20],
                    ee_pos=[0.0, 0.4, 1.26],
                    distance_to_target=1.0,
                    termination_reason="running",
                )

    def test_v2_frame_writes_complete_scene_state(self):
        state = self.make_scene_state()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trajectory.jsonl"
            control_arm.write_dataset_step(
                jsonl_path=path,
                schema_version="expert_multi_v2",
                episode_idx=0,
                step_idx=0,
                random_seed=1000,
                image_path="ep_0_step_0.jpg",
                instruction="悬停在蓝色积木上方",
                action=[0.0] * 7 + [1.0, 0],
                camera_eye=[1.05, 0.4, 1.65],
                block_pos=[0.15, 0.40, 0.05],
                target_pos=[0.15, 0.40, 0.20],
                ee_pos=[0.0, 0.4, 1.26],
                distance_to_target=1.0,
                termination_reason="running",
                scene_state=state,
            )

            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(row["scene_state"], state)

    def test_v2_summary_writes_initial_and_final_scene_state(self):
        initial = self.make_scene_state()
        final = self.make_scene_state()
        final["robot"]["ee_position"] = [0.15, 0.40, 0.20]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.jsonl"
            control_arm.write_episode_summary(
                summary_jsonl_path=path,
                schema_version="expert_multi_v2",
                episode_idx=0,
                random_seed=1000,
                initial_ee_pos=[0.0, 0.0, 1.261],
                initial_block_pos=[0.15, 0.40, 0.05],
                num_steps=25,
                num_frames=2,
                final_distance=0.01,
                termination_reason="success",
                camera_eye=[1.05, 0.4, 1.65],
                final_block_pos=[0.15, 0.40, 0.05],
                final_target_pos=[0.15, 0.40, 0.20],
                final_ee_pos=[0.15, 0.40, 0.19],
                task_instruction="悬停在蓝色积木上方",
                target_block="blue",
                initial_scene_state=initial,
                final_scene_state=final,
            )

            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(row["initial_scene_state"], initial)
            self.assertEqual(row["final_scene_state"], final)


class SettledBlockPairTests(unittest.TestCase):
    """阻止加载器重新采样位置，并在采集前拒绝物理漂移。"""

    @patch("vla_project.simulation.control_arm.p.changeVisualShape")
    @patch("vla_project.simulation.control_arm.p.loadURDF", return_value=12)
    def test_loader_uses_explicit_position(self, load_urdf, _change_visual):
        block_id = control_arm.load_block_at_position(
            task_config={"block_urdf_path": "cube.urdf"},
            position=[-0.15, 0.40, 0.10],
            color_rgba=[1, 0, 0, 1],
            global_scaling=0.1,
        )

        self.assertEqual(block_id, 12)
        self.assertEqual(
            load_urdf.call_args.kwargs["basePosition"],
            [-0.15, 0.40, 0.10],
        )

    def test_settled_pair_rejects_excessive_xy_drift(self):
        sampled = {
            "red": [-0.15, 0.40, 0.10],
            "blue": [0.15, 0.40, 0.10],
        }
        scene_state = {
            "blocks": {
                "red": {"position": [-0.14, 0.40, 0.05]},
                "blue": {"position": [0.15, 0.40, 0.05]},
            }
        }
        with self.assertRaisesRegex(ValueError, "settle drift"):
            control_arm._validate_settled_pair(
                sampled,
                scene_state,
                {
                    "min_axis_separation_xy": 0.12,
                    "max_settle_drift_xy": 0.005,
                },
            )

    def test_settled_pair_accepts_small_drift_and_safe_separation(self):
        control_arm._validate_settled_pair(
            {
                "red": [-0.15, 0.40, 0.10],
                "blue": [0.15, 0.40, 0.10],
            },
            {
                "blocks": {
                    "red": {"position": [-0.148, 0.401, 0.05]},
                    "blue": {"position": [0.149, 0.399, 0.05]},
                }
            },
            {
                "min_axis_separation_xy": 0.12,
                "max_settle_drift_xy": 0.005,
            },
        )


class EpisodeReproducibilityTests(unittest.TestCase):
    """保护独立复位、确定性 seed 和 expert_v1 写入契约。"""

    @patch("vla_project.simulation.control_arm.apply_joint_targets")
    @patch("vla_project.simulation.control_arm.p.resetJointState")
    def test_reset_robot_to_home_clears_position_velocity_and_motor_target(
        self,
        reset_joint,
        apply_targets,
    ):
        robot_config = {"controlled_joints": 7}
        dataset_config = {
            "reset_robot_each_episode": True,
            "home_joint_positions": [0.0] * 7,
        }

        control_arm.reset_robot_to_home(
            robot_id=3,
            robot_config=robot_config,
            dataset_config=dataset_config,
        )

        self.assertEqual(reset_joint.call_count, 7)
        for joint, reset_call in enumerate(reset_joint.call_args_list):
            self.assertEqual(reset_call.args, (3, joint, 0.0))
            self.assertEqual(reset_call.kwargs["targetVelocity"], 0.0)
        apply_targets.assert_called_once_with(
            3,
            robot_config,
            [0.0] * 7,
        )

    @patch("vla_project.simulation.control_arm.apply_joint_targets")
    @patch("vla_project.simulation.control_arm.p.resetJointState")
    def test_reset_robot_to_home_can_be_disabled(
        self,
        reset_joint,
        apply_targets,
    ):
        control_arm.reset_robot_to_home(
            robot_id=3,
            robot_config={"controlled_joints": 7},
            dataset_config={
                "reset_robot_each_episode": False,
                "home_joint_positions": [0.0] * 7,
            },
        )

        reset_joint.assert_not_called()
        apply_targets.assert_not_called()

    def test_write_dataset_step_adds_expert_v1_identity_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = Path(temp_dir) / "trajectory_expert.jsonl"
            control_arm.write_dataset_step(
                jsonl_path=jsonl_path,
                schema_version="expert_v1",
                episode_idx=4,
                step_idx=24,
                random_seed=1004,
                image_path="ep_4_step_24.jpg",
                instruction="悬停在红色积木上方",
                action=[0.0] * 7 + [1.0, 1],
                camera_eye=[1.0, 0.4, 1.6],
                block_pos=[0.1, 0.4, 0.1],
                target_pos=[0.1, 0.4, 0.25],
                ee_pos=[0.1, 0.4, 0.25],
                distance_to_target=0.0,
                termination_reason="success",
            )

            row = json.loads(jsonl_path.read_text(encoding="utf-8"))
            self.assertEqual(row["schema_version"], "expert_v1")
            self.assertEqual(row["episode_idx"], 4)
            self.assertEqual(row["step_idx"], 24)
            self.assertEqual(row["random_seed"], 1004)
            self.assertEqual(len(row["action"]), 9)
            self.assertEqual(row["action"][-1], 1)
            self.assertNotEqual(row["termination_reason"], "running")

    def test_write_episode_summary_adds_seed_and_initial_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "episode_summary.jsonl"
            control_arm.write_episode_summary(
                summary_jsonl_path=summary_path,
                schema_version="expert_v1",
                episode_idx=4,
                random_seed=1004,
                initial_ee_pos=[0.0, 0.0, 1.261],
                initial_block_pos=[0.1, 0.4, 0.05],
                num_steps=25,
                num_frames=2,
                final_distance=0.01,
                termination_reason="success",
                camera_eye=[1.0, 0.4, 1.6],
                final_block_pos=[0.1, 0.4, 0.05],
                final_target_pos=[0.1, 0.4, 0.2],
                final_ee_pos=[0.1, 0.4, 0.19],
            )

            row = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(row["schema_version"], "expert_v1")
            self.assertEqual(row["random_seed"], 1004)
            self.assertEqual(row["initial_ee_pos"], [0.0, 0.0, 1.261])
            self.assertEqual(row["initial_block_pos"], [0.1, 0.4, 0.05])

    def test_run_episode_seeds_and_resets_before_sampling(self):
        events = []
        config = {
            "enable_time_sleep": False,
            "simulation_hz": 240,
            "robot": {
                "controlled_joints": 7,
                "ee_link_index": 6,
            },
            "dataset": {
                "schema_version": "expert_v1",
                "reset_robot_each_episode": True,
                "home_joint_positions": [0.0] * 7,
                "max_steps_per_episode": 1,
                "capture_interval_steps": 24,
                "default_gripper_state": 1.0,
                "instruction": "悬停在红色积木上方",
            },
            "task": {
                "initial_settle_steps": 0,
                "hover_height": 0.15,
                "success_distance": 0.03,
                "stuck_window_steps": 120,
                "stuck_min_improvement": 0.0005,
                "force_terminal_after_step": 976,
            },
            "camera": {},
        }
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "vla_project.simulation.control_arm.random.seed",
            side_effect=lambda seed: events.append(("seed", seed)),
        ), patch(
            "vla_project.simulation.control_arm.reset_robot_to_home",
            side_effect=lambda *_args: events.append(("reset", None)),
        ), patch(
            "vla_project.simulation.control_arm.load_block",
            side_effect=lambda *_args: events.append(("block", None)) or 9,
        ), patch(
            "vla_project.simulation.control_arm.sample_camera_eye",
            side_effect=lambda *_args: events.append(("camera", None))
            or [1.0, 0.4, 1.6],
        ), patch(
            "vla_project.simulation.control_arm.settle_object"
        ), patch(
            "vla_project.simulation.control_arm.get_link_position",
            return_value=[0.0, 0.0, 0.25],
        ), patch(
            "vla_project.simulation.control_arm.get_object_position",
            return_value=[0.0, 0.0, 0.1],
        ), patch(
            "vla_project.simulation.control_arm.calculate_target_joints",
            return_value=[0.0] * 7,
        ), patch(
            "vla_project.simulation.control_arm.apply_joint_targets"
        ), patch(
            "vla_project.simulation.control_arm.capture_rgb",
            return_value=np.zeros((2, 2, 3), dtype=np.uint8),
        ), patch(
            "vla_project.simulation.control_arm.cv2.imwrite",
            return_value=True,
        ), patch(
            "vla_project.simulation.control_arm.p.stepSimulation"
        ), patch(
            "vla_project.simulation.control_arm.p.removeBody"
        ):
            control_arm.run_episode(
                episode_idx=4,
                robot_id=3,
                config=config,
                dataset_dir=temp_dir,
                jsonl_path=Path(temp_dir) / "trajectory_expert.jsonl",
                summary_jsonl_path=Path(temp_dir) / "episode_summary.jsonl",
                random_seed=1004,
            )

        self.assertEqual(events[0], ("seed", 1004))
        self.assertLess(events.index(("reset", None)), events.index(("block", None)))
        self.assertLess(events.index(("block", None)), events.index(("camera", None)))

    def test_main_records_episode_error_and_continues(self):
        config = {
            "connection_mode": "DIRECT",
            "dataset": {
                "schema_version": "expert_v1",
                "random_seed": 1000,
                "num_episodes": 2,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "episode_summary.jsonl"
            with patch(
                "vla_project.simulation.control_arm.load_config",
                return_value=config,
            ), patch(
                "vla_project.simulation.control_arm.connect_physics"
            ), patch(
                "vla_project.simulation.control_arm.prepare_dataset",
                return_value=(
                    temp_dir,
                    str(Path(temp_dir) / "trajectory_expert.jsonl"),
                    str(summary_path),
                ),
            ), patch(
                "vla_project.simulation.control_arm.setup_world",
                return_value=(1, 3),
            ), patch(
                "vla_project.simulation.control_arm.next_episode_index",
                return_value=0,
            ), patch(
                "vla_project.simulation.control_arm.run_episode",
                side_effect=[RuntimeError("ik failed"), None],
            ) as run_episode, patch(
                "vla_project.simulation.control_arm.p.disconnect"
            ):
                control_arm.main()

            rows = [
                json.loads(line)
                for line in summary_path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(run_episode.call_count, 2)
        self.assertEqual(rows[0]["episode_idx"], 0)
        self.assertEqual(rows[0]["random_seed"], 1000)
        self.assertEqual(rows[0]["termination_reason"], "episode_error")
        self.assertIn("ik failed", rows[0]["error"])
        self.assertEqual(rows[0]["num_frames"], 0)
        self.assertEqual(
            run_episode.call_args_list[1].kwargs["random_seed"],
            1001,
        )


if __name__ == "__main__":
    # 允许把本文件当脚本单独运行，同时兼容 unittest discover。
    unittest.main()
