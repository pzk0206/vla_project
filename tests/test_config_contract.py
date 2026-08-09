"""检查 ``sim_config.yaml`` 是否满足项目代码依赖的最低配置契约。

这类测试不启动 PyBullet，也不评价机械臂表现。它的作用是在正式运行采集或
评估前，尽早发现配置缺字段、数值非法或两个参数互相矛盾的问题。
"""

import unittest

from vla_project.simulation.control_arm import load_config


class ConfigContractTests(unittest.TestCase):
    """保护采集脚本和 Stage 3 probe 共同依赖的配置结构。"""

    @classmethod
    def setUpClass(cls):
        """整个测试类只读取一次 YAML，避免每个测试重复访问文件。"""
        cls.config = load_config("sim_config.yaml")

    def test_required_sections_exist(self):
        """所有主流程必需的顶层配置区块都必须存在。

        ``required <= actual_keys`` 是集合的“子集”判断：允许 YAML 以后增加新
        区块，但不能缺少现有代码会读取的区块。
        """
        self.assertTrue(
            {
                "connection_mode",
                "gravity",
                "robot",
                "dataset",
                "task",
                "camera",
                "probe",
                "probe_evaluation",
                "vlm_evaluation",
                "grounding_smoke",
            }
            <= self.config.keys()
        )

    def test_grounding_smoke_config_is_frozen_and_safe(self):
        """在线小测必须锁定案例、标定、步长和安全阈值。"""
        smoke = self.config["grounding_smoke"]

        self.assertEqual(
            smoke["output_dir"],
            "outputs/vlm_evaluations/grounding_world_smoke",
        )
        self.assertEqual(
            smoke["calibration_path"],
            "outputs/vlm_evaluations/grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1/calibration_validation/calibration.json",
        )
        self.assertEqual(smoke["seeds"], [56, 55, 59])
        self.assertEqual(smoke["start_directions"], ["left", "right", "front"])
        self.assertEqual(smoke["start_offset_xy"], 0.10)
        self.assertEqual(smoke["max_control_steps"], 10)
        self.assertEqual(smoke["max_stale_target_steps"], 4)
        self.assertEqual(smoke["move_step_xy"], 0.02)
        self.assertEqual(smoke["hover_z"], 0.20)
        self.assertEqual(smoke["stop_distance_xy"], 0.02)
        self.assertEqual(smoke["visibility_reference_pixels"], 378)
        self.assertEqual(smoke["clear_visibility_threshold"], 0.75)
        self.assertEqual(smoke["max_target_jump_xy"], 0.03)
        self.assertEqual(smoke["min_progress_xy"], 0.005)
        self.assertEqual(smoke["no_progress_limit"], 2)
        self.assertEqual(smoke["workspace_x"], [-0.30, 0.30])
        self.assertEqual(smoke["workspace_y"], [0.30, 0.70])
        self.assertEqual(smoke["required_successes"], 3)
        self.assertEqual(smoke["max_total_api_calls"], 30)
        self.assertEqual(smoke["api_max_retries"], 0)
        self.assertEqual(smoke["expected_calibration_samples"], 15)
        self.assertEqual(
            smoke["expected_calibration_sample_ids"],
            [
                f"seed_{seed}_d020_{direction}"
                for seed in range(42, 47)
                for direction in ("front", "left", "right")
            ],
        )
        screening = smoke["screening"]
        self.assertEqual(
            screening["output_dir"],
            "outputs/vlm_evaluations/grounding_world_smoke_screening",
        )
        self.assertEqual(screening["seed_range"], [55, 100])
        self.assertEqual(
            screening["directions"], ["left", "right", "front"]
        )
        self.assertEqual(screening["max_pose_error"], 0.005)
        self.assertNotIn("num_actions", screening)
        self.assertNotIn("max_final_distance_xy", screening)

    def test_vlm_evaluation_config_is_valid(self):
        """离线采样与在线评估必须使用非空目录和正数 episode 上限。"""
        evaluation = self.config["vlm_evaluation"]
        self.assertEqual(
            evaluation["sample_output_dir"],
            "outputs/vlm_samples/448_calibration_validation_d020",
        )
        self.assertEqual(
            evaluation["grounding_run_name"],
            "grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1",
        )
        self.assertTrue(evaluation["run_output_dir"])
        self.assertTrue(evaluation["offline_run_name"])
        self.assertTrue(evaluation["ground_then_decide_run_name"])
        camera_override = evaluation["camera_override"]
        self.assertEqual(len(camera_override["eye_offset_base"]), 3)
        self.assertEqual(camera_override["eye_offset_base"], [0.0, 0.0, 3.0])
        self.assertEqual(len(camera_override["up_vector"]), 3)
        self.assertEqual(len(camera_override["eye_offset_random_range"]), 2)
        self.assertIs(evaluation["use_dual_view"], False)
        self.assertEqual(evaluation["sample_strategy"], "stratified_balanced_poses")
        self.assertEqual(evaluation["balanced_pose_offsets_xy"], [0.20])
        self.assertTrue(all(offset > 0 for offset in evaluation["balanced_pose_offsets_xy"]))
        self.assertEqual(evaluation["stratified_num_seeds"], 5)
        self.assertEqual(evaluation["stratified_seeds"], [47, 48, 49, 50, 51])
        self.assertEqual(len(set(evaluation["stratified_seeds"])), 5)
        self.assertTrue(
            all(type(seed) is int for seed in evaluation["stratified_seeds"])
        )
        self.assertEqual(camera_override["image_width"], 448)
        self.assertEqual(camera_override["image_height"], 448)
        self.assertGreater(evaluation["offline_num_episodes"], 0)
        self.assertGreater(evaluation["max_samples_per_episode"], 0)
        self.assertGreater(evaluation["online_smoke_episodes"], 0)
        self.assertGreater(evaluation["online_eval_episodes"], 0)

    def test_generated_outputs_are_grouped_under_outputs(self):
        """所有生成图片和实验记录必须进入统一 outputs 根目录。"""
        self.assertEqual(
            self.config["dataset"]["output_dir"],
            "outputs/dataset/expert_multi_v2",
        )
        self.assertEqual(self.config["probe"]["output_dir"], "outputs/probe")
        self.assertEqual(
            self.config["probe_evaluation"]["output_dir"],
            "outputs/probe_evaluations",
        )
        self.assertEqual(
            self.config["vlm_evaluation"]["sample_output_dir"],
            "outputs/vlm_samples/448_calibration_validation_d020",
        )
        self.assertEqual(
            self.config["vlm_evaluation"]["run_output_dir"],
            "outputs/vlm_evaluations",
        )
        self.assertTrue(
            self.config["grounding_smoke"]["output_dir"].startswith(
                "outputs/vlm_evaluations/"
            )
        )
        self.assertTrue(
            self.config["grounding_smoke"]["screening"]["output_dir"].startswith(
                "outputs/vlm_evaluations/"
            )
        )

    def test_expert_multi_v2_collection_config_is_frozen(self):
        """双积木 v2 必须使用斜视、联合采样和可追加的300条计划。"""
        dataset = self.config["dataset"]
        task = self.config["task"]
        pair = task["pair_sampling"]

        self.assertEqual(
            dataset["output_dir"],
            "outputs/dataset/expert_multi_v2",
        )
        self.assertEqual(dataset["schema_version"], "expert_multi_v2")
        self.assertEqual(dataset["task_selection"], "balanced_alternating")
        self.assertEqual(dataset["random_seed"], 1000)
        self.assertTrue(dataset["reset_robot_each_episode"])
        self.assertEqual(dataset["home_joint_positions"], [0.0] * 7)
        self.assertFalse(dataset["clean_before_run"])
        self.assertEqual(dataset["num_episodes"], 300)
        self.assertEqual(dataset["pilot_num_episodes"], 10)
        self.assertEqual(dataset["target_num_episodes"], 300)
        self.assertEqual(pair["x_range"], [-0.2, 0.2])
        self.assertEqual(pair["y_range"], [0.38, 0.5])
        self.assertEqual(pair["min_axis_separation_xy"], 0.12)
        self.assertEqual(pair["max_attempts"], 100)
        self.assertEqual(pair["max_settle_drift_xy"], 0.005)
        self.assertEqual(pair["max_episode_drift_xy"], 0.005)
        self.assertEqual(self.config["camera"]["eye_offset_base"], [1.05, 0.0, 1.65])

    def test_probe_evaluation_config_is_valid(self):
        """批量评估必须有正数次数、非空目录和可复现的整数种子。"""
        evaluation = self.config["probe_evaluation"]
        self.assertGreater(evaluation["num_episodes"], 0)
        self.assertTrue(evaluation["output_dir"])
        self.assertIsInstance(evaluation["random_seed"], int)

    def test_probe_distances_and_steps_are_positive(self):
        """移动距离、控制次数和停止阈值不能为零或负数。

        如果这些值非法，闭环可能完全不动、循环次数为零，或者使用没有物理
        意义的负距离，所以应当在仿真启动前失败。
        """
        probe = self.config["probe"]
        self.assertGreater(probe["move_step_xy"], 0)
        self.assertGreater(probe["max_control_steps"], 0)
        self.assertGreater(probe["sim_steps_per_action"], 0)
        self.assertGreater(probe["stop_distance_xy"], 0)

    def test_probe_mode_is_supported(self):
        """当前决策入口只实现 heuristic 和 api 两种模式。"""
        self.assertIn(self.config["probe"]["mode"].lower(), {"heuristic", "api"})

    def test_capture_and_termination_limits_are_consistent(self):
        """采样间隔应为正数，强制终止点必须早于 episode 的绝对上限。

        后一个关系保证代码有机会明确记录 ``max_steps`` 终止原因，而不是先被
        外层循环上限截断，导致 episode 没有可诊断的结束状态。
        """
        dataset = self.config["dataset"]
        task = self.config["task"]
        self.assertGreater(dataset["capture_interval_steps"], 0)
        self.assertLess(
            task["force_terminal_after_step"],
            dataset["max_steps_per_episode"],
        )
    def test_probe_api_limits_are_valid(self):
        api = self.config["probe"]["api"]
        self.assertGreater(api["timeout_seconds"], 0)
        self.assertGreaterEqual(api["max_retries"], 0)

    def test_end_effector_marker_config_is_valid(self):
        """VLM 末端标记开关和 RGBA 颜色配置必须合法。"""
        marker = self.config["probe"]["end_effector_marker"]

        self.assertIs(marker["enabled"], False)
        self.assertEqual(len(marker["color_rgba"]), 4)
        self.assertTrue(all(0 <= channel <= 1 for channel in marker["color_rgba"]))
        self.assertGreater(marker["radius"], 0)

        evaluation = self.config["vlm_evaluation"]
        self.assertGreater(evaluation["balanced_pose_offset_xy"], 0)


if __name__ == "__main__":
    # 既支持测试发现，也支持直接执行：python tests/test_config_contract.py
    unittest.main()
