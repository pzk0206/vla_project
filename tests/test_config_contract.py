"""检查 ``sim_config.yaml`` 是否满足项目代码依赖的最低配置契约。

这类测试不启动 PyBullet，也不评价机械臂表现。它的作用是在正式运行采集或
评估前，尽早发现配置缺字段、数值非法或两个参数互相矛盾的问题。
"""

import unittest

from control_arm import load_config


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
            }
            <= self.config.keys()
        )

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


if __name__ == "__main__":
    # 既支持测试发现，也支持直接执行：python tests/test_config_contract.py
    unittest.main()
