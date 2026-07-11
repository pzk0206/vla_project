import unittest

from control_arm import load_config


class ConfigContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("sim_config.yaml")

    def test_required_sections_exist(self):
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
        evaluation = self.config["probe_evaluation"]
        self.assertGreater(evaluation["num_episodes"], 0)
        self.assertTrue(evaluation["output_dir"])
        self.assertIsInstance(evaluation["random_seed"], int)

    def test_probe_distances_and_steps_are_positive(self):
        probe = self.config["probe"]
        self.assertGreater(probe["move_step_xy"], 0)
        self.assertGreater(probe["max_control_steps"], 0)
        self.assertGreater(probe["sim_steps_per_action"], 0)
        self.assertGreater(probe["stop_distance_xy"], 0)

    def test_probe_mode_is_supported(self):
        self.assertIn(self.config["probe"]["mode"].lower(), {"heuristic", "api"})

    def test_capture_and_termination_limits_are_consistent(self):
        dataset = self.config["dataset"]
        task = self.config["task"]
        self.assertGreater(dataset["capture_interval_steps"], 0)
        self.assertLess(
            task["force_terminal_after_step"],
            dataset["max_steps_per_episode"],
        )


if __name__ == "__main__":
    unittest.main()
