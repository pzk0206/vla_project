"""保护标准 src 包结构和命令入口，防止目录整理后再次漂移。"""

import importlib
import importlib.metadata
import unittest


EXPECTED_SCRIPTS = {
    "vla-collect": "vla_project.simulation.control_arm:main",
    "vla-audit-action-tokenization": (
        "vla_project.simulation.audit_action_tokenization:main"
    ),
    "vla-audit-dataset-visibility": (
        "vla_project.simulation.audit_dataset_visibility:main"
    ),
    "vla-evaluate-dataset": "vla_project.simulation.evaluate_dataset:main",
    "vla-probe": "vla_project.simulation.stage3_probe:main",
    "vla-evaluate-probe": "vla_project.simulation.evaluate_probe:main",
    "vla-collect-vlm-samples": "vla_project.vlm.collect_vlm_eval_samples:main",
    "vla-evaluate-vlm-decisions": "vla_project.vlm.evaluate_vlm_decisions:main",
    "vla-diagnose-grounding": "vla_project.vlm.diagnose_vlm_grounding:main",
    "vla-ground-then-decide": "vla_project.vlm.evaluate_ground_then_decide:main",
    "vla-evaluate-backprojection": (
        "vla_project.vlm.evaluate_grounding_backprojection:main"
    ),
    "vla-validate-grounding-calibration": (
        "vla_project.vlm.validate_grounding_calibration:main"
    ),
    "vla-migrate-generated-outputs": (
        "vla_project.tools.migrate_generated_outputs:main"
    ),
    "vla-run-grounding-smoke": "vla_project.vlm.grounding_smoke.runner:main",
    "vla-screen-grounding-smoke": (
        "vla_project.vlm.grounding_smoke.screening:main"
    ),
}


class PackageLayoutTests(unittest.TestCase):
    def test_domain_packages_are_importable(self):
        for module_name in (
            "vla_project",
            "vla_project.simulation",
            "vla_project.vlm",
            "vla_project.vlm.grounding_smoke",
            "vla_project.tools",
        ):
            with self.subTest(module_name=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))


class ConsoleScriptTests(unittest.TestCase):
    def test_installed_distribution_exposes_exact_console_scripts(self):
        distribution = importlib.metadata.distribution("vla-project")
        actual_scripts = {
            entry_point.name: entry_point.value
            for entry_point in distribution.entry_points
            if entry_point.group == "console_scripts"
        }
        self.assertEqual(actual_scripts, EXPECTED_SCRIPTS)

    def test_console_script_targets_are_importable_main_functions(self):
        for script_name, target in EXPECTED_SCRIPTS.items():
            module_name, function_name = target.split(":", maxsplit=1)
            with self.subTest(script_name=script_name):
                module = importlib.import_module(module_name)
                self.assertTrue(callable(getattr(module, function_name)))


if __name__ == "__main__":
    unittest.main()
