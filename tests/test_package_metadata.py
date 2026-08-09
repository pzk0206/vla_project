"""保护标准 src 包结构和命令入口，防止目录整理后再次漂移。"""

import importlib
import importlib.metadata
from pathlib import Path
import unittest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 development environment
    from setuptools._vendor import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def declared_scripts() -> dict[str, str]:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["scripts"]


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

    def test_training_runtime_dependencies_are_declared(self):
        requirements = {
            line.split("#", maxsplit=1)[0].strip().lower().split(">=", maxsplit=1)[0]
            for line in (PROJECT_ROOT / "requirements.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.split("#", maxsplit=1)[0].strip()
        }
        self.assertIn("pillow", requirements)
        self.assertIn("sentence-transformers", requirements)


class ConsoleScriptTests(unittest.TestCase):
    def test_installed_distribution_exposes_exact_console_scripts(self):
        distribution = importlib.metadata.distribution("vla-project")
        actual_scripts = {
            entry_point.name: entry_point.value
            for entry_point in distribution.entry_points
            if entry_point.group == "console_scripts"
        }
        self.assertEqual(actual_scripts, declared_scripts())

    def test_console_script_targets_are_importable_main_functions(self):
        for script_name, target in declared_scripts().items():
            module_name, function_name = target.split(":", maxsplit=1)
            with self.subTest(script_name=script_name):
                module = importlib.import_module(module_name)
                self.assertTrue(callable(getattr(module, function_name)))

    def test_v2_pipeline_commands_are_declared(self):
        scripts = declared_scripts()
        self.assertEqual(
            scripts["vla-evaluate-vla-counterfactual"],
            "vla_project.training.vla_counterfactual:main",
        )
        self.assertEqual(
            scripts["vla-split-episodes"],
            "vla_project.simulation.split_episodes:main",
        )


if __name__ == "__main__":
    unittest.main()
