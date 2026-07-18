"""保护标准 src 包结构和命令入口，防止目录整理后再次漂移。"""

import importlib
import unittest


class PackageLayoutTests(unittest.TestCase):
    def test_domain_packages_are_importable(self):
        for module_name in (
            "vla_project",
            "vla_project.simulation",
            "vla_project.vlm",
            "vla_project.tools",
        ):
            with self.subTest(module_name=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))


if __name__ == "__main__":
    unittest.main()
