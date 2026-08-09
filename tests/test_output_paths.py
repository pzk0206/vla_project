"""跨领域输出路径边界测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vla_project import output_paths


class ResolveManagedOutputTests(unittest.TestCase):
    """路径规范化必须在任何文件系统写操作前阻断逃逸。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_root = Path(self.temp_dir.name)
        self.root = self.temp_root / "project"
        self.managed_root = self.root / "outputs" / "dataset"
        self.managed_root.mkdir(parents=True)
        self.outside = self.temp_root / "outside"
        self.outside.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def resolve(self, requested_path):
        return output_paths.resolve_managed_output(
            requested_path,
            allowed_root="outputs/dataset",
            project_root_override=self.root,
        )

    def test_accepts_relative_child_and_returns_absolute_path(self):
        actual = self.resolve("outputs/dataset/expert_v2")

        self.assertEqual(
            actual,
            self.root / "outputs" / "dataset" / "expert_v2",
        )
        self.assertTrue(actual.is_absolute())

    def test_accepts_absolute_path_inside_managed_root(self):
        requested = self.managed_root / "expert_v2"

        self.assertEqual(self.resolve(requested), requested)

    def test_rejects_allowed_root_itself(self):
        with self.assertRaisesRegex(ValueError, "leaf directory"):
            self.resolve("outputs/dataset")

    def test_rejects_parent_segments_even_when_result_would_be_inside_root(self):
        with self.assertRaisesRegex(ValueError, "parent traversal"):
            self.resolve("outputs/dataset/a/../b")

    def test_rejects_escape_to_source_tree(self):
        with self.assertRaisesRegex(ValueError, "parent traversal"):
            self.resolve("outputs/dataset/../../src")

    def test_rejects_absolute_path_outside_managed_root(self):
        with self.assertRaisesRegex(ValueError, "outside managed root"):
            self.resolve(self.outside / "run")

    def test_rejects_symlink_parent_that_resolves_outside_root(self):
        (self.managed_root / "link").symlink_to(
            self.outside,
            target_is_directory=True,
        )

        with self.assertRaisesRegex(ValueError, "outside managed root"):
            self.resolve("outputs/dataset/link/run")

    def test_rejects_empty_current_and_parent_paths(self):
        for requested in ("", ".", ".."):
            with self.subTest(requested=requested):
                with self.assertRaises(ValueError):
                    self.resolve(requested)

    def test_rejects_nul_character(self):
        with self.assertRaises(ValueError):
            self.resolve("outputs/dataset/bad\x00name")


class ValidateRunNameTests(unittest.TestCase):
    """显式运行名不能携带任何目录结构。"""

    def test_accepts_single_safe_component(self):
        self.assertEqual(
            output_paths.validate_run_name("run_20260809_v1"),
            "run_20260809_v1",
        )

    def test_rejects_directory_components(self):
        for value in (
            "../escape",
            "a/b",
            "a\\b",
            ".",
            "..",
            "/tmp/run",
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    output_paths.validate_run_name(value)

    def test_rejects_empty_and_non_string_names(self):
        for value in ("", "   ", None, Path("run")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    output_paths.validate_run_name(value)


class AtomicPublishTests(unittest.TestCase):
    """目录发布失败时必须保留旧证据，且不能移动根外目录。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "project"
        self.managed_root = self.root / "outputs" / "vlm_samples"
        self.managed_root.mkdir(parents=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def publish(self, staging, destination):
        return output_paths.publish_directory_atomically(
            staging,
            destination,
            allowed_root="outputs/vlm_samples",
            project_root_override=self.root,
        )

    def test_publishes_new_directory(self):
        staging = self.managed_root / ".run_staging"
        staging.mkdir()
        (staging / "new.txt").write_text("new", encoding="utf-8")
        destination = self.managed_root / "run"

        actual = self.publish(staging, destination)

        self.assertEqual(actual, destination)
        self.assertFalse(staging.exists())
        self.assertEqual(
            (destination / "new.txt").read_text(encoding="utf-8"),
            "new",
        )

    def test_successful_publish_replaces_existing_directory(self):
        destination = self.managed_root / "run"
        destination.mkdir()
        (destination / "old.txt").write_text("old", encoding="utf-8")
        staging = self.managed_root / ".run_staging"
        staging.mkdir()
        (staging / "new.txt").write_text("new", encoding="utf-8")

        self.publish(staging, destination)

        self.assertFalse((destination / "old.txt").exists())
        self.assertEqual(
            (destination / "new.txt").read_text(encoding="utf-8"),
            "new",
        )

    def test_failed_publish_restores_previous_directory(self):
        destination = self.managed_root / "run"
        destination.mkdir()
        (destination / "old.txt").write_text("old", encoding="utf-8")
        staging = self.managed_root / ".run_staging"
        staging.mkdir()
        (staging / "new.txt").write_text("new", encoding="utf-8")
        original_replace = Path.replace
        call_count = 0

        def fail_second_replace(path, target):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("publish failed")
            return original_replace(path, target)

        with patch.object(
            Path,
            "replace",
            autospec=True,
            side_effect=fail_second_replace,
        ):
            with self.assertRaisesRegex(OSError, "publish failed"):
                self.publish(staging, destination)

        self.assertEqual(
            (destination / "old.txt").read_text(encoding="utf-8"),
            "old",
        )
        self.assertFalse((destination / "new.txt").exists())

    def test_rejects_staging_outside_managed_root_before_moving_destination(self):
        destination = self.managed_root / "run"
        destination.mkdir()
        marker = destination / "old.txt"
        marker.write_text("old", encoding="utf-8")
        outside_staging = self.root / "outside"
        outside_staging.mkdir()

        with self.assertRaisesRegex(ValueError, "outside managed root"):
            self.publish(outside_staging, destination)

        self.assertEqual(marker.read_text(encoding="utf-8"), "old")

    def test_context_failure_preserves_existing_directory(self):
        destination = self.managed_root / "run"
        destination.mkdir()
        marker = destination / "old.txt"
        marker.write_text("old", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            with output_paths.staged_output_directory(
                destination,
                allowed_root="outputs/vlm_samples",
                project_root_override=self.root,
            ) as staging:
                (staging / "new.txt").write_text("new", encoding="utf-8")
                raise RuntimeError("generation failed")

        self.assertEqual(marker.read_text(encoding="utf-8"), "old")

    def test_context_success_publishes_generated_directory(self):
        destination = self.managed_root / "run"

        with output_paths.staged_output_directory(
            destination,
            allowed_root="outputs/vlm_samples",
            project_root_override=self.root,
        ) as staging:
            (staging / "new.txt").write_text("new", encoding="utf-8")

        self.assertEqual(
            (destination / "new.txt").read_text(encoding="utf-8"),
            "new",
        )


if __name__ == "__main__":
    unittest.main()
