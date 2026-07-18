"""验证生成输出迁移不会覆盖数据，并会同步修复记录中的图片路径。"""

import json
import tempfile
import unittest
from pathlib import Path

from vla_project.tools.migrate_generated_outputs import (
    MIGRATIONS,
    MigrationConflictError,
    MigrationError,
    migrate_generated_outputs,
    preflight_migration,
    rewrite_path,
    validate_migrated_outputs,
)


class GeneratedOutputMigrationTests(unittest.TestCase):
    """所有测试只操作临时目录，不接触仓库里的真实实验输出。"""

    def make_source_trees(self, root):
        """创建八个最小旧目录，模拟当前主工作区布局。"""
        for source, _ in MIGRATIONS:
            source_dir = root / source
            source_dir.mkdir(parents=True)
            (source_dir / "marker.txt").write_text(source, encoding="utf-8")

    def test_rewrite_path_uses_exact_prefix_boundaries(self):
        self.assertEqual(
            rewrite_path("vlm_eval_samples_448/images/frame.jpg"),
            "outputs/vlm_samples/448/images/frame.jpg",
        )
        self.assertEqual(
            rewrite_path("dataset/frame.jpg"),
            "outputs/dataset/frame.jpg",
        )
        self.assertEqual(
            rewrite_path("dataset_backup/frame.jpg"),
            "dataset_backup/frame.jpg",
        )

    def test_rewrite_path_updates_absolute_project_paths(self):
        self.assertEqual(
            rewrite_path(
                "/home/pzk/vla_project/vlm_eval_runs/run/backprojection.jsonl"
            ),
            "/home/pzk/vla_project/outputs/vlm_evaluations/run/backprojection.jsonl",
        )

    def test_preflight_reports_inventory_without_moving_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_source_trees(root)

            summary = preflight_migration(root)

            self.assertEqual(summary["source_directories"], len(MIGRATIONS))
            self.assertEqual(summary["files"], len(MIGRATIONS))
            self.assertTrue(all((root / source).is_dir() for source, _ in MIGRATIONS))
            self.assertFalse((root / "outputs").exists())

    def test_migration_moves_files_and_rewrites_jsonl_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_source_trees(root)
            image = root / "vlm_eval_samples_448" / "images" / "frame.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"jpeg-placeholder")
            manifest = root / "vlm_eval_samples_448" / "samples.jsonl"
            manifest.write_text(
                json.dumps(
                    {"image_path": "vlm_eval_samples_448/images/frame.jpg"}
                )
                + "\n",
                encoding="utf-8",
            )

            summary = migrate_generated_outputs(root)

            self.assertEqual(summary["moved_directories"], len(MIGRATIONS))
            self.assertFalse((root / "vlm_eval_samples_448").exists())
            moved_manifest = root / "outputs/vlm_samples/448/samples.jsonl"
            row = json.loads(moved_manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                row["image_path"],
                "outputs/vlm_samples/448/images/frame.jpg",
            )
            self.assertTrue((root / row["image_path"]).is_file())
            validated = validate_migrated_outputs(root)
            self.assertEqual(validated["images"], 1)

    def test_destination_collision_changes_nothing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_source_trees(root)
            (root / "outputs/dataset").mkdir(parents=True)

            with self.assertRaises(MigrationConflictError):
                migrate_generated_outputs(root)

            self.assertTrue(all((root / source).is_dir() for source, _ in MIGRATIONS))

    def test_preexisting_missing_image_reference_is_preserved_not_rejected(self):
        """历史清理策略允许悬空引用，但迁移不能新增悬空引用。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_source_trees(root)
            trace = root / "probe_eval_runs" / "run_001" / "probe_trace.jsonl"
            trace.parent.mkdir(parents=True)
            trace.write_text(
                json.dumps(
                    {"image_path": "probe_eval_runs/run_001/probe_step_00.jpg"}
                )
                + "\n",
                encoding="utf-8",
            )

            before = preflight_migration(root)
            after = migrate_generated_outputs(root)

            self.assertEqual(before["missing_image_references"], 1)
            self.assertEqual(after["missing_image_references"], 1)
            moved_trace = (
                root
                / "outputs/probe_evaluations/run_001/probe_trace.jsonl"
            )
            row = json.loads(moved_trace.read_text(encoding="utf-8"))
            self.assertEqual(
                row["image_path"],
                "outputs/probe_evaluations/run_001/probe_step_00.jpg",
            )

    def test_worktree_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / ".worktrees" / "experiment"
            root.mkdir(parents=True)
            self.make_source_trees(root)

            with self.assertRaises(MigrationError):
                preflight_migration(root)


if __name__ == "__main__":
    unittest.main()
