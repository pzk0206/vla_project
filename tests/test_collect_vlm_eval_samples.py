"""测试离线 VLM 评估样本的最小数据契约。

这些测试只使用临时图片和内存中的样本字典，不启动 PyBullet，也不会请求真实
VLM。目的是在批量采样和付费评估前，尽早发现字段缺失、方向非法、图片损坏或
样本 ID 重复。
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from collect_vlm_eval_samples import (
    build_balanced_ee_positions,
    build_sample_diagnostic,
    build_sampling_config,
    build_stratified_balanced_cases,
    collect_vlm_eval_samples,
    reset_robot_to_target,
    select_evenly_spaced_rows,
    validate_diagnostics,
    validate_samples,
    write_sample_files,
)


class SampleDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.camera_config = {
            "workspace_center": [0.0, 0.4, 0.0],
            "up_vector": [0.0, 1.0, 0.0],
            "image_width": 448,
            "image_height": 448,
            "fov": 45,
            "near_val": 0.1,
            "far_val": 100.0,
        }

    def test_builds_reproducible_diagnostic_without_mutating_sample(self):
        sample = {
            "sample_id": "seed_42_d020_left",
            "image_path": "frame.jpg",
            "instruction": "悬停在红色积木上方",
            "expected_direction": "left",
            "camera_eye": [0.0, 0.4, 3.0],
        }

        diagnostic = build_sample_diagnostic(
            sample["sample_id"],
            [0.01, 0.44, 0.05],
            sample["camera_eye"],
            self.camera_config,
        )

        self.assertNotIn("block_pos", sample)
        self.assertEqual(diagnostic["sample_id"], sample["sample_id"])
        self.assertEqual(diagnostic["block_pos"], [0.01, 0.44, 0.05])
        self.assertEqual(len(diagnostic["view_matrix"]), 16)
        self.assertEqual(len(diagnostic["projection_matrix"]), 16)
        self.assertEqual(diagnostic["image_width"], 448)
        self.assertEqual(diagnostic["image_height"], 448)

    def test_rejects_duplicate_diagnostic_sample_ids(self):
        row = build_sample_diagnostic(
            "duplicate",
            [0.01, 0.44, 0.05],
            [0.0, 0.4, 3.0],
            self.camera_config,
        )

        with self.assertRaisesRegex(ValueError, "重复"):
            validate_diagnostics([row, dict(row)])

    def test_writes_truth_to_separate_file(self):
        sample = {
            "sample_id": "seed_42_d020_left",
            "image_path": "frame.jpg",
            "instruction": "悬停在红色积木上方",
            "expected_direction": "left",
            "camera_eye": [0.0, 0.4, 3.0],
        }
        diagnostic = build_sample_diagnostic(
            sample["sample_id"],
            [0.01, 0.44, 0.05],
            sample["camera_eye"],
            self.camera_config,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path, diagnostics_path = write_sample_files(
                Path(temp_dir), [sample], [diagnostic]
            )
            manifest_row = json.loads(manifest_path.read_text(encoding="utf-8"))
            diagnostic_row = json.loads(
                diagnostics_path.read_text(encoding="utf-8")
            )

        self.assertNotIn("block_pos", manifest_row)
        self.assertIn("block_pos", diagnostic_row)

    @patch("collect_vlm_eval_samples.validate_samples")
    @patch("collect_vlm_eval_samples.capture_balanced_pose_sample")
    @patch("collect_vlm_eval_samples.build_sampling_config")
    def test_balanced_collection_writes_matching_diagnostics(
        self, build_config, capture_sample, _validate_samples
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "probe": {"mode": "heuristic"},
                "probe_evaluation": {"random_seed": 42},
                "dataset": {"instruction": "悬停在红色积木上方"},
                "vlm_evaluation": {
                    "sample_output_dir": temp_dir,
                    "sample_strategy": "stratified_balanced_poses",
                    "balanced_pose_offsets_xy": [0.2],
                    "stratified_num_seeds": 5,
                },
                "camera": self.camera_config,
            }
            build_config.return_value = config
            capture_sample.return_value = {
                "camera_eye": [0.0, 0.4, 3.0],
                "block_pos": [0.01, 0.44, 0.05],
            }

            manifest_path, samples = collect_vlm_eval_samples(config)
            diagnostics_path = Path(temp_dir) / "diagnostics.jsonl"

            diagnostics = [
                json.loads(line)
                for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(samples), 20)
        self.assertEqual(len(diagnostics), 20)
        self.assertEqual(capture_sample.call_count, 20)
        self.assertEqual(
            {row["random_seed"] for row in samples},
            {42, 43, 44, 45, 46},
        )
        self.assertEqual(
            {
                direction: sum(
                    row["expected_direction"] == direction for row in samples
                )
                for direction in ("left", "right", "front", "back")
            },
            {"left": 5, "right": 5, "front": 5, "back": 5},
        )
        self.assertEqual(
            {row["sample_id"] for row in samples},
            {row["sample_id"] for row in diagnostics},
        )
        self.assertTrue(manifest_path.name == "samples.jsonl")
        self.assertTrue(all("block_pos" not in row for row in samples))


class ValidateVlmEvalSamplesTests(unittest.TestCase):
    """固定离线样本的字段、图片、方向和唯一性规则。"""

    def setUp(self):
        """为每个测试创建独立临时项目目录和一张可读取的 JPEG。"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        self.image_path = Path(
            "vlm_eval_samples/images/seed_42_step_000.jpg"
        )
        absolute_image_path = self.project_root / self.image_path
        absolute_image_path.parent.mkdir(parents=True)

        # 使用真实 JPEG 验证图片不仅存在，而且能被 OpenCV 解码。
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(absolute_image_path), image))

    def tearDown(self):
        """测试完成后自动清理临时图片和目录。"""
        self.temp_dir.cleanup()

    def make_valid_sample(self):
        """返回一条满足计划中最小 schema 的合法样本。"""
        return {
            "sample_id": "seed_42_step_000",
            "image_path": self.image_path.as_posix(),
            "instruction": "悬停在红色积木上方",
            "expected_direction": "right",
            "random_seed": 42,
            "control_step": 0,
            "camera_eye": [1.0, 0.4, 1.6],
        }

    def test_valid_sample_is_accepted(self):
        """字段完整、方向合法且图片可读的样本应通过校验。"""
        validate_samples([self.make_valid_sample()], self.project_root)

    def test_missing_required_field_is_rejected(self):
        """缺少任务指令会使样本失去评估含义，因此必须拒绝。"""
        sample = self.make_valid_sample()
        del sample["instruction"]

        with self.assertRaisesRegex(ValueError, "instruction"):
            validate_samples([sample], self.project_root)

    def test_invalid_direction_is_rejected(self):
        """方向必须限制为 left/right/front/back/stop 五个离散标签。"""
        sample = self.make_valid_sample()
        sample["expected_direction"] = "up"

        with self.assertRaisesRegex(ValueError, "expected_direction"):
            validate_samples([sample], self.project_root)

    def test_unreadable_image_is_rejected(self):
        """路径存在但无法解码的文件不能作为视觉评估输入。"""
        sample = self.make_valid_sample()
        absolute_image_path = self.project_root / sample["image_path"]
        absolute_image_path.write_text("not a jpeg", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "image_path"):
            validate_samples([sample], self.project_root)

    def test_duplicate_sample_ids_are_rejected(self):
        """重复 ID 会污染统计和结果对齐，因此必须在评估前失败。"""
        first = self.make_valid_sample()
        duplicate = self.make_valid_sample()

        with self.assertRaisesRegex(ValueError, "sample_id"):
            validate_samples([first, duplicate], self.project_root)


class SelectEvenlySpacedRowsTests(unittest.TestCase):
    """检查长 trace 是否被均匀压缩为少量代表帧。"""

    def test_short_trace_keeps_every_row(self):
        """不足上限时不应丢失任何有效控制步。"""
        rows = [{"control_step": step} for step in range(3)]
        self.assertEqual(select_evenly_spaced_rows(rows, 10), rows)

    def test_long_trace_keeps_limit_and_both_ends(self):
        """长 trace 应保留首尾并严格限制样本数量。"""
        rows = [{"control_step": step} for step in range(25)]
        selected = select_evenly_spaced_rows(rows, 10)

        self.assertEqual(len(selected), 10)
        self.assertEqual(selected[0]["control_step"], 0)
        self.assertEqual(selected[-1]["control_step"], 24)


class BalancedEePositionsTests(unittest.TestCase):
    """固定五类离散方向对应的末端采样位置。"""

    def test_places_end_effector_on_each_side_of_block(self):
        positions = build_balanced_ee_positions(
            block_pos=[0.1, 0.4, 0.05],
            hover_height=0.15,
            offset_xy=0.2,
        )

        self.assertEqual(positions["left"], [0.3, 0.4, 0.2])
        self.assertEqual(positions["right"], [-0.1, 0.4, 0.2])
        self.assertEqual(positions["front"], [0.1, 0.2, 0.2])
        self.assertEqual(positions["back"], [0.1, 0.6, 0.2])
        self.assertEqual(positions["stop"], [0.1, 0.4, 0.2])

    def test_builds_four_directions_for_each_distance(self):
        """分层集必须按距离覆盖四方向，并生成可复现的唯一标签。"""
        cases = build_stratified_balanced_cases([0.20, 0.10, 0.05])

        self.assertEqual(len(cases), 12)
        self.assertEqual(
            cases[:4],
            [
                {"offset_xy": 0.20, "offset_tag": "d020", "direction": "left"},
                {"offset_xy": 0.20, "offset_tag": "d020", "direction": "right"},
                {"offset_xy": 0.20, "offset_tag": "d020", "direction": "front"},
                {"offset_xy": 0.20, "offset_tag": "d020", "direction": "back"},
            ],
        )
        self.assertEqual(cases[-1]["offset_tag"], "d005")
        self.assertEqual(
            len({(case["offset_tag"], case["direction"]) for case in cases}),
            12,
        )

    @patch("collect_vlm_eval_samples.p.resetJointState")
    @patch(
        "collect_vlm_eval_samples.get_link_position",
        side_effect=[[0.2, 0.2, 0.3], [0.1, 0.4, 0.2]],
    )
    @patch(
        "collect_vlm_eval_samples.calculate_target_joints",
        return_value=[0.0] * 7,
    )
    def test_iterates_ik_until_target_pose_converges(
        self,
        calculate_joints,
        get_link_position,
        reset_joint,
    ):
        actual = reset_robot_to_target(
            robot_id=3,
            robot_config={"controlled_joints": 7, "ee_link_index": 6},
            target_pos=[0.1, 0.4, 0.2],
            tolerance=0.001,
            max_iterations=5,
        )

        self.assertEqual(actual, [0.1, 0.4, 0.2])
        self.assertEqual(calculate_joints.call_count, 2)
        self.assertEqual(reset_joint.call_count, 14)


class BuildSamplingConfigTests(unittest.TestCase):
    """验证 VLM 采样只覆盖相机，不修改原始 baseline 配置。"""

    def test_applies_vlm_camera_override_to_a_copy(self):
        config = {
            "probe": {"mode": "heuristic", "save_trace_images": False},
            "camera": {
                "eye_offset_base": [1.0, 0.0, 1.6],
                "up_vector": [0, 0, 1],
            },
            "vlm_evaluation": {
                "use_dual_view": True,
                "camera_override": {
                    "eye_offset_base": [0.0, 0.0, 2.0],
                    "up_vector": [0, 1, 0],
                },
                "secondary_camera_override": {
                    "eye_offset_base": [0.0, -0.8, 2.0],
                    "up_vector": [0, 1, 0],
                },
            },
        }

        sampling_config = build_sampling_config(config)

        self.assertEqual(
            sampling_config["camera"]["eye_offset_base"],
            [0.0, 0.0, 2.0],
        )
        self.assertEqual(sampling_config["camera"]["up_vector"], [0, 1, 0])
        self.assertTrue(sampling_config["probe"]["save_trace_images"])
        self.assertEqual(
            sampling_config["probe"]["secondary_camera"]["eye_offset_base"],
            [0.0, -0.8, 2.0],
        )
        self.assertEqual(config["camera"]["eye_offset_base"], [1.0, 0.0, 1.6])


if __name__ == "__main__":
    unittest.main()
