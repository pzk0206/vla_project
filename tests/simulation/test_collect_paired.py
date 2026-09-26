"""测试 collect_paired：为 expert_multi_v2 补录反色指令专家轨迹。

覆盖纯逻辑（反色映射、场景契约、摘要校验）、原始轨迹复制、补录轨迹执行与
成对数据契约。外部仿真接口使用 mock，文件测试使用临时目录，不打开 PyBullet
窗口，也不改动真实 outputs/dataset/。
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from vla_project.simulation import collect_paired


def _make_scene_state(target_block, red_pos, blue_pos, ee_pos=None, camera_eye=None):
    """构造合法 expert_multi_v2 scene_state。"""
    return {
        "target_block": target_block,
        "blocks": {
            "red": {"position": list(red_pos), "orientation": [0, 0, 0, 1]},
            "blue": {"position": list(blue_pos), "orientation": [0, 0, 0, 1]},
        },
        "robot": {
            "joint_positions": [0.0] * 7,
            "joint_velocities": [0.0] * 7,
            "ee_position": list(ee_pos if ee_pos is not None else red_pos),
        },
        "camera_eye": list(camera_eye if camera_eye is not None else [1.0, 0.4, 1.6]),
    }


def _make_summary(episode_idx, target_block, red_pos, blue_pos):
    """构造 expert_multi_v2 摘要。"""
    return {
        "schema_version": "expert_multi_v2",
        "episode_idx": episode_idx,
        "random_seed": 1000 + episode_idx,
        "initial_ee_pos": [0, 0, 1.26],
        "initial_block_pos": list(red_pos if target_block == "red" else blue_pos),
        "num_steps": 32,
        "num_frames": 32,
        "final_distance": 0.029,
        "termination_reason": "success",
        "camera_eye": [1.0, 0.4, 1.6],
        "final_block_pos": list(red_pos if target_block == "red" else blue_pos),
        "final_target_pos": [0.1, 0.4, 0.25],
        "final_ee_pos": [0.1, 0.4, 0.25],
        "task_instruction": "悬停在红色积木上方" if target_block == "red" else "悬停在蓝色积木上方",
        "target_block": target_block,
        "initial_scene_state": _make_scene_state(target_block, red_pos, blue_pos),
        "final_scene_state": _make_scene_state(target_block, red_pos, blue_pos),
    }


def _make_config(output_dir, schema_version="expert_multi_v2"):
    """构造最小可用采集配置（与 sim_config.yaml 的 dataset/task/robot 契约对齐）。"""
    return {
        "connection_mode": "DIRECT",
        "gravity": [0, 0, -9.8],
        "enable_time_sleep": False,
        "simulation_hz": 240,
        "robot": {
            "urdf_path": "kuka_iiwa/model.urdf",
            "base_position": [0, 0, 0],
            "use_fixed_base": True,
            "ee_link_index": 6,
            "controlled_joints": 7,
            "max_force": 100,
            "max_velocity": 0.6,
            "ik_residual_threshold": 0.001,
            "target_orientation_euler": [0, 3.141592653589793, 0],
        },
        "dataset": {
            "output_dir": output_dir,
            "task_selection": "balanced_alternating",
            "jsonl_name": "trajectory_expert.jsonl",
            "summary_jsonl_name": "episode_summary.jsonl",
            "schema_version": schema_version,
            "clean_before_run": True,
            "num_episodes": 300,
            "pilot_num_episodes": 10,
            "target_num_episodes": 300,
            "random_seed": 1000,
            "reset_robot_each_episode": True,
            "home_joint_positions": [0.0] * 7,
            "max_steps_per_episode": 1000,
            "capture_interval_steps": 24,
            "instruction": "悬停在红色积木上方",
            "default_gripper_state": 1.0,
        },
        "task": {
            "block_urdf_path": "cube.urdf",
            "block_position": {"x_range": [-0.2, 0.2], "y_range": [0.38, 0.5], "z": 0.1},
            "block_global_scaling": 0.1,
            "block_color_rgba": [1, 0, 0, 1],
            "hover_height": 0.15,
            "success_distance": 0.03,
            "force_terminal_after_step": 976,
            "initial_settle_steps": 30,
            "stuck_window_steps": 120,
            "stuck_min_improvement": 0.0005,
            "pair_sampling": {
                "enabled": True,
                "x_range": [-0.2, 0.2],
                "y_range": [0.38, 0.5],
                "z": 0.1,
                "min_axis_separation_xy": 0.12,
                "max_attempts": 100,
                "max_settle_drift_xy": 0.005,
                "max_episode_drift_xy": 0.005,
            },
            "second_block": {
                "enabled": True,
                "color_rgba": [0, 0, 1, 1],
                "x_range": [0.05, 0.2],
                "y_range": [0.40, 0.50],
                "z": 0.1,
                "global_scaling": 0.1,
            },
            "tasks": [
                {"instruction": "悬停在红色积木上方", "target_block": "red"},
                {"instruction": "悬停在蓝色积木上方", "target_block": "blue"},
            ],
        },
        "camera": {
            "workspace_center": [0.0, 0.4, 0.0],
            "eye_offset_base": [1.05, 0.0, 1.65],
            "eye_offset_random_range": [-0.1, 0.1],
            "up_vector": [0, 0, 1],
            "image_width": 224,
            "image_height": 224,
            "fov": 50,
            "near_val": 0.1,
            "far_val": 100.0,
        },
    }


class OppositeTargetTests(unittest.TestCase):
    """反色目标映射纯逻辑。"""

    def test_red_blue_are_opposites(self):
        self.assertEqual(collect_paired._opposite("red"), "blue")
        self.assertEqual(collect_paired._opposite("blue"), "red")

    def test_opposite_red_and_blue_is_identity(self):
        for color in ("red", "blue"):
            self.assertEqual(collect_paired._opposite(collect_paired._opposite(color)), color)

    def test_invalid_target_rejected(self):
        with self.assertRaises(ValueError):
            collect_paired._opposite("green")


class ExtractSceneSpecTests(unittest.TestCase):
    """冻结场景提取契约。"""

    def test_extracts_frozen_scene(self):
        red_pos = [0.1, 0.4, 0.1]
        blue_pos = [-0.15, 0.42, 0.1]
        summary = _make_summary(7, "red", red_pos, blue_pos)
        spec = collect_paired.extract_saved_scene_spec(summary)
        self.assertEqual(spec["episode_idx"], 7)
        self.assertEqual(spec["random_seed"], 1007)
        self.assertEqual(spec["blocks"]["red"]["position"], red_pos)
        self.assertEqual(spec["blocks"]["blue"]["position"], blue_pos)
        self.assertEqual(spec["camera_eye"], [1.0, 0.4, 1.6])
        self.assertEqual(spec["joint_positions"], [0.0] * 7)

    def test_non_v2_summary_rejected(self):
        summary = _make_summary(0, "red", [0.1, 0.4, 0.1], [-0.15, 0.42, 0.1])
        summary["schema_version"] = "expert_v1"
        with self.assertRaises(ValueError):
            collect_paired.extract_saved_scene_spec(summary)

    def test_missing_blocks_rejected(self):
        summary = _make_summary(0, "red", [0.1, 0.4, 0.1], [-0.15, 0.42, 0.1])
        del summary["initial_scene_state"]["blocks"]
        with self.assertRaises(ValueError):
            collect_paired.extract_saved_scene_spec(summary)


class LoadSummariesTests(unittest.TestCase):
    """源摘要加载与 300 条红蓝平衡契约。"""

    def _write_source(self, temp_dir, count=300):
        rows = []
        for idx in range(count):
            target = "red" if idx < 150 else "blue"
            red_pos = [0.1, 0.4, 0.1]
            blue_pos = [-0.15, 0.42, 0.1]
            rows.append(_make_summary(idx, target, red_pos, blue_pos))
        path = Path(temp_dir) / "episode_summary.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def test_loads_300_balanced_summaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_source(temp_dir)
            rows = collect_paired._load_summaries(temp_dir)
            self.assertEqual(len(rows), 300)
            self.assertEqual(
                {color: sum(1 for r in rows if r["target_block"] == color) for color in ("red", "blue")},
                {"red": 150, "blue": 150},
            )
            self.assertEqual([r["episode_idx"] for r in rows], list(range(300)))

    def test_missing_summary_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                collect_paired._load_summaries(temp_dir)

    def test_wrong_count_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_source(temp_dir, count=10)
            with self.assertRaises(ValueError):
                collect_paired._load_summaries(temp_dir)


class CopyOriginalEpisodeTests(unittest.TestCase):
    """原始轨迹复制：图片复制 + image_path 重写。"""

    def test_copies_images_and_rewrites_relative_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source"
            output_dir = Path(temp_dir) / "output"
            source_dir.mkdir()
            output_dir.mkdir()
            # 构造一张源图片
            image_path = source_dir / "ep_3_step_0.jpg"
            cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
            summary = _make_summary(3, "red", [0.1, 0.4, 0.1], [-0.15, 0.42, 0.1])
            trajectory_rows = [
                {
                    "schema_version": "expert_multi_v2",
                    "episode_idx": 3,
                    "step_idx": 0,
                    "image_path": str(image_path),
                    "action": [0.0] * 9,
                }
            ]
            manifest = {"schema_version": "expert_multi_v2"}
            written = collect_paired._copy_original_episode(
                source_dir, output_dir, summary, trajectory_rows, manifest
            )
            self.assertEqual(len(written), 1)
            # 与源数据一致：image_path 写绝对路径
            self.assertEqual(
                Path(written[0]["image_path"]),
                (output_dir / "ep_3_step_0.jpg").resolve(),
            )
            self.assertIs(written[0]["is_paired_copy"], False)
            self.assertTrue((output_dir / "ep_3_step_0.jpg").is_file())

    def test_relative_source_image_resolved_from_source_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source"
            output_dir = Path(temp_dir) / "output"
            source_dir.mkdir()
            output_dir.mkdir()
            image_path = source_dir / "ep_5_step_24.jpg"
            cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
            summary = _make_summary(5, "blue", [0.1, 0.4, 0.1], [-0.15, 0.42, 0.1])
            trajectory_rows = [
                {
                    "schema_version": "expert_multi_v2",
                    "episode_idx": 5,
                    "step_idx": 24,
                    "image_path": "ep_5_step_24.jpg",
                    "action": [0.0] * 9,
                }
            ]
            written = collect_paired._copy_original_episode(
                source_dir, output_dir, summary, trajectory_rows, []
            )
            self.assertEqual(
                Path(written[0]["image_path"]),
                (output_dir / "ep_5_step_24.jpg").resolve(),
            )
            self.assertTrue((output_dir / "ep_5_step_24.jpg").is_file())


class RunPairedBranchTests(unittest.TestCase):
    """补录轨迹执行：物理全程 mock，验证 seed/指令/场景契约。"""

    def _run_branch(self, temp_dir, episode_idx, target_block):
        source_dir = Path(temp_dir) / "source"
        output_dir = Path(temp_dir) / "output"
        source_dir.mkdir()
        output_dir.mkdir()
        summary = _make_summary(episode_idx, target_block, [0.1, 0.4, 0.1], [-0.15, 0.42, 0.1])
        config = _make_config(str(output_dir))
        jsonl_path = output_dir / "trajectory_expert.jsonl"
        summary_path = output_dir / "episode_summary.jsonl"
        return source_dir, output_dir, summary, config, jsonl_path, summary_path

    @patch("vla_project.simulation.collect_paired.write_episode_summary")
    @patch("vla_project.simulation.collect_paired.write_dataset_step")
    @patch("vla_project.simulation.collect_paired.should_capture")
    @patch("vla_project.simulation.collect_paired.determine_termination")
    @patch("vla_project.simulation.collect_paired.capture_scene_state")
    @patch("vla_project.simulation.collect_paired.capture_rgb")
    @patch("vla_project.simulation.collect_paired.get_link_position")
    @patch("vla_project.simulation.collect_paired.calculate_target_joints")
    @patch("vla_project.simulation.collect_paired.apply_joint_targets")
    @patch("vla_project.simulation.collect_paired.p.stepSimulation")
    @patch("vla_project.simulation.collect_paired.p.disconnect")
    @patch("vla_project.simulation.collect_paired.p.getBasePositionAndOrientation")
    @patch("vla_project.simulation.collect_paired._replay_diagnostics")
    @patch("vla_project.simulation.collect_paired._load_saved_block")
    @patch("vla_project.simulation.collect_paired._restore_robot")
    @patch("vla_project.simulation.collect_paired.setup_world")
    @patch("vla_project.simulation.collect_paired.connect_physics")
    def test_paired_branch_writes_opposite_target(
        self,
        connect_physics,
        setup_world,
        restore_robot,
        load_saved_block,
        replay_diagnostics,
        get_base_pos,
        disconnect,
        step_sim,
        apply_joint_targets,
        calculate_target_joints,
        get_link_position,
        capture_rgb,
        capture_scene_state,
        determine_termination,
        should_capture,
        write_dataset_step,
        write_episode_summary,
    ):
        connect_physics.return_value = 0
        setup_world.return_value = (1, 2)
        load_saved_block.side_effect = [100, 200]  # red, blue
        replay_diagnostics.return_value = {"passed": True}
        capture_scene_state.side_effect = lambda *a, **k: _make_scene_state(
            "red", [0.1, 0.4, 0.1], [-0.15, 0.42, 0.1]
        )
        determine_termination.return_value = "success"
        should_capture.return_value = True
        capture_rgb.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
        get_link_position.return_value = [0.1, 0.4, 0.25]
        get_base_pos.return_value = ([0.1, 0.4, 0.1], [0, 0, 0, 1])
        calculate_target_joints.return_value = [0.1] * 7

        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir, output_dir, summary, config, jsonl_path, summary_path = (
                self._run_branch(temp_dir, 3, "red")
            )
            result = collect_paired.run_paired_branch(
                source_dir=source_dir,
                output_dir=output_dir,
                episode_idx=303,
                summary=summary,
                config=config,
                jsonl_path=jsonl_path,
                summary_jsonl_path=summary_path,
            )

        self.assertEqual(result["paired_target"], "blue")
        self.assertEqual(result["paired_with"], 3)
        # 补录轨迹 seed = manifest.random_seed + episode_idx = 1000 + 303
        written_step = write_dataset_step.call_args.kwargs
        self.assertEqual(written_step["random_seed"], 1303)
        self.assertEqual(written_step["instruction"], "悬停在蓝色积木上方")
        self.assertEqual(written_step["episode_idx"], 303)
        written_summary = write_episode_summary.call_args.kwargs
        self.assertEqual(written_summary["target_block"], "blue")
        self.assertEqual(written_summary["random_seed"], 1303)
        self.assertEqual(written_summary["task_instruction"], "悬停在蓝色积木上方")


class CollectPairedIntegrationTests(unittest.TestCase):
    """主流程：复制原始 + 补录，manifest 契约与红蓝平衡。"""

    def _write_source_dataset(self, source_dir):
        """写入 300 条摘要 + 300 条轨迹（红 150 / 蓝 150）+ 图片。"""
        source_dir.mkdir(parents=True, exist_ok=True)
        summaries = []
        trajectory_rows = []
        for idx in range(300):
            target = "red" if idx < 150 else "blue"
            red_pos = [0.1, 0.4, 0.1]
            blue_pos = [-0.15, 0.42, 0.1]
            summary = _make_summary(idx, target, red_pos, blue_pos)
            summaries.append(summary)
            for step in (0, 24):
                image_path = source_dir / f"ep_{idx}_step_{step}.jpg"
                cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
                trajectory_rows.append(
                    {
                        "schema_version": "expert_multi_v2",
                        "episode_idx": idx,
                        "step_idx": step,
                        "random_seed": 1000 + idx,
                        "image_path": str(image_path),
                        "instruction": summary["task_instruction"],
                        "action": [0.0] * 9,
                        "camera_eye": [1.0, 0.4, 1.6],
                        "block_pos": red_pos if target == "red" else blue_pos,
                        "target_pos": [0.1, 0.4, 0.25],
                        "ee_pos": [0.1, 0.4, 0.25],
                        "distance_to_target": 0.029,
                        "termination_reason": "success",
                        "scene_state": summary["initial_scene_state"],
                    }
                )
        with (source_dir / "episode_summary.jsonl").open("w", encoding="utf-8") as handle:
            for summary in summaries:
                handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        with (source_dir / "trajectory_expert.jsonl").open("w", encoding="utf-8") as handle:
            for row in trajectory_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @patch("vla_project.simulation.collect_paired.run_paired_branch")
    def test_manifest_contract_and_copy(self, run_paired_branch):
        def fake_paired_branch(**kwargs):
            # mock 补录：真实写入一条补录摘要，使文件里最终有 600 条。
            paired_idx = kwargs["episode_idx"]
            summary = kwargs["summary"]
            target = collect_paired._opposite(summary["target_block"])
            paired_summary = dict(summary)
            paired_summary["episode_idx"] = paired_idx
            paired_summary["random_seed"] = 1000 + paired_idx
            paired_summary["target_block"] = target
            paired_summary["task_instruction"] = (
                "悬停在蓝色积木上方" if target == "blue" else "悬停在红色积木上方"
            )
            paired_summary["is_paired_copy"] = True
            with Path(kwargs["summary_jsonl_path"]).open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(paired_summary, ensure_ascii=False) + "\n")
            return {
                "episode_idx": paired_idx,
                "paired_with": summary["episode_idx"],
                "paired_target": target,
                "termination_reason": "success",
            }

        run_paired_branch.side_effect = fake_paired_branch
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source"
            self._write_source_dataset(source_dir)
            output_dir = Path(temp_dir) / "outputs" / "dataset" / "expert_paired_v1"
            config = _make_config(str(output_dir))
            results = collect_paired.collect_paired(
                source_dir=source_dir,
                output_dir=str(output_dir),
                config=config,
                project_root_override=temp_dir,
            )
            self.assertEqual(len(results), 300)
            manifest = json.loads(
                (output_dir / "dataset_manifest.json").read_text(encoding="utf-8")
            )
            # 成对数据集 target_num_episodes 翻倍为 600，scale gate task_balance 才能通过
            self.assertEqual(manifest["target_num_episodes"], 600)
            self.assertEqual(manifest["schema_version"], "expert_multi_v2")
            self.assertEqual(manifest["action_dim"], 9)
            # 摘要 600 条：红 300 / 蓝 300
            with (output_dir / "episode_summary.jsonl").open(encoding="utf-8") as handle:
                summaries = [
                    json.loads(line) for line in handle if line.strip()
                ]
            self.assertEqual(len(summaries), 600)
            task_counts = {
                color: sum(1 for s in summaries if s["target_block"] == color)
                for color in ("red", "blue")
            }
            self.assertEqual(task_counts, {"red": 300, "blue": 300})
            # 原始轨迹图片复制到成对目录
            self.assertTrue((output_dir / "ep_0_step_0.jpg").is_file())
            self.assertTrue((output_dir / "ep_299_step_24.jpg").is_file())

    @patch("vla_project.simulation.collect_paired.run_paired_branch")
    def test_refuses_nonempty_output(self, run_paired_branch):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source"
            self._write_source_dataset(source_dir)
            output_dir = Path(temp_dir) / "outputs" / "dataset" / "expert_paired_v1"
            output_dir.mkdir(parents=True)
            (output_dir / "evidence.txt").write_text("x", encoding="utf-8")
            config = _make_config(str(output_dir))
            with self.assertRaises(FileExistsError):
                collect_paired.collect_paired(
                    source_dir=source_dir,
                    output_dir=str(output_dir),
                    config=config,
                    project_root_override=temp_dir,
                )


if __name__ == "__main__":
    unittest.main()
