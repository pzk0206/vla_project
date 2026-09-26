"""为 expert_multi_v2 冻结场景补录反色指令专家轨迹，生成成对训练数据集。

问题背景：
- expert_multi_v2 里红指令和蓝指令出现在不同随机场景，模型可以只靠视觉布局
  挑一个积木就得分，文本指令在训练中成为噪声。
- 成对反事实评估（v4）因此只有偏好切换 40%，成对指令跟随 0/50。

本模块对 expert_multi_v2 的每条轨迹恢复其冻结场景（红蓝积木位姿、机器人状态、
相机位置），用确定性 IK 对**反色**目标积木执行同一套悬停专家，得到「同一画面、
同一机器人状态、只有指令不同」的成对轨迹。这样训练时模型不利用语言就无法同时
拟合红/蓝两条轨迹，被迫学会区分指令。

输出目录结构（expert_paired_v1）：
- 原始轨迹（episode_idx 0-299）：从源数据集复制图片并重写 JSONL。
- 补录轨迹（episode_idx 300-599）：恢复场景后确定性悬停到反色积木。
- schema_version 保持 expert_multi_v2，可复用 evaluate_dataset 的全部悬停契约校验。

CLI: python -m vla_project.simulation.collect_paired
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import yaml

from vla_project.output_paths import resolve_managed_output
from vla_project.simulation.control_arm import (
    apply_joint_targets,
    build_dataset_manifest,
    calculate_target_joints,
    capture_rgb,
    capture_scene_state,
    connect_physics,
    determine_termination,
    euclidean_distance,
    get_link_position,
    get_object_position,
    load_block_at_position,
    load_config,
    reset_robot_to_home,
    should_capture,
    setup_world,
    write_dataset_step,
    write_episode_summary,
)

CONFIG_PATH = "sim_config.yaml"
_PAIRED_OFFSET = 300          # 补录轨迹 episode_idx 从 300 起
_SCENE_REPLAY_TOLERANCE_METERS = 0.001
_SCENE_REPLAY_QUATERNION_TOLERANCE = 0.001
_INSTRUCTION_BY_TARGET = {
    "red": "悬停在红色积木上方",
    "blue": "悬停在蓝色积木上方",
}
_OPPOSITE_TARGET = {"red": "blue", "blue": "red"}


def _opposite(target_block):
    """反色目标：red <-> blue。"""
    if target_block not in _OPPOSITE_TARGET:
        raise ValueError(f"非法 target_block: {target_block!r}")
    return _OPPOSITE_TARGET[target_block]


def _finite_vector(value, length, label):
    if (
        not isinstance(value, list)
        or len(value) != length
        or not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
    ):
        raise ValueError(f"invalid saved scene {label}")
    return [float(item) for item in value]


def extract_saved_scene_spec(summary):
    """从摘要提取用于确定性重放的冻结场景参数（与 vla_rollout 同协议）。"""
    if summary.get("schema_version") != "expert_multi_v2":
        raise ValueError("saved scene requires expert_multi_v2 summary")
    scene = summary.get("initial_scene_state")
    if not isinstance(scene, dict):
        raise ValueError("invalid saved scene initial_scene_state")
    blocks = scene.get("blocks")
    robot = scene.get("robot")
    if not isinstance(blocks, dict) or not isinstance(robot, dict):
        raise ValueError("invalid saved scene blocks or robot")
    saved_blocks = {}
    for color in ("red", "blue"):
        block = blocks.get(color)
        if not isinstance(block, dict):
            raise ValueError(f"invalid saved scene {color} block")
        saved_blocks[color] = {
            "position": _finite_vector(block.get("position"), 3, f"{color} position"),
            "orientation": _finite_vector(
                block.get("orientation"), 4, f"{color} orientation"
            ),
        }
    return {
        "episode_idx": int(summary["episode_idx"]),
        "random_seed": int(summary["random_seed"]),
        "blocks": saved_blocks,
        "joint_positions": _finite_vector(
            robot.get("joint_positions"), 7, "joint positions"
        ),
        "joint_velocities": _finite_vector(
            robot.get("joint_velocities"), 7, "joint velocities"
        ),
        "camera_eye": _finite_vector(scene.get("camera_eye"), 3, "camera eye"),
    }


def _quaternion_error(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    return float(
        min(np.linalg.norm(actual - expected), np.linalg.norm(actual + expected))
    )


def _replay_diagnostics(robot_id, robot_config, block_ids, scene_spec):
    """重放后核对场景误差，确保冻结场景恢复成功。"""
    actual = capture_scene_state(
        robot_id,
        robot_config,
        block_ids,
        scene_spec["camera_eye"],
        "red",
    )
    position_errors = {}
    orientation_errors = {}
    for color in ("red", "blue"):
        position_errors[color] = euclidean_distance(
            actual["blocks"][color]["position"],
            scene_spec["blocks"][color]["position"],
        )
        orientation_errors[color] = _quaternion_error(
            actual["blocks"][color]["orientation"],
            scene_spec["blocks"][color]["orientation"],
        )
    joint_error = max(
        abs(actual_value - expected_value)
        for actual_value, expected_value in zip(
            actual["robot"]["joint_positions"], scene_spec["joint_positions"]
        )
    )
    passed = (
        max(position_errors.values()) <= _SCENE_REPLAY_TOLERANCE_METERS
        and max(orientation_errors.values()) <= _SCENE_REPLAY_QUATERNION_TOLERANCE
        and joint_error <= _SCENE_REPLAY_TOLERANCE_METERS
    )
    return {
        "passed": passed,
        "position_error_m": position_errors,
        "orientation_error": orientation_errors,
        "max_joint_position_error": float(joint_error),
        "actual_scene_state": actual,
    }


def _restore_robot(robot_id, robot_config, scene_spec):
    for joint_index, (position, velocity) in enumerate(
        zip(scene_spec["joint_positions"], scene_spec["joint_velocities"])
    ):
        p.resetJointState(
            robot_id,
            joint_index,
            targetValue=position,
            targetVelocity=velocity,
        )
    apply_joint_targets(robot_id, robot_config, scene_spec["joint_positions"])


def _load_saved_block(task_config, block_state, color_rgba, global_scaling):
    block_id = load_block_at_position(
        task_config,
        block_state["position"],
        color_rgba,
        global_scaling,
    )
    p.resetBasePositionAndOrientation(
        block_id,
        block_state["position"],
        block_state["orientation"],
    )
    p.resetBaseVelocity(block_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    return block_id


def _sha256(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_summaries(dataset_dir):
    """读取源数据集 episode 摘要，严格校验 expert_multi_v2 契约。"""
    summary_path = Path(dataset_dir) / "episode_summary.jsonl"
    if not summary_path.is_file():
        raise FileNotFoundError(f"找不到 episode 摘要: {summary_path}")
    rows = []
    with summary_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 300:
        raise ValueError(f"预期 300 条 episode 摘要，实际 {len(rows)} 条")
    indices = {row["episode_idx"] for row in rows}
    if indices != set(range(300)):
        raise ValueError("expert_multi_v2 必须包含 episode_idx 0-299")
    task_counts = {color: sum(1 for r in rows if r["target_block"] == color) for color in ("red", "blue")}
    if task_counts != {"red": 150, "blue": 150}:
        raise ValueError(f"expert_multi_v2 必须红蓝各 150，实际 {task_counts}")
    rows.sort(key=lambda r: r["episode_idx"])
    return rows


def _copy_original_episode(
    source_dir,
    output_dir,
    summary,
    trajectory_rows,
    manifest,
):
    """复制原始轨迹的图片并重写 JSONL 行到成对目录。

    图片复制为 ep_{idx}_step_{step}.jpg，JSONL 的 image_path 重写为相对路径
    指向成对目录内的同名字段（由 evaluate_dataset 相对解析）。
    """
    episode_idx = summary["episode_idx"]
    written = []
    for row in trajectory_rows:
        source_image = Path(row["image_path"])
        if not source_image.is_file():
            source_image = Path(source_dir) / source_image
        target_name = source_image.name
        shutil.copy2(str(source_image), str(output_dir / target_name))
        new_row = dict(row)
        # 与源数据集一致：image_path 写绝对路径，训练直接可读。
        new_row["image_path"] = str(output_dir / target_name)
        new_row["is_paired_copy"] = False
        written.append(new_row)
    return written


def run_paired_branch(
    *,
    source_dir,
    output_dir,
    episode_idx,
    summary,
    config,
    jsonl_path,
    summary_jsonl_path,
):
    """在冻结场景上对反色目标执行确定性悬停专家，写入补录轨迹。

    补录轨迹的 scene_state.target_block 为反色，红蓝积木位姿与机器人状态
    与原始轨迹完全一致（重放误差校验）。seed 沿用原始（保持 scene 契约）。
    """
    scene_spec = extract_saved_scene_spec(summary)
    original_target = summary["target_block"]
    paired_target = _opposite(original_target)
    instruction = _INSTRUCTION_BY_TARGET[paired_target]

    connect_physics("DIRECT")
    block_ids = None
    try:
        _, robot_id = setup_world(config)
        robot_config = config["robot"]
        task_config = config["task"]
        _restore_robot(robot_id, robot_config, scene_spec)
        scale = float(task_config["block_global_scaling"])
        red_block_id = _load_saved_block(
            task_config,
            scene_spec["blocks"]["red"],
            task_config["block_color_rgba"],
            scale,
        )
        second = task_config["second_block"]
        blue_block_id = _load_saved_block(
            task_config,
            scene_spec["blocks"]["blue"],
            second["color_rgba"],
            float(second.get("global_scaling", scale)),
        )
        block_ids = {"red": red_block_id, "blue": blue_block_id}
        replay = _replay_diagnostics(
            robot_id, robot_config, block_ids, scene_spec
        )
        if not replay["passed"]:
            raise RuntimeError(
                f"episode {episode_idx} 场景重放失败: {replay}"
            )

        hover_height = float(task_config["hover_height"])
        max_steps = int(config["dataset"]["max_steps_per_episode"])
        success_distance = float(task_config["success_distance"])
        capture_interval = int(config["dataset"]["capture_interval_steps"])
        stuck_window = int(task_config["stuck_window_steps"])
        stuck_min_improvement = float(task_config["stuck_min_improvement"])
        force_terminal = int(task_config["force_terminal_after_step"])

        # 补录轨迹的 seed 遵循 manifest 契约：random_seed + episode_idx，
        # 与 evaluate_dataset 的 expected_seed 校验保持一致。
        paired_random_seed = int(config["dataset"]["random_seed"]) + episode_idx

        recent_distances = []
        saved_frames = 0
        final_distance = None
        final_block_pos = None
        final_target_pos = None
        final_ee_pos = None
        termination_reason = "running"
        camera_eye = list(scene_spec["camera_eye"])
        initial_ee_pos = list(
            get_link_position(robot_id, robot_config["ee_link_index"])
        )
        current_scene_state = capture_scene_state(
            robot_id,
            robot_config,
            block_ids,
            camera_eye,
            paired_target,
        )
        initial_scene_state = current_scene_state
        initial_block_pos = list(
            current_scene_state["blocks"][paired_target]["position"]
        )

        for step_idx in range(max_steps):
            block_pos = list(
                get_object_position(block_ids[paired_target])
            )
            target_pos = list(block_pos)
            target_pos[2] += hover_height
            target_joint_angles = calculate_target_joints(
                robot_id, robot_config, target_pos
            )
            apply_joint_targets(robot_id, robot_config, target_joint_angles)
            p.stepSimulation()
            if config.get("enable_time_sleep"):
                import time

                time.sleep(1.0 / config["simulation_hz"])

            current_scene_state = capture_scene_state(
                robot_id,
                robot_config,
                block_ids,
                camera_eye,
                paired_target,
            )
            block_pos = list(
                current_scene_state["blocks"][paired_target]["position"]
            )
            target_pos = list(block_pos)
            target_pos[2] += hover_height
            ee_pos = list(current_scene_state["robot"]["ee_position"])
            distance_to_target = euclidean_distance(ee_pos, target_pos)
            final_distance = distance_to_target
            final_block_pos = block_pos
            final_target_pos = target_pos
            final_ee_pos = ee_pos

            recent_distances.append(distance_to_target)
            if len(recent_distances) > stuck_window:
                recent_distances.pop(0)

            termination_reason = determine_termination(
                distance_to_target=distance_to_target,
                recent_distances=recent_distances,
                step_idx=step_idx,
                success_distance=success_distance,
                stuck_window_steps=stuck_window,
                stuck_min_improvement=stuck_min_improvement,
                force_terminal_after_step=force_terminal,
            )
            terminate_episode = int(termination_reason != "running")

            if should_capture(step_idx, capture_interval, terminate_episode):
                image_name = f"ep_{episode_idx}_step_{step_idx}.jpg"
                image_path = str(output_dir / image_name)
                cv2.imwrite(image_path, capture_rgb(config["camera"], camera_eye))
                saved_frames += 1
                action = (
                    list(target_joint_angles[: robot_config["controlled_joints"]])
                    + [1.0, terminate_episode]
                )
                write_dataset_step(
                    jsonl_path=jsonl_path,
                    schema_version=config["dataset"]["schema_version"],
                    episode_idx=episode_idx,
                    step_idx=step_idx,
                    random_seed=paired_random_seed,
                    image_path=image_path,
                    instruction=instruction,
                    action=action,
                    camera_eye=camera_eye,
                    block_pos=block_pos,
                    target_pos=target_pos,
                    ee_pos=ee_pos,
                    distance_to_target=distance_to_target,
                    termination_reason=termination_reason,
                    scene_state=current_scene_state,
                )

            if terminate_episode:
                break

        write_episode_summary(
            summary_jsonl_path=summary_jsonl_path,
            schema_version=config["dataset"]["schema_version"],
            episode_idx=episode_idx,
            random_seed=paired_random_seed,
            initial_ee_pos=initial_ee_pos,
            initial_block_pos=initial_block_pos,
            num_steps=step_idx + 1,
            num_frames=saved_frames,
            final_distance=final_distance,
            termination_reason=termination_reason,
            camera_eye=camera_eye,
            final_block_pos=final_block_pos,
            final_target_pos=final_target_pos,
            final_ee_pos=final_ee_pos,
            task_instruction=instruction,
            target_block=paired_target,
            initial_scene_state=initial_scene_state,
            final_scene_state=current_scene_state,
            extra_fields={
                "paired_with": scene_spec["episode_idx"],
                "is_paired_copy": True,
            },
        )
        return {
            "episode_idx": episode_idx,
            "paired_with": scene_spec["episode_idx"],
            "original_target": original_target,
            "paired_target": paired_target,
            "termination_reason": termination_reason,
            "frames": saved_frames,
        }
    finally:
        if block_ids is not None:
            for block_id in block_ids.values():
                if block_id is not None:
                    try:
                        p.removeBody(block_id)
                    except Exception:
                        pass
        p.disconnect()


def collect_paired(
    source_dir,
    output_dir,
    config,
    max_episodes=None,
    project_root_override=None,
):
    """主流程：复制原始 300 条 + 补录 300 条反色轨迹，写入成对数据集。"""
    source_dir = Path(source_dir)
    output_dir = Path(
        resolve_managed_output(
            output_dir,
            allowed_root="outputs/dataset",
            project_root_override=project_root_override,
        )
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"成对数据集目录非空，拒绝覆盖: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = _load_summaries(source_dir)
    if max_episodes is not None:
        summaries = summaries[:max_episodes]

    # manifest 与 config 快照
    config = dict(config)
    dataset_config = dict(config["dataset"])
    dataset_config["output_dir"] = str(output_dir)
    dataset_config["schema_version"] = "expert_multi_v2"
    # 成对数据集 = 原始 + 补录，目标规模翻倍；scale gate 的 task_balance
    # 用 target_num_episodes//2 期望红蓝各半，因此必须等于 2×原始规模。
    dataset_config["target_num_episodes"] = len(summaries) * 2
    config["dataset"] = dataset_config
    manifest = build_dataset_manifest(config)
    manifest["paired_source_dir"] = str(source_dir)
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    snapshot_path = output_dir / "config_snapshot.yaml"
    snapshot_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    jsonl_path = output_dir / dataset_config["jsonl_name"]
    summary_jsonl_path = output_dir / dataset_config["summary_jsonl_name"]

    # 读源轨迹并按 episode 分组
    source_trajectory = Path(source_dir) / dataset_config["jsonl_name"]
    source_rows = []
    with source_trajectory.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                source_rows.append(json.loads(line))
    rows_by_episode = {}
    for row in source_rows:
        rows_by_episode.setdefault(row["episode_idx"], []).append(row)

    results = []
    for offset, summary in enumerate(summaries):
        original_idx = summary["episode_idx"]
        # 1. 复制原始轨迹
        trajectory_rows = rows_by_episode.get(original_idx, [])
        if not trajectory_rows:
            raise ValueError(f"源轨迹缺少 episode {original_idx}")
        written = _copy_original_episode(
            source_dir,
            output_dir,
            summary,
            trajectory_rows,
            manifest,
        )
        with jsonl_path.open("a", encoding="utf-8") as jsonl_handle:
            for row in written:
                jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with summary_jsonl_path.open("a", encoding="utf-8") as summary_handle:
            copied_summary = dict(summary)
            copied_summary["is_paired_copy"] = False
            summary_handle.write(
                json.dumps(copied_summary, ensure_ascii=False) + "\n"
            )

        # 2. 补录反色轨迹
        paired_idx = _PAIRED_OFFSET + original_idx
        result = run_paired_branch(
            source_dir=source_dir,
            output_dir=output_dir,
            episode_idx=paired_idx,
            summary=summary,
            config=config,
            jsonl_path=jsonl_path,
            summary_jsonl_path=summary_jsonl_path,
        )
        results.append(result)

    paired_dir = str(output_dir)
    print(
        f"成对数据采集完成：原始 {len(summaries)} 条 + 补录 {len(results)} 条"
        f" -> {paired_dir}"
    )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="为 expert_multi_v2 补录反色指令专家轨迹，生成成对训练数据"
    )
    parser.add_argument("--source-dir", default="outputs/dataset/expert_multi_v2")
    parser.add_argument("--output-dir", default="outputs/dataset/expert_paired_v1")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--config-path",
        default=CONFIG_PATH,
        help="补录专家执行使用的 sim_config 路径（schema 仍写 expert_multi_v2）",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config_path)
        collect_paired(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            config=config,
            max_episodes=args.max_episodes,
        )
    except Exception as exc:
        print(f"成对数据采集失败: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
