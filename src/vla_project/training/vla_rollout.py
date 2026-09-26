"""按保存的 expert_multi_v2 场景确定性执行 VLA rollout。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from vla_project.output_paths import resolve_new_output_directory
from vla_project.simulation.control_arm import (
    apply_joint_targets,
    capture_rgb,
    capture_scene_state,
    connect_physics,
    euclidean_distance,
    get_hover_target,
    get_link_position,
    load_block_at_position,
    load_config,
    p,
    setup_world,
)
from vla_project.training.dataset import IMAGENET_MEAN, IMAGENET_STD


_DATASET_FILES = {
    "dataset_manifest": "dataset_manifest.json",
    "trajectory_expert": "trajectory_expert.jsonl",
    "episode_summary": "episode_summary.jsonl",
    "episode_split": "episode_split.json",
}
_INSTRUCTION_BY_TARGET = {
    "red": "悬停在红色积木上方",
    "blue": "悬停在蓝色积木上方",
}
_SCENE_REPLAY_TOLERANCE_METERS = 0.001
_SCENE_REPLAY_QUATERNION_TOLERANCE = 0.001


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite_vector(value, length, label):
    if (
        not isinstance(value, list)
        or len(value) != length
        or not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
    ):
        raise ValueError(f"invalid saved scene {label}")
    return [float(item) for item in value]


def validate_checkpoint_dataset_binding(metadata, dataset_dir):
    """在构造模型或连接仿真前核对 checkpoint 与数据证据。"""
    dataset_dir = Path(dataset_dir)
    if metadata.get("dataset_schema_version") != "expert_multi_v2":
        raise ValueError("checkpoint is not bound to expert_multi_v2")
    if metadata.get("split_schema_version") not in {
        "episode_split_v2",
        "episode_split_v3",
    }:
        raise ValueError(
            "checkpoint is not bound to a supported episode split "
            "(v2 or paired v3)"
        )
    for key, file_name in _DATASET_FILES.items():
        path = dataset_dir / file_name
        if not path.is_file():
            raise FileNotFoundError(f"missing rollout dataset input: {path}")
        expected = metadata.get(f"{key}_sha256")
        actual = _sha256(path)
        if expected != actual:
            raise ValueError(f"{key} SHA-256 mismatch")
    action_stats = metadata.get("action_stats")
    if not isinstance(action_stats, dict):
        raise ValueError("checkpoint missing action_stats")
    mean = np.asarray(action_stats.get("mean"), dtype=np.float32)
    std = np.asarray(action_stats.get("std"), dtype=np.float32)
    if (
        mean.shape != (7,)
        or std.shape != (7,)
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(std))
        or np.any(std <= 0)
    ):
        raise ValueError("checkpoint action_stats are invalid")
    return mean, std


def extract_saved_scene_spec(summary):
    """从摘要提取唯一允许用于 rollout 的保存场景参数。"""
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


def _preprocess_image(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb).resize((224, 224), Image.BILINEAR)
    image_np = np.asarray(image_pil, dtype=np.float32) / 255.0
    image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)


def load_bound_vla_model(checkpoint_path, dataset_dir, device):
    """先读 checkpoint 元数据并核对输入，再构造有外部依赖的模型。"""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    action_mean, action_std = validate_checkpoint_dataset_binding(
        metadata, dataset_dir
    )
    from vla_project.training.vla_model import VLAModel

    model = VLAModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, metadata, action_mean, action_std


def _load_vla_model(*_args, **_kwargs):
    """保留旧测试/调用边界；禁止绕过 v2 数据哈希绑定加载模型。"""
    raise RuntimeError("use load_bound_vla_model with an audited dataset_dir")


def load_validation_summaries(dataset_dir, max_episodes=None):
    dataset_dir = Path(dataset_dir)
    split_doc = json.loads((dataset_dir / "episode_split.json").read_text())
    strat = split_doc.get("stratification", {})
    paired = strat.get("paired")
    if paired is not None:
        # 成对数据：反事实以 50 个 val 场景为单位（原始轨迹），每个场景
        # 再分别执行红/蓝指令分支。补录伴侣不重复评估，避免场景重复。
        val_ids = paired.get("scene_val")
        if not isinstance(val_ids, list) or len(val_ids) != 50:
            raise ValueError(
                "paired rollout requires exactly 50 val scenes"
            )
    else:
        val_ids = split_doc.get("val")
        if not isinstance(val_ids, list) or len(val_ids) != 50:
            raise ValueError("deterministic rollout requires exactly 50 val episodes")
    by_id = {}
    with (dataset_dir / "episode_summary.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                by_id[row["episode_idx"]] = row
    summaries = [by_id[index] for index in val_ids]
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        summaries = summaries[:max_episodes]
    return summaries


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


def _quaternion_error(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    return float(min(np.linalg.norm(actual - expected), np.linalg.norm(actual + expected)))


def _replay_diagnostics(robot_id, robot_config, block_ids, scene_spec):
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


def _distances_to_blocks(robot_id, robot_config, block_ids, hover_height):
    ee_pos = get_link_position(robot_id, robot_config["ee_link_index"])
    distances = {
        color: float(
            euclidean_distance(
                ee_pos,
                get_hover_target(block_ids[color], hover_height),
            )
        )
        for color in ("red", "blue")
    }
    return [float(value) for value in ee_pos], distances


def run_saved_scene_branch(
    *,
    model,
    device,
    action_mean,
    action_std,
    config,
    episode_summary,
    instruction_target,
    max_steps=200,
    sim_steps_per_action=60,
    gui=False,
):
    """在一个独立物理连接中运行单条颜色指令分支。"""
    if instruction_target not in _INSTRUCTION_BY_TARGET:
        raise ValueError(f"invalid instruction target: {instruction_target}")
    scene_spec = extract_saved_scene_spec(episode_summary)
    instruction = _INSTRUCTION_BY_TARGET[instruction_target]
    result = {
        "episode_idx": scene_spec["episode_idx"],
        "random_seed": scene_spec["random_seed"],
        "instruction": instruction,
        "instruction_target": instruction_target,
        "success": False,
        "system_error": None,
        "termination_reason": None,
        "steps": 0,
        "trace": [],
    }
    red_block_id = None
    blue_block_id = None
    connected = False
    try:
        connect_physics("GUI" if gui else "DIRECT")
        connected = True
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
        result["scene_replay"] = replay
        if not replay["passed"]:
            result["system_error"] = "scene_replay_mismatch"
            result["termination_reason"] = "scene_replay_mismatch"
            return result

        text_embedding = model.encode_texts([instruction]).to(device)
        hover_height = float(task_config["hover_height"])
        success_distance = float(task_config["success_distance"])
        termination_reason = "max_steps"
        actions_executed = 0
        for step_idx in range(max_steps):
            image = capture_rgb(config["camera"], scene_spec["camera_eye"])
            image_tensor = _preprocess_image(image).to(device)
            with torch.no_grad():
                action_pred, aux_pred, _color_logits = model(
                    image_tensor, text_embedding
                )
            normalized_action = action_pred[0].detach().cpu().numpy()
            joint_targets = normalized_action * action_std + action_mean
            terminate_score = float(torch.sigmoid(aux_pred[0, 1]).item())
            trace_row = {
                "step_idx": step_idx,
                "normalized_action": normalized_action.astype(float).tolist(),
                "joint_targets": joint_targets.astype(float).tolist(),
                "terminate_score": terminate_score,
                "action_executed": False,
            }
            if terminate_score > 0.5:
                ee_pos, distances = _distances_to_blocks(
                    robot_id, robot_config, block_ids, hover_height
                )
                trace_row.update({"ee_position": ee_pos, "distances": distances})
                result["trace"].append(trace_row)
                termination_reason = "model_terminate"
                break

            apply_joint_targets(
                robot_id,
                robot_config,
                joint_targets[: robot_config["controlled_joints"]],
            )
            for _ in range(sim_steps_per_action):
                p.stepSimulation()
            actions_executed += 1
            trace_row["action_executed"] = True
            ee_pos, distances = _distances_to_blocks(
                robot_id, robot_config, block_ids, hover_height
            )
            trace_row.update({"ee_position": ee_pos, "distances": distances})
            result["trace"].append(trace_row)
            if distances[instruction_target] < success_distance:
                result["success"] = True
                termination_reason = "success"
                break

        final_ee, final_distances = _distances_to_blocks(
            robot_id, robot_config, block_ids, hover_height
        )
        result.update(
            {
                "termination_reason": termination_reason,
                "steps": actions_executed,
                "final_ee_position": final_ee,
                "final_distances": final_distances,
                "final_distance": final_distances[instruction_target],
                "nearest_block": min(final_distances, key=final_distances.get),
            }
        )
        return result
    except Exception as exc:
        result["system_error"] = "rollout_exception"
        result["termination_reason"] = "rollout_exception"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        return result
    finally:
        if connected:
            for block_id in (red_block_id, blue_block_id):
                if block_id is not None:
                    try:
                        p.removeBody(block_id)
                    except Exception:
                        pass
            p.disconnect()


def run_vla_rollout(
    checkpoint_path,
    config_path,
    dataset_dir,
    output_dir,
    max_episodes=None,
    max_steps=200,
    gui=False,
    project_root_override=None,
):
    """兼容入口：对每个 val episode 的原始目标执行确定性 rollout。"""
    output_dir = resolve_new_output_directory(
        output_dir,
        allowed_root="outputs/rollout",
        project_root_override=project_root_override,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata, action_mean, action_std = load_bound_vla_model(
        checkpoint_path, dataset_dir, device
    )
    config = load_config(config_path)
    summaries = load_validation_summaries(dataset_dir, max_episodes)
    records = [
        run_saved_scene_branch(
            model=model,
            device=device,
            action_mean=action_mean,
            action_std=action_std,
            config=config,
            episode_summary=summary,
            instruction_target=summary["target_block"],
            max_steps=max_steps,
            gui=gui,
        )
        for summary in summaries
    ]
    valid = [record for record in records if record["system_error"] is None]
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dataset_dir": str(dataset_dir),
        "dataset_manifest_sha256": metadata["dataset_manifest_sha256"],
        "episode_split_sha256": metadata["episode_split_sha256"],
        "num_episodes": len(records),
        "valid_episode_count": len(valid),
        "system_error_count": len(records) - len(valid),
        "num_success": sum(record["success"] for record in valid),
        "success_rate": (
            sum(record["success"] for record in valid) / len(valid) if valid else None
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_episode.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="确定性 VLA Rollout 评估")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="sim_config.yaml")
    parser.add_argument("--dataset-dir", default="outputs/dataset/expert_multi_v2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args(argv)
    run_vla_rollout(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
        gui=args.gui,
    )


if __name__ == "__main__":
    main()
