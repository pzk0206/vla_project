"""VLA Rollout: 按指令评估 VLA 模型能否区分红/蓝积木。

用法:
  python -m vla_project.training.vla_rollout \
    --checkpoint outputs/training/vla_regression_full_v1/checkpoint_best.pt \
    --output-dir outputs/rollout/vla_regression_full_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from vla_project.output_paths import resolve_new_output_directory
from PIL import Image

from vla_project.simulation.control_arm import (
    apply_joint_targets,
    capture_rgb,
    connect_physics,
    euclidean_distance,
    get_hover_target,
    get_link_position,
    get_object_position,
    load_block,
    load_config,
    load_second_block,
    p,
    reset_robot_to_home,
    sample_camera_eye,
    settle_object,
    setup_world,
)
from vla_project.training.dataset import IMAGENET_MEAN, IMAGENET_STD


def _preprocess_image(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb).resize((224, 224), Image.BILINEAR)
    image_np = np.asarray(image_pil, dtype=np.float32) / 255.0
    image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)


def _load_vla_model(checkpoint_path, device):
    from vla_project.training.vla_model import VLAModel

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = VLAModel().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("metadata", {})


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
    """VLA rollout: 对每个 val episode 用原始指令评估。"""
    output_dir = resolve_new_output_directory(
        output_dir,
        allowed_root="outputs/rollout",
        project_root_override=project_root_override,
    )

    from vla_project.training.dataset import compute_action_stats

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载
    config = load_config(config_path)
    model, metadata = _load_vla_model(checkpoint_path, device)

    # action_stats
    all_rows = []
    traj_path = Path(dataset_dir) / "trajectory_expert.jsonl"
    with open(traj_path) as f:
        for line in f:
            if line.strip():
                all_rows.append(json.loads(line))
    with open(Path(dataset_dir) / "episode_split.json") as f:
        split = json.load(f)
    train_rows = [r for r in all_rows if r["episode_idx"] in set(split["train"])]
    action_stats = compute_action_stats(train_rows)
    action_mean = np.asarray(action_stats[0], dtype=np.float32)
    action_std = np.asarray(action_stats[1], dtype=np.float32)

    # val episodes
    summaries = []
    with open(Path(dataset_dir) / "episode_summary.jsonl") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                if s["episode_idx"] in set(split["val"]):
                    summaries.append(s)

    if max_episodes:
        summaries = summaries[:max_episodes]

    # 仿真
    mode = "GUI" if gui else "DIRECT"
    connect_physics(mode)
    _, robot_id = setup_world(config)
    robot_cfg = config["robot"]
    task_cfg = config["task"]

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ep_summary in summaries:
        episode_idx = ep_summary["episode_idx"]
        instruction = ep_summary.get("task_instruction", "悬停在红色积木上方")
        block_pos = ep_summary["initial_block_pos"]
        target_block = ep_summary.get("target_block", "red")

        try:
            reset_robot_to_home(robot_id, robot_cfg, config["dataset"])
            for _ in range(task_cfg["initial_settle_steps"]):
                p.stepSimulation()

            red_block_id = load_block(task_cfg)
            blue_block_id = load_second_block(task_cfg)
            settle_object(config, task_cfg["initial_settle_steps"])

            target_block_id = blue_block_id if target_block == "blue" else red_block_id

            camera_eye = sample_camera_eye(config["camera"])

            text_emb = model.encode_texts([instruction]).to(device)

            success = False
            final_distance = None
            steps_taken = 0

            for step in range(max_steps):
                image_bgr = capture_rgb(config["camera"], camera_eye)
                image_tensor = _preprocess_image(image_bgr).to(device)

                with torch.no_grad():
                    action_pred, aux_pred = model(image_tensor, text_emb)

                pred = action_pred[0].cpu().numpy()
                joint_targets = pred * action_std + action_mean

                if aux_pred[0, 1].sigmoid().item() > 0.5:
                    break

                apply_joint_targets(robot_id, robot_cfg,
                                    joint_targets[:robot_cfg["controlled_joints"]])
                for _ in range(60):
                    p.stepSimulation()

                hover_target = get_hover_target(
                    target_block_id, task_cfg["hover_height"])
                ee_pos = get_link_position(robot_id, robot_cfg["ee_link_index"])
                distance = euclidean_distance(ee_pos, hover_target)

                if distance < task_cfg["success_distance"]:
                    success = True
                    final_distance = float(distance)
                    steps_taken = step + 1
                    break

            if not success:
                hover_target = get_hover_target(
                    target_block_id, task_cfg["hover_height"])
                ee_pos = get_link_position(robot_id, robot_cfg["ee_link_index"])
                final_distance = float(euclidean_distance(ee_pos, hover_target))
                steps_taken = max_steps

            results.append({
                "episode_idx": episode_idx,
                "success": success,
                "final_distance": final_distance,
                "steps": steps_taken,
                "instruction": instruction,
                "target_block": target_block,
            })

        except Exception as exc:
            import traceback
            results.append({
                "episode_idx": episode_idx,
                "success": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })

        finally:
            if "red_block_id" in dir():
                p.removeBody(red_block_id)
            if "blue_block_id" in dir():
                p.removeBody(blue_block_id)

    p.disconnect()

    num_success = sum(1 for r in results if r.get("success"))
    success_dists = [r["final_distance"] for r in results if "final_distance" in r and r["final_distance"] is not None]

    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset_dir": str(dataset_dir),
        "num_episodes": len(results),
        "num_success": num_success,
        "success_rate": num_success / len(results) if results else 0,
        "mean_final_distance": float(np.mean(success_dists)) if success_dists else None,
        "median_final_distance": float(np.median(success_dists)) if success_dists else None,
    }

    with open(output_dir / "per_episode.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(output_dir / "summary.json", "w") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="VLA Rollout 评估")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="sim_config.yaml")
    parser.add_argument("--dataset-dir", default="outputs/dataset/expert_multi_v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

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
