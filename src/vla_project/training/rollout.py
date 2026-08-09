"""BC 模型闭环 rollout 评估模块。

对训练好的 BCModel checkpoint 做闭环仿真评估：
- 每次决策前拍摄一张相机照片，前向得到关节动作。
- 按回归（反归一化 q = pred * std + mean）或分类（tokenizer 解码）把预测转成关节角。
- 下发给 KUKA 电机控制器，推进 sim_steps_per_action 个物理仿真步。
- 用末端执行器与悬停目标的三维欧氏距离判定 success（默认 < 0.03m）。

用法:
    python -m vla_project.training.rollout \\
        --checkpoint outputs/training/xxx/checkpoint_best.pt \\
        --output-dir /tmp/rollout_smoke \\
        --max-episodes 1 --max-steps 3
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import torch
import yaml

from vla_project.output_paths import resolve_new_output_directory
from PIL import Image

from vla_project.simulation.control_arm import (
    apply_joint_targets,
    capture_rgb,
    connect_physics,
    euclidean_distance,
    get_hover_target,
    get_link_position,
    load_block,
    load_config,
    reset_robot_to_home,
    sample_camera_eye,
    settle_object,
    setup_world,
)
from vla_project.training.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    compute_action_stats,
)
from vla_project.training.model import BCModel
from vla_project.training.tokenizer_utils import (
    decode_tokens_to_action,
    load_quantile_tokenizer,
)

_DEFAULT_DATASET = "outputs/dataset/expert_scaling_v1"
_DEFAULT_CONFIG = "sim_config.yaml"

# 物理世界生命周期缓存：一次评估进程只连接并 setup 一次，跨 episode 复用。
_world_setup = False
_robot_id = None


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _validate_checkpoint_tokenizer(tokenizer, representation, num_bins):
    if (
        not isinstance(tokenizer, dict)
        or tokenizer.get("representation") != representation
        or tokenizer.get("binning") != "quantile"
        or tokenizer.get("requested_num_bins") != num_bins
    ):
        raise ValueError("invalid delta checkpoint tokenizer")
    edges_list = tokenizer.get("edges")
    reconstruction_values = tokenizer.get("reconstruction_values")
    if (
        not isinstance(edges_list, list)
        or len(edges_list) != 7
        or not isinstance(reconstruction_values, list)
        or len(reconstruction_values) != 7
    ):
        raise ValueError("invalid delta checkpoint tokenizer")
    for edges, reconstruction in zip(edges_list, reconstruction_values):
        edge_array = np.asarray(edges, dtype=np.float64)
        reconstruction_array = np.asarray(
            reconstruction,
            dtype=np.float64,
        )
        if (
            edge_array.shape != (num_bins + 1,)
            or reconstruction_array.shape != (num_bins,)
            or not np.isfinite(edge_array).all()
            or not np.isfinite(reconstruction_array).all()
            or not np.all(np.diff(edge_array) > 0)
        ):
            raise ValueError("invalid delta checkpoint tokenizer")
    return tokenizer


def _load_checkpoint_action_assets(metadata, action_representation):
    """验证并返回 checkpoint 内冻结的分类动作资产。"""
    if action_representation == "regression":
        return None
    if action_representation == "delta_q_64":
        if metadata.get("action_semantics") != (
            "same_episode_saved_target_delta_v2"
        ):
            raise ValueError(
                "invalid_reason=absolute_labels_encoded_as_delta"
            )
        if not _valid_sha256(metadata.get("training_split_sha256")):
            raise ValueError("invalid delta checkpoint training split hash")
        return _validate_checkpoint_tokenizer(
            metadata.get("tokenizer"),
            "delta_q",
            64,
        )
    tokenizer = metadata.get("tokenizer")
    if tokenizer is None:
        return None
    return _validate_checkpoint_tokenizer(tokenizer, "absolute_q", 32)


def _classification_joint_targets(
    action_representation,
    token_ids,
    tokenizer,
    current_q=None,
):
    decoded = np.asarray(
        decode_tokens_to_action(
            token_ids,
            tokenizer["reconstruction_values"],
        ),
        dtype=np.float64,
    )
    if action_representation != "delta_q_64":
        return decoded
    current = np.asarray(current_q, dtype=np.float64)
    if current.shape != decoded.shape or not np.isfinite(current).all():
        raise ValueError("delta rollout requires current joint state")
    return current + decoded


def _ensure_physics_world(config, gui):
    """确保仿真世界已连接并初始化（幂等）。"""
    global _world_setup, _robot_id
    if _world_setup:
        return
    connection_mode = "GUI" if gui else "DIRECT"
    connect_physics(connection_mode)
    _, _robot_id = setup_world(config)
    _world_setup = True
    print(f"🌍 [WORLD] 已连接 {connection_mode} 模式仿真世界。")


def _teardown_physics_world():
    """断开仿真世界，允许下一次评估重新连接。"""
    global _world_setup
    if _world_setup:
        p.disconnect()
        _world_setup = False
        print("🌍 [WORLD] 仿真世界已断开。")


def _preprocess_image(image_bgr):
    """BGR → RGB → PIL resize(224) → [0,1] → ImageNet 归一化 → (1,3,224,224)。

    与 dataset.py 的 BCDataset.__getitem__ 归一化口径保持一致。
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb).resize((224, 224), Image.BILINEAR)
    image_np = np.asarray(image_pil, dtype=np.float32) / 255.0
    image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)


def load_model_for_rollout(checkpoint_path, device):
    """加载 BC checkpoint，返回 (model, metadata_dict)。

    checkpoint 结构（与 train.py _save_checkpoint 对应）：
        epoch / model_state_dict / optimizer_state_dict /
        action_representation / metadata

    metadata 额外注入 checkpoint 同级 config.yaml 里的 dataset_dir 与
    overfit_episodes，供评估阶段按相同口径重算 action_stats。
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

    ckpt = torch.load(
        str(checkpoint_path), map_location=device, weights_only=False
    )
    action_representation = ckpt["action_representation"]
    metadata = dict(ckpt.get("metadata") or {})
    metadata["action_representation"] = action_representation
    _load_checkpoint_action_assets(metadata, action_representation)

    model = BCModel(action_representation=action_representation)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # 只读同级 config.yaml，注入训练口径参数（不影响其他文件）。
    config_yaml = checkpoint_path.parent / "config.yaml"
    if config_yaml.is_file():
        cfg = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))
        metadata["dataset_dir"] = cfg.get("dataset_dir", _DEFAULT_DATASET)
        metadata["overfit_episodes"] = cfg.get("overfit_episodes")
    else:
        metadata["dataset_dir"] = _DEFAULT_DATASET
        metadata["overfit_episodes"] = None

    return model, metadata


def _load_all_trajectory_rows(dataset_dir):
    """加载数据集全部轨迹帧（用于按训练口径重算 action stats）。"""
    rows = []
    with open(
        Path(dataset_dir) / "trajectory_expert.jsonl", encoding="utf-8"
    ) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _compute_action_stats_for_checkpoint(dataset_dir, overfit_episodes):
    """按 train.py 相同口径重算回归 action_stats。

    若训练为 overfit 模式（checkpoint 同级 config.yaml 记录），
    只取前 overfit_episodes 个训练 episode 统计，否则用全部训练 episode。
    """
    all_rows = _load_all_trajectory_rows(dataset_dir)
    split_path = Path(dataset_dir) / "episode_split.json"
    split_doc = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = list(split_doc["train"])
    if overfit_episodes:
        train_ids = train_ids[:overfit_episodes]
    train_set = set(train_ids)
    train_rows = [r for r in all_rows if r["episode_idx"] in train_set]
    return compute_action_stats(train_rows)


def _load_tokenizer_for_checkpoint(dataset_dir, action_representation):
    """分类分支：从 audit_v2 加载 quantile edges；regression 返回 None。"""
    if action_representation == "regression":
        return None
    audit_path = (
        Path(dataset_dir)
        / "action_tokenization_audit_v2"
        / "action_tokenization_audit.json"
    )
    if not audit_path.is_file():
        raise FileNotFoundError(f"找不到 tokenizer 审计报告: {audit_path}")
    use_delta = action_representation == "delta_q_64"
    rep = "delta_q" if use_delta else "absolute_q"
    num_bins = 64 if use_delta else 32
    return load_quantile_tokenizer(str(audit_path), rep, num_bins)


def run_rollout_episode(
    model,
    config,
    block_pos,
    device,
    action_representation,
    action_stats,
    action_tokenizer,
    max_steps=200,
    sim_steps_per_action=60,
    gui=False,
):
    """单 episode 闭环 rollout。

    流程：reset → loop{photo→predict→decode→act→check_distance}。

    返回 dict（至少含）：
        success / final_distance / steps / termination_reason /
        final_ee_pos / final_target_pos / block_pos / camera_eye
    """
    _ensure_physics_world(config, gui)

    robot_cfg = config["robot"]
    task_cfg = config["task"]
    dataset_cfg = config["dataset"]
    camera_cfg = config["camera"]

    if action_representation == "regression":
        action_mean = np.asarray(action_stats[0], dtype=np.float32)
        action_std = np.asarray(action_stats[1], dtype=np.float32)
    else:
        action_mean = action_std = None

    success_distance = task_cfg.get("success_distance", 0.03)

    # reset → home → 初始稳定 → 放积木 → 积木稳定 → 相机取景
    reset_robot_to_home(_robot_id, robot_cfg, dataset_cfg)
    for _ in range(task_cfg["initial_settle_steps"]):
        p.stepSimulation()

    block_id = load_block(task_cfg)
    p.resetBasePositionAndOrientation(
        block_id, list(block_pos), [0.0, 0.0, 0.0, 1.0]
    )
    settle_object(config, task_cfg["initial_settle_steps"])

    camera_eye = sample_camera_eye(camera_cfg)

    final_distance = None
    final_ee_pos = None
    final_target_pos = None
    termination_reason = "max_steps"
    steps = 0

    try:
        for step_idx in range(max_steps):
            steps = step_idx + 1

            # 阶段 A：拍照（capture_rgb 返回 BGR）
            image_bgr = capture_rgb(camera_cfg, camera_eye)
            image_tensor = _preprocess_image(image_bgr).to(device)

            # 阶段 B：模型前向。BCModel.forward 返回 (action_pred, aux_pred)，
            # 其中 aux_pred 第二维为 terminate 头（logits）。
            with torch.no_grad():
                action_pred, aux_pred = model(image_tensor)

            # 阶段 C：解码为 7 维关节目标角。
            if action_representation == "regression":
                pred = action_pred[0].cpu().numpy()
                joint_targets = pred * action_std + action_mean
            else:
                token_ids = action_pred.argmax(dim=-1)[0].cpu().numpy()
                current_q = None
                if action_representation == "delta_q_64":
                    current_q = np.array(
                        [
                            p.getJointState(_robot_id, j)[0]
                            for j in range(robot_cfg["controlled_joints"])
                        ],
                        dtype=np.float32,
                    )
                joint_targets = _classification_joint_targets(
                    action_representation,
                    token_ids,
                    action_tokenizer,
                    current_q=current_q,
                )

            early_terminate = aux_pred[0, 1].sigmoid().item() > 0.5

            # 阶段 D：执行动作（POSITION_CONTROL 下发给前 7 个关节）。
            apply_joint_targets(
                _robot_id,
                robot_cfg,
                joint_targets[: robot_cfg["controlled_joints"]],
            )
            for _ in range(sim_steps_per_action):
                p.stepSimulation()
                if config["enable_time_sleep"]:
                    time.sleep(1.0 / config["simulation_hz"])

            # 阶段 E：读取末端位置，计算离当前悬停目标的三维距离。
            target_pos = get_hover_target(block_id, task_cfg["hover_height"])
            ee_pos = get_link_position(_robot_id, robot_cfg["ee_link_index"])
            distance = euclidean_distance(ee_pos, target_pos)

            final_distance = distance
            final_ee_pos = list(ee_pos)
            final_target_pos = list(target_pos)

            if distance < success_distance:
                termination_reason = "success"
                break
            if early_terminate:
                termination_reason = "early_terminate"
                break
    finally:
        p.removeBody(block_id)

    success = termination_reason == "success"
    return {
        "block_pos": list(block_pos),
        "success": success,
        "final_distance": final_distance,
        "steps": steps,
        "termination_reason": termination_reason,
        "final_ee_pos": final_ee_pos,
        "final_target_pos": final_target_pos,
        "camera_eye": list(camera_eye),
    }


def run_rollout_evaluation(
    checkpoint_path,
    config_path=_DEFAULT_CONFIG,
    dataset_dir=_DEFAULT_DATASET,
    output_dir=None,
    max_episodes=None,
    max_steps=200,
    gui=False,
    project_root_override=None,
):
    """迭代 val episodes 做闭环评估，保存 per_episode.jsonl + summary.json。"""
    if output_dir is None:
        checkpoint = Path(checkpoint_path)
        tag = checkpoint.parent.name or checkpoint.stem
        output_dir = Path("outputs/rollout") / tag
    output_dir = resolve_new_output_directory(
        output_dir,
        allowed_root="outputs/rollout",
        project_root_override=project_root_override,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model, metadata = load_model_for_rollout(checkpoint_path, device)
    action_representation = metadata["action_representation"]
    overfit_episodes = metadata.get("overfit_episodes")
    print(
        f"模型: action_representation={action_representation}, "
        f"overfit_episodes={overfit_episodes}, checkpoint={checkpoint_path}"
    )

    # 与训练口径一致的预处理参数。
    action_stats = None
    action_tokenizer = _load_checkpoint_action_assets(
        metadata,
        action_representation,
    )
    if action_representation == "regression":
        action_stats = _compute_action_stats_for_checkpoint(
            dataset_dir, overfit_episodes
        )
    elif action_tokenizer is None:
        action_tokenizer = _load_tokenizer_for_checkpoint(
            dataset_dir,
            action_representation,
        )

    config = load_config(config_path)
    _ensure_physics_world(config, gui)

    # val episode 列表 + 每 episode 的初始积木位置。
    split_doc = json.loads(
        (Path(dataset_dir) / "episode_split.json").read_text(encoding="utf-8")
    )
    summaries = []
    with open(
        Path(dataset_dir) / "episode_summary.jsonl", encoding="utf-8"
    ) as fh:
        for line in fh:
            line = line.strip()
            if line:
                summaries.append(json.loads(line))
    summaries_by_idx = {s["episode_idx"]: s for s in summaries}

    val_episodes = [
        idx for idx in split_doc["val"] if idx in summaries_by_idx
    ]
    if max_episodes is not None:
        val_episodes = val_episodes[:max_episodes]
    print(f"评估 episodes: {len(val_episodes)} 个 (val 共 {len(split_doc['val'])})")

    output_dir.mkdir(parents=True, exist_ok=True)
    per_episode_path = output_dir / "per_episode.jsonl"
    summary_path = output_dir / "summary.json"

    records = []
    try:
        with open(per_episode_path, "w", encoding="utf-8") as fh:
            for episode_idx in val_episodes:
                summary = summaries_by_idx[episode_idx]
                block_pos = summary["initial_block_pos"]
                # 相机采样用与数据采集一致的 seed，保证可复现。
                random.seed(summary.get("random_seed", episode_idx))
                try:
                    rec = run_rollout_episode(
                        model=model,
                        config=config,
                        block_pos=block_pos,
                        device=device,
                        action_representation=action_representation,
                        action_stats=action_stats,
                        action_tokenizer=action_tokenizer,
                        max_steps=max_steps,
                        gui=gui,
                    )
                    rec["episode_idx"] = episode_idx
                except Exception as exc:
                    # 单 episode 失败不中断整批评估；重置物理世界，下一 episode 重新连接。
                    _teardown_physics_world()
                    rec = {
                        "episode_idx": episode_idx,
                        "success": False,
                        "error": repr(exc),
                    }
                records.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()

                status = "✅" if rec.get("success") else "❌"
                if rec.get("final_distance") is not None:
                    msg = (
                        f"{status} episode {episode_idx}: "
                        f"steps={rec.get('steps')}, "
                        f"final_distance={rec['final_distance']:.4f} m, "
                        f"reason={rec.get('termination_reason')}"
                    )
                else:
                    msg = f"{status} episode {episode_idx}: error={rec.get('error')}"
                print(msg, file=sys.stderr if not rec.get("success") else sys.stdout)
    finally:
        _teardown_physics_world()

    num_episodes = len(records)
    num_success = sum(1 for r in records if r.get("success"))
    final_distances = [
        r["final_distance"] for r in records if r.get("final_distance") is not None
    ]
    steps_list = [r["steps"] for r in records if r.get("steps") is not None]

    result_summary = {
        "checkpoint": str(checkpoint_path),
        "dataset_dir": dataset_dir,
        "action_representation": action_representation,
        "num_episodes": num_episodes,
        "num_success": num_success,
        "success_rate": (num_success / num_episodes) if num_episodes else 0.0,
        "mean_final_distance": (
            float(statistics.mean(final_distances)) if final_distances else None
        ),
        "median_final_distance": (
            float(statistics.median(final_distances)) if final_distances else None
        ),
        "mean_steps": float(statistics.mean(steps_list)) if steps_list else None,
    }
    summary_path.write_text(
        json.dumps(result_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"\n📊 评估完成: {num_success}/{num_episodes} 成功, "
        f"success_rate={result_summary['success_rate']:.3f}"
    )
    print(f"📁 输出: {output_dir}")
    return result_summary


def main():
    parser = argparse.ArgumentParser(description="BC 闭环 rollout 评估")
    parser.add_argument(
        "--checkpoint", required=True, help="BC checkpoint 路径（必需）"
    )
    parser.add_argument(
        "--output-dir", required=True, help="评估输出目录（必需）"
    )
    parser.add_argument(
        "--config", default=_DEFAULT_CONFIG, help="仿真配置文件"
    )
    parser.add_argument(
        "--dataset-dir", default=_DEFAULT_DATASET, help="数据集目录"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="最多评估的 val episode 数（默认全部）",
    )
    parser.add_argument(
        "--max-steps", type=int, default=200, help="单 episode 最大决策步数"
    )
    parser.add_argument(
        "--gui", action="store_true", help="使用 GUI 连接（默认 DIRECT）"
    )
    args = parser.parse_args()

    try:
        run_rollout_evaluation(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            max_episodes=args.max_episodes,
            max_steps=args.max_steps,
            gui=args.gui,
        )
    except Exception as exc:
        print(f"评估失败: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
