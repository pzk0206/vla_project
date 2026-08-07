"""BC 训练循环与 vla-train-bc CLI。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

_DEFAULT_DATASET = "outputs/dataset/expert_scaling_v1"
_DEFAULT_OUTPUT = "outputs/training"


def _make_optimizer(model, lr=1e-4, weight_decay=1e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _make_scheduler(optimizer, T_0=10):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0
    )


def _regression_loss(action_pred, aux_pred, action_target, aux_target):
    """回归损失：SmoothL1(action) + BCE(aux)。"""
    action_loss = nn.functional.smooth_l1_loss(action_pred, action_target)
    aux_loss = nn.functional.binary_cross_entropy_with_logits(
        aux_pred, aux_target
    )
    return action_loss + aux_loss


def _classification_loss(action_pred, aux_pred, action_target, aux_target):
    """分类损失：每关节独立 CrossEntropy (sum) + BCE(aux)。"""
    # action_pred: (B, 7, C), action_target: (B, 7)
    B, J, C = action_pred.shape
    ce = nn.functional.cross_entropy(
        action_pred.view(B * J, C),
        action_target.view(B * J),
        reduction="sum",
    )
    action_loss = ce / B  # 归一化到每样本
    aux_loss = nn.functional.binary_cross_entropy_with_logits(
        aux_pred, aux_target
    )
    return action_loss + aux_loss


def _run_epoch(model, dataloader, optimizer, device, is_train, loss_fn):
    """跑一个 epoch，返回平均 loss。"""
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for batch in dataloader:
            images = batch[0].to(device)
            action_target = batch[1].to(device)
            aux_target = batch[2].to(device)

            action_pred, aux_pred = model(images)
            loss = loss_fn(action_pred, aux_pred, action_target, aux_target)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    return total_loss / max(total_samples, 1)


def _save_checkpoint(model, optimizer, epoch, path, metadata=None):
    """保存 checkpoint。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "action_representation": model.action_representation,
    }
    if metadata:
        state["metadata"] = metadata
    torch.save(state, path)


def _write_jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _compute_overfit_stats(rows, subset_episodes):
    """计算 overfit 子集的 action_stats。"""
    subset_rows = [r for r in rows if r["episode_idx"] in subset_episodes]
    from .dataset import compute_action_stats

    return compute_action_stats(subset_rows)


def _load_all_rows(dataset_dir):
    """加载全部轨迹行（用于计算 action stats）。"""
    trajectory_path = Path(dataset_dir) / "trajectory_expert.jsonl"
    rows = []
    with open(trajectory_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def run_training(
    dataset_dir=_DEFAULT_DATASET,
    output_dir=None,
    action_representation="regression",
    epochs=50,
    batch_size=32,
    lr=1e-4,
    overfit_episodes=None,
):
    """执行 BC 训练，返回输出目录和训练摘要。"""
    from .dataset import BCDataset, compute_action_stats
    from .model import BCModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    dataset_dir = Path(dataset_dir)
    split_path = dataset_dir / "episode_split.json"

    # 确定 tokenizer 审计路径（分类组需要）
    tokenizer_audit_path = None
    if action_representation != "regression":
        tokenizer_audit_path = (
            dataset_dir / "action_tokenization_audit_v2"
            / "action_tokenization_audit.json"
        )
        if not tokenizer_audit_path.is_file():
            raise FileNotFoundError(
                f"找不到 tokenizer 审计报告: {tokenizer_audit_path}"
            )

    # 计算 action stats（回归组需要）
    action_stats = None
    all_rows = _load_all_rows(dataset_dir)
    subset_ids = None  # overfit 模式的 episode 子集

    # Overfit 模式：只选指定数量 episode
    if overfit_episodes is not None:
        with open(split_path, encoding="utf-8") as fh:
            split_doc = json.load(fh)
        train_ids = list(split_doc["train"])
        subset_ids = list(train_ids[:overfit_episodes])
        action_stats = _compute_overfit_stats(all_rows, set(subset_ids))
    else:
        # 全量训练：用全部训练 episode 算 stats
        with open(split_path, encoding="utf-8") as fh:
            split_doc = json.load(fh)
        train_ids = set(split_doc["train"])
        train_rows = [r for r in all_rows if r["episode_idx"] in train_ids]
        action_stats = compute_action_stats(train_rows)

    # 创建数据集
    train_dataset = BCDataset(
        dataset_dir=str(dataset_dir),
        split="train",
        action_representation=action_representation,
        tokenizer_audit_path=(
            str(tokenizer_audit_path) if tokenizer_audit_path else None
        ),
        action_stats=action_stats,
        episode_ids=subset_ids,
    )
    val_dataset = BCDataset(
        dataset_dir=str(dataset_dir),
        split="val",
        action_representation=action_representation,
        tokenizer_audit_path=(
            str(tokenizer_audit_path) if tokenizer_audit_path else None
        ),
        action_stats=action_stats,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    print(
        f"数据: train={len(train_dataset)} samples, "
        f"val={len(val_dataset)} samples"
    )
    if overfit_episodes is not None:
        print(f"Overfit 模式: {overfit_episodes} episodes")
    print(f"动作表示: {action_representation}")

    # 模型
    model = BCModel(action_representation=action_representation).to(device)
    is_classification = model.is_classification()
    loss_fn = _classification_loss if is_classification else _regression_loss

    optimizer = _make_optimizer(model, lr=lr)
    scheduler = _make_scheduler(optimizer)

    # 输出目录
    if output_dir is None:
        tag = (
            f"bc_{action_representation}"
            + (f"_overfit_{overfit_episodes}" if overfit_episodes else "")
            + "_v1"
        )
        output_dir = Path(_DEFAULT_OUTPUT) / tag
    else:
        output_dir = Path(output_dir)

    run_id = datetime.now(timezone.utc).isoformat(timespec="seconds")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置快照
    config = {
        "run_id": run_id,
        "dataset_dir": str(dataset_dir),
        "action_representation": action_representation,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "overfit_episodes": overfit_episodes,
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }
    (output_dir / "config.yaml").write_text(
        yaml.dump(config, allow_unicode=True), encoding="utf-8"
    )

    # 训练循环
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(
            model, train_loader, optimizer, device, True, loss_fn
        )
        val_loss = _run_epoch(
            model, val_loader, optimizer, device, False, loss_fn
        )
        scheduler.step()

        train_losses.append({"epoch": epoch, "loss": round(train_loss, 6)})
        val_losses.append({"epoch": epoch, "loss": round(val_loss, 6)})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(
                model,
                optimizer,
                epoch,
                output_dir / "checkpoint_best.pt",
                metadata={"val_loss": val_loss, "train_loss": train_loss},
            )

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"  epoch {epoch:3d}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )

    elapsed = time.time() - t_start

    # 保存最后一个 epoch
    _save_checkpoint(
        model,
        optimizer,
        epochs,
        output_dir / "checkpoint_last.pt",
        metadata={"val_loss": val_loss, "train_loss": train_loss},
    )

    # 保存 loss 曲线
    _write_jsonl(output_dir / "train_loss.jsonl", train_losses)
    _write_jsonl(output_dir / "val_loss.jsonl", val_losses)

    # 摘要
    summary = {
        "run_id": run_id,
        "action_representation": action_representation,
        "epochs_completed": epochs,
        "best_val_loss": round(best_val_loss, 6),
        "final_train_loss": round(train_losses[-1]["loss"], 6),
        "final_val_loss": round(val_losses[-1]["loss"], 6),
        "elapsed_seconds": round(elapsed, 1),
        "device": str(device),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n训练完成: best_val_loss={best_val_loss:.4f}, elapsed={elapsed:.1f}s")
    print(f"输出: {output_dir}")

    return output_dir, summary


def main():
    parser = argparse.ArgumentParser(
        description="VLA 行为克隆训练"
    )
    parser.add_argument("--dataset-dir", default=_DEFAULT_DATASET)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--action-representation",
        default="regression",
        choices=["regression", "absolute_q_32", "delta_q_64"],
        help="动作表示（默认: regression）",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="训练 epoch 数（默认: 50）"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size（默认: 32）"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率（默认: 1e-4）")
    parser.add_argument(
        "--overfit",
        type=int,
        default=None,
        help="Overfit 模式：只用前 N 个训练 episode",
    )
    args = parser.parse_args()

    try:
        run_training(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            action_representation=args.action_representation,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            overfit_episodes=args.overfit,
        )
    except Exception as exc:
        print(f"训练失败: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
