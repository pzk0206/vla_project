"""VLA 训练循环与 vla-train CLI。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from vla_project.output_paths import resolve_new_output_directory

_DEFAULT_DATASET = "outputs/dataset/expert_multi_v2"
_DEFAULT_OUTPUT = "outputs/training"
_TRAINING_SEED = 42
_TEXT_ENCODER_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_MODEL_ARCHITECTURE = "resnet18_multilingual_minilm_fusion_coloraux_v1"


def _validate_text_encoder_distinguishes_chinese():
    """训练前门禁：验证文本编码器能区分红/蓝中文指令。

    如果红/蓝 token 或 embedding 完全相同，训练没有任何语言信号，
    VLA 退化为 BC——必须在此处硬失败，不允许继续。
    """
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(_TEXT_ENCODER_ID)
    tokenizer = encoder.tokenizer

    red_text = "悬停在红色积木上方"
    blue_text = "悬停在蓝色积木上方"

    # 门禁 A：token 序列必须不同
    red_ids = tokenizer.encode(red_text, add_special_tokens=False)
    blue_ids = tokenizer.encode(blue_text, add_special_tokens=False)
    if red_ids == blue_ids:
        raise ValueError(
            f"文本编码器 {_TEXT_ENCODER_ID} 将红/蓝指令编码为相同 token 序列: "
            f"{red_ids}"
        )

    # 门禁 B：embedding 必须可区分且数值有效
    import torch
    emb = encoder.encode(
        [red_text, blue_text], convert_to_tensor=True, show_progress_bar=False
    )
    if emb.shape != (2, 384):
        raise ValueError(f"embedding 形状异常: {emb.shape}，预期 (2, 384)")
    if not torch.isfinite(emb).all():
        raise ValueError("embedding 包含 NaN 或 Inf")
    sim = torch.nn.functional.cosine_similarity(
        emb[0:1], emb[1:2], dim=1
    ).item()
    if sim > 0.99:
        raise ValueError(
            f"红/蓝 embedding 无法区分: cosine_sim={sim:.6f}（必须 < 0.99）"
        )
    print(
        f"✅ 编码预检通过: 红/蓝 token 不同, embedding cosine_sim={sim:.4f}"
    )
    return True


def _make_optimizer(model, lr=1e-4, weight_decay=1e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _make_scheduler(optimizer, T_0=10):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0)


def _vla_loss(
    action_pred,
    aux_pred,
    color_logits,
    action_target,
    aux_target,
    color_target,
    color_weight=1.0,
):
    """动作回归 + aux(gripper/terminate) + 目标色分类 三路损失。

    color_weight 放大目标色监督，强制模型利用语言区分红/蓝（P1）。
    """
    action_loss = nn.functional.smooth_l1_loss(action_pred, action_target)
    aux_loss = nn.functional.binary_cross_entropy_with_logits(aux_pred, aux_target)
    color_loss = nn.functional.cross_entropy(color_logits, color_target)
    return action_loss + aux_loss + color_weight * color_loss


def _collate_vla(batch, model, device):
    """将 VLADataset batch 整理为模型输入。

    text_emb 使用 encode_texts_grad()（带梯度），使语言分支参与反向传播；
    该输出与 encode_texts() 逐位一致，不影响推理时的一致性。
    """
    images = torch.stack([item[0] for item in batch]).to(device)
    texts = [item[1] for item in batch]
    text_emb = model.encode_texts_grad(texts).to(device)
    action_target = torch.stack([item[2] for item in batch]).to(device)
    aux_target = torch.stack([item[3] for item in batch]).to(device)
    color_target = torch.stack([item[4] for item in batch]).to(device)
    return images, text_emb, action_target, aux_target, color_target


def _run_epoch(model, dataloader, optimizer, device, is_train, color_weight=1.0):
    """跑一个 epoch，返回 (平均 loss, 目标色分类准确率)。"""
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    color_correct = 0

    with torch.set_grad_enabled(is_train):
        for batch in dataloader:
            images, text_emb, action_target, aux_target, color_target = _collate_vla(
                batch, model, device
            )

            action_pred, aux_pred, color_logits = model(images, text_emb)
            loss = _vla_loss(
                action_pred,
                aux_pred,
                color_logits,
                action_target,
                aux_target,
                color_target,
                color_weight=color_weight,
            )
            color_correct += (
                color_logits.argmax(dim=1) == color_target
            ).sum().item()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    color_acc = color_correct / max(total_samples, 1)
    return total_loss / max(total_samples, 1), color_acc


def _save_checkpoint(
    model,
    optimizer,
    epoch,
    path,
    common_metadata=None,
    epoch_metrics=None,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "action_representation": model.action_representation,
    }
    metadata = dict(common_metadata or {})
    metadata.update(epoch_metrics or {})
    if metadata:
        state["metadata"] = metadata
    torch.save(state, path)


def _write_jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_all_rows(dataset_dir):
    trajectory_path = Path(dataset_dir) / "trajectory_expert.jsonl"
    rows = []
    with open(trajectory_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _default_training_output(overfit_episodes):
    scope = (
        f"overfit_{overfit_episodes}"
        if overfit_episodes is not None
        else "full"
    )
    return Path(_DEFAULT_OUTPUT) / f"vla_regression_{scope}_v4"


def _validate_training_output_version(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.name.endswith("_v4"):
        raise ValueError("VLA v4 training output directory must end with _v4")
    return output_dir


def _validate_v2_split(split_doc):
    if split_doc.get("schema_version") != "episode_split_v2":
        raise ValueError("VLA v2 training requires episode_split_v2")
    train_ids = split_doc.get("train")
    val_ids = split_doc.get("val")
    if (
        not isinstance(train_ids, list)
        or not isinstance(val_ids, list)
        or len(train_ids) != 250
        or len(val_ids) != 50
        or len(set(train_ids)) != 250
        or len(set(val_ids)) != 50
        or set(train_ids) & set(val_ids)
        or set(train_ids) | set(val_ids) != set(range(300))
    ):
        raise ValueError("invalid VLA v2 train/val episode split")
    expected_counts = {
        "train": {"red": 125, "blue": 125},
        "val": {"red": 25, "blue": 25},
    }
    if split_doc.get("task_counts") != expected_counts:
        raise ValueError("invalid VLA v2 split task_counts")


def _validate_paired_split(split_doc):
    """成对数据 split 契约：episode_split_v3，train 500 / val 100。"""
    if split_doc.get("schema_version") != "episode_split_v3":
        raise ValueError("paired VLA training requires episode_split_v3")
    train_ids = split_doc.get("train")
    val_ids = split_doc.get("val")
    if (
        not isinstance(train_ids, list)
        or not isinstance(val_ids, list)
        or len(train_ids) != 500
        or len(val_ids) != 100
        or len(set(train_ids)) != 500
        or len(set(val_ids)) != 100
        or set(train_ids) & set(val_ids)
        or set(train_ids) | set(val_ids) != set(range(600))
    ):
        raise ValueError("invalid paired VLA train/val episode split")
    expected_counts = {
        "train": {"red": 250, "blue": 250},
        "val": {"red": 50, "blue": 50},
    }
    if split_doc.get("task_counts") != expected_counts:
        raise ValueError("invalid paired VLA split task_counts")


def _load_v2_contract(dataset_dir):
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "dataset_manifest.json"
    split_path = dataset_dir / "episode_split.json"
    summary_path = dataset_dir / "episode_summary.jsonl"
    for path in (manifest_path, split_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing VLA training input: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_doc = json.loads(split_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "expert_multi_v2":
        raise ValueError("VLA training requires expert_multi_v2 manifest")
    is_paired = bool(manifest.get("paired_source_dir"))
    if is_paired:
        _validate_paired_split(split_doc)
    else:
        _validate_v2_split(split_doc)
    provenance = split_doc.get("provenance", {})
    if provenance.get("source_sha256") != _file_sha256(summary_path):
        raise ValueError("episode split source summary SHA-256 mismatch")
    if provenance.get("manifest_sha256") != _file_sha256(manifest_path):
        raise ValueError("episode split manifest SHA-256 mismatch")
    return manifest, split_doc


def _build_checkpoint_metadata(
    dataset_dir,
    *,
    train_episode_ids,
    action_stats,
    training_seed,
):
    dataset_dir = Path(dataset_dir)
    file_names = {
        "dataset_manifest": "dataset_manifest.json",
        "trajectory_expert": "trajectory_expert.jsonl",
        "episode_summary": "episode_summary.jsonl",
        "episode_split": "episode_split.json",
    }
    manifest = json.loads((dataset_dir / file_names["dataset_manifest"]).read_text())
    split_doc = json.loads((dataset_dir / file_names["episode_split"]).read_text())
    return {
        "action_semantics": "absolute_joint_target_v1",
        "dataset_schema_version": manifest.get("schema_version"),
        "split_schema_version": split_doc.get("schema_version"),
        **{
            f"{key}_sha256": _file_sha256(dataset_dir / file_name)
            for key, file_name in file_names.items()
        },
        "train_episode_ids": [int(value) for value in train_episode_ids],
        "action_stats": {
            "mean": [float(value) for value in action_stats[0]],
            "std": [float(value) for value in action_stats[1]],
        },
        "training_seed": int(training_seed),
        "model_architecture": _MODEL_ARCHITECTURE,
        "text_encoder": _TEXT_ENCODER_ID,
    }


def _set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(
    dataset_dir=_DEFAULT_DATASET,
    output_dir=None,
    epochs=50,
    batch_size=32,
    lr=1e-4,
    overfit_episodes=None,
    color_weight=1.0,
    project_root_override=None,
):
    """执行 VLA 训练，返回输出目录和训练摘要。"""
    if output_dir is None:
        output_dir = _default_training_output(overfit_episodes)
    output_dir = resolve_new_output_directory(
        output_dir,
        allowed_root="outputs/training",
        project_root_override=project_root_override,
    )
    output_dir = _validate_training_output_version(output_dir)

    from .dataset import VLADataset, compute_action_stats
    from .vla_model import VLAModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 训练前门禁：编码器必须能区分中文红/蓝指令
    _validate_text_encoder_distinguishes_chinese()

    _set_reproducible_seed(_TRAINING_SEED)

    dataset_dir = Path(dataset_dir)
    split_path = dataset_dir / "episode_split.json"
    _, split_doc = _load_v2_contract(dataset_dir)

    # 计算 action_stats
    all_rows = _load_all_rows(dataset_dir)
    subset_ids = None

    if overfit_episodes is not None:
        train_ids = list(split_doc["train"])
        subset_ids = list(train_ids[:overfit_episodes])
        subset_rows = [r for r in all_rows if r["episode_idx"] in set(subset_ids)]
        action_stats = compute_action_stats(subset_rows)
    else:
        train_ids = set(split_doc["train"])
        train_rows = [r for r in all_rows if r["episode_idx"] in train_ids]
        action_stats = compute_action_stats(train_rows)

    # 数据集
    train_dataset = VLADataset(
        dataset_dir=str(dataset_dir),
        split="train",
        action_stats=action_stats,
        episode_ids=subset_ids,
    )
    val_dataset = VLADataset(
        dataset_dir=str(dataset_dir),
        split="val",
        action_stats=action_stats,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=False, collate_fn=lambda x: x,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=False, collate_fn=lambda x: x,
    )

    print(
        f"数据: train={len(train_dataset)} samples, "
        f"val={len(val_dataset)} samples"
    )
    if overfit_episodes is not None:
        print(f"Overfit 模式: {overfit_episodes} episodes")

    # 模型
    model = VLAModel().to(device)
    optimizer = _make_optimizer(model, lr=lr)
    scheduler = _make_scheduler(optimizer)

    run_id = datetime.now(timezone.utc).isoformat(timespec="seconds")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = _build_checkpoint_metadata(
        dataset_dir,
        train_episode_ids=sorted(subset_ids or split_doc["train"]),
        action_stats=action_stats,
        training_seed=_TRAINING_SEED,
    )

    config = {
        "run_id": run_id,
        "dataset_dir": str(dataset_dir),
        "action_representation": "regression",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "overfit_episodes": overfit_episodes,
        "color_weight": color_weight,
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "training_seed": _TRAINING_SEED,
        "action_stats": checkpoint_metadata["action_stats"],
        "episode_split_sha256": checkpoint_metadata["episode_split_sha256"],
        "dataset_manifest_sha256": checkpoint_metadata["dataset_manifest_sha256"],
    }
    (output_dir / "config.yaml").write_text(
        yaml.dump(config, allow_unicode=True), encoding="utf-8"
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_color_acc = _run_epoch(
            model, train_loader, optimizer, device, True, color_weight=color_weight
        )
        val_loss, val_color_acc = _run_epoch(
            model, val_loader, optimizer, device, False, color_weight=color_weight
        )
        scheduler.step()

        train_losses.append({"epoch": epoch, "loss": round(train_loss, 6)})
        val_losses.append({"epoch": epoch, "loss": round(val_loss, 6)})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(
                model, optimizer, epoch,
                output_dir / "checkpoint_best.pt",
                common_metadata=checkpoint_metadata,
                epoch_metrics={"val_loss": val_loss, "train_loss": train_loss},
            )

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"  epoch {epoch:3d}/{epochs}  "
                f"train_loss={train_loss:.4f}(color {train_color_acc:.3f})  "
                f"val_loss={val_loss:.4f}(color {val_color_acc:.3f})"
            )

    elapsed = time.time() - t_start

    _save_checkpoint(
        model, optimizer, epochs,
        output_dir / "checkpoint_last.pt",
        common_metadata=checkpoint_metadata,
        epoch_metrics={"val_loss": val_loss, "train_loss": train_loss},
    )

    _write_jsonl(output_dir / "train_loss.jsonl", train_losses)
    _write_jsonl(output_dir / "val_loss.jsonl", val_losses)

    summary = {
        "run_id": run_id,
        "action_representation": "regression",
        "epochs_completed": epochs,
        "best_val_loss": round(best_val_loss, 6),
        "final_train_loss": round(train_losses[-1]["loss"], 6),
        "final_val_loss": round(val_losses[-1]["loss"], 6),
        "elapsed_seconds": round(elapsed, 1),
        "device": str(device),
        "training_seed": _TRAINING_SEED,
        "episode_split_sha256": checkpoint_metadata["episode_split_sha256"],
        "dataset_manifest_sha256": checkpoint_metadata["dataset_manifest_sha256"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n训练完成: best_val_loss={best_val_loss:.4f}, elapsed={elapsed:.1f}s")
    print(f"输出: {output_dir}")

    return output_dir, summary


def main():
    parser = argparse.ArgumentParser(description="VLA 训练")
    parser.add_argument("--dataset-dir", default=_DEFAULT_DATASET)
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int, default=50, help="训练 epoch 数（默认: 50）")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size（默认: 32）")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率（默认: 1e-4）")
    parser.add_argument("--overfit", type=int, default=None, help="Overfit 模式：只用前 N 个训练 episode")
    parser.add_argument("--color-weight", type=float, default=1.0,
                        help="目标色分类损失权重（默认: 1.0；调小可减少对目标选择的强调、加大动作精度占比）")
    args = parser.parse_args()

    try:
        run_training(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            overfit_episodes=args.overfit,
            color_weight=args.color_weight,
        )
    except Exception as exc:
        print(f"训练失败: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
