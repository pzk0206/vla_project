"""Episode 级训练/验证划分。

按 block 初始位置 X/Y 分层，将 episode 划分为训练集和验证集。
输出 episode_split.json 供下游（tokenizer 重拟合、BC 训练）使用。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_DATASET_DIR = "outputs/dataset/expert_scaling_v1"
_TRAIN_RATIO = 250.0 / 300.0  # ≈ 0.833
_TRAIN_COUNT = 250
_VAL_COUNT = 50
_NUM_X_BINS = 5
_NUM_Y_BINS = 5
_SPLIT_SEED = 42
_SCHEMA_VERSION = "episode_split_v1"


def _compute_quantile_edges(values, num_bins):
    """计算等频分箱边界（与 _axis_bin_counts 的五箱概念对齐）。"""
    sorted_values = sorted(values)
    n = len(sorted_values)
    edges = []
    for i in range(num_bins + 1):
        idx = int(round(i * (n - 1) / num_bins))
        edges.append(sorted_values[idx])
    return edges


def _assign_bin(value, edges):
    """将值分配到分箱索引 (0-based)。等于上边界的值放入最后一箱。"""
    for i in range(len(edges) - 2):
        if value < edges[i + 1]:
            return i
    return len(edges) - 2


def _read_episode_summaries(episode_summary_path):
    """读取 episode_summary.jsonl，返回 list[dict]（按 episode_idx 排序）。"""
    rows = []
    with open(episode_summary_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    # 校验
    if len(rows) != 300:
        raise ValueError(f"预期 300 条 episode 摘要，实际 {len(rows)} 条")
    episode_indices = {r["episode_idx"] for r in rows}
    if episode_indices != set(range(300)):
        missing = sorted(set(range(300)) - episode_indices)
        if missing:
            raise ValueError(f"缺失 episode: {missing}")
    rows.sort(key=lambda r: r["episode_idx"])
    return rows


def _stratified_split(rows):
    """按 initial_block_pos X/Y 五箱分层，做 250/50 划分。"""
    xs = [r["initial_block_pos"][0] for r in rows]
    ys = [r["initial_block_pos"][1] for r in rows]

    x_edges = _compute_quantile_edges(xs, _NUM_X_BINS)
    y_edges = _compute_quantile_edges(ys, _NUM_Y_BINS)

    # 分配 episode 到 cell
    cells = {}
    for r in rows:
        xi = _assign_bin(r["initial_block_pos"][0], x_edges)
        yi = _assign_bin(r["initial_block_pos"][1], y_edges)
        key = (xi, yi)
        cells.setdefault(key, []).append(r["episode_idx"])

    # 每个 cell 内随机分配
    rng = random.Random(_SPLIT_SEED)
    train = []
    val = []
    for cell_eps in cells.values():
        shuffled = list(cell_eps)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * (1 - _TRAIN_RATIO))))
        n_val = min(n_val, len(shuffled) - 1) if len(shuffled) > 1 else 0
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])

    # 全局调整：如果验证集不足或多于 50，随机迁移
    while len(val) < _VAL_COUNT:
        rng.shuffle(train)
        val.append(train.pop())
    while len(val) > _VAL_COUNT:
        rng.shuffle(val)
        train.append(val.pop())

    train.sort()
    val.sort()

    # 校验
    assert len(train) == _TRAIN_COUNT, f"train={len(train)}"
    assert len(val) == _VAL_COUNT, f"val={len(val)}"
    assert len(set(train) & set(val)) == 0, "train/val 重叠"
    assert set(train) | set(val) == set(range(300)), "未覆盖全部 episode"

    # 校验 X/Y 五箱覆盖
    train_xs = [xs[i] for i in train]
    train_ys = [ys[i] for i in train]
    val_xs = [xs[i] for i in val]
    val_ys = [ys[i] for i in val]

    for name, t_xs, t_ys in [("train", train_xs, train_ys), ("val", val_xs, val_ys)]:
        t_x_bins = _assign_bin(min(t_xs), x_edges), _assign_bin(max(t_xs), x_edges)
        t_y_bins = _assign_bin(min(t_ys), y_edges), _assign_bin(max(t_ys), y_edges)
        # 检查覆盖了所有 5 箱
        x_covered = len({_assign_bin(v, x_edges) for v in t_xs})
        y_covered = len({_assign_bin(v, y_edges) for v in t_ys})
        if x_covered < _NUM_X_BINS:
            raise ValueError(f"{name} X 箱覆盖不足: {x_covered}/{_NUM_X_BINS}")
        if y_covered < _NUM_Y_BINS:
            raise ValueError(f"{name} Y 箱覆盖不足: {y_covered}/{_NUM_Y_BINS}")

    return {
        "train": train,
        "val": val,
        "x_edges": x_edges,
        "y_edges": y_edges,
    }


def _file_fingerprint(path):
    """文件的 SHA-256 指纹。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_split(dataset_dir):
    """执行分层划分，输出 episode_split.json。"""
    dataset_dir = Path(dataset_dir)
    summary_path = dataset_dir / "episode_summary.jsonl"
    if not summary_path.is_file():
        raise FileNotFoundError(f"找不到 episode 摘要: {summary_path}")

    rows = _read_episode_summaries(summary_path)
    result = _stratified_split(rows)

    split_doc = {
        "schema_version": _SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "split_seed": _SPLIT_SEED,
        "train_count": len(result["train"]),
        "val_count": len(result["val"]),
        "train": result["train"],
        "val": result["val"],
        "stratification": {
            "method": "quantile_5_bin",
            "fields": ["initial_block_pos.x", "initial_block_pos.y"],
            "x_edges": [round(e, 6) for e in result["x_edges"]],
            "y_edges": [round(e, 6) for e in result["y_edges"]],
        },
        "provenance": {
            "source": str(summary_path),
            "source_sha256": _file_fingerprint(summary_path),
        },
    }

    output_path = dataset_dir / "episode_split.json"
    output_path.write_text(
        json.dumps(split_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # 打印摘要
    train_xs = [rows[i]["initial_block_pos"][0] for i in result["train"]]
    train_ys = [rows[i]["initial_block_pos"][1] for i in result["train"]]
    val_xs = [rows[i]["initial_block_pos"][0] for i in result["val"]]
    val_ys = [rows[i]["initial_block_pos"][1] for i in result["val"]]

    print(
        f"Episode 划分完成：train={len(result['train'])}，val={len(result['val'])}，"
        f"split_seed={_SPLIT_SEED}"
    )
    print(
        f"  train X: [{min(train_xs):.4f}, {max(train_xs):.4f}]  "
        f"Y: [{min(train_ys):.4f}, {max(train_ys):.4f}]"
    )
    print(
        f"  val   X: [{min(val_xs):.4f}, {max(val_xs):.4f}]  "
        f"Y: [{min(val_ys):.4f}, {max(val_ys):.4f}]"
    )
    print(f"  输出: {output_path}")

    return split_doc


def main():
    parser = argparse.ArgumentParser(
        description="Episode 级分层训练/验证划分"
    )
    parser.add_argument("--dataset-dir", default=_DEFAULT_DATASET_DIR)
    args = parser.parse_args()
    try:
        run_split(args.dataset_dir)
    except Exception as exc:
        print(f"划分失败: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
