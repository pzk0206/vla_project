"""Episode 级训练/验证划分。

按 block 初始位置 X/Y 分层，将 episode 划分为训练集和验证集。
输出 episode_split.json 供下游（tokenizer 重拟合、BC 训练）使用。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
_V1_SCHEMA_VERSION = "episode_split_v1"
_V2_SCHEMA_VERSION = "episode_split_v2"
_V3_SCHEMA_VERSION = "episode_split_v3"


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


def _read_episode_summaries(episode_summary_path, expected_count=300):
    """读取 episode_summary.jsonl，返回 list[dict]（按 episode_idx 排序）。

    expected_count：v1/v2 单任务数据为 300；成对数据为 600。
    """
    rows = []
    with open(episode_summary_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    # 校验
    if len(rows) != expected_count:
        raise ValueError(f"预期 {expected_count} 条 episode 摘要，实际 {len(rows)} 条")
    episode_indices = {r["episode_idx"] for r in rows}
    if episode_indices != set(range(expected_count)):
        missing = sorted(set(range(expected_count)) - episode_indices)
        if missing:
            raise ValueError(f"缺失 episode: {missing}")
    rows.sort(key=lambda r: r["episode_idx"])
    return rows


def _stratified_split_v1(rows):
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


def _validated_v2_target_position(row):
    target_block = row.get("target_block")
    if target_block not in {"red", "blue"}:
        raise ValueError(
            f"episode {row.get('episode_idx')} target_block 非法: {target_block!r}"
        )
    scene_state = row.get("initial_scene_state")
    try:
        scene_target = scene_state["target_block"]
        position = scene_state["blocks"][target_block]["position"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"episode {row.get('episode_idx')} 缺少目标块 initial_scene_state"
        ) from exc
    if scene_target != target_block:
        raise ValueError(
            f"episode {row.get('episode_idx')} target_block 与 scene state 不一致"
        )
    if (
        not isinstance(position, list)
        or len(position) != 3
        or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in position)
    ):
        raise ValueError(
            f"episode {row.get('episode_idx')} 目标块位置非法"
        )
    initial_block_pos = row.get("initial_block_pos")
    if (
        not isinstance(initial_block_pos, list)
        or len(initial_block_pos) != 3
        or max(abs(float(a) - float(b)) for a, b in zip(position, initial_block_pos))
        > 1e-6
    ):
        raise ValueError(
            f"episode {row.get('episode_idx')} initial_block_pos 与目标 scene state 不一致"
        )
    return [float(value) for value in position]


def _split_v2_task_group(rows, val_count, rng):
    positions = {
        row["episode_idx"]: _validated_v2_target_position(row) for row in rows
    }
    xs = [position[0] for position in positions.values()]
    ys = [position[1] for position in positions.values()]
    x_edges = _compute_quantile_edges(xs, _NUM_X_BINS)
    y_edges = _compute_quantile_edges(ys, _NUM_Y_BINS)
    cells = {}
    for episode_idx, position in positions.items():
        key = (_assign_bin(position[0], x_edges), _assign_bin(position[1], y_edges))
        cells.setdefault(key, []).append(episode_idx)

    train = []
    val = []
    for cell_key in sorted(cells):
        shuffled = list(cells[cell_key])
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * (1 - _TRAIN_RATIO))))
        n_val = min(n_val, len(shuffled) - 1) if len(shuffled) > 1 else 0
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])

    while len(val) < val_count:
        rng.shuffle(train)
        val.append(train.pop())
    while len(val) > val_count:
        rng.shuffle(val)
        train.append(val.pop())

    for split_name, episode_ids in (("train", train), ("val", val)):
        x_covered = len({_assign_bin(positions[index][0], x_edges) for index in episode_ids})
        y_covered = len({_assign_bin(positions[index][1], y_edges) for index in episode_ids})
        if x_covered < _NUM_X_BINS or y_covered < _NUM_Y_BINS:
            raise ValueError(
                f"{split_name} 目标位置分箱覆盖不足: X={x_covered}/5 Y={y_covered}/5"
            )
    return train, val, x_edges, y_edges


def _stratified_split_v2(rows):
    task_rows = {"red": [], "blue": []}
    for row in rows:
        if row.get("schema_version") != "expert_multi_v2":
            raise ValueError("expert_multi_v2 划分包含其他 schema")
        target_block = row.get("target_block")
        if target_block not in task_rows:
            raise ValueError(f"非法 target_block: {target_block!r}")
        task_rows[target_block].append(row)
    if {color: len(items) for color, items in task_rows.items()} != {
        "red": 150,
        "blue": 150,
    }:
        raise ValueError("expert_multi_v2 必须严格包含 red=150、blue=150")

    rng = random.Random(_SPLIT_SEED)
    train = []
    val = []
    per_task_edges = {}
    for color in ("red", "blue"):
        color_train, color_val, x_edges, y_edges = _split_v2_task_group(
            task_rows[color], 25, rng
        )
        train.extend(color_train)
        val.extend(color_val)
        per_task_edges[color] = {
            "x_edges": x_edges,
            "y_edges": y_edges,
        }
    train.sort()
    val.sort()
    if len(train) != _TRAIN_COUNT or len(val) != _VAL_COUNT:
        raise AssertionError(f"v2 split size invalid: train={len(train)} val={len(val)}")
    if set(train) & set(val) or set(train) | set(val) != set(range(300)):
        raise AssertionError("v2 train/val 未严格分离并覆盖全部 episode")
    return {"train": train, "val": val, "per_task_edges": per_task_edges}


def _detect_paired_dataset(rows):
    """成对数据：补录轨迹带 is_paired_copy=True / paired_with 字段。"""
    return any(row.get("is_paired_copy") is True for row in rows)


def _group_paired_scenes(rows):
    """把成对数据按场景分组：以非补录轨迹为场景锚点，补录轨迹归入伴侣场景。

    返回 (scenes, paired_map)：
    - scenes: list[dict]，每个是原始（非补录）轨迹，作为场景代表。
    - paired_map: dict[原始 episode_idx -> list[补录 row]]。
    """
    original = [row for row in rows if row.get("is_paired_copy") is not True]
    paired = [row for row in rows if row.get("is_paired_copy") is True]
    paired_map = {}
    for row in paired:
        anchor = row.get("paired_with")
        if anchor is None:
            raise ValueError(f"补录轨迹缺少 paired_with: episode {row['episode_idx']}")
        paired_map.setdefault(anchor, []).append(row)
    return original, paired_map


def _stratified_split_paired(rows):
    """成对数据（600 条）以场景对为单位分层。

    对 300 个原始场景按目标块位置做 5×5 箱分层（红蓝各 150 → 125/25），
    再把每个场景的补录伴侣放入同一 split，保证「同场景双指令」不跨划分。
    返回的 train/val 各含 500/100 条（含补录）。
    """
    original, paired_map = _group_paired_scenes(rows)
    if len(original) != 300:
        raise ValueError(f"成对数据必须包含 300 个原始场景，实际 {len(original)}")
    if {color: sum(1 for r in original if r["target_block"] == color) for color in ("red", "blue")} != {
        "red": 150,
        "blue": 150,
    }:
        raise ValueError("成对数据原始场景必须红蓝各 150")

    # 复用 v2 的按目标块位置分层逻辑，得到场景级 train/val
    scene_result = _stratified_split_v2(original)
    scene_train = scene_result["train"]
    scene_val = scene_result["val"]

    train = []
    val = []
    for scene_index in scene_train:
        train.append(scene_index)
        train.extend(row["episode_idx"] for row in paired_map.get(scene_index, []))
    for scene_index in scene_val:
        val.append(scene_index)
        val.extend(row["episode_idx"] for row in paired_map.get(scene_index, []))

    # 校验：所有补录都归入了 train 或 val
    all_episodes = {row["episode_idx"] for row in rows}
    split_episodes = set(train) | set(val)
    if all_episodes != split_episodes or set(train) & set(val):
        raise AssertionError("成对分层未严格覆盖且未重叠")

    train.sort()
    val.sort()
    if len(train) != 500 or len(val) != 100:
        raise AssertionError(f"paired split size invalid: train={len(train)} val={len(val)}")
    # 每场景补录伴侣应与场景同 split（已由上述归组保证）
    return {
        "train": train,
        "val": val,
        "per_task_edges": scene_result["per_task_edges"],
        "scene_train": scene_train,
        "scene_val": scene_val,
    }


def _stratified_split(rows, is_paired=False):
    if is_paired:
        return _stratified_split_paired(rows)
    if rows and rows[0].get("schema_version") == "expert_multi_v2":
        return _stratified_split_v2(rows)
    return _stratified_split_v1(rows)


def _file_fingerprint(path):
    """文件的 SHA-256 指纹。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_split(dataset_dir):
    """执行分层划分，输出 episode_split.json。"""
    dataset_dir = Path(dataset_dir)
    summary_path = dataset_dir / "episode_summary.jsonl"
    if not summary_path.is_file():
        raise FileNotFoundError(f"找不到 episode 摘要: {summary_path}")

    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到数据 manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as manifest_handle:
        manifest = json.load(manifest_handle)

    # 成对数据 manifest 由 collect_paired 写入 paired_source_dir，目标规模翻倍。
    is_paired = bool(manifest.get("paired_source_dir"))
    expected_count = 600 if is_paired else 300
    rows = _read_episode_summaries(summary_path, expected_count=expected_count)
    is_paired = is_paired or _detect_paired_dataset(rows)
    result = _stratified_split(rows, is_paired=is_paired)
    is_multi_v2 = rows[0].get("schema_version") == "expert_multi_v2"

    if is_multi_v2:
        by_id = {row["episode_idx"]: row for row in rows}
        task_counts = {
            split_name: {
                color: sum(
                    by_id[index]["target_block"] == color
                    for index in result[split_name]
                )
                for color in ("red", "blue")
            }
            for split_name in ("train", "val")
        }
        stratification = {
            "method": (
                "paired_scene_then_target_block_then_quantile_5_bin_xy"
                if is_paired
                else "target_block_then_quantile_5_bin_xy"
            ),
            "fields": [
                "target_block",
                "initial_scene_state.blocks[target_block].position.x/y",
            ],
            "per_task_edges": {
                color: {
                    axis: [round(edge, 6) for edge in edges]
                    for axis, edges in result["per_task_edges"][color].items()
                }
                for color in ("red", "blue")
            },
        }
        if is_paired:
            stratification["paired"] = {
                "scene_train": result["scene_train"],
                "scene_val": result["scene_val"],
            }
    else:
        task_counts = None
        stratification = {
            "method": "quantile_5_bin",
            "fields": ["initial_block_pos.x", "initial_block_pos.y"],
            "x_edges": [round(e, 6) for e in result["x_edges"]],
            "y_edges": [round(e, 6) for e in result["y_edges"]],
        }

    split_doc = {
        "schema_version": (
            _V3_SCHEMA_VERSION
            if is_paired
            else (_V2_SCHEMA_VERSION if is_multi_v2 else _V1_SCHEMA_VERSION)
        ),
        "generator_version": (
            "split_episodes_v3"
            if is_paired
            else ("split_episodes_v2" if is_multi_v2 else "split_episodes_v1")
        ),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "split_seed": _SPLIT_SEED,
        "train_count": len(result["train"]),
        "val_count": len(result["val"]),
        "train": result["train"],
        "val": result["val"],
        "stratification": stratification,
        "provenance": {
            "source": str(summary_path),
            "source_sha256": _file_fingerprint(summary_path),
        },
    }
    if is_multi_v2:
        split_doc["task_counts"] = task_counts
        split_doc["provenance"]["manifest"] = str(manifest_path)
        split_doc["provenance"]["manifest_sha256"] = _file_fingerprint(
            manifest_path
        )

    output_path = dataset_dir / "episode_split.json"
    output_path.write_text(
        json.dumps(split_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # 打印摘要
    by_id = {row["episode_idx"]: row for row in rows}
    train_positions = [
        _validated_v2_target_position(by_id[index])
        if is_multi_v2
        else by_id[index]["initial_block_pos"]
        for index in result["train"]
    ]
    val_positions = [
        _validated_v2_target_position(by_id[index])
        if is_multi_v2
        else by_id[index]["initial_block_pos"]
        for index in result["val"]
    ]
    train_xs = [position[0] for position in train_positions]
    train_ys = [position[1] for position in train_positions]
    val_xs = [position[0] for position in val_positions]
    val_ys = [position[1] for position in val_positions]

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
