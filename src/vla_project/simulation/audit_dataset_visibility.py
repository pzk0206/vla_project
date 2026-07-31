"""确定性重放 expert 数据，并审计红块逐帧可见率。"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics
import tempfile

import numpy as np
import pybullet as p

from vla_project.simulation.control_arm import (
    capture_rgb_and_segmentation,
)
from vla_project.simulation.expert_dataset_replay import (
    ReplayValidationError,
    load_replay_inputs as load_audit_inputs,
    read_jsonl,
    replay_difference_is_acceptable,
    replay_episode_frames,
    validate_episode_contract,
    validate_replay_image,
)
from vla_project.vlm.evaluate_grounding_backprojection import (
    classify_visibility,
)


OBJECT_ID_MASK = (1 << 24) - 1
DEFAULT_DATASET_DIR = "outputs/dataset/expert_scaling_v1"


def count_block_pixels(segmentation, block_id):
    """统计 segmentation 中属于指定物体主体或链接的像素。"""
    if isinstance(block_id, bool) or not isinstance(block_id, int):
        raise ValueError("block_id 必须是整数")
    object_ids = np.asarray(segmentation, dtype=np.int64) & OBJECT_ID_MASK
    return int(np.count_nonzero(object_ids == block_id))


def build_visibility_rows(frame_observations, reference_pixels=None):
    """用每帧同一物理状态的无遮挡参考像素生成比率和分组。"""
    rows = []
    for observation in frame_observations:
        frame_reference = observation.get(
            "reference_block_pixels",
            reference_pixels,
        )
        if (
            isinstance(frame_reference, bool)
            or not isinstance(frame_reference, int)
            or frame_reference <= 0
        ):
            raise ValueError("reference_block_pixels 必须是正整数")
        visible = observation["visible_block_pixels"]
        if (
            isinstance(visible, bool)
            or not isinstance(visible, int)
            or not 0 <= visible <= frame_reference
        ):
            raise ValueError("visible_block_pixels 必须位于参考范围内")
        ratio = visible / frame_reference
        rows.append(
            {
                **observation,
                "reference_block_pixels": frame_reference,
                "block_visibility_ratio": ratio,
                "visibility_group": classify_visibility(ratio),
            }
        )
    return rows


def numeric_stats(values):
    """返回可 JSON 序列化的基础数值统计。"""
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None}
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def longest_nonclear_run(episode_idx, episode_rows):
    """找出一条 episode 中最长的连续非 clear 保存帧段。"""
    longest = None
    run_start = None
    run_rows = []
    for row in episode_rows + [None]:
        if row is not None and row["visibility_group"] != "clear":
            if run_start is None:
                run_start = row["step_idx"]
            run_rows.append(row)
            continue
        if run_rows:
            candidate = {
                "episode_idx": episode_idx,
                "start_step": run_start,
                "end_step": run_rows[-1]["step_idx"],
                "num_frames": len(run_rows),
            }
            if (
                longest is None
                or candidate["num_frames"] > longest["num_frames"]
            ):
                longest = candidate
        run_start = None
        run_rows = []
    return longest


def summarize_visibility(rows):
    """汇总帧级、episode 级和连续遮挡证据。"""
    rows = sorted(
        rows,
        key=lambda row: (row["episode_idx"], row["step_idx"]),
    )
    if not rows:
        raise ValueError("可见性行不能为空")
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[row["episode_idx"]].append(row)

    episode_longest_runs = {
        episode_idx: longest_nonclear_run(episode_idx, episode_rows)
        for episode_idx, episode_rows in by_episode.items()
    }
    candidates = [
        run for run in episode_longest_runs.values() if run is not None
    ]
    longest = max(
        candidates,
        key=lambda run: run["num_frames"],
        default=None,
    )

    ratios = [row["block_visibility_ratio"] for row in rows]
    group_counts = Counter(row["visibility_group"] for row in rows)
    initial_counts = Counter(
        episode_rows[0]["visibility_group"]
        for episode_rows in by_episode.values()
    )
    terminal_counts = Counter(
        episode_rows[-1]["visibility_group"]
        for episode_rows in by_episode.values()
    )
    initial_ratios = [
        episode_rows[0]["block_visibility_ratio"]
        for episode_rows in by_episode.values()
    ]
    terminal_ratios = [
        episode_rows[-1]["block_visibility_ratio"]
        for episode_rows in by_episode.values()
    ]
    per_episode = []
    for episode_idx, episode_rows in by_episode.items():
        episode_counts = Counter(
            row["visibility_group"] for row in episode_rows
        )
        episode_ratios = [
            row["block_visibility_ratio"] for row in episode_rows
        ]
        per_episode.append(
            {
                "episode_idx": episode_idx,
                "num_frames": len(episode_rows),
                "visibility_group_counts": {
                    group: episode_counts[group]
                    for group in ("clear", "partial", "severe")
                },
                "visibility_ratio_stats": numeric_stats(episode_ratios),
                "initial_visibility_group": (
                    episode_rows[0]["visibility_group"]
                ),
                "terminal_visibility_group": (
                    episode_rows[-1]["visibility_group"]
                ),
                "longest_nonclear_run": episode_longest_runs[episode_idx],
            }
        )

    lowest = sorted(
        rows,
        key=lambda row: row["block_visibility_ratio"],
    )[:10]
    return {
        "num_episodes": len(by_episode),
        "num_frames": len(rows),
        "replay_exact_match_count": sum(
            row["replay_exact_match"] for row in rows
        ),
        "replay_tolerance_match_count": sum(
            not row["replay_exact_match"]
            and row.get("replay_within_tolerance", False)
            for row in rows
        ),
        "replay_mismatch_count": 0,
        "visibility_group_counts": {
            group: group_counts[group]
            for group in ("clear", "partial", "severe")
        },
        "visibility_group_rates": {
            group: group_counts[group] / len(rows)
            for group in ("clear", "partial", "severe")
        },
        "episodes_with_nonclear": sum(
            any(row["visibility_group"] != "clear" for row in episode_rows)
            for episode_rows in by_episode.values()
        ),
        "episodes_with_severe": sum(
            any(row["visibility_group"] == "severe" for row in episode_rows)
            for episode_rows in by_episode.values()
        ),
        "initial_visibility_group_counts": dict(initial_counts),
        "terminal_visibility_group_counts": dict(terminal_counts),
        "initial_visibility_ratio_stats": numeric_stats(initial_ratios),
        "terminal_visibility_ratio_stats": numeric_stats(terminal_ratios),
        "visibility_ratio_stats": numeric_stats(ratios),
        "longest_nonclear_saved_frame_run": (
            0 if longest is None else longest["num_frames"]
        ),
        "longest_nonclear_run": longest,
        "per_episode": per_episode,
        "lowest_visibility_frames": [
            {
                "episode_idx": row["episode_idx"],
                "step_idx": row["step_idx"],
                "image_path": row["image_path"],
                "block_visibility_ratio": row["block_visibility_ratio"],
                "visibility_group": row["visibility_group"],
            }
            for row in lowest
        ],
    }


def replay_episode(config, manifest, summary, frame_rows):
    """按原始 seed 和控制顺序重放一条 episode。"""
    robot_visual_colors = {}

    def observe(frame):
        source = frame.source_row
        if not robot_visual_colors:
            robot_visual_colors.update(
                {
                    shape[1]: shape[7]
                    for shape in p.getVisualShapeData(frame.robot_id)
                }
            )
        replay_bgr, segmentation = capture_rgb_and_segmentation(
            config["camera"], frame.source_camera_eye
        )
        replay_check = validate_replay_image(
            source["image_path"], replay_bgr
        )
        visible_pixels = count_block_pixels(segmentation, frame.block_id)
        for link_index, rgba in robot_visual_colors.items():
            p.changeVisualShape(
                frame.robot_id,
                link_index,
                rgbaColor=[*rgba[:3], 0.0],
            )
        try:
            _, reference_segmentation = capture_rgb_and_segmentation(
                config["camera"], frame.source_camera_eye
            )
            reference_pixels = count_block_pixels(
                reference_segmentation, frame.block_id
            )
        finally:
            for link_index, rgba in robot_visual_colors.items():
                p.changeVisualShape(
                    frame.robot_id, link_index, rgbaColor=rgba
                )
        if reference_pixels <= 0 or visible_pixels > reference_pixels:
            raise ReplayValidationError(
                "invalid_visibility_pixels",
                episode_idx=source["episode_idx"],
                step_idx=source["step_idx"],
                visible_block_pixels=visible_pixels,
                reference_block_pixels=reference_pixels,
            )
        return {
            "schema_version": "visibility_audit_v1",
            "episode_idx": source["episode_idx"],
            "random_seed": source["random_seed"],
            "step_idx": source["step_idx"],
            "image_path": source["image_path"],
            "visible_block_pixels": visible_pixels,
            "reference_block_pixels": reference_pixels,
            **replay_check,
        }

    observations = replay_episode_frames(
        config, manifest, summary, frame_rows, observe
    )
    return build_visibility_rows(observations)


def write_json(path, payload):
    """稳定写入可读 JSON。"""
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_visibility_audit(dataset_dir, output_dir=None):
    """逐 episode 重放并原子发布可信可见性标签。"""
    dataset_dir = Path(dataset_dir)
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else dataset_dir / "visibility_audit_v1"
    )
    if output_dir.exists():
        raise FileExistsError(f"审计输出目录已存在: {output_dir}")
    try:
        manifest, config, frames, summaries = load_audit_inputs(dataset_dir)
        indexed_frames = defaultdict(list)
        for row in frames:
            indexed_frames[row["episode_idx"]].append(row)
        if len({row["episode_idx"] for row in summaries}) != len(summaries):
            raise ReplayValidationError("duplicate_episode_summary")

        all_rows = []
        for summary in sorted(
            summaries,
            key=lambda row: row["episode_idx"],
        ):
            all_rows.extend(
                replay_episode(
                    config,
                    manifest,
                    summary,
                    indexed_frames[summary["episode_idx"]],
                )
            )
        if len(all_rows) != len(frames):
            raise ReplayValidationError(
                "unconsumed_frame_rows",
                expected_frames=len(frames),
                replayed_frames=len(all_rows),
            )
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=False)
        evidence = {
            "schema_version": "visibility_audit_v1",
            "passed": False,
            "reason": (
                exc.reason
                if isinstance(exc, ReplayValidationError)
                else type(exc).__name__
            ),
            "error": repr(exc),
            "evidence": (
                exc.evidence
                if isinstance(exc, ReplayValidationError)
                else {}
            ),
        }
        write_json(output_dir / "visibility_audit_failure.json", evidence)
        raise

    summary = {
        "schema_version": "visibility_audit_v1",
        "dataset_dir": str(dataset_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "replay_validation": {"passed": True},
        **summarize_visibility(all_rows),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temp_dir:
        staging = Path(temp_dir) / output_dir.name
        staging.mkdir()
        with (staging / "frame_visibility.jsonl").open(
            "w",
            encoding="utf-8",
        ) as handle:
            for row in all_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(
            staging / "visibility_audit_summary.json",
            summary,
        )
        staging.replace(output_dir)
    return summary


def main():
    """CLI：确定性重放版本化 expert 数据集并发布可见性报告。"""
    parser = argparse.ArgumentParser(
        description="确定性重放 expert 数据并审计红块可见率"
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    summary = run_visibility_audit(
        args.dataset_dir,
        args.output_dir,
    )
    print(
        "可见性审计完成："
        f"episode={summary['num_episodes']}，"
        f"frames={summary['num_frames']}，"
        f"groups={summary['visibility_group_counts']}"
    )


if __name__ == "__main__":
    main()
