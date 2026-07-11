import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

from control_arm import CONFIG_PATH, load_config
from stage3_probe import run_probe_episode


def cleanup_success_images(episode_dir, success, enabled):
    """只在启用策略且 episode 成功时删除步骤图片。"""
    if success and enabled:
        for image_path in Path(episode_dir).glob("probe_step_*.jpg"):
            image_path.unlink()


def aggregate_probe_summaries(summaries, config_snapshot):
    """汇总一批 episode 的成功率、距离、步数和失败类型。"""
    total = len(summaries)
    successes = sum(row["success"] for row in summaries)
    final_distances = [row["final_distance"] for row in summaries if row["final_distance"] is not None]
    control_steps = [row["num_control_steps"] for row in summaries if row["num_control_steps"] > 0]
    directions = Counter()
    for row in summaries:
        directions.update(row["direction_counts"])
    reasons = Counter(row["termination_reason"] for row in summaries)
    return {
        "num_episodes": total,
        "success_count": successes,
        "failure_count": total - successes,
        "error_count": reasons["error"],
        "success_rate": successes / total if total else 0.0,
        "final_distance_mean": statistics.mean(final_distances) if final_distances else None,
        "final_distance_median": statistics.median(final_distances) if final_distances else None,
        "final_distance_max": max(final_distances) if final_distances else None,
        "control_steps_mean": statistics.mean(control_steps) if control_steps else None,
        "control_steps_median": statistics.median(control_steps) if control_steps else None,
        "termination_reason_counts": dict(reasons),
        "direction_counts": dict(directions),
        "failed_episode_indices": [row["episode_idx"] for row in summaries if not row["success"]],
        "config_snapshot": config_snapshot,
    }


def make_error_summary(episode_idx, random_seed, trace_path, exc):
    return {
        "episode_idx": episode_idx,
        "random_seed": random_seed,
        "success": False,
        "termination_reason": "error",
        "num_control_steps": 0,
        "initial_distance": None,
        "final_distance": None,
        "final_block_pos": None,
        "direction_counts": {},
        "distance_increase_steps": 0,
        "trace_path": str(trace_path),
        "error": f"{type(exc).__name__}: {exc}",
    }


def run_batch(config, num_episodes=None, run_name=None):
    """运行一批 heuristic probe 并返回批次目录与汇总。"""
    if config["probe"]["mode"].lower() != "heuristic":
        raise ValueError("批量评估当前只允许 probe.mode: heuristic")
    evaluation = config["probe_evaluation"]
    count = num_episodes if num_episodes is not None else evaluation["num_episodes"]
    name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    batch_dir = Path(evaluation["output_dir"]) / name
    batch_dir.mkdir(parents=True, exist_ok=False)
    summary_path = batch_dir / "episode_summary.jsonl"
    summaries = []

    for episode_idx in range(count):
        seed = evaluation["random_seed"] + episode_idx
        episode_dir = batch_dir / f"episode_{episode_idx:03d}"
        trace_path = episode_dir / "probe_trace.jsonl"
        try:
            summary = run_probe_episode(config, episode_idx, episode_dir, seed)
        except Exception as exc:
            summary = make_error_summary(episode_idx, seed, trace_path, exc)
        cleanup_success_images(
            episode_dir,
            summary["success"],
            evaluation["save_failure_images_only"],
        )
        summaries.append(summary)
        with summary_path.open("a", encoding="utf-8") as summary_file:
            summary_file.write(json.dumps(summary, ensure_ascii=False) + "\n")

    config_snapshot = {
        "probe": config["probe"],
        "probe_evaluation": evaluation,
    }
    aggregate = aggregate_probe_summaries(summaries, config_snapshot)
    aggregate_path = batch_dir / "probe_eval_summary.json"
    aggregate_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return batch_dir, aggregate


def main():
    config = load_config(CONFIG_PATH)
    batch_dir, summary = run_batch(config)
    print(f"✅ 批量评估完成: {batch_dir}")
    print(
        f"成功率 {summary['success_rate']:.1%} "
        f"({summary['success_count']}/{summary['num_episodes']})"
    )


if __name__ == "__main__":
    main()
