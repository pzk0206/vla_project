"""Stage 3 批量评估入口。

这份脚本不重新实现机械臂控制，而是重复调用 stage3_probe.py 中的
run_probe_episode()。整体数据流是：

读取 YAML 配置 -> 创建批次目录 -> 逐个运行 episode -> 写 episode 摘要
-> 聚合成功率/距离/步数 -> 写整批 summary。

这样单次 probe 和批量评估使用完全相同的控制逻辑，后续替换决策器时，
评估结果不会因为两套控制实现不同而失去可比性。
"""

import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

from vla_project.simulation.control_arm import CONFIG_PATH, load_config
from vla_project.simulation.stage3_probe import run_probe_episode


def cleanup_success_images(episode_dir, success, enabled):
    """成功时删除步骤图片，失败时保留现场。

    trace 的体积很小，所有 episode 都保留；图片数量大，只保留失败样本。
    这既控制磁盘占用，也确保 failure mode analysis 仍有视觉证据。
    """
    if success and enabled:
        for image_path in Path(episode_dir).glob("probe_step_*.jpg"):
            image_path.unlink()


def aggregate_probe_summaries(summaries, config_snapshot):
    """把 episode 级结果聚合成一份批次报告。

    summaries 中一行对应一个 episode。异常 episode 也留在分母中，因此
    success_rate 反映的是整套系统的真实可用率，而不是只统计正常运行样本。
    没有有效距离的异常样本不参与距离均值，避免 None 污染数值统计。
    """
    total = len(summaries)
    successes = sum(row["success"] for row in summaries)
    final_distances = [row["final_distance"] for row in summaries if row["final_distance"] is not None]
    control_steps = [row["num_control_steps"] for row in summaries if row["num_control_steps"] > 0]
    # 把每个 episode 的方向计数再次合并，观察策略是否长期偏向某个方向。
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
    """把未预期异常转换成普通 episode 摘要。

    批量实验中单次异常不应该让剩余 episode 全部丢失，所以这里保存异常
    类型和消息，交给 run_batch 继续写 JSONL 并运行下一条。
    """
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
    """运行一批 heuristic probe，并返回批次目录与汇总字典。

    Args:
        config: sim_config.yaml 加载后的完整配置。
        num_episodes: 测试时可覆盖 YAML 次数；None 表示使用正式配置。
        run_name: 测试时可固定目录名；None 表示自动生成时间戳。

    Returns:
        (batch_dir, aggregate)，分别是结果目录 Path 和整批统计字典。

    Side effects:
        创建批次/episode 目录，写 trace、episode_summary.jsonl 和
        probe_eval_summary.json，并按配置清理成功 episode 图片。
    """
    if config["probe"]["mode"].lower() != "heuristic":
        raise ValueError("批量评估当前只允许 probe.mode: heuristic")
    evaluation = config["probe_evaluation"]
    count = num_episodes if num_episodes is not None else evaluation["num_episodes"]
    # 每次正式运行使用独立时间戳目录，避免覆盖上一次实验结果。
    name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    batch_dir = Path(evaluation["output_dir"]) / name
    batch_dir.mkdir(parents=True, exist_ok=False)
    summary_path = batch_dir / "episode_summary.jsonl"
    summaries = []

    for episode_idx in range(count):
        # seed 随 episode 递增：同一基础 seed 可复现完全相同的一组随机目标。
        seed = evaluation["random_seed"] + episode_idx
        episode_dir = batch_dir / f"episode_{episode_idx:03d}"
        trace_path = episode_dir / "probe_trace.jsonl"
        try:
            summary = run_probe_episode(config, episode_idx, episode_dir, seed)
        except Exception as exc:
            # 错误被记录而不是继续抛出，保证后续 episode 仍能完成。
            summary = make_error_summary(episode_idx, seed, trace_path, exc)
        cleanup_success_images(
            episode_dir,
            summary["success"],
            evaluation["save_failure_images_only"],
        )
        summaries.append(summary)
        # 一完成就追加一行；即使进程稍后中断，已完成结果也不会丢失。
        with summary_path.open("a", encoding="utf-8") as summary_file:
            summary_file.write(json.dumps(summary, ensure_ascii=False) + "\n")

    # 保存关键配置快照，后续比较实验时能知道结果由哪些参数产生。
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
    """命令行入口：读取 YAML，完成整批评估并打印最重要的结果。"""
    config = load_config(CONFIG_PATH)
    batch_dir, summary = run_batch(config)
    print(f"✅ 批量评估完成: {batch_dir}")
    print(
        f"成功率 {summary['success_rate']:.1%} "
        f"({summary['success_count']}/{summary['num_episodes']})"
    )


if __name__ == "__main__":
    main()
