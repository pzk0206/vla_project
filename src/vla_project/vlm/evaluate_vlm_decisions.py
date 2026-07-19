"""离线比较 Qwen 视觉方向预测与 heuristic 标准标签。

脚本只读取已经生成的固定图片，不启动 PyBullet 控制。每完成一个样本就立即追加
一行预测结果；重复运行同一 run 时会跳过已有 sample_id，避免网络中断后重复付费。
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2

from vla_project.vlm.collect_vlm_eval_samples import VALID_DIRECTIONS, read_jsonl
from vla_project.simulation.control_arm import CONFIG_PATH, load_config
from vla_project.simulation.stage3_probe import (
    build_vision_direction_prompt,
    call_openai_compatible_api,
    map_screen_to_world,
    parse_screen_direction,
)


DIRECTION_ORDER = ("left", "right", "front", "back", "stop")
INVALID_PREDICTION = "__invalid__"

def build_offline_prompt(sample):
    """只用任务指令构造方向 prompt，不读取标准标签或仿真真值。

    相机到世界坐标的映射来自项目固定相机布置，不来自当前样本的 block_pos、
    ee_pos 或 expected_direction，因此不会把答案泄露给模型。
    """
    return build_vision_direction_prompt(
        sample["instruction"],
        include_stop=False,
    )


def summarize_predictions(predictions):
    """汇总合法率、准确率、分类指标、混淆矩阵、延迟和错误类型。"""
    predictions = list(predictions)
    total = len(predictions)
    confusion_matrix = {
        actual: {
            predicted: 0
            for predicted in (*DIRECTION_ORDER, INVALID_PREDICTION)
        }
        for actual in DIRECTION_ORDER
    }
    valid_output_count = 0
    exact_matches = 0
    latencies = []
    error_types = Counter()

    for row in predictions:
        actual = row["expected_direction"]
        predicted = row.get("predicted_direction")
        predicted_bucket = (
            predicted if predicted in VALID_DIRECTIONS else INVALID_PREDICTION
        )
        confusion_matrix[actual][predicted_bucket] += 1
        if predicted in VALID_DIRECTIONS:
            valid_output_count += 1
        if predicted == actual:
            exact_matches += 1
        if isinstance(row.get("latency_seconds"), (int, float)):
            latencies.append(row["latency_seconds"])
        if row.get("error_type"):
            error_types[row["error_type"]] += 1

    per_direction = {}
    for direction in DIRECTION_ORDER:
        true_positive = confusion_matrix[direction][direction]
        actual_count = sum(confusion_matrix[direction].values())
        predicted_count = sum(
            confusion_matrix[actual][direction] for actual in DIRECTION_ORDER
        )
        precision = (
            true_positive / predicted_count if predicted_count else None
        )
        recall = true_positive / actual_count if actual_count else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        per_direction[direction] = {
            "support": actual_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "total_samples": total,
        "valid_output_count": valid_output_count,
        "valid_output_rate": valid_output_count / total if total else 0.0,
        "exact_match_count": exact_matches,
        "exact_match_accuracy": exact_matches / total if total else 0.0,
        "per_direction": per_direction,
        "confusion_matrix": confusion_matrix,
        "average_api_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "error_type_counts": dict(error_types),
    }


def append_jsonl(path, row):
    """立即追加并刷新一条结果，减少中断时的数据和费用损失。"""
    with Path(path).open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_file.flush()


def load_existing_predictions(path):
    """按 sample_id 读取已完成结果，供断点续跑跳过。"""
    path = Path(path)
    if not path.exists():
        return {}
    return {row["sample_id"]: row for row in read_jsonl(path)}


def evaluate_offline(config, limit=None):
    """评估固定样本并返回运行目录与最新汇总。"""
    evaluation = config["vlm_evaluation"]
    sample_dir = Path(evaluation["sample_output_dir"])
    samples = read_jsonl(sample_dir / "samples.jsonl")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit 必须大于 0")
        samples = samples[:limit]

    run_dir = Path(evaluation["run_output_dir"]) / evaluation["offline_run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    summary_path = run_dir / "summary.json"
    existing = load_existing_predictions(predictions_path)

    for sample in samples:
        if sample["sample_id"] in existing:
            continue

        image_bgr = cv2.imread(sample["image_path"])
        started_at = time.perf_counter()
        predicted_screen_direction = None
        predicted_direction = None
        raw_response = None
        error_type = None
        error_message = None
        try:
            predicted_screen_direction, raw_response = call_openai_compatible_api(
                image_bgr,
                config["probe"]["api"],
                prompt_text=build_offline_prompt(sample),
                response_parser=parse_screen_direction,
            )
            predicted_direction = map_screen_to_world(
                predicted_screen_direction
            )
        except Exception as exc:
            error_type = type(exc).__name__
            error_message = str(exc)
        latency_seconds = time.perf_counter() - started_at

        result = {
            "sample_id": sample["sample_id"],
            "image_path": sample["image_path"],
            "instruction": sample["instruction"],
            "expected_direction": sample["expected_direction"],
            "predicted_screen_direction": predicted_screen_direction,
            "predicted_direction": predicted_direction,
            "raw_response": raw_response,
            "is_exact_match": predicted_direction == sample["expected_direction"],
            "latency_seconds": latency_seconds,
            "error_type": error_type,
            "error_message": error_message,
        }
        append_jsonl(predictions_path, result)
        existing[sample["sample_id"]] = result

    summary = summarize_predictions(existing.values())
    summary["run_dir"] = str(run_dir)
    summary["sample_manifest"] = str(sample_dir / "samples.jsonl")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir, summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只评估 manifest 开头的 N 个样本，用于单请求和 10 样本检查。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(CONFIG_PATH)
    run_dir, summary = evaluate_offline(config, limit=args.limit)
    print(f"✅ 离线评估结果目录: {run_dir}")
    print(
        f"合法输出率: {summary['valid_output_rate']:.1%}，"
        f"exact-match accuracy: {summary['exact_match_accuracy']:.1%}"
    )


if __name__ == "__main__":
    main()
