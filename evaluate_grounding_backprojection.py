"""离线评估 Qwen 红块框中心到工作平面世界坐标的反投影误差。"""

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from camera_geometry import (
    normalized_box_center_to_pixel,
    pixel_to_world_on_plane,
)


def read_jsonl(path):
    """读取非空 JSONL 行。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _index_unique(rows, label):
    indexed = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{label} 缺少非空 sample_id")
        if sample_id in indexed:
            raise ValueError(f"{label} sample_id 重复: {sample_id}")
        indexed[sample_id] = row
    return indexed


def evaluate_rows(predictions, diagnostics, plane_z=0.0):
    """连接 grounding 和诊断真值，并逐样本计算 xy 定位误差。"""
    prediction_index = _index_unique(predictions, "grounding prediction")
    diagnostic_index = _index_unique(diagnostics, "diagnostics")
    missing = sorted(set(prediction_index) - set(diagnostic_index))
    if missing:
        raise ValueError(f"grounding prediction 缺少诊断: {missing}")

    results = []
    for sample_id, prediction in prediction_index.items():
        diagnostic = diagnostic_index[sample_id]
        boxes = prediction.get("boxes")
        base_result = {
            "sample_id": sample_id,
            "red_block_box": boxes.get("red_block") if boxes else None,
            "box_center_pixel": None,
            "predicted_target_world": None,
            "true_block_pos": diagnostic["block_pos"],
            "localization_error_xy": None,
            "signed_error_x": None,
            "signed_error_y": None,
            "latency_seconds": prediction.get("latency_seconds"),
            "error_type": prediction.get("error_type"),
            "error_message": prediction.get("error_message"),
        }
        if not boxes or not boxes.get("red_block"):
            base_result["error_type"] = (
                base_result["error_type"] or "MissingGroundingBoxError"
            )
            base_result["error_message"] = (
                base_result["error_message"] or "red_block box 缺失"
            )
            results.append(base_result)
            continue
        try:
            pixel = normalized_box_center_to_pixel(
                boxes["red_block"],
                diagnostic["image_width"],
                diagnostic["image_height"],
            )
            predicted = pixel_to_world_on_plane(
                pixel,
                diagnostic["image_width"],
                diagnostic["image_height"],
                diagnostic["view_matrix"],
                diagnostic["projection_matrix"],
                plane_z=plane_z,
            )
            true_block = diagnostic["block_pos"]
            signed_x = predicted[0] - true_block[0]
            signed_y = predicted[1] - true_block[1]
            base_result.update(
                {
                    "box_center_pixel": list(pixel),
                    "predicted_target_world": predicted,
                    "localization_error_xy": math.hypot(signed_x, signed_y),
                    "signed_error_x": signed_x,
                    "signed_error_y": signed_y,
                    "error_type": None,
                    "error_message": None,
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            base_result["error_type"] = type(exc).__name__
            base_result["error_message"] = str(exc)
        results.append(base_result)
    return results


def summarize_results(results):
    """汇总有效率、xy 误差和系统性有符号偏差。"""
    valid = [row for row in results if row["localization_error_xy"] is not None]
    errors = [row["localization_error_xy"] for row in valid]
    error_counts = Counter(
        row["error_type"] for row in results if row["error_type"]
    )
    return {
        "num_samples": len(results),
        "num_valid": len(valid),
        "num_failed": len(results) - len(valid),
        "valid_rate": len(valid) / len(results) if results else 0.0,
        "localization_error_xy_mean": statistics.fmean(errors) if errors else None,
        "localization_error_xy_median": (
            statistics.median(errors) if errors else None
        ),
        "localization_error_xy_max": max(errors) if errors else None,
        "signed_error_x_mean": (
            statistics.fmean(row["signed_error_x"] for row in valid)
            if valid
            else None
        ),
        "signed_error_y_mean": (
            statistics.fmean(row["signed_error_y"] for row in valid)
            if valid
            else None
        ),
        "error_type_counts": dict(sorted(error_counts.items())),
    }


def write_jsonl(path, rows):
    """覆盖写入逐样本 JSONL 结果。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plane-z", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = read_jsonl(args.predictions)
    diagnostics = read_jsonl(args.diagnostics)
    results = evaluate_rows(predictions, diagnostics, plane_z=args.plane_z)
    summary = summarize_results(results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "backprojection_results.jsonl", results)
    with (output_dir / "backprojection_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
