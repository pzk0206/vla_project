"""冻结清晰 grounding 的固定偏差，并在独立反投影结果上验证。"""

import json
import math
import statistics
from pathlib import Path


VALID_VISIBILITY_GROUPS = {"clear", "partial", "severe"}
PASS_THRESHOLD_METERS = 0.03


def read_jsonl(path):
    """读取 JSONL 中的非空行。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _index_unique(rows, label):
    """按 sample_id 建立索引，并验证 ID 与可见率分组。"""
    indexed = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{label} 缺少非空 sample_id")
        if sample_id in indexed:
            raise ValueError(f"{label} sample_id 重复: {sample_id}")
        group = row.get("visibility_group")
        if group not in VALID_VISIBILITY_GROUPS:
            raise ValueError(f"{label} visibility_group 非法: {sample_id}")
        indexed[sample_id] = row
    return indexed


def _finite_number(value, field_name, sample_id):
    """把普通整数或浮点数规范化为有限浮点数。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{sample_id} {field_name} 必须是有限数值")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{sample_id} {field_name} 必须是有限数值")
    return value


def _validated_truth_xy(row):
    """读取离线评分真值；真值不参与模型预测或校准应用。"""
    truth = row.get("true_block_pos")
    if not isinstance(truth, list) or len(truth) < 2:
        raise ValueError(f"{row['sample_id']} true_block_pos 非法")
    return (
        _finite_number(truth[0], "true_block_pos.x", row["sample_id"]),
        _finite_number(truth[1], "true_block_pos.y", row["sample_id"]),
    )


def _validated_prediction_xyz(row):
    """验证存在的世界坐标预测；明确失败记录允许预测为 None。"""
    predicted = row.get("predicted_target_world")
    if predicted is None:
        return None
    if not isinstance(predicted, list) or len(predicted) < 3:
        raise ValueError(f"{row['sample_id']} predicted_target_world 非法")
    return [
        _finite_number(value, "predicted_target_world", row["sample_id"])
        for value in predicted[:3]
    ]


def fit_clear_calibration(rows, source_path):
    """只用定位有效的 clear 校准样本拟合一次固定 XY 补偿。"""
    indexed = _index_unique(rows, "calibration")
    for row in indexed.values():
        _validated_truth_xy(row)
        _validated_prediction_xyz(row)

    clear_rows = [
        row
        for row in indexed.values()
        if row["visibility_group"] == "clear"
    ]
    if not clear_rows:
        raise ValueError("clear 校准样本不能为空")
    for row in clear_rows:
        if row.get("localization_error_xy") is None:
            raise ValueError(f"clear 校准样本定位无效: {row['sample_id']}")

    signed_x = [
        _finite_number(
            row.get("signed_error_x"),
            "signed_error_x",
            row["sample_id"],
        )
        for row in clear_rows
    ]
    signed_y = [
        _finite_number(
            row.get("signed_error_y"),
            "signed_error_y",
            row["sample_id"],
        )
        for row in clear_rows
    ]
    mean_x = statistics.fmean(signed_x)
    mean_y = statistics.fmean(signed_y)
    return {
        "calibration_source": str(source_path),
        "num_clear_calibration_samples": len(clear_rows),
        "calibration_sample_ids": sorted(
            row["sample_id"] for row in clear_rows
        ),
        "signed_error_x_mean": mean_x,
        "signed_error_y_mean": mean_y,
        "correction_x": -mean_x,
        "correction_y": -mean_y,
    }


def apply_frozen_calibration(calibration, validation_rows):
    """将已冻结的 XY 补偿应用到独立验证集，不重新拟合参数。"""
    validation = _index_unique(validation_rows, "validation")
    calibration_ids = set(calibration.get("calibration_sample_ids", []))
    overlap = sorted(calibration_ids & set(validation))
    if overlap:
        raise ValueError(f"校准集和验证集 sample_id 重叠: {overlap}")

    correction_x = _finite_number(
        calibration.get("correction_x"),
        "correction_x",
        "calibration",
    )
    correction_y = _finite_number(
        calibration.get("correction_y"),
        "correction_y",
        "calibration",
    )
    corrected_rows = []
    for row in validation.values():
        true_x, true_y = _validated_truth_xy(row)
        predicted_values = _validated_prediction_xyz(row)
        corrected = dict(row)
        corrected.update(
            {
                "correction_x_applied": correction_x,
                "correction_y_applied": correction_y,
                "corrected_target_world": None,
                "corrected_signed_error_x": None,
                "corrected_signed_error_y": None,
                "corrected_error_xy": None,
            }
        )
        if predicted_values is not None:
            corrected_x = predicted_values[0] + correction_x
            corrected_y = predicted_values[1] + correction_y
            signed_x = corrected_x - true_x
            signed_y = corrected_y - true_y
            corrected.update(
                {
                    "corrected_target_world": [
                        corrected_x,
                        corrected_y,
                        predicted_values[2],
                    ],
                    "corrected_signed_error_x": signed_x,
                    "corrected_signed_error_y": signed_y,
                    "corrected_error_xy": math.hypot(signed_x, signed_y),
                }
            )
        corrected_rows.append(corrected)
    return corrected_rows
