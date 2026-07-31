"""只读审计 expert_v1 动作表示与候选离散化。"""

import math
from collections import Counter

import numpy as np


SCHEMA_VERSION = "action_tokenization_audit_v1"
ALLOWED_VISIBILITY_GROUPS = {"clear", "partial", "severe"}
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
RECOMMENDATION_THRESHOLDS = {
    "minimum_nonempty_bin_occupancy": 0.90,
    "minimum_nonempty_bin_count": 20,
    "maximum_normalized_p95_reconstruction_error": 0.05,
}


class AuditValidationError(ValueError):
    """携带稳定原因和证据的审计输入错误。"""

    def __init__(self, reason, **evidence):
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence


def _validate_action(action, key):
    if not isinstance(action, list) or len(action) != 9:
        raise AuditValidationError("invalid_action", key=key)
    for index, value in enumerate(action):
        if isinstance(value, bool) and index < 8:
            raise AuditValidationError(
                "invalid_action",
                key=key,
                action_index=index,
            )
        if not isinstance(value, (int, float)):
            raise AuditValidationError(
                "invalid_action",
                key=key,
                action_index=index,
            )
        if not math.isfinite(value):
            raise AuditValidationError(
                "non_finite_action",
                key=key,
                action_index=index,
            )
    if action[8] not in (0, 1, False, True):
        raise AuditValidationError("invalid_action", key=key, action_index=8)
    return [float(value) for value in action[:8]] + [int(action[8])]


def _validate_episode_termination(rows):
    episode_rows = {}
    for row in rows:
        episode_rows.setdefault(row["episode_idx"], []).append(row)
    for episode_idx, current_rows in episode_rows.items():
        terminated = [row["terminate"] for row in current_rows]
        if terminated != [0] * (len(current_rows) - 1) + [1]:
            raise AuditValidationError(
                "invalid_episode_termination",
                episode_idx=episode_idx,
                terminate_values=terminated,
            )


def numeric_stats(values):
    """计算一维有限数值序列的稳定描述统计。"""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise AuditValidationError("invalid_numeric_series")
    percentile_values = np.percentile(
        array,
        PERCENTILES,
        method="linear",
    )
    result = {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "non_finite_count": 0,
    }
    result.update(
        {
            f"p{percentile:02d}": float(value)
            for percentile, value in zip(PERCENTILES, percentile_values)
        }
    )
    return result


def _joint_statistics(rows, field):
    vectors = [row[field] for row in rows if row[field] is not None]
    if not vectors:
        return []
    return [
        numeric_stats([vector[joint_index] for vector in vectors])
        for joint_index in range(7)
    ]


def _summarize_rows(rows, visibility_group=None):
    transition_count = sum(row["delta_q"] is not None for row in rows)
    summary = {
        "num_frames": len(rows),
        "episode_count": len({row["episode_idx"] for row in rows}),
        "num_transitions": transition_count,
        "absolute_q": _joint_statistics(rows, "q_target"),
        "delta_q": _joint_statistics(rows, "delta_q"),
        "delta_q_per_step": _joint_statistics(rows, "delta_q_per_step"),
    }
    if visibility_group == "severe":
        summary["insufficient_for_generalization"] = len(rows) < 20
    return summary


def _string_counts(values):
    return {
        str(value): count
        for value, count in sorted(Counter(values).items())
    }


def summarize_action_analysis(rows):
    """汇总动作、特殊维度、step gap 与可见性条件分布。"""
    global_summary = _summarize_rows(rows)
    gripper_values = [row["gripper"] for row in rows]
    terminate_values = [row["terminate"] for row in rows]
    step_gaps = [row["step_gap"] for row in rows if row["step_gap"] is not None]
    global_summary.update(
        {
            "num_episodes": len({row["episode_idx"] for row in rows}),
            "gripper": {
                "counts": _string_counts(gripper_values),
                "is_constant": len(set(gripper_values)) <= 1,
            },
            "terminate": {
                "counts": _string_counts(terminate_values),
                "rate": (
                    sum(terminate_values) / len(terminate_values)
                    if terminate_values
                    else 0.0
                ),
            },
            "step_gap": {"counts": _string_counts(step_gaps)},
            "by_visibility": {
                group: _summarize_rows(
                    [row for row in rows if row["visibility_group"] == group],
                    group,
                )
                for group in ("clear", "partial", "severe")
            },
        }
    )
    return global_summary


def evaluate_binning(values, method, num_bins):
    """模拟一维分箱、重建并返回数据支持度与误差。"""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise AuditValidationError("invalid_binning_series")
    if method not in {"uniform_width", "quantile"}:
        raise ValueError(f"未知分箱方法: {method}")
    if not isinstance(num_bins, int) or num_bins <= 1:
        raise ValueError("num_bins 必须是大于1的整数")

    if float(array.min()) == float(array.max()):
        edges = np.asarray([float(array.min()), float(array.max())])
        assignments = np.zeros(array.size, dtype=int)
        reconstruction_array = np.asarray([float(array.min())])
        reported_reconstruction_values = [float(array.min())]
    else:
        if method == "uniform_width":
            edges = np.linspace(array.min(), array.max(), num_bins + 1)
        else:
            edges = np.unique(
                np.quantile(
                    array,
                    np.linspace(0.0, 1.0, num_bins + 1),
                    method="linear",
                )
            )
        assignments = np.searchsorted(
            edges[1:-1],
            array,
            side="right",
        )
        interval_count = len(edges) - 1
        counts = np.bincount(assignments, minlength=interval_count)
        if method == "uniform_width":
            reconstruction_array = (edges[:-1] + edges[1:]) / 2.0
            reported_reconstruction_values = [
                float(value) for value in reconstruction_array
            ]
        else:
            reconstruction_array = np.asarray(
                [
                    (
                        np.median(array[assignments == index])
                        if counts[index] > 0
                        else (edges[index] + edges[index + 1]) / 2.0
                    )
                    for index in range(interval_count)
                ],
                dtype=float,
            )
            reported_reconstruction_values = [
                (
                    float(reconstruction_array[index])
                    if counts[index] > 0
                    else None
                )
                for index in range(interval_count)
            ]

    effective_num_bins = len(reconstruction_array)
    counts = np.bincount(assignments, minlength=effective_num_bins)
    nonempty_counts = counts[counts > 0]
    reconstructed = reconstruction_array[assignments]
    errors = np.abs(array - reconstructed)
    probabilities = nonempty_counts / array.size
    normalized_entropy = float(
        -np.sum(probabilities * np.log(probabilities)) / np.log(num_bins)
    )
    p01, p99 = np.percentile(array, [1, 99], method="linear")
    robust_range = float(p99 - p01)
    mae = float(errors.mean())
    p95_error = float(np.percentile(errors, 95, method="linear"))
    return {
        "method": method,
        "requested_num_bins": num_bins,
        "effective_num_bins": effective_num_bins,
        "edges": [float(value) for value in edges],
        "reconstruction_values": reported_reconstruction_values,
        "counts": [int(value) for value in counts],
        "nonempty_bin_occupancy": float(len(nonempty_counts) / num_bins),
        "minimum_nonempty_bin_count": int(nonempty_counts.min()),
        "maximum_class_fraction": float(nonempty_counts.max() / array.size),
        "normalized_entropy": normalized_entropy,
        "mae": mae,
        "p95_absolute_error": p95_error,
        "max_absolute_error": float(errors.max()),
        "robust_range_p99_p01": robust_range,
        "normalized_mae": None if robust_range == 0.0 else mae / robust_range,
        "normalized_p95_reconstruction_error": (
            None if robust_range == 0.0 else p95_error / robust_range
        ),
        "eligible_metrics": robust_range > 0.0,
        "metric_failure_reason": (
            "zero_robust_range" if robust_range == 0.0 else None
        ),
    }


def _candidate_worst_joint_metrics(per_joint):
    normalized_errors = [
        result["normalized_p95_reconstruction_error"]
        for result in per_joint
        if result["normalized_p95_reconstruction_error"] is not None
    ]
    return {
        "nonempty_bin_occupancy": min(
            result["nonempty_bin_occupancy"] for result in per_joint
        ),
        "minimum_nonempty_bin_count": min(
            result["minimum_nonempty_bin_count"] for result in per_joint
        ),
        "normalized_p95_reconstruction_error": (
            max(normalized_errors) if normalized_errors else None
        ),
        "zero_robust_range": any(
            not result["eligible_metrics"] for result in per_joint
        ),
    }


def evaluate_candidates(rows):
    """评估固定的12个绝对目标与目标差分候选。"""
    sources = {
        "absolute_q": [row["q_target"] for row in rows],
        "delta_q": [
            row["delta_q"] for row in rows if row["delta_q"] is not None
        ],
    }
    candidates = []
    for representation, vectors in sources.items():
        if not vectors:
            raise AuditValidationError(
                "missing_representation_samples",
                representation=representation,
            )
        for method in ("uniform_width", "quantile"):
            for num_bins in (16, 32, 64):
                per_joint = [
                    evaluate_binning(
                        [vector[joint_index] for vector in vectors],
                        method,
                        num_bins,
                    )
                    for joint_index in range(7)
                ]
                candidates.append(
                    {
                        "representation": representation,
                        "binning": method,
                        "num_bins": num_bins,
                        "per_joint": per_joint,
                        "worst_joint_metrics": (
                            _candidate_worst_joint_metrics(per_joint)
                        ),
                    }
                )
    return candidates


def _candidate_rejection_reasons(candidate, thresholds):
    metrics = candidate["worst_joint_metrics"]
    reasons = []
    if metrics.get("zero_robust_range"):
        reasons.append("zero_robust_range")
    if (
        metrics["nonempty_bin_occupancy"]
        < thresholds["minimum_nonempty_bin_occupancy"]
    ):
        reasons.append("occupancy_below_threshold")
    if (
        metrics["minimum_nonempty_bin_count"]
        < thresholds["minimum_nonempty_bin_count"]
    ):
        reasons.append("minimum_count_below_threshold")
    normalized_error = metrics["normalized_p95_reconstruction_error"]
    if (
        normalized_error is not None
        and normalized_error
        > thresholds["maximum_normalized_p95_reconstruction_error"]
    ):
        reasons.append("normalized_p95_above_threshold")
    return reasons


def _candidate_sort_key(candidate):
    metrics = candidate["worst_joint_metrics"]
    normalized_error = metrics["normalized_p95_reconstruction_error"]
    return (
        candidate["num_bins"],
        float("inf") if normalized_error is None else normalized_error,
        -metrics["minimum_nonempty_bin_count"],
        0 if candidate["binning"] == "uniform_width" else 1,
    )


def select_recommendations(candidates, thresholds=None):
    """按透明门槛为两类离散表示选择最小合格候选。"""
    thresholds = dict(thresholds or RECOMMENDATION_THRESHOLDS)
    result = {
        "rule_version": "action_tokenization_recommendation_v1",
        "thresholds": thresholds,
        "continuous_regression": {
            "included": True,
            "reason": "unquantized_baseline",
        },
    }
    for representation in ("absolute_q", "delta_q"):
        ranked = []
        for source in candidates:
            if source["representation"] != representation:
                continue
            candidate = dict(source)
            candidate["rejection_reasons"] = _candidate_rejection_reasons(
                candidate,
                thresholds,
            )
            candidate["eligible"] = not candidate["rejection_reasons"]
            ranked.append(candidate)
        ranked.sort(key=_candidate_sort_key)
        selected = next(
            (candidate for candidate in ranked if candidate["eligible"]),
            None,
        )
        result[representation] = {
            "status": "selected" if selected else "no_eligible_candidate",
            "candidate": selected,
            "ranked_candidates": ranked,
        }
    return result


def build_frame_analysis(manifest, trajectory_rows, visibility_rows):
    """把动作帧与可见性帧对齐，并计算 episode 内相邻目标差。"""
    if manifest.get("schema_version") != "expert_v1":
        raise AuditValidationError("unsupported_dataset_schema")
    if manifest.get("action_dim") != 9:
        raise AuditValidationError("unexpected_action_dim")
    visibility_by_key = {}
    for row in visibility_rows:
        key = (row["episode_idx"], row["step_idx"])
        if key in visibility_by_key:
            raise AuditValidationError("duplicate_visibility_key", key=key)
        visibility_by_key[key] = row
    previous_by_episode = {}
    seen_trajectory_keys = set()
    output = []
    for row in trajectory_rows:
        episode_idx = row["episode_idx"]
        step_idx = row["step_idx"]
        key = (episode_idx, step_idx)
        if key in seen_trajectory_keys:
            raise AuditValidationError("duplicate_trajectory_key", key=key)
        seen_trajectory_keys.add(key)
        visibility = visibility_by_key.pop(key, None)
        if visibility is None:
            raise AuditValidationError("missing_visibility_key", key=key)
        visibility_group = visibility.get("visibility_group")
        if visibility_group not in ALLOWED_VISIBILITY_GROUPS:
            raise AuditValidationError(
                "invalid_visibility_group",
                key=key,
                visibility_group=visibility_group,
            )
        action = _validate_action(row.get("action"), key)
        q_target = action[:7]
        previous = previous_by_episode.get(episode_idx)
        step_gap = (
            None if previous is None else step_idx - previous["step_idx"]
        )
        if step_gap is not None and step_gap <= 0:
            raise AuditValidationError("non_increasing_step", key=key)
        delta_q = (
            None
            if previous is None
            else [
                current - prior
                for current, prior in zip(q_target, previous["q_target"])
            ]
        )
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "episode_idx": episode_idx,
                "step_idx": step_idx,
                "visibility_group": visibility_group,
                "q_target": q_target,
                "gripper": action[7],
                "terminate": int(action[8]),
                "step_gap": step_gap,
                "delta_q": delta_q,
                "delta_q_per_step": (
                    None
                    if delta_q is None
                    else [value / step_gap for value in delta_q]
                ),
            }
        )
        previous_by_episode[episode_idx] = {
            "step_idx": step_idx,
            "q_target": q_target,
        }
    if visibility_by_key:
        raise AuditValidationError(
            "extra_visibility_keys",
            keys=sorted(visibility_by_key),
        )
    _validate_episode_termination(output)
    return output
