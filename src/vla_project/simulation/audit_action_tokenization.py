"""只读审计 expert_v1 动作表示与候选离散化。"""

import math
from collections import Counter

import numpy as np


SCHEMA_VERSION = "action_tokenization_audit_v1"
ALLOWED_VISIBILITY_GROUPS = {"clear", "partial", "severe"}
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


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
