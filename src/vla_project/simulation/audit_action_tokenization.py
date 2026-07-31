"""只读审计 expert_v1 动作表示与候选离散化。"""

import math


SCHEMA_VERSION = "action_tokenization_audit_v1"
ALLOWED_VISIBILITY_GROUPS = {"clear", "partial", "severe"}


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
