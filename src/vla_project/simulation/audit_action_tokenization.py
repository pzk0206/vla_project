"""只读审计 expert_v1 动作表示与候选离散化。"""

import argparse
from datetime import datetime
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import tempfile

import numpy as np

from vla_project.output_paths import (
    resolve_managed_output,
    staged_output_directory,
)


SCHEMA_VERSION = "action_tokenization_audit_v1"
ALLOWED_VISIBILITY_GROUPS = {"clear", "partial", "severe"}
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
DEFAULT_DATASET_DIR = "outputs/dataset/expert_scaling_v1"
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
    episode_termination_counts = Counter()
    for row in rows:
        episode_termination_counts[row["episode_idx"]] += row["terminate"]
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
                "per_episode_counts": {
                    str(episode_idx): int(count)
                    for episode_idx, count in sorted(
                        episode_termination_counts.items()
                    )
                },
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


def read_jsonl(path):
    """读取对象型 JSONL，并把解析失败转换成稳定审计错误。"""
    rows = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AuditValidationError(
                        "invalid_jsonl",
                        path=str(path),
                        line_number=line_number,
                        error=str(exc),
                    ) from exc
                if not isinstance(row, dict):
                    raise AuditValidationError(
                        "invalid_jsonl_row",
                        path=str(path),
                        line_number=line_number,
                    )
                rows.append(row)
    except FileNotFoundError as exc:
        raise AuditValidationError("missing_input_file", path=str(path)) from exc
    return rows


def _file_metadata(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise AuditValidationError("missing_input_file", path=str(path)) from exc
    return {
        "path": str(path),
        "size_bytes": Path(path).stat().st_size,
        "sha256": digest.hexdigest(),
    }


def load_audit_inputs(dataset_dir):
    """从冻结数据集加载 manifest、轨迹、可见性和输入指纹。"""
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "dataset_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AuditValidationError(
            "missing_input_file",
            path=str(manifest_path),
        ) from exc
    except json.JSONDecodeError as exc:
        raise AuditValidationError(
            "invalid_manifest",
            path=str(manifest_path),
            error=str(exc),
        ) from exc
    trajectory_name = manifest.get("jsonl_name")
    if not isinstance(trajectory_name, str) or not trajectory_name:
        raise AuditValidationError("missing_trajectory_name")
    trajectory_path = dataset_dir / trajectory_name
    visibility_path = (
        dataset_dir / "visibility_audit_v1" / "frame_visibility.jsonl"
    )
    trajectory_rows = read_jsonl(trajectory_path)
    visibility_rows = read_jsonl(visibility_path)
    inputs = {
        "manifest": _file_metadata(manifest_path),
        "trajectory": _file_metadata(trajectory_path),
        "visibility": _file_metadata(visibility_path),
    }
    return manifest, trajectory_rows, visibility_rows, inputs


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_failure(dataset_dir, exc):
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "passed": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reason": (
            exc.reason
            if isinstance(exc, AuditValidationError)
            else type(exc).__name__
        ),
        "error": repr(exc),
        "evidence": (
            exc.evidence if isinstance(exc, AuditValidationError) else {}
        ),
    }
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    failure_path = dataset_dir / "action_tokenization_audit_failure.json"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=dataset_dir,
        prefix=".action_tokenization_failure_",
        suffix=".json",
        delete=False,
    ) as handle:
        json.dump(evidence, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(failure_path)


def _resolve_audit_output(dataset_dir, output_dir):
    dataset_dir = Path(dataset_dir).resolve()
    requested = (
        output_dir
        if output_dir is not None
        else dataset_dir / "action_tokenization_audit_v1"
    )
    return resolve_managed_output(
        requested,
        allowed_root=dataset_dir,
        project_root_override=dataset_dir.parent,
    )


def _publish_success(dataset_dir, output_dir, frame_rows, summary):
    dataset_dir = Path(dataset_dir).resolve()
    with staged_output_directory(
        output_dir,
        allowed_root=dataset_dir,
        project_root_override=dataset_dir.parent,
    ) as staging:
        frame_path = staging / "frame_action_analysis.jsonl"
        with frame_path.open("w", encoding="utf-8") as handle:
            for row in frame_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary_path = staging / "action_tokenization_audit.json"
        _write_json(summary_path, summary)
        if len(read_jsonl(frame_path)) != len(frame_rows):
            raise RuntimeError("staged_frame_count_mismatch")
        if json.loads(summary_path.read_text(encoding="utf-8")) != summary:
            raise RuntimeError("staged_summary_mismatch")


def run_action_tokenization_audit(dataset_dir, output_dir=None, episode_ids=None):
    """运行全量只读动作审计并原子发布结果。

    episode_ids: 可选，int 列表。提供时只用这些 episode 拟合分箱边界；
        其他 episode 的帧仍参与统计但分箱边界不由它们决定。
    """
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = _resolve_audit_output(dataset_dir, output_dir)
    try:
        manifest, trajectories, visibility, inputs = load_audit_inputs(
            dataset_dir
        )
        if episode_ids is not None:
            episode_set = set(episode_ids)
            trajectories = [
                r for r in trajectories
                if r.get("episode_idx") in episode_set
            ]
            visibility = [
                r for r in visibility
                if r.get("episode_idx") in episode_set
            ]
        frame_rows = build_frame_analysis(
            manifest,
            trajectories,
            visibility,
        )
        statistics = summarize_action_analysis(frame_rows)
        candidates = evaluate_candidates(frame_rows)
        recommendations = select_recommendations(candidates)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_dir": str(dataset_dir),
            "inputs": inputs,
            "input_validation": {"passed": True},
            **statistics,
            "candidates": candidates,
            "recommendations": recommendations,
            "boundary_fit_episodes": (
                len(episode_ids)
                if episode_ids is not None
                else statistics.get("num_episodes", 0)
            ),
            "limitations": {
                "delta_q_is_saved_target_difference": True,
                "boundaries_are_exploratory_full_dataset_fits": (
                    episode_ids is None
                ),
                "training_or_rollout_validation_completed": False,
            },
            "passed": True,
        }
        if episode_ids is not None:
            summary["split"] = {
                "boundary_fit_episode_count": len(episode_ids),
                "note": (
                    "分箱边界仅由 boundary_fit_episode_count 个 episode 拟合；"
                    "验证集 episode 不参与边界拟合。"
                ),
            }
        _publish_success(dataset_dir, output_dir, frame_rows, summary)
    except Exception as exc:
        _write_failure(dataset_dir, exc)
        raise

    failure_path = dataset_dir / "action_tokenization_audit_failure.json"
    failure_path.unlink(missing_ok=True)
    return summary


def main():
    """CLI：审计 expert_v1 动作分布与候选离散表示。"""
    parser = argparse.ArgumentParser(
        description="审计 expert_v1 action tokenization 候选"
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--episode-ids-file",
        help="episode_split.json 路径；提供时只用其 train 列表拟合分箱边界",
    )
    args = parser.parse_args()
    episode_ids = None
    if args.episode_ids_file:
        split_doc = json.loads(
            Path(args.episode_ids_file).read_text(encoding="utf-8")
        )
        episode_ids = split_doc.get("train")
        if not isinstance(episode_ids, list) or not episode_ids:
            raise ValueError(
                "episode_split.json 缺少 train 列表或列表为空"
            )
    summary = run_action_tokenization_audit(
        args.dataset_dir,
        args.output_dir,
        episode_ids=episode_ids,
    )
    recommendations = summary["recommendations"]
    print(
        "Action tokenization 审计完成："
        f"episodes={summary['num_episodes']}，"
        f"frames={summary['num_frames']}，"
        f"transitions={summary['num_transitions']}，"
        f"absolute={recommendations['absolute_q']['status']}，"
        f"delta={recommendations['delta_q']['status']}"
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


if __name__ == "__main__":
    main()
