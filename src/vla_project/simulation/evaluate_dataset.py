"""只读扫描 expert 数据集，并生成可机器判断的质量报告。"""

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml


CONFIG_PATH = "sim_config.yaml"
QUALITY_REPORT_NAME = "dataset_quality_report.json"
REQUIRED_FRAME_FIELDS = {
    "schema_version",
    "episode_idx",
    "step_idx",
    "random_seed",
    "image_path",
    "instruction",
    "action",
    "camera_eye",
    "block_pos",
    "target_pos",
    "ee_pos",
    "distance_to_target",
    "termination_reason",
}
REQUIRED_SUMMARY_FIELDS = {
    "schema_version",
    "episode_idx",
    "random_seed",
    "num_steps",
    "num_frames",
    "final_distance",
    "termination_reason",
    "camera_eye",
    "initial_ee_pos",
    "initial_block_pos",
    "final_block_pos",
    "final_target_pos",
    "final_ee_pos",
}
MULTI_V2_FRAME_FIELDS = {"scene_state"}
MULTI_V2_SUMMARY_FIELDS = {
    "task_instruction",
    "target_block",
    "initial_scene_state",
    "final_scene_state",
}
BASE_QUALITY_ERROR_COUNT_FIELDS = (
    "schema_error_count",
    "action_dim_error_count",
    "missing_image_count",
    "unreadable_image_count",
    "image_size_mismatch_count",
    "orphan_image_count",
    "duplicate_step_key_count",
    "seed_error_count",
    "frame_count_mismatch_count",
    "terminal_flag_error_count",
)
MULTI_V2_ERROR_COUNT_FIELDS = (
    "scene_state_error_count",
    "target_consistency_error_count",
    "block_overlap_error_count",
    "block_drift_error_count",
)
QUALITY_ERROR_COUNT_FIELDS = (
    BASE_QUALITY_ERROR_COUNT_FIELDS + MULTI_V2_ERROR_COUNT_FIELDS
)


def _read_jsonl(path):
    """读取 JSONL；单行损坏时保留错误并继续后续记录。"""
    rows = []
    errors = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, TypeError) as exc:
                errors.append(
                    {
                        "type": "invalid_json",
                        "file": path.name,
                        "line": line_number,
                        "detail": str(exc),
                    }
                )
    return rows, errors


def _numeric_stats(values):
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None}
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _axis_bin_counts(values, num_bins=5):
    if not values:
        return []
    low = min(values)
    high = max(values)
    if low == high:
        return [len(values)] + [0] * (num_bins - 1)
    width = (high - low) / num_bins
    counts = [0] * num_bins
    for value in values:
        index = min(int((value - low) / width), num_bins - 1)
        counts[index] += 1
    return counts


def _resolve_image_path(dataset_dir, raw_path):
    path = Path(raw_path)
    if path.is_file():
        return path
    if not path.is_absolute():
        return dataset_dir / path
    return path


def _finite_vector(value, length):
    return (
        isinstance(value, list)
        and len(value) == length
        and all(
            isinstance(item, (int, float)) and math.isfinite(item)
            for item in value
        )
    )


def _valid_scene_state(state):
    if not isinstance(state, dict) or state.get("target_block") not in {"red", "blue"}:
        return False
    blocks = state.get("blocks")
    if not isinstance(blocks, dict):
        return False
    for color in ("red", "blue"):
        block = blocks.get(color)
        if not isinstance(block, dict):
            return False
        if not _finite_vector(block.get("position"), 3):
            return False
        orientation = block.get("orientation")
        if not _finite_vector(orientation, 4):
            return False
        norm = math.sqrt(sum(value * value for value in orientation))
        if abs(norm - 1.0) > 1e-6:
            return False
    robot = state.get("robot")
    return (
        isinstance(robot, dict)
        and _finite_vector(robot.get("joint_positions"), 7)
        and _finite_vector(robot.get("joint_velocities"), 7)
        and _finite_vector(robot.get("ee_position"), 3)
        and _finite_vector(state.get("camera_eye"), 3)
    )


def _vectors_close(left, right, tolerance=1e-6):
    return (
        isinstance(left, list)
        and isinstance(right, list)
        and len(left) == len(right)
        and all(abs(a - b) <= tolerance for a, b in zip(left, right))
    )


def _axis_separation(scene_state):
    red = scene_state["blocks"]["red"]["position"]
    blue = scene_state["blocks"]["blue"]["position"]
    return max(abs(red[0] - blue[0]), abs(red[1] - blue[1]))


def _xy_drift(reference_state, observed_state, color):
    reference = reference_state["blocks"][color]["position"]
    observed = observed_state["blocks"][color]["position"]
    return math.hypot(observed[0] - reference[0], observed[1] - reference[1])


def _target_consistent(row, scene_state, instruction_by_target, hover_height):
    target = scene_state["target_block"]
    if row.get("target_block", target) != target:
        return False
    instruction = row.get("task_instruction", row.get("instruction"))
    if instruction != instruction_by_target.get(target):
        return False
    target_position = scene_state["blocks"][target]["position"]
    if "block_pos" in row and not _vectors_close(row.get("block_pos"), target_position):
        return False
    if "target_pos" in row:
        expected = list(target_position)
        expected[2] += hover_height
        if not _vectors_close(row.get("target_pos"), expected):
            return False
    return True


def evaluate_pilot_gate(report, manifest):
    """严格要求 pilot 数量、成功数和所有完整性计数同时达标。"""
    expected = manifest["pilot_num_episodes"]
    checks = {
        "num_episodes": report["num_episodes"] == expected,
        "success_count": report["success_count"] == expected,
        "schema": report["schema_error_count"] == 0,
        "action_dim": report["action_dim_error_count"] == 0,
        "missing_images": report["missing_image_count"] == 0,
        "unreadable_images": report["unreadable_image_count"] == 0,
        "image_size": report["image_size_mismatch_count"] == 0,
        "orphan_images": report["orphan_image_count"] == 0,
        "duplicate_steps": report["duplicate_step_key_count"] == 0,
        "seeds": report["seed_error_count"] == 0,
        "frame_counts": report["frame_count_mismatch_count"] == 0,
        "terminal_flags": report["terminal_flag_error_count"] == 0,
    }
    if manifest.get("schema_version") == "expert_multi_v2":
        checks.update(
            {
                field.removesuffix("_count"): report[field] == 0
                for field in MULTI_V2_ERROR_COUNT_FIELDS
            }
        )
        expected_per_task = expected // 2
        checks["task_balance"] = (
            expected % 2 == 0
            and report.get("task_counts")
            == {"red": expected_per_task, "blue": expected_per_task}
        )
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
    }


def _has_full_axis_coverage(counts):
    return (
        isinstance(counts, list)
        and len(counts) == 5
        and all(isinstance(count, int) and count > 0 for count in counts)
    )


def evaluate_scale_gate(report, manifest):
    """验收规模化数据数量、成功率、完整性和目标位置覆盖。"""
    checks = {
        "valid_episode_count": (
            report["valid_episode_count"] >= manifest["target_num_episodes"]
        ),
        "success_rate": report["success_rate"] >= 0.99,
        "x_bin_coverage": _has_full_axis_coverage(
            report["block_position"]["x_bin_counts"]
        ),
        "y_bin_coverage": _has_full_axis_coverage(
            report["block_position"]["y_bin_counts"]
        ),
    }
    quality_fields = BASE_QUALITY_ERROR_COUNT_FIELDS
    if manifest.get("schema_version") == "expert_multi_v2":
        quality_fields = QUALITY_ERROR_COUNT_FIELDS
    checks.update(
        {
            field.removesuffix("_count"): report[field] == 0
            for field in quality_fields
        }
    )
    if manifest.get("schema_version") == "expert_multi_v2":
        expected = manifest["target_num_episodes"]
        checks["task_balance"] = (
            expected % 2 == 0
            and report.get("task_counts")
            == {"red": expected // 2, "blue": expected // 2}
        )
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
    }


def select_active_gate(report, manifest, pilot_gate, scale_gate):
    """按已扫描的 episode 数选择 pilot 或规模化门禁。"""
    if report["num_episodes"] >= manifest["target_num_episodes"]:
        return "scale", scale_gate["passed"]
    return "pilot", pilot_gate["passed"]


def evaluate_dataset(dataset_dir):
    """扫描 manifest、JSONL 与 JPEG，一次报告全部可恢复质量问题。"""
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "dataset_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    frame_path = dataset_dir / manifest["jsonl_name"]
    summary_path = dataset_dir / manifest["summary_jsonl_name"]
    frames, frame_json_errors = _read_jsonl(frame_path)
    summaries, summary_json_errors = _read_jsonl(summary_path)
    errors = frame_json_errors + summary_json_errors

    counts = {
        "schema_error_count": len(errors),
        "action_dim_error_count": 0,
        "missing_image_count": 0,
        "unreadable_image_count": 0,
        "image_size_mismatch_count": 0,
        "duplicate_step_key_count": 0,
        "seed_error_count": 0,
        "frame_count_mismatch_count": 0,
        "terminal_flag_error_count": 0,
        "scene_state_error_count": 0,
        "target_consistency_error_count": 0,
        "block_overlap_error_count": 0,
        "block_drift_error_count": 0,
    }
    is_multi_v2 = manifest.get("schema_version") == "expert_multi_v2"
    instruction_by_target = {
        task["target_block"]: task["instruction"]
        for task in manifest.get("tasks", [])
        if isinstance(task, dict)
        and "target_block" in task
        and "instruction" in task
    }
    hover_height = float(manifest.get("hover_height", 0.15))
    pair_config = manifest.get("pair_sampling") or {}
    minimum_separation = float(pair_config.get("min_axis_separation_xy", 0.0))
    maximum_drift = float(pair_config.get("max_episode_drift_xy", 0.0))
    invalid_episode_ids = set()
    frame_counts = Counter()
    seen_step_keys = set()
    referenced_images = set()
    block_positions = []
    frame_scenes_by_episode = defaultdict(list)

    def record_error(error_type, row_type, index, episode_idx=None, detail=None):
        errors.append(
            {
                "type": error_type,
                "record_type": row_type,
                "index": index,
                "episode_idx": episode_idx,
                "detail": detail,
            }
        )
        if episode_idx is not None:
            invalid_episode_ids.add(episode_idx)

    for index, frame in enumerate(frames):
        episode_idx = frame.get("episode_idx")
        required_fields = REQUIRED_FRAME_FIELDS | (
            MULTI_V2_FRAME_FIELDS if is_multi_v2 else set()
        )
        missing_fields = sorted(required_fields - frame.keys())
        schema_invalid = bool(missing_fields) or (
            frame.get("schema_version") != manifest["schema_version"]
        )
        if schema_invalid:
            counts["schema_error_count"] += 1
            record_error(
                "frame_schema",
                "frame",
                index,
                episode_idx,
                {"missing_fields": missing_fields},
            )

        if episode_idx is not None:
            frame_counts[episode_idx] += 1
        step_key = (episode_idx, frame.get("step_idx"))
        if step_key in seen_step_keys:
            counts["duplicate_step_key_count"] += 1
            record_error("duplicate_step", "frame", index, episode_idx)
        else:
            seen_step_keys.add(step_key)

        expected_seed = (
            manifest["random_seed"] + episode_idx
            if isinstance(episode_idx, int)
            else None
        )
        if expected_seed is None or frame.get("random_seed") != expected_seed:
            counts["seed_error_count"] += 1
            record_error("seed", "frame", index, episode_idx)

        action = frame.get("action")
        if not isinstance(action, list) or len(action) != manifest["action_dim"]:
            counts["action_dim_error_count"] += 1
            record_error("action_dim", "frame", index, episode_idx)
        elif (
            frame.get("termination_reason") != "running"
            and action[-1] != 1
        ):
            counts["terminal_flag_error_count"] += 1
            record_error("terminal_flag", "frame", index, episode_idx)

        block_pos = frame.get("block_pos")
        if isinstance(block_pos, list) and len(block_pos) >= 2:
            block_positions.append(block_pos)

        if is_multi_v2:
            scene_state = frame.get("scene_state")
            if not _valid_scene_state(scene_state):
                counts["scene_state_error_count"] += 1
                record_error("scene_state", "frame", index, episode_idx)
            else:
                frame_scenes_by_episode[episode_idx].append(scene_state)
                if not _target_consistent(
                    frame,
                    scene_state,
                    instruction_by_target,
                    hover_height,
                ):
                    counts["target_consistency_error_count"] += 1
                    record_error("target_consistency", "frame", index, episode_idx)

        raw_image_path = frame.get("image_path")
        if not isinstance(raw_image_path, str):
            continue
        image_path = _resolve_image_path(dataset_dir, raw_image_path)
        referenced_images.add(image_path.resolve())
        if not image_path.is_file():
            counts["missing_image_count"] += 1
            record_error("missing_image", "frame", index, episode_idx)
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            counts["unreadable_image_count"] += 1
            record_error("unreadable_image", "frame", index, episode_idx)
            continue
        height, width = image.shape[:2]
        if (
            width != manifest["image_width"]
            or height != manifest["image_height"]
        ):
            counts["image_size_mismatch_count"] += 1
            record_error(
                "image_size",
                "frame",
                index,
                episode_idx,
                {"width": width, "height": height},
            )

    termination_reason_counts = Counter()
    final_distances = []
    frames_per_episode = []
    camera_positions = []
    success_episode_ids = set()
    task_counts = Counter()
    pair_separations = []
    block_drifts = []
    for index, summary in enumerate(summaries):
        episode_idx = summary.get("episode_idx")
        required_fields = REQUIRED_SUMMARY_FIELDS | (
            MULTI_V2_SUMMARY_FIELDS if is_multi_v2 else set()
        )
        missing_fields = sorted(required_fields - summary.keys())
        schema_invalid = bool(missing_fields) or (
            summary.get("schema_version") != manifest["schema_version"]
        )
        if schema_invalid:
            counts["schema_error_count"] += 1
            record_error(
                "summary_schema",
                "summary",
                index,
                episode_idx,
                {"missing_fields": missing_fields},
            )

        expected_seed = (
            manifest["random_seed"] + episode_idx
            if isinstance(episode_idx, int)
            else None
        )
        if expected_seed is None or summary.get("random_seed") != expected_seed:
            counts["seed_error_count"] += 1
            record_error("seed", "summary", index, episode_idx)

        reason = summary.get("termination_reason")
        termination_reason_counts[str(reason)] += 1
        if reason == "success":
            success_episode_ids.add(episode_idx)

        final_distance = summary.get("final_distance")
        if isinstance(final_distance, (int, float)):
            final_distances.append(final_distance)
        num_frames = summary.get("num_frames")
        if isinstance(num_frames, int):
            frames_per_episode.append(num_frames)
            if frame_counts[episode_idx] != num_frames:
                counts["frame_count_mismatch_count"] += 1
                record_error(
                    "frame_count",
                    "summary",
                    index,
                    episode_idx,
                    {
                        "declared": num_frames,
                        "observed": frame_counts[episode_idx],
                    },
                )

        camera_eye = summary.get("camera_eye")
        if isinstance(camera_eye, list) and len(camera_eye) == 3:
            camera_positions.append(camera_eye)

        if is_multi_v2:
            target_block = summary.get("target_block")
            if target_block in {"red", "blue"}:
                task_counts[target_block] += 1
            initial_state = summary.get("initial_scene_state")
            final_state = summary.get("final_scene_state")
            valid_initial = _valid_scene_state(initial_state)
            valid_final = _valid_scene_state(final_state)
            if not valid_initial or not valid_final:
                counts["scene_state_error_count"] += 1
                record_error("scene_state", "summary", index, episode_idx)
                continue
            if not _target_consistent(
                summary,
                initial_state,
                instruction_by_target,
                hover_height,
            ):
                counts["target_consistency_error_count"] += 1
                record_error("target_consistency", "summary", index, episode_idx)
            separation = _axis_separation(initial_state)
            pair_separations.append(separation)
            if separation < minimum_separation:
                counts["block_overlap_error_count"] += 1
                record_error("block_overlap", "summary", index, episode_idx)
            observed_states = [final_state] + frame_scenes_by_episode.get(
                episode_idx, []
            )
            for observed_state in observed_states:
                for color in ("red", "blue"):
                    drift = _xy_drift(initial_state, observed_state, color)
                    block_drifts.append(drift)
                    if drift > maximum_drift:
                        counts["block_drift_error_count"] += 1
                        record_error(
                            "block_drift",
                            "summary",
                            index,
                            episode_idx,
                            {"color": color, "drift_xy": drift},
                        )

    existing_images = {
        image_path.resolve() for image_path in dataset_dir.glob("*.jpg")
    }
    orphan_images = existing_images - referenced_images
    for image_path in sorted(orphan_images):
        errors.append(
            {
                "type": "orphan_image",
                "path": str(image_path),
            }
        )

    x_values = [position[0] for position in block_positions]
    y_values = [position[1] for position in block_positions]
    if camera_positions:
        camera_min = [
            min(position[axis] for position in camera_positions)
            for axis in range(3)
        ]
        camera_max = [
            max(position[axis] for position in camera_positions)
            for axis in range(3)
        ]
    else:
        camera_min = None
        camera_max = None

    success_count = len(success_episode_ids)
    num_episodes = len(summaries)
    valid_episode_count = len(success_episode_ids - invalid_episode_ids)
    report = {
        "num_episodes": num_episodes,
        "valid_episode_count": valid_episode_count,
        "num_frames": len(frames),
        "success_count": success_count,
        "success_rate": success_count / num_episodes if num_episodes else 0.0,
        "termination_reason_counts": dict(termination_reason_counts),
        "final_distance_stats": _numeric_stats(final_distances),
        "frames_per_episode_stats": _numeric_stats(frames_per_episode),
        "block_position": {
            "x_min": min(x_values) if x_values else None,
            "x_max": max(x_values) if x_values else None,
            "y_min": min(y_values) if y_values else None,
            "y_max": max(y_values) if y_values else None,
            "x_bin_counts": _axis_bin_counts(x_values),
            "y_bin_counts": _axis_bin_counts(y_values),
        },
        "camera_position": {
            "min": camera_min,
            "max": camera_max,
        },
        "task_counts": {
            "red": task_counts.get("red", 0),
            "blue": task_counts.get("blue", 0),
        },
        "pair_min_axis_separation_stats": _numeric_stats(pair_separations),
        "block_xy_drift_stats": _numeric_stats(block_drifts),
        **counts,
        "orphan_image_count": len(orphan_images),
        "errors": errors,
    }
    report["pilot_gate"] = evaluate_pilot_gate(report, manifest)
    report["scale_gate"] = evaluate_scale_gate(report, manifest)
    report["active_gate"], report["passed"] = select_active_gate(
        report,
        manifest,
        report["pilot_gate"],
        report["scale_gate"],
    )
    return report


def write_quality_report(dataset_dir, report):
    path = Path(dataset_dir) / QUALITY_REPORT_NAME
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="扫描 expert 数据集质量")
    parser.add_argument("--dataset-dir")
    args = parser.parse_args(argv)
    if args.dataset_dir is None:
        with Path(CONFIG_PATH).open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        dataset_dir = Path(config["dataset"]["output_dir"])
    else:
        dataset_dir = Path(args.dataset_dir)
    report = evaluate_dataset(dataset_dir)
    report_path = write_quality_report(dataset_dir, report)
    print(
        f"数据质量报告: {report_path}；"
        f"episode={report['num_episodes']}，"
        f"success_rate={report['success_rate']:.2%}，"
        f"errors={len(report['errors'])}，"
        f"gate={report['active_gate']}"
    )
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
