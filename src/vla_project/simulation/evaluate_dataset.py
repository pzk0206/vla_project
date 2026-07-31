"""只读扫描 expert 数据集，并生成可机器判断的质量报告。"""

import argparse
import json
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
QUALITY_ERROR_COUNT_FIELDS = (
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
    checks.update(
        {
            field.removesuffix("_count"): report[field] == 0
            for field in QUALITY_ERROR_COUNT_FIELDS
        }
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
    }
    invalid_episode_ids = set()
    frame_counts = Counter()
    seen_step_keys = set()
    referenced_images = set()
    block_positions = []

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
        missing_fields = sorted(REQUIRED_FRAME_FIELDS - frame.keys())
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
    for index, summary in enumerate(summaries):
        episode_idx = summary.get("episode_idx")
        missing_fields = sorted(REQUIRED_SUMMARY_FIELDS - summary.keys())
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
