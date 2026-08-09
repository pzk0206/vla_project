"""从冻结 expert 轨迹确定性生成固定俯视视觉派生数据集。"""

import argparse
from collections import defaultdict
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
import yaml

from vla_project.output_paths import resolve_managed_output
from vla_project.simulation.control_arm import capture_rgb
from vla_project.simulation.expert_dataset_replay import (
    load_replay_inputs,
    replay_episode_frames,
    validate_replay_image,
)


SCHEMA_VERSION = "expert_view_v1"
DATASET_NAME = "expert_topdown_v1"
DEFAULT_SOURCE_DATASET = "outputs/dataset/expert_scaling_v1"
DEFAULT_OUTPUT_DIR = "outputs/dataset/expert_topdown_v1"
TOPDOWN_EYE = [0.0, 0.4, 3.0]
VALUE_ATOL = 1e-9
MINIMUM_FREE_BYTES = 2 * 1024**3
EXPECTED_EPISODES = 300
EXPECTED_FRAMES = 9894


class DerivationValidationError(RuntimeError):
    """携带稳定 reason 和证据的派生数据拒绝。"""

    def __init__(self, reason, **evidence):
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence


def _is_within(path, parent):
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_dataset_paths(repo_root, source_dataset, output_dir):
    """解析并隔离源、输出和允许的数据集根目录。"""
    repo_root = Path(repo_root).resolve()
    try:
        source = resolve_managed_output(
            source_dataset,
            allowed_root="outputs/dataset",
            project_root_override=repo_root,
        )
        output = resolve_managed_output(
            output_dir,
            allowed_root="outputs/dataset",
            project_root_override=repo_root,
        )
    except ValueError as exc:
        raise DerivationValidationError(
            "unsafe_dataset_path",
            error=str(exc),
        ) from exc
    if not source.is_dir():
        raise DerivationValidationError(
            "source_dataset_missing", source_dataset=str(source)
        )
    if source == output or _is_within(output, source) or _is_within(
        source, output
    ):
        raise DerivationValidationError(
            "dataset_path_overlap",
            source_dataset=str(source),
            output_dir=str(output),
        )
    if output.exists():
        raise FileExistsError(f"派生输出目录已存在: {output}")
    stale = sorted(
        output.parent.glob(f".{output.name}.staging-*")
    )
    if stale:
        raise DerivationValidationError(
            "stale_staging_exists",
            paths=[str(path) for path in stale],
        )
    free_bytes = shutil.disk_usage(output.parent)[2]
    if free_bytes < MINIMUM_FREE_BYTES:
        raise DerivationValidationError(
            "insufficient_disk_space",
            required_bytes=MINIMUM_FREE_BYTES,
            free_bytes=free_bytes,
        )
    return source, output


def sha256_file(path):
    """流式计算单文件 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source_image(source_dataset, repo_root, raw_path):
    raw = Path(raw_path)
    candidates = [raw] if raw.is_absolute() else [repo_root / raw, source_dataset / raw]
    image_path = next((path for path in candidates if path.is_file()), candidates[0])
    resolved = image_path.resolve()
    if not resolved.is_file():
        raise DerivationValidationError(
            "source_image_missing", image_path=str(resolved)
        )
    if not _is_within(resolved, source_dataset):
        raise DerivationValidationError(
            "source_image_path_escape", image_path=str(resolved)
        )
    return resolved


def aggregate_source_images_sha256(frame_rows, source_dataset, repo_root):
    """对排序后的图片路径和内容计算无歧义聚合摘要。"""
    source_dataset = Path(source_dataset).resolve()
    repo_root = Path(repo_root).resolve()
    ordered = sorted(
        frame_rows, key=lambda row: (row["episode_idx"], row["step_idx"])
    )
    keys = [(row["episode_idx"], row["step_idx"]) for row in ordered]
    if len(keys) != len(set(keys)):
        raise DerivationValidationError("duplicate_frame_key")
    digest = hashlib.sha256()
    for row in ordered:
        image_path = _resolve_source_image(
            source_dataset, repo_root, row["image_path"]
        )
        relative = image_path.relative_to(source_dataset).as_posix().encode(
            "utf-8"
        )
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with image_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def publish_staging(staging, output_dir):
    """只把完整 staging 一次性改名为最终数据集目录。"""
    staging = Path(staging)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"派生输出目录已存在: {output_dir}")
    staging.replace(output_dir)


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _source_hashes(source, manifest, frames, repo_root):
    return {
        "source_manifest_sha256": sha256_file(
            source / "dataset_manifest.json"
        ),
        "source_config_sha256": sha256_file(source / "config_snapshot.yaml"),
        "source_trajectory_sha256": sha256_file(
            source / manifest["jsonl_name"]
        ),
        "source_summary_sha256": sha256_file(
            source / manifest["summary_jsonl_name"]
        ),
        "source_images_sha256": aggregate_source_images_sha256(
            frames, source, repo_root
        ),
    }


def _repo_relative(path, repo_root):
    try:
        return Path(path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError as exc:
        raise DerivationValidationError(
            "path_outside_repository", path=str(path)
        ) from exc


def _write_checked_jpeg(path, image):
    if not cv2.imwrite(str(path), image):
        raise DerivationValidationError(
            "derived_jpeg_write_failed", image_path=str(path)
        )
    decoded = cv2.imread(str(path))
    if decoded is None or decoded.shape[:2] != (448, 448):
        raise DerivationValidationError(
            "derived_jpeg_validation_failed",
            image_path=str(path),
            actual_shape=None if decoded is None else list(decoded.shape),
        )


def _write_failure_report(output_dir, exc):
    output_dir = Path(output_dir)
    failure_path = output_dir.with_name(f"{output_dir.name}_failure.json")
    payload = {
        "schema_version": "expert_view_generation_failure_v1",
        "passed": False,
        "reason": getattr(exc, "reason", type(exc).__name__),
        "error": repr(exc),
        "evidence": getattr(exc, "evidence", {}),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=failure_path.parent,
        prefix=f".{failure_path.name}.",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(failure_path)


def run_generation(
    source_dataset,
    output_dir,
    repo_root=Path.cwd(),
    expected_episode_count=EXPECTED_EPISODES,
    expected_frame_count=EXPECTED_FRAMES,
):
    """重放源轨迹、渲染俯视图并原子发布完整派生数据集。"""
    repo_root = Path(repo_root).resolve()
    source, output = validate_dataset_paths(
        repo_root, source_dataset, output_dir
    )
    started_at = datetime.now().isoformat(timespec="seconds")
    try:
        manifest, source_config, frames, summaries = load_replay_inputs(source)
        if len(frames) != expected_frame_count:
            raise DerivationValidationError(
                "unexpected_source_frame_count",
                expected=expected_frame_count,
                actual=len(frames),
            )
        if len(summaries) != expected_episode_count:
            raise DerivationValidationError(
                "unexpected_source_episode_count",
                expected=expected_episode_count,
                actual=len(summaries),
            )
        summary_ids = [row["episode_idx"] for row in summaries]
        if len(summary_ids) != len(set(summary_ids)):
            raise DerivationValidationError("duplicate_episode_summary")
        frames_by_episode = defaultdict(list)
        for row in frames:
            frames_by_episode[row["episode_idx"]].append(row)
        if set(frames_by_episode) != set(summary_ids):
            raise DerivationValidationError("episode_frame_set_mismatch")

        source_hashes_before = _source_hashes(
            source, manifest, frames, repo_root
        )
        topdown_config, topdown_eye = build_topdown_config(
            source_config, output
        )
        derived_manifest = derive_manifest(
            manifest,
            _repo_relative(source, repo_root),
            source_hashes_before,
        )

        with tempfile.TemporaryDirectory(
            dir=output.parent,
            prefix=f".{output.name}.staging-",
        ) as temp_dir:
            staging = Path(temp_dir)
            derived_frames = []
            derived_summaries = []
            exact_matches = 0
            tolerance_matches = 0
            source_validation_pass_count = 0
            topdown_render_pass_count = 0

            for summary in sorted(
                summaries, key=lambda row: row["episode_idx"]
            ):
                episode_rows = sorted(
                    frames_by_episode[summary["episode_idx"]],
                    key=lambda row: row["step_idx"],
                )
                validate_episode_termination(episode_rows, summary)

                def validate_source_frame(frame):
                    nonlocal exact_matches, tolerance_matches
                    source_path = _resolve_source_image(
                        source,
                        repo_root,
                        frame.source_row["image_path"],
                    )
                    source_bgr = capture_rgb(
                        source_config["camera"], frame.source_camera_eye
                    )
                    replay_check = validate_replay_image(
                        source_path, source_bgr
                    )
                    if replay_check["replay_exact_match"]:
                        exact_matches += 1
                    else:
                        tolerance_matches += 1
                    validate_replay_values(frame)
                    return replay_check

                replay_episode_frames(
                    source_config,
                    manifest,
                    summary,
                    episode_rows,
                    validate_source_frame,
                )
                source_validation_pass_count += 1

                def render_topdown_frame(frame):
                    validate_replay_values(frame)
                    image_name = (
                        f"ep_{frame.source_row['episode_idx']}_"
                        f"step_{frame.source_row['step_idx']}.jpg"
                    )
                    _write_checked_jpeg(
                        staging / image_name,
                        capture_rgb(
                            topdown_config["camera"], topdown_eye
                        ),
                    )
                    final_path = _repo_relative(output / image_name, repo_root)
                    row = derive_frame_row(
                        frame.source_row, final_path, topdown_eye
                    )
                    derived_frames.append(row)
                    return row

                replay_episode_frames(
                    source_config,
                    manifest,
                    summary,
                    episode_rows,
                    render_topdown_frame,
                )
                topdown_render_pass_count += 1
                derived_summaries.append(
                    derive_summary_row(summary, topdown_eye)
                )

            if len(derived_frames) != expected_frame_count:
                raise DerivationValidationError(
                    "derived_frame_count_mismatch",
                    expected=expected_frame_count,
                    actual=len(derived_frames),
                )
            image_count = len(list(staging.glob("*.jpg")))
            if image_count != expected_frame_count:
                raise DerivationValidationError(
                    "derived_image_count_mismatch",
                    expected=expected_frame_count,
                    actual=image_count,
                )
            source_hashes_after = _source_hashes(
                source, manifest, frames, repo_root
            )
            if source_hashes_after != source_hashes_before:
                raise DerivationValidationError(
                    "source_hash_changed",
                    before=source_hashes_before,
                    after=source_hashes_after,
                )

            report = {
                "schema_version": "expert_view_generation_report_v1",
                "passed": True,
                "source_dataset": _repo_relative(source, repo_root),
                "output_dir": _repo_relative(output, repo_root),
                "started_at": started_at,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "view_name": "vlm_topdown",
                "camera_eye": topdown_eye,
                "image_width": 448,
                "image_height": 448,
                "num_episodes": len(derived_summaries),
                "num_frames": len(derived_frames),
                "num_images": image_count,
                "source_replay_exact_match_count": exact_matches,
                "source_replay_tolerance_match_count": tolerance_matches,
                "source_validation_pass_count": (
                    source_validation_pass_count
                ),
                "topdown_render_pass_count": topdown_render_pass_count,
                "source_hashes_before": source_hashes_before,
                "source_hashes_after": source_hashes_after,
                "source_hashes_unchanged": True,
                "label_equivalence_passed": True,
                "fixed_camera_passed": True,
                "observation_action_timing": (
                    "post_single_sim_step_with_current_target"
                ),
            }
            _write_json(staging / "dataset_manifest.json", derived_manifest)
            (staging / "config_snapshot.yaml").write_text(
                yaml.safe_dump(
                    topdown_config, allow_unicode=True, sort_keys=False
                ),
                encoding="utf-8",
            )
            _write_jsonl(
                staging / manifest["jsonl_name"], derived_frames
            )
            _write_jsonl(
                staging / manifest["summary_jsonl_name"], derived_summaries
            )
            _write_json(staging / "view_generation_report.json", report)
            publish_staging(staging, output)
            return report
    except Exception as exc:
        _write_failure_report(output, exc)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="确定性生成 expert 固定俯视派生数据集"
    )
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    report = run_generation(args.source_dataset, args.output_dir)
    print(
        "俯视派生数据集生成完成："
        f"episodes={report['num_episodes']}，"
        f"frames={report['num_frames']}，"
        f"images={report['num_images']}"
    )


if __name__ == "__main__":
    main()


def build_topdown_config(source_config, output_dir):
    """合并源快照中的 VLM 俯视覆盖，并冻结派生配置。"""
    derived = copy.deepcopy(source_config)
    try:
        camera_override = derived["vlm_evaluation"]["camera_override"]
        camera = {**derived["camera"], **camera_override}
    except (KeyError, TypeError) as exc:
        raise DerivationValidationError(
            "missing_topdown_camera_profile", error=repr(exc)
        ) from exc
    expected = {
        "image_width": 448,
        "image_height": 448,
        "eye_offset_base": [0.0, 0.0, 3.0],
        "eye_offset_random_range": [0.0, 0.0],
        "up_vector": [0, 1, 0],
        "fov": 45,
    }
    mismatches = {
        key: {"expected": value, "actual": camera.get(key)}
        for key, value in expected.items()
        if camera.get(key) != value
    }
    if mismatches:
        raise DerivationValidationError(
            "topdown_camera_profile_mismatch", mismatches=mismatches
        )
    workspace_center = camera.get("workspace_center")
    if workspace_center != [0.0, 0.4, 0.0]:
        raise DerivationValidationError(
            "workspace_center_mismatch", actual=workspace_center
        )
    eye = [
        workspace_center[axis] + camera["eye_offset_base"][axis]
        for axis in range(3)
    ]
    if eye != TOPDOWN_EYE:
        raise DerivationValidationError(
            "topdown_camera_eye_mismatch", actual=eye
        )
    derived["camera"] = camera
    derived["dataset"]["schema_version"] = SCHEMA_VERSION
    derived["dataset"]["output_dir"] = str(output_dir)
    return derived, eye


def _finite_vector(value, expected_length, label):
    if not isinstance(value, list) or len(value) != expected_length:
        raise DerivationValidationError(
            "invalid_source_value", field=label
        )
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        for item in value
    ):
        raise DerivationValidationError(
            "non_finite_source_value", field=label
        )
    return np.asarray(value, dtype=float)


def _assert_close(actual, expected, reason, field):
    if not np.allclose(actual, expected, rtol=0.0, atol=VALUE_ATOL):
        raise DerivationValidationError(
            reason,
            field=field,
            expected=np.asarray(expected).tolist(),
            actual=np.asarray(actual).tolist(),
        )


def validate_replay_values(frame, source_row=None):
    """确认保存帧动作和状态与同一时刻的确定性重放对齐。"""
    row = frame.source_row if source_row is None else source_row
    action = _finite_vector(row.get("action"), 9, "action")
    if action[7] != 1.0 or action[8] not in (0.0, 1.0):
        raise DerivationValidationError("invalid_special_action_dimensions")
    _assert_close(
        action[:7],
        frame.target_joint_angles,
        "replay_action_mismatch",
        "action",
    )
    for field, replay_value in (
        ("ee_pos", frame.ee_pos),
        ("block_pos", frame.block_pos),
        ("target_pos", frame.target_pos),
    ):
        source_value = _finite_vector(row.get(field), 3, field)
        _assert_close(
            source_value,
            replay_value,
            "replay_state_mismatch",
            field,
        )
    distance = row.get("distance_to_target")
    if (
        isinstance(distance, bool)
        or not isinstance(distance, (int, float))
        or not math.isfinite(distance)
    ):
        raise DerivationValidationError(
            "non_finite_source_value", field="distance_to_target"
        )
    _assert_close(
        [distance],
        [frame.distance_to_target],
        "replay_state_mismatch",
        "distance_to_target",
    )


def validate_episode_termination(frame_rows, summary):
    """要求每条 episode 只有最后一帧携带终止标志。"""
    if not frame_rows:
        raise DerivationValidationError("missing_episode_frames")
    ordered = sorted(frame_rows, key=lambda row: row["step_idx"])
    terminate_values = [row["action"][-1] for row in ordered]
    expected = [0] * (len(ordered) - 1) + [1]
    if terminate_values != expected:
        raise DerivationValidationError(
            "invalid_episode_termination",
            terminate_values=terminate_values,
        )
    if ordered[-1]["termination_reason"] != summary["termination_reason"]:
        raise DerivationValidationError(
            "termination_reason_mismatch",
            frame_reason=ordered[-1]["termination_reason"],
            summary_reason=summary["termination_reason"],
        )


def derive_frame_row(source, image_path, camera_eye):
    """复制源标签，只替换视觉身份并增加可追溯字段。"""
    derived = copy.deepcopy(source)
    derived.update(
        {
            "schema_version": SCHEMA_VERSION,
            "image_path": str(image_path),
            "camera_eye": list(camera_eye),
            "source_image_path": source["image_path"],
            "source_camera_eye": copy.deepcopy(source["camera_eye"]),
        }
    )
    return derived


def derive_summary_row(source, camera_eye):
    """保留 episode 结果，只替换派生相机身份。"""
    derived = copy.deepcopy(source)
    derived.update(
        {
            "schema_version": SCHEMA_VERSION,
            "camera_eye": list(camera_eye),
            "source_camera_eye": copy.deepcopy(source["camera_eye"]),
        }
    )
    return derived


def derive_manifest(source, source_dataset, hashes):
    """构造兼容通用质量扫描的派生 manifest。"""
    derived = copy.deepcopy(source)
    derived.update(
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_name": DATASET_NAME,
            "derived_from": str(source_dataset),
            "source_schema_version": source["schema_version"],
            "view_name": "vlm_topdown",
            "camera_profile_source": (
                "config_snapshot.yaml:vlm_evaluation.camera_override"
            ),
            "derived_read_only": True,
            "generation_method": "deterministic_state_replay",
            "image_width": 448,
            "image_height": 448,
            "camera_eye_offset_base": [0.0, 0.0, 3.0],
            "camera_eye_offset_random_range": [0.0, 0.0],
            **hashes,
        }
    )
    return derived
