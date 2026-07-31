"""从冻结 expert 轨迹确定性生成固定俯视视觉派生数据集。"""

import copy
import math
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "expert_view_v1"
DATASET_NAME = "expert_topdown_v1"
DEFAULT_SOURCE_DATASET = "outputs/dataset/expert_scaling_v1"
DEFAULT_OUTPUT_DIR = "outputs/dataset/expert_topdown_v1"
TOPDOWN_EYE = [0.0, 0.4, 3.0]
VALUE_ATOL = 1e-9


class DerivationValidationError(RuntimeError):
    """携带稳定 reason 和证据的派生数据拒绝。"""

    def __init__(self, reason, **evidence):
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence


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
