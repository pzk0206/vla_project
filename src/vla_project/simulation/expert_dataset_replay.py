"""共享的 expert 数据集确定性重放契约。"""

from dataclasses import dataclass
import json
from pathlib import Path
import random

import cv2
import numpy as np
import pybullet as p
import yaml

from vla_project.simulation.control_arm import (
    apply_joint_targets,
    calculate_target_joints,
    connect_physics,
    euclidean_distance,
    get_hover_target,
    get_link_position,
    get_object_position,
    load_block,
    reset_robot_to_home,
    sample_camera_eye,
    settle_object,
    setup_world,
)


MAX_REPLAY_PIXEL_MAE = 0.002
MAX_REPLAY_PIXEL_ERROR = 3


class ReplayValidationError(RuntimeError):
    """携带结构化证据的确定性重放失败。"""

    def __init__(self, reason, **evidence):
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence


@dataclass(frozen=True)
class ReplayFrame:
    """一次保存时刻的源记录和真实重放状态。"""

    source_row: dict
    robot_id: int
    block_id: int
    source_camera_eye: list[float]
    target_joint_angles: list[float]
    target_pos: list[float]
    ee_pos: list[float]
    block_pos: list[float]
    distance_to_target: float


def replay_difference_is_acceptable(replay_pixel_mae, max_pixel_error):
    """只接受全量诊断界定的微小 OpenGL 重渲染差异。"""
    return (
        replay_pixel_mae <= MAX_REPLAY_PIXEL_MAE
        and max_pixel_error <= MAX_REPLAY_PIXEL_ERROR
    )


def validate_replay_image(original_path, replay_bgr):
    """确认重放 JPEG 精确一致或只含已界定的渲染舍入差异。"""
    original_path = Path(original_path)
    original = cv2.imread(str(original_path))
    if original is None:
        raise ReplayValidationError(
            "unreadable_original_image",
            image_path=str(original_path),
        )
    encoded, buffer = cv2.imencode(".jpg", replay_bgr)
    if not encoded:
        raise ReplayValidationError("replay_jpeg_encode_failed")
    roundtrip = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if roundtrip is None or roundtrip.shape != original.shape:
        raise ReplayValidationError(
            "replay_image_shape_mismatch",
            expected_shape=list(original.shape),
            actual_shape=None if roundtrip is None else list(roundtrip.shape),
        )
    diff = np.abs(roundtrip.astype(np.int16) - original.astype(np.int16))
    mae = float(np.mean(diff))
    max_error = int(diff.max())
    changed_values = int(np.count_nonzero(diff))
    exact_match = changed_values == 0
    if not replay_difference_is_acceptable(mae, max_error):
        raise ReplayValidationError(
            "replay_image_mismatch",
            image_path=str(original_path),
            replay_pixel_mae=mae,
            max_pixel_error=max_error,
            changed_pixel_values=changed_values,
        )
    return {
        "replay_exact_match": exact_match,
        "replay_within_tolerance": True,
        "replay_pixel_mae": mae,
        "replay_max_pixel_error": max_error,
        "replay_changed_pixel_values": changed_values,
    }


def read_jsonl(path):
    """读取非空 JSONL 行。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_replay_inputs(dataset_dir):
    """从版本化数据集自身加载重放契约与记录。"""
    dataset_dir = Path(dataset_dir)
    with (dataset_dir / "dataset_manifest.json").open(
        "r", encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)
    with (dataset_dir / "config_snapshot.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        config = yaml.safe_load(handle)
    frames = read_jsonl(dataset_dir / manifest["jsonl_name"])
    summaries = read_jsonl(dataset_dir / manifest["summary_jsonl_name"])
    return manifest, config, frames, summaries


def validate_episode_contract(manifest, summary, frame_rows):
    """拒绝无法一一重放的 episode、seed、step 和相机记录。"""
    episode_idx = summary["episode_idx"]
    expected_seed = manifest["random_seed"] + episode_idx
    if summary["random_seed"] != expected_seed:
        raise ReplayValidationError(
            "seed_mismatch",
            episode_idx=episode_idx,
            expected_seed=expected_seed,
            actual_seed=summary["random_seed"],
        )
    if not frame_rows:
        raise ReplayValidationError(
            "missing_episode_frames", episode_idx=episode_idx
        )
    steps = [row["step_idx"] for row in frame_rows]
    if len(steps) != len(set(steps)):
        raise ReplayValidationError("duplicate_step", episode_idx=episode_idx)
    if steps != sorted(steps):
        raise ReplayValidationError(
            "non_increasing_step", episode_idx=episode_idx
        )
    camera_eyes = {tuple(row["camera_eye"]) for row in frame_rows}
    if len(camera_eyes) != 1:
        raise ReplayValidationError(
            "camera_mismatch", episode_idx=episode_idx
        )
    for row in frame_rows:
        if row["episode_idx"] != episode_idx:
            raise ReplayValidationError(
                "episode_mismatch",
                episode_idx=episode_idx,
                frame_episode_idx=row["episode_idx"],
            )
        if row["random_seed"] != expected_seed:
            raise ReplayValidationError(
                "frame_seed_mismatch",
                episode_idx=episode_idx,
                step_idx=row["step_idx"],
            )


def replay_episode_frames(config, manifest, summary, frame_rows, on_frame):
    """按原采集顺序重放一条 episode，并回调每个保存时刻。"""
    frame_rows = sorted(frame_rows, key=lambda row: row["step_idx"])
    validate_episode_contract(manifest, summary, frame_rows)
    episode_idx = summary["episode_idx"]
    random.seed(summary["random_seed"])
    connect_physics("DIRECT")
    results = []
    try:
        _, robot_id = setup_world(config)
        reset_robot_to_home(robot_id, config["robot"], config["dataset"])
        for _ in range(config["task"]["initial_settle_steps"]):
            p.stepSimulation()
        block_id = load_block(config["task"])
        settle_object(config, config["task"]["initial_settle_steps"])
        source_camera_eye = sample_camera_eye(config["camera"])
        expected_eye = frame_rows[0]["camera_eye"]
        if source_camera_eye != expected_eye:
            raise ReplayValidationError(
                "replayed_camera_mismatch",
                episode_idx=episode_idx,
                expected_camera_eye=expected_eye,
                actual_camera_eye=source_camera_eye,
            )

        rows_by_step = {row["step_idx"]: row for row in frame_rows}
        for step_idx in range(frame_rows[-1]["step_idx"] + 1):
            target_pos = get_hover_target(
                block_id, config["task"]["hover_height"]
            )
            target_joint_angles = calculate_target_joints(
                robot_id, config["robot"], target_pos
            )
            apply_joint_targets(robot_id, config["robot"], target_joint_angles)
            p.stepSimulation()
            if step_idx not in rows_by_step:
                continue
            ee_pos = get_link_position(
                robot_id, config["robot"]["ee_link_index"]
            )
            block_pos = get_object_position(block_id)
            replay_frame = ReplayFrame(
                source_row=rows_by_step[step_idx],
                robot_id=robot_id,
                block_id=block_id,
                source_camera_eye=list(source_camera_eye),
                target_joint_angles=list(
                    target_joint_angles[: config["robot"]["controlled_joints"]]
                ),
                target_pos=list(target_pos),
                ee_pos=list(ee_pos),
                block_pos=list(block_pos),
                distance_to_target=euclidean_distance(ee_pos, target_pos),
            )
            results.append(on_frame(replay_frame))
        if len(results) != len(frame_rows):
            raise ReplayValidationError(
                "unconsumed_episode_frames",
                episode_idx=episode_idx,
                expected_frames=len(frame_rows),
                replayed_frames=len(results),
            )
        return results
    except ReplayValidationError as exc:
        exc.evidence.setdefault("episode_idx", episode_idx)
        raise
    finally:
        p.disconnect()
