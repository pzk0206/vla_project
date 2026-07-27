"""确定性重放 expert 数据，并审计红块逐帧可见率。"""

from collections import Counter, defaultdict
from pathlib import Path
import random
import statistics

import cv2
import numpy as np
import pybullet as p

from vla_project.simulation.control_arm import (
    apply_joint_targets,
    calculate_target_joints,
    capture_rgb_and_segmentation,
    connect_physics,
    get_hover_target,
    load_block,
    reset_robot_to_home,
    sample_camera_eye,
    settle_object,
    setup_world,
)
from vla_project.vlm.evaluate_grounding_backprojection import (
    classify_visibility,
)


OBJECT_ID_MASK = (1 << 24) - 1


class ReplayValidationError(RuntimeError):
    """携带结构化证据的确定性重放失败。"""

    def __init__(self, reason, **evidence):
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence


def count_block_pixels(segmentation, block_id):
    """统计 segmentation 中属于指定物体主体或链接的像素。"""
    if isinstance(block_id, bool) or not isinstance(block_id, int):
        raise ValueError("block_id 必须是整数")
    object_ids = np.asarray(segmentation, dtype=np.int64) & OBJECT_ID_MASK
    return int(np.count_nonzero(object_ids == block_id))


def validate_replay_image(original_path, replay_bgr):
    """确认重放图经过原始 JPEG 编码流程后逐像素一致。"""
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
    diff = np.abs(
        roundtrip.astype(np.int16) - original.astype(np.int16)
    )
    mae = float(np.mean(diff))
    if not np.array_equal(roundtrip, original):
        raise ReplayValidationError(
            "replay_image_mismatch",
            image_path=str(original_path),
            replay_pixel_mae=mae,
            max_pixel_error=int(diff.max()),
        )
    return {
        "replay_exact_match": True,
        "replay_pixel_mae": mae,
    }


def build_visibility_rows(frame_observations, reference_pixels):
    """用同一 episode 的无遮挡参考像素生成逐帧比率和分组。"""
    if (
        isinstance(reference_pixels, bool)
        or not isinstance(reference_pixels, int)
        or reference_pixels <= 0
    ):
        raise ValueError("reference_pixels 必须是正整数")
    rows = []
    for observation in frame_observations:
        visible = observation["visible_block_pixels"]
        if (
            isinstance(visible, bool)
            or not isinstance(visible, int)
            or not 0 <= visible <= reference_pixels
        ):
            raise ValueError("visible_block_pixels 必须位于参考范围内")
        ratio = visible / reference_pixels
        rows.append(
            {
                **observation,
                "reference_block_pixels": reference_pixels,
                "block_visibility_ratio": ratio,
                "visibility_group": classify_visibility(ratio),
            }
        )
    return rows


def numeric_stats(values):
    """返回可 JSON 序列化的基础数值统计。"""
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None}
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def longest_nonclear_run(episode_idx, episode_rows):
    """找出一条 episode 中最长的连续非 clear 保存帧段。"""
    longest = None
    run_start = None
    run_rows = []
    for row in episode_rows + [None]:
        if row is not None and row["visibility_group"] != "clear":
            if run_start is None:
                run_start = row["step_idx"]
            run_rows.append(row)
            continue
        if run_rows:
            candidate = {
                "episode_idx": episode_idx,
                "start_step": run_start,
                "end_step": run_rows[-1]["step_idx"],
                "num_frames": len(run_rows),
            }
            if (
                longest is None
                or candidate["num_frames"] > longest["num_frames"]
            ):
                longest = candidate
        run_start = None
        run_rows = []
    return longest


def summarize_visibility(rows):
    """汇总帧级、episode 级和连续遮挡证据。"""
    rows = sorted(
        rows,
        key=lambda row: (row["episode_idx"], row["step_idx"]),
    )
    if not rows:
        raise ValueError("可见性行不能为空")
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[row["episode_idx"]].append(row)

    episode_longest_runs = {
        episode_idx: longest_nonclear_run(episode_idx, episode_rows)
        for episode_idx, episode_rows in by_episode.items()
    }
    candidates = [
        run for run in episode_longest_runs.values() if run is not None
    ]
    longest = max(
        candidates,
        key=lambda run: run["num_frames"],
        default=None,
    )

    ratios = [row["block_visibility_ratio"] for row in rows]
    group_counts = Counter(row["visibility_group"] for row in rows)
    initial_counts = Counter(
        episode_rows[0]["visibility_group"]
        for episode_rows in by_episode.values()
    )
    terminal_counts = Counter(
        episode_rows[-1]["visibility_group"]
        for episode_rows in by_episode.values()
    )
    initial_ratios = [
        episode_rows[0]["block_visibility_ratio"]
        for episode_rows in by_episode.values()
    ]
    terminal_ratios = [
        episode_rows[-1]["block_visibility_ratio"]
        for episode_rows in by_episode.values()
    ]
    per_episode = []
    for episode_idx, episode_rows in by_episode.items():
        episode_counts = Counter(
            row["visibility_group"] for row in episode_rows
        )
        episode_ratios = [
            row["block_visibility_ratio"] for row in episode_rows
        ]
        per_episode.append(
            {
                "episode_idx": episode_idx,
                "num_frames": len(episode_rows),
                "visibility_group_counts": {
                    group: episode_counts[group]
                    for group in ("clear", "partial", "severe")
                },
                "visibility_ratio_stats": numeric_stats(episode_ratios),
                "initial_visibility_group": (
                    episode_rows[0]["visibility_group"]
                ),
                "terminal_visibility_group": (
                    episode_rows[-1]["visibility_group"]
                ),
                "longest_nonclear_run": episode_longest_runs[episode_idx],
            }
        )

    lowest = sorted(
        rows,
        key=lambda row: row["block_visibility_ratio"],
    )[:10]
    return {
        "num_episodes": len(by_episode),
        "num_frames": len(rows),
        "replay_exact_match_count": sum(
            row["replay_exact_match"] for row in rows
        ),
        "replay_mismatch_count": 0,
        "visibility_group_counts": {
            group: group_counts[group]
            for group in ("clear", "partial", "severe")
        },
        "visibility_group_rates": {
            group: group_counts[group] / len(rows)
            for group in ("clear", "partial", "severe")
        },
        "episodes_with_nonclear": sum(
            any(row["visibility_group"] != "clear" for row in episode_rows)
            for episode_rows in by_episode.values()
        ),
        "episodes_with_severe": sum(
            any(row["visibility_group"] == "severe" for row in episode_rows)
            for episode_rows in by_episode.values()
        ),
        "initial_visibility_group_counts": dict(initial_counts),
        "terminal_visibility_group_counts": dict(terminal_counts),
        "initial_visibility_ratio_stats": numeric_stats(initial_ratios),
        "terminal_visibility_ratio_stats": numeric_stats(terminal_ratios),
        "visibility_ratio_stats": numeric_stats(ratios),
        "longest_nonclear_saved_frame_run": (
            0 if longest is None else longest["num_frames"]
        ),
        "longest_nonclear_run": longest,
        "per_episode": per_episode,
        "lowest_visibility_frames": [
            {
                "episode_idx": row["episode_idx"],
                "step_idx": row["step_idx"],
                "image_path": row["image_path"],
                "block_visibility_ratio": row["block_visibility_ratio"],
                "visibility_group": row["visibility_group"],
            }
            for row in lowest
        ],
    }


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
            "missing_episode_frames",
            episode_idx=episode_idx,
        )
    steps = [row["step_idx"] for row in frame_rows]
    if len(steps) != len(set(steps)):
        raise ReplayValidationError(
            "duplicate_step",
            episode_idx=episode_idx,
        )
    camera_eyes = {tuple(row["camera_eye"]) for row in frame_rows}
    if len(camera_eyes) != 1:
        raise ReplayValidationError(
            "camera_mismatch",
            episode_idx=episode_idx,
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


def replay_episode(config, manifest, summary, frame_rows):
    """按原始 seed 和控制顺序重放一条 episode。"""
    frame_rows = sorted(frame_rows, key=lambda row: row["step_idx"])
    validate_episode_contract(manifest, summary, frame_rows)
    episode_idx = summary["episode_idx"]
    random.seed(summary["random_seed"])
    connect_physics("DIRECT")
    observations = []
    try:
        _, robot_id = setup_world(config)
        reset_robot_to_home(robot_id, config["robot"], config["dataset"])
        for _ in range(config["task"]["initial_settle_steps"]):
            p.stepSimulation()
        block_id = load_block(config["task"])
        settle_object(config, config["task"]["initial_settle_steps"])
        camera_eye = sample_camera_eye(config["camera"])
        expected_eye = frame_rows[0]["camera_eye"]
        if camera_eye != expected_eye:
            raise ReplayValidationError(
                "replayed_camera_mismatch",
                episode_idx=episode_idx,
                expected_camera_eye=expected_eye,
                actual_camera_eye=camera_eye,
            )

        rows_by_step = {row["step_idx"]: row for row in frame_rows}
        for step_idx in range(frame_rows[-1]["step_idx"] + 1):
            target = get_hover_target(
                block_id,
                config["task"]["hover_height"],
            )
            joints = calculate_target_joints(
                robot_id,
                config["robot"],
                target,
            )
            apply_joint_targets(robot_id, config["robot"], joints)
            p.stepSimulation()
            if step_idx not in rows_by_step:
                continue
            source = rows_by_step[step_idx]
            replay_bgr, segmentation = capture_rgb_and_segmentation(
                config["camera"],
                camera_eye,
            )
            replay_check = validate_replay_image(
                source["image_path"],
                replay_bgr,
            )
            observations.append(
                {
                    "schema_version": "visibility_audit_v1",
                    "episode_idx": episode_idx,
                    "random_seed": summary["random_seed"],
                    "step_idx": step_idx,
                    "image_path": source["image_path"],
                    "visible_block_pixels": count_block_pixels(
                        segmentation,
                        block_id,
                    ),
                    **replay_check,
                }
            )

        p.removeBody(robot_id)
        _, reference_segmentation = capture_rgb_and_segmentation(
            config["camera"],
            camera_eye,
        )
        reference_pixels = count_block_pixels(
            reference_segmentation,
            block_id,
        )
        return build_visibility_rows(observations, reference_pixels)
    except ReplayValidationError as exc:
        exc.evidence.setdefault("episode_idx", episode_idx)
        raise
    finally:
        p.disconnect()
