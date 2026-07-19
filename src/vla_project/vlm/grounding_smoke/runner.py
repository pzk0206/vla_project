"""Grounding 世界坐标闭环 smoke test 的编排与运行入口。"""

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
import pybullet as p

from vla_project.simulation.camera_geometry import compute_camera_matrices
from vla_project.vlm.collect_vlm_eval_samples import (
    build_balanced_ee_positions,
    reset_robot_to_target,
)
from vla_project.simulation.control_arm import (
    CONFIG_PATH,
    apply_joint_targets,
    calculate_target_joints,
    capture_rgb_and_segmentation,
    connect_physics,
    get_link_position,
    get_object_position,
    load_block,
    load_config,
    sample_camera_eye,
    settle_object,
    setup_world,
)
from vla_project.vlm.diagnose_vlm_grounding import (
    build_grounding_prompt,
    draw_grounding_boxes,
    parse_grounding_boxes,
)

from vla_project.vlm.grounding_smoke.targeting import (
    SmokeSafetyAbort,
    compute_action_from_world_target,
    compute_grounding_action,
    load_frozen_calibration,
)
from vla_project.simulation.stage3_probe import (
    InvalidModelResponseError,
    call_openai_compatible_api,
    direction_to_target,
)


@dataclass(frozen=True)
class SmokeDependencies:
    """闭环依赖边界；便于离线测试证明动作侧不读取真值。"""

    observe: object
    ground: object
    compute_action: object
    compute_held_action: object
    execute: object
    score: object
    save_images: object


def _append_jsonl(path, row):
    """立即追加一行证据，中止时也保留已经发生的步骤。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_control_loop(
    smoke_config, case, calibration, episode_dir, dependencies
):
    """每步重新观察和定位，直到预测 stop 或明确安全中止。"""
    episode_dir.mkdir(parents=True, exist_ok=False)
    trace_path = episode_dir / "smoke_trace.jsonl"
    initial_scoring = dependencies.score()
    safety_state = {
        "previous_target_xy": None,
        "previous_predicted_distance": None,
        "no_progress_count": 0,
    }
    target_memory = {
        "last_valid_target_world": None,
        "stale_target_steps": 0,
    }
    rows = []
    api_calls = 0
    termination = "max_control_steps"

    def record(visibility, reason, payload=None, score_result=None):
        """统一写 trace，避免异常分支丢失终止原因。"""
        row = {
            **case,
            "control_step": len(rows),
            **visibility,
            **(
                score_result
                if score_result is not None
                else dependencies.score()
            ),
            "termination_reason": reason,
            "decision_source": None,
            "target_age_steps": None,
            "used_target_hold": False,
            "api_called": False,
            **(payload or {}),
        }
        _append_jsonl(trace_path, row)
        rows.append(row)
        return row

    for step in range(smoke_config["max_control_steps"]):
        observation = dependencies.observe()
        visibility = observation["visibility"]

        low_visibility = (
            visibility["block_visibility_ratio"]
            < smoke_config["clear_visibility_threshold"]
        )
        if low_visibility:
            raw_path, annotated_path = dependencies.save_images(
                step, observation["image_bgr"], {}
            )
            cached_target = target_memory["last_valid_target_world"]
            if cached_target is None:
                termination = "visibility_out_of_scope"
                record(
                    visibility,
                    termination,
                    {
                        "image_path": raw_path,
                        "annotated_path": annotated_path,
                    },
                )
                break
            if (
                target_memory["stale_target_steps"]
                >= smoke_config["max_stale_target_steps"]
            ):
                termination = "stale_target_limit"
                record(
                    visibility,
                    termination,
                    {
                        "image_path": raw_path,
                        "annotated_path": annotated_path,
                    },
                )
                break

            target_memory["stale_target_steps"] += 1
            raw_response = None
            boxes = {}
            latency = None
            decision_source = "held_vlm_target"
            target_age_steps = target_memory["stale_target_steps"]
            api_called = False
            try:
                computed_action = dependencies.compute_held_action(
                    target_world=cached_target,
                    ee_pos=observation["ee_pos"],
                    safety_state=safety_state,
                    settings=smoke_config,
                )
                safety_state = computed_action["safety_state"]
            except SmokeSafetyAbort as exc:
                termination = exc.reason
                record(
                    visibility,
                    termination,
                    {
                        "error": str(exc),
                        "image_path": raw_path,
                        "annotated_path": annotated_path,
                    },
                )
                break
            except Exception as exc:
                termination = "held_target_error"
                record(
                    visibility,
                    termination,
                    {
                        "error": repr(exc),
                        "image_path": raw_path,
                        "annotated_path": annotated_path,
                    },
                )
                break
        else:
            api_calls += 1
            api_called = True
            try:
                boxes, raw_response, latency = dependencies.ground(
                    observation["image_bgr"]
                )
            except InvalidModelResponseError as exc:
                termination = "invalid_box"
                raw_path, annotated_path = dependencies.save_images(
                    step, observation["image_bgr"], {}
                )
                record(
                    visibility,
                    termination,
                    {
                        "api_called": True,
                        "error": repr(exc),
                        "image_path": raw_path,
                        "annotated_path": annotated_path,
                    },
                )
                break
            except Exception as exc:
                termination = "api_error"
                raw_path, annotated_path = dependencies.save_images(
                    step, observation["image_bgr"], {}
                )
                record(
                    visibility,
                    termination,
                    {
                        "api_called": True,
                        "error": repr(exc),
                        "image_path": raw_path,
                        "annotated_path": annotated_path,
                    },
                )
                break

            if not boxes or not boxes.get("red_block"):
                termination = "invalid_box"
                raw_path, annotated_path = dependencies.save_images(
                    step, observation["image_bgr"], boxes or {}
                )
                record(
                    visibility,
                    termination,
                    {
                        "api_called": True,
                        "raw_response": raw_response,
                        "boxes": boxes,
                        "latency_seconds": latency,
                        "image_path": raw_path,
                        "annotated_path": annotated_path,
                    },
                )
                break

            raw_path, annotated_path = dependencies.save_images(
                step, observation["image_bgr"], boxes
            )
            try:
                computed_action = dependencies.compute_action(
                    red_block_box=boxes["red_block"],
                    image_size=(
                        observation["image_bgr"].shape[1],
                        observation["image_bgr"].shape[0],
                    ),
                    view_matrix=observation["view_matrix"],
                    projection_matrix=observation["projection_matrix"],
                    calibration=calibration,
                    ee_pos=observation["ee_pos"],
                    safety_state=safety_state,
                    settings=smoke_config,
                )
                safety_state = computed_action["safety_state"]
            except SmokeSafetyAbort as exc:
                termination = exc.reason
                record(
                    visibility,
                    termination,
                    {
                        "api_called": True,
                        "raw_response": raw_response,
                        "boxes": boxes,
                        "latency_seconds": latency,
                        "error": str(exc),
                        "image_path": raw_path,
                        "annotated_path": annotated_path,
                    },
                )
                break
            except Exception as exc:
                termination = "backprojection_error"
                record(
                    visibility,
                    termination,
                    {
                        "api_called": True,
                        "raw_response": raw_response,
                        "boxes": boxes,
                        "latency_seconds": latency,
                        "error": repr(exc),
                        "image_path": raw_path,
                        "annotated_path": annotated_path,
                    },
                )
                break

            target_memory["last_valid_target_world"] = list(
                computed_action["corrected_target_world"]
            )
            target_memory["stale_target_steps"] = 0
            decision_source = "fresh_vlm"
            target_age_steps = 0

        decision_payload = {
            "decision_source": decision_source,
            "target_age_steps": target_age_steps,
            "used_target_hold": decision_source == "held_vlm_target",
            "api_called": api_called,
        }

        execution = None
        if computed_action["direction"] != "stop":
            try:
                execution = dependencies.execute(
                    computed_action["direction"], observation["ee_pos"]
                )
            except Exception as exc:
                termination = "ik_error"
                record(
                    visibility,
                    termination,
                    {
                        "raw_response": raw_response,
                        "boxes": boxes,
                        "latency_seconds": latency,
                        **decision_payload,
                        **computed_action,
                        "error": repr(exc),
                        "image_path": raw_path,
                        "annotated_path": annotated_path,
                    },
                )
                break

        score_result = dependencies.score()
        if computed_action["direction"] == "stop":
            termination = (
                "success"
                if score_result["true_distance_xy"] <= 0.03
                else "false_stop"
            )
        elif step == smoke_config["max_control_steps"] - 1:
            termination = "max_control_steps"
        else:
            termination = "running"

        record(
            visibility,
            termination,
            {
                "raw_response": raw_response,
                "boxes": boxes,
                "latency_seconds": latency,
                **decision_payload,
                **computed_action,
                "execution": execution,
                "image_path": raw_path,
                "annotated_path": annotated_path,
            },
            score_result=score_result,
        )
        if termination != "running":
            break

    fresh_steps = sum(
        row.get("decision_source") == "fresh_vlm" for row in rows
    )
    held_steps = sum(
        row.get("decision_source") == "held_vlm_target" for row in rows
    )
    target_ages = [
        row["target_age_steps"]
        for row in rows
        if isinstance(row.get("target_age_steps"), int)
    ]

    return {
        **case,
        "success": termination == "success",
        "termination_reason": termination,
        "num_control_steps": len(rows),
        "num_actions": sum(row.get("execution") is not None for row in rows),
        "api_calls": api_calls,
        "num_fresh_vlm_steps": fresh_steps,
        "num_held_target_steps": held_steps,
        "max_target_age_steps": max(target_ages, default=0),
        "recovered_from_occlusion": termination == "success"
        and held_steps > 0,
        "initial_true_distance_xy": initial_scoring["true_distance_xy"],
        "final_true_distance_xy": (
            rows[-1]["true_distance_xy"]
            if rows
            else initial_scoring["true_distance_xy"]
        ),
        "max_target_jump_xy": max(
            (
                row["target_jump_xy"]
                for row in rows
                if "target_jump_xy" in row
            ),
            default=None,
        ),
        "all_clear": bool(rows)
        and all(
            row["block_visibility_ratio"]
            >= smoke_config["clear_visibility_threshold"]
            for row in rows
        ),
        "trace_path": str(trace_path),
    }


def aggregate_smoke_summaries(summaries, smoke_config):
    """只有固定三个可恢复案例全部成功且未超安全边界才通过。"""
    rows = list(summaries)
    successes = sum(row["success"] for row in rows)
    calls = sum(row["api_calls"] for row in rows)
    passed = (
        [row["seed"] for row in rows] == smoke_config["seeds"]
        and successes == smoke_config["required_successes"] == 3
        and all(row["num_fresh_vlm_steps"] >= 1 for row in rows)
        and all(
            row["max_target_age_steps"]
            <= smoke_config["max_stale_target_steps"]
            for row in rows
        )
        and calls <= smoke_config["max_total_api_calls"]
    )
    return {
        "num_episodes": len(rows),
        "success_count": successes,
        "failure_count": len(rows) - successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "total_api_calls": calls,
        "fresh_vlm_steps": sum(
            row["num_fresh_vlm_steps"] for row in rows
        ),
        "held_target_steps": sum(
            row["num_held_target_steps"] for row in rows
        ),
        "recovered_episode_count": sum(
            row["recovered_from_occlusion"] for row in rows
        ),
        "termination_reason_counts": dict(
            Counter(row["termination_reason"] for row in rows)
        ),
        "episodes": rows,
        "passed": passed,
    }


def build_smoke_cases(smoke_config):
    """构造不可静默替换的三个固定在线案例。"""
    seeds = smoke_config["seeds"]
    directions = smoke_config["start_directions"]
    if (
        len(seeds) != 3
        or len(directions) != 3
        or len(set(seeds)) != 3
        or directions != ["left", "right", "front"]
    ):
        raise ValueError(
            "smoke cases 必须固定为三个唯一 seed 和 left/right/front"
        )
    return [
        {"episode_idx": index, "seed": seed, "start_direction": direction}
        for index, (seed, direction) in enumerate(zip(seeds, directions))
    ]


def make_run_dir(output_root, run_name=None):
    """创建全新运行目录，拒绝覆盖或续写旧 API 回复。"""
    name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    path = Path(output_root) / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def compute_visibility(segmentation, block_id, reference_pixels):
    """从 PyBullet segmentation 编码中统计红块可见像素。"""
    if (
        isinstance(reference_pixels, bool)
        or not isinstance(reference_pixels, int)
        or reference_pixels <= 0
    ):
        raise ValueError("reference_pixels 必须是正整数")
    object_ids = np.asarray(segmentation, dtype=np.int64) & ((1 << 24) - 1)
    visible = int(np.count_nonzero(object_ids == block_id))
    return {
        "block_visible_pixels": visible,
        "block_reference_pixels": reference_pixels,
        "block_visibility_ratio": min(visible / reference_pixels, 1.0),
    }


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_camera_config(config):
    """把 448 正俯视覆盖应用到相机副本，不污染全局配置。"""
    camera_config = copy.deepcopy(config["camera"])
    camera_config.update(config["vlm_evaluation"]["camera_override"])
    return camera_config


def preflight_smoke_case(config, smoke_config, case):
    """无 API 地验证单个案例的起点与首帧可见率。"""
    random.seed(case["seed"])
    connect_physics("DIRECT")
    try:
        _, robot_id = setup_world(config)
        block_id = load_block(config["task"])
        settle_object(config, config["task"]["initial_settle_steps"])
        initial_block_pos = list(get_object_position(block_id))
        start_positions = build_balanced_ee_positions(
            initial_block_pos,
            smoke_config["hover_z"] - initial_block_pos[2],
            smoke_config["start_offset_xy"],
        )
        requested_start = start_positions[case["start_direction"]]
        actual_start = reset_robot_to_target(
            robot_id,
            config["robot"],
            requested_start,
            tolerance=config["vlm_evaluation"][
                "balanced_pose_tolerance"
            ],
            max_iterations=config["vlm_evaluation"][
                "balanced_pose_ik_iterations"
            ],
        )
        start_error = math.dist(actual_start, requested_start)
        camera_config = _build_camera_config(config)
        camera_eye = sample_camera_eye(camera_config)
        _, segmentation = capture_rgb_and_segmentation(
            camera_config,
            camera_eye,
        )
        visibility = compute_visibility(
            segmentation,
            block_id,
            smoke_config["visibility_reference_pixels"],
        )
        rejection_reason = None
        if start_error > config["vlm_evaluation"][
            "balanced_pose_tolerance"
        ]:
            rejection_reason = "start_pose_error"
        elif (
            visibility["block_visibility_ratio"]
            < smoke_config["clear_visibility_threshold"]
        ):
            rejection_reason = "visibility_below_threshold"
        return {
            **case,
            "qualified": rejection_reason is None,
            "rejection_reason": rejection_reason,
            "requested_start_ee_pos": list(requested_start),
            "actual_start_ee_pos": list(actual_start),
            "start_pose_error": start_error,
            **visibility,
            "camera_eye": list(camera_eye),
        }
    finally:
        p.disconnect()


def _build_episode_dependencies(
    config,
    smoke_config,
    case,
    episode_dir,
    robot_id,
    block_id,
    camera_config,
    camera_eye,
    view_matrix,
    projection_matrix,
):
    """把真实 PyBullet/Qwen 接口封装到真值隔离的依赖边界。"""
    robot_config = config["robot"]
    api_config = copy.deepcopy(config["probe"]["api"])
    if smoke_config["api_max_retries"] != 0:
        raise ValueError("grounding smoke 必须禁用 API 重试")
    api_config["max_retries"] = smoke_config["api_max_retries"]
    instruction = config["dataset"]["instruction"]

    def observe():
        image_bgr, segmentation = capture_rgb_and_segmentation(
            camera_config, camera_eye
        )
        return {
            "image_bgr": image_bgr,
            "view_matrix": view_matrix,
            "projection_matrix": projection_matrix,
            "ee_pos": list(
                get_link_position(robot_id, robot_config["ee_link_index"])
            ),
            "visibility": compute_visibility(
                segmentation,
                block_id,
                smoke_config["visibility_reference_pixels"],
            ),
        }

    def ground(image_bgr):
        started = time.monotonic()
        boxes, raw_response = call_openai_compatible_api(
            image_bgr,
            api_config,
            prompt_text=build_grounding_prompt(instruction),
            response_parser=parse_grounding_boxes,
        )
        return boxes, raw_response, time.monotonic() - started

    def execute(direction, ee_pos):
        target_pos = direction_to_target(
            direction,
            ee_pos,
            smoke_config["hover_z"],
            smoke_config["move_step_xy"],
        )
        target_joints = list(
            calculate_target_joints(robot_id, robot_config, target_pos)[
                : robot_config["controlled_joints"]
            ]
        )
        apply_joint_targets(robot_id, robot_config, target_joints)
        settle_object(config, config["probe"]["sim_steps_per_action"])
        return {
            "direction": direction,
            "target_pos": list(target_pos),
            "target_joint_angles": target_joints,
            "ee_pos_after": list(
                get_link_position(robot_id, robot_config["ee_link_index"])
            ),
        }

    def score():
        # 真值只能在这个事后评分闭包中读取，不返回给 observe/compute_action。
        true_block_pos = list(get_object_position(block_id))
        ee_pos = list(
            get_link_position(robot_id, robot_config["ee_link_index"])
        )
        return {
            "true_block_pos": true_block_pos,
            "true_ee_pos": ee_pos,
            "true_distance_xy": math.hypot(
                true_block_pos[0] - ee_pos[0],
                true_block_pos[1] - ee_pos[1],
            ),
        }

    def save_images(step, image_bgr, boxes):
        raw_path = episode_dir / f"step_{step:02d}_raw.jpg"
        if not cv2.imwrite(str(raw_path), image_bgr):
            raise RuntimeError(f"无法写入 smoke 原图: {raw_path}")

        annotated_path = None
        if boxes and boxes.get("end_effector") and boxes.get("red_block"):
            annotated_path = episode_dir / f"step_{step:02d}_annotated.jpg"
            annotated = draw_grounding_boxes(image_bgr, boxes)
            if not cv2.imwrite(str(annotated_path), annotated):
                raise RuntimeError(f"无法写入 smoke 标注图: {annotated_path}")
        return (
            str(raw_path),
            str(annotated_path) if annotated_path is not None else None,
        )

    return SmokeDependencies(
        observe=observe,
        ground=ground,
        compute_action=compute_grounding_action,
        compute_held_action=compute_action_from_world_target,
        execute=execute,
        score=score,
        save_images=save_images,
    )


def run_smoke_batch(config, run_name=None):
    """运行固定三个真实 PyBullet/Qwen case，并保存批次证据。"""
    smoke_config = config["grounding_smoke"]
    cases = build_smoke_cases(smoke_config)
    calibration = load_frozen_calibration(
        smoke_config["calibration_path"],
        smoke_config["expected_calibration_sample_ids"],
        smoke_config["expected_correction_x"],
        smoke_config["expected_correction_y"],
    )
    batch_dir = make_run_dir(smoke_config["output_dir"], run_name)
    episode_summary_path = batch_dir / "episode_summary.jsonl"
    summaries = []

    for case in cases:
        random.seed(case["seed"])
        connect_physics("DIRECT")
        try:
            _, robot_id = setup_world(config)
            block_id = load_block(config["task"])
            settle_object(config, config["task"]["initial_settle_steps"])

            # 真值仅用于构造固定 10cm 起点；控制开始后不进入 observe/action。
            initial_block_pos = list(get_object_position(block_id))
            start_positions = build_balanced_ee_positions(
                initial_block_pos,
                smoke_config["hover_z"] - initial_block_pos[2],
                smoke_config["start_offset_xy"],
            )
            requested_start = start_positions[case["start_direction"]]
            actual_start = reset_robot_to_target(
                robot_id,
                config["robot"],
                requested_start,
                tolerance=config["vlm_evaluation"][
                    "balanced_pose_tolerance"
                ],
                max_iterations=config["vlm_evaluation"][
                    "balanced_pose_ik_iterations"
                ],
            )
            start_error = math.dist(actual_start, requested_start)
            if start_error > config["vlm_evaluation"][
                "balanced_pose_tolerance"
            ]:
                raise RuntimeError(
                    f"smoke 起始姿态未收敛: error={start_error:.6f}"
                )

            camera_config = _build_camera_config(config)
            camera_eye = sample_camera_eye(camera_config)
            view_matrix, projection_matrix = compute_camera_matrices(
                camera_config, camera_eye
            )
            episode_dir = batch_dir / f"episode_{case['episode_idx']:03d}"
            dependencies = _build_episode_dependencies(
                config,
                smoke_config,
                case,
                episode_dir,
                robot_id,
                block_id,
                camera_config,
                camera_eye,
                view_matrix,
                projection_matrix,
            )
            summary = run_control_loop(
                smoke_config,
                case,
                calibration,
                episode_dir,
                dependencies,
            )
            summary["camera_eye"] = list(camera_eye)
            summary["requested_start_ee_pos"] = list(requested_start)
            summary["actual_start_ee_pos"] = list(actual_start)
            summaries.append(summary)
            _append_jsonl(episode_summary_path, summary)
        finally:
            p.disconnect()

    batch_summary = aggregate_smoke_summaries(summaries, smoke_config)
    batch_summary["run_dir"] = str(batch_dir)
    batch_summary["calibration_path"] = smoke_config["calibration_path"]
    _write_json(batch_dir / "smoke_summary.json", batch_summary)
    return batch_dir, batch_summary


def main():
    parser = argparse.ArgumentParser(
        description="运行 grounding 世界坐标闭环 smoke test"
    )
    parser.add_argument(
        "--run-name",
        help="可选运行目录名；已存在时拒绝覆盖",
    )
    args = parser.parse_args()
    config = load_config(CONFIG_PATH)
    batch_dir, summary = run_smoke_batch(config, run_name=args.run_name)
    print(f"run_dir={batch_dir}")
    print(
        f"success={summary['success_count']}/{summary['num_episodes']} "
        f"api_calls={summary['total_api_calls']} passed={summary['passed']}"
    )


if __name__ == "__main__":
    main()
