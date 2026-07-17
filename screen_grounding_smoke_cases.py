"""不调用 VLM 的动态 smoke 案例筛选。"""

import argparse
from collections import Counter
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import random

import cv2
import numpy as np
import pybullet as p

from collect_vlm_eval_samples import (
    build_balanced_ee_positions,
    reset_robot_to_target,
)
from control_arm import (
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
from stage3_probe import direction_to_target


def classify_candidate(step_rows, settings, error=None):
    """按 clear、位置误差和最终距离对一条候选轨迹分类。"""
    rows = list(step_rows)
    if error is not None:
        return {
            "qualified": False,
            "reason": "candidate_error",
            "error": error,
        }

    for row in rows:
        if (
            row["block_visibility_ratio"]
            < settings["clear_visibility_threshold"]
        ):
            return {
                "qualified": False,
                "reason": "visibility_below_threshold",
            }
        if row["target_error_3d"] > settings["max_pose_error"]:
            reason = (
                "start_pose_error"
                if row["observation_step"] == 0
                else "motion_target_error"
            )
            return {"qualified": False, "reason": reason}

    expected_observations = settings["num_actions"] + 1
    if len(rows) != expected_observations:
        return {"qualified": False, "reason": "incomplete_trace"}

    if rows[-1]["true_distance_xy"] > settings["max_final_distance_xy"]:
        return {"qualified": False, "reason": "final_distance_error"}
    return {"qualified": True, "reason": "qualified"}


def select_qualified_cases(candidate_rows, directions):
    """按方向顺序选择 seed 最小且互不重复的合格案例。"""
    rows = list(candidate_rows)
    selected = []
    used_seeds = set()
    for direction in directions:
        eligible = sorted(
            (
                row
                for row in rows
                if row["qualified"]
                and row["direction"] == direction
                and row["seed"] not in used_seeds
            ),
            key=lambda row: row["seed"],
        )
        if not eligible:
            raise ValueError(f"没有可选的 {direction} 合格案例")
        selected.append(eligible[0])
        used_seeds.add(eligible[0]["seed"])
    return selected


def _append_jsonl(path, row):
    """逐候选落盘，避免长筛选中断后没有诊断证据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _make_run_dir(output_root, run_name=None):
    """创建不可覆盖的筛选运行目录。"""
    name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    path = Path(output_root) / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def _compute_visibility(segmentation, block_id, reference_pixels):
    """统计 segmentation 中属于红块主体的像素。"""
    object_ids = np.asarray(segmentation, dtype=np.int64) & ((1 << 24) - 1)
    visible = int(np.count_nonzero(object_ids == block_id))
    return {
        "block_visible_pixels": visible,
        "block_reference_pixels": reference_pixels,
        "block_visibility_ratio": min(visible / reference_pixels, 1.0),
    }


def _screening_settings(smoke_config):
    return {
        **smoke_config["screening"],
        "clear_visibility_threshold": smoke_config[
            "clear_visibility_threshold"
        ],
    }


def _capture_observation(
    config,
    smoke_config,
    candidate_dir,
    observation_step,
    requested_target,
    actual_ee,
    block_id,
    camera_config,
    camera_eye,
):
    """保存一个筛选观察点，并生成完整资格指标。"""
    image, segmentation = capture_rgb_and_segmentation(
        camera_config, camera_eye
    )
    image_path = candidate_dir / f"step_{observation_step:02d}.jpg"
    if not cv2.imwrite(str(image_path), image):
        raise RuntimeError(f"无法写入筛选图片: {image_path}")

    current_block = list(get_object_position(block_id))
    return {
        "observation_step": observation_step,
        "requested_target": list(requested_target),
        "actual_ee_pos": list(actual_ee),
        "target_error_3d": math.dist(actual_ee, requested_target),
        "true_block_pos": current_block,
        "true_distance_xy": math.hypot(
            current_block[0] - actual_ee[0],
            current_block[1] - actual_ee[1],
        ),
        **_compute_visibility(
            segmentation,
            block_id,
            smoke_config["visibility_reference_pixels"],
        ),
        "image_path": str(image_path),
    }


def run_candidate(config, seed, direction, candidate_dir):
    """用真实 2cm 控制轨迹筛选一个 seed+方向候选。"""
    smoke = config["grounding_smoke"]
    screening = smoke["screening"]
    settings = _screening_settings(smoke)
    candidate_dir = Path(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=False)
    rows = []

    random.seed(seed)
    connect_physics("DIRECT")
    try:
        _, robot_id = setup_world(config)
        block_id = load_block(config["task"])
        settle_object(config, config["task"]["initial_settle_steps"])

        block_pos = list(get_object_position(block_id))
        starts = build_balanced_ee_positions(
            block_pos,
            smoke["hover_z"] - block_pos[2],
            smoke["start_offset_xy"],
        )
        requested_target = starts[direction]
        actual_ee = list(
            reset_robot_to_target(
                robot_id,
                config["robot"],
                requested_target,
                tolerance=config["vlm_evaluation"][
                    "balanced_pose_tolerance"
                ],
                max_iterations=config["vlm_evaluation"][
                    "balanced_pose_ik_iterations"
                ],
            )
        )

        camera_config = copy.deepcopy(config["camera"])
        camera_config.update(config["vlm_evaluation"]["camera_override"])
        camera_eye = sample_camera_eye(camera_config)

        for observation_step in range(screening["num_actions"] + 1):
            row = _capture_observation(
                config,
                smoke,
                candidate_dir,
                observation_step,
                requested_target,
                actual_ee,
                block_id,
                camera_config,
                camera_eye,
            )
            rows.append(row)
            partial = classify_candidate(rows, settings)
            if partial["reason"] != "incomplete_trace":
                break
            if observation_step == screening["num_actions"]:
                break

            requested_target = direction_to_target(
                direction,
                actual_ee,
                smoke["hover_z"],
                smoke["move_step_xy"],
            )
            target_joints = calculate_target_joints(
                robot_id, config["robot"], requested_target
            )
            apply_joint_targets(robot_id, config["robot"], target_joints)
            settle_object(config, config["probe"]["sim_steps_per_action"])
            actual_ee = list(
                get_link_position(
                    robot_id, config["robot"]["ee_link_index"]
                )
            )
        classification = classify_candidate(rows, settings)
    except Exception as exc:
        classification = classify_candidate(
            rows, settings, error=repr(exc)
        )
    finally:
        p.disconnect()

    return {
        "seed": seed,
        "direction": direction,
        "steps": rows,
        **classification,
    }


def run_screening(config, run_name=None):
    """评估完整候选笛卡尔积并写确定性选择摘要。"""
    screening = config["grounding_smoke"]["screening"]
    run_dir = _make_run_dir(screening["output_dir"], run_name)
    trace_path = run_dir / "candidate_trace.jsonl"
    candidates = []

    for seed in range(
        screening["seed_range"][0], screening["seed_range"][1] + 1
    ):
        for direction in screening["directions"]:
            candidate = run_candidate(
                config,
                seed,
                direction,
                run_dir / f"seed_{seed:03d}_{direction}",
            )
            candidates.append(candidate)
            _append_jsonl(trace_path, candidate)

    selection_error = None
    try:
        selected = select_qualified_cases(
            candidates, screening["directions"]
        )
    except ValueError as exc:
        selected = []
        selection_error = str(exc)

    summary = {
        "num_candidates": len(candidates),
        "num_qualified": sum(row["qualified"] for row in candidates),
        "reason_counts": dict(
            Counter(row["reason"] for row in candidates)
        ),
        "selected_cases": [
            {"seed": row["seed"], "direction": row["direction"]}
            for row in selected
        ],
        "selection_error": selection_error,
        "passed": selection_error is None,
    }
    _write_json(run_dir / "screening_summary.json", summary)
    return run_dir, summary


def main():
    parser = argparse.ArgumentParser(
        description="无 API 筛选 grounding smoke 动态轨迹案例"
    )
    parser.add_argument(
        "--run-name", help="可选运行目录名；已存在时拒绝覆盖"
    )
    args = parser.parse_args()
    config = load_config(CONFIG_PATH)
    run_dir, summary = run_screening(config, run_name=args.run_name)
    print(f"run_dir={run_dir}")
    print(
        f"qualified={summary['num_qualified']}/{summary['num_candidates']} "
        f"selected={summary['selected_cases']} passed={summary['passed']}"
    )
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
