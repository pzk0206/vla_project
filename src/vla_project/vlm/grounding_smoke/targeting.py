"""不读取红块真值的 grounding 世界坐标与安全动作策略。"""

import json
import math
from pathlib import Path

from vla_project.simulation.camera_geometry import (
    normalized_box_center_to_pixel,
    pixel_to_world_on_plane,
)


FLOAT_TOLERANCE = 1e-12


class SmokeSafetyAbort(RuntimeError):
    """表示定位结果触发了可诊断、不可继续执行的安全条件。"""

    def __init__(self, reason, message):
        self.reason = reason
        super().__init__(f"{reason}: {message}")


def load_frozen_calibration(path, expected_ids, expected_x, expected_y):
    """读取并核对冻结标定，拒绝静默换用其他拟合结果。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("num_clear_calibration_samples") != len(expected_ids):
        raise ValueError("冻结校准样本数不匹配")
    if payload.get("calibration_sample_ids") != list(expected_ids):
        raise ValueError("冻结校准样本 ID 不匹配")

    for field, expected in (
        ("correction_x", expected_x),
        ("correction_y", expected_y),
    ):
        value = payload.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not math.isclose(value, expected, abs_tol=FLOAT_TOLERANCE)
        ):
            raise ValueError(f"冻结校准 {field} 不匹配")
    return payload


def _at_most(value, limit):
    """包含边界地比较浮点距离。"""
    return value <= limit or math.isclose(
        value, limit, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE
    )


def _at_least(value, limit):
    """包含边界地比较浮点改善量。"""
    return value >= limit or math.isclose(
        value, limit, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE
    )


def _dominant_direction(dx, dy, stop_distance):
    """两轴都够近才停止，否则只选择绝对误差更大的世界轴。"""
    if _at_most(abs(dx), stop_distance) and _at_most(
        abs(dy), stop_distance
    ):
        return "stop"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "front" if dy > 0 else "back"


def compute_action_from_world_target(
    target_world,
    ee_pos,
    safety_state,
    settings,
    target_jump_xy=0.0,
):
    """使用已校验的 VLM 世界目标和当前末端位置重新计算安全动作。"""
    corrected = list(target_world)
    dx = corrected[0] - ee_pos[0]
    dy = corrected[1] - ee_pos[1]
    distance = math.hypot(dx, dy)
    direction = _dominant_direction(dx, dy, settings["stop_distance_xy"])

    previous_distance = safety_state["previous_predicted_distance"]
    no_progress_count = safety_state["no_progress_count"]
    if direction != "stop" and previous_distance is not None:
        improvement = previous_distance - distance
        no_progress_count = (
            0
            if _at_least(improvement, settings["min_progress_xy"])
            else no_progress_count + 1
        )
        if no_progress_count >= settings["no_progress_limit"]:
            raise SmokeSafetyAbort(
                "no_progress", f"improvement={improvement}"
            )

    return {
        "direction": direction,
        "corrected_target_world": corrected,
        "predicted_distance_xy": distance,
        "target_jump_xy": target_jump_xy,
        "safety_state": {
            "previous_target_xy": corrected[:2],
            "previous_predicted_distance": distance,
            "no_progress_count": no_progress_count,
        },
    }


def compute_grounding_action(
    red_block_box,
    image_size,
    view_matrix,
    projection_matrix,
    calibration,
    ee_pos,
    safety_state,
    settings,
):
    """把 VLM 框中心反投影、补偿，并生成单轴离散动作。"""
    width, height = image_size
    pixel = normalized_box_center_to_pixel(red_block_box, width, height)
    raw_world = list(
        pixel_to_world_on_plane(
            pixel,
            width,
            height,
            view_matrix,
            projection_matrix,
            plane_z=settings["plane_z"],
        )
    )
    corrected = [
        raw_world[0] + calibration["correction_x"],
        raw_world[1] + calibration["correction_y"],
        raw_world[2],
    ]

    in_workspace = (
        settings["workspace_x"][0]
        <= corrected[0]
        <= settings["workspace_x"][1]
        and settings["workspace_y"][0]
        <= corrected[1]
        <= settings["workspace_y"][1]
    )
    if not in_workspace:
        raise SmokeSafetyAbort("target_out_of_workspace", str(corrected))

    previous_target = safety_state["previous_target_xy"]
    jump = math.dist(previous_target, corrected[:2]) if previous_target else 0.0
    if not _at_most(jump, settings["max_target_jump_xy"]):
        raise SmokeSafetyAbort("target_jump", f"jump={jump}")

    action = compute_action_from_world_target(
        target_world=corrected,
        ee_pos=ee_pos,
        safety_state=safety_state,
        settings=settings,
        target_jump_xy=jump,
    )
    return {
        **action,
        "box_center_pixel": list(pixel),
        "raw_target_world": raw_world,
    }
