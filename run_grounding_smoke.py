"""Grounding 世界坐标闭环 smoke test 的编排与运行入口。"""

from collections import Counter
from dataclasses import dataclass
import json

from grounding_targeting import SmokeSafetyAbort


@dataclass(frozen=True)
class SmokeDependencies:
    """闭环依赖边界；便于离线测试证明动作侧不读取真值。"""

    observe: object
    ground: object
    compute_action: object
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
            **(payload or {}),
        }
        _append_jsonl(trace_path, row)
        rows.append(row)
        return row

    for step in range(smoke_config["max_control_steps"]):
        observation = dependencies.observe()
        visibility = observation["visibility"]

        if (
            visibility["block_visibility_ratio"]
            < smoke_config["clear_visibility_threshold"]
        ):
            termination = "visibility_out_of_scope"
            raw_path, annotated_path = dependencies.save_images(
                step, observation["image_bgr"], {}
            )
            record(
                visibility,
                termination,
                {
                    "image_path": raw_path,
                    "annotated_path": annotated_path,
                },
            )
            break

        api_calls += 1
        try:
            boxes, raw_response, latency = dependencies.ground(
                observation["image_bgr"]
            )
        except Exception as exc:
            termination = "api_error"
            raw_path, annotated_path = dependencies.save_images(
                step, observation["image_bgr"], {}
            )
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

        if not boxes or not boxes.get("red_block"):
            termination = "invalid_box"
            raw_path, annotated_path = dependencies.save_images(
                step, observation["image_bgr"], boxes or {}
            )
            record(
                visibility,
                termination,
                {
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
                    "raw_response": raw_response,
                    "boxes": boxes,
                    "latency_seconds": latency,
                    "error": repr(exc),
                    "image_path": raw_path,
                    "annotated_path": annotated_path,
                },
            )
            break

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
                **computed_action,
                "execution": execution,
                "image_path": raw_path,
                "annotated_path": annotated_path,
            },
            score_result=score_result,
        )
        if termination != "running":
            break

    return {
        **case,
        "success": termination == "success",
        "termination_reason": termination,
        "num_control_steps": len(rows),
        "num_actions": sum(row.get("execution") is not None for row in rows),
        "api_calls": api_calls,
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
    """只有固定三个 clear case 全部成功且未超调用上限才通过。"""
    rows = list(summaries)
    successes = sum(row["success"] for row in rows)
    calls = sum(row["api_calls"] for row in rows)
    passed = (
        [row["seed"] for row in rows] == smoke_config["seeds"]
        and successes == smoke_config["required_successes"] == 3
        and all(row["all_clear"] for row in rows)
        and calls <= smoke_config["max_total_api_calls"]
    )
    return {
        "num_episodes": len(rows),
        "success_count": successes,
        "failure_count": len(rows) - successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "total_api_calls": calls,
        "termination_reason_counts": dict(
            Counter(row["termination_reason"] for row in rows)
        ),
        "episodes": rows,
        "passed": passed,
    }
