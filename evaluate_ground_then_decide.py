"""让 VLM 先输出目标框再输出方向，并检查两者是否自洽。"""

import argparse
import json
import time
from pathlib import Path

import cv2

from collect_vlm_eval_samples import read_jsonl
from vla_project.simulation.control_arm import CONFIG_PATH, load_config
from diagnose_vlm_grounding import (
    REQUIRED_BOXES,
    append_jsonl,
    draw_grounding_boxes,
    parse_grounding_boxes,
)
from vla_project.simulation.stage3_probe import (
    call_openai_compatible_api,
    map_screen_to_world,
)


SCREEN_DIRECTIONS = {
    "screen_left",
    "screen_right",
    "screen_up",
    "screen_down",
}


def build_ground_then_decide_prompt(instruction):
    """要求模型用同一份结构化回复表达定位和方向结论。"""
    return (
        f"任务指令：{instruction}\n"
        "这是一张 PyBullet 机械臂正俯视图。先找到固定底座，再沿机械臂连接关系"
        "逐节追踪，连接链最后一节的末梢才是真实末端。\n"
        "先完成两个框：真实末端 end_effector 和红色方块 red_block。"
        "框坐标使用 0 到 1000 的 [x1,y1,x2,y2]。\n"
        "然后比较两个框的中心：红块在右侧输出 screen_right，在左侧输出 screen_left，"
        "在下方输出 screen_down，在上方输出 screen_up；同时有两个偏差时选择绝对值更大的轴。\n"
        "只能输出一行 JSON："
        '{"end_effector":[x1,y1,x2,y2],"red_block":[x1,y1,x2,y2],'
        '"direction":"screen_left"}。不要解释或添加 Markdown。'
    )


def parse_ground_then_decide(text):
    """验证两个归一化框和一个屏幕方向标签。"""
    boxes = parse_grounding_boxes(text)
    if boxes is None:
        return None
    start = text.find("{")
    end = text.rfind("}")
    try:
        payload = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return None
    direction = payload.get("direction")
    if direction not in SCREEN_DIRECTIONS:
        return None
    return {**boxes, "direction": direction}


def direction_from_grounding_boxes(boxes):
    """根据两个框中心的主轴偏差确定屏幕方向。"""
    ee_x1, ee_y1, ee_x2, ee_y2 = boxes["end_effector"]
    red_x1, red_y1, red_x2, red_y2 = boxes["red_block"]
    dx = (red_x1 + red_x2 - ee_x1 - ee_x2) / 2
    dy = (red_y1 + red_y2 - ee_y1 - ee_y2) / 2
    if abs(dx) >= abs(dy):
        return "screen_right" if dx > 0 else "screen_left"
    return "screen_down" if dy > 0 else "screen_up"


def evaluate_ground_then_decide(config, limit=4):
    """运行结构化对照并保存模型方向、框推导方向和一致性。"""
    evaluation = config["vlm_evaluation"]
    samples = read_jsonl(Path(evaluation["sample_output_dir"]) / "samples.jsonl")[:limit]
    run_dir = Path(evaluation["run_output_dir"]) / evaluation[
        "ground_then_decide_run_name"
    ]
    annotated_dir = run_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "predictions.jsonl"

    results = []
    for sample in samples:
        image_bgr = cv2.imread(sample["image_path"])
        started_at = time.perf_counter()
        try:
            parsed, raw_response = call_openai_compatible_api(
                image_bgr,
                config["probe"]["api"],
                prompt_text=build_ground_then_decide_prompt(sample["instruction"]),
                response_parser=parse_ground_then_decide,
            )
            boxes = {name: parsed[name] for name in REQUIRED_BOXES}
            model_screen_direction = parsed["direction"]
            box_screen_direction = direction_from_grounding_boxes(boxes)
            model_direction = map_screen_to_world(model_screen_direction)
            box_direction = map_screen_to_world(box_screen_direction)
            annotated_path = annotated_dir / f"{sample['sample_id']}.jpg"
            cv2.imwrite(str(annotated_path), draw_grounding_boxes(image_bgr, boxes))
            error_type = None
            error_message = None
        except Exception as exc:
            boxes = None
            raw_response = None
            model_screen_direction = None
            box_screen_direction = None
            model_direction = None
            box_direction = None
            annotated_path = None
            error_type = type(exc).__name__
            error_message = str(exc)

        result = {
            "sample_id": sample["sample_id"],
            "expected_direction": sample["expected_direction"],
            "boxes": boxes,
            "model_screen_direction": model_screen_direction,
            "box_screen_direction": box_screen_direction,
            "model_direction": model_direction,
            "box_direction": box_direction,
            "model_is_exact_match": model_direction == sample["expected_direction"],
            "box_is_exact_match": box_direction == sample["expected_direction"],
            "internally_consistent": model_screen_direction == box_screen_direction,
            "raw_response": raw_response,
            "annotated_path": str(annotated_path) if annotated_path else None,
            "latency_seconds": time.perf_counter() - started_at,
            "error_type": error_type,
            "error_message": error_message,
        }
        append_jsonl(results_path, result)
        results.append(result)
    return run_dir, results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(CONFIG_PATH)
    run_dir, results = evaluate_ground_then_decide(config, args.limit)
    model_correct = sum(row["model_is_exact_match"] for row in results)
    box_correct = sum(row["box_is_exact_match"] for row in results)
    consistent = sum(row["internally_consistent"] for row in results)
    print(f"✅ Ground-then-decide 结果目录: {run_dir}")
    print(
        f"模型方向: {model_correct}/{len(results)}，"
        f"框推导方向: {box_correct}/{len(results)}，"
        f"内部一致: {consistent}/{len(results)}"
    )


if __name__ == "__main__":
    main()
