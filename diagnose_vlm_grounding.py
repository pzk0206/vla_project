"""让 VLM 标出机械臂末端和红块，用于拆解方向误判的视觉根因。"""

import argparse
import json
import time
from pathlib import Path

import cv2

from collect_vlm_eval_samples import read_jsonl
from control_arm import CONFIG_PATH, load_config
from stage3_probe import call_openai_compatible_api


REQUIRED_BOXES = ("end_effector", "red_block")


def build_grounding_prompt(instruction):
    """只请求两个目标框，不要求模型继续判断移动方向。"""
    return (
        f"任务指令：{instruction}\n"
        "这是一张 PyBullet 机械臂正俯视图。请定位机械臂真实末端和红色方块。\n"
        "判断真实末端时，先找到固定底座，再沿机械臂的连接关系逐节追踪；"
        "连接链最后一节的末梢才是真实末端，不要选择底座或中间关节。\n"
        "不要根据部件颜色、画面上下位置或到基座的画面直线距离判断末端。\n"
        "使用 0 到 1000 的归一化图像坐标，坐标顺序为 [x1,y1,x2,y2]。\n"
        "只能输出一行 JSON，格式必须是："
        '{"end_effector":[x1,y1,x2,y2],"red_block":[x1,y1,x2,y2]}。\n'
        "不要解释，不要添加 Markdown。"
    )


def parse_grounding_boxes(text):
    """解析并验证模型给出的两个 0–1000 归一化目标框。"""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return None

    boxes = {}
    for name in REQUIRED_BOXES:
        box = payload.get(name) if isinstance(payload, dict) else None
        if not isinstance(box, list) or len(box) != 4:
            return None
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in box):
            return None
        normalized = [float(value) for value in box]
        x1, y1, x2, y2 = normalized
        if not all(0 <= value <= 1000 for value in normalized):
            return None
        if x1 >= x2 or y1 >= y2:
            return None
        boxes[name] = normalized
    return boxes


def _box_to_pixels(box, image_width, image_height):
    """把 0–1000 坐标转换成当前图片像素坐标。"""
    x1, y1, x2, y2 = box
    return (
        int(round(x1 / 1000 * (image_width - 1))),
        int(round(y1 / 1000 * (image_height - 1))),
        int(round(x2 / 1000 * (image_width - 1))),
        int(round(y2 / 1000 * (image_height - 1))),
    )


def draw_grounding_boxes(image_bgr, boxes):
    """在图片副本上绘制绿色末端框和黄色红块框。"""
    annotated = image_bgr.copy()
    height, width = annotated.shape[:2]
    styles = {
        "end_effector": ((0, 255, 0), "VLM END EFFECTOR"),
        "red_block": ((0, 255, 255), "VLM RED BLOCK"),
    }
    for name in REQUIRED_BOXES:
        color, label = styles[name]
        x1, y1, x2, y2 = _box_to_pixels(boxes[name], width, height)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def diagnose_grounding(config, limit=None):
    """对固定离线图片调用 VLM，并保存带预测框的诊断图片。"""
    evaluation = config["vlm_evaluation"]
    samples = read_jsonl(Path(evaluation["sample_output_dir"]) / "samples.jsonl")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit 必须大于 0")
        samples = samples[:limit]

    run_dir = Path(evaluation["run_output_dir"]) / evaluation["grounding_run_name"]
    annotated_dir = run_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "grounding_predictions.jsonl"

    results = []
    for sample in samples:
        image_bgr = cv2.imread(sample["image_path"])
        started_at = time.perf_counter()
        boxes = None
        raw_response = None
        error_type = None
        error_message = None
        try:
            boxes, raw_response = call_openai_compatible_api(
                image_bgr,
                config["probe"]["api"],
                prompt_text=build_grounding_prompt(sample["instruction"]),
                response_parser=parse_grounding_boxes,
            )
            annotated_path = annotated_dir / f"{sample['sample_id']}.jpg"
            cv2.imwrite(str(annotated_path), draw_grounding_boxes(image_bgr, boxes))
        except Exception as exc:
            annotated_path = None
            error_type = type(exc).__name__
            error_message = str(exc)

        result = {
            "sample_id": sample["sample_id"],
            "image_path": sample["image_path"],
            "expected_direction": sample["expected_direction"],
            "boxes": boxes,
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
    parser.add_argument("--limit", type=int, default=4, help="诊断 manifest 前 N 张图片。")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(CONFIG_PATH)
    run_dir, results = diagnose_grounding(config, limit=args.limit)
    valid_count = sum(result["boxes"] is not None for result in results)
    print(f"✅ 末端定位诊断目录: {run_dir}")
    print(f"成功生成目标框: {valid_count}/{len(results)}")


if __name__ == "__main__":
    main()
