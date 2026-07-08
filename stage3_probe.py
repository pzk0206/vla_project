import base64
import json
import os
import re
import time
import urllib.error
import urllib.request

import cv2
import pybullet as p

from control_arm import (
    CONFIG_PATH,
    apply_joint_targets,
    calculate_target_joints,
    capture_rgb,
    connect_physics,
    euclidean_distance,
    get_link_position,
    get_object_position,
    load_block,
    load_config,
    sample_camera_eye,
    settle_object,
    setup_world,
)


DIRECTION_TO_DELTA = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "front": (0.0, 1.0),
    "back": (0.0, -1.0),
    "stop": (0.0, 0.0),
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def encode_image_to_data_url(image_bgr):
    ok, buffer = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("无法把观测图编码成 JPEG。")
    image_base64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{image_base64}"


def build_probe_prompt():
    return (
        "你在帮助控制一个仿真机械臂末端接近红色积木上方。"
        "请只输出一个方向词，不要输出解释。"
        "允许输出的内容只有: left, right, front, back, stop。"
        "如果红色积木已经基本位于末端正下方，则输出 stop。"
    )


def parse_direction(text):
    if not text:
        return "stop"
    lowered = text.strip().lower()
    for direction in DIRECTION_TO_DELTA:
        if lowered == direction:
            return direction
    match = re.search(r"\b(left|right|front|back|stop)\b", lowered)
    if match:
        return match.group(1)
    return "stop"


def call_openai_compatible_api(image_bgr, api_config):
    base_url = os.environ.get(api_config["base_url_env"], "").rstrip("/")
    api_key = os.environ.get(api_config["api_key_env"], "")
    model_name = os.environ.get(api_config["model_env"], "")

    if not base_url or not api_key or not model_name:
        raise RuntimeError(
            "API 模式缺少环境变量。请设置 "
            f"{api_config['base_url_env']} / {api_config['api_key_env']} / {api_config['model_env']}。"
        )

    payload = {
        "model": model_name,
        "temperature": api_config["temperature"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_probe_prompt()},
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_to_data_url(image_bgr)},
                    },
                ],
            }
        ],
    }

    request = urllib.request.Request(
        url=base_url + api_config["endpoint_path"],
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"API 请求失败: {exc.code} {error_body}") from exc

    content = body["choices"][0]["message"]["content"]
    if isinstance(content, list):
        text_chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_chunks.append(item.get("text", ""))
        content = " ".join(text_chunks)
    return parse_direction(content), content


def heuristic_direction(ee_pos, block_pos, stop_distance_xy):
    dx = block_pos[0] - ee_pos[0]
    dy = block_pos[1] - ee_pos[1]

    if abs(dx) <= stop_distance_xy and abs(dy) <= stop_distance_xy:
        return "stop", "heuristic_stop"
    if abs(dx) >= abs(dy):
        return ("right", "heuristic_dx_positive") if dx > 0 else ("left", "heuristic_dx_negative")
    return ("front", "heuristic_dy_positive") if dy > 0 else ("back", "heuristic_dy_negative")


def decide_direction(probe_config, image_bgr, ee_pos, block_pos):
    mode = probe_config["mode"].lower()
    if mode == "api":
        direction, raw_response = call_openai_compatible_api(image_bgr, probe_config["api"])
        return direction, raw_response, "api"
    direction, raw_response = heuristic_direction(
        ee_pos,
        block_pos,
        probe_config["stop_distance_xy"],
    )
    return direction, raw_response, "heuristic"


def write_probe_trace(trace_jsonl_path, row):
    with open(trace_jsonl_path, "a", encoding="utf-8") as trace_file:
        trace_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    config = load_config(CONFIG_PATH)
    probe_config = config["probe"]
    robot_config = config["robot"]
    task_config = config["task"]
    camera_config = config["camera"]

    connect_physics(config["connection_mode"])
    _, robot_id = setup_world(config)
    block_id = load_block(task_config)
    settle_object(config, task_config["initial_settle_steps"])

    output_dir = probe_config["output_dir"]
    ensure_dir(output_dir)
    trace_jsonl_path = os.path.join(output_dir, "probe_trace.jsonl")
    if os.path.exists(trace_jsonl_path):
        os.remove(trace_jsonl_path)

    camera_eye = sample_camera_eye(camera_config)
    print(
        f"🎥 [PROBE] 阶段三探路相机位置: "
        f"X:{camera_eye[0]:.2f}, Y:{camera_eye[1]:.2f}, Z:{camera_eye[2]:.2f}"
    )

    try:
        for control_step in range(probe_config["max_control_steps"]):
            block_pos = get_object_position(block_id)
            hover_target = [
                block_pos[0],
                block_pos[1],
                block_pos[2] + probe_config["hover_height"],
            ]
            ee_pos = get_link_position(robot_id, robot_config["ee_link_index"])
            distance_to_hover = euclidean_distance(ee_pos, hover_target)

            image_bgr = capture_rgb(camera_config, camera_eye)
            image_filename = os.path.join(output_dir, f"probe_step_{control_step:02d}.jpg")
            if probe_config["save_trace_images"]:
                cv2.imwrite(image_filename, image_bgr)

            direction, raw_response, decision_source = decide_direction(
                probe_config,
                image_bgr,
                ee_pos,
                block_pos,
            )

            if direction == "stop":
                target_pos = hover_target
            else:
                delta_x, delta_y = DIRECTION_TO_DELTA[direction]
                target_pos = [
                    ee_pos[0] + delta_x * probe_config["move_step_xy"],
                    ee_pos[1] + delta_y * probe_config["move_step_xy"],
                    hover_target[2],
                ]

            target_joint_angles = calculate_target_joints(robot_id, robot_config, target_pos)
            apply_joint_targets(robot_id, robot_config, target_joint_angles)

            for _ in range(probe_config["sim_steps_per_action"]):
                p.stepSimulation()
                if config["enable_time_sleep"]:
                    time.sleep(1.0 / config["simulation_hz"])

            print(
                f"🧭 [PROBE] step={control_step} source={decision_source} "
                f"direction={direction} distance={distance_to_hover:.3f}"
            )
            write_probe_trace(
                trace_jsonl_path,
                {
                    "control_step": control_step,
                    "decision_source": decision_source,
                    "direction": direction,
                    "raw_response": raw_response,
                    "ee_pos": list(ee_pos),
                    "block_pos": list(block_pos),
                    "hover_target": list(hover_target),
                    "target_pos": list(target_pos),
                    "distance_to_hover": distance_to_hover,
                    "image_path": image_filename if probe_config["save_trace_images"] else None,
                },
            )

            if direction == "stop" and distance_to_hover <= task_config["success_distance"]:
                print("✅ [PROBE] 模型已判定 stop，且末端已接近悬停目标。")
                break
    finally:
        p.removeBody(block_id)
        p.disconnect()


if __name__ == "__main__":
    main()
