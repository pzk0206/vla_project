import base64
import json
import os
import re
import time
import urllib.error
import urllib.request

import cv2
import pybullet as p

# 这里复用 control_arm.py 里的基础能力：
# - 加载配置、初始化 PyBullet 世界
# - 采集相机图像
# - 读取红色积木和机械臂末端位置
# - 用 IK 把目标点转换成机械臂关节角
#
# 这样 stage3_probe.py 只负责“闭环探路”逻辑，不重复写仿真环境代码。
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


# 阶段三探路时，模型或规则只输出一个方向词。
# 这里把方向词映射成 x/y 平面上的单位移动方向：
# - left/right 控制 x 轴
# - front/back 控制 y 轴
# - stop 表示已经足够接近，不再横向移动
DIRECTION_TO_DELTA = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "front": (0.0, 1.0),
    "back": (0.0, -1.0),
    "stop": (0.0, 0.0),
}


def ensure_dir(path):
    """确保输出目录存在。

    probe 运行时会把每一步图片和 trace 日志写进 probe_runs/。
    如果目录不存在，先创建；如果已存在，不报错。
    """
    os.makedirs(path, exist_ok=True)


def encode_image_to_data_url(image_bgr):
    """把 OpenCV 的 BGR 图片编码成 API 可接收的 data URL。

    OpenAI 兼容的多模态接口通常接收 image_url。
    本地图片不能直接发文件路径，所以先压缩成 JPEG，再 base64 编码，
    最后拼成 data:image/jpeg;base64,... 这种格式。
    """
    ok, buffer = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("无法把观测图编码成 JPEG。")
    image_base64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{image_base64}"


def build_probe_prompt():
    """构造给多模态模型看的最小控制提示词。

    这个 prompt 故意要求模型只输出 left/right/front/back/stop。
    原因是阶段三先验证闭环骨架，不让模型自由生成长文本，
    否则后面的解析和控制会变得不稳定。
    """
    return (
        "你在帮助控制一个仿真机械臂末端接近红色积木上方。"
        "请只输出一个方向词，不要输出解释。"
        "允许输出的内容只有: left, right, front, back, stop。"
        "如果红色积木已经基本位于末端正下方，则输出 stop。"
    )


def parse_direction(text):
    """从模型回复里解析方向词。

    理想情况：模型只返回 "left" 这种单词。
    兜底情况：模型如果返回 "move left" 或带解释文本，
    这里用正则从中提取第一个合法方向。

    如果完全解析不到，就返回 stop，避免机械臂乱动。
    """
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
    """调用 OpenAI 兼容的多模态 API，让模型根据图片判断移动方向。

    这个函数只在 sim_config.yaml 里 probe.mode = "api" 时会被使用。
    当前默认 mode 是 heuristic，所以一般不会真的请求网络。

    需要的环境变量由 sim_config.yaml 指定：
    - VLA_API_BASE_URL: 接口地址，例如 http://host:port/v1
    - VLA_API_KEY: API Key
    - VLA_MODEL_NAME: 模型名
    """
    base_url = os.environ.get(api_config["base_url_env"], "").rstrip("/")
    api_key = os.environ.get(api_config["api_key_env"], "")
    model_name = os.environ.get(api_config["model_env"], "")

    if not base_url or not api_key or not model_name:
        raise RuntimeError(
            "API 模式缺少环境变量。请设置 "
            f"{api_config['base_url_env']} / {api_config['api_key_env']} / {api_config['model_env']}。"
        )

    # 请求体采用常见的 OpenAI chat/completions 兼容格式：
    # 一条用户消息里同时放文字 prompt 和当前相机图片。
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

    # 不额外依赖 openai SDK，用 Python 标准库 urllib 发送 HTTP 请求。
    # 这样脚本对不同 OpenAI 兼容服务更轻，也更容易迁移。
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

    # 不同兼容服务返回的 content 可能是字符串，也可能是列表。
    # 这里统一整理成普通文本，再交给 parse_direction 解析方向。
    content = body["choices"][0]["message"]["content"]
    if isinstance(content, list):
        text_chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_chunks.append(item.get("text", ""))
        content = " ".join(text_chunks)
    return parse_direction(content), content


def heuristic_direction(ee_pos, block_pos, stop_distance_xy):
    """不用大模型，直接用坐标差写一个“假策略”。

    ee_pos 是机械臂末端位置，block_pos 是红色积木位置。
    当前 Baseline 的目标是让末端移动到红色积木上方，
    所以只需要先在 x/y 平面对齐，再保持悬停高度。

    这个函数的价值：
    - 先验证闭环控制代码能跑通。
    - 排除 API、prompt、视觉理解带来的不确定性。
    - 后续换成真实模型时，可以对比模型表现是否至少接近这个基线。
    """
    dx = block_pos[0] - ee_pos[0]
    dy = block_pos[1] - ee_pos[1]

    # 如果 x/y 都已经足够接近，认为可以停止横向移动。
    if abs(dx) <= stop_distance_xy and abs(dy) <= stop_distance_xy:
        return "stop", "heuristic_stop"

    # 优先修正误差更大的轴，避免一次同时改 x/y 导致控制更难分析。
    if abs(dx) >= abs(dy):
        return ("right", "heuristic_dx_positive") if dx > 0 else ("left", "heuristic_dx_negative")
    return ("front", "heuristic_dy_positive") if dy > 0 else ("back", "heuristic_dy_negative")


def decide_direction(probe_config, image_bgr, ee_pos, block_pos):
    """统一决策入口：根据配置选择真实 API 或启发式规则。

    main() 不关心方向来自哪里，只关心最终拿到：
    - direction: left/right/front/back/stop
    - raw_response: 原始回复或规则原因，方便写日志复盘
    - decision_source: api 或 heuristic
    """
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
    """把每一步闭环控制过程写入 JSONL。

    trace 的目的不是训练，而是调试：
    你可以看每一步模型/规则输出了什么方向、当时末端在哪里、
    红块在哪里、目标点在哪里、距离是否真的在变小。
    """
    with open(trace_jsonl_path, "a", encoding="utf-8") as trace_file:
        trace_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    """阶段三最小闭环探路主流程。

    一次运行只放一个红色积木，最多执行 max_control_steps 次控制：
    1. 拍一张当前相机图。
    2. 根据图片或坐标判断方向。
    3. 把方向转换成下一步目标位置。
    4. 用 IK 求关节角并下发给机械臂。
    5. 推进若干物理仿真 step。
    6. 写 probe_trace.jsonl 方便复盘。
    """
    config = load_config(CONFIG_PATH)
    probe_config = config["probe"]
    robot_config = config["robot"]
    task_config = config["task"]
    camera_config = config["camera"]

    # 初始化仿真世界：连接 PyBullet、加载机械臂、加载红色积木。
    connect_physics(config["connection_mode"])
    _, robot_id = setup_world(config)
    block_id = load_block(task_config)
    settle_object(config, task_config["initial_settle_steps"])

    # probe_runs/ 是阶段三探路输出，不进入 Git 仓库。
    # 每次运行会覆盖旧 trace，图片文件名按 step 编号重写。
    output_dir = probe_config["output_dir"]
    ensure_dir(output_dir)
    trace_jsonl_path = os.path.join(output_dir, "probe_trace.jsonl")
    if os.path.exists(trace_jsonl_path):
        os.remove(trace_jsonl_path)

    # 一次 probe 内相机固定，避免画面变化来自相机抖动。
    # 不同运行之间仍然可以有轻微随机视角，保持后续泛化空间。
    camera_eye = sample_camera_eye(camera_config)
    print(
        f"🎥 [PROBE] 阶段三探路相机位置: "
        f"X:{camera_eye[0]:.2f}, Y:{camera_eye[1]:.2f}, Z:{camera_eye[2]:.2f}"
    )

    try:
        for control_step in range(probe_config["max_control_steps"]):
            # 读取当前红块位置，构造“红块正上方”的悬停目标点。
            block_pos = get_object_position(block_id)
            hover_target = [
                block_pos[0],
                block_pos[1],
                block_pos[2] + probe_config["hover_height"],
            ]
            # 读取机械臂末端位置，用来判断离目标还有多远。
            ee_pos = get_link_position(robot_id, robot_config["ee_link_index"])
            distance_to_hover = euclidean_distance(ee_pos, hover_target)

            # 采集当前相机图片。API 模式会把这张图发给多模态模型；
            # heuristic 模式虽然不靠图片决策，但保存图片便于人肉复盘。
            image_bgr = capture_rgb(camera_config, camera_eye)
            image_filename = os.path.join(output_dir, f"probe_step_{control_step:02d}.jpg")
            if probe_config["save_trace_images"]:
                cv2.imwrite(image_filename, image_bgr)

            # 方向决策是这份脚本的核心抽象：
            # 当前可以来自坐标规则，未来可以来自 VLA/多模态模型。
            direction, raw_response, decision_source = decide_direction(
                probe_config,
                image_bgr,
                ee_pos,
                block_pos,
            )

            # 把离散方向词转换成 PyBullet 世界坐标里的下一步目标点。
            # stop 时直接把目标设为红块上方悬停点；
            # 其他方向则从当前末端位置出发，在 x/y 平面移动一个小步长。
            if direction == "stop":
                target_pos = hover_target
            else:
                delta_x, delta_y = DIRECTION_TO_DELTA[direction]
                target_pos = [
                    ee_pos[0] + delta_x * probe_config["move_step_xy"],
                    ee_pos[1] + delta_y * probe_config["move_step_xy"],
                    hover_target[2],
                ]

            # IK 把“末端应该去哪里”转换成“7 个关节应该转到什么角度”。
            target_joint_angles = calculate_target_joints(robot_id, robot_config, target_pos)
            apply_joint_targets(robot_id, robot_config, target_joint_angles)

            # 每次决策后推进多个物理 step，让机械臂有时间朝目标运动。
            for _ in range(probe_config["sim_steps_per_action"]):
                p.stepSimulation()
                if config["enable_time_sleep"]:
                    time.sleep(1.0 / config["simulation_hz"])

            print(
                f"🧭 [PROBE] step={control_step} source={decision_source} "
                f"direction={direction} distance={distance_to_hover:.3f}"
            )
            # 记录这一控制步的完整上下文，后续可用来排查：
            # 模型判断错了、方向映射错了、还是 IK/物理控制没跟上。
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

            # 只有模型/规则说 stop，并且真实距离也满足 success_distance，
            # 才认为这次闭环探路成功结束。
            if direction == "stop" and distance_to_hover <= task_config["success_distance"]:
                print("✅ [PROBE] 模型已判定 stop，且末端已接近悬停目标。")
                break
    finally:
        # 不管中途是否报错，都清理 PyBullet 资源，避免下次运行残留物体。
        p.removeBody(block_id)
        p.disconnect()


if __name__ == "__main__":
    main()
