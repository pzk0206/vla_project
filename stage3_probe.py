"""Stage 3 单次在线闭环探路。

本文件负责一条 episode 内的“观察 -> 决策 -> 世界坐标小步目标 -> IK
-> 物理执行 -> 再观察”。run_probe_episode() 同时服务单次命令和批量评估，
避免 evaluate_probe.py 复制控制循环后产生行为漂移。
"""

import base64
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter

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


def apply_end_effector_marker(robot_id, robot_config, marker_config):
    """把真实末端 link 改成醒目的颜色，帮助 VLM 区分末端与中间关节。"""
    if not marker_config.get("enabled", False):
        return None
    p.changeVisualShape(
        robot_id,
        robot_config["ee_link_index"],
        rgbaColor=marker_config["color_rgba"],
    )
    visual_shape_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=marker_config["radius"],
        rgbaColor=marker_config["color_rgba"],
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape_id,
    )


def sync_end_effector_marker(marker_id, robot_id, link_index):
    """在拍照前把无碰撞视觉小球同步到真实末端，不影响物理控制。"""
    link_state = p.getLinkState(robot_id, link_index)
    p.resetBasePositionAndOrientation(
        marker_id,
        link_state[4],
        link_state[5],
    )


def combine_camera_views(primary_image, secondary_image):
    """添加面板标签和分隔线，再组成一张双视角模型输入图。"""
    if primary_image.shape[0] != secondary_image.shape[0]:
        raise ValueError("双视图高度必须相同")

    def add_header(image, label):
        # 独立标题栏不会盖住机械臂或红块；英文标签可由 OpenCV 稳定绘制。
        annotated = cv2.copyMakeBorder(
            image,
            24,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        cv2.putText(
            annotated,
            label,
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return annotated

    primary_panel = add_header(primary_image, "MAIN")
    secondary_panel = add_header(secondary_image, "AUX")
    primary_with_divider = cv2.copyMakeBorder(
        primary_panel,
        0,
        0,
        0,
        4,
        cv2.BORDER_CONSTANT,
        value=(0, 255, 255),
    )
    return cv2.hconcat([primary_with_divider, secondary_panel])


def direction_to_target(direction, ee_pos, hover_height, move_step_xy):
    """把世界坐标方向转换成单步末端目标位置。

    heuristic 和 API 都必须经过这个函数，确保日志里的方向词就是
    机械臂实际执行的 x/y 位移，而不是被某个决策来源的特殊分支绕过。
    """
    if direction not in DIRECTION_TO_DELTA:
        raise ValueError(f"未知方向: {direction}")
    delta_x, delta_y = DIRECTION_TO_DELTA[direction]
    return [
        ee_pos[0] + delta_x * move_step_xy,
        ee_pos[1] + delta_y * move_step_xy,
        hover_height,
    ]


def determine_probe_termination(
    distance_after,
    control_step,
    max_control_steps,
    success_distance,
):
    """返回当前 probe 控制步的终止状态。"""
    if distance_after <= success_distance:
        return "success"
    if control_step >= max_control_steps - 1:
        return "max_control_steps"
    return "running"


def joint_angle_errors(target_angles, actual_angles):
    """计算每个关节的目标误差，正负号定义为 target - actual。"""
    return [target - actual for target, actual in zip(target_angles, actual_angles)]


def get_controlled_joint_angles(robot_id, controlled_joints):
    """读取受控关节的实际角度，单位为弧度。"""
    return [p.getJointState(robot_id, joint_idx)[0] for joint_idx in range(controlled_joints)]


def ensure_dir(path):
    """确保输出目录存在。

    probe 运行时会把每一步图片和 trace 日志写进 outputs/probe/。
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


SCREEN_TO_WORLD = {
    "screen_right": "right",
    "screen_left": "left",
    "screen_up": "front",
    "screen_down": "back",
    "stop": "stop",
}


def build_vision_direction_prompt(instruction, include_stop):
    """构造在线、离线共用的图像方向 prompt。

    末端由机械臂连接结构确定，不再依赖颜色或它在画面中的高低位置。
    离线方向集不包含停止样本；在线闭环才启用停止分支。
    """
    lines = [
        f"任务指令：{instruction}",
        "你在控制 PyBullet 机械臂末端，使其移动到红色方块正上方。",
        "输入图片是一张单张正俯视图。",
        "先找到固定在地面上的机械臂底座，再沿相互连接的机械臂连杆逐节追踪；"
        "连接关系中最后一节的末梢才是真实末端。",
        "这里的最后一节由机械臂连接关系决定，不是画面中的直线距离、上下位置或颜色。",
        "不要把固定底座或中间关节当成末端。",
        "比较红色方块中心相对真实末端中心在画面中的位置。",
        "如果红块主要在末端右侧，输出 screen_right。",
        "如果红块主要在末端左侧，输出 screen_left。",
        "如果红块主要在末端下方，输出 screen_down。",
        "如果红块主要在末端上方，输出 screen_up。",
        "如果同时存在水平和垂直偏差，比较两者的像素距离绝对值，选择偏差更大的轴。",
    ]
    if include_stop:
        lines.extend(
            [
                "只有当真实末端中心与红色方块中心已经基本对齐时，才输出 stop。",
                "只能输出一个标签：screen_left、screen_right、screen_up、screen_down、stop。",
            ]
        )
    else:
        lines.append(
            "只能输出一个标签：screen_left、screen_right、screen_up、screen_down。"
        )
    lines.append("不要解释，不要输出标点或其他文字。")
    return "\n".join(lines)


def build_probe_prompt():
    """兼容旧调用：在线闭环使用带停止分支的共享 prompt。"""
    return build_vision_direction_prompt(
        "悬停在红色积木上方",
        include_stop=True,
    )


def parse_screen_direction(text):
    """从模型回复中提取自然图像方向标签。"""
    if not text:
        return None
    lowered = text.strip().lower()
    if lowered in SCREEN_TO_WORLD:
        return lowered
    match = re.search(
        r"\b(screen_left|screen_right|screen_up|screen_down|stop)\b",
        lowered,
    )
    return match.group(1) if match else None


def map_screen_to_world(screen_direction):
    """把图像坐标方向确定性转换成 PyBullet 世界动作。"""
    if screen_direction not in SCREEN_TO_WORLD:
        raise ValueError(f"未知图像方向: {screen_direction!r}")
    return SCREEN_TO_WORLD[screen_direction]


class InvalidModelResponseError(RuntimeError):
    """模型回复中不包含任何允许的离散方向。"""


def parse_direction(text):
    """从模型回复中提取合法方向；无法解析时返回 None。"""
    # 空回复不是 stop，而是一次无效模型回复。
    if not text:
        return None

    lowered = text.strip().lower()

    # 优先处理模型只返回一个方向词的标准情况。
    for direction in DIRECTION_TO_DELTA:
        if lowered == direction:
            return direction

    # 兼容 “move right” 之类带简短解释的回复。
    match = re.search(r"\b(left|right|front|back|stop)\b", lowered)
    if match:
        return match.group(1)

    # 没有合法方向时明确返回 None，避免伪装成 stop。
    return None


def call_openai_compatible_api(
    image_bgr,
    api_config,
    prompt_text=None,
    response_parser=parse_direction,
):
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
                    {
                        "type": "text",
                        "text": prompt_text or build_probe_prompt(),
                    },
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

    # max_retries 表示首次请求失败后最多额外尝试几次。
    for attempt in range(api_config["max_retries"] + 1):
        try:
            # 超时时间由 YAML 配置控制，避免网络策略写死在 Python 代码中。
            with urllib.request.urlopen(
                request,
                timeout=api_config["timeout_seconds"],
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            retryable = exc.code == 429 or 500 <= exc.code < 600

            # 429 和服务端 5xx 通常是瞬时错误，有剩余次数时才重试。
            if retryable and attempt < api_config["max_retries"]:
                continue
            raise RuntimeError(
                f"API 请求失败: {exc.code} {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            # 连接中断、DNS 等网络瞬时错误可以重试；次数耗尽后明确失败。
            if attempt < api_config["max_retries"]:
                continue
            raise RuntimeError(f"API 网络请求失败: {exc.reason}") from exc

    # 不同兼容服务返回的 content 可能是字符串，也可能是列表。
    # 这里统一整理成普通文本，再交给 parse_direction 解析方向。
    content = body["choices"][0]["message"]["content"]
    if isinstance(content, list):
        text_chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_chunks.append(item.get("text", ""))
        content = " ".join(text_chunks)

    # 请求成功不等于模型输出有效，必须单独验证离散方向。
    direction = response_parser(content)
    if direction is None:
        # 限制预览长度，既保留诊断线索，也避免长回复大量写入错误日志。
        response_preview = str(content).strip().replace("\n", " ")[:200]
        raise InvalidModelResponseError(
            f"模型回复不包含合法方向: {response_preview!r}"
        )

    return direction, content


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


def decide_direction(
    probe_config,
    image_bgr,
    ee_pos,
    block_pos,
    instruction="悬停在红色积木上方",
):
    """统一决策入口：根据配置选择真实 API 或启发式规则。

    main() 不关心方向来自哪里，只关心最终拿到：
    - direction: left/right/front/back/stop
    - raw_response: 原始回复或规则原因，方便写日志复盘
    - decision_source: api 或 heuristic
    """
    mode = probe_config["mode"].lower()
    if mode == "api":
        screen_direction, raw_response = call_openai_compatible_api(
            image_bgr,
            probe_config["api"],
            prompt_text=build_vision_direction_prompt(
                instruction,
                include_stop=True,
            ),
            response_parser=parse_screen_direction,
        )
        return map_screen_to_world(screen_direction), raw_response, "api"
    if mode == "heuristic":
        direction, raw_response = heuristic_direction(
            ee_pos,
            block_pos,
            probe_config["stop_distance_xy"],
        )
        return direction, raw_response, "heuristic"
    raise ValueError(
        f"probe.mode 只能是 heuristic 或 api，当前值: {probe_config['mode']}"
    )


def write_probe_trace(trace_jsonl_path, row):
    """把每一步闭环控制过程写入 JSONL。

    trace 的目的不是训练，而是调试：
    你可以看每一步模型/规则输出了什么方向、当时末端在哪里、
    红块在哪里、目标点在哪里、距离是否真的在变小。
    """
    with open(trace_jsonl_path, "a", encoding="utf-8") as trace_file:
        trace_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_probe_trace(rows, episode_idx, random_seed, trace_path):
    """把逐步 trace 压缩成一行 episode 摘要。

    rows 保存每个控制步的完整上下文；摘要只保留批量统计需要的信息：
    首末距离、终止原因、方向分布和距离变差次数。这样评估器无需再次理解
    PyBullet 控制细节，也能比较大量 episode。
    """
    if not rows:
        raise ValueError("probe trace 不能为空")
    last = rows[-1]
    return {
        "episode_idx": episode_idx,
        "random_seed": random_seed,
        "success": last["termination_reason"] == "success",
        "termination_reason": last["termination_reason"],
        "num_control_steps": len(rows),
        "initial_distance": rows[0]["distance_before"],
        "final_distance": last["distance_after"],
        "final_block_pos": list(last["block_pos"]),
        "direction_counts": dict(Counter(row["direction"] for row in rows)),
        "distance_increase_steps": sum(row["distance_delta"] < 0 for row in rows),
        "trace_path": str(trace_path),
        "error": None,
    }


def run_probe_episode(config, episode_idx, episode_dir, random_seed=None):
    """运行一次可复现的 Stage 3 闭环并返回 episode 摘要。

    Args:
        config: sim_config.yaml 加载后的完整配置。
        episode_idx: 当前 episode 编号，只用于输出和汇总定位。
        episode_dir: 本 episode 独立保存图片与 trace 的目录。
        random_seed: 控制红块位置和相机扰动；None 表示不固定随机性。

    Returns:
        summarize_probe_trace() 生成的字典，可直接写入 episode_summary.jsonl。

    一次运行只放一个红色积木，最多执行 max_control_steps 次控制：
    1. 拍一张当前相机图。
    2. 根据图片或坐标判断方向。
    3. 把方向转换成下一步目标位置。
    4. 用 IK 求关节角并下发给机械臂。
    5. 推进若干物理仿真 step。
    6. 写 probe_trace.jsonl 方便复盘。
    """
    # 固定种子后，相同 episode 可以在不同参数下复现，形成公平对照实验。
    if random_seed is not None:
        random.seed(random_seed)
    probe_config = config["probe"]
    robot_config = config["robot"]
    task_config = config["task"]
    camera_config = config["camera"]

    # 初始化仿真世界：连接 PyBullet、加载机械臂、加载红色积木。
    connect_physics(config["connection_mode"])
    _, robot_id = setup_world(config)
    # 标记只改变末端外观，不改变 IK、关节状态或标准方向标签。
    end_effector_marker_id = apply_end_effector_marker(
        robot_id,
        robot_config,
        probe_config["end_effector_marker"],
    )
    block_id = load_block(task_config)
    settle_object(config, task_config["initial_settle_steps"])

    # outputs/probe/ 是阶段三探路输出，不进入 Git 仓库。
    # 每次运行会覆盖旧 trace，图片文件名按 step 编号重写。
    output_dir = str(episode_dir)
    ensure_dir(output_dir)
    trace_jsonl_path = os.path.join(output_dir, "probe_trace.jsonl")
    if os.path.exists(trace_jsonl_path):
        os.remove(trace_jsonl_path)

    # 一次 probe 内相机固定，避免画面变化来自相机抖动。
    # 不同运行之间仍然可以有轻微随机视角，保持后续泛化空间。
    camera_eye = sample_camera_eye(camera_config)
    secondary_camera_config = probe_config.get("secondary_camera")
    secondary_camera_eye = (
        sample_camera_eye(secondary_camera_config)
        if secondary_camera_config
        else None
    )
    print(
        f"🎥 [PROBE] 阶段三探路相机位置: "
        f"X:{camera_eye[0]:.2f}, Y:{camera_eye[1]:.2f}, Z:{camera_eye[2]:.2f}"
    )

    # 内存中的 rows 用于最后生成摘要；同时每一步立即落盘，防止中途异常丢证据。
    trace_rows = []
    try:
        for control_step in range(probe_config["max_control_steps"]):
            # 读取当前红块位置，构造“红块正上方”的悬停目标点。
            block_pos = get_object_position(block_id)
            hover_target = [
                block_pos[0],
                block_pos[1],
                block_pos[2] + probe_config["hover_height"],
            ]
            # 读取动作前的末端位置和距离，后面再和动作后距离对比。
            ee_pos_before = get_link_position(robot_id, robot_config["ee_link_index"])
            distance_before = euclidean_distance(ee_pos_before, hover_target)

            # 采集当前相机图片。API 模式会把这张图发给多模态模型；
            # heuristic 模式虽然不靠图片决策，但保存图片便于人肉复盘。
            if end_effector_marker_id is not None:
                sync_end_effector_marker(
                    end_effector_marker_id,
                    robot_id,
                    robot_config["ee_link_index"],
                )
            image_bgr = capture_rgb(camera_config, camera_eye)
            # 左图提供稳定的方向坐标，右图在机械臂遮挡红块时补充观察。
            # 拼成一张图后仍然只需调用模型一次。
            if secondary_camera_config:
                secondary_image_bgr = capture_rgb(
                    secondary_camera_config,
                    secondary_camera_eye,
                )
                image_bgr = combine_camera_views(image_bgr, secondary_image_bgr)
            image_filename = os.path.join(output_dir, f"probe_step_{control_step:02d}.jpg")
            if probe_config["save_trace_images"]:
                cv2.imwrite(image_filename, image_bgr)

            # 方向决策是这份脚本的核心抽象：
            # 当前可以来自坐标规则，未来可以来自 VLA/多模态模型。
            direction, raw_response, decision_source = decide_direction(
                probe_config,
                image_bgr,
                ee_pos_before,
                block_pos,
                instruction=config["dataset"]["instruction"],
            )

            # heuristic 和 API 共用同一条执行路径：方向词决定真实的单步位移。
            # 这里的方向属于 PyBullet 世界坐标，不等同于相机图像中的左右前后。
            target_pos = direction_to_target(
                direction,
                ee_pos_before,
                hover_target[2],
                probe_config["move_step_xy"],
            )

            # IK 把“末端应该去哪里”转换成“7 个关节应该转到什么角度”。
            target_joint_angles = calculate_target_joints(robot_id, robot_config, target_pos)
            target_joint_angles = list(
                target_joint_angles[: robot_config["controlled_joints"]]
            )
            actual_joint_angles_before = get_controlled_joint_angles(
                robot_id,
                robot_config["controlled_joints"],
            )
            apply_joint_targets(robot_id, robot_config, target_joint_angles)

            # 每次决策后推进多个物理 step，让机械臂有时间朝目标运动。
            for _ in range(probe_config["sim_steps_per_action"]):
                p.stepSimulation()
                if config["enable_time_sleep"]:
                    time.sleep(1.0 / config["simulation_hz"])

            # 动作执行后重新读末端位置，用真实结果判断这一步是否有效。
            ee_pos_after = get_link_position(robot_id, robot_config["ee_link_index"])
            actual_joint_angles_after = get_controlled_joint_angles(
                robot_id,
                robot_config["controlled_joints"],
            )
            joint_error_after = joint_angle_errors(
                target_joint_angles,
                actual_joint_angles_after,
            )
            distance_after = euclidean_distance(ee_pos_after, hover_target)
            distance_delta = distance_before - distance_after
            termination_reason = determine_probe_termination(
                distance_after=distance_after,
                control_step=control_step,
                max_control_steps=probe_config["max_control_steps"],
                success_distance=task_config["success_distance"],
            )

            print(
                f"🧭 [PROBE] step={control_step} source={decision_source} "
                f"direction={direction} before={distance_before:.3f} "
                f"after={distance_after:.3f} delta={distance_delta:.3f}"
            )
            # 记录这一控制步的完整上下文，后续可用来排查：
            # 模型判断错了、方向映射错了、还是 IK/物理控制没跟上。
            trace_row = {
                "control_step": control_step,
                "decision_source": decision_source,
                "direction": direction,
                "raw_response": raw_response,
                "ee_pos": list(ee_pos_after),
                "ee_pos_before": list(ee_pos_before),
                "ee_pos_after": list(ee_pos_after),
                "block_pos": list(block_pos),
                "hover_target": list(hover_target),
                "target_pos": list(target_pos),
                "target_joint_angles": target_joint_angles,
                "actual_joint_angles_before": actual_joint_angles_before,
                "actual_joint_angles_after": actual_joint_angles_after,
                "joint_error_after": joint_error_after,
                "distance_to_hover": distance_after,
                "distance_before": distance_before,
                "distance_after": distance_after,
                "distance_delta": distance_delta,
                "termination_reason": termination_reason,
                "image_path": image_filename if probe_config["save_trace_images"] else None,
                # 离线 VLM 样本必须记录实际视角，才能复现方向语义问题。
                "camera_eye": list(camera_eye),
                "secondary_camera_eye": (
                    list(secondary_camera_eye) if secondary_camera_eye else None
                ),
            }
            write_probe_trace(trace_jsonl_path, trace_row)
            trace_rows.append(trace_row)

            # 终止依据真实动作结果，不强依赖方向词是否已经输出 stop。
            if termination_reason == "success":
                print("✅ [PROBE] 末端已接近悬停目标。")
                break
            if termination_reason == "max_control_steps":
                print(
                    "⚠️ [PROBE] 已达到最大控制步数，"
                    f"最终距离 {distance_after:.3f}m。"
                )
        return summarize_probe_trace(
            trace_rows,
            episode_idx,
            random_seed,
            trace_jsonl_path,
        )
    finally:
        # 每个批量 episode 都独立清理，防止前一次物体或连接污染后续结果。
        p.removeBody(block_id)
        p.disconnect()


def main():
    """保持 `python stage3_probe.py` 单次运行方式兼容。

    批量入口也调用 run_probe_episode，因此两种运行方式的控制行为完全一致。
    """
    config = load_config(CONFIG_PATH)
    run_probe_episode(config, 0, config["probe"]["output_dir"])


if __name__ == "__main__":
    main()
