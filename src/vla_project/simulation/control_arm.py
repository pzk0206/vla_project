import json
import math
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import yaml

from vla_project.simulation.camera_geometry import compute_camera_matrices


# =====================================================================
# control_arm.py
# ---------------------------------------------------------------------
# 这个脚本负责“按配置自动采集 VLA 训练数据”：
# 1. 从 sim_config.yaml 读取所有可调参数。
# 2. 在 PyBullet 中加载地面、机械臂、红色积木。
# 3. 每个 episode 随机生成一个积木位置和一个固定相机位置。
# 4. 用 IK 求解机械臂末端悬停到积木上方所需的关节角。
# 5. 按固定间隔保存 RGB 图片，同时把图片路径、语言指令、动作和调试状态写入 JSONL。
#
# 设计原则：
# - control_arm.py 只写“流程逻辑”，不硬编码实验参数。
# - sim_config.yaml 负责“实验参数”，比如相机、采样频率、episode 数、终止条件。
# - 这样以后调数据质量时，优先改 YAML，不需要频繁改 Python 主逻辑。
# =====================================================================
CONFIG_PATH = "sim_config.yaml"
DATASET_MANIFEST_NAME = "dataset_manifest.json"
CONFIG_SNAPSHOT_NAME = "config_snapshot.yaml"
MANIFEST_COMPATIBILITY_FIELDS = (
    "schema_version",
    "instruction",
    "action_dim",
    "random_seed",
    "seed_rule",
    "image_width",
    "image_height",
    "jsonl_name",
    "summary_jsonl_name",
)


def load_config(config_path):
    """读取外部 YAML 配置，让控制逻辑和可调参数彻底分离。

    YAML 是人类可读的配置格式。这里用它存放所有经常调的参数，
    比如相机高度、图片大小、episode 数、终止阈值等。
    """
    print(f"📂 正在读取外部配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def connect_physics(connection_mode):
    """根据配置选择 GUI 或 DIRECT 仿真模式。

    GUI:
        打开 PyBullet 图形窗口，适合本地观察机械臂运动是否正确。
    DIRECT:
        不开窗口，运行更快，适合批量采集训练数据或服务器运行。
    """
    mode = str(connection_mode).upper()
    if mode == "GUI":
        return p.connect(p.GUI)
    if mode == "DIRECT":
        return p.connect(p.DIRECT)
    raise ValueError(f"connection_mode 只能是 GUI 或 DIRECT，当前值: {connection_mode}")


def validate_versioned_dataset_dir(output_dir):
    """只接受 outputs/dataset 下的版本化子目录。"""
    path = Path(output_dir)
    root = Path("outputs/dataset")
    if path == root or root not in path.parents:
        raise ValueError(
            "dataset.output_dir 必须是 outputs/dataset/ 下的版本化子目录"
        )
    return path


def build_dataset_manifest(config):
    """从完整采集配置生成稳定的数据集契约。"""
    dataset = config["dataset"]
    camera = config["camera"]
    task = config["task"]
    return {
        "schema_version": dataset["schema_version"],
        "dataset_name": Path(dataset["output_dir"]).name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "instruction": dataset["instruction"],
        "action_dim": config["robot"]["controlled_joints"] + 2,
        "random_seed": dataset["random_seed"],
        "seed_rule": "random_seed + episode_idx",
        "image_width": camera["image_width"],
        "image_height": camera["image_height"],
        "block_position_range": task["block_position"],
        "camera_eye_offset_base": camera["eye_offset_base"],
        "camera_eye_offset_random_range": camera[
            "eye_offset_random_range"
        ],
        "pilot_num_episodes": dataset["pilot_num_episodes"],
        "target_num_episodes": dataset["target_num_episodes"],
        "jsonl_name": dataset["jsonl_name"],
        "summary_jsonl_name": dataset["summary_jsonl_name"],
    }


def _validate_append_compatibility(existing_manifest, current_manifest):
    mismatches = [
        field
        for field in MANIFEST_COMPATIBILITY_FIELDS
        if existing_manifest.get(field) != current_manifest.get(field)
    ]
    if mismatches:
        raise ValueError(
            "追加采集配置与现有 dataset manifest 不兼容: "
            + ", ".join(mismatches)
        )


def prepare_dataset(dataset_config, full_config=None):
    """按配置初始化数据集目录与 JSONL 文件路径。

    数据集结构采用：
        outputs/dataset/<version>/
          ep_0_step_0.jpg
          ep_0_step_24.jpg
          ...
          trajectory_expert.jsonl

    JSONL 是“一行一个 JSON 对象”的格式，适合大规模训练数据逐行读取。
    """
    dataset_path = validate_versioned_dataset_dir(dataset_config["output_dir"])
    dataset_dir = str(dataset_path)
    clean_before_run = dataset_config.get("clean_before_run", True)
    if clean_before_run and os.path.exists(dataset_dir):
        print(f"🗑️ 检测到旧数据，正在清空 {dataset_dir} 目录...")
        shutil.rmtree(dataset_dir)

    os.makedirs(dataset_dir, exist_ok=True)
    if full_config is not None:
        manifest = build_dataset_manifest(full_config)
        manifest_path = dataset_path / DATASET_MANIFEST_NAME
        snapshot_path = dataset_path / CONFIG_SNAPSHOT_NAME
        if not clean_before_run and manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as manifest_file:
                existing_manifest = json.load(manifest_file)
            _validate_append_compatibility(existing_manifest, manifest)
        else:
            with manifest_path.open("w", encoding="utf-8") as manifest_file:
                json.dump(
                    manifest,
                    manifest_file,
                    ensure_ascii=False,
                    indent=2,
                )
                manifest_file.write("\n")
            with snapshot_path.open("w", encoding="utf-8") as snapshot_file:
                yaml.safe_dump(
                    full_config,
                    snapshot_file,
                    allow_unicode=True,
                    sort_keys=False,
                )

    jsonl_path = os.path.join(dataset_dir, dataset_config["jsonl_name"])
    summary_jsonl_path = os.path.join(dataset_dir, dataset_config["summary_jsonl_name"])
    print(f"📁 数据集目录已就绪: {dataset_dir}")
    return dataset_dir, jsonl_path, summary_jsonl_path


def next_episode_index(summary_jsonl_path):
    """根据已有摘要返回追加采集应使用的下一个 episode 编号。"""
    if not os.path.exists(summary_jsonl_path):
        return 0

    episode_indices = []
    with open(summary_jsonl_path, "r", encoding="utf-8") as summary_file:
        for line in summary_file:
            if line.strip():
                episode_indices.append(json.loads(line)["episode_idx"])
    return max(episode_indices, default=-1) + 1


def setup_world(config):
    """加载物理世界、地面和机器人主体。

    这里是每次运行脚本只需要做一次的初始化：
    - 设置 PyBullet 自带资源路径，方便加载 plane.urdf、cube.urdf、kuka_iiwa。
    - 设置重力。
    - 加载地面。
    - 加载机械臂，并把机械臂 ID 返回给后续控制函数。
    """
    # PyBullet 自带很多常用模型，设置这个路径后就能直接用相对路径加载。
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # gravity 是 [x, y, z]，正常地球环境通常是 [0, 0, -9.8]。
    p.setGravity(*config["gravity"])

    # 地面不参与训练标签，但给视觉画面和物理碰撞提供环境参照。
    plane_id = p.loadURDF("plane.urdf")
    robot_cfg = config["robot"]

    # robot_id 是 PyBullet 后续控制这台机械臂的唯一句柄。
    robot_id = p.loadURDF(
        robot_cfg["urdf_path"],
        basePosition=robot_cfg["base_position"],
        useFixedBase=robot_cfg["use_fixed_base"],
    )

    print("🤖 [SUCCESS] 仿真世界初始化完成。")
    print(f"📍 机器人型号: {robot_cfg['urdf_path']}")
    print(f"📍 当前重力设定: {config['gravity']}")
    return plane_id, robot_id


def random_uniform_from_range(value_range):
    """从 [min, max] 范围中采样一个随机数。

    这个小函数让 YAML 里的区间配置可以直接被复用，
    例如积木 x/y 范围、相机扰动范围。
    """
    return random.uniform(value_range[0], value_range[1])


def sample_block_position(task_config):
    """根据配置随机生成积木位置。

    每个 episode 换一个积木位置，可以避免模型只记住固定坐标。
    当前只随机 x/y，z 固定，因为积木始终放在地面上。
    """
    pos_cfg = task_config["block_position"]
    return [
        random_uniform_from_range(pos_cfg["x_range"]),
        random_uniform_from_range(pos_cfg["y_range"]),
        pos_cfg["z"],
    ]


def load_block(task_config):
    """加载任务积木并设置视觉颜色。

    注意：每个 episode 都会新建一个 block，episode 结束后再 removeBody。
    这样不会出现多个积木堆在场景里，导致碰撞和视觉标签混乱。
    """
    block_id = p.loadURDF(
        task_config["block_urdf_path"],
        basePosition=sample_block_position(task_config),
        globalScaling=task_config["block_global_scaling"],
    )
    p.changeVisualShape(block_id, -1, rgbaColor=task_config["block_color_rgba"])
    return block_id


def load_second_block(task_cfg):
    """加载蓝色积木（第二目标物）并设置视觉颜色。

    每个 episode 都会和红色积木一起新建一个蓝色积木，
    episode 结束后随红块一起 removeBody。
    """
    second_cfg = task_cfg["second_block"]
    position = [
        random_uniform_from_range(second_cfg["x_range"]),
        random_uniform_from_range(second_cfg["y_range"]),
        second_cfg["z"],
    ]
    block_id = p.loadURDF(
        task_cfg["block_urdf_path"],
        basePosition=position,
        globalScaling=second_cfg.get(
            "global_scaling", task_cfg["block_global_scaling"]
        ),
    )
    p.changeVisualShape(block_id, -1, rgbaColor=second_cfg["color_rgba"])
    return block_id


def select_task(config):
    """按配置选择本 episode 的任务指令和目标积木颜色。

    新配置使用 task.tasks 列表随机挑选“悬停红块 / 悬停蓝块”之一；
    旧配置没有 tasks 列表时，回退到 dataset.instruction 和红色积木。
    返回 (instruction, target_block)，其中 target_block 为 "red" 或 "blue"。
    """
    task_cfg = config["task"]
    dataset_cfg = config["dataset"]
    tasks = task_cfg.get("tasks")
    if tasks:
        selected = random.choice(tasks)
        return selected["instruction"], selected["target_block"]
    return dataset_cfg.get("instruction", "悬停在红色积木上方"), "red"


def settle_object(config, num_steps):
    """让新加载的物体先在物理世界里稳定下来，再开始正式采集。"""
    for _ in range(num_steps):
        p.stepSimulation()
        if config["enable_time_sleep"]:
            time.sleep(1.0 / config["simulation_hz"])


def get_object_position(obj_id):
    """获取物体的当前世界坐标。

    PyBullet 返回的是 (position, orientation)，这里当前任务只需要位置。
    """
    position, _ = p.getBasePositionAndOrientation(obj_id)
    return position


def get_hover_target(block_id, hover_height):
    """生成位于积木正上方的安全悬停目标点。

    不能直接把机械臂末端移动到积木中心，否则会撞进积木。
    所以目标点使用：
        target_x = block_x
        target_y = block_y
        target_z = block_z + hover_height
    这就是“悬停在红色积木上方”的几何定义。
    """
    block_pos = get_object_position(block_id)
    return [block_pos[0], block_pos[1], block_pos[2] + hover_height]


def calculate_target_joints(robot_id, robot_config, target_pos):
    """通过 IK 计算机械臂到达目标点所需的关节角。

    IK(Inverse Kinematics，逆运动学)做的事情是：
        输入：末端执行器想去哪里、姿态是什么。
        输出：每个关节应该转到什么角度。

    target_orientation_euler 目前配置为 [0, pi, 0]，
    含义是让末端朝下，符合从上方靠近积木的任务。
    """
    target_quat = p.getQuaternionFromEuler(robot_config["target_orientation_euler"])
    controlled_joints = robot_config["controlled_joints"]
    lower_limits = []
    upper_limits = []
    rest_poses = []
    for joint_idx in range(controlled_joints):
        joint_info = p.getJointInfo(robot_id, joint_idx)
        lower_limits.append(joint_info[8])
        upper_limits.append(joint_info[9])
        rest_poses.append(p.getJointState(robot_id, joint_idx)[0])
    joint_ranges = [
        upper - lower for lower, upper in zip(lower_limits, upper_limits)
    ]
    return p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=robot_config["ee_link_index"],
        targetPosition=target_pos,
        targetOrientation=target_quat,
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=rest_poses,
        residualThreshold=robot_config["ik_residual_threshold"],
    )


def apply_joint_targets(robot_id, robot_config, target_joint_angles):
    """把 IK 结果下发给前 N 个主动关节。

    POSITION_CONTROL 表示“位置控制”：告诉关节目标角度，
    PyBullet 内部会用电机模型把关节慢慢推过去。

    max_force 和 max_velocity 会明显影响运动：
    - force 太小：机械臂可能推不动，看起来发软。
    - velocity 太小：运动很慢，连续图片变化不明显。
    - velocity 太大：动作跳变大，数据可能不够平滑。
    """
    for joint_idx in range(robot_config["controlled_joints"]):
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=joint_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_joint_angles[joint_idx],
            force=robot_config["max_force"],
            maxVelocity=robot_config["max_velocity"],
        )


def reset_robot_to_home(robot_id, robot_config, dataset_config):
    """把机械臂位置、速度和电机目标同步复位到固定 home pose。"""
    if not dataset_config["reset_robot_each_episode"]:
        return

    joint_count = robot_config["controlled_joints"]
    home = dataset_config["home_joint_positions"]
    if len(home) != joint_count:
        raise ValueError("home_joint_positions 长度必须等于 controlled_joints")

    for joint, position in enumerate(home):
        p.resetJointState(
            robot_id,
            joint,
            position,
            targetVelocity=0.0,
        )
    apply_joint_targets(robot_id, robot_config, home)


def get_link_position(robot_id, link_index):
    """读取指定 link 的世界坐标。

    这里用来读取末端执行器位置，之后和 target_pos 比距离，
    判断当前 episode 是否已经成功到达目标附近。
    """
    link_state = p.getLinkState(robot_id, link_index)
    return link_state[4]


def euclidean_distance(pos_a, pos_b):
    """计算两个 3D 点之间的欧氏距离。

    返回值单位和 PyBullet 世界坐标一致，通常可以理解为“米”。
    例如 0.015 表示 1.5 厘米。
    """
    return math.sqrt(sum((pos_a[idx] - pos_b[idx]) ** 2 for idx in range(3)))


def sample_camera_eye(camera_config):
    """每个 episode 锁定一个相机位置，并加入轻微域随机化。

    相机位置 = workspace_center + eye_offset_base + 随机扰动。

    这样做的目的：
    - 同一个 episode 内相机不动，方便模型理解连续动作。
    - 不同 episode 间视角略变，避免模型死记一个固定视角。
    """
    random_low, random_high = camera_config["eye_offset_random_range"]
    base_offset = camera_config["eye_offset_base"]
    workspace_center = camera_config["workspace_center"]
    eye_offset = [
        base_offset[axis] + random.uniform(random_low, random_high)
        for axis in range(3)
    ]
    return [workspace_center[axis] + eye_offset[axis] for axis in range(3)]


def capture_rgb_and_segmentation(camera_config, camera_eye):
    """使用同一相机矩阵采集 BGR 图片和物体分割掩码。

    PyBullet 相机需要两个矩阵：
    - view_matrix: 相机放在哪里、看向哪里、哪个方向算“上”。
    - projection_matrix: 视场角、近裁剪面、远裁剪面、宽高比。

    getCameraImage 返回 RGBA 和 segmentation；前者转成 OpenCV 常用的 BGR，
    后者保留 PyBullet 的 object/link 编码供离线可见率诊断使用。
    """
    # 渲染和后续反投影必须共用完全相同的相机矩阵。
    view_matrix, projection_matrix = compute_camera_matrices(
        camera_config, camera_eye
    )

    width, height, rgb_img, _, segmentation_img = p.getCameraImage(
        width=camera_config["image_width"],
        height=camera_config["image_height"],
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
    )
    rgb_array = np.reshape(rgb_img, (height, width, 4))[:, :, :3]
    segmentation = np.reshape(segmentation_img, (height, width))
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR), segmentation


def capture_rgb(camera_config, camera_eye):
    """从虚拟相机采集 BGR 图片，保持现有调用方接口不变。"""
    image_bgr, _ = capture_rgb_and_segmentation(camera_config, camera_eye)
    return image_bgr


def write_dataset_step(
    jsonl_path,
    schema_version,
    episode_idx,
    step_idx,
    random_seed,
    image_path,
    instruction,
    action,
    camera_eye,
    block_pos,
    target_pos,
    ee_pos,
    distance_to_target,
    termination_reason,
):
    """写入单帧 VLA 数据，保留调试所需的关键状态。

    每一行 JSON 对应一张图片和一个动作标签：
    - image_path: 当前视觉输入。
    - instruction: 当前语言指令。
    - action: 7 维关节目标 + 夹爪状态 + 终止标志。
    - camera_eye/block_pos/target_pos/ee_pos: 用来排查数据质量。
    - distance_to_target: 用来判断轨迹是否真的在接近目标。
    - termination_reason: 当前帧对应的结束原因，便于区分成功、卡住或跑满步数。
    """
    step_data = {
        "schema_version": schema_version,
        "episode_idx": episode_idx,
        "step_idx": step_idx,
        "random_seed": random_seed,
        "image_path": image_path,
        "instruction": instruction,
        "action": action,
        "camera_eye": camera_eye,
        "block_pos": list(block_pos),
        "target_pos": list(target_pos),
        "ee_pos": list(ee_pos),
        "distance_to_target": distance_to_target,
        "termination_reason": termination_reason,
    }
    with open(jsonl_path, "a", encoding="utf-8") as jsonl_file:
        jsonl_file.write(json.dumps(step_data, ensure_ascii=False) + "\n")


def write_episode_summary(
    summary_jsonl_path,
    schema_version,
    episode_idx,
    random_seed,
    initial_ee_pos,
    initial_block_pos,
    num_steps,
    num_frames,
    final_distance,
    termination_reason,
    camera_eye,
    final_block_pos,
    final_target_pos,
    final_ee_pos,
    task_instruction=None,
    target_block=None,
):
    """为每条轨迹写一行摘要，方便快速审计数据集质量。"""
    summary = {
        "schema_version": schema_version,
        "episode_idx": episode_idx,
        "random_seed": random_seed,
        "initial_ee_pos": list(initial_ee_pos),
        "initial_block_pos": list(initial_block_pos),
        "num_steps": num_steps,
        "num_frames": num_frames,
        "final_distance": final_distance,
        "termination_reason": termination_reason,
        "camera_eye": camera_eye,
        "final_block_pos": list(final_block_pos),
        "final_target_pos": list(final_target_pos),
        "final_ee_pos": list(final_ee_pos),
        "task_instruction": task_instruction,
        "target_block": target_block,
    }
    with open(summary_jsonl_path, "a", encoding="utf-8") as summary_file:
        summary_file.write(json.dumps(summary, ensure_ascii=False) + "\n")


def write_episode_error_summary(
    summary_jsonl_path,
    schema_version,
    episode_idx,
    random_seed,
    error,
    task_instruction=None,
    target_block=None,
):
    """记录单条 episode 异常，并保留未知状态为空值。"""
    summary = {
        "schema_version": schema_version,
        "episode_idx": episode_idx,
        "random_seed": random_seed,
        "initial_ee_pos": None,
        "initial_block_pos": None,
        "num_steps": 0,
        "num_frames": 0,
        "final_distance": None,
        "termination_reason": "episode_error",
        "camera_eye": None,
        "final_block_pos": None,
        "final_target_pos": None,
        "final_ee_pos": None,
        "task_instruction": task_instruction,
        "target_block": target_block,
        "error": error,
    }
    with open(summary_jsonl_path, "a", encoding="utf-8") as summary_file:
        summary_file.write(json.dumps(summary, ensure_ascii=False) + "\n")


def should_capture(step_idx, capture_interval, terminate_episode):
    """固定频率采图，同时保证终止帧一定被采到。

    例如 capture_interval=24，表示每 24 个物理仿真 step 保存一张图。
    即使当前 step 不是 24 的倍数，只要 episode 终止，也强制保存最后一帧。
    """
    return step_idx % capture_interval == 0 or terminate_episode == 1


def determine_termination(
    distance_to_target,
    recent_distances,
    step_idx,
    success_distance,
    stuck_window_steps,
    stuck_min_improvement,
    force_terminal_after_step,
):
    """按成功、卡住、最大步数的优先级返回 episode 状态。"""
    if distance_to_target <= success_distance:
        return "success"

    if len(recent_distances) == stuck_window_steps:
        improvement = recent_distances[0] - recent_distances[-1]
        if improvement < stuck_min_improvement:
            return "stuck"

    # 最大步数必须独立判断，不能被“窗口已满但仍在改善”的分支遮住。
    if step_idx >= force_terminal_after_step:
        return "max_steps"
    return "running"


def run_episode(
    episode_idx,
    robot_id,
    config,
    dataset_dir,
    jsonl_path,
    summary_jsonl_path,
    random_seed,
):
    """执行一条任务轨迹并写入图片/JSONL 数据。

    一个 episode 的生命周期：
    1. 随机选择本 episode 的任务：悬停红块或悬停蓝块。
    2. 同时加载红色积木和（若启用）蓝色积木，并生成固定相机位置。
    3. 循环执行“感知目标 -> IK 求解 -> 电机控制 -> 物理步进 -> 采图写标签”。
    4. 成功到达或达到最大步数后终止。
    5. 删除两个积木，避免影响下一个 episode。
    """
    robot_cfg = config["robot"]
    task_cfg = config["task"]
    dataset_cfg = config["dataset"]
    camera_cfg = config["camera"]

    random.seed(random_seed)
    reset_robot_to_home(robot_id, robot_cfg, dataset_cfg)
    for _ in range(task_cfg["initial_settle_steps"]):
        p.stepSimulation()
    initial_ee_pos = get_link_position(
        robot_id,
        robot_cfg["ee_link_index"],
    )

    # 每条轨迹随机挑选一个任务（悬停红块 / 悬停蓝块），
    # 并同时加载红色积木和（若启用）蓝色积木。
    instruction, target_block = select_task(config)
    red_block_id = load_block(task_cfg)
    blue_block_id = (
        load_second_block(task_cfg)
        if task_cfg.get("second_block", {}).get("enabled")
        else None
    )
    settle_object(config, task_cfg["initial_settle_steps"])
    initial_block_pos = get_object_position(red_block_id)
    target_block_id = (
        blue_block_id
        if target_block == "blue" and blue_block_id is not None
        else red_block_id
    )

    # 同一个 episode 内固定相机，有利于形成稳定的时序视觉输入。
    camera_eye = sample_camera_eye(camera_cfg)
    recent_distances = []
    saved_frame_count = 0
    termination_reason = "running"
    final_distance = None
    final_block_pos = None
    final_target_pos = None
    final_ee_pos = None
    print(
        f"🎥 [CAMERA] episode {episode_idx} 相机位置: "
        f"X:{camera_eye[0]:.2f}, Y:{camera_eye[1]:.2f}, Z:{camera_eye[2]:.2f}"
    )

    try:
        for step_idx in range(dataset_cfg["max_steps_per_episode"]):
            # 阶段 A：根据当前积木位置，计算“悬停点”。
            target_pos = get_hover_target(target_block_id, task_cfg["hover_height"])

            # 阶段 B：把悬停点转换成机械臂 7 个关节的目标角度。
            target_joint_angles = calculate_target_joints(robot_id, robot_cfg, target_pos)

            # 阶段 C：把目标角度下发给 PyBullet 电机控制器。
            apply_joint_targets(robot_id, robot_cfg, target_joint_angles)

            # 阶段 D：推进一小步物理仿真，让机械臂真的开始运动。
            p.stepSimulation()
            if config["enable_time_sleep"]:
                time.sleep(1.0 / config["simulation_hz"])

            # 阶段 E：读取末端位置，计算它离悬停目标还差多远。
            ee_pos = get_link_position(robot_id, robot_cfg["ee_link_index"])
            distance_to_target = euclidean_distance(ee_pos, target_pos)
            block_pos = get_object_position(target_block_id)
            final_distance = distance_to_target
            final_block_pos = block_pos
            final_target_pos = target_pos
            final_ee_pos = ee_pos

            # 最近一段距离历史用于判断“看起来还在跑，但其实已经卡住”。
            recent_distances.append(distance_to_target)
            if len(recent_distances) > task_cfg["stuck_window_steps"]:
                recent_distances.pop(0)

            # 成功优先级最高；其次是卡住；最后才是最大步数兜底。
            termination_reason = determine_termination(
                distance_to_target=distance_to_target,
                recent_distances=recent_distances,
                step_idx=step_idx,
                success_distance=task_cfg["success_distance"],
                stuck_window_steps=task_cfg["stuck_window_steps"],
                stuck_min_improvement=task_cfg["stuck_min_improvement"],
                force_terminal_after_step=task_cfg["force_terminal_after_step"],
            )
            terminate_episode = int(termination_reason != "running")

            if terminate_episode:
                print(
                    f"🏁 [TERMINUS] episode {episode_idx} 结束原因: {termination_reason}，"
                    f"距目标 {distance_to_target * 100:.2f} 厘米。"
                )

            if should_capture(
                step_idx,
                dataset_cfg["capture_interval_steps"],
                terminate_episode,
            ):
                # 图片文件名包含 episode 和 step，方便从文件名反查轨迹位置。
                image_filename = f"ep_{episode_idx}_step_{step_idx}.jpg"
                image_path = os.path.join(dataset_dir, image_filename)
                cv2.imwrite(image_path, capture_rgb(camera_cfg, camera_eye))
                saved_frame_count += 1

                # action 当前定义：
                # 前 7 维：IK 求出来的关节目标角。
                # 第 8 维：夹爪状态，当前任务只悬停，所以保持默认值。
                # 第 9 维：是否终止，用于训练时识别 episode 结束。
                action = (
                    list(target_joint_angles[: robot_cfg["controlled_joints"]])
                    + [dataset_cfg["default_gripper_state"], terminate_episode]
                )
                write_dataset_step(
                    jsonl_path=jsonl_path,
                    schema_version=dataset_cfg["schema_version"],
                    episode_idx=episode_idx,
                    step_idx=step_idx,
                    random_seed=random_seed,
                    image_path=image_path,
                    instruction=instruction,
                    action=action,
                    camera_eye=camera_eye,
                    block_pos=block_pos,
                    target_pos=target_pos,
                    ee_pos=ee_pos,
                    distance_to_target=distance_to_target,
                    termination_reason=termination_reason,
                )
                print(f"📝 [ALIGN] 已追加数据: {image_path}")

            if terminate_episode:
                print(f"🛑 [BREAK] episode {episode_idx} 实际迭代 {step_idx + 1} 步。")
                break

        write_episode_summary(
            summary_jsonl_path=summary_jsonl_path,
            schema_version=dataset_cfg["schema_version"],
            episode_idx=episode_idx,
            random_seed=random_seed,
            initial_ee_pos=initial_ee_pos,
            initial_block_pos=initial_block_pos,
            num_steps=step_idx + 1,
            num_frames=saved_frame_count,
            final_distance=final_distance,
            termination_reason=termination_reason,
            camera_eye=camera_eye,
            final_block_pos=final_block_pos,
            final_target_pos=final_target_pos,
            final_ee_pos=final_ee_pos,
            task_instruction=instruction,
            target_block=target_block,
        )
    finally:
        # 无论 episode 正常结束还是中途报错，都尽量清理当前两个积木。
        for block_id in (red_block_id, blue_block_id):
            if block_id is not None:
                p.removeBody(block_id)


def main():
    """主入口：按顺序完成配置读取、仿真初始化、批量采集和断开连接。"""
    config = load_config(CONFIG_PATH)
    connect_physics(config["connection_mode"])
    dataset_dir, jsonl_path, summary_jsonl_path = prepare_dataset(
        config["dataset"],
        full_config=config,
    )
    _, robot_id = setup_world(config)

    start_episode_idx = next_episode_index(summary_jsonl_path)
    end_episode_idx = start_episode_idx + config["dataset"]["num_episodes"]
    for episode_idx in range(start_episode_idx, end_episode_idx):
        random_seed = config["dataset"]["random_seed"] + episode_idx
        try:
            run_episode(
                episode_idx=episode_idx,
                robot_id=robot_id,
                config=config,
                dataset_dir=dataset_dir,
                jsonl_path=jsonl_path,
                summary_jsonl_path=summary_jsonl_path,
                random_seed=random_seed,
            )
        except Exception as exc:
            write_episode_error_summary(
                summary_jsonl_path=summary_jsonl_path,
                schema_version=config["dataset"]["schema_version"],
                episode_idx=episode_idx,
                random_seed=random_seed,
                error=repr(exc),
                task_instruction=None,
                target_block=None,
            )

    p.disconnect()
    print("✅ 数据采集完成。")


if __name__ == "__main__":
    main()
