"""收集并校验用于离线 VLM 方向评估的视觉样本。

脚本复用 Stage 3 heuristic 闭环，先生成逐步 trace，再从每个 episode 中均匀
选取少量代表帧。这样标签来自已经验证过的 heuristic 决策，同时避免连续图片
高度重复。
"""

import copy
import json
import math
import random
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pybullet as p

from camera_geometry import compute_camera_matrices
from control_arm import (
    CONFIG_PATH,
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
from stage3_probe import (
    apply_end_effector_marker,
    heuristic_direction,
    run_probe_episode,
    sync_end_effector_marker,
)


# VLM 和机械臂控制层共同使用的离散动作集合。
VALID_DIRECTIONS = {"left", "right", "front", "back", "stop"}
BALANCED_DIRECTION_ORDER = ("left", "right", "front", "back")


def build_sample_diagnostic(
    sample_id,
    block_pos,
    camera_eye,
    camera_config,
    visibility_metrics=None,
):
    """构造只供离线评分使用的仿真真值与相机快照。"""
    view_matrix, projection_matrix = compute_camera_matrices(
        camera_config, camera_eye
    )
    diagnostic = {
        "sample_id": sample_id,
        "block_pos": [float(value) for value in block_pos],
        "camera_eye": [float(value) for value in camera_eye],
        "image_width": int(camera_config["image_width"]),
        "image_height": int(camera_config["image_height"]),
        "view_matrix": [float(value) for value in view_matrix],
        "projection_matrix": [float(value) for value in projection_matrix],
    }
    if visibility_metrics is not None:
        diagnostic.update(visibility_metrics)
    return diagnostic


def compute_block_visibility_metrics(
    visible_segmentation,
    reference_segmentation,
    body_id,
):
    """比较当前画面与移除机械臂后的分割掩码，计算红块可见率。"""
    if isinstance(body_id, bool) or not isinstance(body_id, int) or body_id < 0:
        raise ValueError("body_id 必须是非负整数")

    object_id_mask = (1 << 24) - 1

    def count_body_pixels(segmentation):
        object_ids = np.asarray(segmentation, dtype=np.int64) & object_id_mask
        return int(np.count_nonzero(object_ids == body_id))

    visible_pixels = count_body_pixels(visible_segmentation)
    reference_pixels = count_body_pixels(reference_segmentation)
    if reference_pixels <= 0:
        raise ValueError("红块参考像素必须大于 0")
    if visible_pixels > reference_pixels:
        raise ValueError(
            "红块可见像素不能大于参考像素: "
            f"visible={visible_pixels}, reference={reference_pixels}"
        )
    return {
        "block_visible_pixels": visible_pixels,
        "block_reference_pixels": reference_pixels,
        "block_visibility_ratio": visible_pixels / reference_pixels,
    }


def validate_diagnostics(rows):
    """拒绝无法和 VLM 样本可靠连接的诊断行。"""
    seen = set()
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("诊断行缺少非空 sample_id")
        if sample_id in seen:
            raise ValueError(f"诊断 sample_id 重复: {sample_id}")
        seen.add(sample_id)
        if len(row.get("block_pos", [])) != 3:
            raise ValueError(f"诊断 block_pos 非法: {sample_id}")
        if len(row.get("view_matrix", [])) != 16:
            raise ValueError(f"诊断 view_matrix 非法: {sample_id}")
        if len(row.get("projection_matrix", [])) != 16:
            raise ValueError(f"诊断 projection_matrix 非法: {sample_id}")
        visibility_fields = {
            "block_visible_pixels",
            "block_reference_pixels",
            "block_visibility_ratio",
        }
        present_fields = visibility_fields & row.keys()
        if present_fields and present_fields != visibility_fields:
            raise ValueError(f"诊断可见率字段不完整: {sample_id}")
        if present_fields:
            visible = row["block_visible_pixels"]
            reference = row["block_reference_pixels"]
            ratio = row["block_visibility_ratio"]
            if (
                isinstance(visible, bool)
                or not isinstance(visible, int)
                or visible < 0
            ):
                raise ValueError(f"诊断可见像素非法: {sample_id}")
            if (
                isinstance(reference, bool)
                or not isinstance(reference, int)
                or reference <= 0
            ):
                raise ValueError(f"诊断参考像素非法: {sample_id}")
            if visible > reference:
                raise ValueError(f"诊断可见像素超过参考像素: {sample_id}")
            if (
                isinstance(ratio, bool)
                or not isinstance(ratio, (int, float))
                or not math.isfinite(ratio)
                or not 0.0 <= ratio <= 1.0
                or not math.isclose(ratio, visible / reference, abs_tol=1e-12)
            ):
                raise ValueError(f"诊断可见率非法: {sample_id}")


def write_sample_files(output_dir, samples, diagnostics):
    """分别写入可发送样本和仅供评分的诊断真值。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_diagnostics(diagnostics)
    sample_ids = {row["sample_id"] for row in samples}
    diagnostic_ids = {row["sample_id"] for row in diagnostics}
    if sample_ids != diagnostic_ids:
        raise ValueError("samples 与 diagnostics 的 sample_id 不一致")
    manifest_path = output_dir / "samples.jsonl"
    diagnostics_path = output_dir / "diagnostics.jsonl"
    for path, rows in (
        (manifest_path, samples),
        (diagnostics_path, diagnostics),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return manifest_path, diagnostics_path

# 每条离线样本必须提供的最小字段；诊断字段以后可以额外增加。
REQUIRED_SAMPLE_FIELDS = {
    "sample_id",
    "image_path",
    "instruction",
    "expected_direction",
    "random_seed",
    "control_step",
    "camera_eye",
}


def validate_samples(samples, project_root="."):
    """校验样本字段、方向、图片可读性和 ID 唯一性。

    Args:
        samples: 由样本字典组成的可迭代对象。
        project_root: ``image_path`` 相对路径所基于的项目根目录。

    Raises:
        ValueError: 任意样本违反最小数据契约时立即抛出，并指出问题字段。
    """
    root = Path(project_root)
    seen_sample_ids = set()

    for index, sample in enumerate(samples):
        # 先检查字段，避免后续直接索引时只得到含义不清楚的 KeyError。
        missing_fields = REQUIRED_SAMPLE_FIELDS - sample.keys()
        if missing_fields:
            missing_text = ", ".join(sorted(missing_fields))
            raise ValueError(f"样本 {index} 缺少必需字段: {missing_text}")

        direction = sample["expected_direction"]
        if direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"样本 {index} 的 expected_direction 非法: {direction!r}"
            )

        sample_id = sample["sample_id"]
        if sample_id in seen_sample_ids:
            raise ValueError(f"sample_id 重复: {sample_id!r}")
        seen_sample_ids.add(sample_id)

        # 文件存在仍可能是空文件或损坏文件，所以必须让 OpenCV 实际解码一次。
        image_path = root / sample["image_path"]
        if not image_path.is_file() or cv2.imread(str(image_path)) is None:
            raise ValueError(f"样本 {index} 的 image_path 不可读取: {image_path}")


def select_evenly_spaced_rows(rows, max_samples):
    """从 trace 中均匀选择不超过 ``max_samples`` 行，并保留首尾。

    连续控制步的画面通常非常相似。均匀选择比只取开头若干帧更能覆盖机械臂从
    初始位置逐渐接近积木的完整过程。
    """
    rows = list(rows)
    if max_samples <= 0:
        raise ValueError("max_samples 必须大于 0")
    if len(rows) <= max_samples:
        return rows
    if max_samples == 1:
        return [rows[0]]

    last_index = len(rows) - 1
    indices = [
        round(position * last_index / (max_samples - 1))
        for position in range(max_samples)
    ]
    return [rows[index] for index in indices]


def build_balanced_ee_positions(block_pos, hover_height, offset_xy):
    """生成五类标签对应的末端世界坐标位置。"""
    block_x, block_y, block_z = block_pos
    hover_z = block_z + hover_height
    return {
        "left": [round(block_x + offset_xy, 10), block_y, hover_z],
        "right": [round(block_x - offset_xy, 10), block_y, hover_z],
        "front": [block_x, round(block_y - offset_xy, 10), hover_z],
        "back": [block_x, round(block_y + offset_xy, 10), hover_z],
        "stop": [block_x, block_y, hover_z],
    }


def build_stratified_balanced_cases(offsets_xy):
    """按距离从远到近生成四方向分层评估案例。"""
    cases = []
    for offset_xy in offsets_xy:
        offset_tag = f"d{round(offset_xy * 100):03d}"
        for direction in BALANCED_DIRECTION_ORDER:
            cases.append(
                {
                    "offset_xy": offset_xy,
                    "offset_tag": offset_tag,
                    "direction": direction,
                }
            )
    return cases


def validate_stratified_seeds(value):
    """返回显式分层 seeds，并拒绝空值、重复值、布尔值和非整数。"""
    if not isinstance(value, list) or not value:
        raise ValueError("vlm_evaluation.stratified_seeds 必须是非空列表")
    if any(type(seed) is not int for seed in value):
        raise ValueError("vlm_evaluation.stratified_seeds 只能包含整数")
    if len(set(value)) != len(value):
        raise ValueError("vlm_evaluation.stratified_seeds 不能重复")
    return list(value)


def reset_robot_to_target(
    robot_id,
    robot_config,
    target_pos,
    tolerance,
    max_iterations,
):
    """迭代 IK 并重置关节，直到静态末端姿态收敛到指定位置。"""
    actual_pos = None
    for _ in range(max_iterations):
        target_joints = calculate_target_joints(robot_id, robot_config, target_pos)
        for joint_idx, joint_angle in enumerate(
            target_joints[: robot_config["controlled_joints"]]
        ):
            p.resetJointState(robot_id, joint_idx, joint_angle)
        actual_pos = get_link_position(robot_id, robot_config["ee_link_index"])
        distance = sum(
            (actual_pos[axis] - target_pos[axis]) ** 2 for axis in range(3)
        ) ** 0.5
        if distance <= tolerance:
            break
    return actual_pos


def capture_balanced_pose_sample(
    config,
    direction,
    seed,
    image_path,
    offset_xy=None,
):
    """把末端放到红块指定一侧，拍摄一张真实 PyBullet 均衡样本。"""
    random.seed(seed)
    robot_config = config["robot"]
    probe_config = config["probe"]
    task_config = config["task"]
    camera_config = config["camera"]

    connect_physics(config["connection_mode"])
    try:
        _, robot_id = setup_world(config)
        marker_id = apply_end_effector_marker(
            robot_id,
            robot_config,
            probe_config["end_effector_marker"],
        )
        block_id = load_block(task_config)
        settle_object(config, task_config["initial_settle_steps"])
        block_pos = get_object_position(block_id)
        if offset_xy is None:
            offset_xy = config["vlm_evaluation"]["balanced_pose_offset_xy"]
        target_positions = build_balanced_ee_positions(
            block_pos,
            probe_config["hover_height"],
            offset_xy,
        )

        # 离线基准只需要合法静态姿态；直接设置 IK 关节角避免运动路径干扰画面。
        ee_pos = reset_robot_to_target(
            robot_id,
            robot_config,
            target_positions[direction],
            tolerance=config["vlm_evaluation"]["balanced_pose_tolerance"],
            max_iterations=config["vlm_evaluation"]["balanced_pose_ik_iterations"],
        )
        actual_direction, _ = heuristic_direction(
            ee_pos,
            block_pos,
            probe_config["stop_distance_xy"],
        )
        if actual_direction != direction:
            raise RuntimeError(
                f"均衡姿态未达到预期方向: expected={direction}, "
                f"actual={actual_direction}, ee_pos={ee_pos}, block_pos={block_pos}"
            )

        if marker_id is not None:
            sync_end_effector_marker(
                marker_id,
                robot_id,
                robot_config["ee_link_index"],
            )
        camera_eye = sample_camera_eye(camera_config)
        image_bgr, visible_segmentation = capture_rgb_and_segmentation(
            camera_config,
            camera_eye,
        )
        if not cv2.imwrite(str(image_path), image_bgr):
            raise RuntimeError(f"无法写入均衡样本图片: {image_path}")

        # 参考分割只用于测量红块完整投影面积，不保存参考 RGB，也不发送给 VLM。
        if marker_id is not None:
            p.removeBody(marker_id)
        p.removeBody(robot_id)
        _, reference_segmentation = capture_rgb_and_segmentation(
            camera_config,
            camera_eye,
        )
        visibility_metrics = compute_block_visibility_metrics(
            visible_segmentation,
            reference_segmentation,
            block_id,
        )
        return {
            "camera_eye": list(camera_eye),
            "block_pos": list(block_pos),
            **visibility_metrics,
        }
    finally:
        p.disconnect()


def build_sampling_config(config):
    """复制完整配置，并只为 VLM 采样覆盖相机和图片保存选项。"""
    sampling_config = copy.deepcopy(config)
    sampling_config["probe"]["mode"] = "heuristic"
    sampling_config["probe"]["save_trace_images"] = True

    # 双视角只增加视觉信息，不改变 heuristic 产生标准答案的方式。
    # 左右视图从相反方向倾斜俯视，但保持相同图像坐标，避免斜视透视歧义。
    if sampling_config["vlm_evaluation"].get("use_dual_view", False):
        secondary_camera = copy.deepcopy(config["camera"])
        secondary_camera.update(
            sampling_config["vlm_evaluation"]["secondary_camera_override"]
        )
        sampling_config["probe"]["secondary_camera"] = secondary_camera
    sampling_config["camera"].update(
        sampling_config["vlm_evaluation"]["camera_override"]
    )
    if sampling_config["vlm_evaluation"].get("sample_strategy") in {
        "balanced_poses",
        "stratified_balanced_poses",
    }:
        sampling_config["task"]["block_position"].update(
            sampling_config["vlm_evaluation"]["balanced_block_position"]
        )
    return sampling_config


def read_jsonl(path):
    """读取一份 JSONL trace，忽略纯空行。"""
    with Path(path).open("r", encoding="utf-8") as jsonl_file:
        return [json.loads(line) for line in jsonl_file if line.strip()]


def collect_vlm_eval_samples(config):
    """运行固定种子的 heuristic probe 并写出离线视觉方向样本。"""
    if config["probe"]["mode"].lower() != "heuristic":
        raise ValueError("离线标准标签只能在 probe.mode: heuristic 下收集")

    evaluation = config["vlm_evaluation"]
    output_dir = Path(evaluation["sample_output_dir"])
    images_dir = output_dir / "images"
    # 该目录只包含可重新生成的评估样本。每次清理可避免旧图片和新 manifest 混用。
    if output_dir.exists():
        shutil.rmtree(output_dir)
    images_dir.mkdir(parents=True)

    # 使用独立俯视相机并强制保存图片，不修改 baseline 的原始配置字典。
    sampling_config = build_sampling_config(config)
    camera_config = sampling_config["camera"]

    base_seed = config["probe_evaluation"]["random_seed"]
    instruction = config["dataset"]["instruction"]
    samples = []
    diagnostics = []

    # 均衡模式主动构造相对位置；分层模式额外覆盖远、中、近三档距离。
    if evaluation.get("sample_strategy") in {
        "balanced_poses",
        "stratified_balanced_poses",
    }:
        if evaluation["sample_strategy"] == "stratified_balanced_poses":
            cases = build_stratified_balanced_cases(
                evaluation["balanced_pose_offsets_xy"]
            )
            seeds = validate_stratified_seeds(evaluation["stratified_seeds"])
        else:
            cases = [
                {
                    "offset_xy": evaluation["balanced_pose_offset_xy"],
                    "offset_tag": None,
                    "direction": direction,
                }
                for direction in BALANCED_DIRECTION_ORDER
            ]
            seeds = [
                base_seed + episode_idx
                for episode_idx in range(evaluation["offline_num_episodes"])
            ]

        for seed in seeds:
            for case_index, case in enumerate(cases):
                direction = case["direction"]
                if case["offset_tag"]:
                    sample_id = f"seed_{seed}_{case['offset_tag']}_{direction}"
                else:
                    sample_id = f"seed_{seed}_{direction}"
                relative_image_path = images_dir / f"{sample_id}.jpg"
                captured = capture_balanced_pose_sample(
                    sampling_config,
                    direction,
                    seed,
                    relative_image_path,
                    offset_xy=case["offset_xy"],
                )
                camera_eye = captured["camera_eye"]
                samples.append(
                    {
                        "sample_id": sample_id,
                        "image_path": relative_image_path.as_posix(),
                        "instruction": instruction,
                        "expected_direction": direction,
                        "offset_xy": case["offset_xy"],
                        "random_seed": seed,
                        "control_step": case_index,
                        "camera_eye": list(camera_eye),
                    }
                )
                diagnostics.append(
                    build_sample_diagnostic(
                        sample_id,
                        captured["block_pos"],
                        camera_eye,
                        camera_config,
                        visibility_metrics={
                            "block_visible_pixels": captured[
                                "block_visible_pixels"
                            ],
                            "block_reference_pixels": captured[
                                "block_reference_pixels"
                            ],
                            "block_visibility_ratio": captured[
                                "block_visibility_ratio"
                            ],
                        },
                    )
                )

        validate_samples(samples, Path.cwd())
        manifest_path, _ = write_sample_files(output_dir, samples, diagnostics)
        return manifest_path, samples

    # 原始逐步图片只用于挑选，放在临时目录中并在结束后自动清理。
    with tempfile.TemporaryDirectory(prefix="vlm_eval_probe_") as temp_dir:
        temp_root = Path(temp_dir)
        for episode_idx in range(evaluation["offline_num_episodes"]):
            seed = base_seed + episode_idx
            episode_dir = temp_root / f"episode_{episode_idx:03d}"
            summary = run_probe_episode(
                sampling_config,
                episode_idx,
                episode_dir,
                random_seed=seed,
            )
            trace_rows = read_jsonl(summary["trace_path"])
            selected_rows = select_evenly_spaced_rows(
                trace_rows,
                evaluation["max_samples_per_episode"],
            )

            for row in selected_rows:
                control_step = row["control_step"]
                sample_id = f"seed_{seed}_step_{control_step:03d}"
                relative_image_path = images_dir / f"{sample_id}.jpg"
                shutil.copy2(row["image_path"], relative_image_path)

                # block_pos/ee_pos 等仿真真值不进入评估样本，防止后续误发给 VLM。
                samples.append(
                    {
                        "sample_id": sample_id,
                        "image_path": relative_image_path.as_posix(),
                        "instruction": instruction,
                        "expected_direction": row["direction"],
                        "random_seed": seed,
                        "control_step": control_step,
                        "camera_eye": row["camera_eye"],
                    }
                )
                diagnostics.append(
                    build_sample_diagnostic(
                        sample_id,
                        row["block_pos"],
                        row["camera_eye"],
                        camera_config,
                    )
                )

    # 先用同一套契约校验全部样本，通过后才写最终 manifest。
    validate_samples(samples, Path.cwd())
    manifest_path, _ = write_sample_files(output_dir, samples, diagnostics)

    return manifest_path, samples


def main():
    """命令行入口：读取 YAML、运行采样并报告实际样本数。"""
    config = load_config(CONFIG_PATH)
    manifest_path, samples = collect_vlm_eval_samples(config)
    print(f"✅ 离线 VLM 样本已生成: {manifest_path}")
    print(f"样本数量: {len(samples)}")


if __name__ == "__main__":
    main()
