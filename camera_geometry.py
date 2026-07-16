"""PyBullet/OpenGL 相机投影与工作平面反投影纯几何。"""

import math

import numpy as np
import pybullet as p


_EPSILON = 1e-9


def _positive_image_size(image_width, image_height):
    if (
        isinstance(image_width, bool)
        or isinstance(image_height, bool)
        or not isinstance(image_width, (int, float))
        or not isinstance(image_height, (int, float))
        or not math.isfinite(image_width)
        or not math.isfinite(image_height)
        or image_width <= 1
        or image_height <= 1
    ):
        raise ValueError("图片宽高必须是大于 1 的有限数值")


def _matrix4(values, name):
    array = np.asarray(values, dtype=float)
    if array.size != 16:
        raise ValueError(f"{name} 必须包含 16 个数值")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 包含非有限数值")
    return array.reshape((4, 4), order="F")


def _homogeneous_to_cartesian(point, name):
    if abs(point[3]) <= _EPSILON:
        raise ValueError(f"{name} 的齐次坐标无法归一化")
    result = point[:3] / point[3]
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 包含非有限世界坐标")
    return result


def compute_camera_matrices(camera_config, camera_eye):
    """按 PyBullet 渲染约定构造 view 和 projection matrix。"""
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_eye,
        cameraTargetPosition=camera_config["workspace_center"],
        cameraUpVector=camera_config["up_vector"],
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=camera_config["fov"],
        aspect=camera_config["image_width"] / camera_config["image_height"],
        nearVal=camera_config["near_val"],
        farVal=camera_config["far_val"],
    )
    return list(view_matrix), list(projection_matrix)


def normalized_box_center_to_pixel(box, image_width, image_height):
    """把 0–1000 归一化框中心转换成浮点像素坐标。"""
    _positive_image_size(image_width, image_height)
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("目标框必须是 [x1, y1, x2, y2]")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in box
    ):
        raise ValueError("目标框必须包含四个有限数值")
    x1, y1, x2, y2 = map(float, box)
    if not all(0.0 <= value <= 1000.0 for value in (x1, y1, x2, y2)):
        raise ValueError("目标框坐标必须位于 0 到 1000")
    if x1 >= x2 or y1 >= y2:
        raise ValueError("目标框边界顺序非法")
    return (
        ((x1 + x2) / 2.0) / 1000.0 * (image_width - 1),
        ((y1 + y2) / 2.0) / 1000.0 * (image_height - 1),
    )


def pixel_to_world_on_plane(
    pixel_xy,
    image_width,
    image_height,
    view_matrix,
    projection_matrix,
    plane_z=0.0,
):
    """把像素反投影成射线，并返回射线与 z=plane_z 的交点。"""
    _positive_image_size(image_width, image_height)
    pixel = np.asarray(pixel_xy, dtype=float)
    if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
        raise ValueError("像素坐标必须包含两个有限数值")
    u, v = pixel
    if not (0.0 <= u <= image_width - 1 and 0.0 <= v <= image_height - 1):
        raise ValueError("像素坐标超出图片范围")
    if (
        isinstance(plane_z, bool)
        or not isinstance(plane_z, (int, float))
        or not math.isfinite(plane_z)
    ):
        raise ValueError("工作平面高度必须是有限数值")

    view = _matrix4(view_matrix, "view_matrix")
    projection = _matrix4(projection_matrix, "projection_matrix")
    try:
        inverse_view_projection = np.linalg.inv(projection @ view)
    except np.linalg.LinAlgError as exc:
        raise ValueError("view/projection matrix 不可逆") from exc

    ndc_x = 2.0 * u / (image_width - 1) - 1.0
    ndc_y = 1.0 - 2.0 * v / (image_height - 1)
    near = _homogeneous_to_cartesian(
        inverse_view_projection @ np.array([ndc_x, ndc_y, -1.0, 1.0]),
        "near clip point",
    )
    far = _homogeneous_to_cartesian(
        inverse_view_projection @ np.array([ndc_x, ndc_y, 1.0, 1.0]),
        "far clip point",
    )
    direction = far - near
    if abs(direction[2]) <= _EPSILON:
        raise ValueError("相机射线与工作平面平行")
    distance = (float(plane_z) - near[2]) / direction[2]
    if distance < 0.0:
        raise ValueError("工作平面交点位于相机射线反方向")
    world = near + distance * direction
    world[2] = float(plane_z)
    return world.tolist()


def world_to_pixel(
    world_xyz,
    image_width,
    image_height,
    view_matrix,
    projection_matrix,
):
    """把世界坐标投影到左上角为原点的浮点像素坐标。"""
    _positive_image_size(image_width, image_height)
    world = np.asarray(world_xyz, dtype=float)
    if world.shape != (3,) or not np.all(np.isfinite(world)):
        raise ValueError("世界坐标必须包含三个有限数值")
    view = _matrix4(view_matrix, "view_matrix")
    projection = _matrix4(projection_matrix, "projection_matrix")
    clip = projection @ view @ np.append(world, 1.0)
    if clip[3] <= _EPSILON:
        raise ValueError("世界点位于相机后方或无法投影")
    ndc = clip[:3] / clip[3]
    return (
        (ndc[0] + 1.0) * 0.5 * (image_width - 1),
        (1.0 - ndc[1]) * 0.5 * (image_height - 1),
    )
