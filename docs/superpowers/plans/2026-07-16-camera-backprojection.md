# Camera Backprojection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Qwen 的红块归一化框中心反投影到 PyBullet 工作平面世界坐标，并用与 VLM 输入隔离的仿真真值完成离线定位误差评估。

**Architecture:** 新增独立 `camera_geometry.py`，使用 PyBullet/OpenGL view 和 projection matrix 做像素射线与 `z=0` 平面求交；现有渲染路径复用同一矩阵构造函数。采样器把 `block_pos` 与相机矩阵写入独立 `diagnostics.jsonl`，离线评估器只连接已有 grounding 输出与 diagnostics，不调用网络。

**Tech Stack:** Python 3.10、NumPy、PyBullet、OpenCV、JSONL、`unittest`

## Global Constraints

- 第一版只验收 `vlm_evaluation.camera_override` 的固定 `448 x 448` 正俯视单相机。
- 底层反投影必须使用完整 view/projection matrix，不写死俯视比例。
- 工作平面固定为 `z = 0.0`，但几何函数保留 `plane_z` 参数。
- `samples.jsonl` 不得包含 `block_pos` 或相机矩阵；其中的离线
  `expected_direction` 标签不得进入发送给 Qwen 的请求内容。
- 仿真真值只写入 `diagnostics.jsonl` 并用于离线评分。
- 不实现 `api_grounded`，不调用 Qwen，不读取 depth buffer，不修改现有控制语义。
- 所有测试和运行验证使用 `conda run -n vla_env ...`。
- 当前工作区包含用户已有未提交修改；每次只暂存任务明确列出的文件，不清理或覆盖其他修改。

## File Structure

- Create: `camera_geometry.py` — 相机矩阵构造、框中心换算、投影与反投影纯几何。
- Create: `tests/test_camera_geometry.py` — 坐标约定、往返精度与失败分支测试。
- Modify: `control_arm.py` — 图像采集改为复用共享矩阵构造函数。
- Modify: `tests/test_control_arm.py` — 验证共享矩阵被原样传给 PyBullet 渲染。
- Modify: `collect_vlm_eval_samples.py` — 为每个样本生成隔离的 diagnostics。
- Modify: `tests/test_collect_vlm_eval_samples.py` — 验证真值隔离与诊断契约。
- Create: `evaluate_grounding_backprojection.py` — 无网络的离线连接、逐样本误差和汇总。
- Create: `tests/test_evaluate_grounding_backprojection.py` — 连接契约、错误隔离与统计测试。

---

### Task 1: Implement Pure Camera Geometry

**Files:**
- Create: `camera_geometry.py`
- Create: `tests/test_camera_geometry.py`

**Interfaces:**
- Consumes: PyBullet camera configuration keys `workspace_center`, `up_vector`, `fov`, `image_width`, `image_height`, `near_val`, `far_val`.
- Produces: `compute_camera_matrices(...)`, `normalized_box_center_to_pixel(...)`, `pixel_to_world_on_plane(...)`, and `world_to_pixel(...)`.

- [ ] **Step 1: Write failing tests for normalized box centers and top-down backprojection**

Create `tests/test_camera_geometry.py`:

```python
"""测试相机像素、世界坐标与工作平面之间的确定性几何转换。"""

import unittest

import numpy as np

from camera_geometry import (
    compute_camera_matrices,
    normalized_box_center_to_pixel,
    pixel_to_world_on_plane,
    world_to_pixel,
)


CAMERA_CONFIG = {
    "workspace_center": [0.0, 0.4, 0.0],
    "up_vector": [0.0, 1.0, 0.0],
    "image_width": 448,
    "image_height": 448,
    "fov": 45,
    "near_val": 0.1,
    "far_val": 100.0,
}
CAMERA_EYE = [0.0, 0.4, 3.0]


class NormalizedBoxCenterTests(unittest.TestCase):
    def test_maps_normalized_box_center_to_float_pixel(self):
        pixel = normalized_box_center_to_pixel(
            [400, 300, 600, 500], 448, 448
        )

        np.testing.assert_allclose(pixel, [223.5, 178.8], atol=1e-9)


class CameraBackprojectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.view, cls.projection = compute_camera_matrices(
            CAMERA_CONFIG, CAMERA_EYE
        )

    def test_image_center_hits_workspace_center_on_ground_plane(self):
        world = pixel_to_world_on_plane(
            (223.5, 223.5), 448, 448, self.view, self.projection
        )

        np.testing.assert_allclose(world, [0.0, 0.4, 0.0], atol=1e-6)

    def test_world_pixel_world_round_trip(self):
        expected = [0.12, 0.51, 0.0]
        pixel = world_to_pixel(
            expected, 448, 448, self.view, self.projection
        )
        actual = pixel_to_world_on_plane(
            pixel, 448, 448, self.view, self.projection
        )

        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_world_positive_y_projects_above_image_center(self):
        _, pixel_y = world_to_pixel(
            [0.0, 0.5, 0.0], 448, 448, self.view, self.projection
        )

        self.assertLess(pixel_y, 223.5)
```

- [ ] **Step 2: Run the geometry tests and verify RED**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_camera_geometry -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'camera_geometry'`.

- [ ] **Step 3: Implement the minimal matrix, projection, and ray-plane functions**

Create `camera_geometry.py`:

```python
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
    _positive_image_size(image_width, image_height)
    pixel = np.asarray(pixel_xy, dtype=float)
    if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
        raise ValueError("像素坐标必须包含两个有限数值")
    u, v = pixel
    if not (0.0 <= u <= image_width - 1 and 0.0 <= v <= image_height - 1):
        raise ValueError("像素坐标超出图片范围")
    if not isinstance(plane_z, (int, float)) or not math.isfinite(plane_z):
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
```

- [ ] **Step 4: Run the geometry tests and verify GREEN**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_camera_geometry -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Add explicit invalid-input tests**

Append to `CameraBackprojectionTests`:

```python
    def test_rejects_singular_matrix(self):
        with self.assertRaisesRegex(ValueError, "不可逆"):
            pixel_to_world_on_plane(
                (223.5, 223.5), 448, 448, [0.0] * 16, self.projection
            )

    def test_rejects_parallel_ray(self):
        horizontal_config = dict(CAMERA_CONFIG)
        horizontal_config["workspace_center"] = [0.0, 1.4, 1.0]
        horizontal_config["up_vector"] = [0.0, 0.0, 1.0]
        horizontal_view, horizontal_projection = compute_camera_matrices(
            horizontal_config, [0.0, 0.4, 1.0]
        )

        with self.assertRaisesRegex(ValueError, "平行"):
            pixel_to_world_on_plane(
                (223.5, 223.5),
                448,
                448,
                horizontal_view,
                horizontal_projection,
            )

    def test_rejects_intersection_behind_camera_ray(self):
        with self.assertRaisesRegex(ValueError, "反方向"):
            pixel_to_world_on_plane(
                (223.5, 223.5),
                448,
                448,
                self.view,
                self.projection,
                plane_z=4.0,
            )
```

Append to `NormalizedBoxCenterTests`:

```python
    def test_rejects_invalid_normalized_box(self):
        invalid_boxes = (
            [600, 300, 400, 500],
            [-1, 300, 400, 500],
            [100, 200, 300],
        )
        for box in invalid_boxes:
            with self.subTest(box=box):
                with self.assertRaises(ValueError):
                    normalized_box_center_to_pixel(box, 448, 448)
```

- [ ] **Step 6: Run all geometry tests**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_camera_geometry -v
```

Expected: 8 tests pass.

- [ ] **Step 7: Commit the geometry unit**

```bash
git add camera_geometry.py tests/test_camera_geometry.py
git commit -m "feat: add camera plane backprojection geometry"
```

---

### Task 2: Share Camera Matrices With RGB Capture

**Files:**
- Modify: `control_arm.py:20-35,294-324`
- Modify: `tests/test_control_arm.py`

**Interfaces:**
- Consumes: `compute_camera_matrices(camera_config, camera_eye)` from Task 1.
- Produces: unchanged `capture_rgb(camera_config, camera_eye) -> np.ndarray`; guarantees rendering and backprojection use identical matrices.

- [ ] **Step 1: Write a failing capture integration test**

Add NumPy to the existing imports in `tests/test_control_arm.py`:

```python
import numpy as np
```

Extend the existing `from control_arm import ...` statement to include `capture_rgb`:

```python
from control_arm import (
    calculate_target_joints,
    capture_rgb,
    determine_termination,
    next_episode_index,
)
```

Add this test class:

```python
class CaptureRgbCameraMatrixTests(unittest.TestCase):
    @patch("control_arm.p.getCameraImage")
    @patch("control_arm.compute_camera_matrices")
    def test_passes_shared_matrices_to_pybullet_renderer(
        self, compute_matrices, get_camera_image
    ):
        view_matrix = [float(index) for index in range(16)]
        projection_matrix = [float(index + 16) for index in range(16)]
        compute_matrices.return_value = (view_matrix, projection_matrix)
        get_camera_image.return_value = (
            2,
            2,
            np.zeros((2, 2, 4), dtype=np.uint8),
            None,
            None,
        )
        camera_config = {
            "image_width": 2,
            "image_height": 2,
        }

        image = capture_rgb(camera_config, [0.0, 0.4, 3.0])

        compute_matrices.assert_called_once_with(
            camera_config, [0.0, 0.4, 3.0]
        )
        self.assertEqual(get_camera_image.call_args.kwargs["viewMatrix"], view_matrix)
        self.assertEqual(
            get_camera_image.call_args.kwargs["projectionMatrix"], projection_matrix
        )
        self.assertEqual(image.shape, (2, 2, 3))
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_control_arm.CaptureRgbCameraMatrixTests -v
```

Expected: FAIL because `control_arm.compute_camera_matrices` does not exist.

- [ ] **Step 3: Replace duplicated matrix construction in `capture_rgb()`**

Add this import to `control_arm.py`:

```python
from camera_geometry import compute_camera_matrices
```

Replace the direct `p.computeViewMatrix(...)` and `p.computeProjectionMatrixFOV(...)` calls inside `capture_rgb()` with:

```python
    view_matrix, projection_matrix = compute_camera_matrices(
        camera_config, camera_eye
    )
```

Keep the existing `p.getCameraImage(...)`, NumPy reshape, RGB slicing and BGR conversion unchanged.

- [ ] **Step 4: Run targeted and existing control tests**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_control_arm -v
```

Expected: all `tests.test_control_arm` tests pass.

- [ ] **Step 5: Commit shared camera matrices**

```bash
git add control_arm.py tests/test_control_arm.py
git commit -m "refactor: share camera matrices with backprojection"
```

---

### Task 3: Isolate Offline Ground Truth From VLM Samples

**Files:**
- Modify: `collect_vlm_eval_samples.py`
- Modify: `tests/test_collect_vlm_eval_samples.py`

**Interfaces:**
- Consumes: `compute_camera_matrices(camera_config, camera_eye)` from Task 1.
- Produces: `build_sample_diagnostic(...) -> dict`, `validate_diagnostics(...)`, `write_sample_files(...)`, and `diagnostics.jsonl` beside `samples.jsonl`.

- [ ] **Step 1: Write failing tests for the diagnostics contract**

Add these imports to `tests/test_collect_vlm_eval_samples.py`:

```python
import json
import tempfile
from pathlib import Path

from collect_vlm_eval_samples import (
    build_sample_diagnostic,
    validate_diagnostics,
    write_sample_files,
)
```

Add this test class:

```python
class SampleDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.camera_config = {
            "workspace_center": [0.0, 0.4, 0.0],
            "up_vector": [0.0, 1.0, 0.0],
            "image_width": 448,
            "image_height": 448,
            "fov": 45,
            "near_val": 0.1,
            "far_val": 100.0,
        }

    def test_builds_reproducible_diagnostic_without_mutating_sample(self):
        sample = {
            "sample_id": "seed_42_d020_left",
            "image_path": "frame.jpg",
            "instruction": "悬停在红色积木上方",
            "expected_direction": "left",
            "camera_eye": [0.0, 0.4, 3.0],
        }

        diagnostic = build_sample_diagnostic(
            sample["sample_id"],
            [0.01, 0.44, 0.05],
            sample["camera_eye"],
            self.camera_config,
        )

        self.assertNotIn("block_pos", sample)
        self.assertEqual(diagnostic["sample_id"], sample["sample_id"])
        self.assertEqual(diagnostic["block_pos"], [0.01, 0.44, 0.05])
        self.assertEqual(len(diagnostic["view_matrix"]), 16)
        self.assertEqual(len(diagnostic["projection_matrix"]), 16)
        self.assertEqual(diagnostic["image_width"], 448)
        self.assertEqual(diagnostic["image_height"], 448)

    def test_rejects_duplicate_diagnostic_sample_ids(self):
        rows = [
            {"sample_id": "duplicate"},
            {"sample_id": "duplicate"},
        ]

        with self.assertRaisesRegex(ValueError, "重复"):
            validate_diagnostics(rows)

    def test_writes_truth_to_separate_file(self):
        sample = {
            "sample_id": "seed_42_d020_left",
            "image_path": "frame.jpg",
            "instruction": "悬停在红色积木上方",
            "expected_direction": "left",
            "camera_eye": [0.0, 0.4, 3.0],
        }
        diagnostic = build_sample_diagnostic(
            sample["sample_id"],
            [0.01, 0.44, 0.05],
            sample["camera_eye"],
            self.camera_config,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path, diagnostics_path = write_sample_files(
                Path(temp_dir), [sample], [diagnostic]
            )
            manifest_row = json.loads(manifest_path.read_text(encoding="utf-8"))
            diagnostic_row = json.loads(
                diagnostics_path.read_text(encoding="utf-8")
            )

        self.assertNotIn("block_pos", manifest_row)
        self.assertIn("block_pos", diagnostic_row)
```

- [ ] **Step 2: Run diagnostics tests and verify RED**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_collect_vlm_eval_samples.SampleDiagnosticsTests -v
```

Expected: FAIL because the three imported functions do not exist.

- [ ] **Step 3: Implement diagnostic construction, validation, and writing**

Add this import to `collect_vlm_eval_samples.py`:

```python
from camera_geometry import compute_camera_matrices
```

Add these helpers:

```python
def build_sample_diagnostic(
    sample_id, block_pos, camera_eye, camera_config
):
    """构造只供离线评分使用的仿真真值与相机快照。"""
    view_matrix, projection_matrix = compute_camera_matrices(
        camera_config, camera_eye
    )
    return {
        "sample_id": sample_id,
        "block_pos": [float(value) for value in block_pos],
        "camera_eye": [float(value) for value in camera_eye],
        "image_width": int(camera_config["image_width"]),
        "image_height": int(camera_config["image_height"]),
        "view_matrix": [float(value) for value in view_matrix],
        "projection_matrix": [float(value) for value in projection_matrix],
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
```

- [ ] **Step 4: Run the new contract tests and verify GREEN**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_collect_vlm_eval_samples.SampleDiagnosticsTests -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Thread diagnostics through both sample collection strategies**

Change `capture_balanced_pose_sample(...)` to return both values without changing its rendering behavior:

```python
        return {
            "camera_eye": list(camera_eye),
            "block_pos": list(block_pos),
        }
```

In `collect_vlm_eval_samples(config)`, initialize diagnostics beside samples:

```python
    camera_config = sampling_config["camera"]
    samples = []
    diagnostics = []
```

In the balanced branch, replace the single `camera_eye` result with:

```python
                captured = capture_balanced_pose_sample(
                    sampling_config,
                    direction,
                    seed,
                    relative_image_path,
                    offset_xy=case["offset_xy"],
                )
                camera_eye = captured["camera_eye"]
```

After appending each balanced sample, append its diagnostic:

```python
                diagnostics.append(
                    build_sample_diagnostic(
                        sample_id,
                        captured["block_pos"],
                        camera_eye,
                        camera_config,
                    )
                )
```

Replace the balanced branch's manual manifest write with:

```python
        validate_samples(samples, Path.cwd())
        manifest_path, _ = write_sample_files(
            output_dir, samples, diagnostics
        )
        return manifest_path, samples
```

In the trace-based branch, after each selected sample append:

```python
                diagnostics.append(
                    build_sample_diagnostic(
                        sample_id,
                        row["block_pos"],
                        row["camera_eye"],
                        camera_config,
                    )
                )
```

Replace the final manual manifest write with:

```python
    validate_samples(samples, Path.cwd())
    manifest_path, _ = write_sample_files(output_dir, samples, diagnostics)
    return manifest_path, samples
```

- [ ] **Step 6: Run the complete collector tests**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_collect_vlm_eval_samples -v
```

Expected: all collector tests pass.

- [ ] **Step 7: Commit the truth-isolation unit**

```bash
git add collect_vlm_eval_samples.py tests/test_collect_vlm_eval_samples.py
git commit -m "feat: isolate grounding evaluation truth"
```

---

### Task 4: Evaluate Grounding Backprojection Offline

**Files:**
- Create: `evaluate_grounding_backprojection.py`
- Create: `tests/test_evaluate_grounding_backprojection.py`

**Interfaces:**
- Consumes: grounding rows with `sample_id` and `boxes.red_block`; diagnostic rows from Task 3; geometry functions from Task 1.
- Produces: `evaluate_rows(predictions, diagnostics) -> list[dict]`, `summarize_results(results) -> dict`, `backprojection_results.jsonl`, and `backprojection_summary.json`.

- [ ] **Step 1: Write failing evaluator tests**

Create `tests/test_evaluate_grounding_backprojection.py`:

```python
"""测试 grounding 框反投影的离线连接、错误隔离和指标统计。"""

import unittest

import numpy as np

from camera_geometry import compute_camera_matrices, world_to_pixel
from evaluate_grounding_backprojection import evaluate_rows, summarize_results


class GroundingBackprojectionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = {
            "workspace_center": [0.0, 0.4, 0.0],
            "up_vector": [0.0, 1.0, 0.0],
            "image_width": 448,
            "image_height": 448,
            "fov": 45,
            "near_val": 0.1,
            "far_val": 100.0,
        }
        cls.view, cls.projection = compute_camera_matrices(
            config, [0.0, 0.4, 3.0]
        )

    def diagnostic(self, sample_id, block_pos):
        return {
            "sample_id": sample_id,
            "block_pos": block_pos,
            "camera_eye": [0.0, 0.4, 3.0],
            "image_width": 448,
            "image_height": 448,
            "view_matrix": self.view,
            "projection_matrix": self.projection,
        }

    def box_around_world_point(self, world):
        pixel_x, pixel_y = world_to_pixel(
            world, 448, 448, self.view, self.projection
        )
        normalized_x = pixel_x / 447.0 * 1000.0
        normalized_y = pixel_y / 447.0 * 1000.0
        return [
            normalized_x - 5.0,
            normalized_y - 5.0,
            normalized_x + 5.0,
            normalized_y + 5.0,
        ]

    def test_computes_zero_xy_error_for_matching_box_center(self):
        target = [0.08, 0.47, 0.0]
        predictions = [
            {
                "sample_id": "sample-1",
                "boxes": {"red_block": self.box_around_world_point(target)},
                "latency_seconds": 1.2,
                "error_type": None,
                "error_message": None,
            }
        ]
        diagnostics = [self.diagnostic("sample-1", [0.08, 0.47, 0.05])]

        result = evaluate_rows(predictions, diagnostics)[0]

        np.testing.assert_allclose(
            result["predicted_target_world"], target, atol=1e-6
        )
        self.assertAlmostEqual(result["localization_error_xy"], 0.0, places=6)
        self.assertIsNone(result["error_type"])

    def test_keeps_invalid_grounding_as_sample_error(self):
        predictions = [
            {
                "sample_id": "sample-1",
                "boxes": None,
                "error_type": "InvalidModelResponseError",
                "error_message": "missing boxes",
            }
        ]
        diagnostics = [self.diagnostic("sample-1", [0.08, 0.47, 0.05])]

        result = evaluate_rows(predictions, diagnostics)[0]

        self.assertEqual(result["error_type"], "InvalidModelResponseError")
        self.assertIsNone(result["predicted_target_world"])

    def test_rejects_duplicate_or_missing_diagnostics_before_evaluation(self):
        prediction = {"sample_id": "sample-1", "boxes": None}
        duplicate = self.diagnostic("sample-1", [0.0, 0.4, 0.05])

        with self.assertRaisesRegex(ValueError, "重复"):
            evaluate_rows([prediction], [duplicate, duplicate])
        with self.assertRaisesRegex(ValueError, "缺少诊断"):
            evaluate_rows([prediction], [])

    def test_summarizes_valid_errors_and_failure_types(self):
        results = [
            {
                "localization_error_xy": 0.01,
                "signed_error_x": 0.006,
                "signed_error_y": -0.008,
                "error_type": None,
            },
            {
                "localization_error_xy": 0.03,
                "signed_error_x": -0.018,
                "signed_error_y": 0.024,
                "error_type": None,
            },
            {
                "localization_error_xy": None,
                "signed_error_x": None,
                "signed_error_y": None,
                "error_type": "InvalidModelResponseError",
            },
        ]

        summary = summarize_results(results)

        self.assertEqual(summary["num_samples"], 3)
        self.assertEqual(summary["num_valid"], 2)
        self.assertEqual(summary["num_failed"], 1)
        self.assertAlmostEqual(summary["localization_error_xy_mean"], 0.02)
        self.assertAlmostEqual(summary["localization_error_xy_median"], 0.02)
        self.assertAlmostEqual(summary["localization_error_xy_max"], 0.03)
        self.assertEqual(
            summary["error_type_counts"], {"InvalidModelResponseError": 1}
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run evaluator tests and verify RED**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_evaluate_grounding_backprojection -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'evaluate_grounding_backprojection'`.

- [ ] **Step 3: Implement joining, per-sample projection, and summaries**

Create `evaluate_grounding_backprojection.py` with these functions and the usual imports (`argparse`, `json`, `math`, `statistics`, `Counter`, and `Path`):

```python
"""离线评估 Qwen 红块框中心到工作平面世界坐标的反投影误差。"""

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from camera_geometry import (
    normalized_box_center_to_pixel,
    pixel_to_world_on_plane,
)


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _index_unique(rows, label):
    indexed = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{label} 缺少非空 sample_id")
        if sample_id in indexed:
            raise ValueError(f"{label} sample_id 重复: {sample_id}")
        indexed[sample_id] = row
    return indexed


def evaluate_rows(predictions, diagnostics, plane_z=0.0):
    prediction_index = _index_unique(predictions, "grounding prediction")
    diagnostic_index = _index_unique(diagnostics, "diagnostics")
    missing = sorted(set(prediction_index) - set(diagnostic_index))
    if missing:
        raise ValueError(f"grounding prediction 缺少诊断: {missing}")

    results = []
    for sample_id, prediction in prediction_index.items():
        diagnostic = diagnostic_index[sample_id]
        boxes = prediction.get("boxes")
        base_result = {
            "sample_id": sample_id,
            "red_block_box": boxes.get("red_block") if boxes else None,
            "box_center_pixel": None,
            "predicted_target_world": None,
            "true_block_pos": diagnostic["block_pos"],
            "localization_error_xy": None,
            "signed_error_x": None,
            "signed_error_y": None,
            "latency_seconds": prediction.get("latency_seconds"),
            "error_type": prediction.get("error_type"),
            "error_message": prediction.get("error_message"),
        }
        if not boxes or not boxes.get("red_block"):
            base_result["error_type"] = (
                base_result["error_type"] or "MissingGroundingBoxError"
            )
            base_result["error_message"] = (
                base_result["error_message"] or "red_block box 缺失"
            )
            results.append(base_result)
            continue
        try:
            pixel = normalized_box_center_to_pixel(
                boxes["red_block"],
                diagnostic["image_width"],
                diagnostic["image_height"],
            )
            predicted = pixel_to_world_on_plane(
                pixel,
                diagnostic["image_width"],
                diagnostic["image_height"],
                diagnostic["view_matrix"],
                diagnostic["projection_matrix"],
                plane_z=plane_z,
            )
            true_block = diagnostic["block_pos"]
            signed_x = predicted[0] - true_block[0]
            signed_y = predicted[1] - true_block[1]
            base_result.update(
                {
                    "box_center_pixel": list(pixel),
                    "predicted_target_world": predicted,
                    "localization_error_xy": math.hypot(signed_x, signed_y),
                    "signed_error_x": signed_x,
                    "signed_error_y": signed_y,
                    "error_type": None,
                    "error_message": None,
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            base_result["error_type"] = type(exc).__name__
            base_result["error_message"] = str(exc)
        results.append(base_result)
    return results


def summarize_results(results):
    valid = [row for row in results if row["localization_error_xy"] is not None]
    errors = [row["localization_error_xy"] for row in valid]
    error_counts = Counter(
        row["error_type"] for row in results if row["error_type"]
    )
    return {
        "num_samples": len(results),
        "num_valid": len(valid),
        "num_failed": len(results) - len(valid),
        "valid_rate": len(valid) / len(results) if results else 0.0,
        "localization_error_xy_mean": statistics.fmean(errors) if errors else None,
        "localization_error_xy_median": statistics.median(errors) if errors else None,
        "localization_error_xy_max": max(errors) if errors else None,
        "signed_error_x_mean": (
            statistics.fmean(row["signed_error_x"] for row in valid)
            if valid
            else None
        ),
        "signed_error_y_mean": (
            statistics.fmean(row["signed_error_y"] for row in valid)
            if valid
            else None
        ),
        "error_type_counts": dict(sorted(error_counts.items())),
    }
```

- [ ] **Step 4: Run evaluator tests and verify GREEN**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_evaluate_grounding_backprojection -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Add a no-network CLI and output persistence**

Append to `evaluate_grounding_backprojection.py`:

```python
def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plane-z", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = read_jsonl(args.predictions)
    diagnostics = read_jsonl(args.diagnostics)
    results = evaluate_rows(predictions, diagnostics, plane_z=args.plane_z)
    summary = summarize_results(results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "backprojection_results.jsonl", results)
    with (output_dir / "backprojection_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the full test suite**

Run:

```bash
conda run -n vla_env python -m unittest discover -s tests -v
```

Expected: all tests pass with zero failures and zero errors.

- [ ] **Step 7: Regenerate diagnostics without making an API call**

Run:

```bash
conda run -n vla_env python collect_vlm_eval_samples.py
```

Expected: `vlm_eval_samples_448/samples.jsonl` and `vlm_eval_samples_448/diagnostics.jsonl` contain the same unique `sample_id` set. This command only runs PyBullet sampling and does not call Qwen.

- [ ] **Step 8: Evaluate the existing 448px grounding predictions**

Run:

```bash
conda run -n vla_env python evaluate_grounding_backprojection.py \
  --predictions vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_v3/grounding_predictions.jsonl \
  --diagnostics vlm_eval_samples_448/diagnostics.jsonl \
  --output-dir vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_v3/backprojection
```

Expected: exit 0 and a summary with `num_samples`, `num_valid`, `num_failed`, xy error mean/median/max, signed x/y means, and error type counts. Do not claim the `0.03m` gate is met until these real values are read.

- [ ] **Step 9: Commit the offline evaluator**

```bash
git add evaluate_grounding_backprojection.py tests/test_evaluate_grounding_backprojection.py
git commit -m "feat: evaluate grounding backprojection error"
```

## Final Verification

- [ ] Run the complete test suite again:

```bash
conda run -n vla_env python -m unittest discover -s tests -v
```

- [ ] Inspect the final implementation diff only:

```bash
git diff e4dd3aa -- camera_geometry.py control_arm.py collect_vlm_eval_samples.py evaluate_grounding_backprojection.py tests/test_camera_geometry.py tests/test_control_arm.py tests/test_collect_vlm_eval_samples.py tests/test_evaluate_grounding_backprojection.py
```

- [ ] Confirm that no runtime VLM input contains truth coordinates:

```bash
rg -n 'block_pos|view_matrix|projection_matrix' vlm_eval_samples_448/samples.jsonl
```

Expected: no matches.

- [ ] Read the generated summary and make the stage decision from evidence:

```bash
sed -n '1,200p' vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_v3/backprojection/backprojection_summary.json
```

Gate: no systematic x/y axis inversion; only proceed to `api_grounded` smoke tests if the measured localization error is compatible with the current `0.03m` control success threshold.
