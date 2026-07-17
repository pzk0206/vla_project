# Grounding 世界坐标闭环 Smoke Test 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 3 个 clear episode 验证 Qwen grounding、相机反投影、冻结补偿、主轴2cm单步控制和安全中止组成的真实视觉闭环能否达到3/3、最终 XY 误差不超过3cm。

**Architecture:** 新增纯计算模块 `grounding_targeting.py`，其动作接口不接收 PyBullet 红块真值；新增独立入口 `run_grounding_smoke.py`，用可注入依赖的闭环编排连接 Qwen、PyBullet、trace 和评分层。现有 `stage3_probe.py` 只提供已验证的 API、IK 和单步控制函数，不增加新控制模式。

**Tech Stack:** Python 3.10、PyBullet、OpenCV、NumPy、标准库 `unittest`、JSON/JSONL、Qwen OpenAI-compatible API、Git。

## Global Constraints

- 只运行 seeds 52、53、54，初始方向依次为 left、right、front，起始 XY 距离为0.10m。
- 使用固定448×448正俯视相机，每一步重新调用 Qwen；每个 episode 最多10步，单次完整运行最多30次 API 请求。
- 每次动作只沿主误差轴移动0.02m，悬停 Z 固定为0.20m；预测 stop 阈值为 X/Y 各不超过0.02m。
- 控制决策函数不得接收 `block_pos`；真值只允许用于场景搭建、动作后 trace 和最终评分。
- 运行时只读取已有 `calibration.json`，要求 clear 校准样本数为15，冻结补偿为 `(+0.02492227406480192, -0.019467343494422532)m`，不得重新拟合。
- 预测工作区固定为 X `[-0.30, 0.30]`、Y `[0.30, 0.70]`。
- clear 资格使用参考像素378和阈值0.75；低于阈值立即以 `visibility_out_of_scope` 失败。
- 相邻补偿目标跳变大于0.03m时中止；连续两个动作的预测距离均未至少改善0.005m时中止。
- 只有3/3都在10步内由预测坐标触发 stop，且真实 XY 误差均 `<=0.03m`，整批才通过。
- 已有运行目录只读；中断后创建新的时间戳目录并从场景初态重跑，不复用旧 step 回复。
- 所有 Python 测试和脚本使用 conda 环境 `vla_env`。
- 实验图片和运行结果保存在本地，不强制提交批量生成物。

---

## 文件结构

- Create: `grounding_targeting.py` — 校准加载、框反投影、补偿、工作区、主轴动作和安全状态纯函数。
- Create: `tests/test_grounding_targeting.py` — 保护真值无关的定位与动作边界。
- Create: `run_grounding_smoke.py` — 场景搭建、可注入闭环、PyBullet适配、输出和 CLI。
- Create: `tests/test_run_grounding_smoke.py` — 模拟逐步重定位、终止原因、trace 和 summary。
- Modify: `sim_config.yaml`、`tests/test_config_contract.py`、`.gitignore` — 冻结配置并隔离本地输出。
- Modify: `README.md`、`docs/worklog/WORKLOG.md`、`docs/planning/vla_robotic_study_plan.md`、`docs/debugging/BUGLOG.md` — 同步真实结论。

---

### Task 1: Smoke 配置契约与本地输出隔离

**Files:**
- Modify: `sim_config.yaml`
- Modify: `tests/test_config_contract.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: 无。
- Produces: `config["grounding_smoke"]: dict`。

- [ ] **Step 1: 写配置失败测试**

在 `tests/test_config_contract.py` 新增：

```python
def test_grounding_smoke_config_is_frozen_and_safe(self):
    smoke = self.config["grounding_smoke"]
    self.assertEqual(smoke["output_dir"], "vlm_smoke_runs")
    self.assertEqual(smoke["seeds"], [52, 53, 54])
    self.assertEqual(smoke["start_directions"], ["left", "right", "front"])
    self.assertEqual(smoke["start_offset_xy"], 0.10)
    self.assertEqual(smoke["max_control_steps"], 10)
    self.assertEqual(smoke["move_step_xy"], 0.02)
    self.assertEqual(smoke["hover_z"], 0.20)
    self.assertEqual(smoke["stop_distance_xy"], 0.02)
    self.assertEqual(smoke["visibility_reference_pixels"], 378)
    self.assertEqual(smoke["clear_visibility_threshold"], 0.75)
    self.assertEqual(smoke["max_target_jump_xy"], 0.03)
    self.assertEqual(smoke["min_progress_xy"], 0.005)
    self.assertEqual(smoke["no_progress_limit"], 2)
    self.assertEqual(smoke["workspace_x"], [-0.30, 0.30])
    self.assertEqual(smoke["workspace_y"], [0.30, 0.70])
    self.assertEqual(smoke["required_successes"], 3)
    self.assertEqual(smoke["max_total_api_calls"], 30)
    self.assertEqual(smoke["expected_calibration_samples"], 15)
```

同时把 `"grounding_smoke"` 加入 `test_required_sections_exist`。

- [ ] **Step 2: 运行测试并确认先失败**

```bash
conda run -n vla_env python -m unittest \
  tests.test_config_contract.ConfigContractTests.test_required_sections_exist \
  tests.test_config_contract.ConfigContractTests.test_grounding_smoke_config_is_frozen_and_safe -v
```

Expected: FAIL/ERROR，`grounding_smoke` 尚不存在。

- [ ] **Step 3: 写入精确配置和忽略规则**

```yaml
grounding_smoke:
  output_dir: "vlm_smoke_runs"
  calibration_path: "vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1/calibration_validation/calibration.json"
  expected_calibration_samples: 15
  expected_correction_x: 0.02492227406480192
  expected_correction_y: -0.019467343494422532
  seeds: [52, 53, 54]
  start_directions: ["left", "right", "front"]
  start_offset_xy: 0.10
  max_control_steps: 10
  move_step_xy: 0.02
  hover_z: 0.20
  stop_distance_xy: 0.02
  plane_z: 0.0
  visibility_reference_pixels: 378
  clear_visibility_threshold: 0.75
  workspace_x: [-0.30, 0.30]
  workspace_y: [0.30, 0.70]
  max_target_jump_xy: 0.03
  min_progress_xy: 0.005
  no_progress_limit: 2
  required_successes: 3
  max_total_api_calls: 30
```

在 `.gitignore` 增加 `vlm_smoke_runs/`。

- [ ] **Step 4: 验证并提交**

```bash
conda run -n vla_env python -m unittest tests.test_config_contract -v
git check-ignore -v vlm_smoke_runs/example/smoke_summary.json
git add sim_config.yaml tests/test_config_contract.py .gitignore
git commit -m "test: freeze grounding smoke configuration"
```

Expected: 配置测试全部 PASS，路径被新规则忽略。

---

### Task 2: 真值无关的定位、补偿和动作策略

**Files:**
- Create: `grounding_targeting.py`
- Create: `tests/test_grounding_targeting.py`

**Interfaces:**
- Produces: `SmokeSafetyAbort`、`load_frozen_calibration(path, expected_count, expected_x, expected_y) -> dict`、`compute_grounding_action(red_block_box, image_size, view_matrix, projection_matrix, calibration, ee_pos, safety_state, settings) -> dict`。

- [ ] **Step 1: 写核心失败测试**

```python
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from grounding_targeting import (
    SmokeSafetyAbort,
    compute_grounding_action,
    load_frozen_calibration,
)

CALIBRATION = {
    "num_clear_calibration_samples": 15,
    "correction_x": 0.02492227406480192,
    "correction_y": -0.019467343494422532,
}

class GroundingTargetingTests(unittest.TestCase):
    def test_decision_interface_cannot_receive_block_truth(self):
        self.assertNotIn(
            "block_pos",
            inspect.signature(compute_grounding_action).parameters,
        )

    def test_loads_only_expected_frozen_calibration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.json"
            path.write_text(json.dumps(CALIBRATION), encoding="utf-8")
            loaded = load_frozen_calibration(
                path, 15, 0.02492227406480192, -0.019467343494422532
            )
        self.assertEqual(loaded, CALIBRATION)

    @patch("grounding_targeting.pixel_to_world_on_plane")
    @patch("grounding_targeting.normalized_box_center_to_pixel")
    def test_applies_correction_and_selects_dominant_axis(
        self, center_to_pixel, pixel_to_world
    ):
        center_to_pixel.return_value = (223.5, 223.5)
        pixel_to_world.return_value = [0.075, 0.41, 0.0]
        action = compute_grounding_action(
            red_block_box=[450, 450, 550, 550],
            image_size=(448, 448),
            view_matrix=[1.0] * 16,
            projection_matrix=[1.0] * 16,
            calibration=CALIBRATION,
            ee_pos=[0.0, 0.4, 0.2],
            safety_state={
                "previous_target_xy": None,
                "previous_predicted_distance": None,
                "no_progress_count": 0,
            },
            settings={
                "plane_z": 0.0,
                "workspace_x": [-0.30, 0.30],
                "workspace_y": [0.30, 0.70],
                "stop_distance_xy": 0.02,
                "max_target_jump_xy": 0.03,
                "min_progress_xy": 0.005,
                "no_progress_limit": 2,
            },
        )
        self.assertEqual(action["direction"], "right")
        self.assertAlmostEqual(
            action["corrected_target_world"][0], 0.09992227406480192
        )
```

在同一个测试文件中再明确加入以下独立测试方法（每个测试只验证一个边界）：

- `test_stop_requires_both_axes_within_two_centimeters`：`dx=0.019, dy=-0.020` 返回 `stop`；
- `test_equal_axis_error_uses_x_as_dominant_axis`：`abs(dx)==abs(dy)>0.02` 时按 X 轴；
- `test_workspace_boundary_is_inclusive`：目标恰好落在四条边界上不报错；
- `test_target_outside_workspace_aborts`：超出任一边界 `1e-6` 抛出 `target_out_of_workspace`；
- `test_target_jump_equal_to_limit_is_allowed`：跳变恰好0.03m继续；
- `test_target_jump_above_limit_aborts`：跳变0.030001m抛出 `target_jump`；
- `test_two_consecutive_subthreshold_improvements_abort`：两次改善都小于0.005m后抛出 `no_progress`；
- `test_one_sufficient_improvement_resets_no_progress_counter`：改善恰好0.005m将计数清零；
- `test_calibration_rejects_boolean_and_non_finite_values`：`True`、`NaN`、`inf` 均被拒绝；
- `test_public_action_signature_has_no_block_truth`：用 `inspect.signature` 断言参数中没有 `block_pos`、`true_block_pos`。

- [ ] **Step 2: 运行并确认模块缺失**

```bash
conda run -n vla_env python -m unittest tests.test_grounding_targeting -v
```

Expected: ERROR，模块尚不存在。

- [ ] **Step 3: 实现完整纯策略接口**

```python
"""不读取红块真值的 grounding 世界坐标与安全动作策略。"""

import json
import math
from pathlib import Path

from camera_geometry import (
    normalized_box_center_to_pixel,
    pixel_to_world_on_plane,
)

class SmokeSafetyAbort(RuntimeError):
    def __init__(self, reason, message):
        self.reason = reason
        super().__init__(f"{reason}: {message}")

def load_frozen_calibration(path, expected_count, expected_x, expected_y):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("num_clear_calibration_samples") != expected_count:
        raise ValueError("冻结校准样本数不匹配")
    for field, expected in (
        ("correction_x", expected_x),
        ("correction_y", expected_y),
    ):
        value = payload.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not math.isclose(value, expected, abs_tol=1e-12)
        ):
            raise ValueError(f"冻结校准 {field} 不匹配")
    return payload

def _dominant_direction(dx, dy, stop_distance):
    if abs(dx) <= stop_distance and abs(dy) <= stop_distance:
        return "stop"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "front" if dy > 0 else "back"

def compute_grounding_action(
    red_block_box,
    image_size,
    view_matrix,
    projection_matrix,
    calibration,
    ee_pos,
    safety_state,
    settings,
):
    width, height = image_size
    pixel = normalized_box_center_to_pixel(red_block_box, width, height)
    raw_world = pixel_to_world_on_plane(
        pixel, width, height, view_matrix, projection_matrix,
        plane_z=settings["plane_z"],
    )
    corrected = [
        raw_world[0] + calibration["correction_x"],
        raw_world[1] + calibration["correction_y"],
        raw_world[2],
    ]
    if not (
        settings["workspace_x"][0] <= corrected[0] <= settings["workspace_x"][1]
        and settings["workspace_y"][0] <= corrected[1] <= settings["workspace_y"][1]
    ):
        raise SmokeSafetyAbort("target_out_of_workspace", str(corrected))
    previous_target = safety_state["previous_target_xy"]
    jump = math.dist(previous_target, corrected[:2]) if previous_target else 0.0
    if jump > settings["max_target_jump_xy"]:
        raise SmokeSafetyAbort("target_jump", f"jump={jump}")
    dx = corrected[0] - ee_pos[0]
    dy = corrected[1] - ee_pos[1]
    distance = math.hypot(dx, dy)
    direction = _dominant_direction(dx, dy, settings["stop_distance_xy"])
    previous_distance = safety_state["previous_predicted_distance"]
    count = safety_state["no_progress_count"]
    if direction != "stop" and previous_distance is not None:
        improvement = previous_distance - distance
        count = 0 if improvement >= settings["min_progress_xy"] else count + 1
        if count >= settings["no_progress_limit"]:
            raise SmokeSafetyAbort("no_progress", f"improvement={improvement}")
    return {
        "direction": direction,
        "box_center_pixel": list(pixel),
        "raw_target_world": raw_world,
        "corrected_target_world": corrected,
        "predicted_distance_xy": distance,
        "target_jump_xy": jump,
        "safety_state": {
            "previous_target_xy": corrected[:2],
            "previous_predicted_distance": distance,
            "no_progress_count": count,
        },
    }
```

- [ ] **Step 4: 验证并提交**

```bash
conda run -n vla_env python -m unittest tests.test_grounding_targeting -v
git add grounding_targeting.py tests/test_grounding_targeting.py
git commit -m "feat: add truth-isolated grounding action policy"
```

Expected: 全部 PASS，公共动作接口没有 `block_pos`。

---

### Task 3: 可注入依赖的闭环编排和摘要

**Files:**
- Create: `run_grounding_smoke.py`
- Create: `tests/test_run_grounding_smoke.py`

**Interfaces:**
- Produces: `SmokeDependencies`、`run_control_loop(smoke_config, case, calibration, episode_dir, dependencies) -> dict`、`aggregate_smoke_summaries(summaries, smoke_config) -> dict`。

- [ ] **Step 1: 写模拟闭环失败测试**

```python
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np

from run_grounding_smoke import (
    SmokeDependencies,
    aggregate_smoke_summaries,
    run_control_loop,
)

class SmokeLoopTests(unittest.TestCase):
    def test_visibility_aborts_before_paid_grounding(self):
        ground = Mock()
        dependencies = SmokeDependencies(
            observe=lambda: {
                "image_bgr": np.zeros((448, 448, 3), dtype=np.uint8),
                "view_matrix": [1.0] * 16,
                "projection_matrix": [1.0] * 16,
                "ee_pos": [0.0, 0.4, 0.2],
                "visibility": {
                    "block_visible_pixels": 100,
                    "block_visibility_ratio": 100 / 378,
                },
            },
            ground=ground,
            compute_action=Mock(),
            execute=Mock(),
            score=Mock(return_value={
                "true_block_pos": [0.1, 0.4, 0.05],
                "true_distance_xy": 0.1,
            }),
            save_images=Mock(return_value=("raw.jpg", None)),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_control_loop(
                {
                    "max_control_steps": 10,
                    "clear_visibility_threshold": 0.75,
                },
                {"episode_idx": 0, "seed": 52, "start_direction": "left"},
                {},
                Path(temp_dir) / "episode_000",
                dependencies,
            )
        self.assertEqual(
            summary["termination_reason"], "visibility_out_of_scope"
        )
        ground.assert_not_called()
```

在同一个测试类中明确加入以下测试：

- `test_two_frame_success_regrounds_and_executes_once`：两次观测、两次 grounding，第二帧为 `stop`，`execute.call_count == 1`，最终 `success`；
- `test_action_call_never_receives_block_truth`：逐个检查 `compute_action.call_args_list`，kwargs 中没有 `block_pos` 或 `true_block_pos`；
- `test_false_stop_is_scored_only_after_predicted_stop`：预测 stop 但真实距离0.031m，终止为 `false_stop` 且不执行动作；
- `test_last_allowed_action_records_max_control_steps`：达到第10步仍非 stop，最后一行 trace 的终止原因为 `max_control_steps`；
- `test_safety_abort_reason_is_preserved_in_trace`：参数化或分开覆盖 `target_jump`、`target_out_of_workspace`、`no_progress`；
- `test_grounding_exception_becomes_api_error`：API 异常记录 `api_error`，且 API 尝试次数为1；
- `test_missing_red_box_becomes_invalid_box`：返回空框记录 `invalid_box`；
- `test_geometry_exception_becomes_backprojection_error`：`compute_action` 的非安全异常记录 `backprojection_error`；
- `test_execute_exception_becomes_ik_error`：执行器异常记录 `ik_error`；
- `test_every_abort_writes_one_trace_row`：上述每种中止都存在 JSONL 行且含显式 `termination_reason`；
- `test_initial_distance_is_measured_before_first_action`：起始评分0.10m、动作后0.08m，摘要分别保存0.10和0.08；
- `test_batch_pass_requires_exact_three_clear_successes`：少一个 case、任一失败、任一非 clear 或超过30次调用均为 `passed=false`。

- [ ] **Step 2: 运行并确认模块缺失**

```bash
conda run -n vla_env python -m unittest tests.test_run_grounding_smoke -v
```

Expected: ERROR，模块尚不存在。

- [ ] **Step 3: 实现可注入闭环**

```python
from collections import Counter
from dataclasses import dataclass
import json

from grounding_targeting import SmokeSafetyAbort

@dataclass(frozen=True)
class SmokeDependencies:
    observe: object
    ground: object
    compute_action: object
    execute: object
    score: object
    save_images: object

def _append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def run_control_loop(
    smoke_config, case, calibration, episode_dir, dependencies
):
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

    def record(
        step, observation, visibility, reason, payload=None, scoring=None
    ):
        row = {
            **case,
            "control_step": step,
            **visibility,
            **(scoring if scoring is not None else dependencies.score()),
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
            record(step, observation, visibility, termination, {
                "image_path": raw_path,
                "annotated_path": annotated_path,
            })
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
            record(step, observation, visibility, termination, {
                "error": repr(exc),
                "image_path": raw_path,
                "annotated_path": annotated_path,
            })
            break

        if not boxes or not boxes.get("red_block"):
            termination = "invalid_box"
            raw_path, annotated_path = dependencies.save_images(
                step, observation["image_bgr"], boxes or {}
            )
            record(step, observation, visibility, termination, {
                "raw_response": raw_response,
                "boxes": boxes,
                "latency_seconds": latency,
                "image_path": raw_path,
                "annotated_path": annotated_path,
            })
            break

        raw_path, annotated_path = dependencies.save_images(
            step, observation["image_bgr"], boxes
        )
        try:
            action = dependencies.compute_action(
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
            safety_state = action["safety_state"]
        except SmokeSafetyAbort as exc:
            termination = exc.reason
            record(step, observation, visibility, termination, {
                "raw_response": raw_response,
                "boxes": boxes,
                "latency_seconds": latency,
                "error": str(exc),
                "image_path": raw_path,
                "annotated_path": annotated_path,
            })
            break
        except Exception as exc:
            termination = "backprojection_error"
            record(step, observation, visibility, termination, {
                "raw_response": raw_response,
                "boxes": boxes,
                "latency_seconds": latency,
                "error": repr(exc),
                "image_path": raw_path,
                "annotated_path": annotated_path,
            })
            break

        execution = None
        if action["direction"] != "stop":
            try:
                execution = dependencies.execute(
                    action["direction"], observation["ee_pos"]
                )
            except Exception as exc:
                termination = "ik_error"
                record(step, observation, visibility, termination, {
                    "raw_response": raw_response,
                    "boxes": boxes,
                    "latency_seconds": latency,
                    **action,
                    "error": repr(exc),
                    "image_path": raw_path,
                    "annotated_path": annotated_path,
                })
                break

        scoring = dependencies.score()
        if action["direction"] == "stop":
            termination = (
                "success"
                if scoring["true_distance_xy"] <= 0.03
                else "false_stop"
            )
        elif step == smoke_config["max_control_steps"] - 1:
            termination = "max_control_steps"
        else:
            termination = "running"
        record(step, observation, visibility, termination, {
            "raw_response": raw_response,
            "boxes": boxes,
            "latency_seconds": latency,
            **action,
            "execution": execution,
            "image_path": raw_path,
            "annotated_path": annotated_path,
        }, scoring=scoring)
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
            if rows else initial_scoring["true_distance_xy"]
        ),
        "max_target_jump_xy": max(
            (
                row["target_jump_xy"]
                for row in rows if "target_jump_xy" in row
            ),
            default=None,
        ),
        "all_clear": all(
            row["block_visibility_ratio"]
            >= smoke_config["clear_visibility_threshold"]
            for row in rows
        ),
        "trace_path": str(trace_path),
    }

def aggregate_smoke_summaries(summaries, smoke_config):
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
```

- [ ] **Step 4: 验证所有终止原因并提交**

```bash
conda run -n vla_env python -m unittest tests.test_run_grounding_smoke -v
git add run_grounding_smoke.py tests/test_run_grounding_smoke.py
git commit -m "feat: orchestrate grounding smoke control loop"
```

Expected: 全部 PASS；可见率失败不调用 API，stop 不执行动作，真实距离只生成评分标签。

---

### Task 4: PyBullet 适配器、运行目录和 CLI

**Files:**
- Modify: `run_grounding_smoke.py`
- Modify: `tests/test_run_grounding_smoke.py`

**Interfaces:**
- Produces: `build_smoke_cases(smoke_config) -> list[dict]`、`make_run_dir(output_root, run_name=None) -> Path`、`run_smoke_batch(config, run_name=None) -> tuple[Path,dict]`。

- [ ] **Step 1: 写 batch 契约失败测试**

```python
def test_builds_exact_three_cases_without_back(self):
    self.assertEqual(
        build_smoke_cases({
            "seeds": [52, 53, 54],
            "start_directions": ["left", "right", "front"],
        }),
        [
            {"episode_idx": 0, "seed": 52, "start_direction": "left"},
            {"episode_idx": 1, "seed": 53, "start_direction": "right"},
            {"episode_idx": 2, "seed": 54, "start_direction": "front"},
        ],
    )

def test_run_directory_never_overwrites(self):
    with tempfile.TemporaryDirectory() as temp_dir:
        make_run_dir(temp_dir, "run_fixed")
        with self.assertRaises(FileExistsError):
            make_run_dir(temp_dir, "run_fixed")
```

- [ ] **Step 2: 运行并确认函数缺失**

```bash
conda run -n vla_env python -m unittest \
  tests.test_run_grounding_smoke.SmokeBatchContractTests -v
```

Expected: ERROR，两个函数尚不存在。

- [ ] **Step 3: 实现真实适配器**

`run_grounding_smoke.py` 必须导入并复用：

```python
from camera_geometry import compute_camera_matrices
from collect_vlm_eval_samples import (
    build_balanced_ee_positions,
    reset_robot_to_target,
)
from control_arm import (
    apply_joint_targets,
    calculate_target_joints,
    capture_rgb_and_segmentation,
    connect_physics,
    get_link_position,
    get_object_position,
    load_block,
    sample_camera_eye,
    settle_object,
    setup_world,
)
from diagnose_vlm_grounding import (
    build_grounding_prompt,
    draw_grounding_boxes,
    parse_grounding_boxes,
)
from grounding_targeting import (
    compute_grounding_action,
    load_frozen_calibration,
)
from stage3_probe import call_openai_compatible_api, direction_to_target
```

实现精确 case 和目录函数：

```python
def build_smoke_cases(smoke_config):
    seeds = smoke_config["seeds"]
    directions = smoke_config["start_directions"]
    if (
        len(seeds) != 3
        or len(directions) != 3
        or len(set(seeds)) != 3
        or directions != ["left", "right", "front"]
    ):
        raise ValueError("smoke cases 必须固定为三个唯一 seed 和 left/right/front")
    return [
        {"episode_idx": i, "seed": seed, "start_direction": direction}
        for i, (seed, direction) in enumerate(zip(seeds, directions))
    ]

def make_run_dir(output_root, run_name=None):
    name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    path = Path(output_root) / name
    path.mkdir(parents=True, exist_ok=False)
    return path

def compute_visibility(segmentation, block_id, reference_pixels):
    ids = np.asarray(segmentation, dtype=np.int64) & ((1 << 24) - 1)
    visible = int(np.count_nonzero(ids == block_id))
    return {
        "block_visible_pixels": visible,
        "block_reference_pixels": reference_pixels,
        "block_visibility_ratio": min(visible / reference_pixels, 1.0),
    }
```

`run_smoke_batch()` 对每个 case 必须按以下固定顺序执行：

1. `random.seed(seed)`、DIRECT连接、`setup_world`、`load_block`、settle；
2. 读取一次真值，只用 `build_balanced_ee_positions(..., 0.10)` 和 `reset_robot_to_target` 搭建起点；
3. 复制配置并应用 `vlm_evaluation.camera_override`，固定 camera eye 和相机矩阵；
4. 构造 `observe/ground/execute/score/save_images` 五个闭包；
5. `observe` 不返回 `block_pos`；`score` 才读取红块真值；
6. `ground` 调用既有 prompt/parser；`execute` 使用 `direction_to_target(..., 0.20, 0.02)`；
7. 调用 `run_control_loop`；不得用宽泛的 `error` 覆盖具体原因：grounding、空框、几何、安全检查和执行异常必须分别保留为 `api_error`、`invalid_box`、`backprojection_error`、`SmokeSafetyAbort.reason` 和 `ik_error`；场景初始化或配置错误则让 batch 明确失败，不伪造 episode 结果；
8. 每个 episode 的 `finally` 必须断开 PyBullet；
9. 写 `episode_summary.jsonl` 和 `smoke_summary.json`。

CLI 提供可选 `--run-name`，默认时间戳目录；打印运行目录、成功数、API调用数和 `passed`。

- [ ] **Step 4: 用 mock 验证不联网和真值隔离**

Patch API、PyBullet 和 `run_control_loop`，固定 `run_name="run_test"`，断言：

```python
self.assertEqual(run_control_loop.call_count, 3)
self.assertEqual(
    [call.args[1]["seed"] for call in run_control_loop.call_args_list],
    [52, 53, 54],
)
self.assertTrue((batch_dir / "smoke_summary.json").is_file())
self.assertNotIn(
    "block_pos",
    inspect.signature(compute_grounding_action).parameters,
)
```

- [ ] **Step 5: 全量验证并提交**

```bash
conda run -n vla_env python -m unittest discover -s tests -v
git diff --check
git add run_grounding_smoke.py tests/test_run_grounding_smoke.py
git commit -m "feat: add pybullet grounding smoke runner"
```

Expected: 全部测试 PASS，测试过程真实 API 调用数为0。

---

### Task 5: 真实 3-Episode Smoke Test

**Files:**
- Generate: `vlm_smoke_runs/run_<timestamp>/`

- [ ] **Step 1: 检查 API 和冻结校准**

```bash
conda run -n vla_env python -c "import os,yaml,json; from pathlib import Path; c=yaml.safe_load(open('sim_config.yaml')); s=c['grounding_smoke']; assert all(os.getenv(c['probe']['api'][n]) for n in ('base_url_env','api_key_env','model_env')); d=json.loads(Path(s['calibration_path']).read_text()); assert d['num_clear_calibration_samples']==15; assert d['correction_x']==s['expected_correction_x']; assert d['correction_y']==s['expected_correction_y']; print('api_env=ready calibration=frozen cases=3 max_calls=30')"
```

Expected: 只打印就绪状态，不打印密钥。

- [ ] **Step 2: 运行完整测试**

```bash
conda run -n vla_env python -m unittest discover -s tests -v
```

Expected: 全部 PASS。

- [ ] **Step 3: 执行真实 smoke test**

```bash
conda run -n vla_env python run_grounding_smoke.py
```

Expected: 创建新时间戳目录；无论通过或失败都保留证据。中断后重新执行会新建目录。

- [ ] **Step 4: 审计固定 cases、调用上限和接口边界**

将 `<run_dir>` 替换为真实目录：

```bash
conda run -n vla_env python -c "import json,inspect; from pathlib import Path; from grounding_targeting import compute_grounding_action; root=Path('<run_dir>'); s=json.loads((root/'smoke_summary.json').read_text()); traces=[json.loads(x) for p in sorted(root.glob('episode_*/smoke_trace.jsonl')) for x in p.read_text().splitlines() if x.strip()]; assert s['num_episodes']==3; assert [e['seed'] for e in s['episodes']]==[52,53,54]; assert s['total_api_calls']<=30; assert 'block_pos' not in inspect.signature(compute_grounding_action).parameters; print(json.dumps(s,ensure_ascii=False,indent=2))"
```

Expected: 断言通过并打印真实摘要。`passed=false` 仍保留原阈值和原 seeds。

- [ ] **Step 5: 人工检查三组原图和标注图**

逐个查看 episode 目录；若失败，检查终止前最后两帧、红块框、补偿目标、末端位置和可见率。

- [ ] **Step 6: 确认生成物保持本地**

```bash
git check-ignore -v <run_dir>/smoke_summary.json
git status --short
```

Expected: 运行目录被忽略，Git 状态没有批量图片。

---

### Task 6: 同步真实结果和下一步

**Files:**
- Modify: `README.md`
- Modify: `docs/worklog/WORKLOG.md`
- Modify: `docs/planning/vla_robotic_study_plan.md`
- Modify: `docs/debugging/BUGLOG.md`

- [ ] **Step 1: 提取唯一可信指标**

```bash
conda run -n vla_env python -c "import json; from pathlib import Path; s=json.loads(Path('<run_dir>/smoke_summary.json').read_text()); print('passed=',s['passed']); print('success=',s['success_count'],'/',s['num_episodes']); print('api_calls=',s['total_api_calls']); print('reasons=',s['termination_reason_counts']); [(print(e['seed'],e['start_direction'],e['num_control_steps'],e['final_true_distance_xy'])) for e in s['episodes']]"
```

Expected: 输出一组摘要直接读取的指标。

- [ ] **Step 2: 用同一组指标更新四份文档**

每份文档必须写明：

```text
固定 seeds 52–54、left/right/front、10cm起点、2cm单步、最多10步。
每一步重新调用 Qwen，真值只做场景搭建和事后评分。
3个 episode 的终止原因、步数、最终真实 XY 距离、总API调用数和 passed。
clear-only 结论不能外推到 severe 遮挡。
```

若 `passed=true`，唯一下一步是扩大到新的 clear seeds 做小批量在线评估，并单独设计遮挡拒绝/恢复；若为 false，只选择摘要中首个主导失败类型做一次单变量修复。

- [ ] **Step 3: 检查一致性和全量测试**

```bash
rg -n "smoke|52|53|54|API|3cm|下一步" README.md docs/worklog/WORKLOG.md docs/planning/vla_robotic_study_plan.md docs/debugging/BUGLOG.md
rg -n "T[B]D|T[O]DO|待.定|占.位" README.md docs/worklog/WORKLOG.md docs/planning/vla_robotic_study_plan.md docs/debugging/BUGLOG.md
git diff --check
conda run -n vla_env python -m unittest discover -s tests -v
```

Expected: 指标一致、未完成标记扫描无输出、格式检查无输出、全部测试 PASS。

- [ ] **Step 4: 提交文档证据**

```bash
git add README.md docs/worklog/WORKLOG.md docs/planning/vla_robotic_study_plan.md docs/debugging/BUGLOG.md
git commit -m "docs: analyze grounding closed-loop smoke test"
```

---

## 完成判据

- 控制决策接口中没有 `block_pos`，模拟集成测试证明真值不会进入动作调用。
- 每个控制步重新调用 Qwen；stop 不执行动作，真实距离只决定 success/false_stop。
- 三个固定 case 不替换，单次运行最多30次 API 请求，输出目录不覆盖。
- clear、工作区、目标跳变、无进展、API、框、反投影、IK和步数中止均有明确原因。
- 完整自动测试通过，真实 smoke 证据通过结构审计。
- 无论 `passed` 为真或假，都分析实际结果并给出一个证据驱动的下一步。
- README、WORKLOG、学习计划和 BUGLOG 使用同一组真实指标与阶段表述。
