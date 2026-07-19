# Grounding Smoke Batch Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在任何真实 VLM 请求前，无 API 地验证三个固定 smoke 案例的起始姿态和首帧可见率，并在任一初始预检失败时拒绝整个批次。

**Architecture:** 在现有 `runner.py` 中增加单案例 `preflight_smoke_case()`，每次独立创建并释放 PyBullet 场景，返回稳定的资格记录。`run_smoke_batch()` 先收集全部三个预检结果并写 `smoke_preflight.json`，仅在全部合格后进入原有 episode 循环；运行中后续遮挡仍由已有 fresh/held 状态机处理。

**Tech Stack:** Python 3.10、PyBullet、NumPy、`unittest`、`unittest.mock`、JSON

## Global Constraints

- 固定案例保持 seeds `[52, 53, 54]` 和方向 `[left, right, front]`，不得跳过或自动替换。
- 预检阈值继续使用 `balanced_pose_tolerance=0.005` 和 `clear_visibility_threshold=0.75`。
- 只有初始动态预检失败才拒绝整批；正式闭环中的低可见帧继续复用最近可靠目标最多4步。
- 预检不得构建 grounding 依赖、调用 `run_control_loop()` 或触发真实 API。
- 不修改 success、预测 stop、事后真值评分、目标保持状态机或 API 调用上限。
- 实现完成后只运行无 API 动态验证，不运行第二次真实 smoke。
- 所有正式代码留在 `src/vla_project/vlm/grounding_smoke/`，测试留在镜像目录 `tests/vlm/grounding_smoke/`。

---

### Task 1: 单案例无 API 动态预检

**Files:**
- Modify: `src/vla_project/vlm/grounding_smoke/runner.py:480-518`
- Test: `tests/vlm/grounding_smoke/test_runner.py:1-35,477-560`

**Interfaces:**
- Consumes: `config: dict`、`smoke_config: dict`、`case: dict`
- Produces: `preflight_smoke_case(config, smoke_config, case) -> dict`
- Stable result keys: `episode_idx`、`seed`、`start_direction`、`qualified`、`rejection_reason`、`requested_start_ee_pos`、`actual_start_ee_pos`、`start_pose_error`、`block_visible_pixels`、`block_reference_pixels`、`block_visibility_ratio`、`camera_eye`

- [ ] **Step 1: 导入待实现接口并增加最小测试配置 helper**

在 `tests/vlm/grounding_smoke/test_runner.py` 的 runner import 中加入：

```python
    preflight_smoke_case,
```

在 `SmokeBatchContractTests` 中加入：

```python
    def make_preflight_config(self):
        return {
            "robot": {"ee_link_index": 6},
            "task": {"initial_settle_steps": 1},
            "camera": {"workspace_center": [0.0, 0.4, 0.0]},
            "vlm_evaluation": {
                "balanced_pose_tolerance": 0.005,
                "balanced_pose_ik_iterations": 20,
                "camera_override": {
                    "image_width": 448,
                    "image_height": 448,
                },
            },
            "grounding_smoke": {
                "hover_z": 0.20,
                "start_offset_xy": 0.10,
                "visibility_reference_pixels": 4,
                "clear_visibility_threshold": 0.75,
            },
        }
```

- [ ] **Step 2: 写起点失败、可见率失败和包含性边界测试**

增加一个测试 helper，使单案例仿真边界完全可控：

```python
    def run_preflight(self, actual_start, segmentation):
        config = self.make_preflight_config()
        requested_start = [0.1, 0.4, 0.2]
        with (
            patch("vla_project.vlm.grounding_smoke.runner.connect_physics"),
            patch(
                "vla_project.vlm.grounding_smoke.runner.setup_world",
                return_value=(1, 10),
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.load_block",
                return_value=20,
            ),
            patch("vla_project.vlm.grounding_smoke.runner.settle_object"),
            patch(
                "vla_project.vlm.grounding_smoke.runner.get_object_position",
                return_value=[0.0, 0.0, 0.05],
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.build_balanced_ee_positions",
                return_value={"left": requested_start},
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.reset_robot_to_target",
                return_value=actual_start,
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.sample_camera_eye",
                return_value=[0.0, 0.4, 3.0],
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.capture_rgb_and_segmentation",
                return_value=(
                    np.zeros((2, 2, 3), dtype=np.uint8),
                    np.asarray(segmentation, dtype=np.int64),
                ),
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.p.disconnect"
            ) as disconnect,
        ):
            result = preflight_smoke_case(
                config,
                config["grounding_smoke"],
                CASE,
            )
        disconnect.assert_called_once_with()
        return result
```

再增加三个行为测试：

```python
    def test_preflight_records_visibility_even_when_start_pose_fails(self):
        result = self.run_preflight(
            [0.106, 0.4, 0.2],
            [[20, 20], [20, 20]],
        )
        self.assertFalse(result["qualified"])
        self.assertEqual(result["rejection_reason"], "start_pose_error")
        self.assertAlmostEqual(result["start_pose_error"], 0.006)
        self.assertEqual(result["block_visibility_ratio"], 1.0)

    def test_preflight_rejects_low_initial_visibility(self):
        result = self.run_preflight(
            [0.1, 0.4, 0.2],
            [[20, 20], [0, 0]],
        )
        self.assertFalse(result["qualified"])
        self.assertEqual(
            result["rejection_reason"],
            "visibility_below_threshold",
        )
        self.assertEqual(result["block_visibility_ratio"], 0.5)

    def test_preflight_accepts_equal_pose_and_visibility_boundaries(self):
        result = self.run_preflight(
            [0.105, 0.4, 0.2],
            [[20, 20], [20, 0]],
        )
        self.assertTrue(result["qualified"])
        self.assertIsNone(result["rejection_reason"])
        self.assertAlmostEqual(result["start_pose_error"], 0.005)
        self.assertEqual(result["block_visibility_ratio"], 0.75)
```

- [ ] **Step 3: 运行测试并确认 RED 原因是接口不存在**

Run:

```bash
conda run -n vla_env python -m unittest \
  tests.vlm.grounding_smoke.test_runner.SmokeBatchContractTests.test_preflight_records_visibility_even_when_start_pose_fails \
  tests.vlm.grounding_smoke.test_runner.SmokeBatchContractTests.test_preflight_rejects_low_initial_visibility \
  tests.vlm.grounding_smoke.test_runner.SmokeBatchContractTests.test_preflight_accepts_equal_pose_and_visibility_boundaries -v
```

Expected: import failure or three errors because `preflight_smoke_case` 尚未定义；不得是测试夹具拼写错误。

- [ ] **Step 4: 实现最小单案例预检**

在 `_build_camera_config()` 后加入：

```python
def preflight_smoke_case(config, smoke_config, case):
    """无 API 地验证单个案例的起点与首帧可见率。"""
    random.seed(case["seed"])
    connect_physics("DIRECT")
    try:
        _, robot_id = setup_world(config)
        block_id = load_block(config["task"])
        settle_object(config, config["task"]["initial_settle_steps"])
        initial_block_pos = list(get_object_position(block_id))
        start_positions = build_balanced_ee_positions(
            initial_block_pos,
            smoke_config["hover_z"] - initial_block_pos[2],
            smoke_config["start_offset_xy"],
        )
        requested_start = start_positions[case["start_direction"]]
        actual_start = reset_robot_to_target(
            robot_id,
            config["robot"],
            requested_start,
            tolerance=config["vlm_evaluation"]["balanced_pose_tolerance"],
            max_iterations=config["vlm_evaluation"][
                "balanced_pose_ik_iterations"
            ],
        )
        start_error = math.dist(actual_start, requested_start)
        camera_config = _build_camera_config(config)
        camera_eye = sample_camera_eye(camera_config)
        _, segmentation = capture_rgb_and_segmentation(
            camera_config,
            camera_eye,
        )
        visibility = compute_visibility(
            segmentation,
            block_id,
            smoke_config["visibility_reference_pixels"],
        )
        rejection_reason = None
        if start_error > config["vlm_evaluation"]["balanced_pose_tolerance"]:
            rejection_reason = "start_pose_error"
        elif (
            visibility["block_visibility_ratio"]
            < smoke_config["clear_visibility_threshold"]
        ):
            rejection_reason = "visibility_below_threshold"
        return {
            **case,
            "qualified": rejection_reason is None,
            "rejection_reason": rejection_reason,
            "requested_start_ee_pos": list(requested_start),
            "actual_start_ee_pos": list(actual_start),
            "start_pose_error": start_error,
            **visibility,
            "camera_eye": list(camera_eye),
        }
    finally:
        p.disconnect()
```

- [ ] **Step 5: 运行三个测试并确认 GREEN**

Run: 与 Step 3 相同。

Expected: `Ran 3 tests`、`OK`，且 API mock 不需要存在。

- [ ] **Step 6: 增加异常资源清理回归测试**

增加：

```python
    def test_preflight_disconnects_when_setup_raises(self):
        config = self.make_preflight_config()
        with (
            patch("vla_project.vlm.grounding_smoke.runner.connect_physics"),
            patch(
                "vla_project.vlm.grounding_smoke.runner.setup_world",
                side_effect=RuntimeError("setup failed"),
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.p.disconnect"
            ) as disconnect,
        ):
            with self.assertRaisesRegex(RuntimeError, "setup failed"):
                preflight_smoke_case(
                    config,
                    config["grounding_smoke"],
                    CASE,
                )
        disconnect.assert_called_once_with()
```

- [ ] **Step 7: 运行单案例预检测试和 runner 模块测试**

Run:

```bash
conda run -n vla_env python -m unittest tests.vlm.grounding_smoke.test_runner -v
```

Expected: 全部通过，无真实 API 调用。

- [ ] **Step 8: 提交单案例预检**

```bash
git add src/vla_project/vlm/grounding_smoke/runner.py tests/vlm/grounding_smoke/test_runner.py
git commit -m "feat: add grounding smoke case preflight"
```

---

### Task 2: 整批预检门禁与证据文件

**Files:**
- Modify: `src/vla_project/vlm/grounding_smoke/runner.py:45-62,632-718`
- Test: `tests/vlm/grounding_smoke/test_runner.py:13-27,560-700`

**Interfaces:**
- Consumes: `preflight_smoke_case(config, smoke_config, case) -> dict`
- Produces: `SmokePreflightError(RuntimeError)`、`<run_dir>/smoke_preflight.json`
- Preserves: `run_smoke_batch(config, run_name=None) -> tuple[Path, dict]` when all cases qualify

- [ ] **Step 1: 写整批拒绝的失败测试**

在 runner import 中加入 `SmokePreflightError`。先把现有批次测试的大型配置提取为模块级
helper，并让现有 `test_batch_uses_three_isolated_cases_and_writes_summary` 改用同一个
helper：

```python
def batch_config(temp_dir):
    calibration_path = Path(temp_dir) / "calibration.json"
    calibration_path.write_text("{}", encoding="utf-8")
    config = {
        "connection_mode": "DIRECT",
        "gravity": [0, 0, -9.8],
        "enable_time_sleep": False,
        "simulation_hz": 240,
        "robot": {
            "ee_link_index": 6,
            "controlled_joints": 7,
        },
        "task": {"initial_settle_steps": 1},
        "dataset": {"instruction": "悬停在红色积木上方"},
        "probe": {
            "sim_steps_per_action": 1,
            "api": {},
        },
        "camera": {
            "workspace_center": [0.0, 0.4, 0.0],
            "near_val": 0.1,
            "far_val": 100.0,
        },
        "vlm_evaluation": {
            "balanced_pose_tolerance": 0.005,
            "balanced_pose_ik_iterations": 20,
            "camera_override": {
                "image_width": 448,
                "image_height": 448,
                "eye_offset_base": [0.0, 0.0, 3.0],
                "eye_offset_random_range": [0.0, 0.0],
                "up_vector": [0, 1, 0],
                "fov": 45,
            },
        },
        "grounding_smoke": {
            **BASE_CONFIG,
            "output_dir": temp_dir,
            "calibration_path": str(calibration_path),
            "expected_calibration_samples": 15,
            "expected_calibration_sample_ids": [
                f"seed_{seed}_d020_{direction}"
                for seed in range(42, 47)
                for direction in ("front", "left", "right")
            ],
            "api_max_retries": 0,
            "expected_correction_x": 0.02492227406480192,
            "expected_correction_y": -0.019467343494422532,
            "start_directions": ["left", "right", "front"],
            "start_offset_xy": 0.10,
            "hover_z": 0.20,
            "move_step_xy": 0.02,
            "visibility_reference_pixels": 378,
        },
    }
    fake_calibration = {
        "num_clear_calibration_samples": 15,
        "correction_x": 0.02492227406480192,
        "correction_y": -0.019467343494422532,
    }
    return config, fake_calibration
```

增加拒绝测试：

```python
    def test_batch_preflights_all_cases_before_rejecting_without_api(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config, fake_calibration = batch_config(temp_dir)
            rows = [
                {
                    "episode_idx": 0,
                    "seed": 52,
                    "start_direction": "left",
                    "qualified": False,
                    "rejection_reason": "visibility_below_threshold",
                },
                {
                    "episode_idx": 1,
                    "seed": 53,
                    "start_direction": "right",
                    "qualified": True,
                    "rejection_reason": None,
                },
                {
                    "episode_idx": 2,
                    "seed": 54,
                    "start_direction": "front",
                    "qualified": False,
                    "rejection_reason": "start_pose_error",
                },
            ]
            with (
                patch(
                    "vla_project.vlm.grounding_smoke.runner.load_frozen_calibration",
                    return_value=fake_calibration,
                ),
                patch(
                    "vla_project.vlm.grounding_smoke.runner.preflight_smoke_case",
                    side_effect=rows,
                ) as preflight,
                patch(
                    "vla_project.vlm.grounding_smoke.runner.run_control_loop"
                ) as loop,
                patch(
                    "vla_project.vlm.grounding_smoke.runner.call_openai_compatible_api"
                ) as api,
            ):
                with self.assertRaises(SmokePreflightError):
                    run_smoke_batch(config, run_name="run_rejected")

            self.assertEqual(preflight.call_count, 3)
            loop.assert_not_called()
            api.assert_not_called()
            run_dir = Path(temp_dir) / "run_rejected"
            payload = json.loads(
                (run_dir / "smoke_preflight.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["num_cases"], 3)
            self.assertEqual(payload["qualified_count"], 1)
            self.assertFalse(payload["passed"])
            self.assertEqual(payload["cases"], rows)
            self.assertFalse((run_dir / "episode_summary.jsonl").exists())
            self.assertFalse((run_dir / "smoke_summary.json").exists())
```

在现有合格批次测试中，用下面两行替换内联配置和 `fake_calibration`：

```python
config, fake_calibration = batch_config(temp_dir)
```

- [ ] **Step 2: 运行拒绝测试并确认 RED**

Run:

```bash
conda run -n vla_env python -m unittest \
  tests.vlm.grounding_smoke.test_runner.SmokeBatchContractTests.test_batch_preflights_all_cases_before_rejecting_without_api -v
```

Expected: import failure because `SmokePreflightError` 尚未定义，或断言失败因为批次仍进入 episode 循环。

- [ ] **Step 3: 实现错误类型和整批门禁**

在 `SmokeDependencies` 前增加：

```python
class SmokePreflightError(RuntimeError):
    """整批初始动态预检未通过。"""
```

在 `run_smoke_batch()` 创建 `batch_dir` 后、创建 `episode_summary_path` 前增加：

```python
    preflight_rows = [
        preflight_smoke_case(config, smoke_config, case)
        for case in cases
    ]
    preflight_summary = {
        "num_cases": len(preflight_rows),
        "qualified_count": sum(
            row["qualified"] for row in preflight_rows
        ),
        "passed": all(row["qualified"] for row in preflight_rows),
        "cases": preflight_rows,
    }
    _write_json(batch_dir / "smoke_preflight.json", preflight_summary)
    if not preflight_summary["passed"]:
        failures = ", ".join(
            f"episode={row['episode_idx']} seed={row['seed']} "
            f"direction={row['start_direction']} "
            f"reason={row['rejection_reason']}"
            for row in preflight_rows
            if not row["qualified"]
        )
        raise SmokePreflightError(f"smoke 初始动态预检失败: {failures}")
```

- [ ] **Step 4: 运行拒绝测试并确认 GREEN**

Run: 与 Step 2 相同。

Expected: `Ran 1 test`、`OK`，三个预检都执行，控制循环与 API 均为0次。

- [ ] **Step 5: 调整合格批次测试，证明两阶段顺序**

在 `test_batch_uses_three_isolated_cases_and_writes_summary` 的 patch 集合中加入：

```python
patch(
    "vla_project.vlm.grounding_smoke.runner.preflight_smoke_case",
    side_effect=[
        {**case, "qualified": True, "rejection_reason": None}
        for case in build_smoke_cases(config["grounding_smoke"])
    ],
) as preflight,
```

并增加断言：

```python
self.assertEqual(preflight.call_count, 3)
self.assertEqual(loop.call_count, 3)
self.assertTrue((batch_dir / "smoke_preflight.json").is_file())
preflight_payload = json.loads(
    (batch_dir / "smoke_preflight.json").read_text(encoding="utf-8")
)
self.assertTrue(preflight_payload["passed"])
self.assertEqual(preflight_payload["qualified_count"], 3)
```

- [ ] **Step 6: 运行完整 runner 测试并确认 GREEN**

Run:

```bash
conda run -n vla_env python -m unittest tests.vlm.grounding_smoke.test_runner -v
```

Expected: 全部通过，无真实 API 调用。

- [ ] **Step 7: 提交整批门禁**

```bash
git add src/vla_project/vlm/grounding_smoke/runner.py tests/vlm/grounding_smoke/test_runner.py
git commit -m "feat: gate grounding smoke on batch preflight"
```

---

### Task 3: 无 API 动态验收与项目记录

**Files:**
- Modify: `docs/agent/CURRENT_STATUS.md`
- Modify: `docs/debugging/BUGLOG.md`
- Modify: `docs/worklog/WORKLOG.md`
- Verify: `outputs/vlm_evaluations/grounding_world_smoke/run_20260719_batch_preflight_validation_v1/smoke_preflight.json`（生成物，不提交）

**Interfaces:**
- Consumes: `vla_project.vlm.grounding_smoke.runner:main`
- Produces: 当前三个固定案例的无 API 预检证据和更新后的权威文档

- [ ] **Step 1: 运行定向契约测试**

```bash
conda run -n vla_env python -m unittest \
  tests.test_config_contract \
  tests.test_package_metadata \
  tests.vlm.grounding_smoke.test_targeting \
  tests.vlm.grounding_smoke.test_runner \
  tests.vlm.grounding_smoke.test_screening -v
```

Expected: 所有测试通过，0 failures、0 errors。

- [ ] **Step 2: 运行全量自动测试和编译检查**

```bash
conda run -n vla_env python -m unittest discover -s tests -v
conda run -n vla_env python -m compileall -q src tests
```

Expected: 全量测试通过；`compileall` 退出码0。

- [ ] **Step 3: 运行当前固定案例的无 API 动态验收**

在隔离 worktree 根目录运行，显式让 Python 使用当前 worktree 的 `src`：

```bash
conda run -n vla_env env \
  -u VLA_API_BASE_URL \
  -u VLA_API_KEY \
  -u VLA_MODEL_NAME \
  PYTHONPATH=src python -m \
  vla_project.vlm.grounding_smoke.runner \
  --run-name run_20260719_batch_preflight_validation_v1
```

Expected: 命令以 `SmokePreflightError` 非零退出；错误同时列出 `52-left` 的
`visibility_below_threshold` 和 `54-front` 的 `start_pose_error`。这是门禁生效，不是
在线闭环失败；三个 API 环境变量已从该进程显式移除，整个过程不得出现 API 响应或
episode trace。

- [ ] **Step 4: 只读校验预检证据**

```bash
conda run -n vla_env python -c "import json; from pathlib import Path; p=Path('outputs/vlm_evaluations/grounding_world_smoke/run_20260719_batch_preflight_validation_v1/smoke_preflight.json'); d=json.loads(p.read_text()); assert len(d['cases']) == 3; assert d['passed'] is False; assert {r['rejection_reason'] for r in d['cases'] if not r['qualified']} == {'visibility_below_threshold', 'start_pose_error'}; print({'num_cases': d['num_cases'], 'qualified_count': d['qualified_count'], 'passed': d['passed'], 'rejections': [(r['seed'], r['start_direction'], r['rejection_reason']) for r in d['cases'] if not r['qualified']]})"
test ! -e outputs/vlm_evaluations/grounding_world_smoke/run_20260719_batch_preflight_validation_v1/episode_summary.jsonl
test ! -e outputs/vlm_evaluations/grounding_world_smoke/run_20260719_batch_preflight_validation_v1/smoke_summary.json
```

Expected: 恰好3条案例记录、`passed=false`、拒绝原因集合与预期一致，两个在线摘要文件均不存在。结合自动测试中的 API mock 0调用断言，证明整批拒绝发生在任何付费请求前。

- [ ] **Step 5: 更新权威文档**

在 `CURRENT_STATUS.md` 中把整批动态预检标为已实现，并保留“成功评分契约”和“普通闭环
异常完整摘要”作为未解决问题。在 `BUGLOG.md` 的 BUG-006 中记录 RED/GREEN 测试、真实
无 API 预检指标和剩余边界；在 `WORKLOG.md` 追加实现过程与验证命令结果。不得把预检
失败写成真实闭环失败，也不得声称第二轮付费实验已经运行。

- [ ] **Step 6: 最终一致性验证**

```bash
git diff --check
git status --short
```

Expected: `git diff --check` 退出码0；状态只包含本计划范围内的源码、测试和文档修改，
以及进入 worktree 前已知且不属于本任务的文件。

- [ ] **Step 7: 提交验收记录**

```bash
git add docs/agent/CURRENT_STATUS.md docs/debugging/BUGLOG.md docs/worklog/WORKLOG.md
git commit -m "docs: record grounding smoke batch preflight"
```

提交后重新运行 `git status --short`，确认任务文件均已提交且没有纳入无关文件。
