# Grounding Smoke 固定案例重选实施计划

> **面向代理执行者：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，逐项执行本计划。步骤使用复选框（`- [ ]`）跟踪。

**目标：** 从 seeds 55–100 中确定性选择各一个初始资格合格的 `left`、`right` 和 `front` smoke 案例，将这些案例冻结到配置中，并证明三个案例全部通过无 API 的只预检运行。

**架构：** 将现有筛选工作流调整为每个候选只进行一次初始观察，并仅使用起点姿态误差和首帧可见率作为资格输入。把 runner 现有的三案例预检提取为可复用的批次边界，通过 `--preflight-only` 暴露；然后使用真实无 API 输出冻结选中的 seeds，并更新项目证据。

**技术栈：** Python 3、`unittest`、`unittest.mock`、PyBullet、OpenCV、NumPy、YAML 配置、JSON/JSONL 证据。

## 全局约束

- 候选 seeds 严格保持为包含端点的 55–100。
- 方向顺序严格保持为 `["left", "right", "front"]`。
- 资格边界包含等号：`start_pose_error <= 0.005m` 且 `block_visibility_ratio >= 0.75`。
- 三个选中 seed 必须互不重复，并且必须是在方向顺序选择规则下可用的最小 seed。
- 后续帧可见率和运动目标误差不作为候选资格输入。
- 筛选和 `--preflight-only` 绝不能调用 `call_openai_compatible_api()`。
- 不修改 grounding、反投影、冻结 XY 补偿、动作选择、IK 行为、目标保持、评分或 API 上限。
- 不运行真实在线 smoke；付费/API 驱动的运行仍需用户另行明确批准。
- 保留历史筛选输出；每次新运行使用唯一输出目录。
- 保留用户未跟踪的 `docs/superpowers/plans/2026-07-19-agent-file-placement-rules.md`。

---

### 任务 1：用初始观察资格替换全轨迹资格

**文件：**
- 修改：`tests/vlm/grounding_smoke/test_screening.py`
- 修改：`src/vla_project/vlm/grounding_smoke/screening.py`
- 修改：`tests/test_config_contract.py`
- 修改：`sim_config.yaml`

**接口：**
- 输入：`config["grounding_smoke"]`、`config["grounding_smoke"]["screening"]`
- 产出：`classify_candidate(observation: dict | None, settings: dict, error: str | None = None) -> dict`
- 产出：`run_candidate(config: dict, seed: int, direction: str, candidate_dir: Path | str) -> dict`
- 保持：`select_qualified_cases(candidate_rows: Iterable[dict], directions: Iterable[str]) -> list[dict]`
- 保持：`run_screening(config: dict, run_name: str | None = None) -> tuple[Path, dict]`

- [ ] **步骤 1：用初始边界测试替换轨迹分类测试**

编写测试前，明确它们要捕获的破坏：

- 将姿态误差判断从 `>` 改成 `>=` 时，等值边界测试必须失败。
- 将可见率判断从 `<` 改成 `<=` 时，等值边界测试必须失败。
- 在姿态误差之前检查可见率时，拒绝原因优先级测试必须失败。
- 在没有观察记录时返回合格结果，缺失观察测试必须失败。

将 `tests/vlm/grounding_smoke/test_screening.py` 中的 `SETTINGS`、`valid_steps()` 和
`CandidateClassificationTests` 替换为：

```python
SETTINGS = {
    "clear_visibility_threshold": 0.75,
    "max_pose_error": 0.005,
}


def valid_observation():
    return {
        "target_error_3d": 0.004,
        "block_visibility_ratio": 0.80,
    }


class CandidateClassificationTests(unittest.TestCase):
    def test_initial_observation_qualifies_at_inclusive_boundaries(self):
        observation = valid_observation()
        observation["target_error_3d"] = 0.005
        observation["block_visibility_ratio"] = 0.75

        result = classify_candidate(observation, SETTINGS)

        self.assertTrue(result["qualified"])
        self.assertEqual(result["reason"], "qualified")

    def test_start_pose_error_strictly_over_limit_is_rejected(self):
        observation = valid_observation()
        observation["target_error_3d"] = 0.005001

        result = classify_candidate(observation, SETTINGS)

        self.assertFalse(result["qualified"])
        self.assertEqual(result["reason"], "start_pose_error")

    def test_visibility_strictly_below_threshold_is_rejected(self):
        observation = valid_observation()
        observation["block_visibility_ratio"] = 0.749999

        result = classify_candidate(observation, SETTINGS)

        self.assertFalse(result["qualified"])
        self.assertEqual(result["reason"], "visibility_below_threshold")

    def test_start_pose_reason_precedes_visibility_reason(self):
        observation = {
            "target_error_3d": 0.006,
            "block_visibility_ratio": 0.50,
        }

        result = classify_candidate(observation, SETTINGS)

        self.assertEqual(result["reason"], "start_pose_error")

    def test_exception_is_rejected_with_evidence(self):
        result = classify_candidate(None, SETTINGS, error="boom")

        self.assertFalse(result["qualified"])
        self.assertEqual(result["reason"], "candidate_error")
        self.assertEqual(result["error"], "boom")

    def test_missing_observation_is_candidate_error(self):
        result = classify_candidate(None, SETTINGS)

        self.assertFalse(result["qualified"])
        self.assertEqual(result["reason"], "candidate_error")
        self.assertEqual(result["error"], "missing_initial_observation")
```

- [ ] **步骤 2：运行分类测试并验证 RED**

运行：

```bash
python -m unittest \
  tests.vlm.grounding_smoke.test_screening.CandidateClassificationTests -v
```

预期：FAIL，因为现有分类器需要由五次观察组成的可迭代对象，并检查旧的全轨迹契约。

- [ ] **步骤 3：实现最小的初始观察分类器**

将 `src/vla_project/vlm/grounding_smoke/screening.py` 中的
`classify_candidate()` 替换为：

```python
def classify_candidate(observation, settings, error=None):
    """按起点姿态和首帧可见率分类一个候选。"""
    if error is not None:
        return {
            "qualified": False,
            "reason": "candidate_error",
            "error": error,
        }
    if observation is None:
        return {
            "qualified": False,
            "reason": "candidate_error",
            "error": "missing_initial_observation",
        }
    if observation["target_error_3d"] > settings["max_pose_error"]:
        return {"qualified": False, "reason": "start_pose_error"}
    if (
        observation["block_visibility_ratio"]
        < settings["clear_visibility_threshold"]
    ):
        return {
            "qualified": False,
            "reason": "visibility_below_threshold",
        }
    return {"qualified": True, "reason": "qualified"}
```

- [ ] **步骤 4：运行分类测试并验证 GREEN**

运行：

```bash
python -m unittest \
  tests.vlm.grounding_smoke.test_screening.CandidateClassificationTests -v
```

预期：6 个测试 PASS。

- [ ] **步骤 5：用单帧行为测试替换基于运动的候选测试**

编写测试前，明确它要捕获的破坏：重新引入任何对 `direction_to_target()`、关节目标计算、
关节驱动或运动后 link 读取的调用都必须失败，因为这些名称不会被 patch，候选必须只用
一次观察完成。

将 `test_candidate_uses_four_real_control_steps_and_five_observations()`
替换为：

```python
def test_candidate_uses_only_the_initial_observation(self):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    segmentation = np.full((20, 20), 20, dtype=np.int64)
    with tempfile.TemporaryDirectory() as temp_dir:
        config = screening_config(temp_dir)
        candidate_dir = Path(temp_dir) / "seed_055_left"
        with (
            patch(
                "vla_project.vlm.grounding_smoke.screening.connect_physics"
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.setup_world",
                return_value=(1, 10),
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.load_block",
                return_value=20,
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.settle_object"
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.get_object_position",
                return_value=[0.0, 0.4, 0.05],
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.build_balanced_ee_positions",
                return_value={"left": [0.1, 0.4, 0.2]},
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.reset_robot_to_target",
                return_value=[0.1, 0.4, 0.2],
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.sample_camera_eye",
                return_value=[0.0, 0.4, 3.0],
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.capture_rgb_and_segmentation",
                return_value=(image, segmentation),
            ) as capture,
            patch(
                "vla_project.vlm.grounding_smoke.screening.cv2.imwrite",
                return_value=True,
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.p.disconnect"
            ) as disconnect,
        ):
            result = run_candidate(config, 55, "left", candidate_dir)

    self.assertTrue(result["qualified"])
    self.assertEqual(result["reason"], "qualified")
    self.assertEqual(result["seed"], 55)
    self.assertEqual(result["direction"], "left")
    self.assertEqual(result["target_error_3d"], 0.0)
    self.assertEqual(result["block_visibility_ratio"], 1.0)
    self.assertTrue(result["image_path"].endswith("step_00.jpg"))
    capture.assert_called_once()
    disconnect.assert_called_once()
    self.assertNotIn(
        "call_openai_compatible_api",
        vla_project.vlm.grounding_smoke.screening.__dict__,
    )
```

增加以下清理/错误测试：

```python
def test_candidate_error_is_recorded_and_physics_is_released(self):
    with tempfile.TemporaryDirectory() as temp_dir:
        config = screening_config(temp_dir)
        candidate_dir = Path(temp_dir) / "seed_055_left"
        with (
            patch(
                "vla_project.vlm.grounding_smoke.screening.connect_physics"
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.setup_world",
                side_effect=RuntimeError("setup failed"),
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.p.disconnect"
            ) as disconnect,
        ):
            result = run_candidate(config, 55, "left", candidate_dir)

    self.assertFalse(result["qualified"])
    self.assertEqual(result["reason"], "candidate_error")
    self.assertIn("setup failed", result["error"])
    disconnect.assert_called_once()
```

在现有 `test_screening_evaluates_full_cartesian_product_and_writes_summary()`
中增加以下行为断言，使保存的摘要冻结用于生成 `selected_cases` 的规则：

```python
self.assertEqual(
    summary["selection_rule"],
    {
        "seed_range": [55, 100],
        "directions": ["left", "right", "front"],
        "max_pose_error": 0.005,
        "clear_visibility_threshold": 0.75,
        "observation_scope": "initial_only",
        "distinct_seeds": True,
        "selection": "smallest_seed_in_direction_order",
    },
)
```

- [ ] **步骤 6：运行候选测试并验证 RED**

运行：

```bash
python -m unittest \
  tests.vlm.grounding_smoke.test_screening.ScreeningRunnerTests.test_candidate_uses_only_the_initial_observation \
  tests.vlm.grounding_smoke.test_screening.ScreeningRunnerTests.test_candidate_error_is_recorded_and_physics_is_released -v
```

预期：单帧测试 FAIL，因为现有实现尝试执行四个运动步骤；错误测试可能已经通过，并在
重构期间保护清理行为。

- [ ] **步骤 7：重构观察捕获和候选执行**

在 `src/vla_project/vlm/grounding_smoke/screening.py` 中：

1. 删除仅供旧运动轨迹使用的导入：

```python
apply_joint_targets
calculate_target_joints
get_link_position
direction_to_target
```

2. 将 `_capture_observation()` 重命名为 `_capture_initial_observation()`，删除
`config` 和 `observation_step` 参数，始终保存 `step_00.jpg`，并保留现有的请求目标、
实际末端位置、真实积木、距离、可见率、相机位置和图片路径证据。

返回值必须为：

```python
return {
    "requested_target": list(requested_target),
    "actual_ee_pos": list(actual_ee),
    "target_error_3d": math.dist(actual_ee, requested_target),
    "true_block_pos": current_block,
    "true_distance_xy": math.hypot(
        current_block[0] - actual_ee[0],
        current_block[1] - actual_ee[1],
    ),
    **_compute_visibility(
        segmentation,
        block_id,
        smoke_config["visibility_reference_pixels"],
    ),
    "camera_eye": list(camera_eye),
    "image_path": str(image_path),
}
```

3. 将 `run_candidate()` 中的运动循环替换为一次调用：

```python
observation = _capture_initial_observation(
    smoke,
    candidate_dir,
    requested_target,
    actual_ee,
    block_id,
    camera_config,
    camera_eye,
)
classification = classify_candidate(observation, settings)
```

在 `try` 之前初始化 `observation = None`，并返回扁平记录：

```python
return {
    "seed": seed,
    "direction": direction,
    **(observation or {}),
    **classification,
}
```

在 `except` 分支中调用：

```python
classification = classify_candidate(
    observation,
    settings,
    error=repr(exc),
)
```

4. 在 `run_screening()` 构造的摘要中加入冻结的选择规则：

```python
"selection_rule": {
    "seed_range": list(screening["seed_range"]),
    "directions": list(screening["directions"]),
    "max_pose_error": screening["max_pose_error"],
    "clear_visibility_threshold": config["grounding_smoke"][
        "clear_visibility_threshold"
    ],
    "observation_scope": "initial_only",
    "distinct_seeds": True,
    "selection": "smallest_seed_in_direction_order",
},
```

- [ ] **步骤 8：运行全部筛选测试并验证 GREEN**

运行：

```bash
python -m unittest tests.vlm.grounding_smoke.test_screening -v
```

预期：全部筛选测试 PASS。

- [ ] **步骤 9：按 TDD 删除过时的轨迹筛选配置**

更新 `tests/test_config_contract.py`，使筛选断言为：

```python
self.assertEqual(screening["seed_range"], [55, 100])
self.assertEqual(
    screening["directions"], ["left", "right", "front"]
)
self.assertEqual(screening["max_pose_error"], 0.005)
self.assertNotIn("num_actions", screening)
self.assertNotIn("max_final_distance_xy", screening)
```

运行：

```bash
python -m unittest tests.test_config_contract.ConfigContractTests.test_grounding_smoke_config_is_safe_and_frozen -v
```

预期：FAIL，因为 `sim_config.yaml` 仍包含 `num_actions` 和
`max_final_distance_xy`。

然后从 `sim_config.yaml` 的 `grounding_smoke.screening` 中删除这两个键，从
`screening_config()` 测试夹具中删除相同的过时键，并修改注释以描述首帧资格。

运行：

```bash
python -m unittest \
  tests.test_config_contract.ConfigContractTests.test_grounding_smoke_config_is_safe_and_frozen \
  tests.vlm.grounding_smoke.test_screening -v
```

预期：所有选定测试 PASS。

- [ ] **步骤 10：提交任务 1**

```bash
git add \
  src/vla_project/vlm/grounding_smoke/screening.py \
  tests/vlm/grounding_smoke/test_screening.py \
  tests/test_config_contract.py \
  sim_config.yaml
git commit -m "feat: screen smoke cases from initial observations"
```

---

### 任务 2：提取批次预检并增加 `--preflight-only`

**文件：**
- 修改：`tests/vlm/grounding_smoke/test_runner.py`
- 修改：`src/vla_project/vlm/grounding_smoke/runner.py`

**接口：**
- 输入：`preflight_smoke_case(config: dict, smoke_config: dict, case: dict) -> dict`
- 产出：`run_smoke_preflight(config: dict, run_name: str | None = None) -> tuple[Path, dict]`
- 修改：`main(argv: list[str] | None = None) -> None`
- 保持：`run_smoke_batch(config: dict, run_name: str | None = None) -> tuple[Path, dict]`

- [ ] **步骤 1：为可复用的成功预检增加失败测试**

从 `runner` 导入时加入 `run_smoke_preflight`。

编写测试前，明确它要捕获的破坏：第一个案例后就返回、未持久化全部三条记录，或创建
在线 episode 证据时，测试都必须失败。

加入 `SmokeBatchContractTests`：

```python
def test_run_smoke_preflight_writes_three_qualified_cases_only(self):
    with tempfile.TemporaryDirectory() as temp_dir:
        config, _ = batch_config(temp_dir)
        rows = [
            {
                **case,
                "qualified": True,
                "rejection_reason": None,
            }
            for case in build_smoke_cases(config["grounding_smoke"])
        ]
        with patch(
            "vla_project.vlm.grounding_smoke.runner.preflight_smoke_case",
            side_effect=rows,
        ) as preflight:
            run_dir, summary = run_smoke_preflight(
                config,
                run_name="run_preflight_only",
            )

        payload = json.loads(
            (run_dir / "smoke_preflight.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(preflight.call_count, 3)
        self.assertEqual(summary, payload)
        self.assertEqual(summary["num_cases"], 3)
        self.assertEqual(summary["qualified_count"], 3)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["cases"], rows)
        self.assertFalse(
            (run_dir / "episode_summary.jsonl").exists()
        )
        self.assertFalse((run_dir / "smoke_summary.json").exists())
```

- [ ] **步骤 2：运行预检测试并验证 RED**

运行：

```bash
python -m unittest \
  tests.vlm.grounding_smoke.test_runner.SmokeBatchContractTests.test_run_smoke_preflight_writes_three_qualified_cases_only -v
```

预期：ERROR，因为 `run_smoke_preflight` 尚不存在。

- [ ] **步骤 3：提取可复用的批次预检**

在 `runner.py` 的 `run_smoke_batch()` 之前增加：

```python
def run_smoke_preflight(config, run_name=None):
    """只运行三个固定案例的无 API 起点预检。"""
    smoke_config = config["grounding_smoke"]
    cases = build_smoke_cases(smoke_config)
    batch_dir = make_run_dir(smoke_config["output_dir"], run_name)
    preflight_rows = [
        preflight_smoke_case(config, smoke_config, case)
        for case in cases
    ]
    summary = {
        "num_cases": len(preflight_rows),
        "qualified_count": sum(
            row["qualified"] for row in preflight_rows
        ),
        "passed": all(row["qualified"] for row in preflight_rows),
        "cases": preflight_rows,
    }
    _write_json(batch_dir / "smoke_preflight.json", summary)
    if not summary["passed"]:
        failures = ", ".join(
            f"episode={row['episode_idx']} seed={row['seed']} "
            f"direction={row['start_direction']} "
            f"reason={row['rejection_reason']}"
            for row in preflight_rows
            if not row["qualified"]
        )
        raise SmokePreflightError(
            f"smoke 初始动态预检失败: {failures}"
        )
    return batch_dir, summary
```

将 `run_smoke_batch()` 的开头改为：

```python
def run_smoke_batch(config, run_name=None):
    """运行固定三个真实 PyBullet/Qwen case，并保存批次证据。"""
    smoke_config = config["grounding_smoke"]
    cases = build_smoke_cases(smoke_config)
    batch_dir, _ = run_smoke_preflight(config, run_name=run_name)
    calibration = load_frozen_calibration(
        smoke_config["calibration_path"],
        smoke_config["expected_calibration_sample_ids"],
        smoke_config["expected_correction_x"],
        smoke_config["expected_correction_y"],
    )
```

从 `run_smoke_batch()` 删除重复的内联预检代码块。

- [ ] **步骤 4：运行批次契约测试并验证 GREEN**

运行：

```bash
python -m unittest \
  tests.vlm.grounding_smoke.test_runner.SmokeBatchContractTests.test_run_smoke_preflight_writes_three_qualified_cases_only \
  tests.vlm.grounding_smoke.test_runner.SmokeBatchContractTests.test_batch_preflights_all_cases_before_rejecting_without_api \
  tests.vlm.grounding_smoke.test_runner.SmokeBatchContractTests.test_batch_uses_three_isolated_cases_and_writes_summary -v
```

预期：3 个测试全部 PASS。仅当直接 patch `preflight_smoke_case` 已无法覆盖真实提取后的
编排时，才调整现有 patch，使其以 `run_smoke_preflight` 为目标。

- [ ] **步骤 5：为 `--preflight-only` 增加失败的 CLI 测试**

编写测试前，明确它要捕获的破坏：忽略 `--preflight-only` 并进入
`run_smoke_batch()` 时必须立即失败。

从 `runner` 导入 `main`，然后增加：

```python
def test_cli_preflight_only_never_enters_online_batch(self):
    with tempfile.TemporaryDirectory() as temp_dir:
        config, _ = batch_config(temp_dir)
        rows = [
            {
                **case,
                "qualified": True,
                "rejection_reason": None,
            }
            for case in build_smoke_cases(config["grounding_smoke"])
        ]
        with (
            patch(
                "vla_project.vlm.grounding_smoke.runner.load_config",
                return_value=config,
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.preflight_smoke_case",
                side_effect=rows,
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.run_smoke_batch",
                side_effect=AssertionError("online batch entered"),
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.call_openai_compatible_api",
                side_effect=AssertionError("API entered"),
            ),
        ):
            main([
                "--preflight-only",
                "--run-name",
                "run_cli_preflight",
            ])

        run_dir = Path(temp_dir) / "run_cli_preflight"
        payload = json.loads(
            (run_dir / "smoke_preflight.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["qualified_count"], 3)
        self.assertFalse(
            (run_dir / "episode_summary.jsonl").exists()
        )
        self.assertFalse((run_dir / "smoke_summary.json").exists())
```

- [ ] **步骤 6：运行 CLI 测试并验证 RED**

运行：

```bash
python -m unittest \
  tests.vlm.grounding_smoke.test_runner.SmokeBatchContractTests.test_cli_preflight_only_never_enters_online_batch -v
```

预期：ERROR，因为 `main()` 不接受 `argv`，并且解析器没有定义
`--preflight-only`。

- [ ] **步骤 7：实现 `--preflight-only`**

修改：

```python
def main(argv=None):
```

增加参数：

```python
parser.add_argument(
    "--preflight-only",
    action="store_true",
    help="只运行三个固定案例的无 API 起点预检",
)
args = parser.parse_args(argv)
```

在普通 `run_smoke_batch()` 分支之前增加：

```python
if args.preflight_only:
    batch_dir, summary = run_smoke_preflight(
        config,
        run_name=args.run_name,
    )
    print(f"run_dir={batch_dir}")
    print(
        f"qualified={summary['qualified_count']}/"
        f"{summary['num_cases']} passed={summary['passed']}"
    )
    return
```

此分支不得加载 calibration、构造 episode 依赖或检查 API 配置。

- [ ] **步骤 8：运行 runner 和 smoke 测试并验证 GREEN**

运行：

```bash
python -m unittest \
  tests.vlm.grounding_smoke.test_runner.SmokeBatchContractTests.test_cli_preflight_only_never_enters_online_batch \
  tests.vlm.grounding_smoke.test_runner -v
```

预期：全部 runner 测试 PASS。

- [ ] **步骤 9：提交任务 2**

```bash
git add \
  src/vla_project/vlm/grounding_smoke/runner.py \
  tests/vlm/grounding_smoke/test_runner.py
git commit -m "feat: add grounding smoke preflight-only mode"
```

---

### 任务 3：运行真实无 API 筛选并冻结选中的案例

**文件：**
- 生成：`outputs/vlm_evaluations/grounding_world_smoke_screening/run_20260726_initial_qualification_v1/`
- 修改：`sim_config.yaml`
- 修改：`tests/test_config_contract.py`
- 生成：`outputs/vlm_evaluations/grounding_world_smoke/run_20260726_fixed_cases_preflight_v1/`

**接口：**
- 输入：`vla-screen-grounding-smoke --run-name <name>`
- 输入：`screening_summary.json["selected_cases"]`
- 产出：按 `[left_seed, right_seed, front_seed]` 排序的 `grounding_smoke.seeds`
- 输入：`vla-run-grounding-smoke --preflight-only --run-name <name>`
- 产出：满足 `qualified_count == 3` 且 `passed == true` 的 `smoke_preflight.json`

- [ ] **步骤 1：运行真实的 138 候选无 API 筛选**

确认没有真实 smoke 命令正在运行，然后执行：

```bash
vla-screen-grounding-smoke \
  --run-name run_20260726_initial_qualification_v1
```

预期：

- 退出码为 0；
- 生成 138 条候选记录；
- 三个选中案例按 `left`、`right`、`front` 排序；
- 没有 API 请求，也不依赖 API 环境；
- 每个成功捕获的候选目录中只存在 `step_00.jpg`。

如果运行名称已经存在，选择下一个确定性后缀
`run_20260726_initial_qualification_v2`；绝不删除或覆盖现有目录。

- [ ] **步骤 2：修改配置前验证选择证据**

将 `SELECTION_SUMMARY` 设为实际新建的摘要路径，然后运行：

```bash
python -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); s=json.loads(p.read_text()); c=s["selected_cases"]; assert s["passed"] is True; assert s["num_candidates"] == 138; assert len(c) == 3; assert [x["direction"] for x in c] == ["left", "right", "front"]; assert len({x["seed"] for x in c}) == 3; assert all(55 <= x["seed"] <= 100 for x in c); print([x["seed"] for x in c])' \
  "$SELECTION_SUMMARY"
```

预期：准确打印保存证据选出的有序 `[left_seed, right_seed, front_seed]`。

- [ ] **步骤 3：为证据选中的案例增加失败配置契约**

将 `tests/test_config_contract.py` 中旧的固定 seed 断言替换为步骤 2 打印的三个确切
整数 seed：

```python
self.assertEqual(smoke["seeds"], [left_seed, right_seed, front_seed])
```

这里的 `left_seed`、`right_seed` 和 `front_seed` 表示实际保存的 `selected_cases`
中的整数；在测试中写入这些整数字面量，不要写变量名，也不要手工选择替代值。

运行：

```bash
python -m unittest \
  tests.test_config_contract.ConfigContractTests.test_grounding_smoke_config_is_safe_and_frozen -v
```

预期：FAIL，因为 `sim_config.yaml` 仍包含 `[52, 53, 54]`。

- [ ] **步骤 4：在配置中冻结证据选中的案例**

使用 `apply_patch`，仅将 `sim_config.yaml` 中的 `grounding_smoke.seeds`
替换为步骤 2 得到的确切有序整数列表。更新相邻注释，写明作为证据的无 API 筛选运行
目录。不要修改 `start_directions`。

运行：

```bash
python -m unittest \
  tests.test_config_contract.ConfigContractTests.test_grounding_smoke_config_is_safe_and_frozen \
  tests.vlm.grounding_smoke.test_runner.SmokeBatchContractTests.test_builds_exact_three_cases_without_back -v
```

如果 `test_builds_exact_three_cases_without_back` 仍硬编码 `[52, 53, 54]`，将其夹具
和预期案例改成相同的三个选中 seed 字面量。

预期：选定测试 PASS，并且 `build_smoke_cases()` 保持 `left/right/front` 映射。

- [ ] **步骤 5：运行固定三案例只预检命令**

运行：

```bash
vla-run-grounding-smoke \
  --preflight-only \
  --run-name run_20260726_fixed_cases_preflight_v1
```

预期：

- 退出码为 0；
- CLI 报告 `qualified=3/3 passed=True`；
- 运行目录包含 `smoke_preflight.json`；
- 不存在 episode 目录、`episode_summary.jsonl` 或 `smoke_summary.json`；
- 没有 VLM API 调用。

如果运行名称已经存在，使用下一个确定性后缀
`run_20260726_fixed_cases_preflight_v2`；绝不删除或覆盖证据。

- [ ] **步骤 6：验证预检证据**

将 `PREFLIGHT_SUMMARY` 设为实际新建的 `smoke_preflight.json`，然后运行：

```bash
python -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); s=json.loads(p.read_text()); assert s["num_cases"] == 3; assert s["qualified_count"] == 3; assert s["passed"] is True; assert len(s["cases"]) == 3; assert all(x["qualified"] and x["rejection_reason"] is None for x in s["cases"]); print([(x["seed"], x["start_direction"], x["start_pose_error"], x["block_visibility_ratio"]) for x in s["cases"]])' \
  "$PREFLIGHT_SUMMARY"
```

预期：打印三个选中案例及其通过的姿态/可见率指标。

- [ ] **步骤 7：提交任务 3**

生成输出继续保持忽略状态。只提交冻结的配置及其契约：

```bash
git add sim_config.yaml tests/test_config_contract.py \
  tests/vlm/grounding_smoke/test_runner.py
git commit -m "config: freeze qualified grounding smoke cases"
```

---

### 任务 4：记录证据并验证仓库

**文件：**
- 修改：`docs/agent/PROJECT_OVERVIEW.md`
- 修改：`docs/agent/CURRENT_STATUS.md`
- 修改：`docs/worklog/WORKLOG.md`
- 检查：`README.md`
- 检查：`docs/debugging/BUGLOG.md`

**接口：**
- 输入：实际 `screening_summary.json`
- 输入：实际 `smoke_preflight.json`
- 产出：当前权威阶段状态和可复现证据链接

- [ ] **步骤 1：更新稳定与动态项目知识**

使用任务 3 中的指标和路径字面量。

在 `PROJECT_OVERVIEW.md` 中：

- 替换“动态候选合格数为零”这一当前选择前提；
- 保留该结果，作为“全轨迹 clear 是无效前提”的证据；
- 加入选中的固定案例和 `3/3` 初始预检结果；
- 说明尚未运行第二次在线 smoke。

在 `CURRENT_STATUS.md` 中：

- 将最后核对日期更新为 `2026-07-26`；
- 将固定案例重选和 `3/3` 无 API 预检标记为完成；
- 删除关于 `52-left` 和 `54-front` 的过时未解决项；
- 将下一步设为审阅无 API 证据，并在任何真实 smoke 前获得用户明确批准；
- 保留自主 stop 和循环内批次异常处理为未解决问题。

在 `WORKLOG.md` 中增加一条带日期的记录，包含：

- 冻结的选择规则；
- 候选数量和原因计数；
- 选中的 seed-direction 对；
- 筛选输出路径；
- 三个固定案例的预检指标；
- 预检输出路径；
- 明确的 API 调用次数 0；
- 不运行在线 smoke 的决定。

- [ ] **步骤 2：检查 README 和 BUGLOG 中过时的权威表述**

运行：

```bash
rg -n "52-left|53-right|54-front|55–100|55-100|0个合格|固定案例|下一步" \
  README.md docs/debugging/BUGLOG.md
```

仅当文件把过时的固定案例表述为当前下一步时才更新。保留历史 Bug 证据，并明确标注其
历史性质，不要改写旧结果。

- [ ] **步骤 3：运行定向验证**

运行：

```bash
python -m unittest \
  tests.vlm.grounding_smoke.test_screening \
  tests.vlm.grounding_smoke.test_runner \
  tests.vlm.grounding_smoke.test_targeting \
  tests.test_config_contract \
  tests.test_package_metadata -v
```

预期：全部定向测试 PASS，失败数和错误数均为零。

- [ ] **步骤 4：运行全量验证**

运行：

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
git diff --check
git status --short
```

预期：

- 全量测试以 0 退出，失败数和错误数均为零；
- `compileall` 以 0 退出；
- `git diff --check` 以 0 退出；
- 只存在预期任务文件以及保留的既有未跟踪计划。

- [ ] **步骤 5：根据证据审计需求**

读取最终 `screening_summary.json` 和 `smoke_preflight.json` 并检查：

```text
[ ] 138 candidates were evaluated
[ ] selected directions are left/right/front in order
[ ] selected seeds are distinct and within 55–100
[ ] selection uses smallest eligible seed per direction after excluding used seeds
[ ] fixed preflight reports qualified_count=3 and passed=true
[ ] no episode or smoke summary exists in the preflight-only directory
[ ] no real online smoke was run
[ ] documentation records API calls=0
```

检查最小 seed 时，加载 `candidate_trace.jsonl`，独立筛选 `qualified=true`，应用按
方向顺序选择互异 seed 的规则，并将结果与
`screening_summary.json["selected_cases"]` 比较。

- [ ] **步骤 6：提交任务 4**

只暂存实际修改的文档：

```bash
git add \
  docs/agent/PROJECT_OVERVIEW.md \
  docs/agent/CURRENT_STATUS.md \
  docs/worklog/WORKLOG.md
git add README.md docs/debugging/BUGLOG.md
git diff --cached --stat
git commit -m "docs: record qualified grounding smoke cases"
```

如果 README 或 BUGLOG 不需要修改，则从 `git add` 中省略。提交前，确认既有未跟踪文件
`docs/superpowers/plans/2026-07-19-agent-file-placement-rules.md` 未被暂存。

- [ ] **步骤 7：报告执行边界**

报告：

- 选中的固定案例；
- 筛选和预检证据路径；
- 确切的定向和全量测试数量；
- API 调用次数：0；
- 真实在线 smoke：未运行；
- 剩余要求：执行付费/API 驱动的 smoke 命令前，必须获得用户明确批准。
