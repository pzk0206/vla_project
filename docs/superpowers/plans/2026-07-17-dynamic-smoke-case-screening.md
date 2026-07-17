# 动态 Smoke 案例筛选实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 不调用 VLM，使用正式闭环相同的 PyBullet 2cm 动作轨迹，从 seeds 55–100 中为 left、right、front 各固定一个全程 clear、可达且 seed 不重复的正式 smoke 案例。

**Architecture:** 新增独立 `screen_grounding_smoke_cases.py`，把纯判定/选择与 PyBullet 候选执行分开；筛选允许用真值生成固定正确动作，但输出只用于选择案例。正式 `run_grounding_smoke.py` 继续保持真值隔离，并在人工审计筛选结果后才更新 `sim_config.yaml`。

**Tech Stack:** Python 3.10、PyBullet、OpenCV、NumPy、YAML、标准库 `unittest`、JSON/JSONL、Git。

## Global Constraints

- 候选 seeds 固定为闭区间 55–100；方向固定为 left、right、front，不加入 back。
- 每个候选从 0.10m 起点开始，执行 4 次 0.02m 动作，形成 10/8/6/4/约2cm 五个观察点。
- 相机固定为既有 448×448 正俯视配置；可见率参考像素378，五个观察点均须 `>=0.75`。
- 起点和每次动作后的请求目标 3D 误差均须 `<=0.005m`；最终真实 XY 距离须 `<=0.03m`。
- 先评估全部候选，再按 left、right、front 顺序选择最小且未被占用的合格 seed。
- 筛选不得导入或调用 Qwen API，不得降低阈值，不得自动修改 `sim_config.yaml`。
- 全部命令使用 conda 环境 `vla_env`；筛选输出保存在被 Git 忽略的 `vlm_smoke_screening_runs/`。

---

## 文件结构

- Create: `screen_grounding_smoke_cases.py` — 纯资格判定、确定性选择、真实 PyBullet 候选运行和 CLI。
- Create: `tests/test_screen_grounding_smoke_cases.py` — 资格边界、选择顺序、真实适配器 mock 和无 API 契约。
- Modify: `sim_config.yaml` — 增加筛选范围与冻结阈值；筛选审计后写入三个正式 seeds。
- Modify: `tests/test_config_contract.py` — 保护筛选配置与最终正式案例契约。
- Modify: `.gitignore` — 忽略本地筛选图片和 JSON 输出。
- Modify: `README.md`、`docs/worklog/WORKLOG.md`、`docs/planning/vla_robotic_study_plan.md`、`docs/debugging/BUGLOG.md` — 同步首轮失败、筛选结果和重跑结果。

---

### Task 1: 纯资格判定、选择规则和配置契约

**Files:**
- Create: `tests/test_screen_grounding_smoke_cases.py`
- Create: `screen_grounding_smoke_cases.py`
- Modify: `sim_config.yaml`
- Modify: `tests/test_config_contract.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `classify_candidate(step_rows, settings, error=None) -> dict`。
- Produces: `select_qualified_cases(candidate_rows, directions) -> list[dict]`。
- Produces: `config["grounding_smoke"]["screening"]: dict`。

- [ ] **Step 1: 写失败测试**

`tests/test_screen_grounding_smoke_cases.py` 必须包含以下可执行测试：

```python
import unittest

from screen_grounding_smoke_cases import (
    classify_candidate,
    select_qualified_cases,
)

SETTINGS = {
    "num_actions": 4,
    "clear_visibility_threshold": 0.75,
    "max_pose_error": 0.005,
    "max_final_distance_xy": 0.03,
}

def valid_steps():
    return [
        {
            "observation_step": step,
            "block_visibility_ratio": 0.80,
            "target_error_3d": 0.004,
            "true_distance_xy": distance,
        }
        for step, distance in enumerate((0.10, 0.08, 0.06, 0.04, 0.02))
    ]

class CandidateClassificationTests(unittest.TestCase):
    def test_five_valid_observations_qualify(self):
        result = classify_candidate(valid_steps(), SETTINGS)
        self.assertTrue(result["qualified"])
        self.assertEqual(result["reason"], "qualified")

    def test_visibility_equal_to_threshold_is_allowed(self):
        rows = valid_steps()
        rows[2]["block_visibility_ratio"] = 0.75
        self.assertTrue(classify_candidate(rows, SETTINGS)["qualified"])

    def test_visibility_below_threshold_is_rejected(self):
        rows = valid_steps()
        rows[2]["block_visibility_ratio"] = 0.749999
        self.assertEqual(
            classify_candidate(rows, SETTINGS)["reason"],
            "visibility_below_threshold",
        )

    def test_start_and_motion_pose_errors_have_distinct_reasons(self):
        start_rows = valid_steps()
        start_rows[0]["target_error_3d"] = 0.005001
        self.assertEqual(
            classify_candidate(start_rows, SETTINGS)["reason"],
            "start_pose_error",
        )
        motion_rows = valid_steps()
        motion_rows[1]["target_error_3d"] = 0.005001
        self.assertEqual(
            classify_candidate(motion_rows, SETTINGS)["reason"],
            "motion_target_error",
        )

    def test_final_distance_over_three_centimeters_is_rejected(self):
        rows = valid_steps()
        rows[-1]["true_distance_xy"] = 0.030001
        self.assertEqual(
            classify_candidate(rows, SETTINGS)["reason"],
            "final_distance_error",
        )

    def test_exception_and_incomplete_trace_are_rejected(self):
        self.assertEqual(
            classify_candidate([], SETTINGS, error="boom")["reason"],
            "candidate_error",
        )
        self.assertEqual(
            classify_candidate(valid_steps()[:-1], SETTINGS)["reason"],
            "incomplete_trace",
        )

class SelectionTests(unittest.TestCase):
    def test_selects_smallest_distinct_seed_in_direction_order(self):
        candidates = [
            {"seed": 55, "direction": direction, "qualified": True}
            for direction in ("left", "right", "front")
        ] + [
            {"seed": 56, "direction": "right", "qualified": True},
            {"seed": 57, "direction": "front", "qualified": True},
        ]
        selected = select_qualified_cases(
            candidates, ["left", "right", "front"]
        )
        self.assertEqual(
            [(row["seed"], row["direction"]) for row in selected],
            [(55, "left"), (56, "right"), (57, "front")],
        )

    def test_missing_direction_raises_without_partial_selection(self):
        with self.assertRaises(ValueError):
            select_qualified_cases(
                [{"seed": 55, "direction": "left", "qualified": True}],
                ["left", "right", "front"],
            )
```

在 `tests/test_config_contract.py` 断言：

```python
screening = self.config["grounding_smoke"]["screening"]
self.assertEqual(screening["output_dir"], "vlm_smoke_screening_runs")
self.assertEqual(screening["seed_range"], [55, 100])
self.assertEqual(screening["directions"], ["left", "right", "front"])
self.assertEqual(screening["num_actions"], 4)
self.assertEqual(screening["max_pose_error"], 0.005)
self.assertEqual(screening["max_final_distance_xy"], 0.03)
```

- [ ] **Step 2: 验证 RED**

```bash
conda run -n vla_env python -m unittest tests.test_screen_grounding_smoke_cases -v
```

Expected: ERROR，`screen_grounding_smoke_cases` 尚不存在。

- [ ] **Step 3: 实现纯函数与配置**

`screen_grounding_smoke_cases.py` 先实现：

```python
def classify_candidate(step_rows, settings, error=None):
    rows = list(step_rows)
    if error is not None:
        return {"qualified": False, "reason": "candidate_error", "error": error}
    for row in rows:
        if row["block_visibility_ratio"] < settings["clear_visibility_threshold"]:
            return {"qualified": False, "reason": "visibility_below_threshold"}
        if row["target_error_3d"] > settings["max_pose_error"]:
            reason = "start_pose_error" if row["observation_step"] == 0 else "motion_target_error"
            return {"qualified": False, "reason": reason}
    expected = settings["num_actions"] + 1
    if len(rows) != expected:
        return {"qualified": False, "reason": "incomplete_trace"}
    if rows[-1]["true_distance_xy"] > settings["max_final_distance_xy"]:
        return {"qualified": False, "reason": "final_distance_error"}
    return {"qualified": True, "reason": "qualified"}

def select_qualified_cases(candidate_rows, directions):
    selected = []
    used_seeds = set()
    rows = list(candidate_rows)
    for direction in directions:
        eligible = sorted(
            (
                row for row in rows
                if row["qualified"]
                and row["direction"] == direction
                and row["seed"] not in used_seeds
            ),
            key=lambda row: row["seed"],
        )
        if not eligible:
            raise ValueError(f"没有可选的 {direction} 合格案例")
        selected.append(eligible[0])
        used_seeds.add(eligible[0]["seed"])
    return selected
```

在 `sim_config.yaml` 的 `grounding_smoke` 下加入：

```yaml
  screening:
    output_dir: "vlm_smoke_screening_runs"
    seed_range: [55, 100]
    directions: ["left", "right", "front"]
    num_actions: 4
    max_pose_error: 0.005
    max_final_distance_xy: 0.03
```

在 `.gitignore` 加入 `vlm_smoke_screening_runs/`。

- [ ] **Step 4: 验证 GREEN 并提交**

```bash
conda run -n vla_env python -m unittest tests.test_screen_grounding_smoke_cases tests.test_config_contract -v
git diff --check
git add screen_grounding_smoke_cases.py tests/test_screen_grounding_smoke_cases.py sim_config.yaml tests/test_config_contract.py .gitignore
git commit -m "feat: define dynamic smoke case screening"
```

Expected: 新测试全部 PASS，筛选输出路径被忽略。

---

### Task 2: 真实 PyBullet 动态筛选入口

**Files:**
- Modify: `screen_grounding_smoke_cases.py`
- Modify: `tests/test_screen_grounding_smoke_cases.py`

**Interfaces:**
- Produces: `run_candidate(config, seed, direction, candidate_dir) -> dict`。
- Produces: `run_screening(config, run_name=None) -> tuple[Path, dict]`。

- [ ] **Step 1: 写真实适配器失败测试**

使用 mock 断言 `run_candidate()`：连接 DIRECT、搭建 10cm 起点、调用四次 `direction_to_target(direction, ..., 0.20, 0.02)`、产生五个 observation row、始终不调用 `call_openai_compatible_api`。使用 mock 候选结果断言 `run_screening()` 评估 seeds 55–100 的三个方向、写 `candidate_trace.jsonl` 和 `screening_summary.json`、已有目录抛出 `FileExistsError`。

测试必须直接断言：

```python
self.assertEqual(direction_to_target.call_count, 4)
self.assertEqual(len(result["steps"]), 5)
self.assertNotIn("call_openai_compatible_api", screen_grounding_smoke_cases.__dict__)
self.assertEqual(summary["num_candidates"], 46 * 3)
self.assertEqual(
    [(row["seed"], row["direction"]) for row in summary["selected_cases"]],
    [(55, "left"), (56, "right"), (57, "front")],
)
```

- [ ] **Step 2: 验证 RED**

```bash
conda run -n vla_env python -m unittest tests.test_screen_grounding_smoke_cases.ScreeningRunnerTests -v
```

Expected: ERROR/FAIL，真实 runner 尚未实现。

- [ ] **Step 3: 实现真实筛选**

`run_candidate()` 必须按固定顺序：seed、DIRECT、world、block settle、真值10cm起点、迭代 IK、固定448相机、起点观察、四次2cm动作与观察、分类、finally disconnect。核心实现使用以下结构：

```python
def run_candidate(config, seed, direction, candidate_dir):
    smoke = config["grounding_smoke"]
    screening = smoke["screening"]
    settings = {
        **screening,
        "clear_visibility_threshold": smoke["clear_visibility_threshold"],
    }
    candidate_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    random.seed(seed)
    connect_physics("DIRECT")
    try:
        _, robot_id = setup_world(config)
        block_id = load_block(config["task"])
        settle_object(config, config["task"]["initial_settle_steps"])
        block_pos = list(get_object_position(block_id))
        starts = build_balanced_ee_positions(
            block_pos,
            smoke["hover_z"] - block_pos[2],
            smoke["start_offset_xy"],
        )
        requested_target = starts[direction]
        actual_ee = list(reset_robot_to_target(
            robot_id,
            config["robot"],
            requested_target,
            tolerance=config["vlm_evaluation"]["balanced_pose_tolerance"],
            max_iterations=config["vlm_evaluation"]["balanced_pose_ik_iterations"],
        ))
        camera_config = copy.deepcopy(config["camera"])
        camera_config.update(config["vlm_evaluation"]["camera_override"])
        camera_eye = sample_camera_eye(camera_config)

        for observation_step in range(screening["num_actions"] + 1):
            image, segmentation = capture_rgb_and_segmentation(
                camera_config, camera_eye
            )
            image_path = candidate_dir / f"step_{observation_step:02d}.jpg"
            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError(f"无法写入筛选图片: {image_path}")
            current_block = list(get_object_position(block_id))
            row = {
                "observation_step": observation_step,
                "requested_target": list(requested_target),
                "actual_ee_pos": list(actual_ee),
                "target_error_3d": math.dist(actual_ee, requested_target),
                "true_block_pos": current_block,
                "true_distance_xy": math.hypot(
                    current_block[0] - actual_ee[0],
                    current_block[1] - actual_ee[1],
                ),
                **compute_visibility(
                    segmentation,
                    block_id,
                    smoke["visibility_reference_pixels"],
                ),
                "image_path": str(image_path),
            }
            rows.append(row)
            partial = classify_candidate(rows, settings)
            if partial["reason"] != "incomplete_trace":
                break
            if observation_step == screening["num_actions"]:
                break
            requested_target = direction_to_target(
                direction,
                actual_ee,
                smoke["hover_z"],
                smoke["move_step_xy"],
            )
            joints = calculate_target_joints(
                robot_id, config["robot"], requested_target
            )
            apply_joint_targets(robot_id, config["robot"], joints)
            settle_object(config, config["probe"]["sim_steps_per_action"])
            actual_ee = list(get_link_position(
                robot_id, config["robot"]["ee_link_index"]
            ))
        classification = classify_candidate(rows, settings)
    except Exception as exc:
        classification = classify_candidate(
            rows, settings, error=repr(exc)
        )
    finally:
        p.disconnect()
    return {
        "seed": seed,
        "direction": direction,
        "steps": rows,
        **classification,
    }
```

每个观察点的 row 字段必须与上面代码一致，特别是：

```python
{
    "observation_step": step,
    "requested_target": list(requested_target),
    "actual_ee_pos": list(actual_ee_pos),
    "target_error_3d": math.dist(actual_ee_pos, requested_target),
    "true_block_pos": list(block_pos),
    "true_distance_xy": math.hypot(
        block_pos[0] - actual_ee_pos[0],
        block_pos[1] - actual_ee_pos[1],
    ),
    **compute_visibility(segmentation, block_id, 378),
    "image_path": str(image_path),
}
```

`run_screening()` 必须评估完整笛卡尔积 `range(55, 101) × [left,right,front]`，逐候选立即追加 JSONL；完整实现形态为：

```python
def run_screening(config, run_name=None):
    screen = config["grounding_smoke"]["screening"]
    run_dir = make_run_dir(screen["output_dir"], run_name)
    trace_path = run_dir / "candidate_trace.jsonl"
    candidates = []
    for seed in range(screen["seed_range"][0], screen["seed_range"][1] + 1):
        for direction in screen["directions"]:
            candidate = run_candidate(
                config,
                seed,
                direction,
                run_dir / f"seed_{seed:03d}_{direction}",
            )
            candidates.append(candidate)
            append_jsonl(trace_path, candidate)
    selection_error = None
    try:
        selected = select_qualified_cases(candidates, screen["directions"])
    except ValueError as exc:
        selected = []
        selection_error = str(exc)
    summary = {
        "num_candidates": len(candidates),
        "num_qualified": sum(row["qualified"] for row in candidates),
        "reason_counts": dict(Counter(row["reason"] for row in candidates)),
        "selected_cases": [
            {"seed": row["seed"], "direction": row["direction"]}
            for row in selected
        ],
        "selection_error": selection_error,
        "passed": selection_error is None,
    }
    write_json(run_dir / "screening_summary.json", summary)
    return run_dir, summary
```

CLI 只打印路径、合格数、选择结果和 passed。

- [ ] **Step 4: 全量验证并提交**

```bash
conda run -n vla_env python -m unittest discover -s tests -v
conda run -n vla_env python screen_grounding_smoke_cases.py --help
git diff --check
git add screen_grounding_smoke_cases.py tests/test_screen_grounding_smoke_cases.py
git commit -m "feat: add pybullet smoke case screening"
```

Expected: 全部测试 PASS，测试期间真实 API 调用为0。

---

### Task 3: 运行筛选、审计并冻结正式案例

**Files:**
- Generate: `vlm_smoke_screening_runs/run_<timestamp>/`
- Modify: `sim_config.yaml`
- Modify: `tests/test_config_contract.py`
- Modify: `tests/test_run_grounding_smoke.py`

- [ ] **Step 1: 执行无 API 筛选**

```bash
conda run -n vla_env python screen_grounding_smoke_cases.py
```

Expected: 评估138个候选，生成 summary；命令不读取 API 环境变量。

- [ ] **Step 2: 审计选择结果**

```bash
conda run -n vla_env python -c "import json; from pathlib import Path; roots=sorted(Path('vlm_smoke_screening_runs').glob('run_*')); s=json.loads((roots[-1]/'screening_summary.json').read_text()); assert s['num_candidates']==138; assert s['passed']; assert len(s['selected_cases'])==3; assert len({x['seed'] for x in s['selected_cases']})==3; assert [x['direction'] for x in s['selected_cases']]==['left','right','front']; print(json.dumps(s,ensure_ascii=False,indent=2))"
```

Expected: 三个不同 seed，全程阈值通过。人工查看三个 selected candidate 目录的五张图。

- [ ] **Step 3: 用真实选择冻结配置测试**

从已审计 summary 读取三个整数，按 left/right/front 顺序替换 `grounding_smoke.seeds`；不修改方向、阈值或步长。把 `tests/test_config_contract.py` 和 `tests/test_run_grounding_smoke.py` 中固定 seed 断言同步为这三个真实整数。

- [ ] **Step 4: 验证并提交选择**

```bash
conda run -n vla_env python -m unittest tests.test_config_contract tests.test_run_grounding_smoke -v
git check-ignore -v vlm_smoke_screening_runs/example/screening_summary.json
git diff --check
git add sim_config.yaml tests/test_config_contract.py tests/test_run_grounding_smoke.py
git commit -m "test: freeze screened grounding smoke cases"
```

Expected: 配置只包含已审计选择，筛选产物不进入 Git。

---

### Task 4: 正式重跑、证据同步与分支完成

**Files:**
- Generate: `vlm_smoke_runs/run_<timestamp>/`
- Modify: `README.md`
- Modify: `docs/worklog/WORKLOG.md`
- Modify: `docs/planning/vla_robotic_study_plan.md`
- Modify: `docs/debugging/BUGLOG.md`

- [ ] **Step 1: 全量测试和 API/标定检查**

```bash
conda run -n vla_env python -m unittest discover -s tests -v
conda run -n vla_env python -c "import json,os,yaml; from pathlib import Path; c=yaml.safe_load(Path('sim_config.yaml').read_text()); s=c['grounding_smoke']; a=c['probe']['api']; assert all(os.getenv(a[n]) for n in ('base_url_env','api_key_env','model_env')); d=json.loads(Path(s['calibration_path']).read_text()); assert d['num_clear_calibration_samples']==s['expected_calibration_samples']; assert d['correction_x']==s['expected_correction_x']; assert d['correction_y']==s['expected_correction_y']; print('ready cases=',s['seeds'])"
```

- [ ] **Step 2: 执行并审计正式 smoke**

```bash
conda run -n vla_env python run_grounding_smoke.py
conda run -n vla_env python -c "import inspect,json,yaml; from pathlib import Path; from grounding_targeting import compute_grounding_action; c=yaml.safe_load(Path('sim_config.yaml').read_text()); roots=sorted(Path('vlm_smoke_runs').glob('run_*')); s=json.loads((roots[-1]/'smoke_summary.json').read_text()); assert [e['seed'] for e in s['episodes']]==c['grounding_smoke']['seeds']; assert s['total_api_calls']<=30; assert 'block_pos' not in inspect.signature(compute_grounding_action).parameters; print(json.dumps(s,ensure_ascii=False,indent=2))"
```

Expected: 无论 passed 真或假，都存在完整三 episode summary；不得临时换 seed 或阈值。

- [ ] **Step 3: 同步四份文档**

四份文档必须同时记录：首轮 52–54 的三个前置条件问题；筛选范围、合格数、reason 计数和三个入选案例；正式三个 episode 的步数、终止原因、最终真实 XY 距离、API 调用数和 passed；结论仅适用于 clear-only，不外推到 back/遮挡恢复。

- [ ] **Step 4: 最终验证并提交**

```bash
rg -n "screening|smoke|clear|API|下一步" README.md docs/worklog/WORKLOG.md docs/planning/vla_robotic_study_plan.md docs/debugging/BUGLOG.md
git diff --check
conda run -n vla_env python -m unittest discover -s tests -v
git add README.md docs/worklog/WORKLOG.md docs/planning/vla_robotic_study_plan.md docs/debugging/BUGLOG.md
git commit -m "docs: analyze screened grounding smoke test"
```

---

## 完成判据

- 138 个候选全部由真实 PyBullet 动态轨迹筛选，Qwen 调用数为0。
- left/right/front 各固定一个 seed 升序下最小且互不重复的合格案例。
- 五个观察点全部可见率 `>=0.75`，位置误差 `<=0.005m`，最终真实 XY `<=0.03m`。
- 正式控制仍不接收 `block_pos`，三个固定案例不在运行中替换，总 API 调用不超过30。
- 自动测试、筛选证据、正式 summary 和四份项目文档使用一致指标。
