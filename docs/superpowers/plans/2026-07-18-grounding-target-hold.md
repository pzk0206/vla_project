# Grounding 最近可靠目标恢复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在红块短暂不可见时复用最近一次合法 VLM 世界坐标，结合当前末端位置重新计算动作，并在连续使用4步后仍无法重新定位时安全停止。

**Architecture:** 在 `grounding_targeting.py` 中抽出“世界目标坐标 + 当前末端坐标 -> 安全动作”的纯计算入口，使 fresh VLM 目标和 held VLM 目标共用同一控制规则。`run_grounding_smoke.py` 维护目标缓存与年龄，只在低可见率且已有缓存时进入 held 路径；PyBullet 红块真值继续只存在于独立评分闭包。

**Tech Stack:** Python 3.10、PyBullet、OpenCV、NumPy、YAML、标准库 `unittest`、JSON/JSONL、Git、conda 环境 `vla_env`。

## Global Constraints

- `max_stale_target_steps` 固定为 `4`；第1至第4次 held 决策允许执行，第5次尝试以 `stale_target_limit` 中止。
- 缓存内容只能来自已经通过反投影、冻结补偿、工作区和目标跳变检查的 VLM 目标坐标。
- held 步骤不得调用 VLM，不得重复应用冻结补偿，不得复用旧动作方向。
- 每个 held 步骤必须使用当前 `ee_pos` 重新计算方向、预测距离、`stop` 和无进展状态。
- 首帧低可见且无缓存时保持 `visibility_out_of_scope`；清晰画面下 API、无效框、几何和控制错误保持原终止原因。
- `compute_grounding_action()` 和 held 纯计算入口的参数都不得包含 `block_pos` 或 `true_block_pos`。
- trace 必须区分 `fresh_vlm`、`held_vlm_target` 和未产生动作的中止行；真实距离仍只做事后评分。
- 全部测试和执行命令使用 `conda run -n vla_env`。

---

## 文件结构

- Modify: `sim_config.yaml` — 冻结最大连续历史目标步数。
- Modify: `tests/test_config_contract.py` — 保护 `max_stale_target_steps=4`。
- Modify: `grounding_targeting.py` — 新增从已补偿世界目标生成安全动作的纯计算入口。
- Modify: `tests/test_grounding_targeting.py` — 验证 held 目标会随当前末端位置重新计算且不接收真值。
- Modify: `run_grounding_smoke.py` — 维护目标缓存、目标年龄、fresh/held 状态机和摘要字段。
- Modify: `tests/test_run_grounding_smoke.py` — 覆盖首帧拒绝、4步保持、第5步中止、重新定位重置和批次判定。
- Modify after real run: `README.md`、`docs/worklog/WORKLOG.md`、`docs/planning/vla_robotic_study_plan.md`、`docs/debugging/BUGLOG.md` — 写入真实 smoke 指标，不能预写成功。

---

### Task 1: 冻结最大失效步数配置

**Files:**
- Modify: `tests/test_config_contract.py`
- Modify: `sim_config.yaml`

**Interfaces:**
- Produces: `config["grounding_smoke"]["max_stale_target_steps"] == 4`。

- [ ] **Step 1: 写失败测试**

在 `ConfigContractTests.test_grounding_smoke_config_is_frozen_and_safe` 中加入：

```python
self.assertEqual(smoke["max_stale_target_steps"], 4)
```

- [ ] **Step 2: 运行测试并确认 RED**

```bash
conda run -n vla_env python -m unittest \
  tests.test_config_contract.ConfigContractTests.test_grounding_smoke_config_is_frozen_and_safe -v
```

Expected: FAIL/ERROR，`grounding_smoke` 尚无 `max_stale_target_steps`。

- [ ] **Step 3: 加入最小配置**

在 `sim_config.yaml` 的 `grounding_smoke` 下加入：

```yaml
  max_stale_target_steps: 4          # 最多连续复用4步可靠VLM目标，第5步仍不可见则停止。
```

- [ ] **Step 4: 运行测试并确认 GREEN**

```bash
conda run -n vla_env python -m unittest tests.test_config_contract -v
```

Expected: 配置契约测试全部 PASS。

- [ ] **Step 5: 提交配置契约**

```bash
git add sim_config.yaml tests/test_config_contract.py
git commit -m "test: freeze grounding target hold limit"
```

---

### Task 2: 抽出缓存世界目标的安全动作策略

**Files:**
- Modify: `tests/test_grounding_targeting.py`
- Modify: `grounding_targeting.py`

**Interfaces:**
- Produces: `compute_action_from_world_target(target_world, ee_pos, safety_state, settings, target_jump_xy=0.0) -> dict`。
- Preserves: `compute_grounding_action(red_block_box, image_size, view_matrix, projection_matrix, calibration, ee_pos, safety_state, settings) -> dict`。

- [ ] **Step 1: 写 held 目标失败测试**

在 `tests/test_grounding_targeting.py` 导入新函数并加入：

```python
from grounding_targeting import compute_action_from_world_target


def test_held_target_recomputes_direction_from_current_ee(self):
    first = compute_action_from_world_target(
        target_world=[0.10, 0.45, 0.0],
        ee_pos=[0.00, 0.45, 0.20],
        safety_state=empty_safety_state(),
        settings=SETTINGS,
    )
    second = compute_action_from_world_target(
        target_world=[0.10, 0.45, 0.0],
        ee_pos=[0.09, 0.45, 0.20],
        safety_state=first["safety_state"],
        settings=SETTINGS,
    )

    self.assertEqual(first["direction"], "right")
    self.assertEqual(second["direction"], "stop")
    self.assertEqual(second["corrected_target_world"], [0.10, 0.45, 0.0])


def test_held_target_interface_cannot_receive_block_truth(self):
    parameters = inspect.signature(compute_action_from_world_target).parameters
    self.assertNotIn("block_pos", parameters)
    self.assertNotIn("true_block_pos", parameters)
```

- [ ] **Step 2: 运行测试并确认 RED**

```bash
conda run -n vla_env python -m unittest \
  tests.test_grounding_targeting.GroundingTargetingTests.test_held_target_recomputes_direction_from_current_ee \
  tests.test_grounding_targeting.GroundingTargetingTests.test_held_target_interface_cannot_receive_block_truth -v
```

Expected: ERROR，`compute_action_from_world_target` 尚不存在。

- [ ] **Step 3: 实现最小纯计算入口**

在 `grounding_targeting.py` 中加入：

```python
def compute_action_from_world_target(
    target_world,
    ee_pos,
    safety_state,
    settings,
    target_jump_xy=0.0,
):
    """使用已校验的VLM世界目标和当前末端位置重新计算安全动作。"""
    corrected = list(target_world)
    dx = corrected[0] - ee_pos[0]
    dy = corrected[1] - ee_pos[1]
    distance = math.hypot(dx, dy)
    direction = _dominant_direction(dx, dy, settings["stop_distance_xy"])

    previous_distance = safety_state["previous_predicted_distance"]
    no_progress_count = safety_state["no_progress_count"]
    if direction != "stop" and previous_distance is not None:
        improvement = previous_distance - distance
        no_progress_count = (
            0
            if _at_least(improvement, settings["min_progress_xy"])
            else no_progress_count + 1
        )
        if no_progress_count >= settings["no_progress_limit"]:
            raise SmokeSafetyAbort(
                "no_progress", f"improvement={improvement}"
            )

    return {
        "direction": direction,
        "corrected_target_world": corrected,
        "predicted_distance_xy": distance,
        "target_jump_xy": target_jump_xy,
        "safety_state": {
            "previous_target_xy": corrected[:2],
            "previous_predicted_distance": distance,
            "no_progress_count": no_progress_count,
        },
    }
```

把 `compute_grounding_action()` 中方向、距离和无进展计算替换为：

```python
action = compute_action_from_world_target(
    target_world=corrected,
    ee_pos=ee_pos,
    safety_state=safety_state,
    settings=settings,
    target_jump_xy=jump,
)
return {
    **action,
    "box_center_pixel": list(pixel),
    "raw_target_world": raw_world,
}
```

工作区和 `target_jump` 检查继续留在 fresh VLM 入口中；held 路径只能接收此前已经通过这些检查的目标。

- [ ] **Step 4: 运行策略测试并确认 GREEN**

```bash
conda run -n vla_env python -m unittest tests.test_grounding_targeting -v
```

Expected: 所有既有 fresh 行为和新增 held 行为全部 PASS。

- [ ] **Step 5: 提交策略拆分**

```bash
git add grounding_targeting.py tests/test_grounding_targeting.py
git commit -m "feat: compute actions from held VLM targets"
```

---

### Task 3: 实现 fresh/held 目标状态机

**Files:**
- Modify: `tests/test_run_grounding_smoke.py`
- Modify: `run_grounding_smoke.py`

**Interfaces:**
- Consumes: `compute_action_from_world_target(...) -> dict`。
- Modifies: `SmokeDependencies` 增加 `compute_held_action: object`。
- Produces trace fields: `decision_source`、`target_age_steps`、`used_target_hold`、`api_called`。
- Produces summary fields: `num_fresh_vlm_steps`、`num_held_target_steps`、`max_target_age_steps`、`recovered_from_occlusion`。

- [ ] **Step 1: 更新测试依赖构造器**

在测试中导入 `compute_action_from_world_target`，并让 `make_dependencies()` 提供：

```python
compute_held_action=compute_held_action
or Mock(return_value=action("right", 0.06)),
```

所有直接构造 `SmokeDependencies` 的测试也显式传入该字段。

- [ ] **Step 2: 写首帧与历史目标失败测试**

加入以下测试：

```python
def test_visibility_without_cached_target_still_aborts(self):
    ground = Mock()
    held = Mock()
    dependencies = self.make_dependencies(
        observe=Mock(return_value=observation(0.50)),
        ground=ground,
        compute_held_action=held,
    )

    summary, rows = self.run_in_temp(dependencies)

    self.assertEqual(summary["termination_reason"], "visibility_out_of_scope")
    self.assertEqual(rows[-1]["decision_source"], None)
    self.assertEqual(rows[-1]["api_called"], False)
    ground.assert_not_called()
    held.assert_not_called()


def test_low_visibility_reuses_cached_target_without_api_call(self):
    fresh = action("right", 0.08)
    held = action("right", 0.06)
    ground = Mock(return_value=(
        {"red_block": [450, 450, 550, 550]}, "fresh", 0.1
    ))
    compute_held = Mock(return_value=held)
    dependencies = self.make_dependencies(
        observe=Mock(side_effect=[observation(1.0), observation(0.50)]),
        ground=ground,
        compute_action=Mock(return_value=fresh),
        compute_held_action=compute_held,
        score=Mock(side_effect=[scoring(0.10), scoring(0.08), scoring(0.06)]),
    )

    summary, rows = self.run_in_temp(
        dependencies, dict(BASE_CONFIG, max_control_steps=2,
                           max_stale_target_steps=4)
    )

    self.assertEqual(ground.call_count, 1)
    self.assertEqual(rows[0]["decision_source"], "fresh_vlm")
    self.assertEqual(rows[0]["target_age_steps"], 0)
    self.assertEqual(rows[1]["decision_source"], "held_vlm_target")
    self.assertEqual(rows[1]["target_age_steps"], 1)
    self.assertFalse(rows[1]["api_called"])
    self.assertEqual(
        compute_held.call_args.kwargs["target_world"],
        fresh["corrected_target_world"],
    )
```

- [ ] **Step 3: 写4步边界与重新定位重置测试**

加入：

```python
def test_fifth_held_attempt_aborts_without_action(self):
    observations = [observation(1.0)] + [observation(0.50)] * 5
    dependencies = self.make_dependencies(
        observe=Mock(side_effect=observations),
        compute_action=Mock(return_value=action("right", 0.10)),
        compute_held_action=Mock(side_effect=[
            action("right", 0.08), action("right", 0.06),
            action("right", 0.04), action("right", 0.02),
        ]),
        # 初始评分 + 5个已执行动作后的评分 + 超限终止行评分。
        score=Mock(side_effect=[scoring(0.12)] * 7),
    )

    summary, rows = self.run_in_temp(
        dependencies, dict(BASE_CONFIG, max_control_steps=6,
                           max_stale_target_steps=4)
    )

    self.assertEqual(summary["termination_reason"], "stale_target_limit")
    self.assertEqual([row["target_age_steps"] for row in rows[:-1]],
                     [0, 1, 2, 3, 4])
    self.assertIsNone(rows[-1]["decision_source"])
    self.assertEqual(summary["num_held_target_steps"], 4)
    self.assertEqual(summary["num_actions"], 5)


def test_new_fresh_target_resets_target_age(self):
    dependencies = self.make_dependencies(
        observe=Mock(side_effect=[
            observation(1.0), observation(0.50), observation(1.0)
        ]),
        ground=Mock(side_effect=[
            ({"red_block": [450, 450, 550, 550]}, "one", 0.1),
            ({"red_block": [451, 450, 551, 550]}, "two", 0.1),
        ]),
        compute_action=Mock(side_effect=[
            action("right", 0.08), action("stop", 0.01)
        ]),
        compute_held_action=Mock(return_value=action("right", 0.04)),
        score=Mock(side_effect=[
            scoring(0.10), scoring(0.08), scoring(0.04), scoring(0.01)
        ]),
    )

    summary, rows = self.run_in_temp(
        dependencies, dict(BASE_CONFIG, max_stale_target_steps=4)
    )

    self.assertTrue(summary["success"])
    self.assertEqual(
        [(row["decision_source"], row["target_age_steps"]) for row in rows],
        [("fresh_vlm", 0), ("held_vlm_target", 1), ("fresh_vlm", 0)],
    )
```

- [ ] **Step 4: 运行新增测试并确认 RED**

```bash
conda run -n vla_env python -m unittest tests.test_run_grounding_smoke.SmokeLoopTests -v
```

Expected: FAIL/ERROR，依赖字段、状态机和日志字段尚未实现。

- [ ] **Step 5: 实现依赖和目标内存**

`SmokeDependencies` 增加：

```python
compute_held_action: object
```

`run_control_loop()` 初始化：

```python
target_memory = {
    "last_valid_target_world": None,
    "stale_target_steps": 0,
}
```

低可见率分支按以下顺序实现：

```python
cached_target = target_memory["last_valid_target_world"]
if cached_target is None:
    termination = "visibility_out_of_scope"
    # 保存图片并记录 decision_source=None、target_age_steps=None、api_called=False
    break

if target_memory["stale_target_steps"] >= smoke_config["max_stale_target_steps"]:
    termination = "stale_target_limit"
    # 保存图片，不调用 VLM、不执行动作并记录明确终止行
    break

target_memory["stale_target_steps"] += 1
computed_action = dependencies.compute_held_action(
    target_world=cached_target,
    ee_pos=observation["ee_pos"],
    safety_state=safety_state,
    settings=smoke_config,
)
decision_source = "held_vlm_target"
target_age_steps = target_memory["stale_target_steps"]
api_called = False
raw_response = None
boxes = {}
latency = None
```

fresh 计算成功后更新：

```python
target_memory["last_valid_target_world"] = list(
    computed_action["corrected_target_world"]
)
target_memory["stale_target_steps"] = 0
decision_source = "fresh_vlm"
target_age_steps = 0
api_called = True
```

把动作执行和终止判定放到 fresh/held 两条路径汇合之后，避免复制 IK 和评分逻辑。

- [ ] **Step 6: 统一 trace 和 episode summary**

每个产生决策的 payload 加入：

```python
"decision_source": decision_source,
"target_age_steps": target_age_steps,
"used_target_hold": decision_source == "held_vlm_target",
"api_called": api_called,
```

没有产生决策的安全中止行写入：

```python
"decision_source": None,
"target_age_steps": None,
"used_target_hold": False,
"api_called": False,
```

episode summary 从 `rows` 计算：

```python
fresh_steps = sum(row.get("decision_source") == "fresh_vlm" for row in rows)
held_steps = sum(
    row.get("decision_source") == "held_vlm_target" for row in rows
)
target_ages = [
    row["target_age_steps"]
    for row in rows
    if isinstance(row.get("target_age_steps"), int)
]
```

并返回：

```python
"num_fresh_vlm_steps": fresh_steps,
"num_held_target_steps": held_steps,
"max_target_age_steps": max(target_ages, default=0),
"recovered_from_occlusion": termination == "success" and held_steps > 0,
```

- [ ] **Step 7: 接入真实 held 策略并确认 GREEN**

在 `_build_episode_dependencies()` 中设置：

```python
compute_held_action=compute_action_from_world_target,
```

运行：

```bash
conda run -n vla_env python -m unittest \
  tests.test_grounding_targeting tests.test_run_grounding_smoke -v
```

Expected: 策略和闭环测试全部 PASS，held 测试中的 VLM 调用数保持1。

- [ ] **Step 8: 提交状态机**

```bash
git add run_grounding_smoke.py tests/test_run_grounding_smoke.py
git commit -m "feat: hold recent VLM target through occlusion"
```

---

### Task 4: 更新批次通过条件与证据契约

**Files:**
- Modify: `tests/test_run_grounding_smoke.py`
- Modify: `run_grounding_smoke.py`

**Interfaces:**
- Modifies: `aggregate_smoke_summaries(summaries, smoke_config) -> dict`。
- Removes passing dependency on: `all_clear == True`。
- Requires per successful episode: `num_fresh_vlm_steps >= 1` and `max_target_age_steps <= 4`。

- [ ] **Step 1: 写批次判定失败测试**

把原 `test_batch_pass_requires_exact_three_clear_successes` 重命名为
`test_batch_pass_requires_three_successful_recoverable_episodes`，并把合格 episode
fixture 改为：

```python
episodes = [
    {
        "seed": seed,
        "success": True,
        "api_calls": 2,
        "termination_reason": "success",
        "num_fresh_vlm_steps": 2,
        "num_held_target_steps": 3,
        "max_target_age_steps": 3,
        "recovered_from_occlusion": True,
    }
    for seed in (52, 53, 54)
]
```

断言该批次通过，并分别断言以下变体失败：少于3个 episode、任一失败、任一 episode
没有 fresh VLM、任一 `max_target_age_steps=5`、总 API 调用数超过30。

- [ ] **Step 2: 运行测试并确认 RED**

```bash
conda run -n vla_env python -m unittest \
  tests.test_run_grounding_smoke.SmokeLoopTests.test_batch_pass_requires_three_successful_recoverable_episodes -v
```

Expected: FAIL，旧实现仍要求 `all_clear`。

- [ ] **Step 3: 实现新的通过条件**

`aggregate_smoke_summaries()` 使用：

```python
passed = (
    [row["seed"] for row in rows] == smoke_config["seeds"]
    and successes == smoke_config["required_successes"] == 3
    and all(row["num_fresh_vlm_steps"] >= 1 for row in rows)
    and all(
        row["max_target_age_steps"]
        <= smoke_config["max_stale_target_steps"]
        for row in rows
    )
    and calls <= smoke_config["max_total_api_calls"]
)
```

批次 summary 增加：

```python
"fresh_vlm_steps": sum(row["num_fresh_vlm_steps"] for row in rows),
"held_target_steps": sum(row["num_held_target_steps"] for row in rows),
"recovered_episode_count": sum(
    row["recovered_from_occlusion"] for row in rows
),
```

- [ ] **Step 4: 运行 smoke 测试并确认 GREEN**

```bash
conda run -n vla_env python -m unittest tests.test_run_grounding_smoke -v
```

Expected: 全部 PASS；批次允许受控 held 步骤，但不允许无 fresh 定位或超过年龄上限。

- [ ] **Step 5: 提交批次契约**

```bash
git add run_grounding_smoke.py tests/test_run_grounding_smoke.py
git commit -m "test: define target hold smoke acceptance"
```

---

### Task 5: 全量验证、真实 smoke 与文档收尾

**Files:**
- Generate: `vlm_smoke_runs/run_<timestamp>/`
- Modify after evidence: `README.md`
- Modify after evidence: `docs/worklog/WORKLOG.md`
- Modify after evidence: `docs/planning/vla_robotic_study_plan.md`
- Modify after evidence: `docs/debugging/BUGLOG.md`

**Interfaces:**
- Consumes: 已冻结配置、冻结校准文件和现有 Qwen API 环境变量。
- Produces: 三个 episode 的图片、`smoke_trace.jsonl`、`episode_summary.jsonl` 和 `smoke_summary.json`。

- [ ] **Step 1: 运行完整自动测试**

```bash
conda run -n vla_env python -m unittest discover -s tests -v
git diff --check
```

Expected: 全部测试 PASS，`git diff --check` 无输出。

- [ ] **Step 2: 检查 API 与冻结校准，不发请求**

```bash
conda run -n vla_env python -c "import json,os,yaml; from pathlib import Path; c=yaml.safe_load(Path('sim_config.yaml').read_text()); s=c['grounding_smoke']; a=c['probe']['api']; assert all(os.getenv(a[n]) for n in ('base_url_env','api_key_env','model_env')); d=json.loads(Path(s['calibration_path']).read_text()); assert d['num_clear_calibration_samples']==s['expected_calibration_samples']; assert d['correction_x']==s['expected_correction_x']; assert d['correction_y']==s['expected_correction_y']; assert s['max_stale_target_steps']==4; print('ready cases=',s['seeds'])"
```

Expected: 输出 `ready cases=` 和三个冻结 seed；此步骤 API 调用为0。

- [ ] **Step 3: 执行三个真实在线 smoke cases**

```bash
conda run -n vla_env python run_grounding_smoke.py
```

Expected: 创建一个全新的 `vlm_smoke_runs/run_<timestamp>/`，无论通过或失败都保留完整中止前证据。

- [ ] **Step 4: 审计真值隔离和状态机证据**

```bash
conda run -n vla_env python -c "import inspect,json,yaml; from pathlib import Path; from grounding_targeting import compute_action_from_world_target,compute_grounding_action; c=yaml.safe_load(Path('sim_config.yaml').read_text()); root=sorted(Path('vlm_smoke_runs').glob('run_*'))[-1]; s=json.loads((root/'smoke_summary.json').read_text()); traces=[json.loads(x) for p in sorted(root.glob('episode_*/smoke_trace.jsonl')) for x in p.read_text().splitlines() if x.strip()]; assert s['num_episodes']==3; assert s['total_api_calls']<=30; assert all(t.get('decision_source') in (None,'fresh_vlm','held_vlm_target') for t in traces); assert all(not t.get('api_called') for t in traces if t.get('decision_source')=='held_vlm_target'); assert all(n not in inspect.signature(compute_grounding_action).parameters for n in ('block_pos','true_block_pos')); assert all(n not in inspect.signature(compute_action_from_world_target).parameters for n in ('block_pos','true_block_pos')); print(json.dumps(s,ensure_ascii=False,indent=2))"
```

Expected: 结构断言全部通过，并打印真实 success、fresh/held 步数、终止原因和 API 调用数；不得要求 `passed=true` 才保存结果。

- [ ] **Step 5: 用真实结果同步四份项目文档**

四份文档必须同时写入：

- run 目录；
- 三个 seed/方向；
- 每个 episode 的控制步数、fresh/held 步数、最大目标年龄、终止原因和最终真实 XY 距离；
- 总 API 调用数、成功数和 `passed`；
- 若失败，只选择首个主导失败类型作为下一步，不降低阈值、不增加多个变量；
- 若通过，结论限定为“静态目标短时遮挡恢复通过3次 smoke”，不能外推到永久遮挡、目标移动或真实机器人。

- [ ] **Step 6: 最终验证并提交文档**

```bash
conda run -n vla_env python -m unittest discover -s tests -v
rg -n "target|目标|held|遮挡|smoke|下一步" \
  README.md docs/worklog/WORKLOG.md \
  docs/planning/vla_robotic_study_plan.md docs/debugging/BUGLOG.md
git diff --check
git status --short
git add README.md docs/worklog/WORKLOG.md \
  docs/planning/vla_robotic_study_plan.md docs/debugging/BUGLOG.md
git commit -m "docs: analyze grounding target hold smoke"
```

Expected: 完整测试仍全部 PASS，四份文档的数字、结论和唯一下一步一致。

---

## 完成标准

- `max_stale_target_steps=4` 有配置测试保护。
- fresh 与 held 路径共用同一世界坐标动作计算，不重复旧方向。
- 首帧不可见立即拒绝；第1至第4次 held 允许；第5次尝试安全停止。
- trace 和 summary 可以明确区分 fresh/held、目标年龄与 API 是否调用。
- 动作接口不能接收 PyBullet 红块真值。
- 完整自动测试通过，真实三个 smoke cases 有可审计证据。
- 无论真实结果成功或失败，均分析结果并只给出一个证据驱动的下一步。
