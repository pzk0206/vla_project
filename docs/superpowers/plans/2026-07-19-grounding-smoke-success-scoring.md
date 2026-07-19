# Grounding Smoke Success Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将最终任务到达、自主 stop 和允许参与批次通过的成功拆成独立指标，使到达目标的 `stale_target_limit` episode 可以参与主要通过，同时不掩盖系统错误。

**Architecture:** `run_control_loop()` 在循环完全结束后从最终真值距离派生三个只读评分字段，不把评分回流到动作或终止决策。`aggregate_smoke_summaries()` 分别聚合主要成功、任务到达和自主停止，并继续用主要 `success` 保护原有3/3、fresh、目标年龄和 API 上限契约。

**Tech Stack:** Python 3.10、`unittest`、`unittest.mock`、JSON

## Global Constraints

- `task_success = final_true_distance_xy <= 0.03`。
- `autonomous_stop_success = termination_reason == "success"`。
- 主要 `success = task_success and termination_reason in {"success", "stale_target_limit"}`。
- API、IK、反投影、invalid box、false stop、visibility、max steps 等终止即使距离达标也不得主要通过。
- 真值只在事后 summary 评分中使用，不得改变控制循环、动作、stop、held 或 VLM 调用。
- 保留原始 `termination_reason` 和 `recovered_from_occlusion` 语义。
- 不修改整批初始预检、固定案例或 API 上限；不运行真实 API。

---

### Task 1: Episode 三层成功评分

**Files:**
- Modify: `src/vla_project/vlm/grounding_smoke/runner.py:380-425`
- Test: `tests/vlm/grounding_smoke/test_runner.py:230-495`

**Interfaces:**
- Consumes: 循环结束后的 `termination: str`、最终 trace/初始评分距离
- Produces: episode summary 字段 `task_success: bool`、`autonomous_stop_success: bool`、`success: bool`

- [ ] **Step 1: 写正常 stop 与 stale 到达的失败测试**

扩展 `test_two_frame_success_regrounds_and_executes_once`：

```python
self.assertTrue(summary["task_success"])
self.assertTrue(summary["autonomous_stop_success"])
self.assertTrue(summary["success"])
```

新增等价于真实 `53-right` 的测试：

```python
def test_stale_target_limit_counts_as_task_success_after_reaching_target(self):
    dependencies = self.make_dependencies(
        observe=Mock(
            side_effect=[observation(1.0)] + [observation(0.50)] * 5
        ),
        compute_action=Mock(return_value=action("right", 0.10)),
        compute_held_action=Mock(
            side_effect=[
                action("right", 0.08),
                action("right", 0.06),
                action("right", 0.04),
                action("right", 0.02),
            ]
        ),
        score=Mock(
            side_effect=[
                scoring(0.12),
                scoring(0.10),
                scoring(0.08),
                scoring(0.06),
                scoring(0.04),
                scoring(0.004),
                scoring(0.004),
            ]
        ),
    )

    summary, _ = self.run_in_temp(
        dependencies,
        dict(BASE_CONFIG, max_control_steps=6),
    )

    self.assertEqual(summary["termination_reason"], "stale_target_limit")
    self.assertEqual(summary["final_true_distance_xy"], 0.004)
    self.assertTrue(summary["task_success"])
    self.assertFalse(summary["autonomous_stop_success"])
    self.assertTrue(summary["success"])
```

在已有 `test_fifth_held_attempt_aborts_without_action` 增加三个 false 断言，保护 stale 未
到达。再新增错误与步数上限测试：

```python
def test_system_errors_cannot_pass_when_final_distance_is_within_3cm(self):
    cases = {
        "api_error": self.make_dependencies(
            ground=Mock(side_effect=TimeoutError("timeout")),
            score=Mock(side_effect=[scoring(0.10), scoring(0.01)]),
        ),
        "backprojection_error": self.make_dependencies(
            compute_action=Mock(side_effect=ValueError("bad matrix")),
            score=Mock(side_effect=[scoring(0.10), scoring(0.01)]),
        ),
        "ik_error": self.make_dependencies(
            compute_action=Mock(return_value=action("right", 0.08)),
            execute=Mock(side_effect=RuntimeError("ik failed")),
            score=Mock(side_effect=[scoring(0.10), scoring(0.01)]),
        ),
    }
    for reason, dependencies in cases.items():
        with self.subTest(reason=reason):
            summary, _ = self.run_in_temp(dependencies)
            self.assertEqual(summary["termination_reason"], reason)
            self.assertTrue(summary["task_success"])
            self.assertFalse(summary["autonomous_stop_success"])
            self.assertFalse(summary["success"])

def test_max_steps_cannot_pass_when_final_distance_is_within_3cm(self):
    dependencies = self.make_dependencies(
        compute_action=Mock(return_value=action("right", 0.08)),
        score=Mock(side_effect=[scoring(0.10), scoring(0.01)]),
    )
    summary, _ = self.run_in_temp(
        dependencies,
        dict(BASE_CONFIG, max_control_steps=1),
    )
    self.assertEqual(summary["termination_reason"], "max_control_steps")
    self.assertTrue(summary["task_success"])
    self.assertFalse(summary["autonomous_stop_success"])
    self.assertFalse(summary["success"])
```

- [ ] **Step 2: 运行全部新评分测试并确认 RED**

```bash
conda run -n vla_env env PYTHONPATH=src python -m unittest \
  tests.vlm.grounding_smoke.test_runner.SmokeLoopTests.test_two_frame_success_regrounds_and_executes_once \
  tests.vlm.grounding_smoke.test_runner.SmokeLoopTests.test_stale_target_limit_counts_as_task_success_after_reaching_target \
  tests.vlm.grounding_smoke.test_runner.SmokeLoopTests.test_system_errors_cannot_pass_when_final_distance_is_within_3cm \
  tests.vlm.grounding_smoke.test_runner.SmokeLoopTests.test_max_steps_cannot_pass_when_final_distance_is_within_3cm \
  tests.vlm.grounding_smoke.test_runner.SmokeLoopTests.test_fifth_held_attempt_aborts_without_action -v
```

Expected: 新字段 KeyError，且 stale episode 旧 `success` 为 false。

- [ ] **Step 3: 最小实现三个评分字段**

在 `run_control_loop()` return 前计算：

```python
final_true_distance_xy = (
    rows[-1]["true_distance_xy"]
    if rows
    else initial_scoring["true_distance_xy"]
)
task_success = final_true_distance_xy <= 0.03
autonomous_stop_success = termination == "success"
success = task_success and termination in {
    "success",
    "stale_target_limit",
}
```

将 return 中旧字段替换为：

```python
"success": success,
"task_success": task_success,
"autonomous_stop_success": autonomous_stop_success,
...
"final_true_distance_xy": final_true_distance_xy,
```

不要移动此计算到循环内部，也不要用它 break。

- [ ] **Step 4: 运行全部新评分测试并确认 GREEN**

Run: 与 Step 2 相同。

Expected: `Ran 5 tests`、`OK`。

- [ ] **Step 5: 运行完整 runner 测试**

```bash
conda run -n vla_env env PYTHONPATH=src python -m unittest tests.vlm.grounding_smoke.test_runner -v
```

Expected: 全部通过，无真实 API 调用。

- [ ] **Step 6: 提交 episode 评分**

```bash
git add src/vla_project/vlm/grounding_smoke/runner.py tests/vlm/grounding_smoke/test_runner.py
git commit -m "feat: split grounding smoke episode success"
```

---

### Task 2: Batch 聚合、兼容测试与文档

**Files:**
- Modify: `src/vla_project/vlm/grounding_smoke/runner.py:426-465`
- Test: `tests/vlm/grounding_smoke/test_runner.py:494-530,790-880`
- Modify: `docs/agent/PROJECT_OVERVIEW.md`
- Modify: `docs/agent/CURRENT_STATUS.md`
- Modify: `docs/debugging/BUGLOG.md`
- Modify: `docs/worklog/WORKLOG.md`

**Interfaces:**
- Consumes: episode `success`、`task_success`、`autonomous_stop_success`
- Produces: 三组 batch count/rate；主要 `passed` 继续使用 `success`

- [ ] **Step 1: 写混合 episode 聚合失败测试**

新增：

```python
def test_batch_reports_task_and_autonomous_stop_metrics_separately(self):
    episodes = [
        {
            "seed": 52,
            "success": True,
            "task_success": True,
            "autonomous_stop_success": True,
            "api_calls": 1,
            "termination_reason": "success",
            "num_fresh_vlm_steps": 1,
            "num_held_target_steps": 0,
            "max_target_age_steps": 0,
            "recovered_from_occlusion": False,
        },
        {
            "seed": 53,
            "success": True,
            "task_success": True,
            "autonomous_stop_success": False,
            "api_calls": 1,
            "termination_reason": "stale_target_limit",
            "num_fresh_vlm_steps": 1,
            "num_held_target_steps": 4,
            "max_target_age_steps": 4,
            "recovered_from_occlusion": False,
        },
        {
            "seed": 54,
            "success": False,
            "task_success": True,
            "autonomous_stop_success": False,
            "api_calls": 1,
            "termination_reason": "api_error",
            "num_fresh_vlm_steps": 1,
            "num_held_target_steps": 0,
            "max_target_age_steps": 0,
            "recovered_from_occlusion": False,
        },
    ]
    summary = aggregate_smoke_summaries(episodes, BASE_CONFIG)
    self.assertEqual(summary["success_count"], 2)
    self.assertAlmostEqual(summary["success_rate"], 2 / 3)
    self.assertEqual(summary["task_success_count"], 3)
    self.assertEqual(summary["task_failure_count"], 0)
    self.assertEqual(summary["task_success_rate"], 1.0)
    self.assertEqual(summary["autonomous_stop_success_count"], 1)
    self.assertAlmostEqual(
        summary["autonomous_stop_success_rate"],
        1 / 3,
    )
    self.assertFalse(summary["passed"])
```

更新现有三个成功 episode fixture，为每行增加：

```python
"task_success": True,
"autonomous_stop_success": True,
```

更新 batch contract 的 `fake_loop()` success summary，增加相同两个字段。

- [ ] **Step 2: 运行聚合测试并确认 RED**

```bash
conda run -n vla_env env PYTHONPATH=src python -m unittest \
  tests.vlm.grounding_smoke.test_runner.SmokeLoopTests.test_batch_reports_task_and_autonomous_stop_metrics_separately -v
```

Expected: 新计数字段 KeyError。

- [ ] **Step 3: 实现 batch 三组统计**

在 `aggregate_smoke_summaries()` 中增加：

```python
task_successes = sum(row["task_success"] for row in rows)
autonomous_stop_successes = sum(
    row["autonomous_stop_success"] for row in rows
)
```

return 增加：

```python
"task_success_count": task_successes,
"task_failure_count": len(rows) - task_successes,
"task_success_rate": task_successes / len(rows) if rows else 0.0,
"autonomous_stop_success_count": autonomous_stop_successes,
"autonomous_stop_success_rate": (
    autonomous_stop_successes / len(rows) if rows else 0.0
),
```

不要改变 `passed` 使用 `successes` 的逻辑。

- [ ] **Step 4: 运行 runner 与定向契约测试**

```bash
conda run -n vla_env env PYTHONPATH=src python -m unittest tests.vlm.grounding_smoke.test_runner -v
conda run -n vla_env env PYTHONPATH=src python -m unittest \
  tests.test_config_contract \
  tests.test_package_metadata \
  tests.vlm.grounding_smoke.test_targeting \
  tests.vlm.grounding_smoke.test_runner \
  tests.vlm.grounding_smoke.test_screening -v
```

Expected: 全部通过，无真实 API 调用。

- [ ] **Step 5: 更新权威文档**

在四份权威文档记录：新 `success` 从本版本起表示“任务到达且终止原因允许”，旧批次
`success_rate` 不可与新批次直接合并；`task_success` 只陈述最终物理位置，
`autonomous_stop_success` 单独陈述自主 stop。明确未选择新固定案例、未运行真实 API。

- [ ] **Step 6: 运行完整验证**

```bash
conda run -n vla_env env PYTHONPATH=src python -m unittest discover -s tests -v
conda run -n vla_env env PYTHONPATH=src python -m compileall -q src tests
conda run -n vla_env env PYTHONPATH=src python -c "import inspect; from vla_project.vlm.grounding_smoke.targeting import compute_grounding_action, compute_action_from_world_target; forbidden={'block_pos','true_block_pos'}; assert forbidden.isdisjoint(inspect.signature(compute_grounding_action).parameters); assert forbidden.isdisjoint(inspect.signature(compute_action_from_world_target).parameters); print('truth isolation passed')"
git diff --check
git status --short
```

Expected: 全量测试、编译、真值隔离和补丁检查全部通过；状态只包含本任务文件和进入任务前已知的用户未跟踪文件。

- [ ] **Step 7: 提交聚合与文档**

```bash
git add src/vla_project/vlm/grounding_smoke/runner.py tests/vlm/grounding_smoke/test_runner.py docs/agent/PROJECT_OVERVIEW.md docs/agent/CURRENT_STATUS.md docs/debugging/BUGLOG.md docs/worklog/WORKLOG.md
git commit -m "feat: report grounding smoke success metrics"
```

提交后重新检查分支状态，准备收尾审查和集成选择。
