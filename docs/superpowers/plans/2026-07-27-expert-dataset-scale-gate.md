# 专家数据规模化验收门禁实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `expert_v1` 增加300条规模化质量门禁，安全追加290条真实专家轨迹，并以机器可判定报告决定能否进入 action tokenization。

**Architecture:** 复用现有只读扫描结果，在 `evaluate_dataset.py` 中保留原 pilot gate，并新增独立 scale gate；报告根据 episode 数量选择当前门禁并暴露统一顶层 `passed`。采集器逻辑保持不变，只把配置从10条清理模式切换为290条追加模式，最终由同一 CLI 扫描全部数据。

**Tech Stack:** Python 3、PyBullet、OpenCV、PyYAML、`unittest`、JSON/JSONL、YAML

## Global Constraints

- 不改变 `schema_version: "expert_v1"`、9维动作、`random_seed: 1000` 或 `episode_seed = random_seed + episode_idx`。
- 不改变机械臂控制、grounding、相机补偿或 smoke stop 行为。
- 已验收的10条 pilot 不得删除、覆盖或重编号。
- 追加采集必须使用 `clean_before_run: false` 和 `num_episodes: 290`。
- scale gate 要求有效 episode 至少300、成功率至少99%、所有完整性错误为0、X/Y五个分箱均非空。
- 生成数据只写入 `outputs/dataset/expert_scaling_v1/`，不提交 Git。

---

### Task 1: 增加规模化质量门禁

**Files:**
- Modify: `src/vla_project/simulation/evaluate_dataset.py`
- Test: `tests/simulation/test_evaluate_dataset.py`

**Interfaces:**
- Consumes: `evaluate_dataset(dataset_dir)` 已生成的统计字段和 manifest 的 `pilot_num_episodes`、`target_num_episodes`。
- Produces: `evaluate_scale_gate(report: dict, manifest: dict) -> dict`，以及报告字段 `scale_gate`、`active_gate`、`passed`。

- [ ] **Step 1: 在测试中导入新门禁并补齐 manifest 契约**

把导入更新为：

```python
from vla_project.simulation.evaluate_dataset import (
    evaluate_dataset,
    evaluate_pilot_gate,
    evaluate_scale_gate,
    write_quality_report,
)
```

在 `manifest()` 返回值中增加：

```python
"target_num_episodes": 300,
```

- [ ] **Step 2: 写合格规模化报告的失败测试**

在 `EvaluateDatasetTests` 增加：

```python
def valid_scale_report(self):
    report = {
        "valid_episode_count": 300,
        "success_rate": 0.99,
        "block_position": {
            "x_bin_counts": [60, 60, 60, 60, 60],
            "y_bin_counts": [60, 60, 60, 60, 60],
        },
    }
    for field in (
        "schema_error_count",
        "action_dim_error_count",
        "missing_image_count",
        "unreadable_image_count",
        "image_size_mismatch_count",
        "orphan_image_count",
        "duplicate_step_key_count",
        "seed_error_count",
        "frame_count_mismatch_count",
        "terminal_flag_error_count",
    ):
        report[field] = 0
    return report

def test_scale_gate_accepts_clean_covered_target_dataset(self):
    gate = evaluate_scale_gate(
        self.valid_scale_report(),
        {"target_num_episodes": 300},
    )

    self.assertTrue(gate["passed"])
    self.assertEqual(gate["failed_checks"], [])
```

- [ ] **Step 3: 运行测试确认 RED**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest \
  tests.simulation.test_evaluate_dataset.EvaluateDatasetTests.test_scale_gate_accepts_clean_covered_target_dataset
```

Expected: 导入失败，因为 `evaluate_scale_gate` 尚不存在。

- [ ] **Step 4: 实现最小 scale gate**

在 `evaluate_dataset.py` 的门禁函数附近增加统一错误字段常量和实现：

```python
QUALITY_ERROR_COUNT_FIELDS = (
    "schema_error_count",
    "action_dim_error_count",
    "missing_image_count",
    "unreadable_image_count",
    "image_size_mismatch_count",
    "orphan_image_count",
    "duplicate_step_key_count",
    "seed_error_count",
    "frame_count_mismatch_count",
    "terminal_flag_error_count",
)


def _has_full_axis_coverage(counts):
    return (
        isinstance(counts, list)
        and len(counts) == 5
        and all(isinstance(count, int) and count > 0 for count in counts)
    )


def evaluate_scale_gate(report, manifest):
    checks = {
        "valid_episode_count": (
            report["valid_episode_count"] >= manifest["target_num_episodes"]
        ),
        "success_rate": report["success_rate"] >= 0.99,
        "x_bin_coverage": _has_full_axis_coverage(
            report["block_position"]["x_bin_counts"]
        ),
        "y_bin_coverage": _has_full_axis_coverage(
            report["block_position"]["y_bin_counts"]
        ),
    }
    checks.update(
        {
            field.removesuffix("_count"): report[field] == 0
            for field in QUALITY_ERROR_COUNT_FIELDS
        }
    )
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
    }
```

- [ ] **Step 5: 运行合格路径测试确认 GREEN**

Run: Step 3 的相同命令。

Expected: 1 test passes。

- [ ] **Step 6: 写规模门禁各类失败条件测试**

增加：

```python
def test_scale_gate_rejects_size_rate_integrity_and_coverage_failures(self):
    corruptions = {
        "valid_episode_count": 299,
        "success_rate": 0.989,
        "schema_error_count": 1,
        "x_bin_counts": [0, 75, 75, 75, 75],
        "y_bin_counts": [75, 75, 75, 75, 0],
    }
    for field, value in corruptions.items():
        with self.subTest(field=field):
            report = self.valid_scale_report()
            if field in ("x_bin_counts", "y_bin_counts"):
                report["block_position"][field] = value
            else:
                report[field] = value

            gate = evaluate_scale_gate(
                report,
                {"target_num_episodes": 300},
            )

            self.assertFalse(gate["passed"])
            self.assertTrue(gate["failed_checks"])
```

- [ ] **Step 7: 运行失败条件测试确认 GREEN**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest tests.simulation.test_evaluate_dataset
```

Expected: module tests pass。

- [ ] **Step 8: 写阶段选择和统一结果的失败测试**

扩展最小数据集测试：

```python
self.assertEqual(report["active_gate"], "pilot")
self.assertTrue(report["passed"])
self.assertFalse(report["scale_gate"]["passed"])
```

再为纯函数提取的选择逻辑增加：

```python
def test_selects_scale_gate_only_at_target_size(self):
    pilot_gate = {"passed": False}
    scale_gate = {"passed": True}

    active_gate, passed = select_active_gate(
        {"num_episodes": 300},
        {"target_num_episodes": 300},
        pilot_gate,
        scale_gate,
    )

    self.assertEqual(active_gate, "scale")
    self.assertTrue(passed)

def test_intermediate_dataset_does_not_pass_pilot_gate(self):
    pilot_gate = {"passed": False}
    scale_gate = {"passed": False}

    active_gate, passed = select_active_gate(
        {"num_episodes": 11},
        {"target_num_episodes": 300},
        pilot_gate,
        scale_gate,
    )

    self.assertEqual(active_gate, "pilot")
    self.assertFalse(passed)
```

同时从模块导入 `select_active_gate`。

- [ ] **Step 9: 运行阶段选择测试确认 RED**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest tests.simulation.test_evaluate_dataset
```

Expected: 因 `select_active_gate` 不存在或报告缺少新字段而失败。

- [ ] **Step 10: 实现阶段选择并接入报告和 CLI**

增加：

```python
def select_active_gate(report, manifest, pilot_gate, scale_gate):
    if report["num_episodes"] >= manifest["target_num_episodes"]:
        return "scale", scale_gate["passed"]
    return "pilot", pilot_gate["passed"]
```

在 `evaluate_dataset()` 构建基础 `report` 后：

```python
report["pilot_gate"] = evaluate_pilot_gate(report, manifest)
report["scale_gate"] = evaluate_scale_gate(report, manifest)
report["active_gate"], report["passed"] = select_active_gate(
    report,
    manifest,
    report["pilot_gate"],
    report["scale_gate"],
)
```

把 `main()` 的输出增加 `gate={report['active_gate']}`，并将退出条件改为：

```python
if not report["passed"]:
    sys.exit(1)
```

- [ ] **Step 11: 运行定向测试确认 GREEN**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest tests.simulation.test_evaluate_dataset
```

Expected: module tests pass。

- [ ] **Step 12: 提交门禁实现**

```bash
git add \
  src/vla_project/simulation/evaluate_dataset.py \
  tests/simulation/test_evaluate_dataset.py
git commit -m "feat: gate scaled expert datasets"
```

---

### Task 2: 切换安全追加配置并更新使用说明

**Files:**
- Modify: `sim_config.yaml`
- Modify: `tests/test_config_contract.py`
- Modify: `README.md`
- Modify: `docs/agent/PROJECT_OVERVIEW.md`
- Modify: `docs/agent/CURRENT_STATUS.md`

**Interfaces:**
- Consumes: Task 1 的顶层 `passed`、`active_gate` 和 `scale_gate` 报告契约。
- Produces: 追加采集配置 `clean_before_run: false`、`num_episodes: 290`，以及与新门禁一致的运行说明。

- [ ] **Step 1: 修改配置契约测试**

把 `test_expert_dataset_scaling_config_is_frozen` 中 episode 断言更新为：

```python
self.assertFalse(dataset["clean_before_run"])
self.assertEqual(dataset["num_episodes"], 290)
self.assertEqual(dataset["pilot_num_episodes"], 10)
self.assertEqual(dataset["target_num_episodes"], 300)
```

- [ ] **Step 2: 运行配置测试确认 RED**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest tests.test_config_contract.ConfigContractTests.test_expert_dataset_scaling_config_is_frozen
```

Expected: 当前配置仍为 `clean_before_run: true`、`num_episodes: 10`，测试失败。

- [ ] **Step 3: 修改追加采集配置**

在 `sim_config.yaml` 中设置：

```yaml
clean_before_run: false
num_episodes: 290
```

注释明确这是在已验收10条 pilot 后追加到总计300条。

- [ ] **Step 4: 运行配置测试确认 GREEN**

Run: Step 2 的相同命令。

Expected: 1 test passes。

- [ ] **Step 5: 更新项目说明**

更新 README 的评估说明：

- pilot 阶段要求恰好10条全部成功且完整性错误为0；
- 达到300条后自动执行 scale gate；
- scale gate 要求至少300条有效 episode、成功率不低于99%、完整性错误为0、X/Y五箱非空；
- 顶层 `passed` 是能否进入 action tokenization 的机器判断字段。

更新 `PROJECT_OVERVIEW.md` 和 `CURRENT_STATUS.md`：

- 当前配置已经切到290条追加模式；
- 当前下一动作是运行真实采集；
- 在真实结果出现前不提前声明300条验收通过。

- [ ] **Step 6: 运行配置和数据评估测试**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest \
  tests.test_config_contract \
  tests.simulation.test_evaluate_dataset
```

Expected: selected modules pass。

- [ ] **Step 7: 提交追加配置与说明**

```bash
git add \
  sim_config.yaml \
  tests/test_config_contract.py \
  README.md \
  docs/agent/PROJECT_OVERVIEW.md \
  docs/agent/CURRENT_STATUS.md
git commit -m "chore: prepare expert dataset append run"
```

---

### Task 3: 采集290条真实轨迹并执行规模化验收

**Files:**
- Generated: `outputs/dataset/expert_scaling_v1/`
- Modify after evidence: `docs/agent/PROJECT_OVERVIEW.md`
- Modify after evidence: `docs/agent/CURRENT_STATUS.md`
- Modify after evidence: `docs/worklog/WORKLOG.md`

**Interfaces:**
- Consumes: Task 2 的追加配置和 Task 1 的统一门禁。
- Produces: 至少300条 `expert_v1` episode、完整质量报告和是否允许进入 action tokenization 的阶段结论。

- [ ] **Step 1: 全量代码验证**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest discover -s tests -v
conda run -n vla_env env PYTHONPATH=src \
  python -m compileall -q src tests
git diff --check
```

Expected: 所有测试通过，`compileall` 和 `git diff --check` 退出码为0。

- [ ] **Step 2: 只读核对 pilot 恢复点**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path

dataset_dir = Path("outputs/dataset/expert_scaling_v1")
summaries = [
    json.loads(line)
    for line in (dataset_dir / "episode_summary.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]
report = json.loads(
    (dataset_dir / "dataset_quality_report.json").read_text(encoding="utf-8")
)
assert len(summaries) == 10, len(summaries)
assert max(row["episode_idx"] for row in summaries) == 9
assert report["pilot_gate"]["passed"] is True
print("pilot episodes=10, max_episode_idx=9, pilot_gate=true")
PY
```

Expected: 输出恢复点信息且退出码为0。

- [ ] **Step 3: 运行真实追加采集**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src vla-collect
```

Expected: 不清理现有目录，从 episode 10 继续采集290条，不调用 VLM API。

- [ ] **Step 4: 运行规模化质量检查**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src vla-evaluate-dataset
```

Expected: `gate=scale`，命令退出码为0，报告顶层 `passed=true`。

- [ ] **Step 5: 核对最终门禁证据**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path

report_path = Path(
    "outputs/dataset/expert_scaling_v1/dataset_quality_report.json"
)
report = json.loads(report_path.read_text(encoding="utf-8"))
assert report["active_gate"] == "scale"
assert report["passed"] is True
assert report["valid_episode_count"] >= 300
assert report["success_rate"] >= 0.99
assert report["scale_gate"]["passed"] is True
print(
    json.dumps(
        {
            "num_episodes": report["num_episodes"],
            "valid_episode_count": report["valid_episode_count"],
            "num_frames": report["num_frames"],
            "success_rate": report["success_rate"],
            "x_bin_counts": report["block_position"]["x_bin_counts"],
            "y_bin_counts": report["block_position"]["y_bin_counts"],
            "failed_checks": report["scale_gate"]["failed_checks"],
        },
        ensure_ascii=False,
        indent=2,
    )
)
PY
```

Expected: 所有断言通过并打印规模化指标。

- [ ] **Step 6: 根据真实证据更新阶段文档**

把实际 episode 数、有效数量、帧数、成功率、失败原因、X/Y 分箱和门禁结果写入
`WORKLOG.md`。同步改写 `PROJECT_OVERVIEW.md` 与 `CURRENT_STATUS.md`：

- 若顶层 `passed=true`，当前阶段切换到 action tokenization；
- 若失败，保留原始输出和失败检查，当前阶段保持专家数据质量诊断，不修改 schema 或
  直接重采全部数据。

- [ ] **Step 7: 最终验证**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest discover -s tests -v
conda run -n vla_env env PYTHONPATH=src \
  python -m compileall -q src tests
git diff --check
git status --short
```

Expected: 测试、编译和 diff 检查通过；Git 状态只包含预期文档变更和用户原有未跟踪文件。

- [ ] **Step 8: 提交实验结论**

```bash
git add \
  docs/agent/PROJECT_OVERVIEW.md \
  docs/agent/CURRENT_STATUS.md \
  docs/worklog/WORKLOG.md
git commit -m "docs: record scaled expert dataset results"
```

`outputs/dataset/expert_scaling_v1/` 继续作为本地实验证据，不提交 Git。
