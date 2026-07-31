# Action Tokenization Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个只读、可复现的 expert_v1 action tokenization 审计命令，联合动作与可见性标签，模拟离散候选并推荐下一轮最小训练对照组。

**Architecture:** 在 `simulation` 领域新增单模块审计器：先严格加载并对齐冻结轨迹与可见性 JSONL，再构造逐帧派生值，计算统计和12个分箱候选，最后原子发布逐帧 JSONL 与 summary。数据完整性 scale gate 保持不变；新命令、报告和文档独立演进。

**Tech Stack:** Python 3.10+、标准库、NumPy、`unittest`/pytest、JSON/JSONL、setuptools console scripts。

## Global Constraints

- 原始 `expert_v1` 数据、manifest、轨迹、图片、可见性报告和质量报告保持只读。
- 正式源码只放 `src/vla_project/simulation/`，镜像测试只放 `tests/simulation/`。
- 不生成 bin id、训练 token 或 tokenized dataset，不划分训练/验证集。
- `delta_q` 只称为相邻保存帧 IK 目标差，不称为实际关节位移。
- 候选固定为 `absolute_q|delta_q × uniform_width|quantile × 16|32|64`。
- 推荐门槛固定为 occupancy `>=0.90`、minimum nonempty count `>=20`、normalized p95 error `<=0.05`。
- severe 组只做描述统计并标记样本不足，不能驱动数据删除。
- 成功结果原子发布；失败不得替换上一次完整成功输出。
- 不触碰工作区内用户已有的未跟踪文件。

---

### Task 1: 输入硬门禁与逐帧动作派生

**Files:**
- Create: `src/vla_project/simulation/audit_action_tokenization.py`
- Create: `tests/simulation/test_audit_action_tokenization.py`

**Interfaces:**
- Consumes: `dataset_manifest.json`、`trajectory_expert.jsonl`、`visibility_audit_v1/frame_visibility.jsonl`。
- Produces: `AuditValidationError(reason: str, **evidence)`、`read_jsonl(path: Path) -> list[dict]`、`load_audit_inputs(dataset_dir: Path) -> tuple[dict, list[dict], list[dict], dict]`、`build_frame_analysis(manifest: dict, trajectory_rows: list[dict], visibility_rows: list[dict]) -> list[dict]`。

- [ ] **Step 1: 写出正常对齐与差分的失败测试**

在新测试文件中使用 `unittest.TestCase`，构造两个 episode；明确验证首帧 `null`、episode 边界不跨越、非固定 `step_gap` 和可见性合并：

```python
def action_row(episode_idx, step_idx, q, terminate=0):
    return {
        "schema_version": "expert_v1",
        "episode_idx": episode_idx,
        "step_idx": step_idx,
        "action": list(q) + [1.0, terminate],
    }

def visibility_row(episode_idx, step_idx, group="clear"):
    return {
        "episode_idx": episode_idx,
        "step_idx": step_idx,
        "visibility_group": group,
    }

def test_builds_frame_analysis_without_cross_episode_delta(self):
    manifest = {"schema_version": "expert_v1", "action_dim": 9}
    trajectories = [
        action_row(0, 0, [0.0] * 7),
        action_row(0, 24, [0.24] * 7, terminate=1),
        action_row(1, 0, [1.0] * 7),
        action_row(1, 12, [0.88] * 7, terminate=1),
    ]
    visibility = [
        visibility_row(0, 0),
        visibility_row(0, 24, "partial"),
        visibility_row(1, 0),
        visibility_row(1, 12, "severe"),
    ]

    rows = build_frame_analysis(manifest, trajectories, visibility)

    assert rows[0]["delta_q"] is None
    assert rows[1]["step_gap"] == 24
    assert rows[1]["delta_q"] == [0.24] * 7
    assert rows[1]["delta_q_per_step"] == [0.01] * 7
    assert rows[2]["delta_q"] is None
    assert rows[3]["delta_q"] == [-0.12] * 7
    assert rows[3]["visibility_group"] == "severe"
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `pytest tests/simulation/test_audit_action_tokenization.py::ActionInputTests::test_builds_frame_analysis_without_cross_episode_delta -v`

Expected: FAIL，错误包含 `ModuleNotFoundError` 或缺少 `build_frame_analysis`。

- [ ] **Step 3: 实现最小异常类型、读取和逐帧派生**

在新模块中定义稳定 schema、允许的 visibility group，并按轨迹原始顺序验证：

```python
SCHEMA_VERSION = "action_tokenization_audit_v1"
ALLOWED_VISIBILITY_GROUPS = {"clear", "partial", "severe"}

class AuditValidationError(ValueError):
    def __init__(self, reason, **evidence):
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence

def build_frame_analysis(manifest, trajectory_rows, visibility_rows):
    if manifest.get("schema_version") != "expert_v1":
        raise AuditValidationError("unsupported_dataset_schema")
    if manifest.get("action_dim") != 9:
        raise AuditValidationError("unexpected_action_dim")
    visibility_by_key = _index_visibility_rows(visibility_rows)
    seen = set()
    previous_by_episode = {}
    output = []
    for line_number, row in enumerate(trajectory_rows, start=1):
        key = (row.get("episode_idx"), row.get("step_idx"))
        if key in seen:
            raise AuditValidationError("duplicate_trajectory_key", key=key)
        seen.add(key)
        action = _validate_action(row.get("action"), key, line_number)
        visibility = visibility_by_key.pop(key, None)
        if visibility is None:
            raise AuditValidationError("missing_visibility_key", key=key)
        previous = previous_by_episode.get(key[0])
        step_gap = None if previous is None else key[1] - previous["step_idx"]
        if step_gap is not None and step_gap <= 0:
            raise AuditValidationError("non_increasing_step", key=key)
        delta = None if previous is None else [
            current - prior
            for current, prior in zip(action[:7], previous["q_target"])
        ]
        output.append(_analysis_row(row, visibility, action, step_gap, delta))
        previous_by_episode[key[0]] = {"step_idx": key[1], "q_target": action[:7]}
    if visibility_by_key:
        raise AuditValidationError(
            "extra_visibility_keys", keys=sorted(visibility_by_key)
        )
    _validate_episode_termination(output)
    return output
```

辅助函数必须拒绝非列表、非9维、布尔关节值、NaN/Inf、非法 visibility group、重复 visibility key，以及每个 episode 非末帧 terminate 或末帧不 terminate。

- [ ] **Step 4: 增加每种硬门禁的定向测试**

使用 `subTest` 或独立测试覆盖：`unsupported_dataset_schema`、`unexpected_action_dim`、`duplicate_trajectory_key`、`duplicate_visibility_key`、`missing_visibility_key`、`extra_visibility_keys`、`invalid_action`、`non_finite_action`、`non_increasing_step`、`invalid_visibility_group` 和 `invalid_episode_termination`。每个测试断言 `exception.reason`，不只断言抛出任意异常。

- [ ] **Step 5: 运行 Task 1 测试并提交**

Run: `pytest tests/simulation/test_audit_action_tokenization.py -v`

Expected: PASS。

```bash
git add src/vla_project/simulation/audit_action_tokenization.py tests/simulation/test_audit_action_tokenization.py
git commit -m "feat: validate action audit inputs"
```

### Task 2: 数值、特殊维度与可见性联合统计

**Files:**
- Modify: `src/vla_project/simulation/audit_action_tokenization.py`
- Modify: `tests/simulation/test_audit_action_tokenization.py`

**Interfaces:**
- Consumes: Task 1 的 `frame_analysis: list[dict]`。
- Produces: `numeric_stats(values: Sequence[float]) -> dict`、`summarize_action_analysis(rows: list[dict]) -> dict`。

- [ ] **Step 1: 写出固定样本的统计失败测试**

用 `[0.0, 1.0, 2.0, 3.0]` 锁定总体标准差和线性分位数；用小型 frame rows 锁定 `step_gap`、夹爪、终止、分组样本数和 severe 标记：

```python
def test_numeric_stats_uses_population_std_and_linear_percentiles(self):
    result = numeric_stats([0.0, 1.0, 2.0, 3.0])
    assert result["count"] == 4
    assert result["mean"] == 1.5
    assert result["std"] == pytest.approx(1.11803398875)
    assert result["p25"] == pytest.approx(0.75)
    assert result["p99"] == pytest.approx(2.97)

def test_summarizes_special_dimensions_and_visibility_groups(self):
    summary = summarize_action_analysis(make_analysis_rows())
    assert summary["num_frames"] == 4
    assert summary["num_transitions"] == 2
    assert summary["gripper"]["is_constant"] is True
    assert summary["terminate"]["counts"] == {"0": 2, "1": 2}
    assert summary["step_gap"]["counts"] == {"12": 1, "24": 1}
    assert summary["by_visibility"]["severe"]["num_frames"] == 1
    assert summary["by_visibility"]["severe"][
        "insufficient_for_generalization"
    ] is True
```

- [ ] **Step 2: 运行新测试并确认缺少统计函数**

Run: `pytest tests/simulation/test_audit_action_tokenization.py -k 'numeric_stats or summarizes_special' -v`

Expected: FAIL，缺少 `numeric_stats` 或 `summarize_action_analysis`。

- [ ] **Step 3: 实现稳定数值统计和7维展开**

使用 NumPy 明确 `ddof=0` 和默认线性 percentile；把 NumPy 标量转换为原生 Python 数值：

```python
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)

def numeric_stats(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise AuditValidationError("invalid_numeric_series")
    percentiles = np.percentile(array, PERCENTILES, method="linear")
    result = {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "non_finite_count": 0,
    }
    result.update(
        {f"p{percentile:02d}": float(value)
         for percentile, value in zip(PERCENTILES, percentiles)}
    )
    return result
```

`summarize_action_analysis()` 分别产生 `absolute_q`、`delta_q`、`delta_q_per_step` 的7维数组统计，并为每个 visibility group 重用相同逻辑；空的差分组明确写 `num_transitions=0` 和空关节列表，不调用 `numeric_stats([])`。

- [ ] **Step 4: 增加 episode 覆盖和空 severe 组测试**

断言每个组的 `episode_count` 使用集合去重；即使 fixture 没有 severe 帧，summary 仍包含 severe 键、`num_frames=0` 和 `insufficient_for_generalization=true`。

- [ ] **Step 5: 运行 Task 2 测试并提交**

Run: `pytest tests/simulation/test_audit_action_tokenization.py -v`

Expected: PASS。

```bash
git add src/vla_project/simulation/audit_action_tokenization.py tests/simulation/test_audit_action_tokenization.py
git commit -m "feat: summarize action visibility distributions"
```

### Task 3: 分箱模拟与确定性推荐

**Files:**
- Modify: `src/vla_project/simulation/audit_action_tokenization.py`
- Modify: `tests/simulation/test_audit_action_tokenization.py`

**Interfaces:**
- Consumes: Task 1 的 frame rows。
- Produces: `evaluate_binning(values: Sequence[float], method: str, num_bins: int) -> dict`、`evaluate_candidates(rows: list[dict]) -> list[dict]`、`select_recommendations(candidates: list[dict], thresholds: dict = RECOMMENDATION_THRESHOLDS) -> dict`。

- [ ] **Step 1: 写等宽分箱与重建误差失败测试**

固定输入 `[0, 1, 2, 3]`、2箱，断言边界 `[0, 1.5, 3]`、重建中心 `[0.75, 2.25]`、计数 `[2, 2]`、MAE `0.5` 和 p95 `0.75`：

```python
def test_evaluates_uniform_width_bins(self):
    result = evaluate_binning([0.0, 1.0, 2.0, 3.0], "uniform_width", 2)
    assert result["edges"] == [0.0, 1.5, 3.0]
    assert result["reconstruction_values"] == [0.75, 2.25]
    assert result["counts"] == [2, 2]
    assert result["mae"] == pytest.approx(0.5)
    assert result["p95_absolute_error"] == pytest.approx(0.75)
```

- [ ] **Step 2: 运行测试并确认缺少分箱函数**

Run: `pytest tests/simulation/test_audit_action_tokenization.py::BinningTests::test_evaluates_uniform_width_bins -v`

Expected: FAIL，缺少 `evaluate_binning`。

- [ ] **Step 3: 实现等宽、等频和重复边界语义**

`uniform_width` 使用 `np.linspace(min, max, num_bins + 1)`；`quantile` 使用 `np.quantile(values, np.linspace(0, 1, num_bins + 1), method="linear")` 后 `np.unique()` 合并边界。统一用 `np.searchsorted(edges[1:-1], values, side="right")` 分配边界值；等宽重建取边缘中点，等频重建取箱内中位数。返回请求箱数、有效箱数、占用率、计数、熵和误差。若稳健范围 `p99-p01 == 0`，返回 `eligible_metrics=false` 和 `zero_robust_range`。

- [ ] **Step 4: 写重复分位数和归一化熵测试**

输入 `[0, 0, 0, 1]` 请求4个 quantile bins，断言重复边界被合并、`effective_num_bins < 4`、occupancy 分母仍为4、没有零宽度边缘。对 `[0, 0, 1, 1]` 的2个等宽箱断言 normalized entropy 为1。

- [ ] **Step 5: 写12候选和最坏关节汇总测试**

构造每个关节略有不同的 frame rows，断言候选键集合恰好是2种 representation × 2种 method × 3种 bins；每个候选的 `worst_joint` 来自7维最差 occupancy、minimum count 和 normalized p95，而不是平均值。

- [ ] **Step 6: 写推荐排序和无合格候选测试**

用手工 candidate dict 锁定选择顺序：16箱优先于32箱；同箱数误差更低优先；再比较 minimum count；完全相同优先等宽。另一 fixture 让全部离散候选 occupancy 小于0.90，断言：

```python
assert recommendations["continuous_regression"]["included"] is True
assert recommendations["absolute_q"]["status"] == "no_eligible_candidate"
assert recommendations["delta_q"]["status"] == "no_eligible_candidate"
```

- [ ] **Step 7: 实现候选资格与推荐理由**

定义：

```python
RECOMMENDATION_THRESHOLDS = {
    "minimum_nonempty_bin_occupancy": 0.90,
    "minimum_nonempty_bin_count": 20,
    "maximum_normalized_p95_reconstruction_error": 0.05,
}
```

候选必须逐关节全部过线；`rejection_reasons` 使用稳定枚举：`occupancy_below_threshold`、`minimum_count_below_threshold`、`normalized_p95_above_threshold`、`zero_robust_range`。推荐输出同时保留所有候选排序和被选候选的完整标识。

- [ ] **Step 8: 运行 Task 3 测试并提交**

Run: `pytest tests/simulation/test_audit_action_tokenization.py -v`

Expected: PASS。

```bash
git add src/vla_project/simulation/audit_action_tokenization.py tests/simulation/test_audit_action_tokenization.py
git commit -m "feat: evaluate action binning candidates"
```

### Task 4: 报告构建、失败证据、原子发布与 CLI

**Files:**
- Modify: `src/vla_project/simulation/audit_action_tokenization.py`
- Modify: `tests/simulation/test_audit_action_tokenization.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_package_metadata.py`

**Interfaces:**
- Consumes: Tasks 1–3 的 frame rows、statistics、candidates 和 recommendations。
- Produces: `run_action_tokenization_audit(dataset_dir: str | Path, output_dir: str | Path | None = None) -> dict`、`main() -> None`、console script `vla-audit-action-tokenization`。

- [ ] **Step 1: 写小型临时数据集端到端失败测试**

在 `TemporaryDirectory` 中写 manifest、两条 episode 的 trajectory JSONL 和 visibility JSONL，调用 runner 后断言：

```python
summary = run_action_tokenization_audit(dataset_dir)
output_dir = dataset_dir / "action_tokenization_audit_v1"
assert summary["passed"] is True
assert summary["num_frames"] == 4
assert (output_dir / "frame_action_analysis.jsonl").is_file()
assert (output_dir / "action_tokenization_audit.json").is_file()
saved = json.loads(
    (output_dir / "action_tokenization_audit.json").read_text()
)
assert saved == summary
assert saved["inputs"]["trajectory"]["sha256"]
```

- [ ] **Step 2: 运行端到端测试并确认 runner 尚不存在**

Run: `pytest tests/simulation/test_audit_action_tokenization.py::PublicationTests::test_publishes_complete_audit -v`

Expected: FAIL，缺少 `run_action_tokenization_audit`。

- [ ] **Step 3: 实现 hash、summary 和原子发布**

用分块 SHA-256 记录 manifest、trajectory 和 visibility 输入；summary 包含 schema、ISO 秒级创建时间、dataset dir、输入 metadata、门禁 passed、episode/frame/transition 数、Task 2 统计、Task 3 candidates/recommendations 和顶层 `passed=true`。在 `output_dir.parent` 创建 `TemporaryDirectory`，先写两个文件并重新读取校验，再用目录 rename 发布；已有成功目录通过同父目录 backup rename 实现可恢复替换，成功后删除 backup。

- [ ] **Step 4: 写失败不替换旧成功输出测试**

先放置包含 sentinel 的旧成功目录，再提供非法动作。断言 runner 抛出 `AuditValidationError`、旧 sentinel 仍存在、数据集根目录生成 `action_tokenization_audit_failure.json`、failure 中 `passed=false` 和稳定 reason。再修复输入成功运行，断言成功目录被完整替换且过期 failure 文件被删除。

- [ ] **Step 5: 实现失败证据和 CLI**

runner 捕获异常后原子写根目录 failure JSON，但重新抛出原异常使 CLI 非零退出。CLI 参数固定为 `--dataset-dir` 和可选 `--output-dir`，成功打印 episode、frames、transitions 以及 absolute/delta 推荐状态。

- [ ] **Step 6: 注册第15个命令并先运行元数据失败测试**

先在 `tests/test_package_metadata.py` 的 `EXPECTED_SCRIPTS` 增加：

```python
"vla-audit-action-tokenization": (
    "vla_project.simulation.audit_action_tokenization:main"
),
```

Run: `pytest tests/test_package_metadata.py -v`

Expected: FAIL，因为 `pyproject.toml` 尚未注册新命令或已安装 editable metadata 尚未刷新。

- [ ] **Step 7: 修改 pyproject、刷新 editable 安装并通过测试**

在 `[project.scripts]` 增加同名入口，然后运行：

```bash
python -m pip install -e .
pytest tests/simulation/test_audit_action_tokenization.py tests/test_package_metadata.py -v
```

Expected: 全部 PASS，安装后的 console scripts 与15项精确契约一致。

- [ ] **Step 8: 提交发布与 CLI**

```bash
git add src/vla_project/simulation/audit_action_tokenization.py tests/simulation/test_audit_action_tokenization.py pyproject.toml tests/test_package_metadata.py
git commit -m "feat: publish action tokenization audit"
```

### Task 5: 真实全量审计、项目知识回写与最终验证

**Files:**
- Modify: `README.md`
- Modify: `docs/agent/PROJECT_OVERVIEW.md`
- Modify: `docs/agent/CURRENT_STATUS.md`
- Modify: `docs/agent/PROJECT_STRUCTURE.md`
- Modify: `docs/worklog/WORKLOG.md`
- Generated: `outputs/dataset/expert_scaling_v1/action_tokenization_audit_v1/frame_action_analysis.jsonl`
- Generated: `outputs/dataset/expert_scaling_v1/action_tokenization_audit_v1/action_tokenization_audit.json`

**Interfaces:**
- Consumes: Task 4 的 console script 和冻结300条数据。
- Produces: 真实 action tokenization 审计证据、更新后的稳定架构/当前状态/使用说明/工作日志。

- [ ] **Step 1: 运行定向和全量自动测试**

Run: `pytest tests/simulation/test_audit_action_tokenization.py tests/test_package_metadata.py -v`

Expected: PASS。

Run: `pytest -q`

Expected: 全部测试通过；记录精确通过数量供 CURRENT_STATUS/WORKLOG 使用。

- [ ] **Step 2: 运行300条真实只读审计**

Run:

```bash
vla-audit-action-tokenization --dataset-dir outputs/dataset/expert_scaling_v1
```

Expected: 退出码0，`frames=9894`、`transitions=9594`，成功生成两个输出文件且不修改输入 SHA-256。

- [ ] **Step 3: 独立核验真实报告关键契约**

使用只读 Python 命令读取 summary，断言：schema 正确、`passed=true`、300 episodes、9,894 frames、9,594 transitions、12 candidates、continuous baseline included，并打印 absolute/delta 推荐和拒绝理由。再运行 `git status --short`，确认 `outputs/` 没有进入 Git 变更。

- [ ] **Step 4: 根据真实证据更新项目文档**

只写真实报告产生的数值，不预填结果：

- README：命令表新增 `vla-audit-action-tokenization`，补充运行示例和报告位置；
- PROJECT_OVERVIEW：命令数14改15，新增已验证动作分布/候选结论和路线恢复点；
- CURRENT_STATUS：记录审计指标、推荐候选、未解决边界、下一步 tokenizer/episode split；
- PROJECT_STRUCTURE：登记新源码/测试职责，并把14个命令改为15个；
- WORKLOG：记录设计理由、输入门禁、真实联合分布、12候选结果和阶段判断。

- [ ] **Step 5: 运行文档后的最终验证**

Run:

```bash
pytest -q
python -m compileall -q src tests
git diff --check
git status --short
```

Expected: 全部测试通过，compileall 退出0，diff check 无输出；status 只包含本任务文档变更和用户原有未跟踪计划，不包含 outputs。

- [ ] **Step 6: 提交真实审计结论与文档**

```bash
git add README.md docs/agent/PROJECT_OVERVIEW.md docs/agent/CURRENT_STATUS.md docs/agent/PROJECT_STRUCTURE.md docs/worklog/WORKLOG.md
git commit -m "docs: record action tokenization audit"
```

- [ ] **Step 7: 最终提交审计**

Run: `git log --oneline -6`

Expected: 能看到设计、实施计划以及每个实现阶段的独立提交；`git status --short` 只剩用户进入任务前已有的未跟踪文件。
