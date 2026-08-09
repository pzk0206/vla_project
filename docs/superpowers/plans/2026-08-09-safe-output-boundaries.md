# Safe Output Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让所有可配置的目录创建、覆盖、删除和运行名都被限制在项目管理的 `outputs/` 子树内，并保证失败时不丢失旧证据。

**Architecture:** 在包根新增无领域依赖的 `output_paths.py`，统一完成项目根定位、规范化路径校验、单段运行名校验和原子目录发布。各领域编排入口在任何写操作、仿真初始化或模型加载前调用该模块；低层纯写函数仍允许测试使用临时目录，但不能作为未校验的 CLI 边界。

**Tech Stack:** Python 3.10、`pathlib`、`tempfile`、`shutil`、`contextlib`、`unittest`

## Global Constraints

- 不运行真实专家数据采集、GPU 训练、付费 VLM API 或 QLoRA。
- 不覆盖或删除 `expert_multi_v1`、旧 checkpoint、旧 rollout 和用户 notebook 修改。
- 相对输出路径固定相对于项目根目录，不依赖调用命令时的当前工作目录。
- 绝对路径只有在规范化后仍位于对应允许根目录内才接受。
- 数据集非空版本目录不可清理；使用新版本名替代覆盖。
- 每项生产代码修改前必须先运行能稳定复现缺陷的失败测试。

---

### Task 1: 统一安全路径与原子发布模块

**Files:**
- Create: `src/vla_project/output_paths.py`
- Create: `tests/test_output_paths.py`
- Modify: `docs/agent/PROJECT_STRUCTURE.md`

**Interfaces:**
- Produces: `project_root() -> Path`
- Produces: `resolve_managed_output(requested_path, *, allowed_root, project_root_override=None, require_child=True) -> Path`
- Produces: `validate_run_name(run_name: str) -> str`
- Produces: `publish_directory_atomically(staging_dir, destination_dir, *, allowed_root, project_root_override=None) -> Path`
- Produces: `staged_output_directory(destination_dir, *, allowed_root, project_root_override=None)` context manager

- [ ] **Step 1: 写路径规范化失败测试**

在 `tests/test_output_paths.py` 创建 `ResolveManagedOutputTests`，覆盖：

```python
def test_accepts_relative_child_and_returns_absolute_path(self):
    actual = resolve_managed_output(
        "outputs/dataset/expert_v2",
        allowed_root="outputs/dataset",
        project_root_override=self.root,
    )
    self.assertEqual(actual, self.root / "outputs/dataset/expert_v2")

def test_rejects_allowed_root_itself(self):
    with self.assertRaisesRegex(ValueError, "leaf directory"):
        resolve_managed_output(
            "outputs/dataset",
            allowed_root="outputs/dataset",
            project_root_override=self.root,
        )

def test_rejects_parent_segments_even_when_result_would_be_inside_root(self):
    with self.assertRaisesRegex(ValueError, "parent traversal"):
        resolve_managed_output(
            "outputs/dataset/a/../b",
            allowed_root="outputs/dataset",
            project_root_override=self.root,
        )

def test_rejects_escape_to_source_tree(self):
    with self.assertRaisesRegex(ValueError, "outside managed root"):
        resolve_managed_output(
            "outputs/dataset/../../src",
            allowed_root="outputs/dataset",
            project_root_override=self.root,
        )

def test_rejects_symlink_parent_that_resolves_outside_root(self):
    (self.root / "outputs/dataset/link").symlink_to(self.outside, target_is_directory=True)
    with self.assertRaisesRegex(ValueError, "outside managed root"):
        resolve_managed_output(
            "outputs/dataset/link/run",
            allowed_root="outputs/dataset",
            project_root_override=self.root,
        )
```

同时覆盖：绝对的根内路径被接受、绝对根外路径被拒绝、空字符串、`.`、`..` 和 NUL
字符被拒绝。

- [ ] **Step 2: 运行测试并确认 RED**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.test_output_paths.ResolveManagedOutputTests -v`

Expected: `ModuleNotFoundError: No module named 'vla_project.output_paths'`。

- [ ] **Step 3: 实现最小路径校验**

`src/vla_project/output_paths.py` 使用以下核心逻辑：

```python
def project_root():
    return Path(__file__).resolve().parents[2]


def _resolve_from_project(value, root):
    path = Path(value)
    return path.resolve(strict=False) if path.is_absolute() else (root / path).resolve(strict=False)


def resolve_managed_output(
    requested_path,
    *,
    allowed_root,
    project_root_override=None,
    require_child=True,
):
    raw = str(requested_path)
    if not raw or "\x00" in raw or Path(raw) in {Path("."), Path("..")}:
        raise ValueError("invalid managed output path")
    if ".." in Path(raw).parts:
        raise ValueError("parent traversal is not allowed")
    root = Path(project_root_override).resolve() if project_root_override else project_root()
    managed_root = _resolve_from_project(allowed_root, root)
    resolved = _resolve_from_project(requested_path, root)
    try:
        relative = resolved.relative_to(managed_root)
    except ValueError as exc:
        raise ValueError("output path is outside managed root") from exc
    if require_child and relative == Path("."):
        raise ValueError("output path must be a leaf directory")
    return resolved
```

- [ ] **Step 4: 运行路径测试并确认 GREEN**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.test_output_paths.ResolveManagedOutputTests -v`

Expected: 全部通过。

- [ ] **Step 5: 写运行名单段测试并确认 RED**

```python
def test_run_name_accepts_single_safe_component(self):
    self.assertEqual(validate_run_name("run_20260809_v1"), "run_20260809_v1")

def test_run_name_rejects_directory_components(self):
    for value in ("../escape", "a/b", "a\\b", ".", "..", "/tmp/run"):
        with self.subTest(value=value), self.assertRaises(ValueError):
            validate_run_name(value)
```

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.test_output_paths.ValidateRunNameTests -v`

Expected: import 或符号缺失失败。

- [ ] **Step 6: 实现 `validate_run_name` 并确认 GREEN**

```python
def validate_run_name(run_name):
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name must be a non-empty directory name")
    if run_name in {".", ".."} or "/" in run_name or "\\" in run_name:
        raise ValueError("run_name must be one directory component")
    if Path(run_name).is_absolute() or Path(run_name).name != run_name:
        raise ValueError("run_name must be one directory component")
    return run_name
```

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.test_output_paths.ValidateRunNameTests -v`

Expected: 全部通过。

- [ ] **Step 7: 写原子发布失败恢复测试并确认 RED**

测试必须验证：无旧目录时发布成功；有旧目录时成功替换；发布第二次 rename 失败时旧目录
内容恢复；staging 和 destination 任一逃逸时在移动前拒绝。

```python
def test_failed_publish_restores_previous_directory(self):
    destination = self.managed_root / "run"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")
    staging = self.managed_root / ".staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")
    original_replace = Path.replace
    call_count = 0

    def fail_second_replace(path, target):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("publish failed")
        return original_replace(path, target)

    with patch.object(Path, "replace", autospec=True, side_effect=fail_second_replace):
        with self.assertRaisesRegex(OSError, "publish failed"):
            publish_directory_atomically(
                staging,
                destination,
                allowed_root=self.managed_root,
                project_root_override=self.root,
            )
    self.assertEqual((destination / "old.txt").read_text(encoding="utf-8"), "old")
```

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.test_output_paths.AtomicPublishTests -v`

Expected: 符号缺失失败。

- [ ] **Step 8: 实现原子发布与 staging context manager**

实现必须在 destination 父目录创建临时目录，先把旧目录改名为 backup，再发布 staging；
任何异常恢复 backup。context manager 只在调用方正常退出后发布：

```python
@contextmanager
def staged_output_directory(
    destination_dir,
    *,
    allowed_root,
    project_root_override=None,
):
    destination = resolve_managed_output(
        destination_dir,
        allowed_root=allowed_root,
        project_root_override=project_root_override,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent,
        prefix=f".{destination.name}_staging_",
    ) as temp_dir:
        staging = Path(temp_dir) / destination.name
        staging.mkdir()
        yield staging
        publish_directory_atomically(
            staging,
            destination,
            allowed_root=allowed_root,
            project_root_override=project_root_override,
        )
```

- [ ] **Step 9: 运行模块测试、编译并更新结构文档**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.test_output_paths -v`

Run: `env PYTHONPYCACHEPREFIX=/tmp/vla_safe_paths_pycache conda run -n vla_env python -m compileall -q src/vla_project/output_paths.py tests/test_output_paths.py`

在 `PROJECT_STRUCTURE.md` 的包根目录说明中登记 `output_paths.py` 和
`tests/test_output_paths.py`，说明它们只提供跨领域安全输出边界。

- [ ] **Step 10: 提交 Task 1**

```bash
git add src/vla_project/output_paths.py tests/test_output_paths.py docs/agent/PROJECT_STRUCTURE.md
git commit -m "feat: add managed output path boundaries"
```

---

### Task 2: 专家数据集不可覆盖与路径穿越修复

**Files:**
- Modify: `src/vla_project/simulation/control_arm.py`
- Modify: `tests/simulation/test_control_arm.py`

**Interfaces:**
- Consumes: `resolve_managed_output`
- Produces: `validate_versioned_dataset_dir(output_dir, project_root_override=None) -> Path`
- Changes: `prepare_dataset` 拒绝清理非空版本目录，不再调用 `shutil.rmtree`

- [ ] **Step 1: 写失败测试**

新增测试：

```python
def test_dataset_path_rejects_parent_traversal(self):
    with self.assertRaisesRegex(ValueError, "parent traversal"):
        control_arm.validate_versioned_dataset_dir(
            "outputs/dataset/../../src",
            project_root_override=self.root,
        )

def test_clean_before_run_refuses_nonempty_version_directory(self):
    dataset = self.root / "outputs/dataset/expert_v1"
    dataset.mkdir(parents=True)
    evidence = dataset / "trajectory_expert.jsonl"
    evidence.write_text("evidence\n", encoding="utf-8")
    config = self.dataset_config(dataset, clean_before_run=True)
    with self.assertRaisesRegex(FileExistsError, "new dataset version"):
        control_arm.prepare_dataset(config, project_root_override=self.root)
    self.assertEqual(evidence.read_text(encoding="utf-8"), "evidence\n")
```

还要测试绝对根内版本目录被接受、根目录本身被拒绝、空版本目录可以初始化。

- [ ] **Step 2: 运行测试并确认 RED**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.simulation.test_control_arm.DatasetPreparationTests -v`

Expected: traversal 测试未抛异常，或函数不接受 `project_root_override`；非空目录被删除。

- [ ] **Step 3: 最小实现**

- `validate_versioned_dataset_dir` 委托 `resolve_managed_output`；
- `prepare_dataset` 增加仅测试注入使用的 `project_root_override=None`；
- 删除 `shutil.rmtree`；
- `clean_before_run=true` 且目录非空时抛 `FileExistsError`；
- 空目录或不存在目录继续初始化；
- 错误发生在 manifest、snapshot 或 JSONL 写入之前。

- [ ] **Step 4: 定向与回归验证**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.simulation.test_control_arm -v`

Expected: 全部通过。

- [ ] **Step 5: 提交 Task 2**

```bash
git add src/vla_project/simulation/control_arm.py tests/simulation/test_control_arm.py
git commit -m "fix: prevent destructive dataset path escapes"
```

---

### Task 3: VLM 样本生成改为 staging 后发布

**Files:**
- Modify: `src/vla_project/vlm/collect_vlm_eval_samples.py`
- Modify: `tests/vlm/test_collect_vlm_eval_samples.py`

**Interfaces:**
- Consumes: `resolve_managed_output`, `staged_output_directory`
- Changes: `collect_vlm_eval_samples(config, project_root_override=None)` 只向 staging 写入

- [ ] **Step 1: 写旧证据保留失败测试**

```python
def test_generation_failure_preserves_existing_output(self):
    output = self.root / "outputs/vlm_samples/run"
    output.mkdir(parents=True)
    marker = output / "old.json"
    marker.write_text("old", encoding="utf-8")
    config = self.config_with_output(output)
    with patch.object(module, "capture_balanced_pose_sample", side_effect=RuntimeError("boom")):
        with self.assertRaisesRegex(RuntimeError, "boom"):
            module.collect_vlm_eval_samples(config, project_root_override=self.root)
    self.assertEqual(marker.read_text(encoding="utf-8"), "old")
```

另写测试确保 `outputs/vlm_samples/../../src` 在 capture mock 被调用前拒绝，以及成功生成后
旧目录被新目录原子替换。

- [ ] **Step 2: 运行测试并确认 RED**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.vlm.test_collect_vlm_eval_samples -v`

Expected: 旧输出在捕获异常前已被 `rmtree` 删除，或新参数不存在。

- [ ] **Step 3: 最小实现**

- 在函数开头将配置路径限制到 `outputs/vlm_samples/`；
- 删除 `shutil.rmtree(output_dir)`；
- 使用 `staged_output_directory` 获取 staging；
- 图片和 JSONL 全部写入 staging；记录中的 `image_path` 转换为最终目录的项目相对路径；
- `validate_samples` 增加 `image_root_override=None`，提供时按文件 basename 到
  `staging/images/` 校验真实图片，同时保持记录中的最终可移植路径不变；
- context manager 正常退出后发布并返回最终 manifest 路径。

- [ ] **Step 4: 验证并提交**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.vlm.test_collect_vlm_eval_samples -v`

Expected: 全部通过。

```bash
git add src/vla_project/vlm/collect_vlm_eval_samples.py tests/vlm/test_collect_vlm_eval_samples.py
git commit -m "fix: publish VLM samples without deleting evidence"
```

---

### Task 4: 审计与派生数据发布边界

**Files:**
- Modify: `src/vla_project/simulation/audit_action_tokenization.py`
- Modify: `src/vla_project/simulation/audit_dataset_visibility.py`
- Modify: `src/vla_project/simulation/render_expert_dataset_view.py`
- Modify: `tests/simulation/test_audit_action_tokenization.py`
- Modify: `tests/simulation/test_audit_dataset_visibility.py`
- Modify: `tests/simulation/test_render_expert_dataset_view.py`

**Interfaces:**
- Consumes: `resolve_managed_output`, `staged_output_directory`
- Rule: 审计输出必须是输入数据集的直接或嵌套子目录，派生数据集必须是
  `outputs/dataset/` 下与源目录不同的版本叶子目录

- [ ] **Step 1: 写三个入口的失败测试**

每个入口至少增加：输出等于数据集根目录时拒绝；`../` 逃逸时拒绝；拒绝发生在读取、
重放或写失败报告之前。action audit 额外测试原子发布异常恢复旧审计目录。

```python
def test_action_audit_rejects_output_equal_to_dataset(self):
    with patch.object(module, "load_audit_inputs") as load_inputs:
        with self.assertRaises(ValueError):
            module.run_action_tokenization_audit(self.dataset, self.dataset)
    load_inputs.assert_not_called()
```

- [ ] **Step 2: 运行测试并确认 RED**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.simulation.test_audit_action_tokenization tests.simulation.test_audit_dataset_visibility tests.simulation.test_render_expert_dataset_view -v`

Expected: 至少路径逃逸测试失败。

- [ ] **Step 3: 接入共享校验与原子发布**

- action audit 与 visibility audit 的默认输出保持不变；显式输出必须位于 dataset 内；
- action audit 用 `staged_output_directory` 替换自有 backup/replace 实现；
- visibility audit 保持“已存在即拒绝”，但在任何失败报告创建前验证路径；
- render 的既有 `validate_dataset_paths` 改为委托共享模块，同时保留源、目标不得相同和
  源目录只读 hash 校验；
- 失败报告只能写入经过校验的 dataset 或目标父目录。

- [ ] **Step 4: 验证并提交**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.simulation.test_audit_action_tokenization tests.simulation.test_audit_dataset_visibility tests.simulation.test_render_expert_dataset_view -v`

Expected: 全部通过。

```bash
git add src/vla_project/simulation/audit_action_tokenization.py src/vla_project/simulation/audit_dataset_visibility.py src/vla_project/simulation/render_expert_dataset_view.py tests/simulation/test_audit_action_tokenization.py tests/simulation/test_audit_dataset_visibility.py tests/simulation/test_render_expert_dataset_view.py
git commit -m "fix: constrain dataset audit output publication"
```

---

### Task 5: Probe 与 smoke 运行名路径逃逸

**Files:**
- Modify: `src/vla_project/simulation/evaluate_probe.py`
- Modify: `src/vla_project/vlm/grounding_smoke/runner.py`
- Modify: `src/vla_project/vlm/grounding_smoke/screening.py`
- Modify: `tests/simulation/test_evaluate_probe.py`
- Modify: `tests/vlm/grounding_smoke/test_runner.py`
- Modify: `tests/vlm/grounding_smoke/test_screening.py`

**Interfaces:**
- Consumes: `validate_run_name`, `resolve_managed_output`
- Rule: 自动时间戳或显式 `run_name` 都只能形成输出根下的一个新目录

- [ ] **Step 1: 写失败测试**

为三个 `make_run_dir` 边界测试 `../escape`、`a/b`、绝对路径；patch `Path.mkdir` 并断言
非法名字时从未调用。测试合法固定名称仍创建在对应临时输出根内。

- [ ] **Step 2: 运行测试并确认 RED**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.simulation.test_evaluate_probe tests.vlm.grounding_smoke.test_runner tests.vlm.grounding_smoke.test_screening -v`

Expected: 非法 run name 当前会拼接并逃逸，测试失败。

- [ ] **Step 3: 最小实现**

三个入口统一调用 `validate_run_name`，并对配置输出根使用各自的允许目录：

```text
evaluate_probe             -> outputs/probe_evaluations/
grounding_smoke.runner     -> outputs/vlm_evaluations/
grounding_smoke.screening  -> outputs/vlm_evaluations/
```

低层函数增加 `project_root_override=None` 供临时目录测试注入；生产 CLI 不暴露绕过参数。

- [ ] **Step 4: 验证并提交**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.simulation.test_evaluate_probe tests.vlm.grounding_smoke.test_runner tests.vlm.grounding_smoke.test_screening -v`

Expected: 全部通过。

```bash
git add src/vla_project/simulation/evaluate_probe.py src/vla_project/vlm/grounding_smoke/runner.py src/vla_project/vlm/grounding_smoke/screening.py tests/simulation/test_evaluate_probe.py tests/vlm/grounding_smoke/test_runner.py tests/vlm/grounding_smoke/test_screening.py
git commit -m "fix: reject run name path traversal"
```

---

### Task 6: 其余 CLI 输出目录边界

**Files:**
- Modify: `src/vla_project/simulation/stage3_probe.py`
- Modify: `src/vla_project/vlm/diagnose_vlm_grounding.py`
- Modify: `src/vla_project/vlm/evaluate_vlm_decisions.py`
- Modify: `src/vla_project/vlm/evaluate_ground_then_decide.py`
- Modify: `src/vla_project/vlm/evaluate_grounding_backprojection.py`
- Modify: `src/vla_project/vlm/validate_grounding_calibration.py`
- Modify: `src/vla_project/training/train.py`
- Modify: `src/vla_project/training/vla_train.py`
- Modify: `src/vla_project/training/rollout.py`
- Modify: `src/vla_project/training/vla_rollout.py`
- Modify: 对应现有测试文件
- Create: `tests/training/__init__.py`
- Create: `tests/training/test_output_boundaries.py`

**Interfaces:**
- Consumes: `resolve_managed_output`, `validate_run_name`
- Rule: CLI/config 编排入口在任何写操作、模型加载、仿真连接或 API 调用前验证输出

- [ ] **Step 1: 写参数化边界测试并确认 RED**

`tests/training/test_output_boundaries.py` 分别调用四个训练/rollout 顶层函数，传入
`outputs/training/../../src` 或 `outputs/rollout/../../src`，patch 模型构造与数据加载，
断言路径错误先发生且重依赖未调用。

现有 VLM 测试中给每个编排入口增加根外输出和含 `/` 的配置运行名用例，断言 API client、
图片写入和 PyBullet 未调用。

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.training.test_output_boundaries tests.vlm.test_diagnose_vlm_grounding tests.vlm.test_evaluate_vlm_decisions tests.vlm.test_evaluate_ground_then_decide tests.vlm.test_evaluate_grounding_backprojection tests.vlm.test_validate_grounding_calibration -v`

Expected: 路径在重依赖之后才使用或被允许，测试失败。

- [ ] **Step 2: 接入对应允许根**

```text
stage3_probe                    -> outputs/probe/
diagnose/evaluate VLM           -> outputs/vlm_evaluations/
grounding backprojection        -> outputs/vlm_evaluations/
calibration validation          -> outputs/vlm_evaluations/
BC/VLA train                    -> outputs/training/
BC/VLA rollout                  -> outputs/rollout/（兼容读取旧 outputs/rollouts/，新写统一到单数目录）
```

训练和 rollout 的新运行目录若非空则拒绝，防止 checkpoint、loss 和评估结果混写。现有
只读 checkpoint 与 dataset 输入不受输出根限制。

- [ ] **Step 3: 运行全部相关测试并确认 GREEN**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.training.test_output_boundaries tests.simulation.test_stage3_probe tests.vlm.test_diagnose_vlm_grounding tests.vlm.test_evaluate_vlm_decisions tests.vlm.test_evaluate_ground_then_decide tests.vlm.test_evaluate_grounding_backprojection tests.vlm.test_validate_grounding_calibration -v`

Expected: 全部通过，且测试不调用真实模型/API。

- [ ] **Step 4: 提交 Task 6**

```bash
git add src/vla_project/simulation/stage3_probe.py src/vla_project/vlm/diagnose_vlm_grounding.py src/vla_project/vlm/evaluate_vlm_decisions.py src/vla_project/vlm/evaluate_ground_then_decide.py src/vla_project/vlm/evaluate_grounding_backprojection.py src/vla_project/vlm/validate_grounding_calibration.py src/vla_project/training/train.py src/vla_project/training/vla_train.py src/vla_project/training/rollout.py src/vla_project/training/vla_rollout.py tests/simulation/test_stage3_probe.py tests/vlm/test_diagnose_vlm_grounding.py tests/vlm/test_evaluate_vlm_decisions.py tests/vlm/test_evaluate_ground_then_decide.py tests/vlm/test_evaluate_grounding_backprojection.py tests/vlm/test_validate_grounding_calibration.py tests/training
git commit -m "fix: constrain configurable experiment outputs"
```

---

### Task 7: 安全修复证据、全量回归与阶段收口

**Files:**
- Modify: `docs/debugging/BUGLOG.md`
- Modify: `docs/worklog/WORKLOG.md`
- Modify: `docs/agent/CURRENT_STATUS.md`

**Interfaces:**
- Consumes: Tasks 1–6 的测试结果与 commit
- Produces: 路径安全失败—根因—修复—验证证据链

- [ ] **Step 1: 运行攻击输入回归集合**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.test_output_paths tests.simulation.test_control_arm tests.vlm.test_collect_vlm_eval_samples tests.simulation.test_audit_action_tokenization tests.simulation.test_audit_dataset_visibility tests.simulation.test_render_expert_dataset_view tests.simulation.test_evaluate_probe tests.vlm.grounding_smoke.test_runner tests.vlm.grounding_smoke.test_screening tests.training.test_output_boundaries -v`

Expected: 全部通过。

- [ ] **Step 2: 运行完整验证**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest discover -q`

Expected at this阶段: 不新增失败；基线已有的两个 config contract 和一个 package metadata
失败仍准确记录，留到工程一致性阶段修复。

Run: `env PYTHONPYCACHEPREFIX=/tmp/vla_safe_full_pycache conda run -n vla_env python -m compileall -q src tests`

Expected: exit 0。

Run: `git diff --check`

Expected: exit 0。

- [ ] **Step 3: 更新权威记录**

`BUGLOG.md` 记录四类根因：词法 `Path.parents` 校验、先删后生成、任意 audit output
替换、run name 目录穿越；记录 RED 和 GREEN 命令结果。`WORKLOG.md` 记录实施提交和未运行
采集/训练。`CURRENT_STATUS.md` 把“目录安全修复”标记为完成，把 delta 修复标记为下一步。

- [ ] **Step 4: 提交阶段文档**

```bash
git add docs/debugging/BUGLOG.md docs/worklog/WORKLOG.md docs/agent/CURRENT_STATUS.md
git commit -m "docs: record managed output security fixes"
```
