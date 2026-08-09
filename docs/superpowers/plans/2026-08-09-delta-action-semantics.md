# Delta Action Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 `delta_q_64` 训练标签与运行时解码语义，使每个样本严格表示同一 episode 相邻保存帧的绝对关节目标差，并让新 checkpoint 自包含可审计 tokenizer 资产。

**Architecture:** 原始 `trajectory_expert.jsonl` 继续只保存绝对关节目标。`BCDataset` 在内存中按 `(episode_idx, step_idx)` 排序并构造 delta 样本；tokenizer 加载器同时返回边界和审计生成的重建值；训练 checkpoint 固化动作语义、tokenizer 和 split 哈希；rollout 对 delta v2 checkpoint 强校验后执行 `current_q + decoded_delta`。

**Tech Stack:** Python 3.10、NumPy、PyTorch、Pillow、`unittest`、JSON/YAML

## Global Constraints

- 不修改冻结的原始轨迹 JSONL，不跨 episode 做差。
- 不删除或改写 `bc_delta_q_64_*_v1` checkpoint 和旧 rollout 证据。
- 旧 delta 0% 结果标记为 `invalid_reason=absolute_labels_encoded_as_delta`，不得用于动作表示结论。
- 新 delta 训练目录必须使用 `*_v2`；实现和 CPU 测试完成前不启动 GPU 重训。
- tokenizer 边界必须只由 train episode 拟合；checkpoint 必须保存 split SHA-256。
- 运行时只能使用审计报告的 `reconstruction_values`，不得退回箱中心近似。
- 每项生产修改前先运行能稳定复现缺陷的失败测试；测试命令使用 `env PYTHONPATH=src`。

---

### Task 1: 同 Episode 相邻 Delta 标签

**Files:**
- Modify: `src/vla_project/training/dataset.py`
- Create: `tests/training/test_dataset.py`

**Interfaces:**
- Produces: `_build_delta_rows(rows) -> list[dict]`
- Rule: 输出行按 `(episode_idx, step_idx)` 排序，每个 episode 首行删除，其余行携带 `_delta_q`

- [ ] **Step 1: 写失败测试**

构造两个交错且乱序的 episode，每个动作前7维使用容易核对的数值。测试必须断言：

```python
delta_rows = _build_delta_rows(rows)
self.assertEqual(
    [(row["episode_idx"], row["step_idx"]) for row in delta_rows],
    [(0, 20), (0, 40), (1, 30)],
)
self.assertEqual(delta_rows[0]["_delta_q"], [0.2] * 7)
self.assertEqual(delta_rows[1]["_delta_q"], [-0.05] * 7)
self.assertEqual(delta_rows[2]["_delta_q"], [0.4] * 7)
```

另测重复 `(episode_idx, step_idx)`、非递增/非法 action 维度和跨 episode 首帧不产生样本。

- [ ] **Step 2: 确认 RED**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.training.test_dataset.DeltaRowTests -v`

Expected: `_build_delta_rows` 不存在。

- [ ] **Step 3: 实现最小差分构造**

```python
def _build_delta_rows(rows):
    ordered = sorted(rows, key=lambda row: (row["episode_idx"], row["step_idx"]))
    previous = {}
    seen = set()
    result = []
    for row in ordered:
        key = (row["episode_idx"], row["step_idx"])
        if key in seen:
            raise ValueError(f"duplicate trajectory key: {key}")
        seen.add(key)
        q_target = _validated_q_target(row, key)
        prior = previous.get(row["episode_idx"])
        previous[row["episode_idx"]] = q_target
        if prior is None:
            continue
        derived = dict(row)
        derived["_delta_q"] = (q_target - prior).astype(np.float32).tolist()
        result.append(derived)
    return result
```

`BCDataset` 的 delta 分支使用 `_build_delta_rows`；`__getitem__` 编码 `_delta_q`，absolute 与 regression 继续编码 `action[:7]`。

- [ ] **Step 4: 确认 GREEN 并回归**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.training.test_dataset -v`

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.simulation.test_audit_action_tokenization -q`

- [ ] **Step 5: 提交**

```bash
git add src/vla_project/training/dataset.py tests/training/test_dataset.py
git commit -m "fix: generate true per-episode delta labels"
```

---

### Task 2: 使用审计重建值的 Tokenizer 资产

**Files:**
- Modify: `src/vla_project/training/tokenizer_utils.py`
- Modify: `src/vla_project/training/dataset.py`
- Modify: `src/vla_project/training/rollout.py`
- Create: `tests/training/test_tokenizer_utils.py`

**Interfaces:**
- Produces: `load_quantile_tokenizer(audit_path, representation, num_bins) -> dict`
- Produces: `decode_tokens_to_action(tokens, reconstruction_values) -> np.ndarray`
- Tokenizer dict: `representation`, `binning`, `requested_num_bins`, `edges`, `reconstruction_values`

- [ ] **Step 1: 写并运行 RED 测试**

测试报告中的某箱使用非中心重建值，例如边界 `[0.0, 1.0, 3.0]`、重建值 `[0.1, 2.8]`；断言 token `[0, 1]` 解码为 `[0.1, 2.8]` 而非 `[0.5, 2.0]`。同时拒绝 joint 数不为7、重建值为 `None`、边界与重建长度不一致的报告。

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.training.test_tokenizer_utils -v`

Expected: `load_quantile_tokenizer` 缺失或旧 decode 返回箱中心。

- [ ] **Step 2: 实现严格 tokenizer 加载和解码**

`load_quantile_edges` 保留为编码兼容包装，但内部调用 `load_quantile_tokenizer`。新 decode 只接受7组明确重建值，校验 token 范围，不从 edges 推导中心。

- [ ] **Step 3: 让数据集持有完整 tokenizer**

`BCDataset` 分类分支保存 `self.tokenizer` 与 `self._edges`；训练可从数据集读取完全相同的资产写入 checkpoint。

现有 rollout 的外部 audit 兼容加载也返回完整 tokenizer，并把
`tokenizer["reconstruction_values"]` 传给 decode；Task 4 再把 delta 的资产来源收紧为
checkpoint 内嵌 metadata。

- [ ] **Step 4: 验证并提交**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.training.test_tokenizer_utils tests.training.test_dataset -q`

```bash
git add src/vla_project/training/tokenizer_utils.py src/vla_project/training/dataset.py src/vla_project/training/rollout.py tests/training/test_tokenizer_utils.py tests/training/test_dataset.py docs/superpowers/plans/2026-08-09-delta-action-semantics.md
git commit -m "fix: decode action tokens with audited reconstructions"
```

---

### Task 3: Delta v2 训练元数据与输出命名

**Files:**
- Modify: `src/vla_project/training/train.py`
- Create: `tests/training/test_train_metadata.py`

**Interfaces:**
- Produces: `_training_split_sha256(split_path) -> str`
- Produces: `_build_checkpoint_metadata(action_representation, split_sha256, tokenizer=None, action_stats=None) -> dict`
- Delta semantic version: `same_episode_saved_target_delta_v2`

- [ ] **Step 1: 写并运行 RED 测试**

测试必须断言 delta 默认输出分别为 `bc_delta_q_64_full_v2` 和 `bc_delta_q_64_overfit_10_v2`；metadata 包含：

```python
{
    "action_semantics": "same_episode_saved_target_delta_v2",
    "training_split_sha256": "<64 hex>",
    "tokenizer": {
        "representation": "delta_q",
        "binning": "quantile",
        "requested_num_bins": 64,
        "edges": [[0.0, 0.1]],
        "reconstruction_values": [[0.05]],
    },
}
```

absolute 分类保存 `absolute_joint_target_v1` 与 tokenizer；regression 保存 `absolute_joint_target_v1` 和 action stats。测试 `_save_checkpoint` 合并公共 metadata 与每 epoch loss，而不是覆盖任一方。

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.training.test_train_metadata -v`

- [ ] **Step 2: 实现元数据构造与 v2 命名**

在模型/数据加载之后、训练循环之前从 train dataset 取得 tokenizer，并计算实际 `episode_split.json` SHA-256。所有 best/last checkpoint 使用同一公共 metadata；`config.yaml` 和 `summary.json` 同步写入动作语义与 split hash。

- [ ] **Step 3: 验证并提交**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.training.test_train_metadata tests.training.test_output_boundaries -q`

```bash
git add src/vla_project/training/train.py tests/training/test_train_metadata.py
git commit -m "feat: version delta checkpoints with frozen semantics"
```

---

### Task 4: Rollout 强制使用 Checkpoint Tokenizer

**Files:**
- Modify: `src/vla_project/training/rollout.py`
- Create: `tests/training/test_rollout_delta.py`

**Interfaces:**
- Produces: `_load_checkpoint_action_assets(metadata, action_representation) -> dict`
- Consumes: Task 2 tokenizer dict、Task 3 checkpoint metadata

- [ ] **Step 1: 写并运行 RED 测试**

测试旧 delta metadata 缺少 `action_semantics` 时拒绝并包含
`absolute_labels_encoded_as_delta`；新 metadata 使用保存的 `reconstruction_values` 解码，且一次执行目标严格等于 mock 当前关节角加 decoded delta。patch 模型、相机与 PyBullet，不启动真实仿真。

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.training.test_rollout_delta -v`

- [ ] **Step 2: 实现 checkpoint 资产校验**

delta rollout 不再从数据集目录重新加载 tokenizer；缺少 v2 semantic、tokenizer 或 split hash 时在连接仿真前拒绝。absolute legacy checkpoint 可继续使用外部 audit 兼容路径，但新 checkpoint 优先使用内嵌资产。

- [ ] **Step 3: 验证并提交**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.training.test_rollout_delta tests.training.test_output_boundaries -q`

```bash
git add src/vla_project/training/rollout.py tests/training/test_rollout_delta.py
git commit -m "fix: enforce delta checkpoint semantics during rollout"
```

---

### Task 5: 废弃旧 Checkpoint 并建立重训门禁

**Files:**
- Modify: `docs/debugging/BUGLOG.md`
- Modify: `docs/worklog/WORKLOG.md`
- Modify: `docs/agent/CURRENT_STATUS.md`
- Modify: `docs/agent/PROJECT_OVERVIEW.md`

**Interfaces:**
- Invalid reason: `absolute_labels_encoded_as_delta`
- Invalid artifacts: `outputs/training/bc_delta_q_64_overfit_10_v1/`、`outputs/training/bc_delta_q_64_full_v1/` 及其旧 rollout

- [ ] **Step 1: 记录不可变旧证据**

文档逐项记录旧 checkpoint 路径、原始训练摘要、0% rollout 及 invalid reason；不改写、不移动、不删除旧 artifact。明确旧结果不能比较 delta 与 regression。

- [ ] **Step 2: 运行 CPU 语义回归与完整回归**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.training.test_dataset tests.training.test_tokenizer_utils tests.training.test_train_metadata tests.training.test_rollout_delta -v`

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest discover -q`

Expected: delta 定向测试全绿；完整回归不新增失败，仍只允许两个 config contract 与一个 package metadata 基线失败。

Run: `env PYTHONPYCACHEPREFIX=/tmp/vla_delta_pycache conda run -n vla_env python -m compileall -q src tests`

Run: `git diff --check`

- [ ] **Step 3: 提交文档与重训决策**

```bash
git add docs/debugging/BUGLOG.md docs/worklog/WORKLOG.md docs/agent/CURRENT_STATUS.md docs/agent/PROJECT_OVERVIEW.md
git commit -m "docs: invalidate legacy delta checkpoints"
```

只有上述验证完成后，才允许按新目录依次运行 overfit 与 full 重训；任何 GPU/长时运行开始前先记录命令、数据集哈希和目标输出目录，并确保目录不存在。

---

### Task 6: 重训 Delta v2 Checkpoint

**Files:**
- Read: `outputs/dataset/expert_scaling_v1/action_tokenization_audit_v2/action_tokenization_audit.json`
- Create: `outputs/training/bc_delta_q_64_overfit_10_v2/`
- Create: `outputs/training/bc_delta_q_64_full_v2/`
- Modify: `docs/worklog/WORKLOG.md`
- Modify: `docs/agent/CURRENT_STATUS.md`

**Interfaces:**
- Consumes: Task 1–4 的 delta v2 训练和 checkpoint 契约
- Produces: 两个全新、不可覆盖的 delta v2 checkpoint 目录

- [ ] **Step 1: 只读预检 tokenizer 与 split 证据**

确认 audit 报告存在、`boundary_fit_episodes == 250`、`split.boundary_fit_episode_count == 250`、`delta_q/quantile/64` 候选的7个 joint 都包含有限的 edges 和无 `None` 的 reconstruction values；计算并记录 `episode_split.json` SHA-256。不得重新生成或覆盖现有 audit。

- [ ] **Step 2: 确认两个新输出目录均不存在**

Run: `test ! -e outputs/training/bc_delta_q_64_overfit_10_v2`

Run: `test ! -e outputs/training/bc_delta_q_64_full_v2`

任一目录存在时停止，不清理、不续写，改用新的明确版本号并先更新计划与状态。

- [ ] **Step 3: 运行 overfit 重训并验证产物**

Run: `env PYTHONPATH=src conda run -n vla_env python -m vla_project.training.train --dataset-dir outputs/dataset/expert_scaling_v1 --output-dir outputs/training/bc_delta_q_64_overfit_10_v2 --action-representation delta_q_64 --overfit 10 --epochs 200 --batch-size 16`

验证 `checkpoint_best.pt`、`checkpoint_last.pt`、`config.yaml`、loss JSONL 和 `summary.json` 全部存在；两个 checkpoint 的 semantic、split hash 和 tokenizer 与预检完全一致，loss 全部有限。

- [ ] **Step 4: 运行 full 重训并验证产物**

Run: `env PYTHONPATH=src conda run -n vla_env python -m vla_project.training.train --dataset-dir outputs/dataset/expert_scaling_v1 --output-dir outputs/training/bc_delta_q_64_full_v2 --action-representation delta_q_64 --epochs 50 --batch-size 32`

执行与 overfit 相同的结构和 metadata 验证。训练失败时保留目录作为失败证据，不把它引用为有效 checkpoint，并使用新的版本目录重试。

- [ ] **Step 5: 记录重训事实并提交文档**

`WORKLOG.md` 记录命令、run id、耗时和 loss；`CURRENT_STATUS.md` 只在两个目录完整验证后标记重训完成。不得在尚未执行闭环 rollout 时声称 delta 表示有效或无效。

```bash
git add docs/worklog/WORKLOG.md docs/agent/CURRENT_STATUS.md
git commit -m "docs: record delta v2 retraining evidence"
```
