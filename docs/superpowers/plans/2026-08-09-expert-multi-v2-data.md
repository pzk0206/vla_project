# 双积木 v2 专家数据实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建无初始碰撞、红蓝位置同分布、完整保存双积木场景状态并通过10条 pilot 与300条 scale 门禁的斜视 `expert_multi_v2` 数据。

**Architecture:** `control_arm.py` 用纯联合采样器先生成两个显式位置，再加载并复核物理状态；统一 scene-state 捕获器向帧与摘要提供双积木、机器人和相机状态。`evaluate_dataset.py` 在保留 expert_v1 兼容扫描的同时，对 expert_multi_v2 启用额外 schema、间距、漂移、目标一致性和任务平衡门禁。

**Tech Stack:** Python 3.10、PyBullet、NumPy、OpenCV、PyYAML、`unittest`、JSON/JSONL

**当前进度（2026-08-09）：** Task 1–6 已完成；真实采集尚未启动。下一执行点为 Task 7
运行前只读检查。全量343项测试中342项通过，唯一失败为既有的安装态 CLI 元数据不一致。

## Global Constraints

- 只修改隔离 worktree；不触碰主工作树中的用户 notebook 修改。
- 旧 `outputs/dataset/expert_multi_v1/` 只读，不删除、不追加、不回写。
- 新目录固定为 `outputs/dataset/expert_multi_v2/`，schema 为 `expert_multi_v2`，相机继续斜视。
- 红蓝位置来自相同范围；至少一个 XY 轴中心间距不小于0.12m，最多拒绝100次。
- settle 与 episode 内任一积木 XY 漂移上限均为0.005m。
- task 由基础 seed 与 episode index 确定性平衡；10条为5/5，300条为150/150。
- 每项生产行为修改前必须观察对应测试按预期失败；命令使用 `env PYTHONPATH=src`。
- 代码和 CPU 测试全绿前不运行采集；pilot 门禁通过前不追加290条。
- 失败数据与报告保留，不清理非空目录后重试。

---

### Task 1: 联合采样与平衡任务

**Files:**
- Modify: `src/vla_project/simulation/control_arm.py`
- Modify: `tests/simulation/test_control_arm.py`
- Modify: `sim_config.yaml`
- Modify: `tests/test_config_contract.py`

**Interfaces:**
- Produces: `sample_block_pair_positions(task_config, rng=random) -> dict[str, list[float]]`
- Produces: `select_task(config, episode_idx=None) -> tuple[str, str]`
- Contract: `pair_sampling.min_axis_separation_xy=0.12`，`max_attempts=100`

- [ ] **Step 1: 写联合采样与任务平衡红测**

新增测试：用可控 RNG 返回先重叠、后合法的坐标，断言结果包含 red/blue 且
`abs(dx) >= 0.12 or abs(dy) >= 0.12`；不可能范围在100次后抛出包含
`unable to sample non-overlapping block pair` 的 `ValueError`。对 episode 0–299 断言
red/blue 各150条，并重复调用得到完全相同序列。

- [ ] **Step 2: 确认 RED**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.simulation.test_control_arm.BlockPairSamplingTests -v`

Expected: FAIL，因为联合采样器不存在，旧 `select_task` 仍使用随机选择。

- [ ] **Step 3: 实现最小联合采样和任务策略**

实现按轴分离判断：

```python
def _pair_is_separated(red, blue, minimum):
    return abs(red[0] - blue[0]) >= minimum or abs(red[1] - blue[1]) >= minimum
```

`sample_block_pair_positions` 从同一个 `pair_sampling` 范围分别采红蓝位置，合法立即返回，
100次后抛错。`balanced_alternating` 使用
`tasks[(episode_idx + dataset.random_seed) % len(tasks)]`；旧配置或未传 episode index 时
保留原 random 兼容行为。

- [ ] **Step 4: 更新 v2 配置契约并验证 GREEN**

把数据目录、schema、计划规模和 pair sampling 写入 `sim_config.yaml`；配置测试断言
`expert_multi_v2`、斜视 eye offset、0.12m/100次/5mm阈值及 balanced 策略。

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.simulation.test_control_arm.BlockPairSamplingTests tests.test_config_contract -q`

- [ ] **Step 5: 提交**

```bash
git add src/vla_project/simulation/control_arm.py tests/simulation/test_control_arm.py sim_config.yaml tests/test_config_contract.py
git commit -m "feat: sample balanced non-overlapping block pairs"
```

---

### Task 2: 完整 Scene State Schema

**Files:**
- Modify: `src/vla_project/simulation/control_arm.py`
- Modify: `tests/simulation/test_control_arm.py`

**Interfaces:**
- Produces: `capture_scene_state(robot_id, robot_config, block_ids, camera_eye, target_block) -> dict`
- Changes: `write_dataset_step(..., scene_state=None)`
- Changes: `write_episode_summary(..., initial_scene_state=None, final_scene_state=None)`

- [ ] **Step 1: 写 scene state 和 writer 红测**

mock `getBasePositionAndOrientation`、`getJointState` 和 `get_link_position`，断言 scene state
同时包含 red/blue position/orientation、7个 joint positions/velocities、EE、camera 和目标色。
另断言 expert_multi_v2 writer 缺少 scene state 时拒绝，写入后兼容 `block_pos` 与目标块位置
在 `1e-6` 内一致；摘要的 initial/final scene state 完整。

- [ ] **Step 2: 确认 RED**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.simulation.test_control_arm.SceneStateTests -v`

Expected: FAIL，因为 `capture_scene_state` 和 writer 参数不存在。

- [ ] **Step 3: 实现捕获器与 schema 校验**

`capture_scene_state` 只读取状态，不推进仿真。四元数原样保存；writer 对
`schema_version == "expert_multi_v2"` 强制 scene state 非空并核对 target/block_pos，
expert_v1 调用保持兼容。

- [ ] **Step 4: 验证 GREEN 与旧 writer 回归**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.simulation.test_control_arm.SceneStateTests tests.simulation.test_control_arm.EpisodeReproducibilityTests -q`

- [ ] **Step 5: 提交**

```bash
git add src/vla_project/simulation/control_arm.py tests/simulation/test_control_arm.py
git commit -m "feat: record complete dual-block scene state"
```

---

### Task 3: Episode 集成与物理前置门禁

**Files:**
- Modify: `src/vla_project/simulation/control_arm.py`
- Modify: `tests/simulation/test_control_arm.py`

**Interfaces:**
- Produces: `load_block_at_position(task_config, position, color_rgba, global_scaling) -> int`
- Produces: `_validate_settled_pair(sampled_positions, scene_state, pair_config) -> None`
- Consumes: Task 1 pair positions、Task 2 scene state

- [ ] **Step 1: 写 episode 集成红测**

断言 `run_episode` 在任何图片/JSONL 写入前：按 episode index 选任务、联合采样一次、用两个
显式位置加载、settle 后捕获状态并验证间距/5mm漂移。断言每帧 scene state 始终有两块，
蓝色任务的旧 `block_pos` 指向 blue，摘要 initial/final 也指向目标块。另测 settle 漂移超限
时抛错且不写图片或轨迹。

- [ ] **Step 2: 确认 RED**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.simulation.test_control_arm.MultiV2EpisodeTests -v`

- [ ] **Step 3: 最小集成**

让 `load_block`/`load_second_block` 旧接口继续可用，新 v2 路径只使用显式位置加载器。
每次保存图像前捕获一次当前 scene state，并从该状态计算 `block_pos`/`target_pos`。结束摘要
使用最后一次真实状态；finally 仍删除两个 body。

- [ ] **Step 4: 验证 GREEN 和 control_arm 全回归**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.simulation.test_control_arm -q`

- [ ] **Step 5: 提交**

```bash
git add src/vla_project/simulation/control_arm.py tests/simulation/test_control_arm.py
git commit -m "feat: enforce expert multi v2 episode state"
```

---

### Task 4: v2 Manifest 与追加采集 CLI

**Files:**
- Modify: `src/vla_project/simulation/control_arm.py`
- Modify: `tests/simulation/test_control_arm.py`

**Interfaces:**
- Changes: `main(argv=None)` accepts `--num-episodes N`
- Manifest adds: `task_selection`, `pair_sampling`, `tasks`, `second_block`

- [ ] **Step 1: 写 manifest/CLI 红测**

断言 v2 manifest 固化联合采样与任务契约，追加时任一字段变化均拒绝。mock `run_episode`
调用 `main(["--num-episodes", "10"])`，断言只运行10条但 manifest 的 target 仍为300；已有
10条摘要后 `--num-episodes 290` 从 episode 10运行到299。

- [ ] **Step 2: 确认 RED**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.simulation.test_control_arm.DatasetRunContractTests tests.simulation.test_control_arm.CollectionCliTests -v`

- [ ] **Step 3: 实现 CLI 和 manifest 兼容字段**

`main(argv=None)` 用 argparse 解析正整数 override，不修改加载后的计划规模字段；循环次数
使用 override。`MANIFEST_COMPATIBILITY_FIELDS` 加入所有 v2 场景字段。

- [ ] **Step 4: 验证包入口与回归**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.simulation.test_control_arm -q`

- [ ] **Step 5: 提交**

```bash
git add src/vla_project/simulation/control_arm.py tests/simulation/test_control_arm.py
git commit -m "feat: add auditable pilot append collection cli"
```

---

### Task 5: Expert Multi v2 质量扫描与门禁

**Files:**
- Modify: `src/vla_project/simulation/evaluate_dataset.py`
- Modify: `tests/simulation/test_evaluate_dataset.py`

**Interfaces:**
- Produces report fields: `task_counts`, `scene_state_error_count`, `target_consistency_error_count`, `block_overlap_error_count`, `block_drift_error_count`, `pair_min_axis_separation_stats`, `block_xy_drift_stats`
- Changes: `QUALITY_ERROR_COUNT_FIELDS` includes all v2 integrity counts

- [ ] **Step 1: 写 v2 schema 红测**

建立最小有效 v2 frame/summary fixture。逐项破坏缺失蓝块、NaN、非单位四元数、7维关节
状态长度、target/instruction、block_pos/target_pos、0.12m间距和5mm漂移，断言各自错误计数
增加且 `passed=false`，扫描继续报告其余错误。

- [ ] **Step 2: 写任务平衡门禁红测**

合成10条5/5和300条150/150报告应通过对应 gate；10条6/4、300条151/149、任一 v2
完整性错误必须失败。expert_v1 的既有 pilot/scale fixture 继续使用原门禁。

- [ ] **Step 3: 确认 RED**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.simulation.test_evaluate_dataset.ExpertMultiV2Tests -v`

- [ ] **Step 4: 实现版本分派扫描与门禁**

只在 manifest schema 为 expert_multi_v2 时执行额外验证；用有限数检查和四元数范数检查
scene state，以 summary initial state 为漂移基准。把详细错误写入 `errors`，所有新计数纳入
pilot/scale checks 和顶层 `passed`。

- [ ] **Step 5: 验证 GREEN 与旧 evaluator 回归**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.simulation.test_evaluate_dataset -q`

- [ ] **Step 6: 提交**

```bash
git add src/vla_project/simulation/evaluate_dataset.py tests/simulation/test_evaluate_dataset.py
git commit -m "feat: gate expert multi v2 scene integrity"
```

---

### Task 6: 代码阶段整体验证与文档门禁

**Files:**
- Modify: `README.md`
- Modify: `docs/agent/PROJECT_OVERVIEW.md`
- Modify: `docs/agent/CURRENT_STATUS.md`
- Modify: `docs/worklog/WORKLOG.md`
- Modify: `docs/debugging/BUGLOG.md`

- [ ] **Step 1: 运行双积木定向与相关回归**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest tests.simulation.test_control_arm tests.simulation.test_evaluate_dataset tests.test_config_contract tests.test_package_metadata -v`

- [ ] **Step 2: 运行全量回归与静态检查**

Run: `env PYTHONPATH=src conda run -n vla_env python -m unittest discover -q`

Expected: 不新增失败；若既有 config/package 基线已被本阶段同步修复，则记录新的实际通过数，
否则明确列出仍存在的同3项失败，不能报告全绿。

Run: `env PYTHONPYCACHEPREFIX=/tmp/vla_multi_v2_pycache conda run -n vla_env python -m compileall -q src tests`

Run: `git diff --check`

- [ ] **Step 3: 更新使用说明与 Bug 证据**

记录旧 v1 的重叠实例、字段歧义根因、v2 命令、schema、门禁和“尚未采集”的状态，不预写
任何成功指标。

- [ ] **Step 4: 提交**

```bash
git add README.md docs/agent/PROJECT_OVERVIEW.md docs/agent/CURRENT_STATUS.md docs/worklog/WORKLOG.md docs/debugging/BUGLOG.md
git commit -m "docs: prepare expert multi v2 collection gate"
```

---

### Task 7: 采集10条 Pilot 并执行门禁

**Artifacts:**
- Create: `outputs/dataset/expert_multi_v2/`

- [ ] **Step 1: 只读运行前检查**

确认目标目录不存在、配置为 DIRECT/斜视/expert_multi_v2、计划规模300、seed规则稳定；记录
配置文件和当前提交 SHA-256。目录若已存在则停止，不清理或复用未知证据。

- [ ] **Step 2: 运行 pilot**

Run: `env PYTHONPATH=src conda run -n vla_env vla-collect --num-episodes 10`

- [ ] **Step 3: 运行 pilot 质量门禁**

Run: `env PYTHONPATH=src conda run -n vla_env vla-evaluate-dataset --dataset-dir outputs/dataset/expert_multi_v2`

Expected: `active_gate=pilot`、`passed=true`、10/10 success、red/blue=5/5、所有完整性计数0。

- [ ] **Step 4: 失败处理或记录通过证据**

若失败，保留目录和报告并停止追加，按系统化调试定位；修复后使用新数据版本目录。若通过，
把真实哈希、帧数、间距和漂移指标写入 WORKLOG/CURRENT_STATUS 并提交。

---

### Task 8: 追加290条并执行最终 Scale 门禁

**Artifacts:**
- Modify by append only: `outputs/dataset/expert_multi_v2/`

- [ ] **Step 1: 确认 pilot 报告仍通过且目录恰有10条**

重读 manifest、summary、quality report 和哈希；任一不一致立即停止。

- [ ] **Step 2: 安全追加290条**

Run: `env PYTHONPATH=src conda run -n vla_env vla-collect --num-episodes 290`

Expected: 从 episode 10/seeds 1010 开始，到 episode 299/seed 1299 结束，不覆盖前10条。

- [ ] **Step 3: 运行最终质量门禁**

Run: `env PYTHONPATH=src conda run -n vla_env vla-evaluate-dataset --dataset-dir outputs/dataset/expert_multi_v2`

Expected: `active_gate=scale`、`passed=true`、300/300 success、red/blue=150/150、所有完整性
错误计数0，最小轴间距不低于0.12m，最大 XY 漂移不超过0.005m。

- [ ] **Step 4: 审计生成物并记录结论**

核对300个唯一 episode、seed 1000–1299、图片引用/文件一一对应、manifest/config/JSONL 哈希，
更新 PROJECT_OVERVIEW、CURRENT_STATUS、WORKLOG 和 BUGLOG。只声明数据门禁通过，不声明
语言跟随或反事实成功。

- [ ] **Step 5: 提交文档恢复点**

```bash
git add docs/agent/PROJECT_OVERVIEW.md docs/agent/CURRENT_STATUS.md docs/worklog/WORKLOG.md docs/debugging/BUGLOG.md
git commit -m "docs: record expert multi v2 dataset evidence"
```
