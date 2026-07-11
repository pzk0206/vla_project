# Stage 3 批量评估系统设计

## 目标

把当前只能运行一次、覆盖一次 trace 的 `stage3_probe.py`，升级为可重复的 Stage 3 批量实验系统。第一轮默认评估 20 个随机 episode，验证运行和统计链路；系统稳定后再把评估规模扩展到 50 个 episode，并以 heuristic 成功率达到 80% 作为当前阶段验收门槛。

## 范围

本次实现包括：

- 将单次 probe 流程整理为可复用的 `run_probe_episode(...)`。
- 新增 `evaluate_probe.py` 批量调用单次流程。
- 为每个 episode 保存独立 trace 和摘要。
- 自动生成整批实验的 JSON 汇总。
- 只为失败 episode 保留观测图片。
- 使用固定基础随机种子，使失败位置可复现。
- 为纯逻辑汇总、输出策略和错误隔离补充自动测试。
- 更新 README、WORKLOG 和学习计划中的运行方式与当前阶段证据。

本次不包括：

- 调用真实 VLM API 做正式对比。
- 根据第一批结果自动调参。
- 训练或微调任何模型。
- 重构整个仓库目录结构。

## 选择的实现方案

`evaluate_probe.py` 直接导入并调用 `stage3_probe.run_probe_episode(...)`。不通过子进程重复启动脚本，也不复制控制循环。这样单次运行和批量评估共享同一套决策、方向映射、IK、终止和 trace 逻辑，后续切换 heuristic 或 VLM 时不会出现两个实现漂移。

## 配置

在 `sim_config.yaml` 新增：

```yaml
probe_evaluation:
  num_episodes: 20
  output_dir: "probe_eval_runs"
  save_failure_images_only: true
  random_seed: 42
```

配置语义：

- `num_episodes`：本批运行的随机 episode 数；第一轮为 20，正式门槛实验改为 50。
- `output_dir`：所有批量评估结果的根目录。
- `save_failure_images_only`：为 true 时成功 episode 删除步骤图片，失败和异常 episode 保留图片。
- `random_seed`：episode `i` 使用 `random_seed + i`，保证同一配置下可以复现目标位置和相机扰动。

批量评估默认沿用 `probe.mode`。当前配置必须保持 `heuristic`；本次实现不会自动调用 API。若将来使用 API，必须由用户显式修改配置并单独控制成本。

## 单次 probe 接口

`stage3_probe.py` 提供：

```python
run_probe_episode(config, episode_idx, episode_dir, random_seed) -> dict
```

该函数负责：

1. 设置 episode 随机种子。
2. 连接并初始化 PyBullet 世界。
3. 在独立目录中运行一次闭环。
4. 写入 `probe_trace.jsonl`。
5. 返回 episode 摘要。
6. 无论成功、失败或异常都断开 PyBullet。

现有 `python stage3_probe.py` 仍然可用。它调用同一个 `run_probe_episode(...)` 完成单次探路，避免破坏当前使用习惯。

## 输出结构

每次启动评估创建一个带时间戳的批次目录：

```text
probe_eval_runs/
└── run_YYYYMMDD_HHMMSS/
    ├── episode_summary.jsonl
    ├── probe_eval_summary.json
    ├── episode_000/
    │   └── probe_trace.jsonl
    ├── episode_001/
    │   ├── probe_trace.jsonl
    │   ├── probe_step_00.jpg
    │   └── ...
    └── ...
```

每个 episode 在运行时先按 `probe.save_trace_images` 保存图片。episode 成功且 `save_failure_images_only: true` 时，只删除该 episode 的 `probe_step_*.jpg`；trace 永远保留。失败或异常时保留所有已生成图片。

## Episode 摘要

每个 episode 返回并写入 `episode_summary.jsonl`：

```text
episode_idx
random_seed
success
termination_reason
num_control_steps
initial_distance
final_distance
final_block_pos
direction_counts
distance_increase_steps
trace_path
error
```

字段规则：

- `success` 仅在 `termination_reason == "success"` 时为 true。
- 正常失败使用 `max_control_steps`。
- 未预期异常使用 `error`，并在 `error` 字段保存异常类型和消息。
- `num_control_steps` 使用实际 trace 行数。
- `distance_increase_steps` 统计 `distance_delta < 0` 的步骤数。

## 批次汇总

`probe_eval_summary.json` 包含：

```text
num_episodes
success_count
failure_count
error_count
success_rate
final_distance_mean
final_distance_median
final_distance_max
control_steps_mean
control_steps_median
termination_reason_counts
direction_counts
failed_episode_indices
config_snapshot
```

统计规则：

- `success_rate = success_count / num_episodes`。
- final distance 统计排除没有产生有效距离的异常 episode，但异常仍计入总 episode 和失败率。
- control steps 使用所有实际产生 trace 的 episode。
- `config_snapshot` 至少保存 probe 与 probe_evaluation 配置，便于结果复现。

## 错误隔离

单个 episode 抛出异常时，评估器：

1. 保留该 episode 已生成的 trace 和图片。
2. 写入 `termination_reason: error` 的摘要。
3. 继续运行后续 episode。

配置缺失、配置值非法、输出根目录无法创建等批次级错误应立即失败，避免生成一批不可信结果。

## 测试策略

使用现有标准库 `unittest`，覆盖：

- episode 摘要从 trace 正确计算。
- success rate、均值、中位数、最大值、方向计数正确汇总。
- 异常 episode 计入失败率但不会破坏整批统计。
- `random_seed + episode_idx` 的种子策略稳定。
- 成功 episode 删除图片但保留 trace。
- 失败 episode 保留图片和 trace。
- 单次入口与批量入口调用相同的 `run_probe_episode(...)`。

运行验证分三层：

1. 全部单元测试。
2. 2-3 个 episode 的快速 DIRECT smoke test。
3. 20 个 episode 的第一批正式评估。

## 完成标准

- `python evaluate_probe.py` 一条命令完成批量 heuristic 评估。
- 单个失败或异常不阻断整批实验。
- 每个 episode 有独立、可追溯的摘要和 trace。
- 成功 episode 不保留步骤图片，失败 episode 保留图片。
- 汇总 JSON 可以直接回答成功率、最终距离、控制步数和失败类型。
- 20 次结果被写入 WORKLOG，且不把单批结果夸大为最终稳定性结论。
- 系统通过 smoke test 后，可以仅修改 `num_episodes: 50` 进入阶段验收实验。
