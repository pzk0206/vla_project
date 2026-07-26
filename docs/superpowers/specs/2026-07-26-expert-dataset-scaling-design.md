# 专家数据规模化设计

**日期：** 2026-07-26  
**状态：** 已确认，待实施

## 目标

把现有50条专家 episode 的可用基线扩展为可复现、可审计、可用于后续
action tokenization 的数据集工作流。先采集10条 pilot 并执行质量门禁，只有 pilot
通过后才扩展到至少300条有效 episode。

本阶段不调用 VLM API，不使用 VLM 预测生成专家标签，也不开始模型训练。

## 已知基线与问题

现有 `outputs/dataset/` 包含50条 episode、286条帧记录和286张图片，摘要均为
`termination_reason=success`。当前采集器存在三个规模化风险：

1. episode 开始时不复位机械臂，后一条轨迹依赖前一条轨迹的终点；已有 episode 只需
   1步即成功，说明轨迹长度受到跨 episode 状态影响。
2. 目标位置和相机扰动没有按 episode 保存确定性 seed，无法独立复现指定轨迹。
3. 单帧 schema 缺少 `schema_version`、`episode_idx`、`step_idx` 和 `random_seed`，
   也没有统一的数据质量报告。

## 数据目录与保护边界

新数据写入：

```text
outputs/dataset/expert_scaling_v1/
```

现有 `outputs/dataset/trajectory_expert.jsonl`、`episode_summary.jsonl` 和图片保持不动。
清理行为只能作用于明确配置的 `expert_scaling_v1/`，不得递归删除
`outputs/dataset/` 根目录。

pilot 和正式扩展使用同一 schema。pilot 通过后以追加模式继续采集，episode 编号从
现有 pilot 最大编号之后继续，避免复制或重写已经验收的样本。

## 可复现的独立 Episode

`dataset` 配置新增：

```yaml
schema_version: "expert_v1"
random_seed: 1000
reset_robot_each_episode: true
home_joint_positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

episode `i` 使用：

```text
episode_seed = random_seed + episode_idx
```

每条 episode 开始前：

1. 使用 `episode_seed` 初始化目标位置和相机扰动的随机源。
2. 将7个受控关节复位到 `home_joint_positions`，关节速度清零。
3. 推进固定的 settle steps，使观测从稳定且一致的初始状态开始。
4. 再创建本 episode 的红块、相机和专家轨迹。

复位只影响机械臂状态，不改变现有 IK、成功距离、卡住检测和动作标签定义。

## 数据 Schema

`trajectory_expert.jsonl` 每行继续对应一张图片和一个9维动作，并增加：

```text
schema_version
episode_idx
step_idx
random_seed
```

保留现有字段：

```text
image_path
instruction
action
camera_eye
block_pos
target_pos
ee_pos
distance_to_target
termination_reason
```

`episode_summary.jsonl` 每行增加：

```text
schema_version
random_seed
initial_ee_pos
initial_block_pos
```

并保留 episode 编号、步数、帧数、最终距离、终止原因、相机和最终状态。

动作 schema 继续定义为：

```text
7维关节目标 + 1维夹爪状态 + 1维终止标志
```

本阶段不修改动作含义，避免在数据规模化时同时引入 action tokenization 变量。

## Manifest 与配置快照

每个数据集目录包含：

```text
dataset_manifest.json
config_snapshot.yaml
trajectory_expert.jsonl
episode_summary.jsonl
dataset_quality_report.json
```

`dataset_manifest.json` 记录：

- `schema_version`
- 数据集名称与创建时间
- 指令和动作维度
- seed 基数与 episode seed 规则
- 图片尺寸
- 目标位置范围和相机扰动范围
- pilot 目标数量与正式目标数量
- JSONL 和摘要文件名

`config_snapshot.yaml` 保存本次采集使用的完整配置副本。追加采集前必须验证关键 schema、
动作维度、图片尺寸和 seed 配置与 manifest 一致，不一致时拒绝追加。

## 数据质量检查

新增独立的 `simulation` 质量检查模块和 CLI。检查器只读取数据集，不修改图片或 JSONL，
生成 `dataset_quality_report.json`。

报告至少包含：

- episode 数、帧数和成功率
- `termination_reason` 计数
- `final_distance` 的 min/mean/median/max
- 每条 episode 帧数的 min/mean/median/max
- 红块 X/Y 范围与分箱计数
- 相机位置范围
- JSONL 引用图片的存在率、可读取率和尺寸一致率
- orphan 图片数量
- schema 缺失、动作维度错误、重复 `(episode_idx, step_idx)` 数量
- episode seed 重复或不符合规则的数量

质量检查遇到单条坏记录时继续扫描并在报告中列出错误；只有无法读取 manifest 或主
JSONL 时才整体失败。

## Pilot 门禁

首次运行只采集10条 episode。pilot 必须同时满足：

```text
num_episodes = 10
success_count = 10
schema_error_count = 0
missing_image_count = 0
unreadable_image_count = 0
image_size_mismatch_count = 0
orphan_image_count = 0
duplicate_step_key_count = 0
seed_error_count = 0
```

此外，每条 episode 必须至少保存1帧，终止帧的动作终止标志必须为1，最终距离必须不超过
当前 `task.success_distance`。

pilot 未通过时保留完整输出作为诊断证据，不自动清理，也不继续生成正式300条数据。

## 正式扩展

pilot 通过后，将同一版本数据集追加到至少300条有效 episode。正式扩展后重新运行完整
质量检查，并要求：

- 有效 episode 至少300条。
- 总体成功率不低于99%，所有失败或异常 episode 均保留摘要。
- 所有图片与 JSONL 一一对应且可读取。
- 所有记录符合 `expert_v1` schema。
- block position 覆盖配置的 X/Y 范围，不集中在单一分箱。

如果采集中出现失败或异常，该 episode 仍写入摘要并进入质量报告，但不计入“有效
episode”。不得通过删除失败摘要制造100%成功率。

## 测试与验收

实施采用 TDD，至少覆盖：

1. 相同 seed 独立复现相同目标和相机位置。
2. 每条 episode 开始前关节位置和速度被复位。
3. 清理只作用于版本化数据目录。
4. 追加采集继续 episode 编号且拒绝 manifest 不兼容。
5. 单帧和摘要 schema 包含新增字段。
6. 质量检查识别缺图、坏图、尺寸错误、孤儿图片、重复 step、错误 seed 和动作维度。
7. pilot 任一门禁失败时不得进入正式扩展。

最终验收包括自动测试、`compileall`、`git diff --check`、10条真实 PyBullet pilot 和
生成的质量报告。pilot 不使用 VLM API。

## 非目标

- 不修改 VLM grounding、相机补偿或 smoke stop 策略。
- 不引入多物体、多语言指令或真实机械臂数据。
- 不在本阶段设计 action tokens 或训练模型。
- 不为了达到数量目标删除失败证据或覆盖原有50条数据。
