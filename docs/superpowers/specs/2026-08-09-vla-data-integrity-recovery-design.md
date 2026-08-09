# VLA 数据完整性与可复现实验恢复设计

**日期：** 2026-08-09  
**状态：** 已确认，待按实施计划执行

## 1. 目标与执行顺序

本轮工作恢复专家数据、动作语义和 VLA 评估的可信证据链。实施必须保持以下顺序：

1. 封闭所有目录删除、覆盖和路径逃逸入口；
2. 修复 `delta_q_64` 标签语义并用测试证明；
3. 建立斜视双积木 `expert_multi_v2` 数据契约和质量门禁；
4. 用保存的场景状态确定性复现 VLA rollout；
5. 实现同场景、只换指令的成对反事实评估；
6. 补齐测试、依赖、CLI、notebook 和权威文档；
7. 代码与门禁通过后才采集 300 条新数据并重训；
8. 最后恢复 Qwen2-VL QLoRA。

旧数据和旧 checkpoint 只读保留，不覆盖、不删除。`expert_multi_v1`、旧 VLA 24.5%
rollout 和文档中的 70% 语言跟随率作为历史证据保存，但不再作为有效结论。

## 2. 非目标与冻结约束

- 主训练视角继续使用当前斜上方随机相机；不切换到严格俯视图。
- `expert_scaling_v1`、`expert_topdown_v1` 和单任务 BC regression/absolute 产物不重采集。
- 本轮不修改任务的 KUKA、IK、悬停高度、成功距离和保存频率，除非新门禁证明现有值
  无法满足双任务数据质量要求。
- 修复代码阶段不得运行真实批量采集、GPU 全量重训、付费 VLM API 或 QLoRA。
- 不用新结果回写旧输出目录；每组新实验使用新的版本化目录。

## 3. 统一安全输出边界

### 3.1 方案

在 `src/vla_project/output_paths.py` 建立跨 simulation、VLM 和 training 使用的稳定接口，
测试放在 `tests/test_output_paths.py`。共同模块只负责路径解析与安全发布，不包含领域逻辑。

核心接口：

```python
def resolve_managed_output(
    requested_path: str | Path,
    *,
    allowed_root: str | Path,
    project_root: str | Path | None = None,
    require_child: bool = True,
) -> Path:
    """返回规范化绝对路径；拒绝根目录本身、路径穿越和符号链接逃逸。"""


def publish_directory_atomically(
    staging_dir: str | Path,
    destination_dir: str | Path,
    *,
    allowed_root: str | Path,
) -> None:
    """校验 staging 后原子发布；失败时恢复原目录。"""
```

解析规则：

- 相对路径以项目根目录解析；绝对路径只有位于允许根目录内时才接受；
- 对路径和允许根目录都执行 `resolve(strict=False)`；
- 目标必须能通过 `relative_to(resolved_allowed_root)`；
- 默认拒绝允许根目录本身；
- 任一已存在父目录是指向允许根目录外的符号链接时拒绝；
- 空路径、`.`、`..`、包含 NUL 的路径和跨根目录路径均拒绝；
- 错误必须在任何删除、移动、建目录或 PyBullet 初始化之前抛出。

领域允许根目录：

| 调用方 | 允许根目录 |
| --- | --- |
| 专家数据采集 | `outputs/dataset/` |
| 动作/可见性审计 | 对应数据集目录，或其明确审计子目录 |
| VLM 固定样本 | `outputs/vlm_samples/` |
| VLM 评估 | `outputs/vlm_evaluations/` |
| probe 批次 | `outputs/probe_evaluations/` |
| 训练与 rollout | `outputs/training/`、`outputs/rollout/` |

### 3.2 覆盖策略

- 可以重建的目录也不再先 `rmtree` 后生成；先在同一父目录创建 staging。
- staging 完成全部 schema、数量和图片检查后，才执行原子替换。
- 替换时旧目录移动到同一文件系统的临时备份；新目录发布失败则恢复旧目录。
- 成功发布后才清理临时备份。
- CLI 的 `run_name` 必须是单一目录名，拒绝 `/`、`\\`、`.`、`..` 和绝对路径。
- 专家数据集是不可覆盖证据：版本化叶子目录非空时即拒绝 `clean_before_run=true`，要求
  使用新版本名；该兼容字段不再触发递归删除。只有尚未写入证据的空目录可以复用。

## 4. `delta_q_64` 动作语义

### 4.1 唯一定义

对同一 episode 中按 `step_idx` 严格递增的相邻保存帧：

```text
delta_q[t] = absolute_q[t] - absolute_q[t-1]
```

每个 episode 的第一帧没有 delta 标签，因此不进入 delta 训练样本。禁止跨 episode
做差。原始 JSONL 中的 `action[:7]` 始终保持绝对关节目标，不修改历史数据格式。

### 4.2 数据接口

`BCDataset` 加载轨迹后先按 `(episode_idx, step_idx)` 建立相邻关系。delta 模式返回
真实差值，再用只在 train episode 上拟合的 `delta_q_64` 边界编码。absolute 和
regression 路径保持原语义。

运行时：

```text
absolute_q_32: joint_target = decoded_absolute_q
delta_q_64:    joint_target = current_q + decoded_delta_q
regression:    joint_target = denormalized_absolute_q
```

tokenizer 报告中的 `reconstruction_values` 是运行时唯一解码值；不再用箱中心替代审计
中位数。checkpoint 必须保存动作表示、分箱边界、重建值、训练 split hash 和必要的
归一化统计。

旧 `bc_delta_q_64_*_v1` checkpoint 和 0% rollout 不删除，但在文档和摘要中标记为
`invalid_reason=absolute_labels_encoded_as_delta`。新训练目录使用 `*_v2`。

## 5. `expert_multi_v2` 采集契约

### 5.1 任务和相机

继续使用斜视主相机及现有随机扰动。两个积木同时存在，任务均衡选择：

```text
红块：x ∈ [-0.20, -0.05], y ∈ [0.40, 0.50]
蓝块：x ∈ [ 0.05,  0.20], y ∈ [0.40, 0.50]
```

两个范围的 X 边界相隔 10 cm；考虑积木尺寸和物理容差，采样结果还必须通过至少
12 cm 的三维中心距离门槛。采样后、加载前执行几何拒绝检查；settle 后再次检查 AABB
和中心距离。发生接触、越界或超过容差的位移时，该采集尝试失败并写入独立失败日志，
不得进入正式轨迹或占用有效 episode ID。

目标目录固定为 `outputs/dataset/expert_multi_v2/`。采集器持续尝试直到得到恰好 300 个
成功且通过场景门禁的有效 episode，红/蓝各 150 个；失败尝试保存到
`collection_failures.jsonl`。任务选择由有效 episode ID 和 seed 确定，不能依赖进程级
随机状态。

### 5.2 场景状态

`episode_summary.jsonl` 每行至少保存：

```json
{
  "schema_version": "expert_multi_v2",
  "episode_idx": 0,
  "episode_seed": 1000,
  "task_instruction": "悬停在红色积木上方",
  "target_block": "red",
  "red_block": {
    "sampled_pose": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]},
    "settled_pose": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}
  },
  "blue_block": {
    "sampled_pose": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]},
    "settled_pose": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}
  },
  "camera_eye": [0, 0, 0],
  "home_joint_positions": [0, 0, 0, 0, 0, 0, 0],
  "termination_reason": "success",
  "num_frames": 1,
  "final_distance": 0.0
}
```

轨迹帧保存 `instruction`、`target_block`、两个积木的当前位姿、相机、动作、末端位姿、
距离和终止原因。manifest 声明两项任务、两个积木配置、斜视相机分布、seed 规则、
质量阈值、schema 版本和代码版本。追加兼容性检查覆盖上述全部影响分布的字段。

### 5.3 质量门禁

正式数据集必须同时满足：

- 有效且成功的 episode 恰好 300；红任务 150、蓝任务 150；
- 正式轨迹成功率为 100%；失败采集尝试只进入独立失败日志；
- episode ID、seed、帧序号、图片和动作维度完整且唯一；
- 所有数值有限，动作和标志满足 schema；
- 每个 episode 恰好一个终止帧，终止帧与摘要一致；
- 两积木采样位姿无重叠，settle 后无接触；
- settle 位移不超过 1 cm，且仍位于所属配置范围；
- 指令、`target_block`、目标位姿和动作目标一致；
- X/Y 覆盖门禁分别对红块和蓝块计算，不用观测范围自适应掩盖分布坍缩；
- manifest、配置快照和实际记录一致；
- 质量报告的 `passed=true` 才允许生成 split 和训练格式。

划分按任务和目标位置分层，固定为 train 250、val 50，并保存 split seed、输入摘要 hash、
schema 版本、任务计数和生成器版本。

## 6. 确定性 VLA rollout

rollout 不再重新随机采样验证场景。每个验证 episode 使用摘要中的：

- episode seed；
- 红蓝块 settled pose；
- 固定 home joint positions；
- camera eye；
- task instruction 和 target block。

重建后先验证实际位姿与记录误差不超过 1 mm，再开始模型控制。重建不一致时记录
`scene_replay_mismatch` 并计为评估错误，不继续该 episode。

每条结果保存实际重建状态、checkpoint hash、dataset/split hash、模型输入指令、每步
预测、停止原因、真实最终距离和成功判定。辅助 terminate 提前停止时记录真实执行步数，
不得写成 `max_steps`。PyBullet 连接和 body 清理必须由作用域明确的 `try/finally` 管理。

## 7. 成对反事实语言评估

每个验证场景运行一对完全相同的 rollout：

```text
原指令：使用 episode 的 target_block
反事实指令：只交换 red ↔ blue，场景、相机、home pose、checkpoint 全部不变
```

每对结果同时报告：

- 原指令是否到达原目标；
- 反事实指令是否到达交换后的目标；
- 最终位置更接近红块还是蓝块；
- 指令交换是否导致目标偏好切换；
- 两次轨迹是否因系统错误而不可比较。

主要指标是有效成对样本中的 `paired_instruction_follow_rate`，分母排除场景重放失败和
系统错误，但同时单独报告排除数量。报告必须包含逐 episode JSONL、总体摘要、红→蓝和
蓝→红分组以及二项比例置信区间。没有该产物前，不得恢复“70% 语言跟随率”或“语言
信号已证明驱动行为”的结论。

## 8. 工程一致性

- 为 training 和新共享路径模块建立镜像测试目录；所有新行为遵循 RED-GREEN-REFACTOR。
- `requirements.txt` 和 `pyproject.toml` 同步声明正式源码的直接依赖。
- CLI 契约从 `pyproject.toml` 单一来源核对，不再手写过期的 16 命令说明。
- 修复模块直接执行顺序、退出码、相对图片路径和 checkpoint 安全加载。
- notebook 清除与当前源码矛盾的陈旧输出，固定依赖版本，启用显式验证策略；在新数据、
  本地确定性 VLA 和反事实结果通过前不恢复 QLoRA 训练。
- `PROJECT_OVERVIEW.md` 保存稳定有效结论，`CURRENT_STATUS.md` 保存当前恢复阶段，
  `BUGLOG.md` 保存本轮失败证据和根因，`WORKLOG.md` 保存实施与实验过程。

## 9. 验收与产物生命周期

每个子阶段先运行定向测试，再运行完整 `unittest discover`、`compileall`、CLI 冒烟和
`git diff --check`。涉及数据的阶段额外运行结构化文件、图片、hash 和质量门禁验证。

新实验的有效顺序为：

```text
安全代码全绿
→ delta 语义测试全绿
→ multi_v2 采集器与门禁全绿
→ 300 条真实采集通过
→ split 与训练格式冻结
→ delta/VLA 重训
→ 确定性 rollout
→ 成对反事实评估
→ 更新有效结论
→ Qwen2-VL QLoRA
```

任何一步失败时保留失败证据，不继续消费该产物。旧目录永不自动删除，新目录只有在完整
验证成功后才能被下游引用。
