# 多任务 VLA 数据采集与训练设计

**日期：** 2026-08-07
**状态：** 待实施

## 目标

采集两任务数据集并训练 VLA 模型，验证语言信号被模型实际使用。

## 1. 任务设计

两块积木同时在场，空间分离：

```
红色积木:  X ∈ [-0.2, -0.05],  Y ∈ [0.40, 0.50]
蓝色积木:  X ∈ [0.05, 0.2],    Y ∈ [0.40, 0.50]
```

| 指令 | 目标 | 积木色 |
|------|------|--------|
| `"悬停在红色积木上方"` | 左边 | 红 |
| `"悬停在蓝色积木上方"` | 右边 | 蓝 |

**关键：** 两块积木同时渲染，图片包含两个目标。指令是唯一区分信号。
模型只看图无法判断去左边还是右边——必须理解语言。

## 2. 数据采集

### 修改 sim_config.yaml

```yaml
task:
  # 保留原单一积木配置做向后兼容
  block_urdf_path: "cube.urdf"
  block_position:
    x_range: [-0.2, 0.2]
    y_range: [0.40, 0.50]
    z: 0.1
  block_color_rgba: [1, 0, 0, 1]  # 红
  
  # 新增第二块积木
  second_block:
    enabled: true
    color_rgba: [0, 0, 1, 1]  # 蓝
    x_range: [0.05, 0.2]
    y_range: [0.40, 0.50]
    z: 0.1
  
  # 新增任务列表
  tasks:
    - instruction: "悬停在红色积木上方"
      target_block: "red"
      hover_height: 0.15
      success_distance: 0.03
    - instruction: "悬停在蓝色积木上方"
      target_block: "blue"
      hover_height: 0.15
      success_distance: 0.03

dataset:
  output_dir: "outputs/dataset/expert_multi_v1"
  num_episodes: 300          # 每任务 ~150
  task_selection: "random"   # 每个 episode 随机选任务
```

### 采集逻辑变更

```
每 episode:
  1. 随机选任务（50/50 概率）
  2. 加载两块积木
  3. 根据任务选 target_block
  4. IK 规划到 target_block 上方 hover_height
  5. 记录 instruction 到 trajectory JSONL
```

### 数据集产出

```
outputs/dataset/expert_multi_v1/
  trajectory_expert.jsonl    # 每帧含 instruction 字段
  episode_summary.jsonl      # 每 episode 含 task + target_block
  episode_split.json         # train 250 / val 50（按任务分层）
  action_tokenization_audit_v1/
  images/                    # JPEG
```

## 3. VLA 模型

### 架构

```
图片 (224×224) ──→ ResNet-18 ──→ 512d ──┐
                                          ├──→ concat(896d) ──→ MLP ──→ Action Head
指令文本 ──────→ MiniLM-L6 ──→ 384d ──┘
```

### 文本编码器

`all-MiniLM-L6-v2`（sentence-transformers）：
- 参数量：22M
- 嵌入维度：384
- 优点：轻量、CPU 友好、已预训练
- 推理时冻结即可（不微调）

### 训练

- 复用现有训练循环（train.py）
- 新文件：`src/vla_project/training/vla_model.py`（VLAModel 类）
- 修改 dataset：`VLADataset` 返回 `(image, instruction_text, action)`
- 动作表示：regression（已证明最优）
- 超参：同 BC（AdamW lr=1e-4, CosineAnnealing, 50 epoch）

### 验证策略

1. Overfit 10 episode：确认 loss 收敛
2. 全量训练 250 episode
3. Rollout：用两种指令分别评估，确认模型根据指令选对了目标

## 4. 实施步骤

| 步骤 | 内容 | 谁 |
|------|------|-----|
| 1 | 修改 sim_config + control_arm 支持双积木 | Luna |
| 2 | 修改采集逻辑支持任务随机 | Luna |
| 3 | 采集 300 episode 多任务数据 | Luna |
| 4 | 实现 VLAModel + VLADataset | Luna |
| 5 | VLA 训练 overfit + 全量 | Luna |
| 6 | Rollout 评估（按指令验证） | Luna |
