# 最小行为克隆训练管线 — 设计

**日期：** 2026-08-06
**状态：** 待实施
**依赖：** [视觉输入策略](2026-08-06-visual-input-strategy-design.md)、[episode 划分 + tokenizer 重拟合](2026-08-06-episode-split-tokenizer-refit-design.md)、[action tokenization 审计](2026-07-31-action-tokenization-audit-design.md)

## 目标

搭建最小 BC 训练管线，验证"从 224×224 RGB 预测专家动作"是否可学习。
不追求 SOTA 性能，只追求管线正确、可复现、可对比。

## 模型架构

```
224×224×3 RGB ──► [ResNet-18 (ImageNet pretrained)] ──► 512-d feature
                                                           │
                              ┌────────────────────────────┤
                              ▼                            ▼
                    [Action Head]                  [Aux Head]
                    7 joint angles            gripper + terminate
```

- **Backbone**：`torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`，去掉最后的 fc 层，输出 512-d。
- **Action Head**：根据实验组不同，输出不同：
  - **连续回归**：512 → 256 → 7（L1/L2 loss）
  - **32 箱 absolute_q 分类**：512 → 256 → 7×32（reshape 为 [7, 32]，每关节独立 softmax，交叉熵）
  - **64 箱 delta_q 分类**：512 → 256 → 7×64（同上）
- **Aux Head**：512 → 128 → 2（gripper: sigmoid + BCE, terminate: sigmoid + BCE）
- **不冻结 backbone**：从头微调全部参数。8,200 帧足够微调 ResNet-18。

## 动作表示三种对照组

| 组 | 输出 | Loss | Tokenizer 边界来源 |
|---|---|---|---|
| `regression` | 7 维浮点（absolute_q 连续值） | SmoothL1 | 不需要。按训练集 z-score 归一化。 |
| `absolute_q_32` | 7×32 logits → 每关节 32 类 | CrossEntropy（每关节独立，sum） | `action_tokenization_audit_v2` 的 32 箱等频绝对边界 |
| `delta_q_64` | 7×64 logits → 每关节 64 类 | CrossEntropy（每关节独立，sum） | `action_tokenization_audit_v2` 的 64 箱等频差分边界 |

三组共享同一个 backbone，只换 head 和 loss。每组独立训练，独立 checkpoint。

### 为什么用 absolute_q 作为回归目标而不是 delta_q

回归天然适合连续值。absolute_q（关节绝对角度）的范围由关节限位定义，
归一化后范围稳定。delta_q 也可以回归，但作为第一个基线，用 simpler target
（"当前状态 → 目标关节角度"）更直觉。后续可以把 delta_q 回归作为第四个对照组。

## 数据流

```
episode_split.json ──► train_episodes = [0, 2, 5, ...]
                              │
trajectory_expert.jsonl ──► 过滤 episode_idx in train_episodes
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              image_path            action[0:7]
                    │              q_target / delta_q
                    ▼
              224×224 JPEG ──► DataLoader
              (normalize with
               ImageNet stats)
```

- **输入**：RGB 224×224，ImageNet 归一化（mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]）
- **目标（regression）**：`q_target = action[0:7]`，按训练集做 z-score 归一化（`(x - mean) / std`）
- **目标（分类）**：`token_id[joint] = searchsorted(edges[joint], q_target[joint])`，每关节独立
- **delta_q 的 target 帧**：只取 `delta_q is not None` 的 transition（排除每 episode 首帧）
- **Aux 目标**：`gripper = action[7]`（始终 1.0），`terminate = action[8]`（3% 为正）

## 训练配置

| 参数 | 值 | 理由 |
|---|---|---|
| Optimizer | AdamW(lr=1e-4, weight_decay=1e-4) | 标准微调配置 |
| Batch size | 32 | ResNet-18 + 单 GPU |
| Epochs（overfit） | 200 | 10 episode 小样本，确认 loss → 0 |
| Epochs（full） | 50 | 250 episode，early stop on val loss |
| LR schedule | CosineAnnealingWarmRestarts(T_0=10) | 周期性重启帮助跳出局部最优 |
| 数据增强 | RandomHorizontalFlip(p=0.5) + ColorJitter(0.1, 0.1, 0.1, 0.05) | 轻微增强，防止 250 episode 过拟合 |
| 验证 | 每个 epoch 在 val 50 episode 上评估 loss | 不跑 PyBullet rollout（训练阶段分离） |
| GPU | CUDA if available, else CPU | 224×224 ResNet-18 在 CPU 上也跑得动 |

## Overfit 验证（先于正式训练）

选前 10 个训练 episode（约 330 帧），用三组分别训练 200 epoch，验证：
1. Regression 组 L1 loss < 0.01（归一化后）
2. 分类组 accuracy > 95%
3. Gripper BCE loss < 0.01（恒为 1.0，模型应快速学会）
4. 无 NaN/Inf

Overfit 不通过的组不进入正式训练。

## CLI

```bash
# Overfit 验证
vla-train-bc --action-representation regression --overfit 10 --epochs 200

# 正式训练
vla-train-bc --action-representation absolute_q_32 --epochs 50

# 指定输出目录
vla-train-bc --action-representation delta_q_64 --output-dir outputs/training/bc_delta_q_64_v1
```

## 输出

```
outputs/training/bc_{action_representation}_v1/
├── config.yaml           # 训练配置快照
├── train_loss.jsonl      # 每 epoch 的训练 loss
├── val_loss.jsonl        # 每 epoch 的验证 loss
├── checkpoint_best.pt    # 最佳验证 loss 的权重
├── checkpoint_last.pt    # 最后一个 epoch 的权重
└── tokenizer.pt          # 分类组的 tokenizer 边界（regression 组不需要）
```

## 验收

- Overfit 三组分别 loss 收敛到接近 0
- 正式训练三组都在 val 上 loss 不上升（不要求 rollout，rollout 是下一步）
- compileall + pytest 通过

## 不在本轮范围

- PyBullet rollout 评估（下一步）
- 多视角训练（俯视留给后续对照）
- 连续 delta_q 回归对照组
- 可见性筛选训练（保留全部数据，不删除 partial/severe）
- 多 GPU / 分布式训练
- TensorBoard / Wandb（先用 JSONL 记录）
