# Episode 级训练/验证划分 + Tokenizer 边界重拟合 — 设计

**日期：** 2026-08-06
**状态：** 待实施
**依赖：** [视觉输入策略决策](2026-08-06-visual-input-strategy-design.md)、[action tokenization 审计](2026-07-31-action-tokenization-audit-design.md)

## 背景

全量 300 episode 的 action tokenization 审计推荐了 32 箱等频 absolute_q 和 64 箱等频
delta_q，但分位数边界是在全部数据上拟合的。验证 episode 的 action 分布已经泄漏到这些
边界中。必须先固定 episode 级划分，再用训练集重拟合 tokenizer 边界。

300 episode 全部成功（`termination_reason=success`），`final_distance` 分布极窄
（min 2.87cm / max 3.00cm），因此分层不依赖 success/failure 或距离。

## 划分参数

| 参数 | 值 | 理由 |
|---|---|---|
| 训练集 | 250 episode（约 83%） | 学习计划 250/50；足够支撑 32/64 箱的最小箱门槛（每箱 ≥20 帧） |
| 验证集 | 50 episode（约 17%） | 足够评估 BC 策略在未见 episode 上的表现 |
| 分层键 | `initial_block_pos` X 和 Y（各 5 等频箱） | X/Y 位置是唯一有实质方差且影响策略难度的 episode 级字段 |
| 随机种子 | 固定（如 42） | 可复现 |

## 不用于分层的字段

| 字段 | 原因 |
|---|---|
| `final_distance` | 分布极窄（2.87–3.00cm），无区分度 |
| `termination_reason` | 全部为 `success`，无区分度 |
| `num_frames` | 29–36，方差小，且主要由 block 位置间接决定 |
| `initial_ee_pos` | 恒定 |

## 分层方法

1. 读取 300 条 `episode_summary.jsonl`，提取 `(episode_idx, initial_block_pos.x, initial_block_pos.y)`
2. 对 X 和 Y 分别计算 5 等频分位数边界（与 `scale_gate` 的 `x_bin_counts`/`y_bin_counts` 保持同构）
3. 每个 episode 分配到一个 (X_bin, Y_bin) cell
4. 对每个 cell，按 250:50 比例随机分配；cell 内 episode 数 ≥6 时保证验证集至少 1 个
5. 全局调整确保恰好 250/50
6. 验证：train 和 val 两个集合的 X 和 Y 5 箱均非空
7. 输出 `episode_split.json`：`{"train": [ep_idx, ...], "val": [ep_idx, ...], "split_seed": 42, ...}`

## Tokenizer 边界重拟合

在 [audit_action_tokenization.py](file:///home/pzk/vla_project/src/vla_project/simulation/audit_action_tokenization.py)
增加 `--episode-ids-file` 参数，指向划分 JSON。当提供时：

- 只从训练集 episode 的 transition 中拟合分箱边界
- 验证集 episode 的 transition 使用训练集边界做误差评估（报告但不参与拟合）
- 输出报告新增 `split` section：训练集和验证集各自的误差统计
- 连续回归不受影响（无离散边界）

## 对下游的影响

- 新的 split JSON 是后续 BC 训练 DataLoader 的输入
- 新的 tokenizer 边界 (`action_tokenization_audit_v2/`) 是训练时的离散动作目标
- 训练时：`absolute_q` 目标 = 32 类分类头；`delta_q` 目标 = 64 类分类头
- `sim_config.yaml` 不改动
