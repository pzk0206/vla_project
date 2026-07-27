# 专家数据规模化验收门禁设计

## 背景

`expert_v1` 的10条真实 pilot 已通过严格门禁，下一阶段需要以追加模式再采集290条，
使数据集达到至少300条。当前 `evaluate_dataset.py` 只实现了 pilot 门禁，并强制要求
episode 数量恰好为10；因此规模化数据即使质量合格，CLI 也会以非零状态退出。

本次工作只补齐规模化验收语义和追加配置，不改变数据 schema、动作维度、seed 规则、
仿真控制、grounding 或 smoke 行为。

## 方案比较

### 方案 A：在同一报告中保留两个门禁，并自动选择当前阶段（采用）

报告同时包含 `pilot_gate` 和 `scale_gate`。CLI 根据 episode 数量选择当前适用门禁：
未达到规模目标时仍执行 pilot gate；达到规模目标时执行 scale gate。

优点是保留历史报告字段和单一命令入口，规模化后不会被 pilot 的“恰好10条”规则误伤。
缺点是 CLI 需要明确记录当前选择的门禁，避免读者只看到两个结果时混淆。

### 方案 B：新增独立规模化评估 CLI

新增类似 `vla-evaluate-scaled-dataset` 的命令。边界清晰，但会产生重复入口、元数据和
文档维护成本，而两种评估共享完全相同的扫描结果。

### 方案 C：直接把 pilot gate 改成规模化 gate

实现最少，但会失去10条 pilot 的严格契约并改变历史语义，不采用。

## 门禁语义

`pilot_gate` 保持现状：

- `num_episodes == pilot_num_episodes`
- `success_count == pilot_num_episodes`
- 所有完整性错误计数为0

新增 `scale_gate`：

- `valid_episode_count >= target_num_episodes`
- `success_rate >= 0.99`
- schema、动作维度、缺失/损坏/尺寸错误图片、孤儿图片、重复 step、seed、帧数和终止
  标志错误计数均为0
- 红块 X 和 Y 的五个分箱均非空

覆盖检查使用现有 `block_position.x_bin_counts` 和 `y_bin_counts`。分箱数组必须恰好有
五项且每项大于0；缺失、长度错误或空箱都使规模化门禁失败。

报告新增：

```text
scale_gate
active_gate: "pilot" | "scale"
passed
```

`passed` 等于当前 `active_gate` 的结果。CLI 只依据顶层 `passed` 决定退出码，同时在
终端输出当前门禁名称。已有 `pilot_gate` 字段保持不变，避免破坏现有消费者。

阶段选择规则：

- `num_episodes >= target_num_episodes`：`active_gate = "scale"`
- 否则：`active_gate = "pilot"`

因此10条 pilot 仍按原规则验收，介于11和299条的中间数据会明确失败，不会被误认为
已经完成规模化验收。

## 追加采集配置

pilot 已经存在，配置改为：

```yaml
clean_before_run: false
num_episodes: 290
```

采集器通过现有摘要计算下一个 episode index，因此从 episode 10 继续写到299，seed
为1010至1299。`schema_version`、`random_seed`、`pilot_num_episodes`、
`target_num_episodes` 和输出目录保持不变。

## 测试与验证

先增加失败测试，再实现最小代码：

1. 合格的300条报告通过 scale gate。
2. 有效 episode 不足、成功率低于99%、任一完整性错误或任一轴存在空箱时分别失败。
3. 10条报告选择 pilot gate；300条报告选择 scale gate；11至299条不会通过。
4. 配置契约要求 `clean_before_run == false` 且 `num_episodes == 290`。
5. 运行定向测试、完整测试、`compileall` 和 `git diff --check`。

真实采集开始前先只读核对现有数据仍为10条且 pilot 报告通过。采集完成后运行同一质量
评估 CLI；只有顶层 `passed=true` 时才进入 action tokenization。
