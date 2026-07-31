# Expert v1 Action Tokenization 只读审计设计

## 背景

`expert_v1` 已完成300条真实 PyBullet 专家轨迹，得到300/300 success、9,894帧、
0项完整性错误，并通过 scale gate。逐帧可见性审计也已通过，clear、partial、severe
分别为9,479、411、4帧。下一阶段需要决定这批数据中的动作应采用连续回归、绝对关节
目标分箱还是相邻关节目标变化量分箱，但目前还没有动作分布、量化误差和可见性条件分布
的证据。

当前每帧动作固定为9维：前7维是 IK 计算出的绝对关节目标角，第8维是夹爪占位状态，
第9维是 episode 终止标志。本阶段只读取冻结数据并模拟候选表示，不生成训练 token，
不划分训练/验证集，不修改数据集 schema、manifest、轨迹、图片或质量报告。

## 目标与非目标

本阶段目标：

- 验证轨迹动作与逐帧可见性标签可以一一对齐；
- 统计7维绝对关节目标、相邻保存帧目标差分和单位 step 差分的联合分布；
- 单独审计夹爪占位值与终止标志；
- 比较不同分箱数和分箱方法的数据支持度与离线重建误差；
- 用透明、版本化规则推荐下一轮最小训练对照组；
- 保存可复算的逐帧派生值、候选边界、指标和推荐理由。

本阶段不负责：

- 生成离散 token 或 tokenized dataset；
- 宣称离线最优候选等于训练或闭环最优候选；
- 修改 `expert_v1` 动作定义或采集 schema；
- 删除 partial/severe 帧或根据可见性重新采样；
- 训练模型、运行 rollout 或重新优化 grounding/autonomous stop。

## 方案比较

### 方案 A：独立只读审计模块（采用）

新增独立的 simulation 数据审计模块，读取冻结轨迹和已有可见性标签，在数据集内的独立
输出目录发布报告。动作表示实验与现有 scale gate 保持隔离，后续 tokenizer 可以复用
其中的纯统计和分箱模拟函数。

优点是职责清楚、可自动测试、可复现，不会改变已经通过的数据质量门禁。代价是新增一个
命令入口和一种报告 schema，但这正好形成明确的阶段证据。

### 方案 B：扩展 `evaluate_dataset.py`

把动作分布和分箱模拟写入现有 `dataset_quality_report.json`。文件更少，但会把冻结数据
完整性门禁与持续变化的训练表示实验耦合，可能使历史 scale gate 语义漂移，因此不采用。

### 方案 C：临时 notebook 或一次性脚本

能快速得到局部统计，但缺少测试、稳定输入输出和命令入口，难以复现，也不符合正式代码
只能进入 `src/vla_project/` 的项目规则，因此不采用。

## 输入与权威来源

输入固定为：

```text
outputs/dataset/expert_scaling_v1/
├── dataset_manifest.json
├── trajectory_expert.jsonl
└── visibility_audit_v1/
    └── frame_visibility.jsonl
```

规则：

- manifest 决定 schema 和 `action_dim`，不得用当前 `sim_config.yaml` 覆盖冻结数据语义；
- 轨迹行提供 episode、step、9维动作和终止原因；
- 可见性行只提供审计标签和对应键，不反向改写轨迹；
- 两份逐帧文件必须通过 `(episode_idx, step_idx)` 完整一一对应；
- 报告记录输入文件的大小和 SHA-256，以便确认重复运行使用了同一份证据。

## 输入硬门禁

以下任一情况发生时，审计失败：

- manifest 或任一 JSONL 缺失、损坏或存在无法解析的行；
- manifest 不是 `expert_v1` 或动作维数不是9；
- episode/step 键重复、缺失、多出或两份文件无法一一对应；
- episode 内各行按轨迹文件原始顺序观察时，保存 step 不严格递增；
- action 不是9维有限数值；
- 终止标志不是布尔值或数值 `0/1`；
- 同一 episode 的终止标志不只出现在最后一个保存帧；
- visibility group 不属于 clear、partial、severe。

失败时在数据集根目录写入 `action_tokenization_audit_failure.json`，包含错误类型、输入
路径和尽可能具体的 episode/step/行号。失败运行不得发布新的成功 JSONL 或成功 summary，
也不得删除或替换上一次完整成功输出；调用者必须根据非零退出码和输入 SHA-256 区分本次
结果。原始数据始终保持不变。成功运行发布新报告后删除同 schema 的过期失败证据。

## 逐帧派生定义

先按轨迹文件原始顺序验证每条 episode 的 `step_idx` 严格递增，再按该顺序计算派生值。
每个保存帧保留：

```text
q_target(t) = action(t)[0:7]
gripper(t)  = action(t)[7]
terminate(t) = action(t)[8]
```

除首帧外计算：

```text
step_gap(t) = step_idx(t) - step_idx(t-1)
delta_q(t) = q_target(t) - q_target(t-1)
delta_q_per_step(t) = delta_q(t) / step_gap(t)
```

`delta_q` 只表示相邻保存帧的 IK 目标差，不称为实际关节位移。第一帧没有前驱，对应三个
派生字段写为 `null`，且不进入差分统计。`step_gap` 必须为正；工具同时报告其频数，避免
把常规采样间隔和终止前较短间隔混为一谈。

## 统计契约

对7个关节的 `q_target`、`delta_q` 和 `delta_q_per_step` 分别输出：

- count、min、max、mean、总体标准差（`ddof=0`）；
- 使用线性插值经验分位数计算的 p01、p05、p25、p50、p75、p95、p99；
- 非有限值计数，成功报告中必须为0。

同时输出：

- 夹爪值的唯一值、频数和是否恒定；
- 终止标志的频数、比例以及每条 episode 的终止帧数量；
- `step_gap` 的唯一值与频数；
- clear、partial、severe 下的样本数以及上述动作统计；
- 各可见性组的 episode 覆盖数。

severe 当前只有4帧，只发布描述统计，并在报告中标记
`insufficient_for_generalization=true`，不得据此推荐删除或重采数据。

## 候选分箱模拟

仅对 `q_target` 和原始 `delta_q` 模拟离散候选；`delta_q_per_step` 只用于诊断。候选
笛卡尔积固定为：

```text
representation: absolute_q | delta_q
binning: uniform_width | quantile
num_bins: 16 | 32 | 64
```

每个关节独立拟合边界：

- `uniform_width` 使用该维观测 min/max 等宽切分，使用箱中心重建；
- `quantile` 使用经验分位数切分，使用每箱样本中位数重建；
- 重复分位数边界必须合并，并记录请求箱数与实际有效箱数，不能制造空宽度箱；
- 非空箱占用率的分母始终是请求箱数；等频重复边界导致的有效箱减少会被占用率门槛捕获；
- 所有边界、每箱重建值和计数都写入报告；
- 本阶段使用全量数据只做可行性审计。正式 tokenizer 必须在 episode 级划分后只用训练集
  拟合边界，不能直接把本报告边界当成最终训练资产。

每个候选、每个关节输出：

- 请求箱数、有效箱数、非空箱占用率；
- 每个非空箱的样本数、最小样本数和最大类别占比；
- 归一化类别熵 `-sum(p * log(p)) / log(requested_num_bins)`；请求箱数大于1是固定前提；
- 重建 MAE、p95 absolute error 和 max absolute error；
- 用该维 `p99-p01` 归一化后的 MAE 和 p95 error；
- 稳健范围为0时的明确不可评估原因。

候选汇总采用7个关节中的最坏值，不用跨关节平均掩盖单维失败。

## 自动推荐规则

离线审计不能证明训练或闭环中的最终优劣，因此报告不宣布唯一“最佳 tokenizer”，而是
推荐下一轮最小训练对照组：

1. 连续回归作为无量化基线，始终保留；
2. 从 absolute_q 候选中选择一个最小合格候选；
3. 从 delta_q 候选中选择一个最小合格候选；
4. 某类没有候选合格时明确返回 `no_eligible_candidate`，不强行推荐。

一个候选只有在7个关节全部满足以下默认门槛时才合格：

```text
nonempty_bin_occupancy >= 0.90
minimum_nonempty_bin_count >= 20
normalized_p95_reconstruction_error <= 0.05
```

选择顺序固定为：箱数更少优先；箱数相同时最坏关节归一化 p95 误差更低者优先；仍相同
时最小箱样本数更多者优先；再相同时优先 `uniform_width`，因为解码规则更简单。门槛、
候选通过状态、拒绝原因和完整排序都必须进入报告。

这些门槛只判断当前数据是否统计上足以支撑候选，不等价于机械臂任务成功门槛。后续计划
必须通过真实训练和 episode 级 rollout 决定最终动作表示。

## 输出契约

成功输出目录：

```text
outputs/dataset/expert_scaling_v1/action_tokenization_audit_v1/
├── frame_action_analysis.jsonl
└── action_tokenization_audit.json
```

`frame_action_analysis.jsonl` 与原轨迹逐帧对应，至少包含：

```json
{
  "schema_version": "action_tokenization_audit_v1",
  "episode_idx": 0,
  "step_idx": 24,
  "visibility_group": "clear",
  "q_target": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "gripper": 1.0,
  "terminate": 0,
  "step_gap": 24,
  "delta_q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "delta_q_per_step": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

示例数值只说明字段形状，不代表真实数据。该文件不包含 bin id 或训练 token。

`action_tokenization_audit.json` 至少包含：

- schema、创建时间、数据集路径、输入文件大小和 SHA-256；
- episode/frame/transition 数量和输入门禁结果；
- 全局与可见性分组的动作、差分、夹爪、终止和 step gap 统计；
- 12个离散候选的边界、重建值、逐关节指标和最坏关节汇总；
- 推荐规则版本、门槛、完整候选排序、选择结果和拒绝原因；
- 连续回归、绝对分箱与差分分箱的限制说明；
- `passed=true`，只表示审计完整执行并发布，不表示任一 tokenizer 已通过训练验证。

成功结果先写入同一父目录下的临时目录，两个文件全部完成并校验后再原子发布，避免中断
留下半份可信报告。重复运行允许完整替换同 schema 的旧生成结果，但不得触碰输入文件。

## 代码位置与命令

正式实现放入：

```text
src/vla_project/simulation/audit_action_tokenization.py
```

测试放入：

```text
tests/simulation/test_audit_action_tokenization.py
```

注册命令：

```text
vla-audit-action-tokenization
```

这是单个数据集离线审计模块，不建立新的源码子包。新增命令后同步更新包元数据契约、
`PROJECT_STRUCTURE.md`、README 命令说明和当前状态中关于命令数量及阶段的表述。

## 测试策略

实现采用 TDD，至少覆盖：

- 两份 JSONL 的 episode/step 完整对齐、缺失、多出和重复键；
- 9维动作、有限数值、step 单调和终止位置硬门禁；
- 首帧 null 差分、跨 episode 不计算差分、非固定 step gap；
- 绝对动作、差分、单位 step 差分和分位数统计；
- 夹爪恒定性、终止比例及 clear/partial/severe 分组；
- 等宽分箱边界、箱中心重建和误差；
- 等频分箱的重复边界合并、中位数重建和有效箱数；
- 候选最坏关节汇总、通过/拒绝原因和确定性排序；
- 没有合格离散候选时仍保留连续回归基线；
- 失败时不发布成功文件，成功时原子发布两个文件；
- CLI、命令入口和包元数据契约。

定向测试使用临时小型 JSONL fixture，不依赖9,894张真实图片。实现完成后运行定向测试、
完整测试、`compileall` 和 `git diff --check`，再对冻结的300条真实数据运行一次全量只读
审计。只有真实报告通过输入门禁，才能更新当前状态和工作日志，并据此设计 tokenizer 与
episode 级训练/验证划分。
