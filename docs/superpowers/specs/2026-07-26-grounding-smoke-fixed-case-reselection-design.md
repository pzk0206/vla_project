# Grounding Smoke 固定案例重选设计

**日期：** 2026-07-26  
**状态：** 已获用户批准  
**范围：** 重选固定 smoke 案例并增加无 API 的只预检入口；不修改在线闭环算法

## 背景

首轮固定案例为 `52-left`、`53-right` 和 `54-front`。整批无 API 预检证明其中只有
`53-right` 合格：

- `52-left` 的首帧红块可见率为 `0.714`，低于 `0.75`；
- `54-front` 的起点姿态误差为 `0.053522m`，高于 `0.005m`；
- `53-right` 能进入闭环，并通过一次 fresh grounding 和四步 held 将真实 XY 距离从
  约 `10.05cm` 降到 `0.41cm`。

现有动态筛选要求候选在起点和四次移动后的五个观察点全部保持 clear，并要求每次动作
姿态误差合格。seeds 55–100 与三个方向组成的138个候选中没有任何候选满足该条件。
这个门槛与当前在线闭环设计不一致：闭环已经允许在首次可靠定位后最多连续四步复用 held
目标，因此固定案例只需在开始付费定位前满足起点姿态和首帧可见率。

## 目标与非目标

目标：

1. 在 seeds 55–100 中为 `left`、`right`、`front` 各选择一个固定案例。
2. 资格只由起点姿态误差和首帧红块可见率决定。
3. 选择规则确定、可重复，并保存全部候选和最终选择证据。
4. 更新固定案例后，以显式只预检模式验证三个案例全部合格。
5. 只预检模式在任何情况下都不构建在线控制依赖或调用 VLM API。

非目标：

- 不修改 grounding、相机反投影、冻结 XY 补偿或工作区校验。
- 不修改动作方向计算、IK 控制、目标保持、成功评分或 API 上限。
- 不要求移动后的观察帧保持 clear。
- 不运行真实在线 smoke；真实 API 实验仍需要用户另行明确批准。
- 不覆盖或重写历史动态筛选输出。

## 方案选择

采用“将现有 screening 修订为起点资格筛选，并为 smoke runner 增加
`--preflight-only`”的方案。

没有保留两套并行筛选模式，因为旧的全轨迹 clear 资格已经被138个候选的真实结果证明
不适合作为当前闭环的运行前提。历史结果继续由既有输出和文档保存，不需要在正式代码中
长期维护一套不再使用的选择规则。

没有采用一次性脚本或手工查看图片选择 seed，因为这种方式不能保证选择可重复，也无法
证明案例没有根据在线 smoke 结果被临时替换。

## 候选资格规则

候选空间固定为配置中的：

```text
seed_range = [55, 100]
directions = [left, right, front]
```

每个 `seed + direction` 候选独立建立 PyBullet 场景，并执行以下步骤：

1. 用 seed 初始化随机状态，创建场景并稳定红块。
2. 用红块仿真真值构造相距 `10cm` 的固定末端起点。
3. 将机械臂重置到该起点并记录请求位置、实际位置和三维姿态误差。
4. 使用正式 smoke 的448px固定俯视相机捕获首帧 RGB 与 segmentation。
5. 计算红块可见像素和相对 `visibility_reference_pixels` 的可见率。
6. 保存首帧图片与结构化候选记录，然后释放 physics 连接。

资格采用包含性边界：

```text
start_pose_error <= 0.005m
block_visibility_ratio >= 0.75
```

拒绝原因保持稳定：

- `start_pose_error`：起点误差严格大于 `0.005m`；
- `visibility_below_threshold`：起点合格，但首帧可见率严格小于 `0.75`；
- `candidate_error`：场景、控制或图像捕获发生非预期异常；
- `qualified`：两个资格条件均满足。

若同一候选同时存在起点误差和低可见率，优先记录 `start_pose_error`，但仍保存两项原始
指标，避免拒绝原因隐藏证据。

## 确定性选择

候选全部评估完成后，按方向顺序 `left → right → front` 选择。每个方向选择 seed 最小
的合格候选，并要求三个方向的 seed 互不重复。已经用于前一个方向的 seed 不得用于后续
方向。

如果任一方向没有可选候选：

- 不产生部分固定案例配置；
- `selected_cases` 为空；
- 摘要记录明确的 `selection_error`；
- 筛选命令以失败状态结束。

筛选不得读取真实在线 smoke 的 episode 结果，也不得根据 VLM 表现改变候选排序。

## 输出证据

筛选继续使用不可覆盖的独立运行目录：

```text
outputs/vlm_evaluations/grounding_world_smoke_screening/<run_name>/
```

至少保存：

- `candidate_trace.jsonl`：138个候选的完整指标、资格和拒绝原因；
- `seed_NNN_direction/step_00.jpg`：每个成功捕获候选的首帧；
- `screening_summary.json`：候选数量、资格数量、原因计数、冻结规则和最终选择。

`screening_summary.json` 的最终选择使用：

```json
{
  "selected_cases": [
    {"seed": 0, "direction": "left"},
    {"seed": 0, "direction": "right"},
    {"seed": 0, "direction": "front"}
  ]
}
```

示例中的 `0` 不是实际值。实际 seed 只能来自真实无 API 筛选输出，不能在实现或规格中
预先猜测。

筛选成功后，将三个选中 seed 按相同方向顺序写入 `sim_config.yaml` 的
`grounding_smoke.seeds`。`start_directions` 继续固定为
`["left", "right", "front"]`。

## 只预检模式

现有 `preflight_smoke_case()` 已能验证单个固定案例的起点姿态和首帧可见率。runner
增加批次级预检函数，将以下现有编排从在线批次中提取为可复用边界：

1. 构造三个固定案例。
2. 对全部案例执行 `preflight_smoke_case()`。
3. 写入 `<run_dir>/smoke_preflight.json`。
4. 返回预检摘要；任意案例不合格时仍保留全部三条记录。

CLI 增加显式 `--preflight-only` 参数。启用后：

- 只运行上述批次预检；
- 合格时正常退出，不继续进入 episode 循环；
- 不合格时以明确错误退出；
- 不加载或调用在线 grounding API；
- 不创建 episode 目录、`episode_summary.jsonl` 或 `smoke_summary.json`。

普通 smoke 模式继续先调用同一批次预检函数，只有预检全部合格才进入原有在线循环。
这避免只预检入口和真实运行入口产生两套资格实现。

## 真值与安全边界

- PyBullet 红块真值只用于构造固定10cm起点和计算预检资格。
- segmentation 只用于可见率门禁，不进入 grounding 坐标或动作决策。
- 候选筛选和只预检均不调用 `call_openai_compatible_api()`。
- `--preflight-only` 不依赖 API 环境变量是否存在；其无 API 性质由控制流和自动测试
  保证，不能只依赖删除密钥。
- 所有候选和固定案例均使用独立 physics 连接，并在 `finally` 中释放。
- 输出目录拒绝覆盖已有运行，防止新结果污染历史证据。

## 错误处理

- 单个候选的非预期异常转为 `candidate_error` 并继续筛选其他候选。
- 筛选结束后若无法为三个方向选出互不重复的合格 seed，写完摘要后返回失败。
- 固定案例只预检中的非预期异常保持原始异常并释放连接，不伪装成资格失败。
- 固定案例资格不合格时，先写完整 `smoke_preflight.json`，再抛
  `SmokePreflightError`。
- 只预检成功不得触发控制循环；这是安全契约，不是调用方约定。

## 测试策略

使用 TDD 修改 `tests/vlm/grounding_smoke/`：

1. 起点误差等于 `0.005m` 时允许，严格超过时拒绝。
2. 首帧可见率等于 `0.75` 时允许，严格低于时拒绝。
3. 候选只捕获一个观察点，不执行方向动作或 IK 动作控制。
4. 单候选异常产生 `candidate_error`，并释放 physics 连接。
5. 完整筛选评估 seeds 55–100 与三个方向的笛卡尔积。
6. 选择结果按方向顺序、最小 seed、seed 互不重复。
7. 缺少任一方向时不输出部分选择。
8. 只预检模式写三个案例的 JSON，并且控制循环、依赖构造和 API 调用次数均为0。
9. 普通 smoke 模式复用同一个批次预检函数，通过后保持现有行为。
10. 配置契约保护 seed 范围、方向顺序和资格阈值。

定向测试通过后运行 grounding smoke 全部测试、配置与包元数据契约、全量测试、
`compileall` 和 `git diff --check`。

## 执行顺序与验收标准

1. 先用测试驱动完成起点资格筛选和证据格式修改。
2. 运行真实 PyBullet、无 API 的 seeds 55–100 候选筛选。
3. 从摘要读取确定性选出的三个案例，不手工挑选。
4. 更新 `sim_config.yaml` 的固定 seeds。
5. 用 `--preflight-only` 运行三个固定案例。
6. 更新 `PROJECT_OVERVIEW.md`、`CURRENT_STATUS.md` 和 `WORKLOG.md` 中的当前结论。

最终验收要求：

- 三个方向各有一个 seed 互不重复的固定案例；
- 固定案例来源可由筛选 trace 和摘要复现；
- `smoke_preflight.json` 报告 `qualified_count=3` 和 `passed=true`；
- 筛选与只预检的真实 API 调用数均为0；
- 没有运行真实在线 smoke；
- 所有相关自动测试和仓库验证命令通过。
