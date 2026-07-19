# Grounding Smoke 成功评分拆分设计

**日期：** 2026-07-19  
**状态：** 已获用户批准，待书面规格复核  
**范围：** 拆分任务到达、自主停止和批次合格成功；不修改控制循环的动作与终止决策

## 背景

首轮真实 `53-right` 在线案例用1次 fresh grounding 和4步 held 将真实 XY 距离从约
10.05cm降到0.41cm，但控制器没有预测 `stop`，最终以 `stale_target_limit` 结束。
现有 episode `success` 只在 `termination_reason == "success"` 时成立，因此把“实际已经
到达目标”和“控制器自主判断停止”合并为同一个指标，无法准确表达这条证据。

本设计把物理任务到达、自主停止和允许参与批次通过的成功拆开。PyBullet 红块真值继续
只用于事后评分，不得影响控制循环是否继续、下一步动作或 VLM 请求。

## 目标与非目标

目标：

1. 独立记录 episode 最终是否真实到达3cm范围。
2. 独立记录控制器是否预测 `stop` 并正确到达。
3. 允许 `stale_target_limit` 结束但最终真实到达的 episode 参与主要批次通过。
4. 防止 API、IK、反投影等系统错误因最终距离碰巧达标而被记为主要成功。
5. 保留原始终止原因，让安全停止与自主停止可区分。

非目标：

- 不使用真值提前终止控制循环。
- 不修改 stop 距离、任务成功距离、held 上限、动作方向或 VLM 调用逻辑。
- 不修改整批初始动态预检。
- 不在本轮选择或替换三个固定案例。
- 不运行真实 API。

## 方案选择

采用 episode 级显式三字段设计，而不是只改变旧 `success` 或只在 batch 聚合时特殊处理。
这样每条 episode 证据自身完整，批次摘要只聚合清晰字段，不需要根据终止原因和距离重新
猜测语义。

## Episode 字段语义

每个 `run_control_loop()` summary 增加或更新以下字段。

### `task_success`

```python
final_true_distance_xy <= 0.03
```

表示 episode 结束时机械臂真实 XY 距离进入3cm任务范围。它是纯事后物理结果，不考虑
终止原因。即使系统错误结束但距离碰巧达标，该字段仍如实为 `true`。

### `autonomous_stop_success`

```python
termination_reason == "success"
```

现有 `success` 终止原因已经同时要求控制器预测 `stop` 和真实距离 `<=0.03`，因此该字段
表示控制器自主停止且停止正确。

### `success`

```python
task_success and termination_reason in {
    "success",
    "stale_target_limit",
}
```

这是参与主要批次通过的 episode 成功。允许的终止原因只有：

- `success`：控制器正确预测 stop；
- `stale_target_limit`：持续遮挡达到保持上限，但最终真实位置已经完成任务。

`api_error`、`ik_error`、`backprojection_error`、`held_target_error`、`invalid_box`、
`false_stop`、`visibility_out_of_scope`、`max_control_steps` 以及其他安全错误，即使最终距离
小于等于3cm，也必须保持 `success=false`。

## 计算时机与数据流

控制循环行为保持不变：

```text
观测 -> fresh/held 决策 -> 动作或停止 -> 事后 score -> 原终止规则
```

循环结束后，从最终 trace 行的 `true_distance_xy`（若没有 trace，则从初始评分）得到
`final_true_distance_xy`。随后只在 summary 构建阶段派生三个评分字段。任何一个评分字段
都不能回流到循环、动作依赖或 VLM 调用边界。

为避免两处3cm阈值产生分歧，在 `run_control_loop()` 内复用同一个局部最终距离值计算
`task_success`；本轮不新增配置项，也不改变现有硬编码 `0.03` 行为。

## 遮挡恢复字段

现有 `recovered_from_occlusion` 保持原语义：只有 `termination_reason == "success"` 且
发生 held 步骤时为 `true`。`stale_target_limit` 后实际到达不会被描述成“恢复后自主
停止”；它通过三个新评分字段和原始终止原因表达。

## Batch 聚合

`aggregate_smoke_summaries()` 输出：

```text
success_count
failure_count
success_rate
task_success_count
task_failure_count
task_success_rate
autonomous_stop_success_count
autonomous_stop_success_rate
```

定义：

- `success_count`：`row["success"]` 为真的数量；
- `task_success_count`：`row["task_success"]` 为真的数量；
- `autonomous_stop_success_count`：`row["autonomous_stop_success"]` 为真的数量；
- 各 failure/rate 使用 episode 总数计算，空批次 rate 为0.0。

主要 `passed` 继续要求：

1. 案例 seed 与固定配置完全一致；
2. `success_count == required_successes == 3`；
3. 每个 episode 至少1次 fresh VLM；
4. 每个 episode 最大目标年龄不超过4；
5. 总 API 调用不超过30。

CLI 摘要继续显示 `success=<success_count>/<num_episodes>`，其语义更新为“允许参与批次
通过的任务成功”。新增任务到达和自主停止计数保存在 JSON 证据中，不在本轮扩展 CLI。

## 兼容性

旧输出中的 `success` 等同于自主停止成功；新输出中的 `success` 是主要批次合格成功，
可能在 `stale_target_limit` 且最终到达时为真。因此这是有意的证据语义升级，不能把新旧
批次的 `success_rate` 直接合并统计。

`termination_reason`、`final_true_distance_xy` 和所有 trace 行保持不变，历史输出文件不
回写。文档必须注明此版本边界，避免把旧批次重新解释为新评分。

## 错误与安全边界

- 系统错误可以产生 `task_success=true`，因为它只陈述最终物理位置；但主要
  `success=false`，批次不能通过。
- `false_stop` 即使最终距离在边界数据中被改写为达标，也不进入允许终止原因集合。
- `max_control_steps` 即使最终到达也不能通过，避免用时间耗尽掩盖控制器没有正确结束。
- `stale_target_limit` 只有最终距离 `<=0.03` 才成功；距离未达标仍失败。
- 初始预检拒绝的批次不产生 episode summary，因此不进入本评分逻辑。

## 测试策略

使用 TDD 扩展 `tests/vlm/grounding_smoke/test_runner.py`：

1. 正常预测 stop 且到达：三个字段均为真。
2. `stale_target_limit` 且最终距离 `<=0.03`：任务到达和主要 success 为真，自主停止为假。
3. `stale_target_limit` 且最终距离 `>0.03`：三个字段均为假。
4. API、IK 或反投影错误且最终距离 `<=0.03`：只有任务到达字段为真。
5. `max_control_steps` 且最终距离 `<=0.03`：不能成为主要 success。
6. 聚合三个混合 episode 时，三组 count/rate 各自准确。
7. 批次 `passed` 只使用主要 `success`，并继续保护 fresh、目标年龄和 API 上限。

所有测试使用 mock/fake score，不访问网络或真实 API。完成后运行 grounding smoke 定向
测试、全量测试、`compileall`、真值接口隔离审计和 `git diff --check`。

## 验收标准

- `53-right` 等价 mock 轨迹在 `stale_target_limit` 且最终0.41cm时记为主要成功，但不记
  自主停止成功。
- 系统错误或 `max_control_steps` 不能因最终距离达标而让 batch 通过。
- 原控制循环的观测、动作、VLM、stop 和 held 行为无变化。
- 批次 JSON 同时报告主要成功、任务到达和自主停止统计。
- 不运行真实 API，所有自动测试和静态验证通过。
