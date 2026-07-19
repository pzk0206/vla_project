# 当前项目状态

**最后核对日期：** 2026-07-19

## 当前阶段

项目已完成首轮真实 `grounding_smoke`，但批次未通过且未完整跑完。三个固定案例中，
`52-left` 首帧可见率不足而在 API 前停止；`53-right` 用1次真实 grounding 和4步目标
保持把真实 XY 距离降到约0.41cm，随后因持续遮挡以 `stale_target_limit` 停止；
`54-front` 的固定起点无法满足5mm姿态容差，批次以异常退出。现在已增加整批无 API
动态预检：三个固定案例的起点与首帧可见率全部合格后才允许进入在线闭环。当前案例已
被门禁在任何 API 请求前整体拒绝。事后成功评分已完成拆分；当前阶段是在不调用付费
API 的情况下修订固定案例，使三个案例都满足运行前提。

## 已完成且仍有效

- Baseline 专家数据采集和诊断输出可用。
- Stage 3 heuristic 固定 seeds 42–91 为 50/50 成功。
- 直接方向预测的能力边界已确定：448px 为 3/4，grounding 中间表示为 4/4。
- 相机反投影几何、可见率分组和离线真值隔离已建立。
- 冻结 XY 补偿在独立 clear 验证集上为 15/15 不超过 3cm。
- 正式源码已迁移到 `src/vla_project/`，并通过 `pyproject.toml` 暴露 12 个命令入口。
- `grounding_smoke` targeting、runner、screening 已迁入正式子包；测试不调用真实 API。
- seeds 55–100 × left/right/front 的138个动态候选筛选结果为0个合格：131个
  `visibility_below_threshold`、7个 `start_pose_error`。
- held 路径只复用已校验的 VLM 世界目标，每步依据当前末端位置重新计算动作；第5次
  连续失效尝试以 `stale_target_limit` 安全中止，且 held 步骤不调用 VLM。
- trace/summary 已记录 fresh/held、目标年龄、API 调用和遮挡恢复字段；当前完整自动
  测试为173/173通过。
- episode 与 batch summary 现在分别报告主 `success`、真实到达 `task_success` 和自主
  停止 `autonomous_stop_success`。主成功允许真实到达后的 `success` 或
  `stale_target_limit`，但拒绝系统错误和步数耗尽；相关定向契约测试为76/76通过，
  完整自动测试为173/173通过，本轮没有调用真实 API。
- 首轮真实 smoke 证据保存在
  `outputs/vlm_evaluations/grounding_world_smoke/run_20260719_target_hold_v1/`；本轮仅发生
  1次真实 API 调用，批次退出码为1，没有生成完整 `batch_summary.json`。
- `53-right` 提供了第一条真实在线目标保持证据：1次 fresh VLM 后连续4步 held 均未
  再调用 API，真实 XY 距离从约10.05cm降至0.41cm。历史摘要按旧口径记为失败；按
  当前新口径解释，同类结果是任务到达、未自主 stop、主要 `success=true`。
- 整批动态预检已通过真实 PyBullet 无 API 验证：固定案例结果为1/3合格，`52-left`
  因初始可见率0.714被拒绝，`54-front` 因起点误差0.053522m被拒绝；只生成
  `smoke_preflight.json`，没有 episode trace 或在线摘要，API 调用为0。

## 未解决问题

1. 三个固定案例本身不满足当前运行前提：`52-left` 初始可见率为0.714，低于0.75；
   `54-front` 起始姿态误差为0.053522m，高于0.005m容差。
2. `53-right` 的历史摘要仍按旧口径记录 `success=false`；新口径下同类结果会记为
   `task_success=true`、`autonomous_stop_success=false`、`success=true`。旧批次未包含
   新字段，其 `success_rate` 不可与新批次直接合并比较。
3. 初始动态资格现在已有完整预检 JSON；但正式闭环开始后的普通案例异常仍可能中断批次
   并缺少完整 summary，该部分不在本轮预检修复范围。
4. 永久遮挡、目标移动和 severe 遮挡恢复仍不在当前范围。
5. README、学习计划、BUGLOG 和 WORKLOG 的阶段表述可能存在时间差；实验结论以
   原始摘要和对应证据链为准。

## 下一步优先级

1. 在整批预检门禁下修订固定案例，使三个案例都满足初始姿态和首帧可见率；不得跳过或
   自动替换失败案例。
2. 固定案例修订后先运行无 API 验证；再次运行真实 smoke 仍需用户明确批准。

## 会话暂停点（2026-07-19）

今日工作已停止，没有启动新案例筛选，也没有调用真实 VLM。下次从“固定案例修订设计”
继续：优先确认是否仍保持 `left/right/front` 各一个；推荐保持方向覆盖，并把筛选资格
从“不可能满足的全程 clear”改为“起点姿态合格且首帧可见率合格”，后续遮挡继续交给
最多4步 held 状态机。选择规则必须预先冻结、确定性执行并保留无 API 证据，不能根据
真实 smoke 结果临时替换案例。

## 当前任务入口

- 项目简介：[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
- 文件路由：[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- Smoke 设计：[2026-07-17-grounding-world-closed-loop-smoke-design.md](../superpowers/specs/2026-07-17-grounding-world-closed-loop-smoke-design.md)
- 已按当前 `src/` 结构修订的 Smoke 实施计划：[2026-07-17-grounding-world-closed-loop-smoke.md](../superpowers/plans/2026-07-17-grounding-world-closed-loop-smoke.md)
- 目标保持设计：[2026-07-17-grounding-target-hold-design.md](../superpowers/specs/2026-07-17-grounding-target-hold-design.md)
- 目标保持计划：[2026-07-18-grounding-target-hold.md](../superpowers/plans/2026-07-18-grounding-target-hold.md)
- 成功评分设计：[2026-07-19-grounding-smoke-success-scoring-design.md](../superpowers/specs/2026-07-19-grounding-smoke-success-scoring-design.md)
- 成功评分计划：[2026-07-19-grounding-smoke-success-scoring.md](../superpowers/plans/2026-07-19-grounding-smoke-success-scoring.md)
- Smoke targeting：[targeting.py](../../src/vla_project/vlm/grounding_smoke/targeting.py)
- Smoke runner：[runner.py](../../src/vla_project/vlm/grounding_smoke/runner.py)
- 动态筛选：[screening.py](../../src/vla_project/vlm/grounding_smoke/screening.py)
- 校准实现：[validate_grounding_calibration.py](../../src/vla_project/vlm/validate_grounding_calibration.py)
- 相机几何：[camera_geometry.py](../../src/vla_project/simulation/camera_geometry.py)
- 单次闭环基线：[stage3_probe.py](../../src/vla_project/simulation/stage3_probe.py)

## 更新规则

阶段、进行中工作、未解决问题或下一步改变时，直接改写本文件中的失效内容。详细实验
过程写入 WORKLOG 或 BUGLOG，不在这里无限追加。
