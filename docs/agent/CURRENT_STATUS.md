# 当前项目状态

**最后核对日期：** 2026-07-19

## 当前阶段

项目已完成首轮真实 `grounding_smoke`，但批次未通过且未完整跑完。三个固定案例中，
`52-left` 首帧可见率不足而在 API 前停止；`53-right` 用1次真实 grounding 和4步目标
保持把真实 XY 距离降到约0.41cm，随后因持续遮挡以 `stale_target_limit` 停止；
`54-front` 的固定起点无法满足5mm姿态容差，批次以异常退出。当前阶段是根据这些真实
证据修订运行边界和批次失败处理，不直接重跑付费实验。

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
- trace/summary 已记录 fresh/held、目标年龄、API 调用和遮挡恢复字段；完整自动测试为
  164/164通过。
- 首轮真实 smoke 证据保存在
  `outputs/vlm_evaluations/grounding_world_smoke/run_20260719_target_hold_v1/`；本轮仅发生
  1次真实 API 调用，批次退出码为1，没有生成完整 `batch_summary.json`。
- `53-right` 提供了第一条真实在线目标保持证据：1次 fresh VLM 后连续4步 held 均未
  再调用 API，真实 XY 距离从约10.05cm降至0.41cm；但按当前终止/通过契约仍记为失败。

## 未解决问题

1. 三个固定案例本身不满足当前运行前提：`52-left` 初始可见率为0.714，低于0.75；
   `54-front` 起始姿态误差为0.053522m，高于0.005m容差。
2. `53-right` 在真实距离已小于3cm后仍因预测目标尚未满足 stop 条件继续保持，最终以
   `stale_target_limit` 结束；需要明确“控制停止”和“事后成功评分”的契约关系。
3. 批次在单个案例起始姿态失败时直接抛异常，只留下前两个 episode summary，没有完整
   批次摘要；后续应保证配置/起点预检发生在任何付费请求之前，或把案例级失败结构化。
4. 永久遮挡、目标移动和 severe 遮挡恢复仍不在当前范围。
5. README、学习计划、BUGLOG 和 WORKLOG 的阶段表述可能存在时间差；实验结论以
   原始摘要和对应证据链为准。

## 下一步优先级

1. 先设计无 API 的批次级预检：三个固定案例都通过起始姿态和首帧可见率检查后，才允许
   发送第一条付费请求，避免再次得到部分批次。
2. 明确事后真实距离 `<=3cm` 是否可将 episode 计为成功，同时保持真值不能进入动作或
   stop 决策；据此补测试并修订 runner。
3. 修订固定案例或失败处理后先运行无 API 验证；再次运行真实 smoke 仍需用户明确批准。

## 当前任务入口

- 项目简介：[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
- 文件路由：[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- Smoke 设计：[2026-07-17-grounding-world-closed-loop-smoke-design.md](../superpowers/specs/2026-07-17-grounding-world-closed-loop-smoke-design.md)
- 已按当前 `src/` 结构修订的 Smoke 实施计划：[2026-07-17-grounding-world-closed-loop-smoke.md](../superpowers/plans/2026-07-17-grounding-world-closed-loop-smoke.md)
- 目标保持设计：[2026-07-17-grounding-target-hold-design.md](../superpowers/specs/2026-07-17-grounding-target-hold-design.md)
- 目标保持计划：[2026-07-18-grounding-target-hold.md](../superpowers/plans/2026-07-18-grounding-target-hold.md)
- Smoke targeting：[targeting.py](../../src/vla_project/vlm/grounding_smoke/targeting.py)
- Smoke runner：[runner.py](../../src/vla_project/vlm/grounding_smoke/runner.py)
- 动态筛选：[screening.py](../../src/vla_project/vlm/grounding_smoke/screening.py)
- 校准实现：[validate_grounding_calibration.py](../../src/vla_project/vlm/validate_grounding_calibration.py)
- 相机几何：[camera_geometry.py](../../src/vla_project/simulation/camera_geometry.py)
- 单次闭环基线：[stage3_probe.py](../../src/vla_project/simulation/stage3_probe.py)

## 更新规则

阶段、进行中工作、未解决问题或下一步改变时，直接改写本文件中的失效内容。详细实验
过程写入 WORKLOG 或 BUGLOG，不在这里无限追加。
