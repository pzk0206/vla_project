# 当前项目状态

**最后核对日期：** 2026-07-19

## 当前阶段

项目已把 grounding targeting、闭环 runner 和无 API screening 迁入正式 `src/`
子包，并通过完整 mock/契约测试。动态筛选已证明末端接近红块时会产生系统性遮挡，
因此当前阶段是实现“最近可靠目标 + 最多4步保持”的安全恢复状态机，而不是直接运行
付费 smoke test。

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
- 合并与迁移后的完整自动测试为157/157通过。

## 未解决问题

1. 冻结补偿后的 grounding 坐标尚无真实在线机械臂闭环证据。
2. 短期目标保持状态机已有设计和计划，但尚未实现；severe 遮挡仍不在当前范围。
3. README、学习计划、BUGLOG 和 WORKLOG 的阶段表述可能存在时间差；实验结论以
   原始摘要和对应证据链为准。

## 下一步优先级

1. 按现有目标保持设计实现 fresh/held 切换、4步保持边界、`stale_target_limit`
   中止和 trace 字段，并用 mock 证明真值隔离。
2. 完成全量自动测试后重新审查三个固定案例是否满足运行边界。
3. 经用户再次明确批准付费实验后，才运行真实 smoke，并根据唯一主导
   失败类型决定后续工作。

## 当前任务入口

- 项目简介：[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
- 文件路由：[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- Smoke 设计：[2026-07-17-grounding-world-closed-loop-smoke-design.md](../superpowers/specs/2026-07-17-grounding-world-closed-loop-smoke-design.md)
- Smoke 实施计划：[2026-07-17-grounding-world-closed-loop-smoke.md](../superpowers/plans/2026-07-17-grounding-world-closed-loop-smoke.md)
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
