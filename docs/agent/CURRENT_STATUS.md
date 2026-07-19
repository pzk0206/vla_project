# 当前项目状态

**最后核对日期：** 2026-07-19

## 当前阶段

项目已完成 Python `src/` 包结构、统一命令入口和生成输出整理。VLM 主线已完成固定
相机 clear 场景的 grounding 世界坐标独立校准验证，下一工程阶段是把该能力接入
小规模、可中止的在线闭环 smoke test。

## 已完成且仍有效

- Baseline 专家数据采集和诊断输出可用。
- Stage 3 heuristic 固定 seeds 42–91 为 50/50 成功。
- 直接方向预测的能力边界已确定：448px 为 3/4，grounding 中间表示为 4/4。
- 相机反投影几何、可见率分组和离线真值隔离已建立。
- 冻结 XY 补偿在独立 clear 验证集上为 15/15 不超过 3cm。
- 正式源码已迁移到 `src/vla_project/`，并通过 `pyproject.toml` 暴露 10 个命令入口。

## 未解决问题

1. 冻结补偿后的 grounding 坐标尚无在线机械臂闭环证据。
2. severe 遮挡仍未解决；现有 clear 结论不能外推。
3. 2026-07-17 smoke 实施计划早于 `src/` 包迁移，文件路径和命令需要先按当前结构
   修订。
4. README、学习计划、BUGLOG 和 WORKLOG 的阶段表述可能存在时间差；实验结论以
   原始摘要和对应证据链为准。

## 下一步优先级

1. 按当前包结构修订 grounding 世界坐标闭环 smoke 实施计划，不改变已批准的真值
   隔离、安全中止、3 个 clear cases 和最多 30 次 API 请求约束。
2. 实现并用 mock 测试纯 targeting、安全状态机和闭环编排，确认测试不会调用真实
   API。
3. 经用户明确批准付费实验后，运行一次 seeds 52–54 的真实 smoke，并根据唯一主导
   失败类型决定后续工作。

## 当前任务入口

- 项目简介：[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
- 文件路由：[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- Smoke 设计：[2026-07-17-grounding-world-closed-loop-smoke-design.md](../superpowers/specs/2026-07-17-grounding-world-closed-loop-smoke-design.md)
- 旧实施计划：[2026-07-17-grounding-world-closed-loop-smoke.md](../superpowers/plans/2026-07-17-grounding-world-closed-loop-smoke.md)
- 校准实现：[validate_grounding_calibration.py](../../src/vla_project/vlm/validate_grounding_calibration.py)
- 相机几何：[camera_geometry.py](../../src/vla_project/simulation/camera_geometry.py)
- 单次闭环基线：[stage3_probe.py](../../src/vla_project/simulation/stage3_probe.py)

## 更新规则

阶段、进行中工作、未解决问题或下一步改变时，直接改写本文件中的失效内容。详细实验
过程写入 WORKLOG 或 BUGLOG，不在这里无限追加。
