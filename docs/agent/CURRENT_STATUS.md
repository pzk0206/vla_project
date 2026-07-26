# 当前项目状态

**最后核对日期：** 2026-07-26

## 当前阶段

专家数据规模化的基础设施和真实 pilot 已完成。`expert_v1` 使用版本化目录、每条
episode 独立 home pose 复位、`random_seed + episode_idx`、manifest/config snapshot
和只读质量门禁。真实 PyBullet pilot 使用 seeds 1000–1009，得到10/10 success、
327帧、0项完整性错误，`pilot_gate.passed=true`，未调用 VLM API。当前阶段允许以
追加模式扩到至少300条；实现已 fast-forward 合并到 `main`（`5481e85`），功能
worktree 与分支已清理。严格 smoke 的自主停止1/3仍作为独立能力边界，不阻塞扩展。

## 已完成且仍有效

- Baseline 专家数据采集和诊断输出可用。
- Stage 3 heuristic 固定 seeds 42–91 为 50/50 成功。
- 直接方向预测的能力边界已确定：448px 为 3/4，grounding 中间表示为 4/4。
- 相机反投影几何、可见率分组和离线真值隔离已建立。
- 冻结 XY 补偿在独立 clear 验证集上为 15/15 不超过 3cm。
- 正式源码已迁移到 `src/vla_project/`，并通过 `pyproject.toml` 暴露 13 个命令入口。
- `grounding_smoke` targeting、runner、screening 已迁入正式子包；测试不调用真实 API。
- 旧的全轨迹 clear 筛选在 seeds 55–100 × left/right/front 的138个候选中为0个
  合格：131个 `visibility_below_threshold`、7个 `start_pose_error`；该结果继续
  证明全程 clear 不是可用前提。
- 新的首帧资格筛选在相同138个候选中得到47个合格：47个
  `visibility_below_threshold`、44个 `start_pose_error`、47个 `qualified`。
- 按 `left -> right -> front` 顺序选择最小且互异 seed，固定案例已冻结为
  `56-left`、`55-right`、`59-front`。
- held 路径只复用已校验的 VLM 世界目标，每步依据当前末端位置重新计算动作；第5次
  连续不可见时使用 `stale_target_recheck` 做只读停止复核。复核不会调用 VLM，也不会
  执行额外动作；仍需移动时才以 `stale_target_limit` 安全中止。
- trace/summary 已记录 fresh/held、目标年龄、API 调用和遮挡恢复字段；新增时序、只读
  复核和严格批次门槛时的完整自动测试为173/173通过。
- episode 与 batch summary 现在分别报告主 `success`、真实到达 `task_success` 和自主
  停止 `autonomous_stop_success`。主成功允许真实到达后的 `success` 或
  `stale_target_limit`，但拒绝系统错误和步数耗尽；成功评分拆分实现时的相关定向测试
  为74/74，本轮没有调用真实 API。
- 首轮真实 smoke 证据保存在
  `outputs/vlm_evaluations/grounding_world_smoke/run_20260719_target_hold_v1/`；本轮仅发生
  1次真实 API 调用，批次退出码为1，没有生成完整 `batch_summary.json`。
- `53-right` 提供了第一条真实在线目标保持证据：1次 fresh VLM 后连续4步 held 均未
  再调用 API，真实 XY 距离从约10.05cm降至0.41cm。历史摘要按旧口径记为失败；按
  当前新口径解释，同类结果是任务到达、未自主 stop、主要 `success=true`。
- 原固定案例的整批预检结果为1/3合格，完整记录了 `52-left` 和 `54-front` 的拒绝
  证据；新固定案例的只预检结果为3/3合格，只生成 `smoke_preflight.json`，没有
  episode trace 或在线摘要，API 调用为0。
- 新固定案例预检指标：`56-left` 姿态误差0.001144m、可见率0.753968；
  `55-right` 姿态误差0.004672m、可见率0.928571；`59-front` 姿态误差0.002442m、
  可见率0.806878。
- 第二轮真实 smoke 证据保存在
  `outputs/vlm_evaluations/grounding_world_smoke/run_20260726_fixed_cases_online_v1/`；
  批次主要成功和任务到达均为3/3，`passed=true`，总 API 调用4次。
- `56-left`、`55-right`、`59-front` 的真实 XY 距离分别从约
  9.93/10.43/9.96cm 降到0.303/0.789/0.371cm；三例均执行5次动作。
- `55-right` 在一次遮挡恢复后自主 stop；`56-left` 和 `59-front` 物理到达但未自主
  stop，因此整批 `autonomous_stop_success=1/3`，终止原因计数为
  `success: 1`、`stale_target_limit: 2`。
- `56-left` 的旧 trace 证明最后一次动作后，末端与缓存目标两轴误差均已低于2cm；
  新的过期复核可识别这一状态。`59-front` 的 VLM 框中心相对真实红块约偏左5.4像素、
  偏上8.7像素，反投影和冻结补偿后仍留下约2.9cm Y 偏差，时序修复不会把它误判为
  stop。
- 新批次 `passed` 现在要求 `autonomous_stop_success_count == required_successes == 3`；
  历史 `run_20260726_fixed_cases_online_v1/smoke_summary.json` 不回写，其
  `passed=true` 仍按旧语义解释。本轮修复没有调用真实 VLM。
- `expert_v1` 已冻结9维动作、帧/摘要 schema、seed 规则和版本化目录保护；
  `vla-evaluate-dataset` 会扫描 schema、动作维度、图片、重复 step、seed、帧数和终止
  标志，并把严格 pilot gate 写入 `dataset_quality_report.json`。
- 真实 pilot 保存在
  `outputs/dataset/expert_scaling_v1/`：10条 episode、327帧、10/10 success；
  最终距离 min/mean/median/max 为
  0.029069/0.029546/0.029523/0.029983m，每条帧数为31/32.7/32.5/35。
- pilot 的12项门禁全部通过，所有错误计数均为0；报告路径为
  `outputs/dataset/expert_scaling_v1/dataset_quality_report.json`。旧数据仍为
  50条摘要、286条帧记录和286张根目录 JPEG。
- 当前完整自动测试为191/191通过，`compileall` 与 `git diff --check` 通过；本轮采集
  和质量评估代码不读取 VLM API 环境变量，也不导入 VLM API 调用函数。

## 未解决问题

1. `53-right` 的历史摘要仍按旧口径记录 `success=false`；新口径下同类结果会记为
   `task_success=true`、`autonomous_stop_success=false`、`success=true`。旧批次未包含
   新字段，其 `success_rate` 不可与新批次直接合并比较。
2. 初始动态资格现在已有完整预检 JSON；但正式闭环开始后的普通案例异常仍可能中断批次
   并缺少完整 summary，该部分不在本轮预检修复范围。
3. `59-front` 的单帧 grounding 残余偏差约为2.9cm，超过当前每轴2cm stop 阈值；
   将其保留为已知能力边界，不通过读取真值、放宽阈值或针对单 seed 调参来消除。
4. 永久遮挡、目标移动和 severe 遮挡恢复仍不在当前范围。
5. README、学习计划和 BUGLOG 的阶段表述可能存在时间差；实验结论以
   原始摘要和对应证据链为准。
6. 10条 pilot 只能验证流程与小样本质量，尚不能证明300条规模下仍有99%以上成功率和
   足够的目标位置覆盖；pilot 的 X 分箱仍有一个空箱，扩展后必须重新审计覆盖。

## 下一步优先级

1. 保持 `expert_v1` schema 不变，将 `clean_before_run` 改为 `false` 并追加290条，
   使有效 episode 总数达到至少300。
2. 扩展后重新运行完整质量扫描，要求有效 episode 至少300、总体成功率不低于99%、
   完整性错误为0，并检查红块 X/Y 分箱覆盖。
3. 通过规模化验收后进入 action tokenization，避免采集与动作表示同时改动。
4. 自主 stop 3/3继续作为严格 smoke 指标；只有后续训练或评估明确依赖时，才重新开启
   grounding 稳健性改进。

## 当前恢复点（2026-07-26）

`expert_v1` 真实10条 pilot 已通过严格门禁，证据位于
`outputs/dataset/expert_scaling_v1/`，当前配置仍是保护性
`clean_before_run: true`、`num_episodes: 10`。下次扩展前必须显式改为追加模式并把新增
数量设为290，不能再次以 clean 模式运行，否则会清理已经验收的 pilot。扩展后重新运行
`vla-evaluate-dataset`，未达到至少300条、99%成功率和零完整性错误时不得进入 action
tokenization。

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
- 专家数据质量检查：[evaluate_dataset.py](../../src/vla_project/simulation/evaluate_dataset.py)
- 专家数据规模化设计：[2026-07-26-expert-dataset-scaling-design.md](../superpowers/specs/2026-07-26-expert-dataset-scaling-design.md)
- 专家数据规模化计划：[2026-07-26-expert-dataset-scaling.md](../superpowers/plans/2026-07-26-expert-dataset-scaling.md)

## 更新规则

阶段、进行中工作、未解决问题或下一步改变时，直接改写本文件中的失效内容。详细实验
过程写入 WORKLOG 或 BUGLOG，不在这里无限追加。
