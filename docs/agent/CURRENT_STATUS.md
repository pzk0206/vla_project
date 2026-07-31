# 当前项目状态

**最后核对日期：** 2026-08-01

## 当前阶段

专家数据规模化、原斜视可见性审计、action tokenization 只读审计和固定垂直俯视视觉
派生均已完成。`expert_topdown_v1` 保留300条 episode 和9,894帧标签，生成9,894张
448×448图片，状态/动作逐项等价且质量门禁通过；原 `expert_scaling_v1` 未被修改。
但俯视可见性 clear/partial/severe=7,024/477/2,393，300个终止帧全部 severe，明显差于
原斜视的9,479/411/4。当前暂停在训练前的视觉策略决策点，下次先决定斜视主基线、俯视
对照或双视角，再做 episode 级划分和轻量行为克隆，不能直接默认俯视单视角。

## 已完成且仍有效

- Baseline 专家数据采集和诊断输出可用。
- Stage 3 heuristic 固定 seeds 42–91 为 50/50 成功。
- 直接方向预测的能力边界已确定：448px 为 3/4，grounding 中间表示为 4/4。
- 相机反投影几何、可见率分组和离线真值隔离已建立。
- 冻结 XY 补偿在独立 clear 验证集上为 15/15 不超过 3cm。
- 正式源码已迁移到 `src/vla_project/`，并通过 `pyproject.toml` 暴露 16 个命令入口。
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
  标志，并把 pilot/scale 两阶段门禁、`active_gate` 和顶层 `passed` 写入
  `dataset_quality_report.json`。
- 真实规模化数据保存在 `outputs/dataset/expert_scaling_v1/`：episode 0–299、seeds
  1000–1299，共300条 episode、9,894帧、300/300 success；最终距离
  min/mean/median/max 为0.028713/0.029419/0.029414/0.029998m，每条帧数为
  29/32.98/33/36。
- scale gate 全部通过，所有完整性错误计数均为0；X 五箱为
  `[2057, 2257, 2492, 1523, 1565]`，Y 五箱为
  `[1830, 2650, 2129, 1876, 1409]`。报告路径为
  `outputs/dataset/expert_scaling_v1/dataset_quality_report.json`。旧数据仍为50条
  摘要、286条帧记录和286张根目录 JPEG。
- 可见性审计报告位于
  `outputs/dataset/expert_scaling_v1/visibility_audit_v1/`。9,894帧中
  clear 9,479（95.81%）、partial 411（4.15%）、severe 4（0.04%）；
  55/300条 episode 至少有一帧非 clear，只有 episode 105 出现4个 severe 帧。
  300个首帧全部 clear，终止帧为245 clear、55 partial；全局最长连续非 clear 段为
  episode 105 的14个保存帧（step 504–807）。
- action tokenization 审计报告位于
  `outputs/dataset/expert_scaling_v1/action_tokenization_audit_v1/`。输入门禁为300个
  episode、9,894帧、9,594 transitions；输入 manifest/trajectory/visibility 的 SHA-256
  在运行前后完全一致。夹爪9,894帧均为1.0，终止标志为300/9,894（3.03%）。
- absolute_q 推荐32箱等频：最坏关节非空箱占用率1.0、最小箱309帧、归一化 p95
  重建误差2.71%；delta_q 推荐64箱等频：对应为1.0、149帧、1.97%。absolute_q 的
  16箱等频因误差5.41%被拒；delta_q 的16/32箱等频因误差6.89%/6.74%被拒。
- 所有等宽候选至少存在尾部箱不足20帧；64箱 absolute/delta 等宽还分别存在占用率
  0.891/0.750，不满足0.90门槛，因此本轮没有推荐等宽分箱。
- 固定垂直俯视派生集位于 `outputs/dataset/expert_topdown_v1/`：300条 episode、9,894帧、
  9,894张448×448 JPEG，相机 eye 固定为`[0.0, 0.4, 3.0]`；状态和动作以`1e-9`容差逐项
  验证，源文件哈希不变，`dataset_quality_report.json` 的 scale gate 通过。
- 派生过程先独立验证原斜视重放，再在新的重放阶段渲染俯视图；这样避免不同相机和分辨率
  交替渲染污染 OpenGL 状态。源重放为9,888 exact、6 tolerance、0 rejected，原有严格
  MAE/max-error门槛未放宽。
- 俯视可见性报告位于 `outputs/dataset/expert_topdown_v1/visibility_audit_v1/`：clear
  7,024（70.99%）、partial 477（4.82%）、severe 2,393（24.19%）；300/300 episode
  出现 severe，300个首帧 clear，300个终止帧 severe。原斜视数据仍有95.81% clear、
  0.04% severe，因此两套图像均保留，尚未选择唯一训练视角。
- 当前完整自动测试为253/253通过；`compileall` 和 `git diff --check` 在本轮收尾时通过。

## 未解决问题

1. `53-right` 的历史摘要仍按旧口径记录 `success=false`；新口径下同类结果会记为
   `task_success=true`、`autonomous_stop_success=false`、`success=true`。旧批次未包含
   新字段，其 `success_rate` 不可与新批次直接合并比较。
2. 初始动态资格现在已有完整预检 JSON；但正式闭环开始后的普通案例异常仍可能中断批次
   并缺少完整 summary，该部分不在本轮预检修复范围。
3. `59-front` 的单帧 grounding 残余偏差约为2.9cm，超过当前每轴2cm stop 阈值；
   将其保留为已知能力边界，不通过读取真值、放宽阈值或针对单 seed 调参来消除。
4. 现有 expert 数据已量化自然遮挡，但永久遮挡、目标移动和 severe 遮挡恢复仍不在
   当前控制范围；4个 severe 帧集中在 episode 105，不能据此声称已解决 severe 遮挡。
5. README、学习计划和 BUGLOG 的阶段表述可能存在时间差；实验结论以
   原始摘要和对应证据链为准。
6. 离线审计已证明当前数据统计上支持32箱等频 absolute_q 和64箱等频 delta_q，但尚未
   通过真实训练或 rollout 证明哪种动作表示最好，也未形成 episode 级训练/验证划分。
   全量审计边界包含未来验证 episode，不能直接作为正式 tokenizer 资产。
7. 垂直俯视会在机械臂接近红块时产生系统性遮挡，severe 比例为24.19%，且所有终止帧
   severe。派生数据技术上有效，但不能据此声称它比原斜视更适合作为唯一视觉输入。

## 下一步优先级

1. 冻结原斜视 `expert_scaling_v1` 和派生 `expert_topdown_v1`，不再运行当前290条追加
   配置，不覆盖任何一套图片，也不改动采集 schema。
2. 先写视觉输入实验决策：至少比较“原斜视单视角主基线”和“垂直俯视对照”；如资源允许，
   再加入双视角融合。不要直接以俯视单视角开始正式训练。
3. 固定 episode 级训练/验证划分，避免同一轨迹的相邻帧跨集合泄漏；只用训练 episode
   重新拟合32箱等频 absolute_q 和64箱等频 delta_q 边界。
4. 实现连续回归、32箱等频 absolute_q 和64箱等频 delta_q 的最小训练对照；当前不因 partial
   或 severe 标签直接删除成功轨迹，后续用保留全部数据与可见性筛选做对照。
5. 自主 stop 3/3继续作为严格 smoke 指标；只有后续训练或评估明确依赖时，才重新开启
   grounding 稳健性改进。

## 当前恢复点（2026-08-01）

原斜视与固定垂直俯视两套300条专家视觉数据均已冻结，派生报告位于
`outputs/dataset/expert_topdown_v1/view_generation_report.json`，可见性报告位于其
`visibility_audit_v1/visibility_audit_summary.json`。恢复时先阅读这两个报告和
`docs/superpowers/specs/2026-07-31-expert-topdown-view-dataset-design.md`，从“斜视主基线、
俯视对照或双视角”的视觉输入决策开始；尚未启动行为克隆训练。当前 `sim_config.yaml`
的 `num_episodes: 290` 是已执行的追加批次配置，不应再次运行 `vla-collect`，否则会继续
追加到590条。

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
- 专家数据可见性审计：[audit_dataset_visibility.py](../../src/vla_project/simulation/audit_dataset_visibility.py)
- 动作表示审计：[audit_action_tokenization.py](../../src/vla_project/simulation/audit_action_tokenization.py)
- 共享专家重放：[expert_dataset_replay.py](../../src/vla_project/simulation/expert_dataset_replay.py)
- 固定视角派生：[render_expert_dataset_view.py](../../src/vla_project/simulation/render_expert_dataset_view.py)
- 俯视派生设计：[2026-07-31-expert-topdown-view-dataset-design.md](../superpowers/specs/2026-07-31-expert-topdown-view-dataset-design.md)
- 俯视派生计划：[2026-07-31-expert-topdown-view-dataset.md](../superpowers/plans/2026-07-31-expert-topdown-view-dataset.md)
- 动作审计设计：[2026-07-31-action-tokenization-audit-design.md](../superpowers/specs/2026-07-31-action-tokenization-audit-design.md)
- 动作审计计划：[2026-07-31-action-tokenization-audit.md](../superpowers/plans/2026-07-31-action-tokenization-audit.md)
- 可见性审计设计：[2026-07-27-expert-dataset-visibility-audit-design.md](../superpowers/specs/2026-07-27-expert-dataset-visibility-audit-design.md)
- 专家数据规模化设计：[2026-07-26-expert-dataset-scaling-design.md](../superpowers/specs/2026-07-26-expert-dataset-scaling-design.md)
- 专家数据规模化计划：[2026-07-26-expert-dataset-scaling.md](../superpowers/plans/2026-07-26-expert-dataset-scaling.md)

## 更新规则

阶段、进行中工作、未解决问题或下一步改变时，直接改写本文件中的失效内容。详细实验
过程写入 WORKLOG 或 BUGLOG，不在这里无限追加。
