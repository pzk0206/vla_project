# 当前项目状态

**最后核对日期：** 2026-08-09

## 当前阶段

目录安全、delta v2 修复/重训、双积木 v2 数据、VLA 重训和正式成对反事实评估均已完成。
当前可审计结论是：**现有 MiniLM VLA 不能区分“悬停在红色积木上方”和“悬停在蓝色积木
上方”这两句中文指令**。旧70%语言跟随结论不恢复。下一阶段先更换或修正中文文本编码链路，
再继续 Qwen2-VL QLoRA。

### 2026-08-09 目录安全完成点

- 工作树：`/home/pzk/vla_project/.worktrees/vla-data-integrity`
- 已完成提交：
  - `4919042`：统一管理输出路径与可回滚原子发布。
  - `f1b8f01`：专家数据集拒绝路径逃逸及非空目录清理/覆盖。
  - `6915498`：VLM 样本 staging 校验后原子发布。
  - `a4fd0d0`：动作审计、可见性审计和派生数据输出边界。
  - `9540e2b`：Probe 与 grounding smoke 运行名路径逃逸。
  - `ff7eaff`：训练、rollout、单次 Probe 和剩余 VLM CLI 输出边界。
- 攻击输入与发布恢复回归：187/187通过；新增/修改入口组合回归：87/87通过。
- 完整回归：303项中300项通过、3项失败，恰好是修复前已记录的两个配置契约和一个
  package CLI 元数据不一致，没有新增失败。
- `compileall -q src tests` 和 `git diff --check` 通过；全程未运行真实采集、GPU 训练或
  付费 API。
- 工作树在实现提交后为 clean；继续工作前仍应检查状态，并且不要触碰主工作树中用户修改
  的 `notebooks/qwen2_vl_qlora_vla.ipynb`。
- 已知全量基线并非全绿：安全修复开始前为253项测试、3项失败（两个配置契约仍期待
  `expert_scaling`，以及 package CLI 元数据不一致）。这些属于后续工程一致性修复，不能
  报告为当前回归通过。

### 当前实验结论边界

- 后续正式采集与训练继续采用斜视图；俯视图遮挡率更高，只保留为对照派生数据。
- 旧 `delta_q_64 = 0%` 因标签/执行语义错误而无效；修复后的 v2 已完成重训和同口径
  rollout，仍为0/50，平均最终距离0.9454m，50条均耗尽200步。在当前工程协议下可以
  排序为 regression 92% > absolute 66% > delta v2 0%，但不能外推为 delta 普遍不可行。
- 多任务 VLA 的24.5%任务成功率可以作为旧设置的观察结果；旧“70%语言跟随率/反事实
  切换”已被正式 v2 审计否定，不能作为红蓝识别结论。
- `expert_multi_v1` 的红蓝块采样与场景状态不满足新门禁；其结果只说明旧场景配置，不能
  外推为已控制积木位置混杂。后续使用新版本目录采集300条通过门禁的数据。

### 后续顺序

1. **已完成：**在新目录重训 `bc_delta_q_64_overfit_10_v2` 与
   `bc_delta_q_64_full_v2`。overfit 最终训练损失为0.516125；full 最佳验证损失为
   9.626452（epoch 30）。两个 checkpoint 均保存 v2 语义、split 哈希和完整 tokenizer。
   新 best checkpoint 的50条 rollout 为0/50，平均/中位最终距离0.9454/0.9665m；旧 v1
   产物继续保留但废弃。
2. **已完成：**`expert_multi_v2` 为300/300 success、红蓝150/150、9,826帧/图片；所有
   完整性错误为0，间距和漂移门禁通过，`active_gate=scale`、`passed=true`。
3. **已完成：**按目标颜色和位姿分层得到250/50 episode split；VLA full rerun 完成50
   epochs，最佳验证损失0.064903（epoch 27），checkpoint 与数据/split 哈希绑定。
4. **已完成：**50对按 episode seed、保存红蓝位姿和机器人状态确定性复现；全部有效且
   重放误差为0。成对指令跟随0/50、偏好切换0/50，红/蓝指令成功11/50和12/50。
5. **已完成：**文本编码审计显示两句中文的 token IDs 完全相同、embedding L2=0；当前
   模型在输入层已经失去红蓝差异，因此旧70%不恢复。
6. **下一步：**更换支持中文且能区分红蓝的文本编码器并重新训练/反事实验证；保持斜视
   主视角。训练测试、CLI 元数据和项目文档已补齐，用户修改中的 notebook 本轮未覆盖。

## 2026-08-07 历史状态（以下不作为当前决策依据）

### 已完成

| 里程碑 | 关键数据 |
|--------|---------|
| BC 三组对照 | regression 92% rollout 成功率 |
| 多任务 VLA | 24.5% 成功率，70% 语言跟随率 |
| 反事实实验 | 换指令后模型切换目标（70%） |

### 当前：Qwen2-VL-2B QLoRA 微调

按学习计划第 9-10 周"轻量微调验证"。方案：Google Colab (T4 16GB) + Qwen2-VL-2B + QLoRA。
步骤：数据格式转换 → Colab 训练 → 下载权重 → 本地 rollout。

### BC 最终结果（regression 胜出）

| 组 | rollout 成功率 | 结论 |
|---|----------------|------|
| regression | **92%** (46/50) | 🏆 VLA 主基线 |
| absolute_q_32 | **66%** (33/50) | 可行，需更大模型 |
| delta_q_64 | **0%** (0/50) | ❌ 不可行 |

### 多任务 VLA 进展

| 步骤 | 状态 |
|------|------|
| 设计文档 | ✅ `docs/superpowers/specs/2026-08-07-multi-task-vla-design.md` |
| 仿真改造（双积木 + 任务随机） | ✅ `control_arm.py`, `sim_config.yaml` |
| 数据采集（290ep，红144/蓝146） | ✅ `outputs/dataset/expert_multi_v1/` |
| VLA 模型代码（vla_model/vla_train） | ❌ 待实现（classifier 故障阻塞） |

### 当前阻塞

`deepseek-v4-pro` safety classifier 持续性故障，导致无法派 Luna、无法跑 Bash 命令。
等 classifier 恢复后第一步：实现 VLA 模型代码 → overfit → 全量训练 → rollout。

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
- 当前完整自动测试为362/362通过；`compileall` 和 `git diff --check` 在本轮收尾时通过。

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
6. Tokenizer 资产已由训练模块通过 `tokenizer_utils.load_quantile_edges()` 从
   `action_tokenization_audit_v2/` 加载，训练时不再依赖手动步骤。但边界本身仍来自
   离线审计阶段拟合——如需换数据集或重跑审计，需手动重新生成该报告。

## 下一步优先级

1. **classifier 恢复后立即**：派 Luna 实现 VLA 模型代码（按设计文档）。
2. VLA overfit 验证（10ep overfit）。
3. VLA 全量训练（250ep）。
4. VLA rollout：按指令评估——同一张图，"红色" vs "蓝色" 指令能否选对目标。
5. 确认语言信号被模型使用后，进入 API VLA 闭环阶段。

## 当前恢复点（2026-08-07）

**BC 阶段完成，多任务 VLA 数据就绪，代码待写。**

关键资产：
- BC 最优模型：`outputs/training/bc_regression_full_v1/checkpoint_best.pt`（92% rollout）
- 多任务数据：`outputs/dataset/expert_multi_v1/`（290ep，红144/蓝146）
- 设计文档：`docs/superpowers/specs/2026-08-07-multi-task-vla-design.md`
- Rollout 模块：`src/vla_project/training/rollout.py`
- 仿真已支持双积木：`sim_config.yaml` + `control_arm.py`

**恢复时第一件事：** 派 Luna 实现 VLAModel + VLADataset + vla_train（按设计文档）。
所需新文件：`vla_model.py`, `vla_train.py`；修改：`dataset.py`, `pyproject.toml`。
依赖：`pip install sentence-transformers`。

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
- BC 训练管线设计：[2026-08-06-bc-training-pipeline-design.md](../superpowers/specs/2026-08-06-bc-training-pipeline-design.md)
- Episode 划分/Tokenizer 重拟合设计：[2026-08-06-episode-split-tokenizer-refit-design.md](../superpowers/specs/2026-08-06-episode-split-tokenizer-refit-design.md)
- 视觉输入策略设计：[2026-08-06-visual-input-strategy-design.md](../superpowers/specs/2026-08-06-visual-input-strategy-design.md)
- 可见性审计设计：[2026-07-27-expert-dataset-visibility-audit-design.md](../superpowers/specs/2026-07-27-expert-dataset-visibility-audit-design.md)
- 专家数据规模化设计：[2026-07-26-expert-dataset-scaling-design.md](../superpowers/specs/2026-07-26-expert-dataset-scaling-design.md)
- 专家数据规模化计划：[2026-07-26-expert-dataset-scaling.md](../superpowers/plans/2026-07-26-expert-dataset-scaling.md)

## 对外表述参考（2026-08-10 审计）

以下为经项目证据核对的对外表述，可直接用于简历或面试。每个数据点均已在本文件中找到
对应审计来源。

### 仿真环境与专家数据采集

在 PyBullet 中搭建 KUKA iiwa 七轴机械臂交互环境，实现 IK/FK 关节控制与多视角虚拟
相机采集。构建 300 条（9,826 帧）双积木悬停轨迹数据集，红/蓝目标各 150 条，成功率
100%，X/Y 五区均衡采样。建立逐帧质量门禁、可见性审计（95.8% clear）和动作
tokenization 量化分析（等频/等宽分箱评估），全部数据经 SHA-256 绑定与原子发布。

### 动作表示与模仿学习

以 ResNet-18 为骨干训练行为克隆基线，经动作量化审计筛选后对比连续回归（regression）
与离散分类（absolute_q_32、delta_q_64）三种动作表示。连续回归以 92%（46/50）Rollout
成功率胜出，验证了连续控制在具身操作任务中的优势；delta_q_64 在当前数据规模下不可行
（0/50），为小样本下的动作表示选择提供消融证据。

### 端到端 VLA 与语言信号分析

构建 ResNet-18 + 多语言 MiniLM 融合 VLA 策略（~129M 参数），实现「RGB + 中文指令 →
7 维关节动作」的端到端映射。建立成对反事实评估协议：固定场景状态与随机种子，仅切换
指令颜色，确定性复现（重放误差 = 0）以严格检验语言信号利用。审计发现旧英文 MiniLM
将中文"红色"/"蓝色"编码为相同 token（embedding L2 = 0），已替换编码器并增加训练前
token 区分度门禁，后续规划 Qwen2-VL QLoRA 增强语言理解。

### VLM 感知与模块化闭环

以 Qwen3-VL-Flash 为开放词汇感知前端，结合 PyBullet 相机反投影几何与冻结 XY 补偿
（标定集 15 样本拟合，验证集 19/20 校正后 ≤ 3cm，达 95%），配合确定性 IK 控制器实现
模块化闭环。3 个未见场景全部成功，末端定位精度 0.30–0.79 cm，借助目标保持策略将 API
调用压缩至 3 个 episode 合计 4 次。按遮挡程度（Clear/Partial/Severe）分层统计误差，
标定与验证严格隔离。

### 工程规范与可复现性

362 项自动化测试，单一 YAML 管理全链路参数。建立完整审计链：数据完整性（9 个 SHA-256
绑定、路径逃逸拒绝）、动作量化评估、可见性分层统计、checkpoint 与数据集哈希绑定。
成对反事实评估中 episode 级确定性复现，重放误差为 0。

---

### 常见错误与纠正

| 错误表述 | 真实数据 | 证据 |
|---------|---------|------|
| 70% 指令跟随率 | 0/50，旧结论已被 v2 审计推翻 | 成对反事实、token 审计 |
| 34M 参数 | ~129M，文本编码器未冻结 | `VLAModel().parameters()` |
| 单 Episode API 4 次 | 3 episode 合计 4 次 | smoke_summary total_api_calls |
| Qwen2-VL 感知前端 | Qwen3-VL-Flash（API 调用） | 输出目录名 qwen3_vl_flash |
| 9,894 帧 | 9,826 帧（expert_multi_v2） | dataset_manifest.json |
| 独立验证 15/15 ≤3cm | 19/20 ≤3cm（95%），clear 15/15 | calibration_validation_summary |

## 更新规则

阶段、进行中工作、未解决问题或下一步改变时，直接改写本文件中的失效内容。详细实验
过程写入 WORKLOG 或 BUGLOG，不在这里无限追加。
