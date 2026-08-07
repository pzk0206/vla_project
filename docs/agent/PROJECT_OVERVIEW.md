# VLA 项目简介（Agent 版）

## 项目定位

这是一个面向 VLA（Vision-Language-Action）学习的个人工程项目。项目先在 PyBullet
中建立可诊断、可复现的 KUKA iiwa 操作闭环，再逐步验证视觉语言感知、世界坐标
定位、动作决策、专家数据和轻量训练，而不是一开始训练大型端到端模型。

当前基础任务是根据“悬停在红色积木上方”的语言指令，让机械臂从视觉观测和本体
状态出发移动到红块上方，并保存可用于诊断、评估和后续训练的图像、动作与 episode
证据。

本项目同时服务三个目标：掌握 VLA 系统各层的实际接口；形成可复现、可测试的工程
成果；沉淀可以在简历和面试中解释的实验决策与失败证据。

## 当前范围

- 已覆盖：PyBullet 仿真、KUKA IK 控制、专家轨迹采集、闭环 probe、批量评估、
  Qwen 离线方向基线、grounding、相机反投影、独立校准验证、真值隔离 smoke runner
  、无 API 动态案例筛选、整批初始动态预检和最多4步的最近可靠目标保持状态机、
  视觉输入策略决策（斜视主基线）、episode 级训练/验证划分、action tokenization
  只读审计、BC 训练管线（ResNet-18 + 3组动作表示对照）、BC rollout 评估。
- 已完成：三组 BC 对照实验，regression 以 **92% rollout 成功率** 胜出。多任务 VLA
  数据已采集（290ep，红蓝双积木双指令），VLA 模型（ResNet-18 + MiniLM 文本编码器）
  已训练并 rollout 评估。反事实实验证明模型**确实在使用语言信号**——换指令后 70%
  的情况下模型跟随指令切换目标。当前 VLA 成功率为 24.5%，瓶颈是每任务数据量不足
  （~120ep/任务 vs BC 的 250ep/任务）。
- 正在推进：多任务 VLA 优化——补数据或接 API VLM 闭环。
- 暂不覆盖：大型端到端 VLA 训练（7B+）、真实机械臂部署、severe 遮挡恢复。

## 系统数据流

### 专家数据基线

`sim_config.yaml` -> 固定 seed 与 home pose 独立复位 -> PyBullet 场景与随机红块 ->
IK 目标和关节控制 -> `expert_v1` RGB、动作和本体状态 -> 版本化数据目录中的
manifest、配置快照、轨迹、episode 摘要与质量报告。

### Stage 3 闭环基线

当前 RGB -> heuristic 或兼容多模态 API 的方向决策 -> 统一世界坐标方向执行 ->
新观测 -> stop 或失败终止 -> `outputs/probe*` 中的 trace 与汇总。

### VLM 定位路线

固定评估图片 -> Qwen 红块 grounding -> 框中心像素 -> 相机反投影到工作平面 ->
冻结 XY 补偿 -> 与末端本体位置比较 -> 确定性动作方向。

PyBullet 红块真值只能用于离线评分、场景资格检查和受控 smoke test 的初始场景
搭建，不得进入正式 grounding 坐标或动作选择接口。这样可以区分视觉定位误差、
几何转换误差和控制误差。

## 模块职责

| 位置 | 职责 |
| --- | --- |
| `src/vla_project/simulation/` | PyBullet 场景、相机、机械臂控制、单次 probe 和批量 probe 评估 |
| `src/vla_project/vlm/` | VLM 样本、方向评估、grounding 诊断、反投影和校准验证 |
| `src/vla_project/vlm/grounding_smoke/` | 在线 smoke 的 targeting、安全编排和无 API 动态筛选 |
| `src/vla_project/tools/` | 仓库生成物迁移等维护工具 |
| `tests/` | 按领域镜像源码，并保护配置、包元数据、几何、控制和评估契约 |
| `sim_config.yaml` | 仿真、相机、任务、probe、VLM 和输出路径的统一参数来源 |
| `outputs/` | 本地生成的图片、trace、预测和实验摘要；默认不提交 Git |
| `docs/` | 当前知识、学习路线、Bug 证据以及历史设计和实施计划 |

正式源码采用 `src/` 布局，`pyproject.toml` 注册 16 个 `vla-*` 命令。新增文件和
测试前查看 [文件路由手册](PROJECT_STRUCTURE.md)，不要在仓库根目录添加正式 Python
脚本。

## 已验证结论与设计决定

1. Stage 3 heuristic 在固定 seeds 42–91 上为 50/50 成功；修复重点是 IK 冗余解
   使用关节限位和当前姿态，不是继续盲调方向步长。
2. Qwen 448px 直接方向基线为 3/4，但 grounding 框中心关系为 4/4；back 样本
   证明模型能给出正确空间表示却输出错误方向。因此 VLM 负责 grounding，代码负责
   确定性坐标或主轴比较，不再通过同义 prompt 反复修补方向标签。
3. seeds 42–46 的 20 张 grounding 均返回合法框，但整体 XY 误差 mean/median/max
   为 3.94/3.77/10.06cm。合法框和方向正确不等于厘米级定位通过，遮挡与固定系统
   偏差必须分开处理。
4. 使用 seeds 42–46 clear 样本拟合并冻结 `(+2.492cm, -1.947cm)` 补偿后，独立
   seeds 47–51 的 15 个 clear 样本补偿误差 mean/median/max 为
   0.77/0.79/1.46cm，15/15 不超过 3cm。唯一 severe 样本仍为 3.27cm，因此结论
   只适用于固定相机 clear 场景。
5. 在线 grounding targeting、runner 和无 API screening 已迁入正式子包；旧的全轨迹
   clear 筛选在 seeds 55–100 × left/right/front 共138个候选中得到0个合格，其中
   131个因可见率不足、7个因起始姿态误差失败。这证明“全程 clear”不是可用的在线
   前提，因此选择了短期目标保持，而不是扩大 seed 搜索或直接运行付费 smoke。
6. 最近可靠目标保持已实现：低可见率时复用已校验的补偿后 VLM 世界坐标，并根据
   当前末端位置重新计算动作；第1至第4次 held 允许。第5次仍不可见时先用当前末端位置
   做一次不执行动作的 `stale_target_recheck`：若已满足预测 stop 则正常结束，否则以
   `stale_target_limit` 中止。trace 区分 fresh、held 和过期复核。
7. 首轮真实 smoke 没有形成3案例成功率：`52-left` 首帧可见率不足，`54-front` 起始
   姿态误差为5.35cm，只有 `53-right` 进入在线闭环。该案例用1次真实 grounding 和
   4步 held 将真实 XY 距离从约10.05cm降至0.41cm，证明目标保持能产生有效在线运动；
   但控制器没有预测 stop，最终以 `stale_target_limit` 结束，不能据此声称端到端通过。
8. 在线 smoke 现在先对三个固定案例全部执行无 API 起点/首帧预检，再决定是否进入
   闭环。原案例真实 PyBullet 验证得到1/3合格，并完整记录两个拒绝原因；门禁只生成
   `smoke_preflight.json`，不创建 episode 证据。运行中后续遮挡仍使用4步 held，
   不受初始门禁替代。
9. Smoke 事后评分从本版本起拆成三层：`task_success` 只表示最终真实 XY 距离不超过
   3cm；`autonomous_stop_success` 只表示控制器自主预测 stop；主要 `success` 要求已经
   物理到达，且终止原因只能是 `success` 或 `stale_target_limit`。API、反投影、IK 等
   系统错误和 `max_control_steps` 即使最终位置碰巧达标也不能通过。真值仍只在循环结束
   后评分，不进入动作或 stop 决策。由于口径改变，旧批次与新批次的 `success_rate`
   不可直接合并比较。
10. 固定案例重选把资格改为起点姿态误差不超过5mm且首帧可见率不低于0.75。相同的
    seeds 55–100 × left/right/front 共有47/138个候选合格；按方向顺序选择最小且互异
    seed，冻结为 `56-left`、`55-right`、`59-front`。独立只预检得到3/3合格，三例
    姿态误差分别为1.14/4.67/2.44mm，可见率为0.754/0.929/0.807；全程没有调用 VLM
    API。
11. 第二轮真实 smoke 在上述三个固定案例上得到主要 `success=3/3` 和
    `task_success=3/3`，最终真实 XY 距离分别为0.303/0.789/0.371cm，总计调用 VLM
    API 4次。`55-right` 自主预测 stop；另外两例虽已到达，但以
    `stale_target_limit` 结束，因此 `autonomous_stop_success=1/3`。这证明固定相机、
    冻结校准和短期目标保持条件下的小规模模块化在线闭环已经打通，但不代表自主停止、
    severe 遮挡或大规模泛化已经解决。
12. 对第二轮 trace 的复查表明，`56-left` 最后一次 held 动作后已满足缓存目标的2cm
    stop 条件，但旧循环先触发过期而漏掉复核；`59-front` 则是 VLM 框中心相对真实
    红块约偏左5.4像素、偏上8.7像素，冻结补偿后仍留下约2.9cm的 Y 方向定位误差。
    前者已用 `stale_target_recheck` 修复；后者不能靠放宽 stop 阈值掩盖。新批次
    `passed` 除主要成功3/3外，还必须要求 `autonomous_stop_success=3/3`。历史输出
    不回写，因此旧 `passed=true` 只代表当时的旧门槛。严格 smoke 未通过不阻塞专家
    数据规模化，避免为单个 seed 过拟合 VLM。
13. `expert_v1` 规模化流程已通过真实10条 PyBullet pilot：episode seeds
    1000–1009，每条先复位7个关节的位置、速度和电机目标；10/10 success，共327帧，
    最终距离 min/mean/median/max 为2.907/2.955/2.952/2.998cm，每条帧数
    min/mean/median/max 为31/32.7/32.5/35。schema、动作维度、图片、重复 step、seed、
    帧数和终止标志共12项门禁全部通过，VLM API 调用为0。旧版50条/286帧数据保持不变。
14. 数据集质量报告保留严格 pilot gate，并新增 scale gate：达到目标规模后要求有效
    episode 至少300、总体成功率不低于99%、全部完整性错误为0，且红块 X/Y 五个分箱
    均非空。报告通过 `active_gate` 和顶层 `passed` 暴露当前阶段的机器判断结果。
15. 真实规模化采集以追加模式保留原10条 pilot，并新增 episode 10–299（seeds
    1010–1299）。最终300/300 success、9,894帧、0项完整性错误，最终距离
    min/mean/median/max 为2.871/2.942/2.941/3.000cm，每条帧数
    min/mean/median/max 为29/32.98/33/36；X/Y 五箱计数分别为
    `[2057, 2257, 2492, 1523, 1565]` 和 `[1830, 2650, 2129, 1876, 1409]`。
    `active_gate=scale`、顶层 `passed=true`，专家数据规模化阶段通过。
16. 对9,894帧执行数据集配置、seed、相机和控制步骤的确定性重放，并用透明机器人
    segmentation 生成逐帧无遮挡参考。9,888帧 JPEG 精确匹配，6帧只含严格容差内的
    OpenGL 舍入差异，0帧拒绝；clear/partial/severe 为
    9,479/411/4。300个首帧全部 clear，但55个终止帧为 partial，说明遮挡主要是机械臂
    接近目标时的任务内现象。当前保留全部成功轨迹和诊断标签，不把可见性审计等同于
    删除数据。
17. action tokenization 只读审计把9,894帧轨迹与可见性标签完整一一对齐，得到9,594个
    episode 内 transition；夹爪9,894帧均为1.0，终止标志300/9,894（3.03%）。固定规则
    在12个离散候选中选择32箱等频 absolute_q（最坏关节归一化 p95 重建误差2.71%、
    最小箱309帧）和64箱等频 delta_q（1.97%、149帧），并始终保留连续回归基线。
    所有等宽候选均因尾部箱样本不足等原因被拒绝。这只确定下一轮最小训练对照组；正式
    边界必须在 episode 级划分后只用训练集拟合，不能直接复用全量审计边界。
18. 在不修改原斜视数据的前提下，300条冻结轨迹已确定性派生为9,894张448×448固定垂直
    俯视图；状态、动作、终止标签逐项等价，质量门禁通过。真实可见性审计得到
    clear/partial/severe=7,024/477/2,393（70.99%/4.82%/24.19%），300个首帧全部 clear，
    300个终止帧全部 severe；相比原斜视的95.81% clear和0.04% severe，俯视并非天然更适合
    行为克隆。当前结论是保留两套视角，在训练前先选择斜视主基线、俯视对照或双视角，
    不把俯视派生成功误写成俯视单视角已经可用。
19. BC 训练三组对照结果：regression（连续回归）rollout 成功率 92%（46/50），
    absolute_q_32（32 箱离散分类）66%（33/50），delta_q_64（64 箱 delta 分类）
    0%。连续回归在当前数据规模（250ep）下是最优动作表示。离散分类路线没有死——66%
    成功率证明大 backbone + 更多数据后可追平。
20. 多任务 VLA 训练（图片 + 文本指令 → 动作）完成首轮实验。ResNet-18 + MiniLM 文本
    编码器（34M 参数）在 241ep 双任务数据上训练，rollout 成功率 24.5%。反事实实验
    （同 episode 用相反指令测试）得到 70% 指令跟随率，证明**语言信号确实驱动模型行为**。
    成功率低的主要原因是每任务数据不足（~120ep），而非架构缺陷。

详细证据与当时的设计边界：

- [Bug 与验证证据](../debugging/BUGLOG.md)
- [工程推进与学习复盘](../worklog/WORKLOG.md)
- [Grounding 闭环 smoke 设计](../superpowers/specs/2026-07-17-grounding-world-closed-loop-smoke-design.md)
- [Grounding 闭环 smoke 历史计划](../superpowers/plans/2026-07-17-grounding-world-closed-loop-smoke.md)

## 评价原则

- 每个阶段先定义固定 seeds、成功阈值、错误分类和证据输出，再运行实验。
- 方向准确率、grounding 合法率、世界坐标误差和闭环成功率是不同指标，不能互相
  替代。
- 校准集与验证集必须隔离；验证集不能反向拟合参数。
- clear、partial、severe 可见率分组分别汇总，不用整体均值隐藏遮挡失败。
- 安全中止证明保护机制有效，但仍计为任务失败。
- 源码与自动测试保护计算契约；本地输出目录保存真实实验结果，两者都不能由文档
  声明替代。

## 学习路线

当前路线：可诊断专家数据 -> 稳定 heuristic 闭环 -> VLM 能力边界 -> grounding
与世界坐标融合 -> 遮挡时的短期目标保持 -> 小规模在线闭环边界修订 -> 专家数据规模化
-> 逐帧视觉可学习性审计 -> action tokenization 审计 -> episode 级数据划分与候选实现
-> 固定多视角视觉证据 -> 视觉输入策略选择 -> BC 训练对照 -> BC rollout 评估 ->
多任务 VLA 数据采集与训练 -> 多任务 VLA rollout 与反事实语言验证 ->
**Qwen2-VL-2B QLoRA 微调（当前）**。

已到达：**VLA 架构验证通过**（语言信号确认驱动行为，70% 指令跟随率）。
下一阶段：方案 B——Google Colab 上完成 Qwen2-VL-2B QLoRA 多任务微调。
- 数据格式转换（现有 290ep → Qwen 训练格式）
- Colab notebook（4-bit 量化 + QLoRA + 训练循环）
- 下载 LoRA 权重 → 本地 rollout 评估
这直接对应学习计划第 9-10 周"轻量微调验证"。

## 关键入口

- 当前状态：[CURRENT_STATUS.md](CURRENT_STATUS.md)
- 文件路由：[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- 使用说明：[README.md](../../README.md)
- 核心配置：[sim_config.yaml](../../sim_config.yaml)
- 学习计划：[vla_robotic_study_plan.md](../planning/vla_robotic_study_plan.md)
- Bug 证据：[BUGLOG.md](../debugging/BUGLOG.md)
- 工作日志：[WORKLOG.md](../worklog/WORKLOG.md)
