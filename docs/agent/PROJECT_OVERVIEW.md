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
  、无 API 动态案例筛选和最多4步的最近可靠目标保持状态机。
- 正在推进：首轮真实在线 smoke 已执行但未通过；当前根据案例可运行性、事后成功评分
  与批次异常证据修订运行边界，在完成无 API 验证前不直接重跑付费闭环。
- 暂不覆盖：大型 VLA 训练、真实机械臂部署、复杂多物体任务、severe 遮挡恢复和
  正式大规模在线 VLM 评估。

## 系统数据流

### 专家数据基线

`sim_config.yaml` -> PyBullet 场景与随机红块 -> IK 目标和关节控制 -> RGB、动作和
本体状态 -> `outputs/dataset/` 中的轨迹与 episode 摘要。

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

正式源码采用 `src/` 布局，`pyproject.toml` 注册 12 个 `vla-*` 命令。新增文件和
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
5. 在线 grounding targeting、runner 和无 API screening 已迁入正式子包；动态筛选
   seeds 55–100 × left/right/front 共138个候选，合格数为0，其中131个因可见率不足、
   7个因起始姿态误差失败。这证明“全程 clear”不是可用的在线前提，因此选择了
   短期目标保持，而不是扩大 seed 搜索或直接运行付费 smoke。
6. 最近可靠目标保持已实现：低可见率时复用已校验的补偿后 VLM 世界坐标，并根据
   当前末端位置重新计算动作；第1至第4次 held 允许，第5次以
   `stale_target_limit` 中止。trace 和 summary 区分 fresh/held，完整离线测试为
   164/164 通过。
7. 首轮真实 smoke 没有形成3案例成功率：`52-left` 首帧可见率不足，`54-front` 起始
   姿态误差为5.35cm，只有 `53-right` 进入在线闭环。该案例用1次真实 grounding 和
   4步 held 将真实 XY 距离从约10.05cm降至0.41cm，证明目标保持能产生有效在线运动；
   但控制器没有预测 stop，最终以 `stale_target_limit` 结束，不能据此声称端到端通过。

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

当前路线是：可诊断专家数据 -> 稳定 heuristic 闭环 -> VLM 能力边界 -> grounding
与世界坐标融合 -> 遮挡时的短期目标保持 -> 小规模在线闭环边界修订 -> 专家数据规模化
-> action tokenization -> 轻量微调验证。是否进入下一阶段由当前阶段证据决定，不因
计划日期自动推进。

## 关键入口

- 当前状态：[CURRENT_STATUS.md](CURRENT_STATUS.md)
- 文件路由：[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- 使用说明：[README.md](../../README.md)
- 核心配置：[sim_config.yaml](../../sim_config.yaml)
- 学习计划：[vla_robotic_study_plan.md](../planning/vla_robotic_study_plan.md)
- Bug 证据：[BUGLOG.md](../debugging/BUGLOG.md)
- 工作日志：[WORKLOG.md](../worklog/WORKLOG.md)
