# VLA 项目工作日志

这份日志用我的视角记录这个项目从零开始的推进思路。重点不是贴代码，而是写清楚：我为什么这样做、每一步解决什么问题、我怎么判断当前阶段是否可以进入下一步。

## 0. 起点：先做一个足够小的闭环

我一开始没有直接去训练大模型，也没有直接做完整抓取。我的第一目标是先把一个最小任务跑通：

```text
让机械臂悬停在红色积木上方
```

我选择这个任务，是因为它足够简单，但已经包含 VLA 项目最核心的链路：

- 视觉里要能看到目标。
- 语言里要有明确任务。
- 机械臂要能根据目标位置产生动作。
- 数据里要能保存图像、动作和状态。
- 失败时要能判断问题在哪里。

这个阶段我接受动作不完美，但不能接受完全不知道为什么失败。

## 1. Baseline 0：先跑通，再谈优化

Baseline 0 的目标是让整个数据采集流程先能工作。我先关心这些基本问题：

- 仿真环境能不能正常加载机械臂和红色积木。
- 红色积木的位置能不能随机变化。
- 机械臂能不能通过 IK 移动到积木上方。
- 每一步能不能保存图片和动作标签。
- 保存出来的数据能不能对上当前任务。

这一阶段的核心思想是“先跑通”。只要图片、动作、目标位置和末端位置能连起来，我就有了后续优化的基础。

## 2. 把可调参数放进配置文件

跑通以后，我发现后面会频繁调整很多东西，比如采集多少条、相机位置、红块随机范围、成功阈值、是否覆盖旧数据等。

所以我把这些经常变化的内容集中放进 `sim_config.yaml`，让 Python 脚本主要负责流程逻辑。

这样做的好处是：

- 每次实验改了什么更容易复盘。
- 调参时不用反复改主逻辑。
- 后面如果数据不好，可以优先检查配置，而不是先怀疑整套代码。

我的原则是：能通过配置表达的实验选择，就不要写死在代码里。

## 3. Baseline 1：让数据变得可诊断

最开始只看图片和轨迹是不够的。即使机械臂没有成功，我也需要知道它为什么没成功。

所以我给数据加了最小可观测性，重点记录：

- 机械臂末端离目标点还有多远。
- 每条轨迹最后是成功、卡住，还是跑满步数。
- 每条 episode 的最终距离、最终积木位置和最终末端位置。

这一步的思想是：不要只收集数据，还要让数据能解释自己。

我在这个阶段形成了一个判断标准：

```text
不完美的数据可以继续优化；不可诊断的数据不能进入下一阶段。
```

也就是说，数据可以暂时不够好，但必须能告诉我下一步该改哪里。

## 4. 分析 stuck：先找原因，再改参数

当数据里出现 `stuck` 时，我没有直接把成功阈值放宽。因为那样可能只是把失败包装成成功。

我先看失败样本的最终距离和红块位置，发现 `stuck` 大致分两类：

- 一类已经很接近目标，只是差一点没过成功阈值。
- 另一类离目标还比较远，更像是目标位置对当前控制范围不稳定。

所以我优先调整红块采样范围，特别是 y 方向，让任务先落在当前机械臂更稳定的区域内。

这个阶段的经验是：失败不是一个标签，失败要继续拆开看。只有知道失败类型，调参才有方向。

## 5. 当前 Baseline 1 数据判断

当前这批数据的结果是：

```text
50 条 episode 全部 success
286 张图片全部能读取
图片尺寸统一为 224x224
抽样看红色积木和机械臂都在画面里
```

这次判断对应的是当前 `outputs/dataset/` 目录和当前 `sim_config.yaml` 的 Baseline 1 配置：

```text
block_position.y_range = [0.38, 0.5]
success_distance = 0.03
num_episodes = 50
```

从 `episode_summary.jsonl` 统计得到：

```text
final_distance min    = 0.0098m
final_distance median = 0.0293m
final_distance max    = 0.0300m
```

我的判断是：这批数据可以作为 Baseline 1 使用。

但我不会把它看成最终数据。原因是很多成功样本接近 3cm 成功阈值边缘，说明它是“可用的基线数据”，不是“高精度最终数据”。

所以当前不需要盲目继续采更多同类图片。更值得做的是进入下一步闭环验证，看看系统能不能从观测出发做连续决策。

## 6. 阶段三：为什么要做 probe

`src/vla_project/simulation/stage3_probe.py` 的目的不是继续采训练数据，而是验证在线闭环。

我想验证的是：

```text
观察当前状态 -> 判断下一步方向 -> 控制机械臂移动 -> 再观察 -> 再判断
```

这和 `src/vla_project/simulation/control_arm.py` 不一样：

- `src/vla_project/simulation/control_arm.py` 是离线专家数据采集。
- `src/vla_project/simulation/stage3_probe.py` 是在线闭环探路。

我先用启发式规则，而不是一开始就接大模型。原因是如果简单规则都不能让机械臂接近目标，那问题很可能在控制逻辑、坐标映射或仿真参数，而不是模型能力。

这个阶段的思想是：先用最可控的规则验证系统骨架，再把决策部分换成模型。

## 7. 方向词和坐标移动的思想

在 probe 里，我让决策结果先保持很简单，只输出：

```text
left / right / front / back / stop
```

这样做是为了降低阶段三的复杂度。模型或规则先不用直接生成连续动作，只需要判断大方向。

然后系统再把方向词转换成 PyBullet 世界坐标里的移动方向：

- `left/right` 对应 x 方向。
- `front/back` 对应 y 方向。
- `stop` 表示已经足够接近。

这里要特别注意：当前 heuristic 使用的是 PyBullet 世界坐标里的末端位置和目标位置差值，不是直接理解图像中的“左/右/前/后”。后面接入 VLM 时，模型看到的是相机图像，所以必须校准“图像方向”和“世界坐标方向”的关系，否则模型可能视觉判断正确，但控制方向映射错误。

这里的关键思想是把问题拆开：

- 决策层只负责判断方向。
- 控制层负责把方向变成实际目标位置。
- 配置文件负责控制每次移动多远。

这样后面如果移动太大或太小，我只需要调步长；如果方向错了，我再检查决策逻辑。

## 8. 为什么要记录末端位置和目标距离

probe 每一步都要记录机械臂末端位置和它离目标悬停点的距离。

我这样做是因为闭环控制不能只看“有没有输出方向”。真正重要的是：

```text
每一步动作之后，机械臂有没有更接近目标？
```

如果方向一直在输出，但距离没有明显下降，就说明闭环并没有真正工作。问题可能在：

- 初始末端位置太远或太高。
- 每次动作执行的仿真步数太少。
- 移动步长不合适。
- 方向映射和真实坐标理解不一致。
- 当前策略只处理 x/y，没有处理 z 方向。

所以 `distance_to_hover` 是阶段三最重要的诊断指标之一。

## 9. 当前 probe 判断

审查旧版 probe 时发现过一个控制路径 bug：heuristic 虽然输出了 `front/right/left/back`，执行分支却直接把完整 `hover_target` 交给 IK。因此旧版 13 步、`1.1567m -> 0.0161m` 的结果只能证明 IK 能接近目标，不能证明方向词真正控制了动作。

修复后 heuristic 和 API 统一经过 `direction -> PyBullet 世界坐标 delta -> 单步 target_pos`。当前有效证据来自重新生成的 `outputs/probe/probe_trace.jsonl`：

```text
63 步闭环控制
世界坐标方向输出：front 23 次，left 40 次
distance_to_hover 从约 1.1524m 降到约 0.0241m
termination_reason: success
```

我的判断是：修复后的 probe 已经不只是空框架，方向词现在确实决定每一步 x/y 世界坐标位移，并在这次样例中把机械臂带到了目标悬停点附近。

但这还不能说明阶段三已经完成。因为单次成功不等于稳定成功。下一步不应该马上把重点放到大模型 API，也不应该继续盲目采更多图片，而应该把 probe 变成可以批量评估的实验系统。

我现在更关心这些问题：

- 随机 50-100 次红块位置时，启发式闭环成功率是多少。
- 成功样本平均需要多少步。
- 失败样本是不是集中在某些 x/y 位置。
- `distance_to_hover` 是否大多数时候单调下降，还是存在震荡。
- `front/back/right/left` 的方向映射是否在不同相机扰动下仍然可靠。

这个判断也解释了为什么要准备 VLM 决策替换接口：不是为了立刻依赖大模型，而是先把“决策层”和“控制层”拆开。这样后面可以在同一套 PyBullet 控制和评估框架里公平比较：

```text
heuristic 规则
API 多模态模型
本地微调模型
```

只有先有统一接口和统一评估指标，后面的 VLM 接入才不是简单 API 拼接，而是一个真正可比较、可复盘的 VLA 闭环实验。

## 10. 首批批量评估与失败诊断

在修复单次方向执行后，我新增了 `src/vla_project/simulation/evaluate_probe.py`，用固定种子 42 连续评估 20 个随机 episode。配置为 `max_control_steps=80`、`sim_steps_per_action=60`、`move_step_xy=0.03`，结果是：

```text
success: 5/20
success_rate: 25%
termination_reason=max_control_steps: 15 个 episode
error: 0
final_distance mean / median / max: 0.4989 / 0.5948 / 0.8137 m
control_steps mean / median: 72.05 / 80
```

这说明批量评估系统已经跑通，但 heuristic 还不稳定。失败 trace 中存在 left/front 往复切换和距离反复增大的现象；下一步应该先按失败位置与动作序列分析振荡原因，再做单变量控制调优，而不是直接进入 VLM。

关节级 trace 最终把问题定位到 IK：失败振荡阶段，第 4 个关节的 IK 目标约为
`2.1713rad`，但 KUKA URDF 上限只有 `2.0944rad`。其他 6 个关节最大跟踪
误差约 `0.00018rad`，只有第 4 关节持续撞限位。于是期望的单轴 `left` 动作
产生额外 y 偏移，heuristic 再用 `front` 纠偏，形成往复振荡。

我没有继续放宽阈值，而是在 IK 中传入关节上下限、活动范围以及当前关节角
`restPoses`，让 7 自由度机械臂沿连续、靠近当前姿态的冗余解移动。完整诊断
过程记录在 `../debugging/BUGLOG.md`。

## 11. Stage 3 heuristic 稳定性验收

修复后先用相同 20 个固定种子回归，结果从 `5/20` 提升为 `20/20`。随后按
学习计划把评估扩大到固定种子 42-91，共 50 个 episode，配置保持：

```text
max_control_steps = 80
sim_steps_per_action = 60
move_step_xy = 0.03m
success_distance = 0.03m
```

正式验收结果来自 `outputs/probe_evaluations/run_20260712_221135/`：

```text
success: 50/50
success_rate: 100%
failure / error: 0 / 0
final_distance mean / median / max: 0.0193 / 0.0197 / 0.0291m
control_steps mean / median: 37.58 / 37
direction: front 243 次，right 1636 次
```

这个结果达到“至少 50 个随机 episode、成功率 80% 以上、失败可诊断”的
Stage 3 门槛。当前可以把 heuristic 看作稳定的控制基线，不需要继续盲目调
步长或仿真执行时间。

当前项目阶段可以这样总结：

- Baseline 0 已完成：采集闭环已经跑通。
- Baseline 1 基本可用：数据能读、图片能看、episode 全部成功。
- 阶段三 heuristic 稳定性验收完成：50 个固定种子全部成功。
- 关节限位故障已经形成“现象、单变量排除、关节级证据、修复、回归”的完整案例。

下一步我应该优先做：

- 保留 heuristic 作为稳定基线和故障排查用 sanity check。
- 让 VLM 只根据相机图像和语言指令输出离散方向，不读取红块真实坐标。
- 保持现有 PyBullet 控制层和 50 种子评估口径不变，公平比较 heuristic 与 VLM。
- 记录 VLM 原始回复、非法输出、方向分布、成功率和失败 trace。

这条路线符合我现在的项目原则：

```text
先跑通，再优化；进入下一阶段前，先保证系统可诊断。
```

## 12. 更新后的学习路线：工程成果 + 模型深度平衡

我选择的路线不是最快做 demo，也不是马上冲大模型训练，而是先把项目做成可复现、可评估、可解释的闭环系统，再逐步加入 VLM 决策和动作 tokenization。

后续路线是：

```text
Stage 3 probe 稳定化
-> 自动化闭环评估
-> VLM 决策替换接口
-> 专家数据集规模化和质量分析
-> action tokenization
-> 轻量 LoRA/QLoRA 微调验证
```

前两步现在已经完成：heuristic probe 通过 50 次稳定性验收，成功率、最终距离和控制步数都能被统一统计。接下来可以进入 VLM 决策替换，因为现在已经有可靠基线判断模型输出到底带来了提升还是退化。

中期重点是 VLM 决策替换接口。这个接口的价值是让项目结构接近真正的 Vision-Language-Action：

```text
图像 + 语言指令 -> 决策模块 -> 动作方向 -> PyBullet 控制
```

这个结构允许我在同一套控制和评估系统中比较 heuristic、API VLM 和未来本地微调模型。

长期深度点是 action tokenization 和轻量微调。即使不一开始训练完整 VLA，也要能说明如何把连续动作离散成 token，如何构造 image-instruction-action 数据格式，以及如何用 LoRA/QLoRA 做小规模验证。

## 13. 简历复盘可提炼点

这份项目日志后续可以直接服务简历和面试复盘。当前最值得提炼的点不是“我调用了某个库”，而是我如何把一个模糊的 VLA 项目拆成可验证的工程系统。

可以提炼为以下几类：

- 工程决策：没有一开始训练大模型，而是先跑通 PyBullet 专家采集闭环，避免在数据和控制都不稳定时盲目上模型。
- 数据质量：通过 `episode_summary.jsonl` 记录 `termination_reason`、最终距离、最终目标位置，让每条轨迹都能被诊断。
- 失败分析：遇到 `stuck` 时没有直接放宽成功阈值，而是先拆分失败类型，再根据红块位置和最终距离调整采样范围。
- 配置驱动：把随机范围、成功阈值、相机参数、采集数量等实验变量放进 `sim_config.yaml`，让实验可复盘。
- 闭环控制：用 `src/vla_project/simulation/stage3_probe.py` 验证“观察 -> 决策 -> 控制 -> 再观察”的在线闭环，而不是只停留在离线数据采集。
- 模型接口：先保留 heuristic 作为 sanity check，再准备 VLM 决策替换接口，使 heuristic、API VLM 和未来本地微调模型可以在同一套评估框架下比较。
- 深度扩展：后续通过 action tokenization 和 LoRA/QLoRA 微调验证，把项目从规则控制推进到真正的 image-instruction-action 学习问题。

如果写进简历，可以压缩成一句：

```text
构建 PyBullet 机械臂 VLA 仿真系统，完成专家轨迹采集、RGB-语言-action 数据对齐、episode-level failure analysis 和在线闭环评估，并设计可插拔 VLM 决策接口与 action tokenization 微调路线。
```

## 14. Alibaba Qwen VLM 离线方向决策接入（2026-07-14）

Stage 3 heuristic 通过 50 次稳定性验收后，我开始把决策来源替换为阿里云
`qwen3-vl-flash`。这一步没有修改已经验证过的 PyBullet 控制层，而是让 VLM
只完成一个受约束的视觉任务：

```text
相机图片 + “悬停在红色积木上方”
-> screen_left / screen_right / screen_up / screen_down / stop
-> 确定性映射为 PyBullet 世界坐标动作
```

API 地址、Key 和模型名称通过环境变量读取，没有写入代码或日志。模型回复如果
不包含合法标签会明确报错，限流和暂时性服务错误可以重试，认证错误不会盲目重试。
这些行为均有单元测试覆盖。

为了避免每次改 prompt 都重新运行仿真，我新增了两段离线工具链：

- `src/vla_project/vlm/collect_vlm_eval_samples.py`：使用固定种子的 heuristic trace 生成图片和标准方向。
- `src/vla_project/vlm/evaluate_vlm_decisions.py`：只读取固定图片，调用 VLM，保存逐样本预测并支持断点续跑。

当前离线样本共 50 张，每个样本只向模型提供图片和任务指令；`block_pos`、
`ee_pos` 等仿真真值只用于采集标准答案，不进入模型 prompt。评估指标包括合法
输出率、exact-match accuracy、分类指标、混淆矩阵、错误类型和 API 延迟。

### 14.1 从低准确率到定位视觉根因

首轮实验不是简单地反复改 prompt，而是逐步分离输出格式、方向映射、相机透视、
目标遮挡和末端识别问题。主要结果如下：

```text
v1  世界坐标标签：10 张准确率 10%，模型大量输出 stop
v2  收紧 stop 规则：10 张准确率 50%
v3  明确图像方向到世界方向：单张准确率 0%
v4  模型输出 screen_*、代码确定性映射：首张 100%，10 张只有 10%
v5  正俯视：红块在部分步骤被机械臂遮挡，未调用 API
v6  倾斜俯视：仍存在单视角遮挡，未调用 API
v7  双视角拼接：单张准确率 0%，模型可能跨面板配对目标
v8  增加 MAIN/AUX 标题和分隔线：单张仍为 0%
v9  3m 高位正俯视单视角：首张 100%，10 张准确率 70%
```

关键诊断不是来自猜测。使用 PyBullet 真值把真实末端和红块投影回 224×224
画面后发现，旧 v7/v8 主视图中真实末端的纵坐标约为 `y=313`，已经超出画面；
画面里的黑色圆环是中间关节，不是真实末端。补充视图虽然能看到真实末端，
红块却被机械臂遮挡。因此旧双视角没有任何一个面板同时清楚包含“真实末端 + 红块”，
继续修改文字描述无法解决输入信息缺失。

我随后用本地仿真对比 4m 正俯视、4m 倾斜俯视和 3m 正俯视。最终选择：

```text
camera eye_offset_base = [0.0, 0.0, 3.0]
camera up_vector       = [0, 1, 0]
camera fov             = 45
use_dual_view          = false
```

3m 正俯视能够让初始和中间阶段的真实末端与红块同时进入单张 224×224 图片，
同时避免双面板交叉配对。对应结果目录为：

```text
outputs/vlm_evaluations/offline_qwen3_vl_flash_high_topdown_v9/
```

10 张评估结果：

```text
合法输出率：100% (10/10)
exact-match accuracy：70% (7/10)
平均 API 延迟：约 1.20s
标准方向分布：front 1 张，right 9 张
right recall：6/9 = 66.7%
```

这证明相机可见性是早期连续失败的主要原因：首张样本从 v7/v8 的错误
`screen_left` 变为正确的 `screen_up -> front`。但 70% 还不能证明策略稳定，
当前 10 张样本方向分布也不均衡。3 个错误样本中，前两个仍可能选错机械臂部件
或视觉方向，最后一个发生在末端接近目标并遮住红块时。

### 14.2 当前阶段判断

VLM 离线接口、固定样本、断点续跑和统计链路已经跑通；高位正俯视把 10 张准确率
提高到 70%，但当前仍属于“视觉基线验证中”，还不能进入付费在线闭环正式验收。

下一步应保持 v9 的 70% 结果作为无标记基线，在真实末端 `link 6` 上增加醒目的
绿色视觉标记，再用同一批样本和同一评估口径做消融对照。如果标记版显著提高
准确率，说明剩余瓶颈主要是末端部件识别；如果没有提高，再继续检查遮挡样本和
标准方向的视觉可判定性，而不是继续堆叠 prompt。

## 15. 从直接方向预测转向目标感知与本体状态融合（2026-07-15）

在 v9 之后，我继续完成了三个受控诊断：

```text
v10 绿色末端标记：10 张准确率从无标记 v9 的 70% 降到 50%
v11 四方向均衡小样本：4 张准确率 50%，left/back 被误判为 right
v12 连接链结构 prompt：4 张准确率仍为 50%，输出与 v11 完全相同
```

这些结果不能说明直接方向方案已经失败：v9 的 70% 证明它具有潜力，但该批次中
9/10 样本都是 `right`；v11/v12 又只有每方向 1 张。当前准确定位应是“受限但有
潜力的直接方向基线”，后续保留为消融对照，而不是直接放弃。

这说明剩余问题不能靠增加醒目颜色或继续堆叠末端描述解决。为了直接观察模型选中
的参照物，我新增 `src/vla_project/vlm/diagnose_vlm_grounding.py`，让 Qwen 分别返回机械臂末端与红块
的归一化框，并由代码绘制诊断图。四张图片均返回合法框，模型基本能找到机械臂
连接链末梢；真正的问题是接近目标后两个框明显重叠，部分框中心的图像方向与
PyBullet 世界动作标签不一致。

### 15.1 为什么调整主路线

最终目标不是让通用 VLM 猜中每一个低级方向词，而是构建稳定、可解释、能够继续
产生 action-token 训练数据的 VLA 系统。真实机器人本身可以通过关节编码器和正向
运动学获得末端状态，因此在仿真中使用 robot joint state 或 `getLinkState()` 获取
自身末端位置属于本体感知，不是作弊。作弊的是直接读取红块世界坐标，或把标准
动作传给模型。

新的主路线为：

```text
图像 + 语言指令
-> VLM 定位指令指定的红块
-> 相机标定把红块框中心反投影到工作平面
-> 机器人本体状态 / FK 提供末端位置
-> 几何控制器计算方向并沿用现有 IK 闭环
```

VLM prompt 仍然不能接收 `block_pos`、`ee_pos` 或标准答案；红块 PyBullet 真值只
允许用于离线定位误差评分。原“VLM 直接输出 screen direction”路径不会删除，
而是作为消融基线，用来量化模块化融合为什么更稳定。

### 15.2 对最终 VLA 阶段的帮助

融合闭环成功后，每一步可以保存：

```text
image + instruction + proprioceptive state + target grounding + delta action
```

这些字段既能支持失败分析，也可以转换成 action token 数据。后续先验证动作离散化
的重建误差，再用小样本 LoRA/QLoRA 检查模型是否能从图像、指令和本体状态预测
动作 token。这样项目形成“直接 VLM 方向基线 -> 模块化融合闭环 -> 学习式动作策略”
的递进关系，而不是把 API 调用夸写成端到端 VLA。

下一步只实现一个可测试的基础模块：将 VLM 红块框中心通过已知相机内外参反投影到
桌面世界坐标，并用离线 `block_pos` 仅做误差评分；在定位误差达标前不进入在线控制。

## 16. 直接方向基线的距离分层评估 v13（2026-07-15）

为了避免因 v9 类别不均衡或 v11/v12 样本过少而过早放弃直接方向方案，我保持
模型、3m 正俯视相机和 v12 prompt 不变，只改变末端与红块的平面距离：

```text
距离：0.20m / 0.10m / 0.05m
每档：left / right / front / back 各 1 张
总计：12 张，固定 seed 42
```

结果：

```text
合法输出率：100% (12/12)
exact-match accuracy：50% (6/12)
0.20m：2/4，50%
0.10m：2/4，50%
0.05m：2/4，50%
left recall：0/3
right recall：3/3
front recall：3/3
back recall：0/3
平均 API 延迟：约 0.97s
输出：outputs/vlm_evaluations/offline_qwen3_vl_flash_distance_stratified_v13/
```

三个距离得到完全相同的方向模式：`left/back` 都被预测为 `right`，`right/front`
全部正确。即使 0.20m 图片中的红块与机械臂末梢已经明显分开，错误仍然存在。
因此当前直接方向基线的限制不是“样本距离太近”，而是模型在单次方向回答中存在
稳定的参照物选择或方向推理偏置。下一步若继续研究这条路线，应比较“VLM 先输出
目标框后由代码比较中心”和“VLM 直接输出方向”，而不是继续放大距离或堆叠 prompt。

### 16.1 20cm 画框对照

继续对 v13 的 0.20m 四方向图片运行末端与红块画框诊断，4/4 返回合法 JSON：

```text
left：两个预测框的相对中心关系支持 left；直接方向仍错误输出 right
right：框中心关系支持 right；直接方向正确
front：框中心关系以向上偏差为主；直接方向正确输出 front
back：红块框没有稳定覆盖真实红块，框中心关系也不支持 back；直接方向错误
输出：outputs/vlm_evaluations/grounding_qwen3_vl_flash_distance20_v2/
```

这证明 50% 不是一个单一故障。`left` 是“预测框的相对中心关系已经支持正确方向，
但最终方向标签推理错误”；`back` 则包含目标定位或遮挡错误。预测框本身只是近似框，
不能当成像素级准确定位。把 grounding 和方向选择拆开有望修复 `left` 这一类推理
错误，但仍需要针对 `back` 的目标可见性和时序信息继续处理。

本地用纯红像素阈值检查原图后，红块主体在 224×224 图片中只有约 `10×9` 像素。
left/front 等姿态约有 87 个纯红像素，而 back 只有 50 个，约 43% 的纯红区域因机械臂
姿态被遮挡；VLM 在 back 中给出的 `red_block` 框与纯红主体没有重合。因此 back 的
直接原因已收窄为小目标和局部遮挡，而不是方向映射。

## 17. 448px 分辨率消融 v14（2026-07-15）

保持 seed 42、20cm 距离、3m 正俯视、FOV、模型和 prompt 不变，只把输入从
224×224 提升到 448×448。使用独立目录保存图片，避免覆盖 v13：

```text
样本：outputs/vlm_samples/448/
方向结果：outputs/vlm_evaluations/offline_qwen3_vl_flash_distance20_448_v14/
画框结果：outputs/vlm_evaluations/grounding_qwen3_vl_flash_distance20_448_v3/
```

像素测量显示红块主体从约 `10×9` 增加到 `20×18`；back 的纯红像素从 50 增加
到 220。直接方向结果：

```text
合法输出率：100% (4/4)
exact-match accuracy：75% (3/4)
left/right/front：正确
back：仍错误输出 right
```

相比同一 seed、同一 20cm 距离的 224px 结果 2/4，448px 把 `left` 修正为正确，
证明输入分辨率确实影响直接方向能力。

448px 画框诊断中，四组预测框的相对中心关系全部支持标准方向；尤其 back 的红块框
已经位于末端框下方，目标定位不再是该帧的主要问题。若由代码比较两个框中心并选择
偏差更大的轴，这四张为 4/4；但模型直接方向只有 3/4。因此剩余 back 错误已经收窄
为“grounding 表示正确但最终方向推理不一致”。

下一步如果继续研究直接方向路线，应测试一次结构化 `ground-then-decide`：要求模型
在同一回复中先给两个框，再给方向，并检查方向是否与它自己的框中心一致。模块化
融合路线则可以直接由代码从框中心计算动作，避免再次引入这一推理错误。

### 17.1 Ground-then-decide 内部一致性实验

新增 `src/vla_project/vlm/evaluate_ground_then_decide.py`，要求同一个 API 回复同时包含两个归一化框和
一个 `screen_*` 方向；代码独立根据框中心主轴计算第二个方向，用于检查模型是否
遵循自己输出的空间关系。448px 四方向结果：

```text
模型最终方向：3/4
框中心推导方向：4/4
模型方向与框中心内部一致：3/4
输出：outputs/vlm_evaluations/ground_then_decide_qwen3_vl_flash_448_v1/
```

left/right/front 三张完全一致且正确。back 回复为：

```text
末端框中心约：(571, 340)
红块框中心约：(567, 431)
dx 约 -4，dy 约 +91
代码主轴方向：screen_down -> back
模型最终方向：screen_left -> left
```

back 的纵向偏差远大于横向偏差，模型却没有执行 prompt 中明确规定的主轴比较。
这直接证明框定位和低级方向选择应拆开：VLM 提供有语义的目标感知，确定性代码
完成坐标比较。直接方向路径仍保留为 75% 对照基线，但不再继续通过增加文字规则
解决已经被证明的算术一致性问题。

当前阶段结论：直接方向基线已经完成能力边界诊断；下一步进入目标坐标接口，先实现
红块框中心到工作平面世界坐标的相机反投影，并用红块仿真真值只做离线误差评分。

## 18. 五位置 grounding 反投影稳定性实验（2026-07-16）

这轮不是继续评价 `left/right/front/back` 是否正确，而是检查 VLM 红块框中心能否
转换成控制器可用的厘米级世界坐标。保持正俯视相机、448×448、20cm 相对距离、
模型和 prompt 不变，使用 seeds 42–46；每个红块位置拍 left/right/front/back
四种机械臂姿态，共 20 张。

为避免把遮挡误认为随机框抖动，我使用 PyBullet segmentation mask 比较“当前姿态”
和“移除机械臂后的红块完整投影”，把可见率只写入 `diagnostics.jsonl`。它不进入
VLM 输入。实际可见率为：15 张 left/right/front 全部 `1.0`；back 分别为
`0.640/0.100/0.214/0.249/0.360`。按 `clear>=0.75`、`partial>=0.25`、
`severe<0.25` 分组后，得到 15/2/3 张。

输出目录：

```text
样本：outputs/vlm_samples/448_multiseed_d020/
grounding：outputs/vlm_evaluations/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/
反投影：outputs/vlm_evaluations/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/backprojection/
```

20/20 返回合法框，`num_failed=0`。整体指标：

```text
localization_error_xy mean / median / max:
0.03943699 / 0.03769951 / 0.10062393 m
signed_error_x_mean: -0.02682766 m
signed_error_y_mean: +0.02244968 m
```

逐方向 mean：left `3.10cm`、right `3.55cm`、front `3.37cm`、back `5.76cm`。
逐可见率分组：

```text
clear:   15/15 valid, mean/median/max = 3.34/3.45/4.00cm
partial:  2/2  valid, mean/median/max = 4.02/4.02/4.41cm
severe:   3/3  valid, mean/median/max = 6.92/6.50/10.06cm
```

最差的 `seed_43_d020_back` 可见率只有 `0.100`，定位误差 `10.06cm`。因此遮挡
确实会显著放大 grounding 误差，不能和清晰样本混成一个无法解释的均值。

清晰样本的 15 个 X 误差全部为负、Y 误差全部为正，平均偏差为
`(-2.49cm, +1.95cm)`，说明还存在稳定系统偏差。同一批数据减去该均值后，探索性
残差 mean/median/max 为 `1.17/1.20/1.97cm`；但这是在拟合数据本身上计算，存在
数据泄漏，不能作为校准验收成绩。

结论：先建立独立校准集与验证集，只用校准集估计清晰样本偏差，再在验证集检查
是否稳定低于 3cm。严重遮挡不参与固定偏差拟合，后续单独测试双视角、历史帧或
主动避让。在这两类问题分开验收前，不进入在线控制。

## 19. Grounding 固定偏差独立验证（2026-07-16）

上一轮 seeds 42–46 的同集残差只能说明固定补偿值得测试，不能证明泛化。本轮把
旧结果固定为校准集，只用其中 15 条 clear 记录拟合一次 XY 补偿：

```text
correction_x = +0.02492227406480192 m
correction_y = -0.019467343494422532 m
```

然后使用完全独立的 seeds 47–51；每个位置仍拍 left/right/front/back，共 20 张。
验证脚本先检查校准 ID 和验证 ID 的交集为空，再只读应用冻结补偿，禁止从验证结果
重新估计参数。新数据的可见率分组为 clear/partial/severe=`15/4/1`，Qwen 为 20/20
图片返回合法框，反投影也为 20/20 有效。

输出目录：

```text
样本：outputs/vlm_samples/448_calibration_validation_d020/
grounding：outputs/vlm_evaluations/grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1/
反投影：outputs/vlm_evaluations/grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1/backprojection/
校准验证：outputs/vlm_evaluations/grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1/calibration_validation/
```

独立验证指标：

```text
overall raw mean/median/max:       3.32/3.39/5.31cm
overall corrected mean/median/max: 1.07/0.87/3.27cm（19/20 <=3cm）

clear raw mean/median/max:          2.99/3.13/3.92cm
clear corrected mean/median/max:    0.77/0.79/1.46cm（15/15 <=3cm）

partial corrected mean/median/max:  1.66/1.63/2.64cm（4/4 <=3cm）
severe corrected error:             3.27cm（0/1 <=3cm）
```

因此预先定义的验收结果是 `passed=true`：固定偏差确实能泛化到新的清晰红块位置，
不是同集数据泄漏造成的假改善。最差 clear 样本 `seed_50_d020_right` 也只有
`1.46cm`。但唯一 severe 样本仍超过 3cm，所以这轮解决的是固定相机、清晰视野下的
系统偏差，不是遮挡问题。

下一步不扩大付费离线样本，而是先设计一个小规模、可中止的闭环 smoke test：运行时
只允许 VLM grounding、相机反投影、冻结补偿和机械臂本体状态参与控制，PyBullet
红块真值继续只做事后评分；同时对 severe 可见率保留拒绝或恢复策略，不把本轮
`passed=true` 外推到遮挡场景。

### 19.1 今日工程收尾与下一阶段约束

本轮不仅生成了实验结论，也把验证过程固化为可重复执行的工程接口：采样器使用显式
seeds 47–51；`src/vla_project/vlm/validate_grounding_calibration.py` 负责冻结校准、检查数据集 ID 隔离、
输出逐样本补偿结果和严格通过判定；新增测试覆盖空 clear、clear 定位失败、超过 3cm、
校准/验证 ID 重叠和遮挡分组。最终完整测试为 `98/98` 通过，相关代码和四份项目文档
已通过功能分支快进合并回 `main`。20 张原图、Qwen 标注图、反投影结果和校准验证
摘要继续保存在上述本地证据目录，不提交批量图片到 Git。

对下一阶段 smoke test，当前已确认两项约束：

```text
场景范围：第一轮只使用红块清晰可见的 3 个 episode，不混入遮挡恢复实验。
通过标准：每个 episode 最多 10 个控制步，3/3 都达到真实 XY 误差 <=3cm。
```

API、框解析、反投影、工作区边界或运动趋势检查失败时必须立即安全中止；安全中止只
证明保护机制工作，不算任务成功。下一次继续设计时，还需要确定每一步是否都重新调用
Qwen。确定完整设计并审查通过前，不改在线控制代码。

## 20. Grounding 闭环前置筛选与遮挡恢复决策（2026-07-17 至 2026-07-18）

闭环 smoke runner、真值隔离动作策略和无 API 动态筛选已经在
`feat/grounding-world-smoke` worktree 中实现。为了先验证 clear-only 闭环，我曾要求
候选轨迹从约 10cm 起点到约 2cm 终点的五个观察点全部满足红块可见率 `>=0.75`，并
在 seeds 55–100 的 left/right/front 三个方向上执行确定性筛选。

真实筛选结果为：

```text
候选总数：138
合格案例：0
visibility_below_threshold：131
start_pose_error：7
筛选结论：passed=false
```

证据保存在：

```text
vlm_smoke_screening_runs/run_20260717_231406/screening_summary.json
vlm_smoke_screening_runs/run_20260717_231406/candidate_trace.jsonl
```

轨迹位置误差大多约为毫米级，主导失败不是机械臂没有按计划移动，而是末端靠近红块后
从固定正俯视相机方向遮挡目标。因此，继续扩大 seed 范围寻找“全程无遮挡”案例会把
系统性遮挡藏起来，不能作为当前主路线。

本轮最终选择“最近可靠目标坐标 + 最大失效步数”，不选择“重复上一动作方向”：

```text
清晰且定位有效：VLM 框 -> 反投影 -> 冻结补偿 -> 更新目标缓存
短暂不可见：缓存的 VLM 目标坐标 - 当前末端坐标 -> 重新计算本步动作
连续保持上限：4 步
第 5 次仍不可见：stale_target_limit，停止且不再执行动作
```

这个机制保存的是 VLM 曾经产生的目标估计，不是 PyBullet 红块真值，也不是旧动作。
每一步仍根据 `getLinkState` 得到的当前末端位置重新判断主误差轴和 `stop`。清晰画面下
出现 API、无效框、反投影、工作区、目标跳变、无进展或 IK 错误时仍按原原因中止，
不能用缓存目标掩盖其他故障。

下一步只实现并验证该状态机：先用自动测试证明 fresh/held 切换、4 步边界、日志字段
和真值隔离，再重新运行三个固定在线 smoke cases。双相机、主动避让和长期遮挡恢复
仍作为后续独立对照，不在本轮同时加入。

## 21. Python 源码按标准包结构整理（2026-07-18）

为避免根目录同时混放配置、正式代码和维护脚本，源码已统一迁移到标准 `src/`
布局：仿真与控制放在 `src/vla_project/simulation/`，VLM 评估放在
`src/vla_project/vlm/`，仓库维护工具放在 `src/vla_project/tools/`；测试继续独立
放在 `tests/`，并按相同领域镜像分组。

新增 `pyproject.toml` 和 10 个 `vla-*` 命令入口。开发环境执行 `pip install -e .`
后，可以用 `vla-collect`、`vla-probe`、`vla-evaluate-probe` 等稳定命令运行流程，
不再依赖根目录 Python 文件名。迁移过程中保持业务逻辑不变，包导入、命令元数据和
原有行为测试合计 `109/109` 通过。

## 22. Grounding smoke 合入正式包结构（2026-07-19）

将 `feat/grounding-world-smoke` 已有的 targeting、runner 和无 API screening 与主分支
`src/` 迁移结果合并，正式放入 `src/vla_project/vlm/grounding_smoke/`，测试镜像到
`tests/vlm/grounding_smoke/`。新增 `vla-run-grounding-smoke` 和
`vla-screen-grounding-smoke` 两个命令，所有新输出统一进入
`outputs/vlm_evaluations/`。

本轮没有调用真实 API。新增契约要求校准文件精确匹配 seeds 42–46 的15个固定 ID，
smoke 专用 API `max_retries=0`，从而保证3个 episode × 10步就是最多30次真实 HTTP
请求；模型框解析失败记录为 `invalid_box`，与网络/API 的 `api_error` 分开。迁移前旧
分支基线为143/143通过，迁移及新增契约完成后的全量测试为157/157通过。

动态筛选的0/138结果仍然有效，说明不能直接进入付费 clear-only smoke。下一步按已确认
设计实现“最近可靠 grounding 目标 + 最多4步保持”，完成 mock 验证后再请求真实实验
批准。

## 23. 最近可靠 Grounding 目标保持状态机（2026-07-19）

本轮按既有设计以 TDD 实现短时遮挡恢复，没有调用真实 API。`sim_config.yaml` 冻结
`max_stale_target_steps=4`；`targeting.py` 抽出“已校验世界目标 + 当前末端位置 ->
安全动作”的纯计算入口，fresh 与 held 共用同一方向、停止距离和无进展规则。缓存只
保存已通过反投影、冻结补偿、工作区和目标跳变检查的 VLM 世界坐标，不保存旧动作，
也不读取 PyBullet 红块真值。

runner 现在维护最近可靠目标和连续失效年龄。首帧低可见且没有缓存时仍以
`visibility_out_of_scope` 中止；第1至第4次低可见步骤不调用 VLM，而是使用当前
`ee_pos` 重新计算动作；第5次尝试以 `stale_target_limit` 中止且不执行动作。新的
fresh 定位会更新缓存并把年龄重置为0。trace 和 episode summary 增加
`decision_source`、`target_age_steps`、`used_target_hold`、`api_called`、fresh/held
步数、最大目标年龄和是否从遮挡恢复等证据字段。

批次通过条件不再要求 `all_clear=true`，改为三个固定案例全部成功、每个至少一次
fresh VLM 定位、目标年龄不超过4且总 API 调用不超过30。新增测试按 RED->GREEN
验证首帧拒绝、缓存复用、4步边界、第5步停止、重新定位重置、真值隔离和批次汇总；
独立代码审查发现 held 纯计算路径的未知异常曾被误记为 `backprojection_error`。新增
回归测试先复现错误分类，再改为 `held_target_error`，避免真实证据把缓存状态或配置
问题错误归因到相机反投影。全量自动测试因此更新为164/164通过；`compileall`、
`git diff --check` 和动作接口真值隔离审计通过。

这些结果只证明离线状态机和证据契约满足设计，不证明 Qwen 在线定位或 PyBullet 闭环
成功。下一步只审查三个固定案例、冻结校准和请求上限；真实付费 smoke 必须由用户再次
明确批准。

## 24. 首轮真实 Grounding 闭环 Smoke（2026-07-19）

用户明确批准后运行：

```text
conda run -n vla_env vla-run-grounding-smoke --run-name run_20260719_target_hold_v1
```

运行前已完成无 API 预检：editable 包指向当前 `main` 的 `src/`，三个固定案例、冻结
校准15个样本 ID、`max_stale_target_steps=4`、`api_max_retries=0` 和
`max_total_api_calls=30` 均通过契约检查，相关定向测试为67/67通过。

真实运行没有通过，进程退出码为1，只发生1次真实 API 调用：

```text
52-left：  visibility_out_of_scope，初始可见率270/378=0.7143，0次API，0次动作
53-right：stale_target_limit，1次API，1步fresh + 4步held，5次动作
54-front： start pose未收敛，误差0.053522m，批次抛异常退出
```

`53-right` 是本轮最重要的正向但不充分证据：真实 XY 距离从约10.05cm降到0.41cm；
首步 VLM 给出世界目标后，连续4个低可见步骤均使用缓存目标和当前末端位置重新计算动作，
没有再次请求 API。红块可见率依次从约0.844降为0.648、0.455、0.254、0.063，最后仅
0.003；第5次失效尝试按设计停止。由于缓存预测目标与当前末端的距离仍约3.58cm，控制器
没有预测 stop，因此该 episode 的 summary 仍为 `success=false`。这说明目标保持在这个
案例中确实完成了有效运动，但尚未满足当前端到端通过契约。

`54-front` 随后用独立无 API 初始化精确复现：请求起点为
`[0.1654557627, 0.2964970011, 0.2]`，实际末端为
`[0.1908597797, 0.3420210481, 0.1878824681]`，误差仍为0.053522m，证明失败发生在
确定性的 IK/reset 起点边界，不是 VLM 或网络波动。

证据目录：

```text
outputs/vlm_evaluations/grounding_world_smoke/run_20260719_target_hold_v1/
```

目录包含 episode 0、1 的逐步图片、trace 和 `episode_summary.jsonl`；由于 episode 2
初始化时抛异常，没有完整 `batch_summary.json`。下一步不直接重跑付费实验：先建立
批次级无 API 起点/首帧预检，并明确“动作不读真值”前提下事后成功评分与预测 stop 的
关系，再决定固定案例和 runner 的最小修订。

## 25. Grounding Smoke 整批无 API 动态预检（2026-07-19）

首轮真实 smoke 暴露出顺序执行漏洞：后置案例尚未验证动态资格时，前置案例已经可能
产生付费请求。本轮没有重跑真实 API，而是以两阶段门禁修复该问题。第一阶段对三个固定
案例分别创建独立 PyBullet 场景，只检查固定10cm起点能否在5mm容差内到达，以及首帧
红块可见率是否达到0.75；三个全部合格后，第二阶段才允许进入原有 VLM 闭环。

实现新增 `preflight_smoke_case()`、`SmokePreflightError` 和
`smoke_preflight.json`。预检记录 seed、方向、请求/实际起点、姿态误差、可见像素、
可见率与拒绝原因。任一初始预检失败时仍完成三个案例检查，但不调用
`run_control_loop()`，不构建 episode trace，也不调用 VLM。正式闭环运行中后续出现
遮挡时，已有“最近可靠目标 + 最多4步 held”逻辑保持不变。

TDD 先确认接口缺失和门禁缺失的 RED，再验证起点失败、低可见率、阈值包含性、异常连接
清理、整批拒绝和3/3合格放行。grounding smoke 定向测试为72/72，全量测试由164项增加
到169/169；`compileall` 通过。

随后在真实 PyBullet 中运行无 API 动态验收，并从进程环境显式移除 API 地址、密钥和
模型名。结果与首轮证据一致：

```text
52-left：  起点误差0.000817m，可见率270/378=0.714286，拒绝
53-right： 起点误差0.000771m，可见率319/378=0.843915，合格
54-front： 起点误差0.053522m，可见率215/378=0.568783，以起点误差优先拒绝
整批：     qualified_count=1/3，passed=false，API调用0次
```

证据保存在：

```text
outputs/vlm_evaluations/grounding_world_smoke/run_20260719_batch_preflight_validation_v1/smoke_preflight.json
```

该目录只包含预检 JSON，没有 `episode_summary.jsonl`、`smoke_summary.json` 或 episode
图片。这是门禁按设计成功拒绝一个不具备3/3资格的批次，不是第二轮在线 smoke 失败。
下一步仍需分别解决事后真实成功与预测 stop 的评分契约，以及选择三个能够通过初始预检
的固定案例；完成前不再申请付费运行。

## 26. Grounding Smoke 成功评分拆分（2026-07-19）

为避免把首轮 `53-right` 的真实任务到达与自主 stop 混成同一个布尔值，本轮按批准的
设计拆分 episode 和 batch 指标。循环控制逻辑、held 状态机与真值隔离边界均未改变；
所有新字段只在循环结束后根据已有评分证据计算：

```text
task_success = final_true_distance_xy <= 0.03
autonomous_stop_success = termination_reason == "success"
success = task_success and termination_reason in {"success", "stale_target_limit"}
```

系统错误和控制预算耗尽不在允许终止原因中，所以即使最后真实距离碰巧不超过3cm，
API、反投影、IK 错误或 `max_control_steps` 也不能成为主要成功。batch summary 新增
`task_success_count/rate` 与 `autonomous_stop_success_count/rate`，原有
`success_count/rate` 和 `passed` 使用新的主要成功语义。

TDD 先用5个 episode 场景和1个混合 batch 场景确认旧实现缺字段，再完成最小实现；
grounding smoke runner 为32/32通过，配置、包入口、targeting、runner、screening 的
定向契约测试为76/76通过。测试使用 mock，不调用真实 API。本轮没有选择或替换三个
固定案例，也没有运行第二轮真实 smoke；下一步仍是让三个固定案例先通过整批无 API
动态预检。

最终全量自动测试为173/173通过，`compileall`、`git diff --check` 和动作接口真值隔离
审计通过。

本次变更构成指标语义版本边界。历史输出不回写，旧 `success_rate` 不能与新批次直接
合并比较；历史 `53-right` 应解释为“真实任务到达，但未自主 stop”，而不是宣称当时的
整批实验已经通过。

## 27. 当日收尾与下次恢复点（2026-07-19）

成功评分功能已 fast-forward 合并到 `main`，合并后的全量测试为173/173通过；功能
worktree 与已合并分支已清理。用户原有未跟踪文件
`docs/superpowers/plans/2026-07-19-agent-file-placement-rules.md` 保持不变。

随后只讨论了固定案例修订方向，没有修改筛选代码、配置或案例，也没有调用 VLM。
当前推荐方案是继续保留 `left/right/front` 三个方向各一个，以预先声明的 seed 范围和
确定性规则选择互不重复的案例；候选资格只要求起点姿态误差 `<=5mm`、首帧可见率
`>=0.75`，不再要求整条接近轨迹全程 clear。选定案例仍须经过整批无 API 预检，只有
3/3合格才可请求第二轮真实 smoke。

该方案尚未获得用户对“继续保持三个方向覆盖”的确认，因此今天在设计澄清阶段暂停。
下次恢复时先确认这一点，再比较候选选择方案并形成设计文档；不要直接改配置或运行
真实 API。

## 28. Grounding Smoke 固定案例重选与3/3只预检（2026-07-26）

本轮按已批准设计将候选资格从“不可能满足的五帧全轨迹 clear”改为“起点姿态误差
`<=0.005m` 且首帧可见率 `>=0.75`”。候选范围继续冻结为 seeds 55–100 ×
`left/right/front`，选择规则为按方向顺序选择 seed 最小且互不重复的合格案例。后续
遮挡仍由最多4步 held 状态机处理，不进入筛选资格。

真实 PyBullet 无 API 筛选共评估138个候选，结果为：

```text
qualified：47
visibility_below_threshold：47
start_pose_error：44
selected：56-left、55-right、59-front
```

筛选证据保存在：

```text
outputs/vlm_evaluations/grounding_world_smoke_screening/run_20260726_initial_qualification_v1/
```

该目录包含138条 `candidate_trace.jsonl` 记录、每个候选唯一的 `step_00.jpg` 和
`screening_summary.json`。独立重算最小互异 seed 规则与摘要选择一致。

随后将 `sim_config.yaml` 的固定 seeds 按 `left/right/front` 顺序冻结为
`[56, 55, 59]`，并通过新增的 `--preflight-only` 运行三案例整批预检：

```text
56-left：  起点误差0.001143823m，可见率285/378=0.753968，合格
55-right： 起点误差0.004671998m，可见率351/378=0.928571，合格
59-front： 起点误差0.002441856m，可见率305/378=0.806878，合格
整批：     qualified_count=3/3，passed=true，API调用0次
```

预检证据保存在：

```text
outputs/vlm_evaluations/grounding_world_smoke/run_20260726_fixed_cases_preflight_v1/smoke_preflight.json
```

预检目录只包含 `smoke_preflight.json`，没有 episode 目录、
`episode_summary.jsonl` 或 `smoke_summary.json`。本轮没有调用真实 VLM，也没有运行
第二轮在线 smoke。独立重算最小互异 seed 规则与摘要完全一致；grounding smoke、
配置和包入口定向测试为74/74通过，全量自动测试为171/171通过，`compileall` 和
`git diff --check` 通过。下一步只能在用户明确批准付费/API运行后继续。

## 29. 第二轮真实 Grounding 闭环 Smoke（2026-07-26）

用户明确批准后，在3/3无 API 预检通过的固定案例上运行：

```text
conda run -n vla_env env PYTHONPATH=src vla-run-grounding-smoke \
  --run-name run_20260726_fixed_cases_online_v1
```

进程退出码为0，批次 `passed=true`。三个案例的主要成功与真实任务到达均为3/3，总计
调用 VLM API 4次；具体结果为：

```text
56-left：  主要成功，未自主stop，stale_target_limit，1次API，
           真实XY距离9.934cm -> 0.303cm
55-right： 主要成功，自主stop，success，2次API，
           真实XY距离10.433cm -> 0.789cm，发生1次遮挡恢复
59-front： 主要成功，未自主stop，stale_target_limit，1次API，
           真实XY距离9.958cm -> 0.371cm
整批：     success=3/3，task_success=3/3，
           autonomous_stop_success=1/3，API调用4次
```

三例各执行5次动作；整批包含4个 fresh VLM 步骤和12个 held 目标步骤。`56-left` 与
`59-front` 虽然没有预测 stop，但最终真实距离已低于3cm，并且终止原因为评分契约允许
的 `stale_target_limit`，所以主要 `success=true`。这与自主停止成功是两个独立指标。

完整证据保存在：

```text
outputs/vlm_evaluations/grounding_world_smoke/run_20260726_fixed_cases_online_v1/
```

其中 `smoke_summary.json` 是批次权威摘要，`episode_summary.jsonl` 和三个 episode
目录保存逐案例结果、trace 与图片。该结果证明固定相机、冻结校准、三个固定方向和最多
4步目标保持条件下，小规模模块化在线闭环已满足当前主要通过条件；它不证明自主停止、
severe 遮挡、移动目标或大规模泛化已经解决。按此前约定的决策规则，下一阶段进入专家
数据规模化，自主 stop 以独立改进项保留。

## 30. 自主 Stop 时序修复与严格批次门槛（2026-07-26）

复查第二轮真实 smoke 后，确认 `56-left` 和 `59-front` 未自主 stop 的根因不同。
`56-left` 最后一次 held 动作后已经进入缓存目标的每轴2cm停止范围，但旧循环在下一帧
先触发缓存过期，没有做动作后复核。`59-front` 的首帧 grounding 框约偏左5.4像素、
偏上8.7像素，冻结补偿后仍残留约2.9cm Y 定位误差；它是固定步长经过真实红块，不是
控制器正确识别到达。

本轮以 TDD 修改 runner：第5次连续不可见时只使用缓存目标和当前末端位置做
`stale_target_recheck`，允许输出 stop，但禁止执行第5个动作，也不调用 VLM。仍需移动
时继续以 `stale_target_limit` 安全中止。批次 `passed` 同时增加自主停止3/3要求，
避免只凭事后真实距离通过整批 smoke。

新增测试先在旧实现上得到预期失败，再完成最小实现；runner 定向测试36/36、全量测试
173/173通过，`compileall` 和 `git diff --check` 通过。本轮没有调用真实 API，
也没有回写历史输出。

结合整体学习路线复核后，自主 stop 3/3继续作为严格 smoke 指标，但不再作为进入专家
数据规模化的硬门槛。`59-front` 保留为单帧 grounding 残余偏差的已知案例，不通过
针对 seed 调参、读取真值或放宽阈值来制造通过。下一步转入专家数据规模化和小批量质量
验收；只有后续训练或评估明确依赖自主 stop 时，才重新处理 grounding 稳健性。
