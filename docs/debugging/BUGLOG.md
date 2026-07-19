# VLA 项目 Bug 日志

这份日志记录可复现故障、证据、根因假设、单变量实验和最终结论。普通项目进度写入 `docs/worklog/WORKLOG.md`。

## BUG-001：负 x 目标下离散控制振荡

- 状态：已修复并通过固定批量回归
- 发现日期：2026-07-11
- 影响模块：`src/vla_project/simulation/stage3_probe.py`、`src/vla_project/simulation/evaluate_probe.py`
- 基线批次：`outputs/probe_evaluations/run_20260711_185439/`

### 现象

```text
20 个固定种子 episode
success: 5/20 (25%)
termination_reason=max_control_steps: 15 个 episode
error: 0
final_distance mean / median / max: 0.4989 / 0.5948 / 0.8137 m
```

12 个 `block_x < 0` 的 episode 全部失败；4 个 `block_x >= 0.1` 的 episode 全部成功。成功目标 x 均值为 0.149，失败目标 x 均值为 -0.072。

### 代表案例

文件：`outputs/probe_evaluations/run_20260711_185439/episode_001/probe_trace.jsonl`

seed 43，`block_x=-0.18458`，`control_step=21`：

```text
direction: left
ee_pos_before: [0.38908, -0.10311, 0.19839]
target_pos:    [0.35908, -0.10311, 0.19999]
ee_pos_after:  [0.38233, -0.12508, 0.19565]

期望 dx: -0.03000 m    实际 dx: -0.00675 m
期望 dy:  0.00000 m    实际 dy: -0.02197 m
distance_before / after / delta: 0.80639 / 0.81729 / -0.01090 m
```

### 已确认与已排除

- `left` 正确转换成 `target_x = ee_x - 0.03`，不是方向映射错误。
- 实际 x 位移不足，并出现非预期 y 负方向偏移。
- Baseline 1 的直接 hover IK 能覆盖负 x，现有证据不足以判定绝对不可达。
- 失败后半段常见 `left/front` 往复切换和距离振荡。

### 已完成诊断实验

#### 实验 A：只增加最大决策次数

```text
3cm × 80：5/20，25%
3cm × 120：5/20，25%
```

负 x 仍为 0/12 成功，失败样本只是振荡更久。排除“允许更多次相同决策即可成功”。

#### 实验 B：只增加每次物理执行时间

```text
sim_steps_per_action 60：5/20，25%
sim_steps_per_action 120：5/20，25%
```

seed 43 step 21 的非预期 dy 几乎不变（-0.02197m 对 -0.02195m）。排除“电机只是缺少执行时间”。

#### 实验 C：减小离散步长

```text
3cm × 80：5/20，25%
1cm × 80：1/20，5%
1cm × 240：4/20，20%
```

1cm 将 seed 43 单步非预期 dy 从 -0.02197m 降到 -0.00625m，但延长到相同 2.4m 移动预算后仍无负 x 成功，只把振荡变小、变久。排除“单纯减小步长即可解决”。

### 当前根因范围

方向到 Cartesian `target_pos` 的映射已确认正确；控制次数、执行时间和步长均不能消除负 x 系统性失败。根因范围已收窄到：

```text
Cartesian target_pos
-> IK 目标关节角
-> 关节电机实际角
-> 末端实际位置
```

优先假设：固定末端姿态下的 7 自由度 IK 在负 x 区域产生不连续/不利的冗余解，或关节目标未被实际电机跟踪。

### 关节级诊断结果

固定 seed 43 的诊断输出：

```text
outputs/probe_evaluations/diagnostic_seed43_joints/probe_trace.jsonl
```

KUKA iiwa 第 4 关节（数组下标 3）的 URDF 限位为：

```text
lower = -2.094395 rad
upper =  2.094395 rad
```

失败振荡阶段的代表值：

```text
step 17, left
IK target joint 4: 2.1713 rad
actual joint 4:    2.0944 rad
error:             0.0769 rad

step 18, front
IK target joint 4: 2.1185 rad
actual joint 4:    2.0944 rad
error:             0.0241 rad
```

从 step 17 开始统计，另外 6 个关节的最大跟踪误差都不超过约
`0.00018 rad`，只有第 4 关节的平均绝对误差为 `0.05074 rad`、最大为
`0.07867 rad`。因此不是整体电机跟不上，而是无约束 IK 持续给第 4 关节
生成超出物理上限的目标。

当第 4 关节被物理限位截住时，`left` 的笛卡尔目标无法按原方向实现：

```text
期望：x -0.03m，y 不变
实际：x 只减少约 0.0068m，同时 y 减少约 0.022m
```

启发式策略随后用 `front` 修正 y，下一步又需要 `left`，最终形成
`left/front` 往复振荡。这也解释了为什么增加决策次数、增加物理执行时间、
缩小单步距离都没有消除失败。

### 根因

`control_arm.calculate_target_joints()` 调用 7 自由度 IK 时只传入末端位置、
固定姿态和残差阈值，没有传入 KUKA 的关节上下限、关节活动范围，也没有用
当前关节姿态约束冗余解。IK 因此沿着不利的冗余分支求解，并将第 4 关节推到
限位外；电机层只能停在 URDF 限位，导致末端实际动作偏离笛卡尔目标。

### 修复

`calculate_target_joints()` 现在从 PyBullet 读取每个受控关节的：

```text
lowerLimits
upperLimits
jointRanges
当前关节角 restPoses
```

并把它们传给 `calculateInverseKinematics()`。其中当前姿态 `restPoses` 让每一步
IK 优先沿连续、靠近现状的冗余解移动；关节限位信息为求解器提供真实机器人
约束。新增单元测试验证这些参数确实传入 IK。

注意：PyBullet 的 null-space IK 仍可能返回略微超过限位的数值，限位参数不是
严格优化约束；但关键变化是它不再选择原先导致横向串扰和死循环的冗余分支，
实际关节控制与末端轨迹已经通过批量结果验证。

### 修复验证

固定 seed 43 单案例：

```text
修复前：80 步失败，final_distance=0.8087m，distance_increase_steps=33
修复后：32 步成功，final_distance=0.0165m，distance_increase_steps=0
输出：outputs/probe_evaluations/diagnostic_seed43_limited_ik/
```

相同 seeds 42-61 的 20 次批量回归：

```text
修复前： 5/20，成功率 25%
修复后：20/20，成功率 100%
失败数：0
错误数：0
最终距离 mean / median / max：0.0180 / 0.0173 / 0.0285m
平均控制步数：38
全批次距离变差步数：5 / 760
输出：outputs/probe_evaluations/run_20260711_221233/
```

原先 12 个负 x 目标从 `0/12` 成功变为全部成功，说明系统性负 x 故障已消失。

### 扩大到 50 次的正式验收

20 次修复回归证明故障已经消失后，继续保持控制参数不变，把固定随机种子扩大
到 42-91，共运行 50 个 episode：

```text
max_control_steps = 80
sim_steps_per_action = 60
move_step_xy = 0.03m
success_distance = 0.03m
```

正式验收结果：

```text
success: 50/50
success_rate: 100%
failure / error: 0 / 0
final_distance mean / median / max: 0.0193 / 0.0197 / 0.0291m
control_steps mean / median: 37.58 / 37
输出：outputs/probe_evaluations/run_20260712_221135/
```

50 次结果仍然没有复现负 x 系统性失败，且最大最终距离低于 0.03m 成功阈值。
因此 20 次结果不是小样本偶然现象，BUG-001 可以作为已完成回归验证的关闭
故障；这批结果同时作为 Stage 3 heuristic 的正式稳定性验收证据。

### 诊断过程新增的可观测字段

在 trace 中增加：

```text
target_joint_angles
actual_joint_angles_before
actual_joint_angles_after
joint_error_after
```

这些字段保留在 `src/vla_project/simulation/stage3_probe.py`，以后出现类似轨迹偏移时，可以直接区分
“IK 目标错误”和“电机未跟踪”。

### 最终结论

BUG-001 的直接根因是 IK 未使用关节约束和当前姿态作为冗余解参考，导致第 4
关节撞限位，进而产生 x/y 串扰和闭环振荡。修复后固定 20 种子成功率从 25%
提升到 100%，扩大到 50 种子后仍为 100%，该故障关闭。

## BUG-002：VLM 连续方向误判与真实末端不可见

- 状态：主要根因已确认；直接方向路径保留为对照，主路线转向目标感知与本体状态融合
- 发现日期：2026-07-14
- 影响模块：`src/vla_project/vlm/collect_vlm_eval_samples.py`、`src/vla_project/vlm/evaluate_vlm_decisions.py`、
  `src/vla_project/simulation/stage3_probe.py`、`sim_config.yaml`
- 对照基线：`outputs/vlm_evaluations/offline_qwen3_vl_flash_high_topdown_v9/`

### 现象

Alibaba Qwen `qwen3-vl-flash` 能稳定返回合法离散标签，但早期版本的方向准确率很低。
代表结果：

```text
v1：10 张合法输出率 100%，准确率 10%
v2：10 张准确率 50%
v4：首张 100%，10 张准确率 10%
v7：双视角首张 0%，期望 front，实际 left
v8：带 MAIN/AUX 标记的双视角首张仍为 0%，期望 front，实际 left
```

v7/v8 在增加双视角和面板标签后仍产生完全相同的 `screen_left`，说明问题不只是
输出格式或模型不知道哪个面板是主视图。

### 尝试过的方法与结果

#### 方法 A：限制输出格式并修复非法回复处理

让模型只输出五个标签，并把“无法解析”从默认 `stop` 改为明确错误；为 API 增加
可恢复错误重试和认证错误快速失败。

结果：合法输出率达到 100%，但方向准确率仍低。说明输出协议已经稳定，视觉判断
仍然错误。

#### 方法 B：收紧 stop 判定

明确规定只有末端与红块中心已经对齐时才能输出 `stop`，不确定时必须选择移动方向。

结果：10 张准确率从 10% 提高到 50%，但仍有系统性方向错误。说明过早停止是一个
问题，但不是全部根因。

#### 方法 C：把图像方向与世界坐标方向拆开

模型改为输出 `screen_left/right/up/down`，代码再通过固定映射转换为 PyBullet
`left/right/front/back`，避免让模型直接推理世界坐标轴。

结果：首张一度达到 100%，但 10 张只有 10%，模型在斜视图中几乎总是选择同一
屏幕方向。说明确定性映射消除了接口歧义，但相机透视和目标可见性仍不稳定。

#### 方法 D：正俯视与倾斜俯视相机

依次尝试正俯视和轻微倾斜俯视，希望让屏幕方向直接对应世界 x/y。

结果：正俯视中红块会被机械臂遮挡；倾斜后部分帧仍然遮挡。单视角在“方向语义
清楚”和“目标可见”之间存在冲突。

#### 方法 E：双视角拼接

左侧使用倾斜俯视主视图，右侧使用反方向补充视图，两个面板保持相同屏幕方向，
仍然只调用一次 API。

结果：图片尺寸为 448×224，部分遮挡确实得到补充，但首张预测仍为
`screen_left`，准确率 0%。模型可能把一个面板的红块和另一个面板的机械臂部件
跨面板比较。

#### 方法 F：增加 MAIN/AUX 标签和黄色分隔线

为两个面板增加独立标题栏和明显分隔线，prompt 明确要求优先判断 MAIN，只有遮挡
时才参考 AUX。

结果：首张仍输出 `screen_left`，和未标记双视角完全相同。排除“模型只是没分清
左右面板边界”这一假设。

### 根因诊断证据

复现 seed 42，并使用 `ee_link_index=6` 的 PyBullet 世界坐标及相机矩阵，把真实
末端和红块投影到 224×224 图像：

```text
v7/v8 MAIN：真实末端约 (112, 313)，红块约 (119, 116)
v7/v8 AUX： 真实末端约 (112, 81)， 红块约 (119, 112)
```

MAIN 中真实末端的 y 坐标已经超出画面。画面内被 prompt 误称为“末端”的黑色
圆环实际是中间关节。AUX 中末端虽然在画面内，但红块位置被机械臂遮住。也就是说：

```text
没有任何一个面板同时清楚提供真实末端和红块。
```

双面板模型只能猜测或跨面板配对，单靠修改 prompt 不可能恢复缺失的视觉信息。
这是 v7/v8 连续输出相同错误方向的直接原因。

### 修正方案

在不调用 API 的情况下，对同一 seed 42 比较：

```text
4m 正俯视
4m 轻微倾斜俯视
3m 正俯视
```

3m 正俯视既能让初始和中间阶段的真实末端、红块同时入画，又比 4m 版本具有更大
的目标像素尺寸。因此 v9 使用单视角配置：

```text
eye_offset_base = [0.0, 0.0, 3.0]
up_vector = [0, 1, 0]
fov = 45
use_dual_view = false
```

同时修改 prompt：将真实末端描述为机械臂最下端的深灰色圆形法兰，并明确不要选择
机械臂中间的黑色关节。

### 验证结果

v9 首张样本：

```text
expected_direction: front
raw_response: screen_up
mapped_direction: front
exact-match: true
latency: 约 1.12s
```

扩大到同一 episode 均匀抽取的 10 张样本：

```text
合法输出率：100% (10/10)
exact-match accuracy：70% (7/10)
平均延迟：约 1.20s
right：9 张，正确 6 张
front：1 张，正确 1 张
```

错误样本：

```text
seed_42_step_017：expected right，predicted left
seed_42_step_026：expected right，predicted front
seed_42_step_039：expected right，predicted front
```

step 39 已接近成功位置，红块被末端遮挡，属于视觉上难以判断的样本。step 17 和
step 26 仍可能存在真实末端部件识别错误。

### 当前结论与下一实验

相机高度和视角是早期低准确率的主要根因。修正输入可见性后，10 张准确率提升到
70%，证明继续堆叠 prompt 不是正确方向。但该故障尚未完全关闭：当前样本类别不
均衡，且末端部件识别、接近目标后的遮挡仍会导致错误。

下一实验是在真实末端 `link 6` 增加绿色视觉标记，并保持 v9 相机、样本种子、
模型和评估指标不变，与无标记 70% 基线进行消融对照。绿色标记只帮助指出哪个部件
是末端，不直接提供正确移动方向；如果准确率明显上升，可以确认剩余瓶颈主要来自
末端识别，而不是方向映射。

### 后续实验：绿色末端标记 v10

在真实末端增加纯绿色无碰撞视觉球，并保持 3m 正俯视、模型和评估口径不变：

```text
合法输出率：100% (10/10)
exact-match accuracy：50% (5/10)
right：9 张，正确 4 张，其中 5 张被误判为 front
front：1 张，正确 1 张
输出：outputs/vlm_evaluations/offline_qwen3_vl_flash_green_ee_v10/
```

绿色标记没有提高准确率，反而比 v9 的 70% 下降到 50%。这说明“增加醒目颜色”
并不等于模型能正确理解该标记与任务的关系，也可能改变原图视觉分布并吸引模型
过度关注。配置已关闭绿色标记，后续不再把颜色作为末端识别的主要依据。

### 后续实验：四方向均衡集 v11

v9/v10 的 10 张轨迹样本中有 9 张标准答案是 `right`，70% 不能代表四方向能力。
因此新增 `balanced_poses` 采样：把末端主动放在红块四周，生成
`left/right/front/back` 各 5 张、共 20 张图片；停止判断暂不混入方向基准。

先对每个方向各评估 1 张，共 4 张：

```text
合法输出率：100% (4/4)
exact-match accuracy：50% (2/4)
left  -> right（错误）
right -> right（正确）
front -> front（正确）
back  -> right（错误）
输出：outputs/vlm_evaluations/offline_qwen3_vl_flash_balanced_v11/
```

这个小样本结果证明 v9 的 70% 受到方向类别不均衡影响。模型仍可能把中间关节或
底座当成末端，随后稳定地比较错对象；继续强调某种颜色不能解决该问题。

### Prompt 复盘与 v12 设计

结合 v7-v11，保留有证据支持的规则：

1. 模型只输出 `screen_left/right/up/down`，世界坐标映射由代码完成。
2. 从固定底座出发，沿机械臂的连接关系逐节追踪；连接链最后一节的末梢才是末端。
3. “最后一节”不是画面中的直线距离、上下位置或颜色，禁止选择底座和中间关节。
4. 同时存在水平和垂直偏差时，比较像素距离绝对值，只选择偏差更大的轴。
5. 四方向离线基准不允许输出 `stop`；在线闭环才单独启用停止规则。

删除没有稳定证据支持的描述：`黑色末端`、`深灰色圆形法兰`、`画面最下端`、
`绿色末端` 和 `更明显的偏差`。这些表述会随视角、材质或模型主观理解改变。

在线和离线现在共用 `stage3_probe.build_vision_direction_prompt()`，避免两套 prompt
继续漂移。新运行目录为：

```text
outputs/vlm_evaluations/offline_qwen3_vl_flash_balanced_chain_prompt_v12/
```

v12 已先评估四方向各 1 张：

```text
合法输出率：100% (4/4)
exact-match accuracy：50% (2/4)
left  -> right（错误）
right -> right（正确）
front -> front（正确）
back  -> right（错误）
平均 API 延迟：约 1.04s
输出：outputs/vlm_evaluations/offline_qwen3_vl_flash_balanced_chain_prompt_v12/
```

结果与 v11 的四个输出完全相同，说明删除颜色描述、改用连接链描述后，模型仍选择
了相同的错误参照物；当前瓶颈不能继续归因于方向标签格式或某一句 prompt。

人工查看 `seed_42_left.jpg` 和 `seed_42_back.jpg` 后发现，均衡集使用的 0.10m 偏移
在 224×224 图片中只形成很小的像素间隔，红块与真实末端部件明显重叠，中心位置
不够清楚。下一步不应继续堆叠 prompt，而应只增大 `balanced_pose_offset_xy`，重新
生成四方向图片并保持 v12 prompt 不变，以验证视觉分离度是否为当前主要限制。

### 末端与红块画框诊断

为了区分“模型找不到末端”和“找到末端但无法比较方向”，新增
`src/vla_project/vlm/diagnose_vlm_grounding.py`。模型只返回 `end_effector` 与 `red_block` 的 0–1000
归一化框，代码再把绿色末端框和黄色红块框画到原图上。四张均衡图片均返回合法框：

```text
成功生成目标框：4/4
输出：outputs/vlm_evaluations/grounding_qwen3_vl_flash_ee_red_v1/
```

人工核对标框图后，模型选择的末端框基本位于机械臂连接链末梢，并没有持续框住
底座或中间关节。因此 v11/v12 的主要错误不能再简单归因于“完全找不到末端”。

框中心关系提供了更直接的证据：

```text
left： 红块框中心在末端框中心左侧，与标准方向一致
right：末端框与红块框完全相同，视觉上无法分离两个中心
front：红块框中心在末端框中心上方，与标准方向一致
back： 红块框中心相对末端呈右下偏移，且水平框偏差大于垂直框偏差
```

这恰好解释了 v12：`front` 正确，而 `back` 按“选择像素偏差更大的轴”输出
`screen_right`。当前主要瓶颈是 0.10m 均衡位姿使红块和末端在低分辨率图中重叠，
导致可见框中心与世界坐标标准方向不一致或不可区分，而不是缺少更长的方向 prompt。

### 架构层处置决定

继续放大离线样本间距可以验证远距离可见性，但机器人最终必须接近并遮挡红块，
因此不能把它当成最终修复。继续修改 prompt 也无法从被遮挡的单帧中恢复不存在的
空间证据。

主路线改为把问题拆成三个可独立评分的模块：

```text
VLM：根据图像和语言定位红块，不接收仿真真值
本体感知：通过关节状态和 FK/getLinkState 获得机器人自身末端位置
几何控制：相机反投影得到目标世界坐标，再计算离散动作
```

这里允许控制器读取末端状态，因为真实机器人同样能由编码器获得自身姿态；仍然
禁止运行时读取红块 PyBullet 位姿。`block_pos` 只可在离线诊断中衡量反投影误差，
不能参与目标估计或动作计算。

这个决定不删除 BUG-002 的直接方向实验。v9 的 70% 证明方案具有潜力，但其 10 张
样本中有 9 张是 `right`；v11/v12 的四方向测试又只有每方向 1 张。因此应将它描述为
“受限但有潜力的直接方向基线”，而不是失败路线。v9-v12、绿色标记消融和画框诊断
共同构成能力边界与根因证据，并作为未来三路对照中的 `VLM direct direction` 基线。下一诊断门槛
改为“红块像素框 -> 工作平面世界坐标”的定位误差；达标后才进入融合闭环 smoke test。

### v13 距离分层实验：排除近距离是唯一原因

保持模型、相机和 v12 prompt 不变，构造 `0.20m / 0.10m / 0.05m` 三档距离，
每档包含四个方向，固定 seed 42，共 12 张：

```text
合法输出率：100% (12/12)
总准确率：50% (6/12)
0.20m / 0.10m / 0.05m：每档均为 2/4
right、front：每档都正确
left、back：每档都被预测为 right
```

0.20m 的错误图中目标和机械臂末梢已经有明显视觉间隔，但预测模式与 0.05m 完全
相同。因此“只把均衡样本距离增大”已被排除为直接方向方案的充分修复。当前更具体
的故障表现是固定方向偏置：模型可能在直接回答方向时选择了错误参照部件，或虽然
识别目标位置却没有执行 prompt 中要求的中心比较。

这仍不等于宣布直接方向路线失败，因为当前只覆盖一个 seed；但它证明下一实验必须
拆分“目标/末端定位”和“根据定位结果选方向”，不能继续把二者压在一个标签回复里
反复调整距离。

### v13 20cm 画框对照：确认两类错误同时存在

对 0.20m 的四方向图片单独请求 `end_effector` 和 `red_block` 框，4/4 返回合法框。
将框中心关系与直接方向回复逐张对照：

```text
left： 两个预测框的相对中心关系支持 left，但直接方向输出 right
right：框中心关系支持 right，直接方向也输出 right
front：框中心的纵向偏差支持 screen_up，直接方向输出 front
back： 红块框未稳定覆盖真实红块，框中心关系不支持 back，直接方向输出 right
```

因此 BUG-002 包含至少两种机制：

1. `left` 属于方向推理错误：模型的两个近似框已经表示出支持 left 的相对中心关系，
   却在直接标签任务中给出相反方向。
2. `back` 属于视觉定位错误：特定机械臂姿态遮挡或干扰红块定位，后续方向自然不可靠。

这支持保留直接方向基线，同时把 grounding 作为可观测中间表示：由代码比较框中心
可以消除第一类错误，但第二类仍需要更好的目标可见性、时序缓存或相机方案。

进一步用纯红像素阈值测量 224×224 原图：正常姿态中的红块主体约为 `10×9` 像素、
约 87 个纯红像素；back 姿态只剩 50 个，约 43% 的纯红区域被机械臂遮挡。VLM 在
back 返回的红块框与纯红主体没有重合。因此 back 已确认属于小目标局部遮挡导致的
grounding 失败。预测框整体只能作为近似中间表示，不能声称达到像素级定位精度。

### v14 448px 消融：感知改善后剩余方向推理不一致

保持其他变量不变，只把图片从 224×224 提升到 448×448。红块主体扩大到约
`20×18` 像素；back 的纯红像素由 50 增加到 220。

直接方向准确率从相同 20cm 四方向的 2/4 提升为 3/4：`left/right/front` 正确，
只有 back 仍输出 `right`。这确认分辨率是前一轮 left 与小目标识别问题的重要变量。

随后对同四张 448px 图片做 grounding，四组末端框与红块框的相对中心关系全部支持
标准方向；back 中红块框中心已经明确位于末端框中心下方。按框中心主轴由代码计算
可以得到 4/4，但 VLM 直接标签仍为 3/4。

因此当前 back 错误不能再归因于红块不可见或框错目标，而是同一模型在两个任务中的
输出不一致：grounding 任务表达了正确的上下关系，直接方向任务却输出 `screen_right`。
后续应把“预测框是否正确”和“方向是否与预测框一致”作为两个独立指标。

### Ground-then-decide：同一回复内确认推理不一致

为排除“分开的 grounding prompt 和 direction prompt 导致注意力不同”，要求模型在
同一个 JSON 中先输出两个框，再输出方向。四张 448px 结果为：模型方向 3/4、框中心
推导方向 4/4、内部一致性 3/4。

唯一不一致的 back 中：末端框中心约 `(571,340)`，红块框中心约 `(567,431)`；
`dx≈-4`、`dy≈+91`，按绝对值更大的轴必须输出 `screen_down`，模型却输出
`screen_left`。因此该错误已经从“可能的感知问题”收窄为可复现的空间关系推理错误。

处置：VLM 输出 grounding，代码用框中心或后续世界坐标执行确定性比较。继续给直接
方向 prompt 增加同义规则的预期收益很低；直接方向 3/4 结果保留为受限基线。

BUG-002 当前处置边界已经明确：不再继续迭代同义方向 prompt；后续实现相机反投影，
并分别记录 grounding 误差、坐标转换误差和控制误差。直接方向 75% 作为消融基线保留。

## BUG-003：合法 grounding 框不等于厘米级定位达标

### 现象

固定正俯视相机、448px、20cm 相对距离和 Qwen grounding prompt，使用 seeds 42–46
生成五个红块位置、四种机械臂姿态，共 20 张。20/20 都返回合法框，但反投影整体
XY mean/median/max 为 `3.94/3.77/10.06cm`，没有通过 3cm 定位门槛。

### 根因拆分

PyBullet segmentation 真值显示，15 张 left/right/front 的红块可见率均为 `1.0`；
五张 back 为 `0.640/0.100/0.214/0.249/0.360`。清晰组平均 `3.34cm`，严重遮挡组
平均 `6.92cm`、最大 `10.06cm`，证明机械臂遮挡是大误差放大器。

但遮挡不是唯一原因。15 张清晰样本的 X 误差全部为负、Y 误差全部为正，平均有符号
偏差约 `(-2.49cm, +1.95cm)`；这表明 VLM 近似框中心或相机/工作平面模型中还存在
稳定系统偏差。方向级 `4/4` 只要求相对中心关系不跨越决策边界，不能证明绝对坐标精度。

### 排除项与处置

- 排除反投影轴翻转：几何 round-trip 测试通过，误差方向不是随机交换 X/Y。
- 排除“只要返回合法框就能控制”：20/20 合法输出仍有 3.94cm 整体平均误差。
- 不通过缩小红块位置范围隐藏遮挡；遮挡样本保留并按可见率单独汇总。
- 不直接使用本轮均值补偿。本轮清晰数据减均值后的 1.17cm 平均残差属于同集拟合，
  只能说明校准有潜力，不能作为泛化证据。

据此后续建立独立校准集/验证集：校准集估计固定偏差，验证集验收 `<3cm`；严重
遮挡继续单独测试双视角、历史帧或主动避让。两条路径完成前不接入在线控制。

### 独立验证结果

固定 seeds 42–46 为校准集，只用 15 条 clear 记录得到冻结补偿
`(+2.492cm, -1.947cm)`；全新的 seeds 47–51 作为验证集，共 20 张，校准 ID 与验证
ID 交集为零，验证阶段没有重新拟合。

20/20 grounding 和反投影有效。验证集 clear/partial/severe 为 `15/4/1`：

```text
clear raw mean/median/max:       2.99/3.13/3.92cm
clear corrected mean/median/max: 0.77/0.79/1.46cm，15/15 <=3cm
partial corrected max:           2.64cm，4/4 <=3cm
severe corrected error:          3.27cm，0/1 <=3cm
```

状态：固定系统偏差在独立 clear 样本上通过离线验证，排除了“只在拟合集上变好”的
数据泄漏解释；但在线控制尚未验证，BUG-003 也不能对 severe 遮挡关闭。下一步只为
clear 场景设计可中止闭环 smoke test，并为 severe 场景保留拒绝、双视角、历史帧或
主动避让路径。

### 当前状态与回归保护

BUG-003 目前是“部分解决”，不能整体关闭：

```text
清晰场景固定偏差：已解决并通过独立离线验证。
严重遮挡定位：未解决，唯一 severe 验证样本仍为 3.27cm。
补偿坐标在线控制：未验证，尚未接入正式控制路径。
```

工程侧已增加显式 seeds 47–51、校准/验证 ID 隔离、冻结补偿、clear 严格 3cm 判定
以及三个独立证据文件。完整测试为 `98/98` 通过，代码已合并回 `main`。这些测试只
保护数据边界和离线计算，不等价于机械臂闭环成功。

下一轮 smoke test 已确认的边界为：只运行 3 个 clear episode；每个最多 10 个控制
步；必须 3/3 达到真实 XY 误差 `<=3cm` 才通过。API、框解析、反投影、工作区边界或
运动趋势检查失败时立即安全中止，但安全中止的 episode 仍记为任务失败。每一步是否
重新调用 Qwen 尚待设计确认；在书面设计通过前不修改在线控制代码。

证据目录：

```text
outputs/vlm_samples/448_multiseed_d020/
outputs/vlm_evaluations/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/
outputs/vlm_evaluations/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/backprojection/
outputs/vlm_samples/448_calibration_validation_d020/
outputs/vlm_evaluations/grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1/
outputs/vlm_evaluations/grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1/calibration_validation/
```

## BUG-004：全程 clear 的闭环案例不存在

- 状态：根因已确认，短时遮挡恢复状态机已有单案例在线运动证据，但批次未通过
- 发现日期：2026-07-17
- 影响模块：`run_grounding_smoke.py`、`screen_grounding_smoke_cases.py`
- 证据批次：`vlm_smoke_screening_runs/run_20260717_231406/`

### 现象

为了避免在首次在线 smoke 中混入遮挡变量，动态筛选要求每条轨迹从约 10cm 起点到约
2cm 终点的五个观察点全部满足可见率 `>=0.75`。seeds 55–100 与
left/right/front 共形成 138 个候选，结果没有任何合格案例：

```text
num_candidates=138
num_qualified=0
visibility_below_threshold=131
start_pose_error=7
```

方向分组中，left 的 46 个候选全部因可见率失败；right 为 44 个可见率失败和 2 个
起点姿态误差；front 为 41 个可见率失败和 5 个起点姿态误差。代表性最长 clear 轨迹
也会在继续靠近时跌破阈值，例如 right seed 55 的可见率依次为
`0.929 -> 0.786 -> 0.595`。

### 根因与排除项

- 轨迹位置误差大多约为 1mm，排除“机械臂动作不准确导致筛选失败”为主因。
- 三个方向、46 个 seed 均出现接近后可见率下降，排除少数坏 seed 为主因。
- 不降低 0.75 阈值，也不继续扩大 seed 范围；这两种做法只会掩盖固定相机下的系统性
  接近遮挡。

### 处置决策

采用“最近可靠目标坐标 + 最大失效步数”，不采用“重复上一次方向”：

- 只缓存已经通过反投影、冻结补偿、工作区和跳变检查的 VLM 目标世界坐标；
- 遮挡步骤不调用 VLM，使用缓存目标与当前末端位置重新计算方向；
- 最多连续使用 4 步，第 5 次仍不可见则以 `stale_target_limit` 停止；
- 新的可靠 VLM 定位会更新缓存并把目标年龄清零；
- PyBullet 红块真值继续只允许进入事后评分闭包。

清晰画面下的 API、无效框、反投影、目标跳变、无进展和 IK 错误不回退到缓存，避免
遮挡恢复掩盖其他问题。BUG-004 只有在自动测试通过并完成三个真实在线 smoke cases
后才能更新状态；成功结论也只代表短时静态目标遮挡恢复，不外推到目标移动或永久遮挡。

### 首轮真实在线证据（2026-07-19）

真实批次 `run_20260719_target_hold_v1` 中，`53-right` 首帧可见率为0.844，发生1次
真实 grounding；随后可见率依次降到0.648、0.455、0.254、0.063，连续4步使用 held
目标且没有再次调用 API。真实 XY 距离从约10.05cm降到0.41cm，证明状态机在该案例中
确实把机械臂移动到目标附近，而不只是通过 mock 测试。

该案例仍不能关闭 BUG-004：缓存预测目标与末端的距离尚未满足2cm stop 条件，第5次
不可见按设计触发 `stale_target_limit`，summary 为 `success=false`。此外另外两个固定
案例没有进入相同闭环条件，因此尚无3/3端到端通过证据。

## BUG-005：held 计算异常被误记为反投影错误

- 状态：已修复并有回归测试
- 发现日期：2026-07-19
- 影响模块：`src/vla_project/vlm/grounding_smoke/runner.py`

### 现象与根因

独立代码审查发现，低可见率下调用 `compute_held_action()` 时，除
`SmokeSafetyAbort` 以外的未知异常统一记录为 `backprojection_error`。held 路径只接收
已缓存的世界目标和当前末端位置，不执行像素反投影，因此该分类会把缓存状态、配置或
实现错误错误归因到相机几何。

### 修复与验证

新增 `test_held_compute_exception_becomes_held_target_error`，先确认旧实现返回
`backprojection_error`，再把该分支改为独立的 `held_target_error`。fresh 路径的真实
反投影异常仍保持 `backprojection_error`，两类证据不再混淆。完整自动测试为
164/164通过，真实 API 调用为0。

## BUG-006：真实 Smoke 缺少整批动态预检和完整失败摘要

- 状态：部分修复；整批初始动态预检已完成，正式闭环异常的完整摘要仍待处理
- 发现日期：2026-07-19
- 影响模块：`src/vla_project/vlm/grounding_smoke/runner.py`
- 证据批次：`outputs/vlm_evaluations/grounding_world_smoke/run_20260719_target_hold_v1/`

### 现象

首轮真实 smoke 在发送请求前只验证了静态配置、冻结校准、CLI 和自动测试，没有先把
三个固定案例的起始姿态与首帧可见率全部动态验证一遍。实际顺序执行后得到：

```text
52-left：初始可见率270/378=0.7143，低于0.75，API调用0次
53-right：进入闭环，API调用1次，最终stale_target_limit
54-front：起始姿态误差0.053522m，高于0.005m，抛RuntimeError
```

第三个案例的异常使整个命令以退出码1结束。输出目录只包含 episode 0、1 的
`episode_summary.jsonl`，没有 episode 2 的结构化失败记录，也没有完整
`batch_summary.json`。因此不能把已落盘的0/2直接解释为三案例成功率。

### 已确认原因与边界

`54-front` 已用不调用 API 的独立初始化精确复现，姿态误差仍为0.053522m，说明它是
当前 seed、front 起点、IK/reset 与5mm容差组合下的确定性失败，不是网络或 VLM 波动。
`52-left` 的低可见率也发生在调用 API 前。问题不在请求重试或30次硬上限；本轮实际
总调用数只有1。

当前 runner 按案例串行执行“初始化 -> 可能调用 API -> 下一个案例初始化”，所以后置
案例的动态资格尚未检查时，前置案例已经可能产生付费请求。单个案例起始姿态失败又以
未捕获的批次级异常退出，导致完整摘要缺失。

### 后续修复约束

- 在任何真实 API 请求前，无 API 地验证整批案例的起始姿态和首帧可见率；（已完成）
- 动态预检失败时不得进入付费闭环，并输出可审计的案例级拒绝原因；（已完成）
- 正式闭环开始后的单案例异常仍应生成结构化 episode 与 batch 摘要；（未完成）
- 另行明确事后真实距离已 `<=3cm`、但控制器未预测 stop 时的成功评分契约；
- 修订固定案例和成功评分并通过无 API 验证前，不直接重跑真实 smoke。

### 整批初始动态预检修复与验证

以 TDD 增加 `preflight_smoke_case()` 和 `SmokePreflightError`。单案例预检独立创建并释放
PyBullet 场景，记录请求/实际起点、姿态误差、首帧可见像素、可见率和稳定拒绝原因；
批次先执行全部三个预检并写 `smoke_preflight.json`，只有3/3合格才进入原有控制循环。
测试证明任意初始预检失败时仍收集三个案例，`run_control_loop()` 与真实 API 边界均为
0次；阈值相等时合格，异常时 physics 连接仍释放。全量自动测试为169/169通过。

真实 PyBullet 验收显式移除了 `VLA_API_BASE_URL`、`VLA_API_KEY` 和
`VLA_MODEL_NAME`，结果为：

```text
num_cases=3
qualified_count=1
52-left：visibility_below_threshold，270/378=0.714286，起点误差0.000817m
53-right：qualified，319/378=0.843915，起点误差0.000771m
54-front：start_pose_error，215/378=0.568783，起点误差0.053522m
```

证据位于
`outputs/vlm_evaluations/grounding_world_smoke/run_20260719_batch_preflight_validation_v1/smoke_preflight.json`。
目录中没有 `episode_summary.jsonl`、`smoke_summary.json` 或 episode trace，证明当前固定
案例在任何在线控制和付费请求前被整体拒绝。本修复不改变运行中遮挡的4步 held 行为，
也不解决正式闭环已经开始后的通用异常汇总。
