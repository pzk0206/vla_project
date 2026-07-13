# VLA 项目 Bug 日志

这份日志记录可复现故障、证据、根因假设、单变量实验和最终结论。普通项目进度写入 `docs/worklog/WORKLOG.md`。

## BUG-001：负 x 目标下离散控制振荡

- 状态：已修复并通过固定批量回归
- 发现日期：2026-07-11
- 影响模块：`stage3_probe.py`、`evaluate_probe.py`
- 基线批次：`probe_eval_runs/run_20260711_185439/`

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

文件：`probe_eval_runs/run_20260711_185439/episode_001/probe_trace.jsonl`

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
probe_eval_runs/diagnostic_seed43_joints/probe_trace.jsonl
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
输出：probe_eval_runs/diagnostic_seed43_limited_ik/
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
输出：probe_eval_runs/run_20260711_221233/
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
输出：probe_eval_runs/run_20260712_221135/
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

这些字段保留在 `stage3_probe.py`，以后出现类似轨迹偏移时，可以直接区分
“IK 目标错误”和“电机未跟踪”。

### 最终结论

BUG-001 的直接根因是 IK 未使用关节约束和当前姿态作为冗余解参考，导致第 4
关节撞限位，进而产生 x/y 串扰和闭环振荡。修复后固定 20 种子成功率从 25%
提升到 100%，扩大到 50 种子后仍为 100%，该故障关闭。
