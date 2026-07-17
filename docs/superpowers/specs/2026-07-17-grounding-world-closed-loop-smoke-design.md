# Grounding 世界坐标闭环 Smoke Test 设计

## 目标

在不改变现有 Stage 3 heuristic/API 基线行为的前提下，用 3 个清晰场景验证以下完整链路能否真正控制机械臂：

```text
当前 RGB
-> Qwen 红块 grounding
-> 框中心反投影到工作平面
-> 应用冻结 XY 补偿
-> 与末端本体位置比较
-> 选择主误差轴并移动 2cm
-> 重新拍照和定位
```

本轮是小规模在线冒烟测试，不是正式 20 episode 评估，不处理 severe 遮挡，也不把补偿接入现有 `stage3_probe.py` 的正式控制模式。

## 已确认的实验范围

- 先在 seeds 55–100 中离线筛选，再固定运行 3 个 episode；
- 初始姿态分别覆盖 left、right、front，三者必须使用不同 seed，避免已知 back 遮挡；
- 测试准备阶段把末端放到距红块约 10cm 的指定方向；
- 每一步移动 2cm；
- 每个 episode 最多 10 个控制步；
- 每一步都重新拍照并调用 Qwen，最多 30 次真实 API 请求；
- 使用固定 448×448 正俯视相机，与定位校准实验保持一致；
- 必须 3/3 episode 都在 10 步内达到真实 XY 误差 `<=0.03m` 才通过。

之所以不从 KUKA 默认姿态开始，是因为实测随机场景的初始 XY 曼哈顿距离约为 53–58cm；2cm × 10 步无法覆盖该距离，会让 smoke test 因预算设计错误而必然失败。10cm 起点使测试聚焦于 VLM 定位和闭环纠偏。

## 首轮固定案例暴露的问题

首轮设计直接固定 seeds 52、53、54。真实运行证明这三个案例不满足 clear-only smoke test 的前置条件：

- seed 52 left 起点红块可见率为 `270/378 = 0.7143`，低于 0.75；
- seed 53 right 起点可见率为 0.8439，Qwen 框经冻结补偿后与真值相差约 1.5cm，第一步正确选择 right，使真实 XY 距离从 10.05cm 降至 8.12cm；但动作后可见率降至 0.6481；
- seed 54 front 请求起点为 `[0.1655, 0.2965, 0.20]`，实际 IK 误差为 5.35cm，无法满足 5mm 起点要求。

因此首轮结果证明单步 grounding 世界坐标链路有正向信号，同时证明固定案例的可见性和可达性假设错误。不得降低 0.75 阈值来掩盖该问题；正式重跑前增加不调用 VLM 的动态轨迹筛选。

## 架构

### 独立运行入口

新增 `run_grounding_smoke.py`，负责场景搭建、episode 循环、文件输出和最终汇总。它可以复用现有 PyBullet 世界初始化、Qwen 请求、grounding 解析、相机几何、IK、关节执行和 `direction_to_target()`，但不修改现有 `stage3_probe.py` 的控制分支。

### 纯定位与动作模块

新增 `grounding_targeting.py`，只实现不依赖 PyBullet 红块真值的纯计算：

- 读取并验证冻结校准参数；
- 红块归一化框中心转像素；
- 像素反投影到 `z=0` 工作平面；
- 应用固定 `correction_x/correction_y`；
- 校验预测目标是否位于工作区；
- 比较末端 XY 与预测目标 XY；
- 在 `right/left/front/back/stop` 中选择动作；
- 检查目标跳变和连续无进展。

运行时校准文件固定为：

```text
vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1/calibration_validation/calibration.json
```

脚本必须验证该文件含 15 个 clear 校准 ID，且补偿值为已有实验冻结的：

```text
correction_x = +0.02492227406480192 m
correction_y = -0.019467343494422532 m
```

验证值只用于防止加载错误文件，不允许 smoke 结果反向更新它。

## 真值边界

PyBullet 红块真实位置 `block_pos` 是仿真 ground truth。它只允许进入两个隔离环节：

1. 正式控制开始前，用于加载红块和构造距红块 10cm 的 left/right/front 起始姿态；
2. 每步动作完成后，由评分层计算真实 XY 距离、可见率标签和最终实验结果。

控制决策接口不得接收 `block_pos`。建议接口形态为：

```text
compute_grounding_action(
    red_block_box,
    image_size,
    view_matrix,
    projection_matrix,
    calibration,
    ee_pos,
    safety_state,
) -> GroundingAction
```

该接口只接收图片派生结果、相机参数、冻结补偿、末端本体位置和历史安全状态。测试准备层与评分层可以记录真值，但不得把它传回 `compute_grounding_action()`。

## 每步数据流

每个 episode 开始时，测试框架使用真值搭建固定起始场景；正式计步从场景搭建完成后开始。

每个控制步依次执行：

1. 用固定 448px 正俯视相机采集 RGB 和 segmentation；
2. 用 segmentation 计算仿真场景资格，可见率必须 `>=0.75`；
3. 调用既有 Qwen grounding prompt 和解析器；
4. 只取 `red_block` 框，机械臂末端位置来自 PyBullet 本体状态；
5. 将红块框中心反投影到 `z=0`，再应用冻结 XY 补偿；
6. 校验补偿目标位于 X `[-0.30, 0.30]`、Y `[0.30, 0.70]`；
7. 计算 `dx=target_x-ee_x`、`dy=target_y-ee_y`；
8. 若 `abs(dx)<=0.02` 且 `abs(dy)<=0.02`，输出 `stop`；
9. 否则选择绝对误差较大的轴：X 正/负对应 right/left，Y 正/负对应 front/back；
10. 复用 `direction_to_target()` 生成 2cm 单步目标，Z 固定为 `0.20m`；
11. 通过 IK 和现有关节控制执行动作；
12. 保存动作后状态和评分真值，再进入下一次重新定位。

`0.20m` 悬停高度来自固定任务几何：红块稳定后的中心 Z 约 0.05m，加既有 0.15m hover height；运行时不读取红块实时 Z 来设置控制高度。

## 无 API 动态轨迹筛选

新增独立入口 `screen_grounding_smoke_cases.py`。筛选阶段允许使用 PyBullet 真值决定固定正确方向，因为它只选择实验场景，不生成正式模型动作，也不评价 VLM。筛选不得调用 Qwen，且不得把真值接口接入 `run_grounding_smoke.py` 的动作策略。

候选范围固定为 seeds 55–100，方向固定为 left、right、front。每个 `seed + direction` 都在独立 DIRECT 场景中执行：

1. 用真值构造距红块 10cm、Z 为 0.20m 的起点；
2. 验证起点 IK 位置误差不超过 0.005m；
3. 使用与正式控制相同的 `direction_to_target()`、2cm 单步、IK、关节控制和物理步数；
4. 在起点及每次动作后的 8cm、6cm、4cm、约2cm位置采集同一 448px 正俯视 RGB 与 segmentation；
5. 每个观察点的红块可见率必须 `>=0.75`；
6. 每次动作后的末端与请求目标 3D 距离必须 `<=0.005m`；
7. 四次动作后的真实 XY 距离必须 `<=0.03m`。

任一条件失败即淘汰该候选，并保存首个明确原因。先评估并记录全部候选，再按固定方向顺序 left、right、front 选择：每个方向取 seed 最小且未被前面方向占用的合格案例。三个入选案例因此使用不同 seed，选择结果不受文件遍历顺序影响。若某方向在 55–100 中没有可选案例，筛选整体失败，不扩大范围、不降低阈值、不调用 VLM。

筛选结果保存在：

```text
vlm_smoke_screening_runs/run_<timestamp>/
├── seed_055_left/
│   ├── step_00.jpg
│   └── ...
├── candidate_trace.jsonl
└── screening_summary.json
```

逐候选记录 seed、方向、每步末端位置、请求目标、控制误差、可见像素、可见率、最终真实 XY 距离、`qualified` 和淘汰原因。筛选脚本只输出推荐选择，不自动修改 `sim_config.yaml`。通过程序和人工图片审计后，再把三个 `seed + direction` 固定写入配置并提交，避免实验输出直接改源码。

## Clear 场景资格

segmentation 只作为仿真 smoke test 的场景资格检查，不提供目标坐标或动作方向。固定相机下，独立验证 diagnostics 的红块完整参考像素中位数为 378，因此：

```text
visibility_ratio = block_visible_pixels / 378
```

比例裁剪到 `[0,1]`，每一步必须 `>=0.75`。若低于阈值，episode 以 `visibility_out_of_scope` 中止并记为失败。筛选完成并冻结正式案例后，不得在正式运行中临时更换 seed、删除样本或声称是控制算法失败。

## 安全中止

以下任一情况立即停止当前 episode，保存已有证据并记为失败：

- `api_error`：Qwen 请求最终失败；
- `invalid_box`：回复无法解析或缺少合法 `red_block` 框；
- `backprojection_error`：像素或相机矩阵无法得到有限世界坐标；
- `target_out_of_workspace`：补偿目标越过工作区边界；
- `visibility_out_of_scope`：任一步可见率低于 0.75；
- `target_jump`：相邻两次补偿目标的 XY 跳变大于 0.03m；
- `no_progress`：连续两个动作后，基于新一帧预测的末端到目标距离均未至少改善 0.005m；
- `ik_error`：IK 返回非法关节值或执行过程异常；
- `max_control_steps`：第 10 步后仍未输出 stop。

安全中止只证明保护机制正常，不算任务成功。

## Stop 与评分

控制器根据预测目标决定 stop，不读取真值。输出 stop 后不再移动，由评分层计算末端与真实红块中心的 XY 距离：

- 距离 `<=0.03m`：`success`；
- 距离 `>0.03m`：`false_stop`。

真实距离只决定实验评分标签，不会生成下一步动作。若 10 步内没有 stop，则结果为 `max_control_steps`，即使事后真实距离偶然小于 3cm，也不修改终止原因。

## 输出与可追溯性

每次运行创建独立时间戳目录：

```text
vlm_smoke_runs/run_<timestamp>/
├── episode_000/
│   ├── step_00.jpg
│   ├── step_00_annotated.jpg
│   └── smoke_trace.jsonl
├── episode_001/
├── episode_002/
└── smoke_summary.json
```

已有运行目录保持只读，入口不得覆盖。如果进程中断，保留不完整目录并创建新的时间戳运行；新的运行从场景搭建阶段重新开始，也重新发起对应 Qwen 请求。闭环物理状态不允许从旧 step 中途续接，避免旧模型回复与新的机器人状态错配。30 次 API 上限按单次完整运行计算。

逐步 trace 至少保存：

- episode、seed、起始方向和 control step；
- 原图和标注图路径；
- Qwen 原始回复、red block 框和 API latency；
- 框中心像素、原始世界坐标、冻结补偿和补偿后坐标；
- 动作前后末端位置、预测距离、目标跳变和进展计数；
- 选择方向、2cm 目标位置和关节执行诊断；
- 可见像素、可见率和 clear 资格；
- 仅评分使用的真实红块位置、真实 XY 距离和终止原因。

`smoke_summary.json` 至少保存：

- 3 个 episode 的成功数、成功率和通过判定；
- 每个 episode 的 seed、起始方向、步数、API 调用数和终止原因；
- 初始/最终真实 XY 距离；
- 预测目标跳变最大值；
- error/termination reason 计数；
- clear 资格是否全程满足；
- 总 API 调用数，必须不超过 30。

## 验收规则

只有同时满足以下条件，smoke test 才 `passed=true`：

1. 正式案例来自 seeds 55–100 的动态轨迹筛选，left/right/front 各一个且 seed 不重复，冻结后各运行一次且没有替换；
2. 三个 episode 全程可见率均 `>=0.75`；
3. 三个 episode 都在最多 10 步内由预测坐标触发 stop；
4. stop 时三个真实 XY 距离都 `<=0.03m`；
5. 没有安全中止、非法坐标或真值参与动作选择；
6. 总 Qwen 请求次数 `<=30`；
7. 冻结补偿在整个运行中没有被修改或重新拟合。

任何一项不满足均为 `passed=false`。失败仍是有效实验结果，必须根据 trace 区分视觉定位、坐标转换、安全保护、IK 执行或步数预算问题。

## 测试策略

### 纯函数单元测试

- 校准文件来源、样本数和补偿值校验；
- 红块框中心到补偿世界坐标；
- stop、X/Y 主轴选择和方向符号；
- 2cm 单步与 0.20m 固定 Z；
- 工作区边界包含等号，越界拒绝；
- 3cm 目标跳变边界；
- 连续两步、最小 0.005m 改善的 no-progress 状态机；
- 非有限坐标、布尔数值、空框和非法相机矩阵。

### 模拟 API 集成测试

- 使用固定合法框驱动多步移动并生成 success；
- 每一步都会重新调用 grounding；
- stop 只由预测坐标触发；
- false stop、安全中止、最大步数和 trace 字段完整性；
- 控制决策函数参数和 mock 调用中不出现 `block_pos`；
- 校准参数在 episode 前后完全相同。

### 动态筛选测试

- 候选必须覆盖起点和四次 2cm 动作后的五个观察点；
- 任一观察点可见率低于 0.75 时，以 `visibility_below_threshold` 淘汰；
- 起点或动作目标误差超过 0.005m 时，以 `start_pose_error` 或 `motion_target_error` 淘汰；
- 最终真实 XY 距离超过 0.03m 时，以 `final_distance_error` 淘汰；
- left/right/front 分别选择 seed 升序下第一个合格且互不重复的案例；
- 某方向没有合格案例时整体失败且不生成配置修改；
- mock Qwen 调用数必须为 0，筛选输出不得包含可被正式控制读取的动作真值文件。

### 真实验证

先运行完整自动测试，再执行无 API 动态筛选并审计三个入选案例的轨迹和图片。人工确认后把选择固定进配置，再显式执行一次 3 episode smoke test。真实请求不复用旧运行的逐步回复；如果运行中断，新运行必须从场景初始状态重新开始。运行完成后审计 3 个 trace、最多 30 条 API 结果、summary 判定和真值隔离字段。

## 非目标

- 不修改或替代 Stage 3 的 50/50 heuristic 基线；
- 不运行 20 episode 正式在线评估；
- 不处理 partial/severe 遮挡恢复；
- 不在本轮正式运行中加入 back；它的确定性方向映射由单元测试保护，待遮挡恢复阶段再做在线验证；
- 不使用双视角、历史帧或主动避让；
- 不训练 LoRA 或 action-token 模型；
- 不把 PyBullet 红块真值提供给 Qwen、反投影或动作选择函数。
