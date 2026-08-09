# 双积木 v2 专家数据设计

## 目标

重新建立可用于语言条件 VLA 训练和成对反事实评估的300条斜视双积木专家数据。新数据
必须消除红蓝块重叠与颜色—位置捷径，保存可确定性复现场景的完整动态状态，并通过专门
的 pilot 与 scale 质量门禁。旧 `expert_multi_v1` 保持只读，不回写或复用。

## 已知问题

`expert_multi_v1` 独立采样红蓝块，两个范围重叠。episode 0 的红块在 settle 后从配置范围
内被推到 `y=0.3517`，低于红块配置下限0.38，证明积木初始碰撞会改变场景。蓝块还只在
正 X 区域采样，模型可能利用位置而非语言识别颜色。

旧帧行的 `block_pos` 只表示当前目标块，而摘要的 `initial_block_pos` 固定表示红块；蓝色
任务因此无法从数据恢复另一个积木的位置和姿态，也不能按原场景做确定性成对 rollout。

## 方案选择

采用“同分布联合拒绝采样”，不采用红左蓝右的固定分区，也不采用目标块与干扰块不同
分布：

- 红蓝块都从同一个 XY 工作区采样，颜色边际位置分布一致；
- 两个中心必须在 X 或 Y 至少一个轴上达到配置的最小间距，按轴间距判断可避免仅靠欧氏
  距离合格但 AABB 仍重叠；
- 达到最大尝试次数仍无合法组合时立即失败，不降低间距或静默接受碰撞；
- task 选择与位置采样解耦，300条数据确定性平衡为红/蓝各150条。

这种方案比固定左右分区稍多拒绝采样，但不会引入颜色与绝对位置的人工相关，是当前任务
最小且可审计的选择。

## 配置与版本

新数据目录固定为 `outputs/dataset/expert_multi_v2/`，schema 固定为
`expert_multi_v2`。配置新增联合采样块，至少包含：

```yaml
pair_sampling:
  enabled: true
  x_range: [-0.2, 0.2]
  y_range: [0.38, 0.5]
  z: 0.1
  min_axis_separation_xy: 0.12
  max_attempts: 100
  max_settle_drift_xy: 0.005
  max_episode_drift_xy: 0.005
```

红蓝块仍分别保留颜色和缩放参数，但位置只能由联合采样器产生。任务策略使用
`balanced_alternating`，由 dataset 基础 seed 与 episode index 决定起始相位；任意偶数
规模中两种任务严格等量，追加采集不会改变既有 episode 的任务。

配置中的计划规模为300。采集 CLI 支持仅覆盖本次运行的 episode 数：第一次运行10条
pilot，门禁通过后向同一版本目录追加290条。manifest 保存计划规模和稳定采样契约，配置
快照不因 pilot 命令缩小为10条。

## 数据 schema

### 帧记录

为兼容已有训练读取器，保留 `block_pos`，其含义冻结为“当前目标块位置”。每帧新增
`scene_state`：

```json
{
  "target_block": "red",
  "blocks": {
    "red": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]},
    "blue": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}
  },
  "robot": {
    "joint_positions": [0, 0, 0, 0, 0, 0, 0],
    "joint_velocities": [0, 0, 0, 0, 0, 0, 0],
    "ee_position": [0, 0, 0]
  },
  "camera_eye": [0, 0, 0]
}
```

`target_pos` 必须由同一帧 `scene_state.blocks[target_block].position` 加 hover height 得到；
`block_pos` 必须与目标块位置在 `1e-6` 容差内一致。所有数组长度和有限性都属于 schema
契约；四元数范数必须在 `1e-6` 容差内等于1。

### Episode 摘要

摘要新增 `initial_scene_state` 和 `final_scene_state`，两者使用相同结构。初始状态在两个
积木 settle 完成、正式动作开始前捕获；最终状态在最后一步之后捕获。保留旧摘要字段作为
兼容投影，但 `initial_block_pos`/`final_block_pos` 明确表示目标块而不是固定红块。

异常摘要仍写入 episode 级错误证据；若完整场景尚未建立，相应 scene state 为 `null`，
质量门禁必须把它计为失败，不能当作普通空值跳过。

## 运行时流程

每个 episode 使用已有规则 `random_seed = dataset.random_seed + episode_idx`：

1. 重置机器人到 home pose并稳定；
2. 确定性选择本 episode 的红/蓝任务；
3. 从同一 RNG 流联合采样两个不重叠位置；
4. 使用显式位置加载红蓝块，settle 后再次验证 XY 间距和位置漂移；
5. 不合格时在写任何图片或轨迹前以 episode error 终止；
6. 固定斜视相机，捕获初始完整场景状态；
7. 专家始终控制到指令指定颜色上方，同时每个保存帧记录两个积木状态；
8. 写最终完整状态并删除两个积木。

相机继续使用当前斜视配置；俯视数据只作为历史派生对照，不进入 v2 主数据。

## 质量门禁

`evaluate_dataset` 对 `expert_multi_v2` 增加以下检查，同时保留原图片、动作9维、step、seed、
终止帧和成功距离检查：

- 每个成功 episode 和每帧均包含两个颜色的有限位置与单位四元数姿态；
- `target_block` 只能是 red/blue，且与中文指令、`block_pos`、`target_pos` 一致；
- 初始红蓝块满足配置的轴向最小间距，没有重叠；
- settle 后相对采样 XY 漂移不超过5mm，episode 内任一积木相对初始状态的 XY 漂移不超过
  5mm；
- episode seed、相机和初始/最终 scene state 完整；
- 10条 pilot 必须5红5蓝且全部成功；
- 300条 scale 必须150红150蓝、全部成功且完整性错误计数为0。

报告保存每类错误计数、红蓝任务计数、最小观测间距、最大积木 XY 漂移和 active gate。
任何门禁失败都保留原始数据与报告，不删除后重采或覆盖。

## 测试策略

先写失败测试，再修改生产代码：

1. 联合采样器在相同颜色分布下始终满足轴向间距，并在不可能配置上有限次数失败；
2. balanced task 对任意连续偶数 episode 精确等量且按 seed/index 可复现；
3. 帧与摘要写入完整双积木、机器人和相机状态，兼容字段语义一致；
4. episode 集成测试证明显式采样位置传入两个加载器，目标块选择不改变另一块记录；
5. evaluator 对缺块、NaN、错误四元数、目标/指令不一致、重叠、漂移和任务不平衡逐项红灯；
6. pilot 10条与 scale 300条的合成报告分别验证5/5和150/150门禁；
7. 输出路径、非空目录追加兼容和旧 `expert_v1` 数据评估保持回归通过。

代码、测试、编译和格式检查通过后才允许采集10条 pilot；pilot 报告通过后才允许追加290
条。最终300条门禁通过前，不进入 VLA 重训或成对反事实评估。

## 结论边界

新数据可以检验语言条件模型在受控双积木场景中的行为，但仅凭采集成功不能恢复旧70%
语言跟随结论。该结论必须等待相同 episode seed 和保存位姿的成对反事实 rollout；每对
红/蓝指令都要独立记录系统错误、目标选择和物理成功。
