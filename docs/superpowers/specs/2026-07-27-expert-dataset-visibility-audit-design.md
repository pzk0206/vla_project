# Expert v1 视觉可学习性与遮挡审计设计

## 背景

`expert_v1` 已完成300条真实 PyBullet 专家轨迹，得到300/300 success、9,894帧、
0项完整性错误，并通过 scale gate。但当前成功只证明使用 PyBullet 世界真值和 IK 的
专家控制器能够到达悬停目标，不证明每张 RGB 输入都足以让视觉模型推断相同动作。

历史采集调用 `capture_rgb_and_segmentation()`，但写盘时只保存 RGB JPEG，没有保存
segmentation。直接对 JPEG 做红色阈值会把 KUKA 的橙红色关节误认为红块，因此不能作为
可信遮挡标签。

本阶段只审计红块视觉可见性及其时序分布，不修改、删除、移动或筛选现有训练数据，也不
改变现有 scale gate。

## 方案比较

### 方案 A：确定性重放并恢复 segmentation（采用）

使用数据集的配置快照、episode seed 和保存的 step 编号重新执行相同仿真，在对应 step
获取 RGB 与 segmentation。逐帧验证重放 JPEG 与原始 JPEG 精确一致，或只含全量诊断
界定的严格渲染舍入差异后，才使用 segmentation 生成红块可见率真值。

优点：

- 对现有9,894帧补充物体身份真值，无需重采数据；
- 不受 KUKA 橙红色外观干扰；
- 能对每一帧建立原图与 segmentation 的可追溯关系。

风险是硬件渲染或仿真状态可能不能完全重现，因此重放一致性必须是发布标签前的硬门禁。

### 方案 B：RGB 颜色阈值加投影区域

根据已保存的相机和红块世界坐标，把预期红块区域投影到图像，再在局部区域统计红色像素。
运行更快，但机器人遮挡区域中的橙色、JPEG 压缩和光照仍会产生误差，只适合作为重放失败
后的近似诊断，不能称为真值标签。

### 方案 C：重新采集带 segmentation 的数据集

修改 schema 并重采300条，能直接保存 segmentation，但会引入新的数据版本、额外存储和
与现有已验收数据的对应问题。本阶段没有证据需要放弃现有数据，因此不采用。

## 可行性证据

在不写入文件的探针中，使用 episode 298（seed 1298）重放 step 0、360、720、744：

```text
camera_eye 最大差值：0
四个重放 JPEG 与原始 JPEG 解码后 MAE：0
无遮挡红块参考像素：221
step 720 红块可见像素：92
step 720 可见率：92 / 221 = 0.416
```

该结果证明当前机器和渲染环境能够精确重放至少一个包含明显遮挡的真实 episode。全量
审计仍必须逐帧重新验证，不能把单个探针推广成无条件保证。

## 输入与权威来源

审计输入固定为：

```text
outputs/dataset/expert_scaling_v1/
├── config_snapshot.yaml
├── dataset_manifest.json
├── trajectory_expert.jsonl
├── episode_summary.jsonl
└── ep_*_step_*.jpg
```

规则：

- 仿真、相机、任务和控制参数来自数据集内的 `config_snapshot.yaml`，不使用可能已经
  改变的当前 `sim_config.yaml` 作为重放真值。
- episode 集合和 seed 来自 `episode_summary.jsonl`，并继续验证
  `random_seed == manifest.random_seed + episode_idx`。
- 每个 episode 需要捕获的 step 来自 `trajectory_expert.jsonl`。
- 原始 JPEG 路径来自逐帧记录；路径不存在、重复 step 或无法读取时立即失败。

## 重放流程

每条 episode 独立执行：

1. 连接新的 DIRECT PyBullet 世界。
2. 使用摘要中的 seed 设置 Python 随机状态。
3. 按原流程把机器人复位到 home pose，运行初始 settle。
4. 生成红块、运行物体 settle、采样本 episode 固定相机。
5. 校验重放相机位置与该 episode 记录的 `camera_eye` 完全一致。
6. 从 step 0 运行到该 episode 的最后保存 step；每个 step 重新计算悬停目标、IK 和电机
   目标，并推进一次物理仿真。
7. 在记录要求的 step 同时捕获 RGB 和 segmentation。
8. 使用与原采集相同的 OpenCV 默认 JPEG 编码把重放 RGB 编码并解码。精确一致直接
   接受；非精确帧仅在 `MAE <= 0.002` 且最大单通道误差不超过3时作为严格容差匹配
   接受，否则拒绝该帧 segmentation。
9. 统计 segmentation 中 object id 等于红块 body id 的像素数。
10. 在每个保存 step 临时把机器人所有 visual shape 的 alpha 设为0，在物理状态、红块
    位姿和相机完全不变时再次渲染 segmentation，得到该帧自己的无遮挡参考像素数；
    随后恢复机器人原始颜色，再继续物理仿真。
11. 每帧参考像素必须大于0；该帧可见像素不得大于同帧参考像素。
12. 使用逐帧参考值计算可见率并分类，然后断开本 episode 的 PyBullet 连接。

PyBullet 的 segmentation 编码包含 object id 和 link index；提取 object id 时使用低
24位，背景值不得误算为红块。

不能在 episode 末尾只渲染一次无遮挡参考并把它用于整条轨迹。全量首次运行发现，
episode 0 的红块在初始 settle 后仍有约0.52毫米位姿变化，导致无遮挡投影在251和252
像素之间变化。跨帧共用终止参考会把合法的252像素错误判为大于251。透明机器人探针
把 episode 298 step 720 的红块从92像素恢复到221像素、机器人 mask 变为0，并且恢复
颜色后的 RGB 完全一致；这种方法不保存/恢复或修改物理状态。

全量只读 RGB 诊断还发现，9,894帧中9,888帧逐像素完全一致，6帧只有稀疏 OpenGL
重渲染舍入差异：最坏 MAE 为0.001256，最大单通道误差为3，最多改变132/196,608个
通道值。严格容差以略高于实测 MAE 上界的0.002和实测最大通道误差3为门禁；结果必须
分别记录 exact 和 tolerance 数量，不能把容差匹配称为精确匹配。

## 重放一致性硬门禁

以下任一条件发生时，本次全量审计失败：

- 输入 manifest、配置快照、JSONL 或原始 JPEG 缺失/损坏；
- episode、seed、step 或 camera_eye 不一致；
- 重放 JPEG 与原始 JPEG 的 MAE 大于0.002或最大单通道误差大于3；
- 保存 step 没有全部重放；
- 参考像素不为正；
- 可见像素为负或大于参考像素；
- PyBullet、IK、渲染或文件读取发生异常。

失败时写入 `visibility_audit_failure.json`，包含 episode、step、原因和可恢复的差异指标；
不得发布 `frame_visibility.jsonl` 或一个看似成功的 summary。原始数据始终保持不变。

成功输出采用临时文件完成后原子改名，避免中断留下半份可信标签。

## 可见率与分组

每帧：

```text
block_visibility_ratio = visible_block_pixels / reference_block_pixels
```

沿用项目已有分组边界：

```text
severe：  0.00 <= ratio < 0.25
partial： 0.25 <= ratio < 0.75
clear：   0.75 <= ratio <= 1.00
```

边界包含关系必须与现有 `classify_visibility()` 保持一致，不新建第二套分类语义。

## 输出契约

成功输出目录：

```text
outputs/dataset/expert_scaling_v1/visibility_audit_v1/
├── frame_visibility.jsonl
└── visibility_audit_summary.json
```

`frame_visibility.jsonl` 每行对应原始一帧，至少包含：

```json
{
  "schema_version": "visibility_audit_v1",
  "episode_idx": 298,
  "random_seed": 1298,
  "step_idx": 720,
  "image_path": "outputs/dataset/expert_scaling_v1/ep_298_step_720.jpg",
  "visible_block_pixels": 92,
  "reference_block_pixels": 221,
  "block_visibility_ratio": 0.416,
  "visibility_group": "partial",
  "replay_pixel_mae": 0.0,
  "replay_max_pixel_error": 0,
  "replay_changed_pixel_values": 0,
  "replay_exact_match": true,
  "replay_within_tolerance": true
}
```

`visibility_audit_summary.json` 至少包含：

- schema、输入数据集和创建时间；
- episode 数、帧数、重放精确匹配数、严格容差匹配数和拒绝数；
- clear、partial、severe 帧数与比例；
- 至少出现一次 partial/severe 的 episode 数；
- 初始帧与终止帧的分组和可见率统计；
- 每条 episode 以及全局最长连续 partial/severe 保存帧数；
- 可见率 min/mean/median/max；
- 可见率最低若干帧的原始图片路径；
- `replay_validation.passed=true`。

连续遮挡按保存帧序列计算，而不是按仿真 step 数伪装精度；报告同时保留起止 step，便于
换算真实控制跨度。

输出是诊断证据，不新增“数据集视觉门禁通过”结论。是否筛选训练数据必须根据真实分布
另行设计。

## 代码位置与命令

正式实现放入：

```text
src/vla_project/simulation/audit_dataset_visibility.py
```

测试放入：

```text
tests/simulation/test_audit_dataset_visibility.py
```

注册命令：

```text
vla-audit-dataset-visibility
```

该职责属于仿真数据集离线审计，不放入 VLM grounding 包，也不在仓库根目录新增脚本。

## 测试策略

采用 TDD，覆盖：

- segmentation object id 解码与红块像素计数；
- 0、0.25、0.75、1.0 分类边界复用；
- 参考像素和可见像素非法值拒绝；
- episode/step/seed/camera 不一致拒绝；
- JPEG 精确匹配、严格渲染容差边界和明显不匹配拒绝；
- clear/partial/severe、初始/终止帧和最长连续遮挡汇总；
- 失败时不发布成功 JSONL；
- 小型真实 DIRECT episode 的确定性重放集成测试；
- CLI、配置路径与包元数据契约。

实现完成后运行定向测试、完整测试、`compileall`、`git diff --check`，再对300条数据运行
全量审计。只有全量逐帧重放匹配全部通过，才记录遮挡分布并更新项目当前状态。
