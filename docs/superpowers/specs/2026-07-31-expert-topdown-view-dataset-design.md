# Expert 垂直俯视派生数据集设计

## 背景与目标

现有 `expert_scaling_v1` 已通过300条规模化门禁、逐帧可见性审计和动作表示审计：
300/300 success、9,894帧、0项完整性错误。它使用每个 episode 带少量随机扰动的
224×224斜视相机，而已经验证的 VLM grounding、反投影和在线 smoke 使用固定
448×448垂直俯视相机。

后续轻量行为克隆控制器预计使用与 VLM 相同的固定俯视观测。直接用斜视图片训练会造成
训练和部署的相机分布不一致；现有少量 VLM 样本又只有方向或 grounding 评估标签，不能
替代完整专家轨迹。

本阶段的目标是：保持原300条物理轨迹、动作、本体状态和终止标签不变，确定性重放每个
已保存状态，使用现有 VLM 垂直相机重新渲染全部9,894帧，发布一个可独立扫描和训练的
派生数据集。原始斜视数据不得修改、移动或删除。

## 已确认的方案

采用“完整俯视重渲染 + 保留斜视源数据”的方案：

- 主训练候选使用完整俯视派生数据，与最终 VLM/控制相机一致；
- 原斜视数据保留为回放真值、单视角对照、遮挡比较和未来多视角训练候选；
- 现有少量 VLM 校准、验证和 smoke 图片继续只作为评估证据，不混入训练数据；
- 第一轮不同时实现多视角训练，避免把相机一致性验证和泛化实验混为一个变量。

未采用的替代方案：

1. 直接训练斜视图片：实现最快，但与预期部署相机不一致。
2. 直接使用旧 VLM 图片：样本少且缺少逐帧7关节动作、`ee_pos` 和 `stop` 标签，还会
   污染既有评估边界。
3. 第一轮同时训练斜视与俯视图片：可能提高视角鲁棒性，但增加实验变量；留到俯视单视角
   基线完成后再决定。

## 权威输入与派生关系

唯一源数据集为：

```text
outputs/dataset/expert_scaling_v1/
├── config_snapshot.yaml
├── dataset_manifest.json
├── trajectory_expert.jsonl
├── episode_summary.jsonl
└── ep_*_step_*.jpg
```

派生过程只从源数据集自身读取重放契约，不用可能已经变化的当前 `sim_config.yaml` 代替
源配置。具体规则如下：

- episode、seed、保存 step、动作和本体状态来自源 JSONL；
- 仿真、机器人、任务和原斜视相机来自源 `config_snapshot.yaml`；
- 新相机由源配置中的基础 `camera` 与 `vlm_evaluation.camera_override` 合并得到；
- 合并后的相机必须为448×448、`eye_offset_base=[0.0, 0.0, 3.0]`、
  `eye_offset_random_range=[0.0, 0.0]`、`up_vector=[0, 1, 0]`、`fov=45`；
- 在工作区中心 `[0.0, 0.4, 0.0]` 下，新相机眼位置必须固定为
  `[0.0, 0.4, 3.0]`。

派生数据集输出为新的同级版本：

```text
outputs/dataset/expert_topdown_v1/
├── config_snapshot.yaml
├── dataset_manifest.json
├── trajectory_expert.jsonl
├── episode_summary.jsonl
├── view_generation_report.json
└── ep_*_step_*.jpg
```

它与 `expert_scaling_v1` 是“一套轨迹、两套视觉观察”的关系，不覆盖源目录，也不把新
图片写入源目录。

源 `config_snapshot.yaml` 是最初10条 pilot 创建时冻结的物理配置，因此其中
`dataset.num_episodes=10` 和 `clean_before_run=true` 不描述后来已经追加完成的300条规模。
episode 数量和 seed 集合继续以 manifest 与两个 JSONL 为权威；这些 pilot 时代字段在
派生快照中原样保留为历史配置证据，不用于决定派生数量，也不得拿派生快照重新运行采集。

## 代码边界

新增共享重放模块：

```text
src/vla_project/simulation/expert_dataset_replay.py
```

职责是读取并验证 expert 数据集契约、按 episode 恢复随机状态和 PyBullet 世界、按原
控制顺序推进到保存 step，并向调用方提供同一物理状态下的机器人、红块和相机上下文。
它不决定输出图片命名、可见性标签或训练划分。

新增派生视图生成器：

```text
src/vla_project/simulation/render_expert_dataset_view.py
```

职责是调用共享重放能力，在每个保存 step 先验证源斜视 JPEG，再渲染固定俯视 JPEG，
复制并改写派生 manifest、轨迹和 episode 摘要，执行发布门禁并原子发布完整目录。

现有 `audit_dataset_visibility.py` 随本次工作改为复用共享重放模块，但保持原命令、输出
schema、阈值和已有 `expert_scaling_v1` 审计语义不变。只做消除重放逻辑重复所需的定向
重构，不修改 grounding 或控制策略。

测试镜像放置：

```text
tests/simulation/test_expert_dataset_replay.py
tests/simulation/test_render_expert_dataset_view.py
```

新增命令：

```text
vla-render-expert-dataset-view \
  --source-dataset outputs/dataset/expert_scaling_v1 \
  --output-dir outputs/dataset/expert_topdown_v1
```

同时为现有质量扫描命令增加可选参数：

```text
vla-evaluate-dataset \
  --dataset-dir outputs/dataset/expert_topdown_v1
```

不传参数时继续读取 `sim_config.yaml` 中的现有 `dataset.output_dir`，因此原采集数据的默认
行为和配置冻结测试不变。只扩展 CLI 入口，不改变 `evaluate_dataset(dataset_dir)` 的扫描
语义。

## 确定性生成流程

每条 episode 在独立的 DIRECT PyBullet 世界中执行：

1. 校验 manifest、配置快照、episode 摘要和帧记录完整，并验证
   `random_seed == manifest.random_seed + episode_idx`。
2. 使用 episode seed 重置 Python 随机状态，按原配置建立世界、机器人 home pose、红块
   和初始 settle。
3. 使用源斜视相机配置采样原 episode 相机，并要求与源逐帧和摘要记录的 `camera_eye`
   完全相同。
4. 从 step 0 按原专家控制顺序推进到最后一个保存 step。
5. 每逢保存 step，先以源相机渲染并使用现有 JPEG 严格一致性规则验证源图片：精确匹配
   直接接受；只允许 `MAE <= 0.002` 且最大单通道误差不超过3的已知 OpenGL 舍入差异。
6. 在同一保存 step 重新读取计算出的7个关节目标、末端位置、红块位置、悬停目标和距离，
   分别与源行的 `action[:7]`、`ee_pos`、`block_pos`、`target_pos` 和
   `distance_to_target` 对齐；有限浮点量使用 `rtol=0, atol=1e-9`。夹爪和终止维度继续
   按源 schema 与 episode 终止契约验证。任何数值不一致都拒绝该 episode。
7. 在不推进仿真、不修改机器人或红块状态的情况下，使用固定俯视相机渲染448×448 RGB，
   以 OpenCV 默认 JPEG 编码写入临时目录。
8. 复制该帧的指令、9维动作、`block_pos`、`target_pos`、`ee_pos`、距离和终止原因；只把
   `schema_version`、`image_path` 和 `camera_eye` 改为派生数据契约，并增加
   `source_image_path`、`source_camera_eye` 供追溯。
9. episode 摘要保留所有运动和成功指标，把摘要相机改为固定俯视相机，并保存
   `source_camera_eye`。
10. 全部 episode 完成后执行数量、字段、标签等价、图片可读性、图片尺寸、seed、step、
   终止标志和源文件只读检查，通过后一次性原子发布目录。

生成前先解析真实路径并拒绝以下情况：输出等于源目录、输出位于源目录内部、源目录位于
输出内部、输出不在 `outputs/dataset/` 下，或者输出路径经过符号链接回到源数据。输出
图片路径统一写成仓库相对路径，图片直接位于派生数据集根目录，以兼容现有孤立图片扫描。

生成器拒绝覆盖已存在的输出目录，并在开始全量渲染前检查输出文件系统至少有2 GiB可用
空间。运行失败时在输出目录旁写入
`expert_topdown_v1_failure.json`，不得发布目标目录、成功 manifest 或半份可训练数据集。
正常异常由临时目录自动清理；进程被强制终止后遗留的 `.expert_topdown_v1.staging-*`
不视为已发布数据，下次运行拒绝自动删除它，而是在失败报告中列出并要求确认后清理。

## 派生数据契约

派生 manifest 使用 `schema_version: expert_view_v1` 和
`dataset_name: expert_topdown_v1`，继续保留现有质量扫描所需字段，并增加：

```json
{
  "derived_from": "outputs/dataset/expert_scaling_v1",
  "source_schema_version": "expert_v1",
  "view_name": "vlm_topdown",
  "camera_profile_source": "config_snapshot.yaml:vlm_evaluation.camera_override",
  "source_manifest_sha256": "...",
  "source_config_sha256": "...",
  "source_trajectory_sha256": "...",
  "source_summary_sha256": "...",
  "source_images_sha256": "..."
}
```

`source_images_sha256` 是对按 `(episode_idx, step_idx)` 排序后的每个源图片相对路径和文件
内容依次更新得到的聚合摘要；生成前后各计算一次，用来发现任何意外图片写入。它不代替
逐帧 JPEG 回放验证。

派生轨迹保持一帧一行、episode/step 排序和9维动作语义。除新增追溯字段及三项视图字段
外，下列字段必须与源记录逐值相等：

```text
episode_idx, step_idx, random_seed, instruction, action,
block_pos, target_pos, ee_pos, distance_to_target, termination_reason
```

派生摘要中所有轨迹结果字段必须与源摘要逐值相等；只有 schema、相机和追溯字段允许
变化。图片保留原始448×448分辨率，未来输入 ResNet-18 时由训练预处理缩放到224×224，
不得为了本阶段训练便利覆盖原始俯视渲染。

派生 `config_snapshot.yaml` 从源配置复制，只允许把顶层 `camera` 替换为合并后的 VLM
俯视相机，并把 `dataset.schema_version` 和 `dataset.output_dir` 改成派生值。manifest 中
的 `dataset_name` 单独设为 `expert_topdown_v1`。机器人、任务、控制、
随机 seed 和保存频率等重放参数必须保持不变。这样现有可见性审计读取派生快照时，会在
相同物理轨迹上重放并验证新的俯视 JPEG，而不会重新使用源斜视相机。

`view_generation_report.json` 记录源与派生路径、开始/完成时间、源文件哈希、重放匹配
数量、episode/帧/图片数量、字段等价检查、固定相机检查以及最终 `passed`。它只证明
派生关系和文件完整性，不代替后续通用质量报告或可见性审计。

manifest 额外声明 `derived_read_only: true` 和
`generation_method: deterministic_state_replay`。这表示该目录只能由派生生成器一次性
发布，不能作为 `vla-collect` 的追加或清理目标。

## 继承的时序语义

源采集每个仿真 step 的顺序是：计算 IK 关节目标、下发电机目标、推进一次物理仿真、读取
末端和物体状态、拍摄图片、把本 step 计算的关节目标写为动作。因此一行数据表示“动作
目标下发并执行一个1/240秒仿真 step 后的观察”，不是动作下发前的观察。

派生视图必须在源图片相同的仿真时刻渲染，并原样继承这项时序语义；本阶段不能通过前移
图片、后移 action 或重新计算标签悄悄改变它。`view_generation_report.json` 明确记录
`observation_action_timing: post_single_sim_step_with_current_target`。后续行为克隆设计必须
把这一点列为模型输入标签契约，并通过训练/验证结果判断1个仿真 step 的偏移是否需要另开
数据版本处理。

## 发布硬门禁

以下条件必须全部满足：

- 源数据仍为300个 episode、9,894帧，且 episode/step 键唯一；
- 9,894个源斜视保存帧均通过确定性重放 JPEG 验证；
- 每个保存 step 的关节目标、末端位置、红块位置、悬停目标和距离均与源记录在规定浮点
  容差内一致；
- 生成300个派生摘要和9,894张可读取的448×448 JPEG；
- 每个源帧恰好对应一个派生帧，没有缺失、重复或孤立图片；
- 所有动作、本体状态、物体状态、距离、seed 和终止标签与源数据逐值相等；
- 每个派生帧和摘要的相机眼位置均为 `[0.0, 0.4, 3.0]`；
- 生成前后源 manifest、配置、两个 JSONL 和源图片聚合 SHA-256 不变；
- 通用 `vla-evaluate-dataset` 对派生数据的完整性和 scale gate 报告通过。

本阶段不规定俯视图片必须达到某个 clear 比例，因为新视角的真实遮挡分布尚未测量。
不得先假设它优于斜视图，也不得为了通过门禁删除 partial/severe 帧。

## 视觉审计与训练门槛

派生数据发布后，立即复用 `vla-audit-dataset-visibility` 对
`expert_topdown_v1` 全量重放，生成它自己的：

```text
outputs/dataset/expert_topdown_v1/visibility_audit_v1/
├── frame_visibility.jsonl
└── visibility_audit_summary.json
```

审计沿用 severe `<0.25`、partial `[0.25, 0.75)`、clear `>=0.75` 的同一分类语义，
但参考像素必须在新相机下逐帧重新计算。原斜视审计结果继续描述源数据，不能复制为俯视
标签。

只有以下条件满足才进入行为克隆实施：

- 派生数据生成硬门禁通过；
- 派生数据通用质量报告通过；
- 新相机可见性审计完成，所有帧均有可信标签；
- 本阶段在发布新相机可见性分布后结束，不自动开始训练或筛除数据；下一份行为克隆设计
  必须引用该真实分布，再明确保留全部帧、按可见性筛选或增加多视角对照的选择。

动作 tokenization 的原始统计结论仍有效，因为动作没有改变；正式32箱 absolute_q 和
64箱 delta_q 边界仍按既定规则在 episode 级划分后只用训练集拟合，不能复用全量审计
边界。

## 测试与验证

实现采用 TDD，至少覆盖：

- 源数据契约、seed、重复 step 和相机不一致拒绝；
- 共享重放按保存 step 回调且不漏帧；
- 源 JPEG 精确匹配、严格容差和明显不匹配路径；
- 关节目标和逐帧状态数值精确容差边界及不一致拒绝；
- VLM camera override 合并后得到固定448×448相机；
- 派生帧只改变允许变化的字段，动作和状态逐值相等；
- 派生摘要只改变允许变化的字段；
- 图片命名、尺寸、数量、孤立文件和固定相机检查；
- 源/输出相同、互相嵌套、符号链接回源、越出 `outputs/dataset/` 和空间不足拒绝；
- 已存在输出目录拒绝覆盖；
- 中途异常不发布半份成功数据，强制中断残留可识别且不自动删除；
- 源元数据和全部源图片聚合哈希前后不变；
- 原可见性审计重构后的现有测试不回归；
- 新生成 CLI、质量扫描 `--dataset-dir` 和包元数据契约。

代码完成后依次运行定向测试、全量测试、`compileall` 和 `git diff --check`，再运行真实
300条全量生成、通用质量扫描与新相机可见性审计。最终只根据生成报告和审计原始摘要
更新 `PROJECT_OVERVIEW.md`、`CURRENT_STATUS.md`、`PROJECT_STRUCTURE.md` 和
`WORKLOG.md`，不预写实验指标。

## 非目标

本阶段不包含：

- 修改或删除 `expert_scaling_v1`；
- 重新采集专家动作或改变9维动作 schema；
- 使用旧 VLM 评估图片训练；
- 筛除任何可见性分组；
- 训练 ResNet-18、融合 MLP 或动作输出头；
- 斜视与俯视多视角联合训练；
- PyBullet 学习策略在线 rollout。

这些工作必须在俯视派生数据及视觉审计通过后，按后续行为克隆实施计划推进。
