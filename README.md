# VLA Project: PyBullet 机械臂数据采集基线

这是一个面向 VLA（Vision-Language-Action）学习的个人实践项目。当前目标不是一开始训练大模型，而是先在 PyBullet 中跑通一个可诊断、可复现的数据采集闭环：

`红色积木随机位置 -> 生成语言指令 -> 机械臂 IK 控制 -> 保存 RGB 图像和动作标签 -> 记录 episode 诊断信息`

当前任务指令是：

```text
悬停在红色积木上方
```

## 当前阶段

当前主线是：Baseline 1 数据基线可用，Stage 3 heuristic 闭环稳定性验收已经通过，VLM 直接方向基线、目标坐标反投影和清晰样本独立校准验证已经完成。

也就是说，离线专家采集已经能生成可诊断的悬停数据；在线 heuristic 闭环已在 50 个固定随机种子下稳定运行。Qwen 的 448px 四方向框可以支持方向级 `4/4` 判断；冻结 seeds 42–46 清晰样本拟合出的固定 XY 补偿后，未参与拟合的 seeds 47–51 清晰样本已全部低于 3cm。该结论仍限于离线固定相机和清晰视野，尚未直接接入在线控制。

已经具备：

- PyBullet KUKA iiwa 仿真环境。
- 配置驱动的数据采集脚本。
- 红色积木随机位置采样。
- IK 控制机械臂移动到积木上方悬停点。
- RGB 图像、语言指令、动作标签的 JSONL 数据输出。
- `termination_reason`、`distance_to_target`、episode 摘要等基础可观测性。
- 阶段三最小闭环探路脚本，可用启发式或 OpenAI 兼容多模态 API 输出方向控制。
- heuristic 和 API 共用同一条世界坐标方向执行路径，trace 会记录 `running`、`success` 或 `max_control_steps`。
- VLM 红块框中心到工作平面世界坐标的相机反投影与离线真值评分。
- PyBullet segmentation 可见率诊断，可把清晰、部分遮挡和严重遮挡样本分开统计。
- 校准集/验证集严格隔离的固定 XY 偏差验证，验证阶段不会重新拟合补偿。

五位置 grounding 评估使用 seeds 42–46、固定正俯视相机、448px 和 20cm 相对距离，共 20 张。20/20 返回合法框；整体 XY 误差 mean/median/max 为 `3.94/3.77/10.06cm`。其中 15 张清晰样本为 `3.34/3.45/4.00cm`，3 张严重遮挡样本平均 `6.92cm`。清晰样本全部表现为 X 负偏、Y 正偏，平均有符号偏差约 `(-2.49cm, +1.95cm)`；因此后续使用独立校准集和验证集测试固定补偿，没有直接把同集均值写进控制器。

独立验证冻结上述校准集得到的补偿 `(+2.492cm, -1.947cm)`，再使用全新的 seeds 47–51、20 张图片评分，没有用验证数据重新估计参数。20/20 返回合法框；可见率分组为 clear/partial/severe=`15/4/1`。clear 原始 mean/median/max 为 `2.99/3.13/3.92cm`，补偿后降为 `0.77/0.79/1.46cm`，15/15 全部 `<=3cm`，因此固定补偿通过预设离线验收。partial 也为 4/4 通过，但唯一 severe 样本补偿后仍为 `3.27cm`；整体为 19/20，不应把清晰场景结论外推到严重遮挡。

暂时不做：

- 不训练大型 VLA 模型。
- 不做真实机械爪完整抓取。
- 不处理复杂多物体任务。
- 不把采集生成的大量图片直接提交到 GitHub。
- 在完成小规模闭环 smoke test 设计和遮挡拒绝边界前，不把补偿后的 VLM 坐标直接接入正式在线控制。

## 文件结构

```text
.
├── pyproject.toml       # Python 包元数据和命令入口
├── src/vla_project/
│   ├── simulation/      # 仿真、采集、闭环控制与 probe 评估
│   ├── vlm/             # VLM 采样、grounding、反投影与校准评估
│   └── tools/           # 生成输出迁移等维护工具
├── tests/
│   ├── simulation/      # simulation 模块测试
│   ├── vlm/             # VLM 模块测试
│   └── tools/           # 维护工具测试
├── sim_config.yaml      # 仿真、相机、任务、数据集配置
├── requirements.txt     # Python 依赖
├── README.md            # 项目说明
├── docs/
│   ├── worklog/         # 项目推进和实验复盘
│   ├── planning/        # 学习计划及原始 PDF
│   ├── debugging/       # Bug 证据、实验和结论
│   └── superpowers/     # 设计规格与实施计划
└── .gitignore           # Git 忽略规则
```

运行后会生成：

```text
outputs/
├── dataset/                    # Baseline 1 训练图片、轨迹和 episode 汇总
├── probe/                      # Stage 3 单次闭环图片和 trace
├── probe_evaluations/          # Stage 3 批量评估运行目录
├── vlm_samples/                # Qwen 离线输入图片、样本清单和诊断信息
│   ├── default/
│   ├── 448/
│   ├── 448_multiseed_d020/
│   └── 448_calibration_validation_d020/
└── vlm_evaluations/            # VLM 预测、标注图、反投影和评估统计
```

这些目录是实验输出，不建议直接提交到 GitHub。需要分享数据时，建议单独打包、上传到网盘、Hugging Face Dataset 或 GitHub Release。

2026-07-18 以前分散在仓库根目录的生成输出已经迁移到 `outputs/`；旧路径与
新路径的完整映射记录在
[`docs/superpowers/specs/2026-07-18-generated-output-organization-design.md`](docs/superpowers/specs/2026-07-18-generated-output-organization-design.md)。

## 环境准备

建议使用 Python 3.10。PyBullet、OpenCV 等库对 Python 3.10 的兼容性更稳。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

如果本机只有 `python3`，也可以先尝试：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` 会以可编辑模式安装本项目，并注册下文使用的 `vla-*` 命令；
修改 `src/` 中的代码后无需重复安装。

安装后可直接使用以下命令：

| 命令 | 用途 |
| --- | --- |
| `vla-collect` | 采集 Baseline 训练数据 |
| `vla-probe` | 运行 Stage 3 单次闭环 |
| `vla-evaluate-probe` | 批量评估 heuristic 闭环 |
| `vla-collect-vlm-samples` | 生成固定 VLM 离线评估样本 |
| `vla-evaluate-vlm-decisions` | 评估 VLM 方向决策 |
| `vla-diagnose-grounding` | 诊断末端与红块 grounding 框 |
| `vla-ground-then-decide` | 评估 grounding 后的确定性方向判断 |
| `vla-evaluate-backprojection` | 评估框中心到世界坐标的反投影 |
| `vla-validate-grounding-calibration` | 验证冻结的 grounding 校准参数 |
| `vla-migrate-generated-outputs` | 迁移和检查历史生成输出 |

## 采集数据

主要参数在 `sim_config.yaml` 中调整。常用模式：

- 想看窗口调试：`connection_mode: "GUI"`，`enable_time_sleep: true`
- 想快速批量采集：`connection_mode: "DIRECT"`，`enable_time_sleep: false`
- 想覆盖旧数据：`dataset.clean_before_run: true`
- 想追加数据：`dataset.clean_before_run: false`

运行：

```bash
vla-collect
```

采集结果会写入 `outputs/dataset/`：

- `trajectory_expert.jsonl`：训练样本，一行对应一帧图像。
- `episode_summary.jsonl`：每条轨迹的摘要，便于判断成功、卡住、超步数等失败原因。
- `ep_*_step_*.jpg`：RGB 观测图。

## 阶段三闭环探路

`src/vla_project/simulation/stage3_probe.py` 用来验证“观测图 -> 决策方向 -> 机械臂移动”的最小闭环。

默认使用启发式模式：

```yaml
probe:
  mode: "heuristic"
```

运行：

```bash
vla-probe
```

如果要接入阿里云百炼 OpenAI 兼容的多模态 API，使用华北 2（北京）地域的
`qwen3-vl-flash`，把 `sim_config.yaml` 中的 `probe.mode` 改为 `api`，并设置环境变量：

```bash
export VLA_API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export VLA_API_KEY="<填写真实 Key，不要写入仓库>"
export VLA_MODEL_NAME="qwen3-vl-flash"
vla-probe
```

API Key 只能通过环境变量传入，禁止写入 YAML、Python、Markdown 或日志。如果以后
使用 `.env` 文件，必须先把 `.env` 加入 `.gitignore`。

## Stage 3 批量评估

批量评估使用固定随机种子运行 heuristic probe，并为每个 episode 保存独立 trace：

```bash
conda run -n vla_env vla-evaluate-probe
```

运行次数、输出目录和失败图片策略由 `sim_config.yaml` 的 `probe_evaluation` 控制。结果写入 `outputs/probe_evaluations/run_*/`；成功 episode 只保留 trace，失败 episode 保留 trace 和步骤图片。

修复 IK 冗余解未使用关节限位和当前姿态的问题后，Stage 3 使用固定种子
42-91 完成 50 次 heuristic 回归：50 次全部成功，成功率 100%，无失败和运行
异常；最终距离 mean / median / max 为 0.0193 / 0.0197 / 0.0291m，平均控制
步数为 37.58。结果保存在 `outputs/probe_evaluations/run_20260712_221135/`，已经达到
“至少 50 次且成功率不低于 80%”的阶段门槛。

## 数据格式

`trajectory_expert.jsonl` 每行是一个样本，核心字段包括：

- `image_path`：当前帧图像路径。
- `instruction`：语言指令。
- `action`：动作标签，当前为 7 维关节目标角 + gripper 占位 + terminate 标记。
- `camera_eye`：相机位置。
- `block_pos`：红色积木位置。
- `target_pos`：悬停目标点。
- `ee_pos`：机械臂末端位置。
- `distance_to_target`：末端到目标点距离。
- `termination_reason`：episode 终止原因。

`episode_summary.jsonl` 用来做数据质量审计，重点看：

- 成功率。
- `stuck`、`max_steps`、`success` 的比例。
- 每条轨迹最终距离。
- 失败轨迹是否集中在某些积木位置范围。

## Baseline 路线

项目文档：

- [工作日志](docs/worklog/WORKLOG.md)：推进过程、调参原因和阶段判断。
- [学习计划](docs/planning/vla_robotic_study_plan.md)：当前路线和验收门槛。
- [Bug 日志](docs/debugging/BUGLOG.md)：故障证据、根因假设和单变量实验。

### Baseline 0：跑通悬停闭环

目标：机械臂能根据红色积木位置移动到其正上方，并保存可训练的数据。

验收标准：

- 图片和 JSONL 一一对应。
- 红色积木在多数图片中清晰可见。
- `distance_to_target` 整体下降。
- 每条 episode 有明确终止原因。

### Baseline 1：提升数据质量

目标：减少卡住、重复帧和不可诊断失败。

当前重点：

- 调整相机视角和 FOV，让红色积木更清晰。
- 收窄或逐步放开积木随机范围。
- 用 `episode_summary.jsonl` 判断失败原因。
- 先保证数据可诊断，再进入下一阶段。

### Baseline 2：从悬停升级到接近

目标：机械臂先悬停，再沿 z 轴下降接近积木，但暂不闭合夹爪。

### Baseline 3：加入夹爪开合

目标：加入 open/close 动作标签和接触状态记录，但先不强求稳定提起。

### Baseline 4：完整抓取和评估

目标：随机初始位置下完成抓取、闭合夹爪、提起，并统计成功率和失败原因。

## 配置优先原则

这个项目刻意把可调参数放在 `sim_config.yaml`，而不是写死在 Python 代码里。

优先改配置：

- episode 数量。
- 是否清空旧数据。
- 相机位置、FOV、图像大小。
- 积木随机范围。
- 成功距离阈值。
- 卡住检测窗口。
- 机械臂最大速度。

只有当流程逻辑本身要变化时，才改 `src/vla_project/` 中的对应模块。

## GitHub 仓库管理建议

建议提交：

- `src/vla_project/`
- `tests/`
- `pyproject.toml`
- `sim_config.yaml`
- `requirements.txt`
- `README.md`
- `.gitignore`

建议忽略：

- `outputs/`
- `__pycache__/`
- `.venv/`
- IDE 本地配置
- 大模型权重、临时日志、压缩包

如果后续要保存一批“代表性小样本”，可以新建 `examples/`，只放少量图片和脱敏 JSONL 片段，不要把完整数据集混进源码仓库。
