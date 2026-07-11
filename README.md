# VLA Project: PyBullet 机械臂数据采集基线

这是一个面向 VLA（Vision-Language-Action）学习的个人实践项目。当前目标不是一开始训练大模型，而是先在 PyBullet 中跑通一个可诊断、可复现的数据采集闭环：

`红色积木随机位置 -> 生成语言指令 -> 机械臂 IK 控制 -> 保存 RGB 图像和动作标签 -> 记录 episode 诊断信息`

当前任务指令是：

```text
悬停在红色积木上方
```

## 当前阶段

当前主线是：Baseline 1 数据基线基本可用，正在推进 Stage 3 闭环评估。

也就是说，离线专家采集已经能生成可诊断的悬停数据；下一步重点不是继续盲目采图，而是验证在线闭环控制在多次随机目标位置下是否稳定。

已经具备：

- PyBullet KUKA iiwa 仿真环境。
- 配置驱动的数据采集脚本。
- 红色积木随机位置采样。
- IK 控制机械臂移动到积木上方悬停点。
- RGB 图像、语言指令、动作标签的 JSONL 数据输出。
- `termination_reason`、`distance_to_target`、episode 摘要等基础可观测性。
- 阶段三最小闭环探路脚本，可用启发式或 OpenAI 兼容多模态 API 输出方向控制。
- heuristic 和 API 共用同一条世界坐标方向执行路径，trace 会记录 `running`、`success` 或 `max_control_steps`。

暂时不做：

- 不训练大型 VLA 模型。
- 不做真实机械爪完整抓取。
- 不处理复杂多物体任务。
- 不把采集生成的大量图片直接提交到 GitHub。

## 文件结构

```text
.
├── control_arm.py       # 数据采集主脚本
├── stage3_probe.py      # 阶段三闭环探路脚本
├── sim_config.yaml      # 仿真、相机、任务、数据集配置
├── requirements.txt     # Python 依赖
├── README.md            # 项目说明
├── WORKLOG.md           # 从零开始的项目工作日志
└── .gitignore           # Git 忽略规则
```

运行后会生成：

```text
dataset/
├── trajectory_expert.jsonl
├── episode_summary.jsonl
└── ep_x_step_y.jpg

probe_runs/
├── probe_trace.jsonl
└── probe_step_xx.jpg
```

这些目录是实验输出，不建议直接提交到 GitHub。需要分享数据时，建议单独打包、上传到网盘、Hugging Face Dataset 或 GitHub Release。

## 环境准备

建议使用 Python 3.10。PyBullet、OpenCV 等库对 Python 3.10 的兼容性更稳。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果本机只有 `python3`，也可以先尝试：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 采集数据

主要参数在 `sim_config.yaml` 中调整。常用模式：

- 想看窗口调试：`connection_mode: "GUI"`，`enable_time_sleep: true`
- 想快速批量采集：`connection_mode: "DIRECT"`，`enable_time_sleep: false`
- 想覆盖旧数据：`dataset.clean_before_run: true`
- 想追加数据：`dataset.clean_before_run: false`

运行：

```bash
python control_arm.py
```

采集结果会写入 `dataset/`：

- `trajectory_expert.jsonl`：训练样本，一行对应一帧图像。
- `episode_summary.jsonl`：每条轨迹的摘要，便于判断成功、卡住、超步数等失败原因。
- `ep_*_step_*.jpg`：RGB 观测图。

## 阶段三闭环探路

`stage3_probe.py` 用来验证“观测图 -> 决策方向 -> 机械臂移动”的最小闭环。

默认使用启发式模式：

```yaml
probe:
  mode: "heuristic"
```

运行：

```bash
python stage3_probe.py
```

如果要接入 OpenAI 兼容的多模态 API，把 `sim_config.yaml` 中的 `probe.mode` 改为 `api`，并设置环境变量：

```bash
export VLA_API_BASE_URL="http://host:port/v1"
export VLA_API_KEY="your_api_key"
export VLA_MODEL_NAME="your_model_name"
python stage3_probe.py
```

## Stage 3 批量评估

批量评估使用固定随机种子运行 heuristic probe，并为每个 episode 保存独立 trace：

```bash
conda run -n vla_env python evaluate_probe.py
```

运行次数、输出目录和失败图片策略由 `sim_config.yaml` 的 `probe_evaluation` 控制。结果写入 `probe_eval_runs/run_*/`；成功 episode 只保留 trace，失败 episode 保留 trace 和步骤图片。

当前首批 20 次结果为 5 成功、15 次 `max_control_steps`，成功率 25%。这说明评估系统已经可用，但 heuristic 控制尚未达到 80% 的阶段门槛。

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

更细的推进过程、调参原因、数据质量判断和阶段三 probe 细节，记录在 `WORKLOG.md`。

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

只有当流程逻辑本身要变化时，才改 `control_arm.py` 或 `stage3_probe.py`。

## GitHub 仓库管理建议

建议提交：

- `control_arm.py`
- `stage3_probe.py`
- `sim_config.yaml`
- `requirements.txt`
- `README.md`
- `.gitignore`

建议忽略：

- `dataset/`
- `probe_runs/`
- `__pycache__/`
- `.venv/`
- IDE 本地配置
- 大模型权重、临时日志、压缩包

如果后续要保存一批“代表性小样本”，可以新建 `examples/`，只放少量图片和脱敏 JSONL 片段，不要把完整数据集混进源码仓库。
