# VLA 具身智能项目学习计划

这份计划是基于同目录的 `vla_robotic_study_plan.pdf` 重新整理后的项目版路线。原 PDF 更像从零开始的通用 12 周指南；当前仓库已经完成了 PyBullet 数据采集基线和阶段三 probe 雏形，所以这里不再重复基础环境扫盲，而是按当前进度重排为“工程成果 + 模型深度平衡”的路线。

## 当前项目定位

当前项目已经具备：

- PyBullet KUKA iiwa 仿真环境。
- 红色积木随机位置采样。
- IK 专家规则控制机械臂悬停到目标上方。
- RGB 图像、语言指令、动作标签 JSONL 输出。
- episode 级诊断摘要，包括 `termination_reason`、最终距离、最终目标位置等。
- 阶段三 `stage3_probe.py`，可以用启发式或 API 模式做在线闭环探路。

当前数据状态：

```text
dataset/episode_summary.jsonl: 50 条 episode
termination_reason: 50 条 success
dataset/trajectory_expert.jsonl: 286 帧训练样本
final_distance: min 0.0098m, median 0.0293m, max 0.0300m
```

当前 probe 状态：

```text
50 个固定种子 episode（42-91）
success: 50/50，success_rate: 100%
failure / error: 0 / 0
final_distance mean / median / max: 0.0193 / 0.0197 / 0.0291m
control_steps mean / median: 37.58 / 37
```

旧版 13 步 trace 虽然成功接近目标，但 heuristic 分支实际绕过了方向词，直接把完整悬停坐标交给 IK，不能作为离散方向闭环证据。修复后 heuristic 和 API 共用 `direction -> 世界坐标单步位移 -> IK` 执行路径；当前 50 次批量结果才是 Stage 3 的正式稳定性证据。

这说明 Baseline 0 已完成，Baseline 1 可用，Stage 3 heuristic 已通过稳定性验收。

首批 20 次批量评估曾只有 5 次成功。关节级诊断发现无约束 IK 将 KUKA 第 4
关节推到 `2.0944rad` 物理上限，造成 x/y 串扰和方向振荡。IK 加入关节限位、
活动范围和当前姿态 `restPoses` 后，相同 20 种子提升到 20/20；继续扩大到
50 个固定种子后仍为 50/50。当前阶段应描述为“heuristic 闭环稳定性验收
完成，准备进入 VLM 决策替换与对比”。

## 总路线

选择路线 B：工程成果 + 模型深度平衡。

这个路线的目标不是最快做一个 demo，也不是马上冲大模型训练，而是先把项目做成可复现、可评估、可解释的闭环系统，再逐步加入 VLM 决策和动作 tokenization。

主线如下：

```text
稳定专家采集
-> 自动化闭环评估
-> VLM 决策替换接口
-> 数据集规模化和质量分析
-> action tokenization
-> 轻量 LoRA/QLoRA 微调验证
```

## 第 1-2 周：Stage 3 probe 稳定化（已完成）

目标：让启发式闭环在随机红块位置下稳定接近目标，而不是只展示单次成功。

要做的事：

- 批量运行 `stage3_probe.py`，先从 20 次开始，再扩到 50-100 次。
- 统计每次最终 `distance_to_hover`、控制步数、是否到达停止阈值。
- 检查 `left/right/front/back/stop` 的方向映射是否持续让距离下降。
- 保存失败 trace，单独分析失败时的红块位置、末端位置和动作序列。
- 更新 `../worklog/WORKLOG.md`，把旧的 probe 失败判断改成当前真实状态。

重点学习：

- 闭环控制不是看单次截图，而是看多次随机测试的成功率。
- `distance_to_hover` 比“模型有没有输出方向”更重要。
- 启发式规则是后续 VLM 的 sanity check。如果规则都跑不稳，不能把锅甩给模型。

验收标准：

- 一次批量评估至少覆盖 50 个随机 episode。
- 启发式 probe 成功率达到 80% 以上。
- 每种失败都有明确的日志和解释。

## 第 3-4 周：自动化评估系统（已完成）

目标：把项目从“能跑”升级为“能评估”。

要做的事：

- 新增 `evaluate_probe.py`。
- 支持配置评估次数、输出目录和是否保存图片。
- 自动汇总：
  - success rate
  - mean / median final distance
  - mean / median control steps
  - failure reason
  - direction 分布
- 输出 `probe_eval_summary.json` 或 `probe_eval_summary.jsonl`。
- 在 `../worklog/WORKLOG.md` 中记录第一版评估结果。

重点学习：

- 实验指标设计。
- failure mode analysis。
- 如何把项目从脚本变成可复现实验。

验收标准：

- 一条命令能完成批量 probe 评估。
- README 或 WORKLOG 中有清晰的结果表。
- 能回答：系统什么时候成功、什么时候失败、失败后下一步改哪里。

## 第 5-6 周：VLM 决策替换接口

目标：把当前规则决策层抽象成可替换模块，为 API VLM 和本地模型做准备。

当前结构可以理解为：

```text
状态信息 -> heuristic 决策 -> direction -> PyBullet 控制
```

目标结构：

```text
图像 + 语言指令 -> decision module -> direction/action -> PyBullet 控制
```

要做的事：

- 统一决策接口，例如：

```python
def decide_direction(image_path, instruction, state, config):
    return {
        "direction": "right",
        "raw_response": "...",
        "decision_source": "heuristic"
    }
```

- 保留 `heuristic` 作为默认决策器。
- 新增 `api` 决策器，读取 `VLA_API_BASE_URL`、`VLA_API_KEY`、`VLA_MODEL_NAME`。
- 强制 VLM 只输出 `left/right/front/back/stop`。
- 对非法输出做兜底，例如记录 `invalid_response` 并回退到 heuristic 或提前终止。
- 对比 heuristic 和 VLM 的成功率、最终距离、失败类型。

重点学习：

- VLA 系统里的决策层和控制层解耦。
- VLM prompt 结构化输出。
- 模型错误不只看“答错”，还要看是否导致控制发散。

验收标准：

- `probe.mode: heuristic` 和 `probe.mode: api` 共用同一套控制逻辑。
- API 模式能跑通至少 20 次闭环测试。
- 有 heuristic vs VLM 的对比结果。

## 第 7 周：专家数据集规模化

目标：把当前 50 条 episode 扩展成可用于训练准备的数据集。

要做的事：

- 在当前稳定配置上采集 300-500 条 episode。
- 记录每次采集的配置，不盲目覆盖实验背景。
- 统计数据质量：
  - 成功率
  - final distance 分布
  - block position 分布
  - 每个 episode 的帧数分布
  - 图片是否存在、能否读取、尺寸是否一致
- 抽样保存代表性图片和 JSONL 片段。

重点学习：

- 数据量不是唯一目标，数据可诊断性更重要。
- 训练前的数据 schema 必须稳定。
- 采集数据时要保留实验上下文，否则结果不可复现。

验收标准：

- 至少 300 条有效 episode。
- JSONL 和图片一一对应。
- 有数据质量报告，可以写进 README 或 WORKLOG。

## 第 8 周：Action Tokenization 原型

目标：做出 VLA 项目的深度亮点，把连续动作转换成离散 token。

要做的事：

- 新增 `action_tokenizer.py`。
- 选择动作表示：
  - 方案 A：离散化末端目标位置 `[x, y, z]`
  - 方案 B：离散化 delta action `[dx, dy, dz]`
  - 方案 C：离散化 7 维关节目标角
- 先推荐方案 B，因为它更接近闭环控制里的下一步动作。
- 将连续值映射到固定 bins，例如 256 个 bin。
- 支持 `encode_action()` 和 `decode_action()`。
- 统计 decode 后的重建误差。

重点学习：

- 为什么 VLA 需要 action tokenization。
- bin 数量、动作范围和控制精度之间的权衡。
- 模型训练时预测离散 token 比直接回归连续浮点数更容易接入语言模型框架。

验收标准：

- 原始 JSONL 能转换成 tokenized dataset。
- token 能还原成近似动作。
- 有 mean / max reconstruction error。

## 第 9-10 周：轻量微调验证

目标：不追求一次训练出完美 VLA，而是验证训练数据管线和小规模模型学习是否成立。

要做的事：

- 准备微调格式：

```text
image: 当前观察图
instruction: 悬停在红色积木上方
target: action tokens 或 direction token
```

- 先用 50-100 条样本做 overfit 验证。
- 再用 train/val split 检查模型是否能泛化到不同红块位置。
- 如果本地算力不足，先完成数据转换和小样本训练脚本，不强求大规模训练。

重点学习：

- LoRA / QLoRA 的作用是低成本调整模型，而不是替代数据质量。
- 训练前要先证明数据格式和标签逻辑正确。
- 小规模 overfit 是排查训练管线的有效手段。

验收标准：

- 能生成可训练的数据格式。
- 能完成一次小规模训练或至少完成训练前验证。
- 能解释 heuristic、API VLM、微调模型三者的区别和实验价值。

## 每周 20 小时安排

建议分配：

- 8 小时：写代码和跑实验。
- 5 小时：补当前任务直接需要的知识。
- 4 小时：分析结果、整理日志、写数据报告。
- 3 小时：更新 README、WORKLOG、简历表达。

不要把时间全部投入看课。这个项目最有价值的部分是实际闭环、失败分析和可复现实验。

## 简历目标表达

完成这条路线后，可以把项目写成：

```text
构建 PyBullet 机械臂 VLA 仿真系统，完成专家轨迹采集、RGB-语言-action 数据对齐、在线闭环控制与自动化评估；设计 episode-level failure analysis，统计 success rate/final distance 等指标；接入多模态模型进行视觉决策替换，并探索 action tokenization 与 LoRA 微调数据管线。
```

当前最优先的下一步是：

```text
实现 Stage 3 自动化评估脚本，让 probe 从单次成功升级为可复现实验结果。
```
