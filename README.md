# VLA Project: PyBullet 机械臂视觉-语言-动作学习

面向桌面抓取与悬停任务的 VLA（Vision-Language-Action）学习与闭环控制系统。
在 PyBullet 仿真中构建了从视觉感知到动作执行的完整管线，覆盖三条技术路线：
**行为克隆（BC）**、**端到端 VLA** 和 **模块化 VLM Grounding**。

## 核心结果

| 路线 | 关键指标 | 说明 |
|------|---------|------|
| BC 行为克隆 | **92%** rollout 成功率（46/50） | ResNet-18 + 连续回归，对比 regression / absolute_q_32 / delta_q_64 |
| VLA 端到端 | **74%** 成对指令跟随（37/50） | ResNet-18 + 多语言 MiniLM（~129M），成对数据逼模型用语言 |
| VLM 模块化 | **3/3** 任务到达，**19/20** 定位 ≤3cm | Qwen3-VL-Flash + 相机反投影 + 冻结 XY 补偿 + IK 控制器 |

详细结果与审计链见下方各节。

## 架构

```
src/vla_project/
├── simulation/    # PyBullet 场景、KUKA iiwa IK/FK 控制、专家数据采集与审计
├── training/      # BC 模型与训练、VLA 模型与训练、rollout 评估、反事实实验
└── vlm/           # VLM 样本采集、grounding 诊断、反投影校准、在线闭环 smoke
tests/             # 362 项自动化测试，与源码目录结构镜像
```

- **仿真引擎**：PyBullet，KUKA iiwa 7-DoF 机械臂
- **视觉骨干**：ResNet-18（ImageNet 预训练）
- **文本编码**：`paraphrase-multilingual-MiniLM-L12-v2`
- **VLM 感知**：Qwen3-VL-Flash（API）+ Grounding DINO 框 → 相机反投影 → 世界坐标
- **动作表示**：regression（连续）/ absolute_q / delta_q（离散分箱）
- **参数管理**：单一 `sim_config.yaml` 管理全链路

## 环境准备

Python 3.10，PyBullet + PyTorch：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` 以可编辑模式安装，注册所有 `vla-*` 命令。

## CLI 命令

| 命令 | 用途 |
| --- | --- |
| `vla-collect` | 采集版本化专家训练数据 |
| `vla-evaluate-dataset` | 扫描专家数据并生成质量报告 |
| `vla-audit-action-tokenization` | 审计动作分布与分箱候选 |
| `vla-audit-dataset-visibility` | 确定性重放并审计逐帧可见率 |
| `vla-render-expert-dataset-view` | 从冻结轨迹生成固定视角派生集 |
| `vla-split-episodes` | 按目标颜色和位姿分层划分 train/val |
| `vla-probe` | 运行 Stage 3 单次闭环 |
| `vla-evaluate-probe` | 批量评估 heuristic 闭环 |
| `vla-collect-vlm-samples` | 生成 VLM 离线评估样本 |
| `vla-evaluate-vlm-decisions` | 评估 VLM 方向决策 |
| `vla-diagnose-grounding` | 诊断 grounding 框与真值关系 |
| `vla-ground-then-decide` | 评估 grounding 后确定性方向判断 |
| `vla-evaluate-backprojection` | 评估框中心反投影到世界坐标 |
| `vla-validate-grounding-calibration` | 验证冻结校准参数 |
| `vla-run-grounding-smoke` | 运行真实 grounding 在线闭环 smoke |
| `vla-screen-grounding-smoke` | 无 API 预筛选 smoke 候选 |
| `vla-train-bc` | 训练单任务 BC 动作表示基线 |
| `vla-rollout` | 评估 BC checkpoint 闭环表现 |
| `vla-train` | 训练多任务 VLA |
| `vla-evaluate-vla-counterfactual` | 确定性成对反事实评估 |
| `vla-convert-qwen` | 转换数据为 Qwen2-VL 微调格式 |

## 专家数据采集

参数由 `sim_config.yaml` 统一管理。当前双积木 v2 数据（`expert_multi_v2`）：

```bash
# Pilot：10 条，验证门禁
vla-collect --num-episodes 10
vla-evaluate-dataset --dataset-dir outputs/dataset/expert_multi_v2

# 门禁通过后追加 290 条
vla-collect --num-episodes 290
vla-evaluate-dataset --dataset-dir outputs/dataset/expert_multi_v2
```

输出目录结构：

```
outputs/dataset/expert_multi_v2/
├── dataset_manifest.json       # schema、seed、尺寸等契约
├── config_snapshot.yaml        # 采集配置快照
├── trajectory_expert.jsonl     # 逐帧训练样本
├── episode_summary.jsonl       # 逐 episode 诊断摘要
├── episode_split.json          # train/val 划分
└── ep_*_step_*.jpg             # RGB 观测图
```

## 数据格式

`trajectory_expert.jsonl` 每行一个样本，核心字段：

| 字段 | 说明 |
|------|------|
| `image_path` | 当前帧图像路径 |
| `instruction` | 中文语言指令 |
| `action` | 7 维关节目标 + gripper + terminate |
| `camera_eye` | 相机位置 |
| `block_pos` | 积木世界坐标 |
| `ee_pos` | 末端位置 |
| `distance_to_target` | 末端到目标距离 |
| `termination_reason` | episode 终止原因 |

## BC 行为克隆

ResNet-18 骨干 + 可切换动作头，三组对照：

| 动作表示 | Rollout 成功率 | 结论 |
|---------|---------------|------|
| regression（连续） | **92%** (46/50) | 主基线 |
| absolute_q_32（32 箱分类） | 66% (33/50) | 可行 |
| delta_q_64（64 箱分类） | 0% (0/50) | 当前协议下失效 |

训练和评估：

```bash
vla-train-bc --action-representation regression
vla-rollout --checkpoint outputs/training/bc_regression_full_v1/checkpoint_best.pt
```

## VLA 端到端

ResNet-18 + 多语言 MiniLM 后融合，输入 RGB + 中文指令，输出 7 维关节目标（~129M 参数）。

**成对反事实评估**：对每个验证 episode 恢复相同 seed、积木位姿和机器人状态（重放误差 = 0），分别用红/蓝指令执行。

- 早期（`all-MiniLM-L6-v2` 英文模型）：指令跟随 0/50、偏好切换 0/50、红/蓝单分支 11/50 和 12/50。
- 修复后（multilingual MiniLM + 成对数据）：**指令跟随 37/50（74%）、偏好切换 50/50（100%）、红 82% / 蓝 90%**，`supports_red_blue_instruction_recognition = true`。

**根因定位与修复**（两层问题，分别修）：

1. **编码层**：旧 `all-MiniLM-L6-v2` 是纯英文模型，将"红色"/"蓝色"都映射为 `[UNK]`（token IDs 相同，embedding L2 = 0）。→ 换成 `paraphrase-multilingual-MiniLM-L12-v2` + 训练前区分度门禁。
2. **数据层**：红/蓝指令出现在不同随机场景，模型可只靠视觉布局得分、语言成噪声。→ 用成对数据（`expert_paired_v1`：同场景反色指令 600 条）堵死捷径，逼模型用语言。

后续规划：分拣环境（expert_sort_v1）与 Qwen2-VL QLoRA。

```bash
vla-train --dataset-dir outputs/dataset/expert_multi_v2
vla-evaluate-vla-counterfactual --checkpoint outputs/training/vla_regression_full_v3/checkpoint_best.pt
```

## VLM 模块化闭环

Qwen3-VL-Flash 做目标 grounding → 相机反投影到工作平面 → 冻结 XY 补偿 → IK 控制器执行。

**离线校准与验证**：
- 标定集：15 个 clear 样本拟合补偿 `(+2.49cm, -1.95cm)`
- 验证集：20 个独立样本（clear/partial/severe = 15/4/1）
- 校正后：19/20 ≤ 3cm（95%），clear 15/15 通过，severe 1/1 超出

**在线 smoke**（3 个固定场景，经无 API 预筛选）：
- 任务到达：3/3
- 末端定位精度：0.30–0.79 cm
- VLM API 调用：3 episode 合计 4 次（目标保持机制）
- 自主停止：1/3

```bash
vla-screen-grounding-smoke    # 无 API 资格预筛选
vla-run-grounding-smoke       # 需 API Key 的真实闭环
```

## 工程规范

- **362 项自动化测试**，与源码目录结构镜像
- **单一 YAML 配置**：`sim_config.yaml` 管理仿真、相机、任务、训练全链路参数
- **数据完整性**：manifest / 轨迹 / 划分 / checkpoint 经 SHA-256 绑定
- **审计链**：质量门禁 → 可见性分层 → 动作量化 → episode 划分 → 训练 → 反事实
- **标定-验证隔离**：校准集与验证集使用不同 seeds，验证阶段不重新拟合

## 配置优先

可调参数集中在 `sim_config.yaml`：

- 连接模式（GUI / DIRECT）
- 相机位置、FOV、图像尺寸
- 积木随机范围
- 成功距离阈值、卡住检测窗口
- 机械臂速度、力控参数
- 训练 epoch、batch size、学习率

只有当流程逻辑变化时才修改 `src/vla_project/` 中的模块。
