# Agent Project Knowledge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立由 `AGENTS.md` 自动路由的项目知识层，让新 Agent 先获得准确的项目目标、架构、当前状态和文件地图，再按任务读取源码与历史证据。

**Architecture:** 把知识按变化频率分成三个权威文件：`PROJECT_OVERVIEW.md` 保存稳定心智模型，`CURRENT_STATUS.md` 保存动态工作快照，`PROJECT_STRUCTURE.md` 保存文件路由。根目录 `AGENTS.md` 只描述何时读取哪一层，不复制正文；README 只补充面向人类和 Agent 的导航链接。

**Tech Stack:** Markdown、Git、Python 3.10、`rg`、现有 `unittest` 测试套件。

## Global Constraints

- “节省 token”指减少重复扫描、重复分析和无关上下文，不通过删除必要背景追求最短文档。
- 项目简介必须同时覆盖工程架构与 VLA 学习路线。
- 不修改 `src/`、`tests/`、`sim_config.yaml`、`pyproject.toml` 或实验输出。
- 不调用付费 API，不重新运行 PyBullet 或 VLM 实验；所有结论必须来自现有代码、测试、Git 历史和项目文档。
- 保留 `BUGLOG.md`、`WORKLOG.md` 和历史 specs/plans 的完整正文，不批量移动、拆分或删除历史记录。
- 工作区当前已有用户发起的移动：已跟踪的 `docs/PROJECT_STRUCTURE.md` 被删除，等内容文件位于未跟踪的 `docs/agent/PROJECT_STRUCTURE.md`。实施时沿用该目标位置，不恢复旧路径，不覆盖用户内容。
- `2026-07-17-grounding-world-closed-loop-smoke` 设计与计划是已批准但尚未实现的历史方案；其根目录源码路径早于后续 `src/` 包迁移，不能写成当前已有能力，也不能未经修订直接执行。
- 每次提交只包含当前任务列出的文件；提交前用 `git diff --cached --name-only` 排除其他工作区变更。

---

## File Structure

| 文件 | 操作 | 单一职责 |
| --- | --- | --- |
| `AGENTS.md` | 新建 | Codex 自动读取的项目知识路由与维护规则 |
| `docs/agent/PROJECT_OVERVIEW.md` | 新建 | 稳定的项目目标、系统数据流、模块边界、关键结论和学习路线 |
| `docs/agent/CURRENT_STATUS.md` | 新建 | 当前阶段、已完成能力、未解决问题和下一步优先级 |
| `docs/agent/PROJECT_STRUCTURE.md` | 完成现有移动并修改 | 当前目录树、文件职责和新文件存放规则 |
| `README.md` | 修改 | 增加项目知识入口并把旧结构文档链接更新到 `docs/agent/` |

---

### Task 1: 固化 `docs/agent/` 文件地图与导航

**Files:**
- Move: `docs/PROJECT_STRUCTURE.md` -> `docs/agent/PROJECT_STRUCTURE.md`（使用工作区已有移动）
- Modify: `docs/agent/PROJECT_STRUCTURE.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: 当前仓库目录、`pyproject.toml` 的 10 个命令入口，以及工作区已有的结构文档移动。
- Produces: 后续 `AGENTS.md`、`PROJECT_OVERVIEW.md` 和 `CURRENT_STATUS.md` 可依赖的唯一文件路由文档。

- [ ] **Step 1: 确认移动内容没有丢失**

Run:

```bash
git show HEAD:docs/PROJECT_STRUCTURE.md > /tmp/vla-project-structure-head.md
diff -u /tmp/vla-project-structure-head.md docs/agent/PROJECT_STRUCTURE.md
```

Expected: 无输出，证明现有移动未改写正文。如果有输出，先审查并保留用户有意修改；不得用 `git checkout` 覆盖。

- [ ] **Step 2: 更新结构文档中的自身路径和目标目录树**

在 `docs/agent/PROJECT_STRUCTURE.md` 做以下精确替换：

```text
docs/PROJECT_STRUCTURE.md
-> docs/agent/PROJECT_STRUCTURE.md
```

将文档目录树改为包含：

```text
docs/
├── agent/
│   ├── PROJECT_OVERVIEW.md
│   ├── CURRENT_STATUS.md
│   └── PROJECT_STRUCTURE.md
├── worklog/WORKLOG.md
├── planning/
├── debugging/BUGLOG.md
└── superpowers/
    ├── specs/
    └── plans/
```

在“文档目录”表中增加以下职责，不把内容复制到其他章节：

```markdown
| `docs/agent/PROJECT_OVERVIEW.md` | Agent 使用的稳定项目目标、架构、关键决策和学习背景 |
| `docs/agent/CURRENT_STATUS.md` | 当前阶段、进行中工作、未解决问题和下一步 |
| `docs/agent/PROJECT_STRUCTURE.md` | 当前目录职责和未来新文件的唯一存放规则 |
```

- [ ] **Step 3: 更新 README 的目录树和文档入口**

把 README 的 `docs/` 树更新为：

```text
├── docs/
│   ├── agent/           # Agent 项目简介、当前状态和文件路由
│   ├── worklog/         # 项目推进和实验复盘
│   ├── planning/        # 学习计划及原始 PDF
│   ├── debugging/       # Bug 证据、实验和结论
│   └── superpowers/     # 设计规格与实施计划
```

在“项目文档”列表最前面增加：

```markdown
- [Agent 项目简介](docs/agent/PROJECT_OVERVIEW.md)：项目目标、架构、关键决策和学习路线。
- [当前工作状态](docs/agent/CURRENT_STATUS.md)：当前阶段、未解决问题和下一步。
- [文件路由手册](docs/agent/PROJECT_STRUCTURE.md)：目录职责和新增文件存放规则。
```

- [ ] **Step 4: 验证旧路径已清除且新导航唯一**

Run:

```bash
rg -n 'docs/PROJECT_STRUCTURE\.md' README.md docs/agent --glob '*.md'
rg -n 'docs/agent/(PROJECT_OVERVIEW|CURRENT_STATUS|PROJECT_STRUCTURE)\.md' README.md docs/agent/PROJECT_STRUCTURE.md
git diff --check
```

Expected: 第一条在当前入口文档中无输出；第二条显示 README 和结构文档中的新路径；格式检查无输出。历史 specs/plans 可以继续保留实施当时的旧路径。

- [ ] **Step 5: 提交文件地图迁移**

```bash
git add README.md docs/PROJECT_STRUCTURE.md docs/agent/PROJECT_STRUCTURE.md
git diff --cached --name-only
git commit -m "docs: group agent project guidance"
```

Expected staged paths:

```text
README.md
docs/PROJECT_STRUCTURE.md
docs/agent/PROJECT_STRUCTURE.md
```

Git 应将两条结构文档路径识别为 rename；不得暂存其他文件。

---

### Task 2: 编写稳定项目简介

**Files:**
- Create: `docs/agent/PROJECT_OVERVIEW.md`

**Interfaces:**
- Consumes: `README.md`、`sim_config.yaml`、`pyproject.toml`、`src/vla_project/`、`tests/`、`docs/debugging/BUGLOG.md`、`docs/worklog/WORKLOG.md`、学习计划和相关 specs/plans。
- Produces: Agent 默认读取的稳定项目心智模型；不保存频繁变化的任务队列。

- [ ] **Step 1: 写项目定位、边界和当前总体路线**

创建 `docs/agent/PROJECT_OVERVIEW.md`，开头使用以下内容：

```markdown
# VLA 项目简介（Agent 版）

## 项目定位

这是一个面向 VLA（Vision-Language-Action）学习的个人工程项目。项目先在 PyBullet 中建立可诊断、可复现的 KUKA iiwa 操作闭环，再逐步验证视觉语言感知、世界坐标定位、动作决策、专家数据和轻量训练，而不是一开始训练大型端到端模型。

当前基础任务是根据“悬停在红色积木上方”的语言指令，让机械臂从视觉观测和本体状态出发移动到红块上方，并保存可用于诊断、评估和后续训练的图像、动作与 episode 证据。

本项目同时服务三个目标：掌握 VLA 系统各层的实际接口；形成可复现、可测试的工程成果；沉淀可以在简历和面试中解释的实验决策与失败证据。

## 当前范围

- 已覆盖：PyBullet 仿真、KUKA IK 控制、专家轨迹采集、闭环 probe、批量评估、Qwen 离线方向基线、grounding、相机反投影和独立校准验证。
- 正在推进：把 clear 场景中验证过的 grounding 世界坐标安全接入小规模在线闭环。
- 暂不覆盖：大型 VLA 训练、真实机械臂部署、复杂多物体任务、severe 遮挡恢复和正式大规模在线 VLM 评估。
```

- [ ] **Step 2: 写系统数据流与真值边界**

加入以下架构说明：

```markdown
## 系统数据流

### 专家数据基线

`sim_config.yaml` -> PyBullet 场景与随机红块 -> IK 目标和关节控制 -> RGB/动作/本体状态 -> `outputs/dataset/` 中的轨迹与 episode 摘要。

### Stage 3 闭环基线

当前 RGB -> heuristic 或兼容多模态 API 的方向决策 -> 统一世界坐标方向执行 -> 新观测 -> stop 或失败终止 -> `outputs/probe*` 中的 trace 与汇总。

### VLM 定位路线

固定评估图片 -> Qwen 红块 grounding -> 框中心像素 -> 相机反投影到工作平面 -> 冻结 XY 补偿 -> 与末端本体位置比较 -> 确定性动作方向。

PyBullet 红块真值只能用于离线评分、场景资格检查和受控 smoke test 的初始场景搭建，不得进入正式 grounding 坐标或动作选择接口。这样可以区分视觉定位误差、几何转换误差和控制误差。
```

- [ ] **Step 3: 写模块职责和配置/测试/输出关系**

加入以下表格：

```markdown
## 模块职责

| 位置 | 职责 |
| --- | --- |
| `src/vla_project/simulation/` | PyBullet 场景、相机、机械臂控制、单次 probe 和批量 probe 评估 |
| `src/vla_project/vlm/` | VLM 样本、方向评估、grounding 诊断、反投影和校准验证 |
| `src/vla_project/tools/` | 仓库生成物迁移等维护工具 |
| `tests/` | 按领域镜像源码，并保护配置、包元数据、几何、控制和评估契约 |
| `sim_config.yaml` | 仿真、相机、任务、probe、VLM 和输出路径的统一参数来源 |
| `outputs/` | 本地生成的图片、trace、预测和实验摘要；默认不提交 Git |
| `docs/` | 当前知识、学习路线、Bug 证据以及历史设计和实施计划 |

正式源码采用 `src/` 布局，`pyproject.toml` 注册 10 个 `vla-*` 命令。新增文件和测试前查看 `PROJECT_STRUCTURE.md`，不要在仓库根目录添加正式 Python 脚本。
```

- [ ] **Step 4: 写已验证结论与关键设计决定**

必须准确记录以下事实，并在每组结论后链接 `BUGLOG.md` 或 `WORKLOG.md`：

```markdown
## 已验证结论与设计决定

1. Stage 3 heuristic 在固定 seeds 42–91 上为 50/50 成功；修复重点是 IK 冗余解使用关节限位和当前姿态，不是继续盲调方向步长。
2. Qwen 448px 直接方向基线为 3/4，但 grounding 框中心关系为 4/4；back 样本证明模型能给出正确空间表示却输出错误方向。因此 VLM 负责 grounding，代码负责确定性坐标或主轴比较，不再通过同义 prompt 反复修补方向标签。
3. seeds 42–46 的 20 张 grounding 均返回合法框，但整体 XY 误差 mean/median/max 为 3.94/3.77/10.06cm。合法框和方向正确不等于厘米级定位通过，遮挡与固定系统偏差必须分开处理。
4. 使用 seeds 42–46 clear 样本拟合并冻结 `(+2.492cm, -1.947cm)` 补偿后，独立 seeds 47–51 的 15 个 clear 样本补偿误差 mean/median/max 为 0.77/0.79/1.46cm，15/15 不超过 3cm。唯一 severe 样本仍为 3.27cm，因此结论只适用于固定相机 clear 场景。
5. 在线 grounding 闭环 smoke test 已有 2026-07-17 设计和计划，但当前仓库没有对应运行模块或真实运行证据；后续实施前必须把旧计划的根目录脚本路径适配到现有 `src/vla_project/` 包结构。
```

证据链接使用仓库相对路径：

```markdown
- [Bug 与验证证据](../debugging/BUGLOG.md)
- [工程推进与学习复盘](../worklog/WORKLOG.md)
- [Grounding 闭环 smoke 设计](../superpowers/specs/2026-07-17-grounding-world-closed-loop-smoke-design.md)
- [Grounding 闭环 smoke 历史计划](../superpowers/plans/2026-07-17-grounding-world-closed-loop-smoke.md)
```

- [ ] **Step 5: 写评价方法、学习路线与入口**

加入以下内容：

```markdown
## 评价原则

- 每个阶段先定义固定 seeds、成功阈值、错误分类和证据输出，再运行实验。
- 方向准确率、grounding 合法率、世界坐标误差和闭环成功率是不同指标，不能互相替代。
- 校准集与验证集必须隔离；验证集不能反向拟合参数。
- clear、partial、severe 可见率分组分别汇总，不用整体均值隐藏遮挡失败。
- 安全中止证明保护机制有效，但仍计为任务失败。
- 源码与自动测试保护计算契约；本地输出目录保存真实实验结果，两者都不能由文档声明替代。

## 学习路线

当前路线是：可诊断专家数据 -> 稳定 heuristic 闭环 -> VLM 能力边界 -> grounding 与世界坐标融合 -> clear 场景在线闭环 -> 专家数据规模化 -> action tokenization -> 轻量微调验证。是否进入下一阶段由当前阶段证据决定，不因计划日期自动推进。

## 关键入口

- 当前状态：`CURRENT_STATUS.md`
- 文件路由：`PROJECT_STRUCTURE.md`
- 使用说明：`../../README.md`
- 核心配置：`../../sim_config.yaml`
- 学习计划：`../planning/vla_robotic_study_plan.md`
- Bug 证据：`../debugging/BUGLOG.md`
- 工作日志：`../worklog/WORKLOG.md`
```

- [ ] **Step 6: 校对简介中的事实与路径**

Run:

```bash
rg -n '50/50|3/4|4/4|3\.94|0\.77|1\.46|3\.27|src/vla_project|CURRENT_STATUS|PROJECT_STRUCTURE' docs/agent/PROJECT_OVERVIEW.md
test -f docs/debugging/BUGLOG.md
test -f docs/worklog/WORKLOG.md
test -f docs/superpowers/specs/2026-07-17-grounding-world-closed-loop-smoke-design.md
test -f docs/superpowers/plans/2026-07-17-grounding-world-closed-loop-smoke.md
git diff --check
```

Expected: 关键事实均出现，四个证据文件存在，格式检查无输出。

- [ ] **Step 7: 提交稳定项目简介**

```bash
git add docs/agent/PROJECT_OVERVIEW.md
git diff --cached --name-only
git commit -m "docs: add agent project overview"
```

Expected: 只提交 `docs/agent/PROJECT_OVERVIEW.md`。

---

### Task 3: 编写当前状态并建立自动入口

**Files:**
- Create: `docs/agent/CURRENT_STATUS.md`
- Create: `AGENTS.md`

**Interfaces:**
- Consumes: Task 1 的文件地图、Task 2 的稳定简介和当前 Git/测试事实。
- Produces: 新 Codex 会话自动获得的阅读顺序，以及可整体改写的当前工作快照。

- [ ] **Step 1: 写当前状态快照**

创建 `docs/agent/CURRENT_STATUS.md`，内容如下：

```markdown
# 当前项目状态

**最后核对日期：** 2026-07-19

## 当前阶段

项目已完成 Python `src/` 包结构、统一命令入口和生成输出整理。VLM 主线已完成固定相机 clear 场景的 grounding 世界坐标独立校准验证，下一工程阶段是把该能力接入小规模、可中止的在线闭环 smoke test。

## 已完成且仍有效

- Baseline 专家数据采集和诊断输出可用。
- Stage 3 heuristic 固定 seeds 42–91 为 50/50 成功。
- 直接方向预测的能力边界已确定：448px 为 3/4，grounding 中间表示为 4/4。
- 相机反投影几何、可见率分组和离线真值隔离已建立。
- 冻结 XY 补偿在独立 clear 验证集上为 15/15 不超过 3cm。
- 正式源码已迁移到 `src/vla_project/`，并通过 `pyproject.toml` 暴露 10 个命令入口。

## 未解决问题

1. 冻结补偿后的 grounding 坐标尚无在线机械臂闭环证据。
2. severe 遮挡仍未解决；现有 clear 结论不能外推。
3. 2026-07-17 smoke 实施计划早于 `src/` 包迁移，文件路径和命令需要先按当前结构修订。
4. README、学习计划、BUGLOG 和 WORKLOG 的阶段表述可能存在时间差；实验结论以原始摘要和对应证据链为准。

## 下一步优先级

1. 按当前包结构修订 grounding 世界坐标闭环 smoke 实施计划，不改变已批准的真值隔离、安全中止、3 个 clear cases 和最多 30 次 API 请求约束。
2. 实现并用 mock 测试纯 targeting、安全状态机和闭环编排，确认测试不会调用真实 API。
3. 经用户明确批准付费实验后，运行一次 seeds 52–54 的真实 smoke，并根据唯一主导失败类型决定后续工作。

## 当前任务入口

- 项目简介：`PROJECT_OVERVIEW.md`
- 文件路由：`PROJECT_STRUCTURE.md`
- Smoke 设计：`../superpowers/specs/2026-07-17-grounding-world-closed-loop-smoke-design.md`
- 旧实施计划：`../superpowers/plans/2026-07-17-grounding-world-closed-loop-smoke.md`
- 校准实现：`../../src/vla_project/vlm/validate_grounding_calibration.py`
- 相机几何：`../../src/vla_project/simulation/camera_geometry.py`
- 单次闭环基线：`../../src/vla_project/simulation/stage3_probe.py`

## 更新规则

阶段、进行中工作、未解决问题或下一步改变时，直接改写本文件中的失效内容。详细实验过程写入 WORKLOG 或 BUGLOG，不在这里无限追加。
```

- [ ] **Step 2: 用仓库事实校验当前状态**

Run:

```bash
test ! -e src/vla_project/vlm/grounding_targeting.py
test ! -e src/vla_project/vlm/run_grounding_smoke.py
test -e src/vla_project/vlm/validate_grounding_calibration.py
test -e src/vla_project/simulation/camera_geometry.py
test -e src/vla_project/simulation/stage3_probe.py
git log --oneline --all -- src/vla_project | head -20
```

Expected: smoke 模块不存在，三个现有入口存在；Git 历史显示最近的 `src/` 包迁移提交。若 smoke 模块已由其他工作创建，停止并基于真实实现改写状态，不能继续声称“尚未实现”。

- [ ] **Step 3: 创建根目录 `AGENTS.md`**

写入以下完整内容：

```markdown
# VLA Project Agent Guidance

## Start Here

1. 开始项目任务前先读 `docs/agent/PROJECT_OVERVIEW.md`，不要先扫描整个仓库。
2. 任务涉及当前进展、实验延续或下一步决策时，再读 `docs/agent/CURRENT_STATUS.md`。
3. 定位、创建或移动文件前，读 `docs/agent/PROJECT_STRUCTURE.md`。
4. 只按当前任务需要读取源码、测试、`WORKLOG.md`、`BUGLOG.md` 和历史 specs/plans；不要默认全文加载全部历史文档。

## Source of Truth

- 源码和测试决定当前行为；本地实验摘要决定真实指标。
- `PROJECT_OVERVIEW.md` 保存稳定架构与已验证结论。
- `CURRENT_STATUS.md` 保存可被整体改写的动态状态。
- `PROJECT_STRUCTURE.md` 是新增文件位置的权威规则。
- 历史 specs/plans 记录当时设计，不自动代表已经实现；先用当前文件和 Git 历史核对。

## Keep Knowledge Current

- 项目目标、架构、数据流或关键结论改变时更新 `PROJECT_OVERVIEW.md`。
- 当前阶段、未解决问题或下一步改变时更新 `CURRENT_STATUS.md`。
- 目录或文件职责改变时更新 `PROJECT_STRUCTURE.md`。
- 详细实验过程写入 `WORKLOG.md`；Bug 证据、根因和验证写入 `BUGLOG.md`。
- 同一事实只在一个权威位置保存完整正文，其他位置使用摘要和链接。
```

- [ ] **Step 4: 验证自动入口引用与状态边界**

Run:

```bash
for path in docs/agent/PROJECT_OVERVIEW.md docs/agent/CURRENT_STATUS.md docs/agent/PROJECT_STRUCTURE.md; do test -f "$path"; done
rg -n 'PROJECT_OVERVIEW|CURRENT_STATUS|PROJECT_STRUCTURE|不要默认全文' AGENTS.md
rg -n '尚无在线|severe|早于.*src|下一步优先级' docs/agent/CURRENT_STATUS.md
rg -n 'T[B]D|T[O]DO|待.定|占.位' AGENTS.md docs/agent/CURRENT_STATUS.md
git diff --check
```

Expected: 三个目标文件存在；入口包含四项路由规则；状态明确在线、遮挡和历史计划边界；占位符扫描和格式检查无输出。

- [ ] **Step 5: 提交动态状态和自动入口**

```bash
git add AGENTS.md docs/agent/CURRENT_STATUS.md
git diff --cached --name-only
git commit -m "docs: add agent project entrypoint"
```

Expected staged paths:

```text
AGENTS.md
docs/agent/CURRENT_STATUS.md
```

---

### Task 4: 交叉验证知识层并做冷启动演练

**Files:**
- Modify only if validation finds a factual error: `AGENTS.md`, `README.md`, `docs/agent/PROJECT_OVERVIEW.md`, `docs/agent/CURRENT_STATUS.md`, `docs/agent/PROJECT_STRUCTURE.md`

**Interfaces:**
- Consumes: Tasks 1–3 的完整知识层。
- Produces: 路径有效、职责不重叠、事实与仓库一致的最终文档，以及一次可复述的 Agent 冷启动结果。

- [ ] **Step 1: 检查路径、职责和历史边界**

Run:

```bash
rg -n 'docs/PROJECT_STRUCTURE\.md' AGENTS.md README.md docs/agent --glob '*.md'
rg -n 'PROJECT_OVERVIEW|CURRENT_STATUS|PROJECT_STRUCTURE' AGENTS.md README.md docs/agent --glob '*.md'
rg -n '已实现|已完成|尚未|历史计划' docs/agent/PROJECT_OVERVIEW.md docs/agent/CURRENT_STATUS.md
git diff --check
```

Expected: 当前入口文档中无旧结构路径；三层文档相互链接；smoke 明确是“有设计/计划但尚未实现”；格式检查无输出。历史 specs/plans 不在路径清理范围内。

- [ ] **Step 2: 检查文档事实所依赖的代码入口**

Run:

```bash
find src/vla_project -maxdepth 2 -type f -name '*.py' | sort
find tests -maxdepth 2 -type f -name 'test_*.py' | sort
rg -n '^vla-[a-z-]+\s*=' pyproject.toml
rg -n '^## (项目定位|当前范围|系统数据流|模块职责|已验证结论与设计决定|评价原则|学习路线|关键入口)$' docs/agent/PROJECT_OVERVIEW.md
```

Expected: 源码和测试领域与结构文档一致；`pyproject.toml` 显示 10 个入口；简介包含全部八个核心章节。

- [ ] **Step 3: 运行无付费依赖的完整测试**

```bash
conda run -n vla_env python -m unittest discover -s tests -v
```

Expected: 当前完整测试全部 PASS，且没有真实 API 请求。现有工作日志最后记录的包迁移基线为 `109/109`；若数量变化，以当前收集到的测试数为准，但任何失败都必须先解释。

- [ ] **Step 4: 执行冷启动复述检查**

启动一个新的 Codex 会话或等价只读检查，让它仅依据 `AGENTS.md` 路由读取项目知识，并回答：

```text
1. 项目解决什么问题？
2. 当前主要模块和数据流是什么？
3. 已验证的 grounding 结论是什么？
4. 现在尚未解决的首要问题是什么？
5. 如果要实现下一步，应先读哪些文件？
```

Expected answer must include:

```text
PyBullet KUKA 悬停/VLA 学习；simulation 与 vlm 分层；
clear 独立验证 15/15 <=3cm；severe 未解决；在线 smoke 尚无实现证据；
先读 CURRENT_STATUS、smoke 设计/计划、camera_geometry、stage3_probe 和校准模块。
```

若复述需要扫描整个仓库才能回答，说明简介缺少关键上下文；只补充缺失事实，不复制完整日志。

- [ ] **Step 5: 提交验证阶段发现的最小修正**

如果没有文档修正，跳过本步骤。若有修正：

```bash
git add AGENTS.md README.md docs/agent/PROJECT_OVERVIEW.md docs/agent/CURRENT_STATUS.md docs/agent/PROJECT_STRUCTURE.md
git diff --cached --name-only
git diff --cached --check
git commit -m "docs: verify agent project knowledge"
```

Expected: 只包含验证实际修改过的项目知识文件，不为空提交。

---

## 完成判据

- 新 Codex 会话会从根目录 `AGENTS.md` 获得明确的按需阅读顺序。
- `PROJECT_OVERVIEW.md` 同时覆盖工程架构、VLA 学习目标、关键结论、评价原则和证据入口。
- `CURRENT_STATUS.md` 清楚区分已完成能力、尚未解决问题和优先下一步，并能被整体改写而不是无限追加。
- `PROJECT_STRUCTURE.md` 位于 `docs/agent/`，内部及 README 不再引用旧路径。
- 历史 smoke 设计/计划没有被误写成当前实现，且指出执行前必须适配 `src/` 包结构。
- `BUGLOG.md`、`WORKLOG.md`、学习计划和历史 specs/plans 的正文未被批量迁移或删除。
- 路径检查、占位符扫描、`git diff --check` 和完整自动测试通过。
- 冷启动复述能在不扫描整个仓库的情况下准确回答项目目标、架构、已验证结论、当前阻塞和相关入口。
