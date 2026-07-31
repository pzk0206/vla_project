# 项目文件结构与新文件存放规则

> 这是本仓库的文件路由手册。开始新增代码、测试、文档或生成图片前，先查本文件。

## Codex 必须先遵守的规则

1. 正式 Python 代码只能放入 `src/vla_project/` 的对应领域包。
2. 测试必须放入 `tests/` 的镜像领域目录。
3. 所有运行生成的图片和结果只能放入 `outputs/` 的对应分类。
4. 仓库根目录不新增正式 `.py` 文件或生成图片。
5. `.worktrees/` 只用于隔离开发，完成后通过 Git 合并，不能作为正式归档目录。
6. 如果现有分类无法容纳新内容，先询问用户，不自行创建新的顶层目录。

## 新文件应该放在哪里

这是最重要的路由表。新增文件前，先按其主要职责选择唯一位置。

| 新内容 | 放置位置 | 配套位置或说明 |
| --- | --- | --- |
| PyBullet、相机、机械臂控制代码 | `src/vla_project/simulation/` | 测试放 `tests/simulation/` |
| VLM、grounding、反投影、校准代码 | `src/vla_project/vlm/` | 测试放 `tests/vlm/` |
| 仓库维护和迁移工具 | `src/vla_project/tools/` | 测试放 `tests/tools/` |
| 仿真、相机、任务和输出路径参数 | `sim_config.yaml` | 契约测试放 `tests/test_config_contract.py` |
| Python 包信息和 `vla-*` 命令入口 | `pyproject.toml` | 契约测试放 `tests/test_package_metadata.py` |
| 第三方 Python 依赖 | `requirements.txt` | 修改后重新安装并运行全量测试 |
| 安装方法、运行命令和阶段概览 | `README.md` | 不在这里记录详细实验过程 |
| 文件职责和新文件路由规则 | `docs/agent/PROJECT_STRUCTURE.md` | 目录发生变化时同步更新 |
| 当前工程进展和实验复盘 | `docs/worklog/WORKLOG.md` | 记录过程、指标和阶段结论 |
| 后续路线和验收计划 | `docs/planning/` | 可编辑计划使用 Markdown |
| Bug 证据、根因和修复结论 | `docs/debugging/BUGLOG.md` | 保留失败到修复的完整证据链 |
| 新功能或重构的设计规格 | `docs/superpowers/specs/` | 设计确认后再实施 |
| 已确认设计的实施步骤 | `docs/superpowers/plans/` | 使用日期和主题命名 |
| Baseline 训练图片和轨迹 | `outputs/dataset/` | 图片、JSONL 和摘要放在同类目录 |
| Stage 3 单次闭环图片和 trace | `outputs/probe/` | 只放单次 probe 输出 |
| Stage 3 批量评估图片和统计 | `outputs/probe_evaluations/<批次>/` | 每次运行建立独立批次目录 |
| VLM 固定输入图片 | `outputs/vlm_samples/<实验名>/` | 同目录保存样本清单和诊断信息 |
| VLM 预测、标注图和评估结果 | `outputs/vlm_evaluations/<实验名>/` | 同目录保存结果和摘要 |
| 项目专用 Codex Skill | `.agents/skills/<skill-name>/` | 每个 Skill 必须包含 `SKILL.md` |
| 项目级 Codex 配置 | `.codex/config.toml` | 仅确有项目级设置时创建，不保存密钥 |

无法直接归类时，先按文件的主要职责选择最接近的领域。若需要改变包边界或新增顶层
分类，必须先询问用户，不能为了方便临时在根目录创建文件夹。

## 新任务的文件放置决策

创建正式代码文件前，必须按以下顺序判断。判断依据是职责和依赖关系，不是文件名相似、
日期接近或当前对话方便。

```text
明确新内容的主要职责
  -> 查找职责相同的现有文件
      -> 能保持单一职责：修改现有文件
      -> 无法保持单一职责：继续判断
  -> 查找同一功能的现有子文件夹
      -> 存在：放入该子文件夹
      -> 不存在：继续判断
  -> 选择最接近的领域目录
      -> simulation / vlm / tools
  -> 是否形成独立、可扩展的多文件工作流？
      -> 是：建立功能名子文件夹
      -> 否：在领域目录新增单个模块
  -> 仍无法明确归类：询问用户
```

查找位置时按以下优先级执行：

1. 相同功能的现有子文件夹；
2. 相同职责的现有代码文件；
3. 对应领域根目录；
4. 经确认的共同上层；
5. 询问用户。

能扩展现有文件且不破坏单一职责时，默认修改现有文件，不为每个新任务创建文件或目录。
多个领域都会使用的能力不能复制到各任务目录；只有职责稳定且确实与具体领域无关时，才
考虑放到共同上层。不能用模糊的 `common.py` 或 `utils.py` 收纳暂时找不到位置的代码。

## 什么时候新建功能子文件夹

只有同时满足以下条件时，才新建功能子文件夹：

1. 新任务预计包含至少两个紧密关联的正式代码文件；
2. 这些文件共同构成一个可独立理解和测试的功能或工作流；
3. 放在领域根目录会导致职责或文件列表明显混乱；
4. 子文件夹可以使用长期稳定的功能名称。

有独立 CLI、独立输入输出、独立错误边界或明确后续扩展计划，可以进一步支持建立子文件夹，
但任何单项都不能单独决定建目录。满足“两个文件”也不是充分条件；如果两个文件分别属于
已有模块职责，仍应放回对应的现有位置。

以下情况默认不新建子文件夹：

- 只增加一个小型模块、通用函数或适配器；
- 只是修复、扩展或测试现有行为；
- 只新增一份测试、文档或配置；
- 新内容可以放入现有目录而不造成职责混杂；
- 任务名称只是临时实验名、日期、提交名或对话描述。

功能子文件夹放在最接近的领域之下，并使用小写蛇形功能名，例如
`src/vla_project/vlm/grounding_smoke/`。不要使用 `stage4/`、`new_task/`、`temp/`、
`test_version/` 或日期作为代码目录名；阶段顺序写入当前状态、工作日志、设计或计划文档。

新增源码子文件夹时，测试目录必须保持同样层级：

```text
src/vla_project/vlm/grounding_smoke/targeting.py
-> tests/vlm/grounding_smoke/test_targeting.py
```

如果正式代码放在已有领域目录，测试也放在对应的现有测试领域目录，不为了测试单独创建
任务层级。

示例：新增 grounding 框解析逻辑时，优先扩展已有
`src/vla_project/vlm/diagnose_vlm_grounding.py`；新增相机投影数学函数时，优先扩展
`src/vla_project/simulation/camera_geometry.py`。只有一个工作流同时包含 targeting、
screening、runner 等多个协作组件时，才适合建立 `vlm/grounding_smoke/` 功能子包。

## 当前目录树

下面只展示有助于理解职责的层级，不展开缓存、批次和大量图片。

```text
vla_project/
├── src/vla_project/                 # 正式 Python 包
│   ├── simulation/                  # 仿真、相机、采集和闭环控制
│   ├── vlm/                         # VLM 样本、grounding、反投影和校准
│   └── tools/                       # 仓库维护工具
├── tests/                           # 与正式源码镜像对应的自动化测试
│   ├── simulation/
│   ├── vlm/
│   ├── tools/
│   ├── test_config_contract.py
│   └── test_package_metadata.py
├── docs/                            # 项目说明、过程记录和设计文档
│   ├── agent/                       # Agent 项目知识和文件路由
│   │   ├── PROJECT_OVERVIEW.md
│   │   ├── CURRENT_STATUS.md
│   │   └── PROJECT_STRUCTURE.md     # 本文件：结构和新文件路由规则
│   ├── worklog/WORKLOG.md
│   ├── planning/
│   ├── debugging/BUGLOG.md
│   └── superpowers/
│       ├── specs/
│       └── plans/
├── outputs/                         # 运行生成内容，不属于正式源码
│   ├── dataset/
│   ├── probe/
│   ├── probe_evaluations/
│   ├── vlm_samples/
│   └── vlm_evaluations/
├── .agents/                         # 可选：项目专用 Codex Skills
├── .codex/                          # 可选：项目级 Codex 配置
├── .vscode/                         # 本机 VS Code 设置
├── .worktrees/                      # Git 隔离开发工作区
├── __pycache__/                     # Python 自动生成缓存
├── .git/                            # Git 内部数据库
├── .gitignore                       # Git 忽略规则
├── pyproject.toml                   # Python 包和命令入口
├── README.md                        # 项目入口说明
├── requirements.txt                 # 第三方依赖
└── sim_config.yaml                  # 仿真和实验参数
```

## 根目录文件

| 文件 | 作用 | 什么时候修改 |
| --- | --- | --- |
| `.gitignore` | 指定不提交的缓存、生成输出和本机配置 | 新增可再生成或仅本机使用的文件类型时 |
| `pyproject.toml` | 定义 `vla-project` 包、`src/` 布局和 16 个 `vla-*` 命令 | 增加包元数据或命令入口时 |
| `README.md` | 给使用者说明项目目标、安装方法、运行命令和当前阶段 | 使用方式或阶段结论变化时 |
| `requirements.txt` | 记录 PyBullet、OpenCV、NumPy 等第三方依赖 | 正式代码新增或移除外部依赖时 |
| `sim_config.yaml` | 保存仿真、相机、采集、probe、VLM 和输出目录参数 | 调整实验变量时，优先改这里而非写死在代码中 |

## 正式源码 `src/vla_project/`

### `simulation/`：仿真与控制

| 文件 | 职责 |
| --- | --- |
| `camera_geometry.py` | PyBullet/OpenGL 相机投影、像素坐标和工作平面反投影几何 |
| `control_arm.py` | 创建 PyBullet 场景、独立复位 KUKA，并采集版本化专家图片与轨迹 |
| `stage3_probe.py` | 运行单个“观察 -> 决策 -> 控制 -> 再观察”闭环 episode |
| `evaluate_probe.py` | 用固定种子批量调用 Stage 3 probe 并汇总成功率和失败证据 |
| `evaluate_dataset.py` | 只读扫描专家数据完整性、生成质量报告并执行严格 pilot 门禁 |
| `audit_dataset_visibility.py` | 确定性重放专家帧，验证 JPEG 一致性并生成红块可见率审计 |
| `audit_action_tokenization.py` | 对齐专家动作与可见性，模拟分箱并生成动作表示审计 |
| `expert_dataset_replay.py` | 统一加载、校验并确定性重放冻结的专家 episode |
| `render_expert_dataset_view.py` | 保留源数据并原子生成固定相机视觉派生数据集 |

### `vlm/`：视觉语言模型评估

| 文件 | 职责 |
| --- | --- |
| `collect_vlm_eval_samples.py` | 从固定 heuristic trace 生成和校验离线 VLM 输入样本 |
| `evaluate_vlm_decisions.py` | 比较 Qwen 方向预测与固定标准方向，支持断点续跑 |
| `diagnose_vlm_grounding.py` | 让 VLM 标出末端和红块，用标注图诊断方向误判原因 |
| `evaluate_ground_then_decide.py` | 检查 VLM 输出的目标框与最终方向是否自洽 |
| `evaluate_grounding_backprojection.py` | 把红块框中心反投影到工作平面并离线计算定位误差 |
| `validate_grounding_calibration.py` | 在独立数据上验证冻结的 grounding XY 偏差补偿 |

#### `vlm/grounding_smoke/`：在线 grounding 冒烟工作流

| 文件 | 职责 |
| --- | --- |
| `targeting.py` | 冻结校准、框反投影、补偿、工作区校验和真值隔离动作策略 |
| `runner.py` | 可注入闭环、安全中止、PyBullet/Qwen 适配、trace、summary 和正式 smoke CLI |
| `screening.py` | 不调用 VLM 的动态候选轨迹筛选和遮挡证据汇总 |

### `tools/`：仓库维护

| 文件 | 职责 |
| --- | --- |
| `migrate_generated_outputs.py` | 将历史生成内容安全迁移到统一 `outputs/` 分类，并校验路径引用 |

`__init__.py` 用来声明 Python 包。除非需要公开包级接口，否则保持简短即可。

## 测试目录 `tests/`

测试目录按领域镜像正式源码：

```text
src/vla_project/simulation/x.py  -> tests/simulation/test_x.py
src/vla_project/vlm/x.py         -> tests/vlm/test_x.py
src/vla_project/tools/x.py       -> tests/tools/test_x.py
```

- `tests/test_config_contract.py`：保护 `sim_config.yaml` 的跨模块配置约束和统一输出路径。
- `tests/test_package_metadata.py`：保护 `src/` 包结构和 16 个控制台命令入口。
- `tests/simulation/test_evaluate_dataset.py`：保护专家数据质量扫描和 pilot 门禁。
- `tests/simulation/test_audit_dataset_visibility.py`：保护确定性重放、可见率和原子输出。
- `tests/simulation/test_audit_action_tokenization.py`：保护动作/可见性对齐、统计、分箱推荐和原子输出。
- `tests/simulation/test_expert_dataset_replay.py`：保护共享重放输入、episode 契约和帧次序。
- `tests/simulation/test_render_expert_dataset_view.py`：保护派生 schema、路径安全、标签等价、
  独立双阶段重放和原子发布。
- `tests/vlm/grounding_smoke/`：镜像测试 targeting、runner 和 screening。
- 新增或修复行为时，应同时新增对应领域测试；不要把测试文件放进 `src/`。

## 文档目录 `docs/`

| 位置 | 内容性质 |
| --- | --- |
| `docs/agent/PROJECT_OVERVIEW.md` | Agent 使用的稳定项目目标、架构、关键决策和学习背景 |
| `docs/agent/CURRENT_STATUS.md` | 当前阶段、进行中工作、未解决问题和下一步 |
| `docs/agent/PROJECT_STRUCTURE.md` | 当前目录职责和未来新文件的唯一存放规则 |
| `docs/worklog/WORKLOG.md` | 实时工程进展、实验指标、失败复盘和阶段判断 |
| `docs/planning/` | 当前学习路线、下一阶段任务和验收标准 |
| `docs/debugging/BUGLOG.md` | Bug 现象、证据、根因、单变量实验和最终结论 |
| `docs/superpowers/specs/` | 实施前确认的设计规格，属于历史决策证据 |
| `docs/superpowers/plans/` | 对应设计的分步实施计划，属于历史执行记录 |

`README.md`、工作日志、学习计划和 Bug 日志属于持续维护的工程文档；历史设计规格和
实施计划用于解释当时为什么这样做，不应为了匹配新结构而随意重写旧记录。

## 生成图片和实验结果 `outputs/`

`outputs/` 只保存运行生成、可重新产生的本地证据，不保存正式源码。五类图片必须这样
路由：

```text
Legacy Baseline RGB frame         -> outputs/dataset/
Versioned expert dataset          -> outputs/dataset/<version>/
Single Stage 3 control-step image -> outputs/probe/
Batch probe failure image         -> outputs/probe_evaluations/<run>/
Fixed Qwen input image            -> outputs/vlm_samples/<experiment>/
Qwen annotated/evaluation image   -> outputs/vlm_evaluations/<experiment>/
```

| 目录 | 内容 |
| --- | --- |
| `outputs/dataset/` | 旧 Baseline 数据，以及按版本子目录保存的新专家数据、manifest、配置快照和质量报告 |
| `outputs/probe/` | Stage 3 单次闭环步骤图和 trace |
| `outputs/probe_evaluations/` | 批量 probe 的独立运行目录、失败图片和汇总 |
| `outputs/vlm_samples/` | 固定 VLM 输入图片、样本清单和不进入 prompt 的诊断真值 |
| `outputs/vlm_evaluations/` | 模型预测、grounding 标注图、反投影结果和评估摘要 |

`outputs/` 已被 Git 忽略。需要分享数据时单独打包或上传数据平台，不要把批量图片提交
到源码仓库。

## 隐藏目录和工具目录

| 目录 | 作用 | 管理规则 |
| --- | --- | --- |
| `.agents/` | 项目专用 Skills 的可选位置；当前为空 | 仅在确实需要复用工作流时创建 `.agents/skills/<name>/SKILL.md` |
| `.codex/` | 项目级 Codex 设置的可选位置；当前为空 | 仅在需要项目覆盖配置时创建 `config.toml`，禁止写 API Key |
| `.vscode/` | VS Code 本机项目设置 | 不属于业务代码，当前被 Git 忽略 |
| `.worktrees/` | 其他 Git 分支的隔离工作目录 | 通过 `git worktree` 管理；分支合并和验证后才能清理 |
| `__pycache__/` | Python 生成的 `.pyc` 字节码缓存 | 可以清理，运行 Python 后会自动生成 |
| `.git/` | Git 提交、分支和 worktree 元数据 | 不能手工编辑或删除 |

当前没有活动 worktree。以后在 `.worktrees/` 创建的隔离分支不会自动进入 `main`；
必须先提交分支、通过 Git 合并并在主工作区复验，完成后再删除 worktree。生成实验证据
如需保留，应先安全复制到正式 `outputs/` 位置并比较，不能把整个 worktree 当归档。

## 常见任务快速定位

| 想做什么 | 首先查看或修改 |
| --- | --- |
| 调整相机、episode 数量或输出路径 | `sim_config.yaml` |
| 修改机械臂采集或 IK 控制 | `src/vla_project/simulation/control_arm.py` |
| 检查专家数据质量或 pilot 门禁 | `src/vla_project/simulation/evaluate_dataset.py` |
| 审计专家图片遮挡和红块可见率 | `src/vla_project/simulation/audit_dataset_visibility.py` |
| 修改单次闭环决策和执行 | `src/vla_project/simulation/stage3_probe.py` |
| 修改批量 probe 统计 | `src/vla_project/simulation/evaluate_probe.py` |
| 修改相机反投影数学 | `src/vla_project/simulation/camera_geometry.py` |
| 修改 VLM 样本生成 | `src/vla_project/vlm/collect_vlm_eval_samples.py` |
| 修改 Qwen 方向或 grounding 评估 | `src/vla_project/vlm/` 中对应评估模块 |
| 修改在线 grounding smoke 或筛选 | `src/vla_project/vlm/grounding_smoke/` 中对应模块 |
| 新增维护脚本 | `src/vla_project/tools/`，并在 `tests/tools/` 添加测试 |
| 记录今天完成了什么 | `docs/worklog/WORKLOG.md` |
| 记录新 Bug 和根因 | `docs/debugging/BUGLOG.md` |
| 规划下一阶段 | `docs/planning/` |
| 查找某类生成图片 | `outputs/` 的对应分类目录 |

## 最后检查

新增文件前确认：

- 它属于正式源码、测试、文档、配置还是生成结果？
- 正式代码和测试是否位于相互镜像的领域目录？
- 生成图片是否进入正确的 `outputs/` 子目录？
- 是否错误地把 `.worktrees/` 当成正式归档目录？
- 是否准备在根目录增加 `.py` 文件或图片？如果是，应重新选择位置。
- 是否要保存 API Key、访问令牌或其他密钥？如果是，立即停止，改用环境变量。
- 是否已有职责相同的文件或同功能子文件夹？如果有，优先复用。
- 新文件是否只是为了当前任务方便，而不是承担新的稳定职责？
- 新子文件夹是否确实包含多个紧密关联、可独立测试的组件？
- 子文件夹名称是否使用稳定功能名，而不是阶段编号、日期或临时名称？
- 新测试路径是否严格镜像源码路径？
- 如果无法说明现有位置为什么不合适，是否应停止创建并先询问用户？
