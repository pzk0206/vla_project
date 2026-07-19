# 项目文件结构说明文档设计

## 目标

在 `docs/PROJECT_STRUCTURE.md` 新增一份同时面向 Codex 和项目维护者的中文文件路由
手册。下次开始任务时，只要先读这一份文档，就能立即判断新代码、对应测试、工程
文档和不同类别生成图片的唯一目标位置，不需要重新猜测或扫描整个仓库。

这份文档的首要问题不是“现有文件是什么”，而是“接下来产生的新文件必须放哪里”。

## 文档范围

文档以当前 `main` 分支的真实结构为准，覆盖：

- 仓库根目录的项目配置文件。
- `src/vla_project/` 下的 `simulation`、`vlm`、`tools` 正式源码分类。
- 与正式源码镜像对应的 `tests/` 分类。
- `docs/` 下的工作日志、学习计划、Bug 日志和设计/实施记录。
- `outputs/` 下五类生成输出，只说明分类，不展开批次和图片文件。
- `.agents/`、`.codex/`、`.vscode/`、`.worktrees/`、`__pycache__/` 等隐藏或工具目录。

不修改代码、配置、输出数据或现有文档内容，不把 `.git/` 内部对象展开说明。

## 文档结构

最终文档采用一份文件，按下面顺序组织：

1. **阅读提示**：区分正式源码、测试、文档、生成结果和工具目录。
2. **带注释的目录树**：展示到能解释职责的层级，不列举缓存和图片明细。
3. **根目录文件说明**：解释 `.gitignore`、`pyproject.toml`、`README.md`、
   `requirements.txt` 和 `sim_config.yaml`。
4. **正式源码说明**：逐个解释 11 个 Python 模块的职责，并明确三类包的边界。
5. **测试目录说明**：说明测试与源码的镜像关系，以及两个跨领域契约测试。
6. **文档目录说明**：解释哪些是实时工程文档，哪些是历史设计和实施记录。
7. **输出与隐藏目录说明**：说明生成内容、缓存、编辑器配置、Skills、Codex 配置和
   Git worktree 的生命周期。
8. **常见任务快速定位表**：用“想做什么 -> 去哪里”的方式提供导航。
9. **新文件路由规则**：按文件类型给出唯一推荐位置、配套测试位置和禁止位置。
10. **维护规则**：标明可自动重建、应由 Git 管理、不能手动删除的目录。

## 必须固化的路由规则

最终文档必须让 Codex 能直接执行以下判断：

| 新内容 | 唯一推荐位置 | 配套位置 |
| --- | --- | --- |
| PyBullet、相机、机械臂控制代码 | `src/vla_project/simulation/` | `tests/simulation/` |
| VLM、grounding、反投影、校准代码 | `src/vla_project/vlm/` | `tests/vlm/` |
| 仓库维护和迁移工具 | `src/vla_project/tools/` | `tests/tools/` |
| 仿真、相机、任务和输出路径参数 | `sim_config.yaml` | `tests/test_config_contract.py` |
| Python 包、版本和命令入口 | `pyproject.toml` | `tests/test_package_metadata.py` |
| 第三方 Python 依赖 | `requirements.txt` | 安装和全量测试 |
| 项目安装、运行和阶段概览 | `README.md` | 不适用 |
| 文件职责和新文件路由规则 | `docs/PROJECT_STRUCTURE.md` | 不适用 |
| 当前工程进展和实验复盘 | `docs/worklog/WORKLOG.md` | 不适用 |
| 后续路线和验收计划 | `docs/planning/` | 不适用 |
| Bug 证据、根因和修复结论 | `docs/debugging/BUGLOG.md` | 不适用 |
| 功能或重构设计规格 | `docs/superpowers/specs/` | 不适用 |
| 已确认设计的实施步骤 | `docs/superpowers/plans/` | 不适用 |
| Baseline 训练图片和轨迹 | `outputs/dataset/` | 同目录 JSONL/摘要 |
| Stage 3 单次闭环图片 | `outputs/probe/` | 同目录 trace |
| Stage 3 批量评估图片 | `outputs/probe_evaluations/` | 对应批次目录 |
| VLM 固定输入图片 | `outputs/vlm_samples/<实验名>/` | 同目录样本清单和诊断 |
| VLM 预测、标注图和评估结果 | `outputs/vlm_evaluations/<实验名>/` | 同目录结果和摘要 |
| 项目专用 Codex Skill | `.agents/skills/<skill-name>/` | `SKILL.md` 及可选资源 |
| 项目级 Codex 设置 | `.codex/config.toml` | 当前没有时不要凭空创建 |

文档还必须明确以下禁止规则：

- 仓库根目录不新增正式 `.py` 文件。
- 正式代码不放进 `tests/`、`docs/` 或 `outputs/`。
- 测试文件不放进 `src/`。
- 任何生成图片不放在仓库根目录、`src/`、`tests/` 或 `docs/`。
- `outputs/` 只保存可再生成的实验结果，不提交批量图片到 Git。
- `.worktrees/` 是独立分支工作区，不能作为正式代码的最终归档位置。

对于不能直接归类的新内容，文档要求 Codex 先根据其主要职责选择最接近的领域；如果
仍会改变包边界或新增顶层分类，再向用户确认，不能自行在根目录创建临时文件夹。

## 表达方式

- 使用短句、表格和一棵紧凑目录树，避免大段抽象描述。
- 所有路径使用仓库相对路径并用反引号标记。
- 模块说明同时回答“做什么”和“什么时候会用到”。
- 路由表同时回答“放哪里”“配套测试放哪里”和“不能放哪里”。
- 对 `outputs/` 和 `.worktrees/` 明确提示它们不是正式源码。
- 对当前为空的 `.agents/` 和 `.codex/` 说明其预期用途，不虚构已存在的配置文件。

## 正确性检查

完成最终文档后执行以下检查：

- 从目录树和表格中提取的本地路径都能在当前仓库中找到；预期用途路径需要明确标注
  “可选”或“当前为空”。
- 文档不再使用迁移前的根目录 Python 路径。
- 使用至少一个代码文件、一个测试文件、一份工程文档和五类生成图片案例检查路由表，
  每个案例都必须得到唯一目标路径。
- `outputs/` 只读检查，不能移动、删除或重新生成任何实验结果。
- `git diff --check` 通过，且只包含计划内的文档改动。
