# Agent File Placement Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有文件路由手册中固化“优先复用现有位置，必要时才新建功能子文件夹”的 Agent 决策规则。

**Architecture:** 只修改 `docs/agent/PROJECT_STRUCTURE.md`，在现有路由表之后增加文件放置决策流程、子文件夹准入条件、命名和测试镜像规则，并扩展末尾检查项。根目录 `AGENTS.md` 已要求创建或移动文件前读取该文档，因此无需增加新的入口文件。

**Tech Stack:** Markdown、Git、`rg`。

## Global Constraints

- 实施阶段只修改 `docs/agent/PROJECT_STRUCTURE.md`。
- 不修改 `AGENTS.md`、源码、测试、配置、README 或其他项目知识文档。
- 优先复用职责相同的现有文件或目录；文件名相似不能替代职责判断。
- 只有形成独立、可扩展的多文件工作流时才新建功能子文件夹。
- 测试目录必须镜像源码目录。
- 无法明确归类时询问用户，不自行创建新的顶层领域。
- 功能子文件夹使用小写蛇形功能名，不使用阶段编号、日期或临时名称。

---

## File Structure

| 文件 | 操作 | 职责 |
| --- | --- | --- |
| `docs/agent/PROJECT_STRUCTURE.md` | 修改 | 增加 Agent 创建文件和子文件夹的统一决策规则 |

---

### Task 1: 增加文件放置与子文件夹决策规则

**Files:**
- Modify: `docs/agent/PROJECT_STRUCTURE.md`

**Interfaces:**
- Consumes: 现有 `simulation`、`vlm`、`tools` 领域路由和源码/测试镜像规则。
- Produces: Agent 在新建文件前必须遵循的单一权威决策流程。

- [ ] **Step 1: 验证现有入口和插入位置**

Run:

```bash
rg -n '^## (新文件应该放在哪里|当前目录树|最后检查)$' docs/agent/PROJECT_STRUCTURE.md
rg -n '创建或移动文件前.*PROJECT_STRUCTURE' AGENTS.md
```

Expected:

```text
PROJECT_STRUCTURE.md 包含“新文件应该放在哪里”“当前目录树”“最后检查”三个章节；
AGENTS.md 明确要求创建或移动文件前读取 PROJECT_STRUCTURE.md。
```

- [ ] **Step 2: 在路由表与当前目录树之间加入完整决策规则**

在“无法直接归类时……”段落之后、`## 当前目录树` 之前插入以下 Markdown：

````markdown
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
````

- [ ] **Step 3: 扩展末尾检查清单**

在 `## 最后检查` 的列表末尾追加：

```markdown
- 是否已有职责相同的文件或同功能子文件夹？如果有，优先复用。
- 新文件是否只是为了当前任务方便，而不是承担新的稳定职责？
- 新子文件夹是否确实包含多个紧密关联、可独立测试的组件？
- 子文件夹名称是否使用稳定功能名，而不是阶段编号、日期或临时名称？
- 新测试路径是否严格镜像源码路径？
- 如果无法说明现有位置为什么不合适，是否应停止创建并先询问用户？
```

- [ ] **Step 4: 验证规则覆盖与变更范围**

Run:

```bash
rg -n '^## (新任务的文件放置决策|什么时候新建功能子文件夹)$' docs/agent/PROJECT_STRUCTURE.md
rg -n '优先复用|独立、可扩展的多文件工作流|小写蛇形|严格镜像|询问用户' docs/agent/PROJECT_STRUCTURE.md
rg -n 'stage4/|new_task/|grounding_smoke/' docs/agent/PROJECT_STRUCTURE.md
rg -n 'T[B]D|T[O]DO|待.定|占.位' docs/agent/PROJECT_STRUCTURE.md
git diff --check
git status --short
```

Expected:

- 两个新章节均存在；
- 复用、建目录条件、命名、测试镜像和询问用户规则均可检索；
- 正反例均存在；
- 占位符扫描和格式检查无输出；
- Git 状态只显示 `docs/agent/PROJECT_STRUCTURE.md` 被修改。

- [ ] **Step 5: 提交文档规则**

```bash
git add docs/agent/PROJECT_STRUCTURE.md
git diff --cached --name-only
git diff --cached --check
git commit -m "docs: define agent file placement rules"
```

Expected: 只提交 `docs/agent/PROJECT_STRUCTURE.md`。

---

## 完成判据

- Agent 能在扩展现有文件、领域目录新增模块和建立功能子包之间做出明确选择。
- 没有明确理由时默认复用现有文件或目录，不为每个任务创建新文件夹。
- 功能子文件夹必须使用稳定功能名，并具有多文件独立工作流边界。
- 测试目录严格镜像源码目录。
- 无法归类时要求询问用户。
- 实施变更只包含 `docs/agent/PROJECT_STRUCTURE.md`。
