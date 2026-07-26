# Superpowers 文档中文规范实施计划

> **面向代理执行者：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，逐项执行本计划。步骤使用复选框（`- [ ]`）跟踪。

**目标：** 将当前 grounding smoke 固定案例重选计划的说明文字完整翻译为中文，并在项目级 `AGENTS.md` 中加入两条文档语言规则。

**架构：** `AGENTS.md` 只保存长期语言约束；现有实施计划原地翻译，不创建中英双语副本。翻译前后对代码围栏内容、复选框数量和关键技术字面量进行机械校验，确保只改变说明文字。

**技术栈：** Markdown、CommonMark 代码围栏、`rg`、`awk`、`diff`、Git。

## 全局约束

- 新建或更新 `docs/superpowers/specs/` 和 `docs/superpowers/plans/` 时，说明文字默认使用中文。
- 技术标识符和可执行内容保持原文。
- 不改变当前 grounding smoke 实施计划的任务顺序、步骤、代码、命令、路径、参数、数值或验收标准。
- 不修改源码、测试、配置或实验输出。
- 保留用户已有的未跟踪文件 `docs/superpowers/plans/2026-07-19-agent-file-placement-rules.md`，不得暂存或修改。

---

### 任务 1：写入项目级文档语言规则

**文件：**
- 修改：`AGENTS.md`

**接口：**
- 输入：用户批准的两条中文规则
- 输出：`AGENTS.md` 中的 `## Documentation Language` 章节

- [ ] **步骤 1：确认规则尚未存在**

运行：

```bash
rg -n "说明文字默认使用中文|技术标识符和可执行内容保持原文" AGENTS.md
```

预期：没有匹配，退出码为 1。

- [ ] **步骤 2：只加入用户指定的两条规则**

使用 `apply_patch` 在 `AGENTS.md` 末尾增加：

```markdown
## Documentation Language

- 新建或更新 `docs/superpowers/specs/` 和 `docs/superpowers/plans/` 时，说明文字默认使用中文。
- 技术标识符和可执行内容保持原文。
```

不得增加语言回退、双语、例外或其他未获批准的规则。

- [ ] **步骤 3：验证两条规则只出现一次**

运行：

```bash
rg -n "说明文字默认使用中文|技术标识符和可执行内容保持原文" AGENTS.md
```

预期：恰好输出两行，每条规则各一行。

---

### 任务 2：将当前实施计划的说明文字翻译为中文

**文件：**
- 修改：`docs/superpowers/plans/2026-07-26-grounding-smoke-fixed-case-reselection.md`

**接口：**
- 输入：现有 4 个任务、全部复选步骤及代码围栏
- 输出：任务结构和可执行内容不变的中文实施计划

- [ ] **步骤 1：保存翻译前的机械基线**

运行：

```bash
awk '/^```/{inside=!inside; print; next} inside{print}' \
  docs/superpowers/plans/2026-07-26-grounding-smoke-fixed-case-reselection.md \
  > /tmp/grounding-smoke-plan-fences.before
```

运行：

```bash
rg -c '^- \[ \]' \
  docs/superpowers/plans/2026-07-26-grounding-smoke-fixed-case-reselection.md
```

记录复选框数量；翻译后必须完全相同。

- [ ] **步骤 2：原地翻译说明文字**

使用 `apply_patch` 翻译以下内容：

- 文档标题；
- 代理执行提示中的自然语言；
- `Goal`、`Architecture`、`Tech Stack` 和 `Global Constraints` 标题及说明；
- 任务标题、`Files`、`Interfaces`、`Consumes`、`Produces`、`Preserves`；
- 每个步骤标题、操作说明和预期结果；
- 普通段落、列表和验收说明。

以下内容逐字符保留：

- 所有代码围栏内部内容；
- 反引号包裹的文件路径、函数名、字段名、CLI 参数和命令；
- Python、Shell、JSON 和配置字面量；
- 数值、seed、方向标签和 Git 提交消息。

- [ ] **步骤 3：验证代码围栏内容没有变化**

运行：

```bash
awk '/^```/{inside=!inside; print; next} inside{print}' \
  docs/superpowers/plans/2026-07-26-grounding-smoke-fixed-case-reselection.md \
  > /tmp/grounding-smoke-plan-fences.after
```

运行：

```bash
diff -u \
  /tmp/grounding-smoke-plan-fences.before \
  /tmp/grounding-smoke-plan-fences.after
```

预期：无输出，退出码为 0。

- [ ] **步骤 4：验证计划结构和关键字面量**

再次运行复选框计数，确认与步骤 1 完全相同：

```bash
rg -c '^- \[ \]' \
  docs/superpowers/plans/2026-07-26-grounding-smoke-fixed-case-reselection.md
```

运行：

```bash
rg -n \
  "55–100|left|right|front|0\\.005m|0\\.75|--preflight-only|call_openai_compatible_api|python -m unittest discover" \
  docs/superpowers/plans/2026-07-26-grounding-smoke-fixed-case-reselection.md
```

预期：所有关键技术字面量仍存在。

---

### 任务 3：检查补丁并提交

**文件：**
- 修改：`AGENTS.md`
- 修改：`docs/superpowers/plans/2026-07-26-grounding-smoke-fixed-case-reselection.md`

**接口：**
- 输入：任务 1 和任务 2 的文档修改
- 输出：单个仅包含语言规则和翻译的 Git 提交

- [ ] **步骤 1：检查格式和范围**

运行：

```bash
git diff --check
git diff --stat
git status --short
```

预期：

- `git diff --check` 退出码为 0；
- 只有 `AGENTS.md` 和目标实施计划发生预期修改；
- 已有未跟踪计划仍未暂存。

- [ ] **步骤 2：人工核对差异**

运行：

```bash
git diff -- AGENTS.md \
  docs/superpowers/plans/2026-07-26-grounding-smoke-fixed-case-reselection.md
```

确认没有技术步骤、代码、命令、数值或验收条件被删除。

- [ ] **步骤 3：提交文档修改**

```bash
git add AGENTS.md \
  docs/superpowers/plans/2026-07-26-grounding-smoke-fixed-case-reselection.md
git commit -m "docs: use Chinese for superpowers documentation"
```

- [ ] **步骤 4：报告结果**

报告修改后的两个文件链接、机械校验结果和提交哈希；不要声称源码或实验行为发生变化。
