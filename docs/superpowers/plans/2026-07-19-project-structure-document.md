# Project Structure Routing Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `docs/PROJECT_STRUCTURE.md` as the canonical routing guide that tells Codex exactly where new code, mirrored tests, engineering documents, and every generated-image category belong.

**Architecture:** Keep the guide as one standalone Chinese Markdown file. Lead with enforceable placement rules and a routing table, then explain the current tree and individual files so future agents can choose one destination without rescanning the repository.

**Tech Stack:** Markdown, Git, shell path checks, `rg`, `find`.

## Global Constraints

- Create only `docs/PROJECT_STRUCTURE.md`; do not change production code, tests, configuration, or generated outputs.
- Use repository-relative paths throughout.
- Do not list individual generated images or every timestamped experiment run.
- Do not describe optional `.agents/` or `.codex/` files as already present.
- Do not add formal Python files to the repository root.
- Do not move, delete, rewrite, or regenerate anything under `outputs/`.
- Make every new-file category resolve to one primary destination; when a new top-level category would be required, instruct Codex to ask the user first.

---

## File Structure

- Create: `docs/PROJECT_STRUCTURE.md` — canonical current-tree explanation and new-file routing rules.
- Read only: `src/vla_project/`, `tests/`, `docs/`, `outputs/`, `.agents/`, `.codex/`, `.vscode/`, `.worktrees/` — current path evidence.
- Read only: `sim_config.yaml`, `pyproject.toml`, `.gitignore` — canonical output, package, and ignore contracts.

### Task 1: Write and verify the canonical routing guide

**Files:**
- Create: `docs/PROJECT_STRUCTURE.md`
- Reference: `docs/superpowers/specs/2026-07-19-project-structure-document-design.md`

**Interfaces:**
- Consumes: the current checked-out repository tree and the approved routing design.
- Produces: one Markdown guide whose routing table maps each artifact category to one primary repository-relative path.

- [ ] **Step 1: Record the current paths without modifying them**

Run:

```bash
find src/vla_project tests docs -maxdepth 3 -mindepth 1 -not -path '*/__pycache__/*' | sort
find outputs -maxdepth 1 -mindepth 1 -type d | sort
find .agents .codex .vscode .worktrees -maxdepth 2 -mindepth 1 | sort
```

Expected: the three source domains, three mirrored test domains, four live documentation groups,
five output categories, optional hidden-tool directories, and the active `grounding-world-smoke`
worktree are visible. This step is read-only.

- [ ] **Step 2: Create the guide with an action-first opening**

Create `docs/PROJECT_STRUCTURE.md` with these opening rules before the explanatory tree:

```markdown
# 项目文件结构与新文件存放规则

> 这是本仓库的文件路由手册。开始新增代码、测试、文档或生成图片前，先查本文件。

## Codex 必须先遵守的规则

1. 正式 Python 代码只能放入 `src/vla_project/` 的对应领域包。
2. 测试必须放入 `tests/` 的镜像领域目录。
3. 所有运行生成的图片和结果只能放入 `outputs/` 的对应分类。
4. 仓库根目录不新增正式 `.py` 文件或生成图片。
5. `.worktrees/` 只用于隔离开发，完成后通过 Git 合并，不能作为正式归档目录。
6. 如果现有分类无法容纳新内容，先询问用户，不自行创建新的顶层目录。
```

- [ ] **Step 3: Add the unique routing table**

Add one table with columns `新内容`、`放置位置`、`配套位置/说明`. It must include these
unambiguous mappings:

```text
PyBullet/相机/控制代码       -> src/vla_project/simulation/  -> tests/simulation/
VLM/grounding/反投影/校准    -> src/vla_project/vlm/         -> tests/vlm/
仓库维护与迁移工具           -> src/vla_project/tools/       -> tests/tools/
仿真与输出路径参数           -> sim_config.yaml              -> tests/test_config_contract.py
包元数据与命令入口           -> pyproject.toml               -> tests/test_package_metadata.py
第三方 Python 依赖           -> requirements.txt             -> 安装后跑全量测试
安装/运行/阶段概览            -> README.md                     -> 不记录详细历史
文件结构和路由规则           -> docs/PROJECT_STRUCTURE.md
工程进展与实验复盘           -> docs/worklog/WORKLOG.md
路线和验收计划               -> docs/planning/
Bug 证据与结论               -> docs/debugging/BUGLOG.md
设计规格                     -> docs/superpowers/specs/
实施步骤                     -> docs/superpowers/plans/
Baseline 图片和轨迹          -> outputs/dataset/
Stage 3 单次闭环             -> outputs/probe/
Stage 3 批量评估             -> outputs/probe_evaluations/<批次>/
VLM 固定输入图片             -> outputs/vlm_samples/<实验名>/
VLM 预测/标注图/评估         -> outputs/vlm_evaluations/<实验名>/
项目专用 Skill              -> .agents/skills/<skill-name>/
项目级 Codex 配置            -> .codex/config.toml（仅确有需要时创建）
```

- [ ] **Step 4: Add the annotated current tree and file responsibilities**

Add a compact tree that includes root configuration, `src/vla_project/{simulation,vlm,tools}`,
mirrored `tests/`, four `docs/` groups, five `outputs/` groups, and hidden/tool directories.
Then add concise tables that explain:

- the five root configuration files;
- all 11 production modules;
- mirrored test files plus `test_config_contract.py` and `test_package_metadata.py`;
- live engineering docs versus historical specs/plans;
- `__pycache__/` as rebuildable, `.worktrees/` as Git-managed isolation, and `.git/` as internal Git state.

The module descriptions must distinguish collection, single-probe control, batch evaluation,
VLM sample generation, direct-decision evaluation, grounding diagnosis, ground-then-decide,
backprojection evaluation, calibration validation, and output migration.

- [ ] **Step 5: Add generated-image routing examples and maintenance rules**

Include these five examples exactly as behavior checks:

```text
Baseline RGB frame                -> outputs/dataset/
Single Stage 3 control-step image -> outputs/probe/
Batch probe failure image         -> outputs/probe_evaluations/<run>/
Fixed Qwen input image            -> outputs/vlm_samples/<experiment>/
Qwen annotated/evaluation image   -> outputs/vlm_evaluations/<experiment>/
```

State that `outputs/` is ignored by Git, caches can be regenerated, worktrees must be removed with
Git only after their branch is integrated, and API keys must never be stored in the repository.

- [ ] **Step 6: Verify the guide against the repository**

Run:

```bash
test -f docs/PROJECT_STRUCTURE.md
rg -n 'src/vla_project/(simulation|vlm|tools)|tests/(simulation|vlm|tools)|outputs/(dataset|probe|probe_evaluations|vlm_samples|vlm_evaluations)' docs/PROJECT_STRUCTURE.md
! rg -n '(^|`)python (control_arm|stage3_probe|evaluate_probe)\.py|(^|`)\w+\.py -> 仓库根目录' docs/PROJECT_STRUCTURE.md
git diff --check
git status --short
```

Expected: the guide exists, all three source/test domains and five output categories are present,
no obsolete root-script instruction is found, `git diff --check` exits 0, and only
`docs/PROJECT_STRUCTURE.md` is uncommitted.

- [ ] **Step 7: Commit the guide**

```bash
git add docs/PROJECT_STRUCTURE.md
git commit -m "docs: add project file routing guide"
```

Expected: one documentation-only commit; production files and `outputs/` remain unchanged.
