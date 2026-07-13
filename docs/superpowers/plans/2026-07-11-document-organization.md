# Project Document Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 归档项目工作文档并建立带真实证据的 BUGLOG。

**Architecture:** 保留根目录代码与入口不变，仅将工作日志和学习计划移动到 `docs/` 的职责目录；README 提供统一导航，BUGLOG 独立记录调查过程。

**Tech Stack:** Git、Markdown、unittest。

## Global Constraints

- 不移动 Python、YAML、requirements、README 或 tests。
- 保留文件 Git 历史并修正所有当前文档引用。
- BUG-001 的根因保持“待验证”，不能把假设写成结论。

### Task 1: 移动现有文档

- [ ] 创建 `docs/worklog`、`docs/planning`、`docs/debugging`。
- [ ] 将 `WORKLOG.md` 移到 `docs/worklog/WORKLOG.md`。
- [ ] 将学习计划 Markdown 和 PDF 移到 `docs/planning/`。

### Task 2: 创建 BUGLOG 并修正导航

- [ ] 创建 `docs/debugging/BUGLOG.md`，写入 BUG-001 的批次证据、step 21 坐标、当前假设和 60→120 单变量实验。
- [ ] 更新 README 目录树和三个文档链接。
- [ ] 更新学习计划内部对 PDF 和 WORKLOG 的相对引用。

### Task 3: 验证

- [ ] 搜索旧路径引用并确认历史设计/计划中的旧路径只作为历史记录保留。
- [ ] 运行全部单元测试和 Python 语法检查。
- [ ] 运行 `git diff --check` 并检查 Git rename 状态。
