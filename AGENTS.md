# VLA Project Agent Guidance

## Start Here

1. 开始项目任务前先读 `docs/agent/PROJECT_OVERVIEW.md`，不要先扫描整个仓库。
2. 任务涉及当前进展、实验延续或下一步决策时，再读 `docs/agent/CURRENT_STATUS.md`。
3. 定位、创建或移动文件前，读 `docs/agent/PROJECT_STRUCTURE.md`。
4. 只按当前任务需要读取源码、测试、`WORKLOG.md`、`BUGLOG.md` 和历史 specs/plans；
   不要默认全文加载全部历史文档。

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

## Documentation Language

- 新建或更新 `docs/superpowers/specs/` 和 `docs/superpowers/plans/` 时，说明文字默认使用中文。
- 技术标识符和可执行内容保持原文。
