# 项目文档整理与 Bug 日志设计

## 目标

在不改变 Python 运行入口、配置路径和测试导入方式的前提下，把项目工作文档按用途归档，并新增可持续维护的故障日志。

## 目标结构

```text
vla_project/
├── README.md
├── control_arm.py
├── stage3_probe.py
├── evaluate_probe.py
├── sim_config.yaml
├── requirements.txt
├── docs/
│   ├── debugging/
│   │   └── BUGLOG.md
│   ├── worklog/
│   │   └── WORKLOG.md
│   ├── planning/
│   │   ├── vla_robotic_study_plan.md
│   │   └── vla_robotic_study_plan.pdf
│   └── superpowers/
│       ├── specs/
│       └── plans/
└── tests/
```

## 文件职责

- 根目录 `README.md`：GitHub 项目入口、运行命令、当前阶段和文档导航。
- `docs/worklog/WORKLOG.md`：按时间和阶段记录项目推进、实验结果与工程判断。
- `docs/planning/`：长期学习路线及其原始 PDF 资料。
- `docs/debugging/BUGLOG.md`：按 bug 编号记录故障现象、证据、根因假设、实验和最终结论。
- `docs/superpowers/`：保留设计规格和实施计划，不与项目业务文档混放。

## BUGLOG 首条记录

新增 `BUG-001：负 x 目标下离散控制振荡`，状态为“调查中”。记录：

- 20 次固定种子评估成功率为 25%。
- 12 个 `block_x < 0` episode 全部失败。
- 成功目标 x 均值 0.149，失败目标 x 均值 -0.072。
- `episode_001` step 21 的 `left` 目标映射正确。
- 期望 dx=-0.03m，实际 dx=-0.00675m。
- 期望 dy=0，实际 dy=-0.02197m。
- `distance_delta=-0.01090m`，动作后更远。
- 当前假设是 60 个物理 step 不足或 IK/关节耦合导致执行偏移。
- 下一实验只把 `sim_steps_per_action` 从 60 调到 120，保持 seed、步长和最大控制步数不变。

## 路径迁移规则

- 使用 Git 感知的移动保留文件历史。
- 全仓库搜索 `WORKLOG.md`、`vla_robotic_study_plan.md` 和 PDF 旧路径。
- README 目录树和文档链接统一更新为新路径。
- 文档内部如果引用同目录文件，使用相对路径；代码和配置路径不变。
- 不移动 README、Python 脚本、YAML、requirements 或 tests。

## 验证

- `rg` 检查旧路径不存在未更新引用。
- Markdown 文件和 PDF 均存在于目标目录。
- Python 全部单元测试继续通过。
- Python 语法检查通过。
- `git diff --check` 无格式错误。

## 完成标准

- 根目录只保留项目入口、代码、配置和依赖文件。
- 工作日志、学习计划和 bug 日志职责清晰。
- README 能从根目录导航到三类文档。
- `BUG-001` 包含可复现实验和待验证结论，不把假设写成已确认根因。
