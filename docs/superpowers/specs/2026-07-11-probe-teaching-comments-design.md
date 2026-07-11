# Probe 教学型注释设计

## 目标

为 `stage3_probe.py` 和 `evaluate_probe.py` 增加面向学习、调试和面试复盘的中文注释，不改变任何运行行为。

## 注释原则

- 解释数据流、指标语义和设计原因，不逐行翻译 Python 语法。
- 每个公共函数说明输入、输出、异常或副作用。
- 关键流程用分段注释串起“配置→episode→trace→摘要→批次统计”。
- 明确世界坐标方向、随机种子、错误隔离和失败图片保留策略。
- 保留现有有效注释，删除或改写已经不符合批量运行结构的旧描述。

## `stage3_probe.py`

- 增加模块级职责说明，区分单次闭环和批量调用。
- 扩充 `summarize_probe_trace` 与 `run_probe_episode` 的参数、返回值及指标来源。
- 解释每个 episode 独立初始化和 finally 清理 PyBullet 的原因。
- 解释动作前后坐标、`distance_delta`、termination reason 和 trace 行的用途。
- 解释 `main()` 只是兼容单次运行入口。

## `evaluate_probe.py`

- 增加模块级批量评估数据流。
- 解释成功图片删除、失败证据保留和磁盘空间权衡。
- 解释异常 episode 为什么计入失败率但不终止批次。
- 解释汇总指标对 failure mode analysis 的价值。
- 解释基础 seed 加 episode_idx 的复现方式和配置快照。

## 非目标

- 不修改函数签名、配置、统计公式、输出格式或控制参数。
- 不运行新的 20 次评估。
- 不调整 IK、方向策略或任何 bug 行为。

## 验证

- 21 个单元测试全部通过。
- 两个脚本语法检查通过。
- `git diff --check` 通过。
- 最终 diff 除缩进格式外只包含注释和 docstring。
