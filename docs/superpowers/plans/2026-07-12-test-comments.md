# Test Comments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为四个现有测试文件补充中文教学注释，使初学者能理解每项测试保护的行为和失败含义。

**Architecture:** 只修改测试文件中的模块说明、类说明、方法 docstring、关键步骤注释和长数据排版。测试输入、mock 行为、断言与生产代码保持不变，最后用完整测试套件验证行为没有改变。

**Tech Stack:** Python 3.10、unittest、unittest.mock、PyBullet 项目测试套件

## Global Constraints

- 不改变生产代码。
- 不改变测试数据、断言、mock 行为和测试覆盖范围。
- 允许仅为可读性重新换行过长的测试数据，但不能改变数据内容。
- 不新增依赖。

---

### Task 1: 配置契约测试注释

**Files:**
- Modify: `tests/test_config_contract.py`

**Interfaces:**
- Consumes: `control_arm.load_config(config_path)` 和 `sim_config.yaml`
- Produces: 带有配置结构、参数合法性和边界关系解释的现有测试

- [ ] **Step 1:** 添加模块 docstring，解释“配置契约测试”用于在运行仿真前发现缺字段和非法参数。
- [ ] **Step 2:** 给测试类、`setUpClass()` 和五个测试方法添加中文 docstring。
- [ ] **Step 3:** 在集合包含关系和终止步数关系旁解释断言原因。
- [ ] **Step 4:** 运行 `conda run -n vla_env python -m unittest tests.test_config_contract -v`，预期 5 项通过。

### Task 2: 控制与终止逻辑测试注释

**Files:**
- Modify: `tests/test_control_arm.py`

**Interfaces:**
- Consumes: `calculate_target_joints()`、`determine_termination()`、`next_episode_index()`
- Produces: 带有 IK mock、边界优先级和临时文件解释的现有测试

- [ ] **Step 1:** 添加模块和测试类 docstring，说明三组测试各自保护的功能。
- [ ] **Step 2:** 解释四层 `@patch` 的替换对象、注入顺序和为何不启动真实 PyBullet。
- [ ] **Step 3:** 解释 `joint_info[8]`、`joint_info[9]`、IK 关键字参数及关节范围断言。
- [ ] **Step 4:** 解释 success、stuck、max_steps 的边界值和判定优先级。
- [ ] **Step 5:** 解释 `TemporaryDirectory` 如何避免污染真实 dataset，以及 episode 编号为何取最大值加一。
- [ ] **Step 6:** 运行 `conda run -n vla_env python -m unittest tests.test_control_arm -v`，预期 6 项通过。

### Task 3: 批量评估测试注释

**Files:**
- Modify: `tests/test_evaluate_probe.py`

**Interfaces:**
- Consumes: `aggregate_probe_summaries()`、`cleanup_success_images()`
- Produces: 带有汇总口径和成功/失败文件保留策略解释的现有测试

- [ ] **Step 1:** 添加模块、测试类和三个测试方法的中文 docstring。
- [ ] **Step 2:** 将三条 episode 样例按字段换行，分别注明成功、普通失败和异常案例。
- [ ] **Step 3:** 解释成功率、中位数、方向计数和失败编号的统计口径。
- [ ] **Step 4:** 解释成功 episode 删除图片但保留 trace、失败 episode 保留图片的诊断价值。
- [ ] **Step 5:** 运行 `conda run -n vla_env python -m unittest tests.test_evaluate_probe -v`，预期 3 项通过。

### Task 4: Stage 3 闭环测试注释与全量验证

**Files:**
- Modify: `tests/test_stage3_probe.py`

**Interfaces:**
- Consumes: Stage 3 方向映射、模式校验、终止判断、trace 汇总和关节误差函数
- Produces: 带有世界坐标、边界优先级、摘要字段和误差符号约定解释的现有测试

- [ ] **Step 1:** 添加模块、六个测试类和所有测试方法的中文 docstring。
- [ ] **Step 2:** 解释 `subTest` 为何让四个方向分别报告失败，以及方向属于世界坐标而非图像坐标。
- [ ] **Step 3:** 解释故意拼错的 mode、最后一步边界、trace 正负改善值和 `target - actual` 符号约定。
- [ ] **Step 4:** 仅重新排版过长的 trace 样例字典，不改变字段和值。
- [ ] **Step 5:** 运行 `conda run -n vla_env python -m unittest tests.test_stage3_probe -v`，预期 9 项通过。
- [ ] **Step 6:** 运行 `conda run -n vla_env python -m unittest discover -s tests -v`，预期全部 23 项通过。
- [ ] **Step 7:** 运行 `git diff --check`，预期无空白错误。
