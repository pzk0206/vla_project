# 测试代码教学注释设计

## 目标

为以下四个测试文件补充适合初学者阅读的中文教学注释：

- `tests/test_config_contract.py`
- `tests/test_control_arm.py`
- `tests/test_evaluate_probe.py`
- `tests/test_stage3_probe.py`

## 注释原则

采用教学式注释：文件开头解释测试模块的职责；每个测试类说明保护的功能；
每个测试方法解释测试输入、预期结果和防止的回归。对 mock、临时目录、
边界值、子测试和浮点断言补充必要解释。

注释重点解释“为什么测试”，不逐字翻译 Python 语法，也不在显而易见的代码
旁重复描述，以免注释比测试本身更难阅读。

## 约束

- 不改变生产代码。
- 不改变测试数据、断言、mock 行为和测试覆盖范围。
- 允许仅为可读性重新换行过长的测试数据，但不能改变数据内容。
- 不新增依赖。

## 验收

修改后运行：

```bash
conda run -n vla_env python -m unittest discover -s tests -v
```

所有现有测试必须继续通过，并用 `git diff --check` 检查格式问题。
