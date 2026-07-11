# Stage 3 Control Fix and Project Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 Stage 3 离散方向被 heuristic 分支绕过的问题，并用可重复检查确认配置、执行、日志、终止与文档之间不存在同类的不一致。

**Architecture:** 在 `stage3_probe.py` 中增加一个不依赖 PyBullet 状态的纯函数，将世界坐标方向转换成单步目标位置；heuristic 与 API 共用该函数。审查保持现有脚本结构，通过标准库测试、AST/配置检查、JSONL 审计和一次 DIRECT probe 验证五条数据流，只修复能被证据复现的问题。

**Tech Stack:** Python 3、标准库 `unittest`、PyYAML、OpenCV、PyBullet、JSONL。

## Global Constraints

- 保留用户当前未提交改动，不覆盖或回退无关内容。
- 不接入真实 VLM API，不新增 `evaluate_probe.py`，不做大规模模块拆分。
- 所有方向名表示 PyBullet 世界坐标，不等同于相机图像方向。
- 每个确认的行为错误必须先有失败测试或可重复审计证据，再做最小修复。
- 缺失依赖时准确记录验证边界，不把静态检查表述成运行成功。

---

### Task 1: 为离散方向执行建立失败回归测试

**Files:**
- Create: `tests/test_stage3_probe.py`
- Inspect: `stage3_probe.py:28-36,295-315`

**Interfaces:**
- Consumes: `DIRECTION_TO_DELTA: dict[str, tuple[float, float]]`。
- Produces: 期望接口 `direction_to_target(direction, ee_pos, hover_height, move_step_xy) -> list[float]`。

- [ ] **Step 1: 创建标准库测试并定义期望行为**

```python
import unittest

from stage3_probe import direction_to_target


class DirectionToTargetTests(unittest.TestCase):
    def test_cardinal_directions_move_one_world_axis_step(self):
        ee_pos = [1.0, 2.0, 3.0]
        cases = {
            "left": [0.75, 2.0, 0.5],
            "right": [1.25, 2.0, 0.5],
            "front": [1.0, 2.25, 0.5],
            "back": [1.0, 1.75, 0.5],
        }
        for direction, expected in cases.items():
            with self.subTest(direction=direction):
                self.assertEqual(
                    direction_to_target(direction, ee_pos, 0.5, 0.25),
                    expected,
                )

    def test_stop_keeps_xy_and_uses_hover_height(self):
        self.assertEqual(
            direction_to_target("stop", [1.0, 2.0, 3.0], 0.5, 0.25),
            [1.0, 2.0, 0.5],
        )

    def test_unknown_direction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知方向"):
            direction_to_target("up", [1.0, 2.0, 3.0], 0.5, 0.25)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认 RED**

Run: `python -m unittest tests.test_stage3_probe -v`

Expected: import 失败，明确指出 `stage3_probe` 尚无 `direction_to_target`。

---

### Task 2: 统一 heuristic 与 API 的方向执行路径

**Files:**
- Modify: `stage3_probe.py:28-36,295-315`
- Test: `tests/test_stage3_probe.py`

**Interfaces:**
- Consumes: `direction`, `ee_pos_before`, `hover_target[2]`, `probe_config["move_step_xy"]`。
- Produces: `direction_to_target(direction, ee_pos, hover_height, move_step_xy) -> list[float]`，供主控制循环统一调用。

- [ ] **Step 1: 实现最小纯函数**

```python
def direction_to_target(direction, ee_pos, hover_height, move_step_xy):
    """把世界坐标方向转换成单步末端目标位置。"""
    if direction not in DIRECTION_TO_DELTA:
        raise ValueError(f"未知方向: {direction}")
    delta_x, delta_y = DIRECTION_TO_DELTA[direction]
    return [
        ee_pos[0] + delta_x * move_step_xy,
        ee_pos[1] + delta_y * move_step_xy,
        hover_height,
    ]
```

- [ ] **Step 2: 主循环删除 decision source 特判并统一调用**

```python
target_pos = direction_to_target(
    direction,
    ee_pos_before,
    hover_target[2],
    probe_config["move_step_xy"],
)
```

- [ ] **Step 3: 运行单元测试确认 GREEN**

Run: `python -m unittest tests.test_stage3_probe -v`

Expected: 3 tests pass。

- [ ] **Step 4: 检查测试是否真正覆盖原 bug**

Run: `rg -n 'decision_source == "heuristic"|target_pos = hover_target' stage3_probe.py`

Expected: 无匹配；主循环不再按决策来源改变动作语义。

---

### Task 3: 审查配置到代码的消费链

**Files:**
- Inspect: `sim_config.yaml`
- Inspect: `control_arm.py`
- Inspect: `stage3_probe.py`
- Create: `tests/test_config_contract.py`

**Interfaces:**
- Consumes: `sim_config.yaml` 顶层和嵌套键。
- Produces: 配置契约测试，保证主脚本需要的配置键存在且关键数值有效。

- [ ] **Step 1: 创建配置契约测试**

```python
import unittest

from control_arm import load_config


class ConfigContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("sim_config.yaml")

    def test_required_sections_exist(self):
        self.assertTrue(
            {"connection_mode", "gravity", "robot", "dataset", "task", "camera", "probe"}
            <= self.config.keys()
        )

    def test_probe_distances_and_steps_are_positive(self):
        probe = self.config["probe"]
        self.assertGreater(probe["move_step_xy"], 0)
        self.assertGreater(probe["max_control_steps"], 0)
        self.assertGreater(probe["sim_steps_per_action"], 0)
        self.assertGreater(probe["stop_distance_xy"], 0)

    def test_probe_mode_is_supported(self):
        self.assertIn(self.config["probe"]["mode"].lower(), {"heuristic", "api"})

    def test_capture_and_termination_limits_are_consistent(self):
        dataset = self.config["dataset"]
        task = self.config["task"]
        self.assertGreater(dataset["capture_interval_steps"], 0)
        self.assertLess(task["force_terminal_after_step"], dataset["max_steps_per_episode"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行配置契约测试**

Run: `python -m unittest tests.test_config_contract -v`

Expected: 4 tests pass；若失败，失败字段即为可重复审计证据，先定位其代码消费点再最小修复。

- [ ] **Step 3: 静态列出 YAML 键与代码索引访问并逐项核对**

Run: `rg -n 'config\[|_config\[|_cfg\[' control_arm.py stage3_probe.py`

Expected: 人工核对每个运行时索引均能在 `sim_config.yaml` 找到；记录任何读取后被硬编码覆盖的键。

---

### Task 4: 审查终止、trace 与输出证据

**Files:**
- Inspect: `control_arm.py:300-535`
- Inspect: `stage3_probe.py:225-372`
- Inspect: `dataset/episode_summary.jsonl`
- Inspect: `dataset/trajectory_expert.jsonl`
- Inspect: `probe_runs/probe_trace.jsonl`
- Modify if evidence requires: `control_arm.py` or `stage3_probe.py`
- Test if evidence requires: `tests/test_stage3_probe.py` or a focused new `tests/test_control_arm.py`

**Interfaces:**
- Consumes: episode/probe JSONL rows and termination fields。
- Produces: 对字段完整性、距离一致性、终止原因和输出覆盖语义的审查结论。

- [ ] **Step 1: 用只读脚本验证 JSONL 结构与数值一致性**

Run:

```bash
python -c 'import json,math,pathlib; files=["dataset/episode_summary.jsonl","dataset/trajectory_expert.jsonl","probe_runs/probe_trace.jsonl"]; rows={p:[json.loads(x) for x in pathlib.Path(p).read_text(encoding="utf-8").splitlines() if x.strip()] for p in files}; assert all(rows.values()); assert all(r["termination_reason"] in {"success","stuck","max_steps"} for r in rows[files[0]]); assert all(abs(math.dist(r["ee_pos_after"],r["hover_target"])-r["distance_after"])<1e-9 for r in rows[files[2]]); print({p:len(v) for p,v in rows.items()})'
```

Expected: exit 0，并输出三份文件行数；失败断言提供具体审计入口。

- [ ] **Step 2: 检查终止阈值和比较运算**

Run: `rg -n 'success_distance|stop_distance_xy|force_terminal_after_step|max_steps_per_episode|termination_reason|break' control_arm.py stage3_probe.py sim_config.yaml`

Expected: 成功阈值、方向停止阈值和最大步数职责能被逐条解释；任何边界不一致必须先新增最小纯逻辑测试再修复。

- [ ] **Step 3: 检查输出覆盖与路径语义**

Run: `rg -n 'rmtree|remove\(|open\(|imwrite|output_dir|jsonl_path|image_path' control_arm.py stage3_probe.py README.md WORKLOG.md`

Expected: 数据集清理受 `clean_before_run` 控制；probe 单次运行覆盖 trace 的行为与当前文档一致；不存在日志指向未写入图片的情况。

---

### Task 5: 运行 DIRECT probe 并同步文档结论

**Files:**
- Modify if behavior wording is stale: `README.md`
- Modify if evidence wording is stale: `WORKLOG.md`
- Modify if milestone wording is stale: `vla_robotic_study_plan.md`
- Inspect: `probe_runs/probe_trace.jsonl`

**Interfaces:**
- Consumes: 修复后的 `stage3_probe.py` 和 `sim_config.yaml`。
- Produces: 一次新的真实离散方向 trace，以及与证据一致的阶段描述。

- [ ] **Step 1: 运行一次 DIRECT heuristic probe**

Run: `python stage3_probe.py`

Expected: exit 0；trace 至少一行；每行 `target_pos.x/y - ee_pos_before.x/y` 与对应方向和 `move_step_xy` 一致。

- [ ] **Step 2: 自动核对新 trace 的方向动作语义**

Run:

```bash
python -c 'import json,pathlib,yaml; cfg=yaml.safe_load(pathlib.Path("sim_config.yaml").read_text()); s=cfg["probe"]["move_step_xy"]; d={"left":(-s,0),"right":(s,0),"front":(0,s),"back":(0,-s),"stop":(0,0)}; rows=[json.loads(x) for x in pathlib.Path("probe_runs/probe_trace.jsonl").read_text().splitlines()]; assert rows; assert all(abs((r["target_pos"][0]-r["ee_pos_before"][0])-d[r["direction"]][0])<1e-9 and abs((r["target_pos"][1]-r["ee_pos_before"][1])-d[r["direction"]][1])<1e-9 for r in rows); print(len(rows), rows[-1]["distance_after"])'
```

Expected: exit 0，并输出控制步数和最终距离。

- [ ] **Step 3: 同步文档中的 Stage 3 证据边界**

文档必须明确：旧的 13 步结果验证了 IK 接近能力，但绕过了离散方向执行；新 trace 才验证修复后的世界坐标方向控制；单次 trace 仍不代表 Stage 3 批量稳定性完成。

---

### Task 6: 完整验证与变更审计

**Files:**
- Verify: all modified Python, YAML, JSONL, and Markdown files。

**Interfaces:**
- Consumes: Tasks 1-5 的所有改动。
- Produces: 可引用的最终验证证据和问题清单。

- [ ] **Step 1: 运行全部单元测试**

Run: `python -m unittest discover -s tests -v`

Expected: 全部测试通过，0 failures，0 errors。

- [ ] **Step 2: 运行 Python 语法检查**

Run: `python -m py_compile control_arm.py stage3_probe.py tests/test_stage3_probe.py tests/test_config_contract.py`

Expected: exit 0，无输出。

- [ ] **Step 3: 检查最终差异和用户原有改动**

Run: `git diff --check && git status --short && git diff -- control_arm.py stage3_probe.py sim_config.yaml README.md WORKLOG.md vla_robotic_study_plan.md tests`

Expected: `git diff --check` exit 0；差异只包含本次修复、测试、证据同步以及用户进入任务前已有的改动，不出现无关重构。

- [ ] **Step 4: 输出审查报告**

报告按“已修复 bug、确认无问题的数据流、未修复建议、运行验证结果”四组呈现，并为每项给出文件位置或命令证据。
