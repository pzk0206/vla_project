# 专家数据规模化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立独立复位、固定 seed、版本化 schema 和自动质量门禁的专家数据采集流程，并完成10条真实 PyBullet pilot 验收。

**Architecture:** 保留 `control_arm.py` 作为专家轨迹采集入口，在其中增加版本化目录保护、manifest、配置快照、episode 独立复位和可复现字段；新增只读的 `evaluate_dataset.py` 负责数据质量扫描与 pilot 门禁。采集和质量评估通过 JSON/JSONL 文件衔接，质量检查不修改原始数据。

**Tech Stack:** Python 3、PyBullet、OpenCV、NumPy、PyYAML、`unittest`、JSON/JSONL、YAML

## Global Constraints

- 新数据只能写入 `outputs/dataset/expert_scaling_v1/`，不得删除或覆盖现有 `outputs/dataset/` 根目录中的50条数据。
- 每条 episode 使用 `episode_seed = dataset.random_seed + episode_idx`，并在开始前把7个关节复位到 `home_joint_positions`、速度清零。
- 单帧和 episode 摘要使用 `schema_version: "expert_v1"`。
- 动作保持9维：7维关节目标、1维夹爪状态、1维终止标志。
- pilot 固定为10条；未通过质量门禁时不得继续扩到300条。
- VLM API 调用必须为0；不得修改 grounding、相机补偿或 smoke stop 逻辑。
- 新建或更新 `docs/superpowers/specs/` 和 `docs/superpowers/plans/` 时说明文字使用中文；技术标识符和可执行内容保持原文。

---

### Task 1: 冻结版本化数据目录、配置与 Manifest 契约

**Files:**
- Modify: `sim_config.yaml`
- Modify: `src/vla_project/simulation/control_arm.py`
- Modify: `tests/test_config_contract.py`
- Modify: `tests/simulation/test_control_arm.py`

**Interfaces:**
- Consumes: 现有 `prepare_dataset(dataset_config)` 和 `load_config(CONFIG_PATH)`。
- Produces: `validate_versioned_dataset_dir(output_dir) -> Path`、`build_dataset_manifest(config) -> dict`、`prepare_dataset(dataset_config, full_config=None) -> tuple[str, str, str]`。

- [ ] **Step 1: 为版本化配置写失败测试**

在 `tests/test_config_contract.py` 增加断言：

```python
dataset = self.config["dataset"]
self.assertEqual(
    dataset["output_dir"],
    "outputs/dataset/expert_scaling_v1",
)
self.assertEqual(dataset["schema_version"], "expert_v1")
self.assertEqual(dataset["random_seed"], 1000)
self.assertTrue(dataset["reset_robot_each_episode"])
self.assertEqual(dataset["home_joint_positions"], [0.0] * 7)
self.assertEqual(dataset["num_episodes"], 10)
```

在 `tests/simulation/test_control_arm.py` 增加：

```python
def test_rejects_dataset_root_as_clean_target(self):
    with self.assertRaises(ValueError):
        control_arm.validate_versioned_dataset_dir("outputs/dataset")

def test_accepts_versioned_child_directory(self):
    path = control_arm.validate_versioned_dataset_dir(
        "outputs/dataset/expert_scaling_v1"
    )
    self.assertEqual(
        path.as_posix(),
        "outputs/dataset/expert_scaling_v1",
    )
```

- [ ] **Step 2: 运行测试确认 RED**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src python -m unittest \
  tests.test_config_contract \
  tests.simulation.test_control_arm -v
```

Expected: FAIL，配置字段缺失且 `validate_versioned_dataset_dir` 尚不存在。

- [ ] **Step 3: 实现配置与目录保护**

把 `sim_config.yaml` 的 `dataset` 更新为：

```yaml
output_dir: "outputs/dataset/expert_scaling_v1"
schema_version: "expert_v1"
clean_before_run: true
num_episodes: 10
pilot_num_episodes: 10
target_num_episodes: 300
random_seed: 1000
reset_robot_each_episode: true
home_joint_positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

在 `control_arm.py` 增加：

```python
def validate_versioned_dataset_dir(output_dir):
    path = Path(output_dir)
    root = Path("outputs/dataset")
    if path == root or root not in path.parents:
        raise ValueError(
            "dataset.output_dir 必须是 outputs/dataset/ 下的版本化子目录"
        )
    return path
```

`prepare_dataset()` 在任何 `shutil.rmtree()` 前调用该函数，只允许清理版本化子目录。

- [ ] **Step 4: 为 manifest 和配置快照写失败测试**

新增测试，使用 `TemporaryDirectory` 和最小完整配置：

```python
manifest = control_arm.build_dataset_manifest(config)
self.assertEqual(manifest["schema_version"], "expert_v1")
self.assertEqual(manifest["action_dim"], 9)
self.assertEqual(manifest["seed_rule"], "random_seed + episode_idx")
self.assertEqual(manifest["pilot_num_episodes"], 10)
self.assertEqual(manifest["target_num_episodes"], 300)
```

调用 `prepare_dataset(dataset_config, full_config=config)` 后断言：

```python
self.assertTrue((run_dir / "dataset_manifest.json").is_file())
self.assertTrue((run_dir / "config_snapshot.yaml").is_file())
```

再写不兼容追加测试：已有 manifest 的 `schema_version` 或图片尺寸不同时，
`clean_before_run=false` 必须抛出 `ValueError`。

- [ ] **Step 5: 运行 manifest 测试确认 RED**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src python -m unittest \
  tests.simulation.test_control_arm -v
```

Expected: FAIL，`build_dataset_manifest` 和兼容性检查尚不存在。

- [ ] **Step 6: 实现 manifest、配置快照和追加兼容检查**

`build_dataset_manifest(config)` 返回：

```python
{
    "schema_version": dataset["schema_version"],
    "dataset_name": Path(dataset["output_dir"]).name,
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "instruction": dataset["instruction"],
    "action_dim": config["robot"]["controlled_joints"] + 2,
    "random_seed": dataset["random_seed"],
    "seed_rule": "random_seed + episode_idx",
    "image_width": camera["image_width"],
    "image_height": camera["image_height"],
    "block_position_range": task["block_position"],
    "camera_eye_offset_base": camera["eye_offset_base"],
    "camera_eye_offset_random_range": camera[
        "eye_offset_random_range"
    ],
    "pilot_num_episodes": dataset["pilot_num_episodes"],
    "target_num_episodes": dataset["target_num_episodes"],
    "jsonl_name": dataset["jsonl_name"],
    "summary_jsonl_name": dataset["summary_jsonl_name"],
}
```

首次创建时写 `dataset_manifest.json` 和 `config_snapshot.yaml`。追加模式读取现有
manifest，并精确比较：

```text
schema_version
instruction
action_dim
random_seed
seed_rule
image_width
image_height
jsonl_name
summary_jsonl_name
```

任一不一致即拒绝追加。

- [ ] **Step 7: 运行定向测试确认 GREEN**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src python -m unittest \
  tests.test_config_contract \
  tests.simulation.test_control_arm -v
```

Expected: PASS。

- [ ] **Step 8: 提交 Task 1**

```bash
git add sim_config.yaml \
  src/vla_project/simulation/control_arm.py \
  tests/test_config_contract.py \
  tests/simulation/test_control_arm.py
git commit -m "feat: version expert dataset runs"
```

---

### Task 2: 让 Episode 独立复位、固定 Seed 并写入稳定 Schema

**Files:**
- Modify: `src/vla_project/simulation/control_arm.py`
- Modify: `tests/simulation/test_control_arm.py`

**Interfaces:**
- Consumes: Task 1 的 `dataset.schema_version`、`random_seed`、`reset_robot_each_episode` 和 `home_joint_positions`。
- Produces: `reset_robot_to_home(robot_id, robot_config, dataset_config) -> None`、`write_episode_error_summary(...) -> None`、扩展后的 `run_episode(..., random_seed)`、带 `expert_v1` 字段的帧和摘要。

- [ ] **Step 1: 为机械臂独立复位写失败测试**

使用 mock 检查7个关节：

```python
@patch("vla_project.simulation.control_arm.p.resetJointState")
def test_reset_robot_to_home_clears_position_and_velocity(reset_joint):
    control_arm.reset_robot_to_home(
        robot_id=3,
        robot_config={"controlled_joints": 7},
        dataset_config={
            "reset_robot_each_episode": True,
            "home_joint_positions": [0.0] * 7,
        },
    )
    self.assertEqual(reset_joint.call_count, 7)
    for joint, call in enumerate(reset_joint.call_args_list):
        self.assertEqual(call.args, (3, joint, 0.0))
        self.assertEqual(call.kwargs["targetVelocity"], 0.0)
```

再验证 `reset_robot_each_episode=false` 时不调用 `resetJointState`。

- [ ] **Step 2: 运行复位测试确认 RED**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src python -m unittest \
  tests.simulation.test_control_arm -v
```

Expected: FAIL，`reset_robot_to_home` 尚不存在。

- [ ] **Step 3: 实现最小复位函数**

```python
def reset_robot_to_home(robot_id, robot_config, dataset_config):
    if not dataset_config["reset_robot_each_episode"]:
        return
    joint_count = robot_config["controlled_joints"]
    home = dataset_config["home_joint_positions"]
    if len(home) != joint_count:
        raise ValueError("home_joint_positions 长度必须等于 controlled_joints")
    for joint, position in enumerate(home):
        p.resetJointState(
            robot_id,
            joint,
            position,
            targetVelocity=0.0,
        )
    apply_joint_targets(robot_id, robot_config, home)
```

复位后必须同步更新电机控制目标，避免上一条 episode 的 motor target 在 settle steps
期间把关节从 home pose 拉走。测试除 `resetJointState` 外，还要断言
`apply_joint_targets(robot_id, robot_config, home)` 调用一次。

- [ ] **Step 4: 为固定 seed 和 schema 写失败测试**

扩展 `write_dataset_step()` 测试，断言写出的行包含：

```python
{
    "schema_version": "expert_v1",
    "episode_idx": 4,
    "step_idx": 24,
    "random_seed": 1004,
}
```

扩展 `write_episode_summary()` 测试，断言包含：

```python
{
    "schema_version": "expert_v1",
    "random_seed": 1004,
    "initial_ee_pos": [0.0, 0.0, 1.261],
    "initial_block_pos": [0.1, 0.4, 0.05],
}
```

给 `run_episode()` 写 mock 测试，验证调用顺序满足：

```text
random.seed(episode_seed)
reset_robot_to_home(...)
load_block(...)
sample_camera_eye(...)
```

并验证 `main()` 对 episode `i` 传入：

```python
random_seed = config["dataset"]["random_seed"] + episode_idx
```

- [ ] **Step 5: 为单 Episode 异常留证写失败测试**

mock `run_episode()` 在第一条抛出 `RuntimeError("ik failed")`、第二条正常返回，断言：

```python
self.assertEqual(run_episode.call_count, 2)
self.assertEqual(error_row["episode_idx"], 0)
self.assertEqual(error_row["random_seed"], 1000)
self.assertEqual(error_row["termination_reason"], "episode_error")
self.assertIn("ik failed", error_row["error"])
self.assertEqual(error_row["num_frames"], 0)
```

异常摘要必须符合 `expert_v1` episode schema，未知的初始或最终状态写 `None`，不得删除
异常证据或中止后续 episode。

- [ ] **Step 6: 运行 schema、seed 与异常测试确认 RED**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src python -m unittest \
  tests.simulation.test_control_arm -v
```

Expected: FAIL，写入函数签名和 `run_episode` 尚未接受新字段。

- [ ] **Step 7: 实现 episode seed、初始状态和 schema 字段**

把 `run_episode` 改为：

```python
def run_episode(
    episode_idx,
    robot_id,
    config,
    dataset_dir,
    jsonl_path,
    summary_jsonl_path,
    random_seed,
):
```

入口依次执行：

```python
random.seed(random_seed)
reset_robot_to_home(robot_id, robot_cfg, dataset_cfg)
for _ in range(task_cfg["initial_settle_steps"]):
    p.stepSimulation()
initial_ee_pos = get_link_position(
    robot_id,
    robot_cfg["ee_link_index"],
)
block_id = load_block(task_cfg)
settle_object(config, task_cfg["initial_settle_steps"])
initial_block_pos = get_object_position(block_id)
camera_eye = sample_camera_eye(camera_cfg)
```

调用 `write_dataset_step()` 时传入 `schema_version`、`episode_idx`、`step_idx`、
`random_seed`；调用 `write_episode_summary()` 时传入相同版本与 seed，并传入初始状态。

- [ ] **Step 8: 实现批次异常摘要并继续采集**

`main()` 的 episode 循环使用：

```python
try:
    run_episode(
        episode_idx,
        robot_id,
        config,
        dataset_dir,
        jsonl_path,
        summary_jsonl_path,
        random_seed,
    )
except Exception as exc:
    write_episode_error_summary(
        summary_jsonl_path=summary_jsonl_path,
        schema_version=dataset["schema_version"],
        episode_idx=episode_idx,
        random_seed=random_seed,
        error=repr(exc),
    )
```

`write_episode_error_summary()` 写入 `termination_reason="episode_error"`、`num_steps=0`、
`num_frames=0`、所有未知位置和距离为 `None`。循环继续下一个 episode，最终质量门禁
负责阻止包含错误的 pilot 扩展。

- [ ] **Step 9: 验证终止帧与动作维度契约**

新增测试断言：

```python
self.assertEqual(len(row["action"]), 9)
self.assertEqual(row["action"][-1], 1)
self.assertNotEqual(row["termination_reason"], "running")
```

Run:

```bash
conda run -n vla_env env PYTHONPATH=src python -m unittest \
  tests.simulation.test_control_arm -v
```

Expected: PASS。

- [ ] **Step 10: 提交 Task 2**

```bash
git add src/vla_project/simulation/control_arm.py \
  tests/simulation/test_control_arm.py
git commit -m "feat: make expert episodes reproducible"
```

---

### Task 3: 新增只读数据质量检查器

**Files:**
- Create: `src/vla_project/simulation/evaluate_dataset.py`
- Create: `tests/simulation/test_evaluate_dataset.py`

**Interfaces:**
- Consumes: `dataset_manifest.json`、`trajectory_expert.jsonl`、`episode_summary.jsonl` 和图片文件。
- Produces: `evaluate_dataset(dataset_dir: Path) -> dict`、`write_quality_report(dataset_dir: Path, report: dict) -> Path`、`main()`。

- [ ] **Step 1: 为合法最小数据集写失败测试**

在临时目录创建1条摘要、1条帧记录和一张224×224 JPEG：

```python
report = evaluate_dataset(dataset_dir)
self.assertEqual(report["num_episodes"], 1)
self.assertEqual(report["num_frames"], 1)
self.assertEqual(report["success_count"], 1)
self.assertEqual(report["schema_error_count"], 0)
self.assertEqual(report["missing_image_count"], 0)
self.assertEqual(report["unreadable_image_count"], 0)
self.assertEqual(report["image_size_mismatch_count"], 0)
self.assertEqual(report["orphan_image_count"], 0)
self.assertEqual(report["duplicate_step_key_count"], 0)
self.assertEqual(report["seed_error_count"], 0)
```

- [ ] **Step 2: 运行合法数据集测试确认 RED**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src python -m unittest \
  tests.simulation.test_evaluate_dataset -v
```

Expected: FAIL，模块尚不存在。

- [ ] **Step 3: 实现 JSONL 读取和基础聚合**

在 `evaluate_dataset.py` 实现：

```python
REQUIRED_FRAME_FIELDS = {
    "schema_version",
    "episode_idx",
    "step_idx",
    "random_seed",
    "image_path",
    "instruction",
    "action",
    "camera_eye",
    "block_pos",
    "target_pos",
    "ee_pos",
    "distance_to_target",
    "termination_reason",
}

REQUIRED_SUMMARY_FIELDS = {
    "schema_version",
    "episode_idx",
    "random_seed",
    "num_steps",
    "num_frames",
    "final_distance",
    "termination_reason",
    "camera_eye",
    "initial_ee_pos",
    "initial_block_pos",
    "final_block_pos",
    "final_target_pos",
    "final_ee_pos",
}
```

`evaluate_dataset()` 读取 manifest 指定的两个 JSONL，累计 episode、帧、成功率、
终止原因、距离和帧数统计。统计函数对空列表返回 `None`，不得产生 NaN。

- [ ] **Step 4: 为损坏数据写失败测试**

分别构造以下错误并断言计数：

```text
缺失字段
action 长度不是9
重复 (episode_idx, step_idx)
random_seed != manifest.random_seed + episode_idx
图片路径不存在
JPEG 无法读取
图片不是 manifest 指定尺寸
目录中存在未被 JSONL 引用的 JPEG
摘要 num_frames 与帧记录数量不一致
终止帧 action[-1] != 1
```

所有错误应在一次扫描中同时报告，不因第一条坏记录提前退出。

- [ ] **Step 5: 运行损坏数据测试确认 RED**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src python -m unittest \
  tests.simulation.test_evaluate_dataset -v
```

Expected: FAIL，对应错误计数尚未实现。

- [ ] **Step 6: 实现完整质量扫描**

报告必须包含：

```python
{
    "num_episodes": int,
    "valid_episode_count": int,
    "num_frames": int,
    "success_count": int,
    "success_rate": float,
    "termination_reason_counts": dict,
    "final_distance_stats": dict,
    "frames_per_episode_stats": dict,
    "block_position": {
        "x_min": float,
        "x_max": float,
        "y_min": float,
        "y_max": float,
        "x_bin_counts": list,
        "y_bin_counts": list,
    },
    "camera_position": {
        "min": list,
        "max": list,
    },
    "schema_error_count": int,
    "action_dim_error_count": int,
    "missing_image_count": int,
    "unreadable_image_count": int,
    "image_size_mismatch_count": int,
    "orphan_image_count": int,
    "duplicate_step_key_count": int,
    "seed_error_count": int,
    "frame_count_mismatch_count": int,
    "terminal_flag_error_count": int,
    "errors": list,
}
```

图片路径先按仓库相对路径读取；若记录只保存文件名，则相对 `dataset_dir` 解析。JPEG
集合只扫描 `dataset_dir.glob("*.jpg")`，不进入其他数据集目录。

- [ ] **Step 7: 实现质量报告落盘**

```python
def write_quality_report(dataset_dir, report):
    path = Path(dataset_dir) / "dataset_quality_report.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
```

`main()` 默认从 `sim_config.yaml` 读取 `dataset.output_dir`，执行扫描、写报告并打印
摘要；结构错误存在时退出码为1。

- [ ] **Step 8: 运行质量检查测试确认 GREEN**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src python -m unittest \
  tests.simulation.test_evaluate_dataset -v
```

Expected: PASS。

- [ ] **Step 9: 提交 Task 3**

```bash
git add src/vla_project/simulation/evaluate_dataset.py \
  tests/simulation/test_evaluate_dataset.py
git commit -m "feat: audit expert dataset quality"
```

---

### Task 4: 注册 CLI 并实现 Pilot 门禁

**Files:**
- Modify: `src/vla_project/simulation/evaluate_dataset.py`
- Modify: `tests/simulation/test_evaluate_dataset.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_package_metadata.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 3 的质量报告字段和 Task 1 manifest 的 `pilot_num_episodes`。
- Produces: `evaluate_pilot_gate(report: dict, manifest: dict) -> dict`、CLI `vla-evaluate-dataset`。

- [ ] **Step 1: 为 pilot 门禁写失败测试**

合法报告必须得到：

```python
gate = evaluate_pilot_gate(report, manifest)
self.assertTrue(gate["passed"])
self.assertEqual(gate["failed_checks"], [])
```

使用 `subTest` 逐个改变以下字段并断言失败：

```text
num_episodes != 10
success_count != 10
schema_error_count > 0
action_dim_error_count > 0
missing_image_count > 0
unreadable_image_count > 0
image_size_mismatch_count > 0
orphan_image_count > 0
duplicate_step_key_count > 0
seed_error_count > 0
frame_count_mismatch_count > 0
terminal_flag_error_count > 0
```

- [ ] **Step 2: 运行 gate 测试确认 RED**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src python -m unittest \
  tests.simulation.test_evaluate_dataset -v
```

Expected: FAIL，`evaluate_pilot_gate` 尚不存在。

- [ ] **Step 3: 实现严格 pilot gate**

```python
def evaluate_pilot_gate(report, manifest):
    expected = manifest["pilot_num_episodes"]
    checks = {
        "num_episodes": report["num_episodes"] == expected,
        "success_count": report["success_count"] == expected,
        "schema": report["schema_error_count"] == 0,
        "action_dim": report["action_dim_error_count"] == 0,
        "missing_images": report["missing_image_count"] == 0,
        "unreadable_images": report["unreadable_image_count"] == 0,
        "image_size": report["image_size_mismatch_count"] == 0,
        "orphan_images": report["orphan_image_count"] == 0,
        "duplicate_steps": report["duplicate_step_key_count"] == 0,
        "seeds": report["seed_error_count"] == 0,
        "frame_counts": report["frame_count_mismatch_count"] == 0,
        "terminal_flags": report["terminal_flag_error_count"] == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
    }
```

把结果写入质量报告的 `pilot_gate` 字段。CLI 在 `pilot_gate.passed=false` 时退出码为1。

- [ ] **Step 4: 注册 CLI 并更新元数据测试**

在 `pyproject.toml` 增加：

```toml
vla-evaluate-dataset = "vla_project.simulation.evaluate_dataset:main"
```

更新 `tests/test_package_metadata.py` 的命令集合与数量断言，并在 README 的命令表增加：

```text
vla-evaluate-dataset：扫描专家数据并生成 dataset_quality_report.json
```

- [ ] **Step 5: 运行 CLI 与元数据测试确认 GREEN**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src python -m unittest \
  tests.simulation.test_evaluate_dataset \
  tests.test_package_metadata -v
```

Expected: PASS。

- [ ] **Step 6: 提交 Task 4**

```bash
git add src/vla_project/simulation/evaluate_dataset.py \
  tests/simulation/test_evaluate_dataset.py \
  pyproject.toml \
  tests/test_package_metadata.py \
  README.md
git commit -m "feat: gate expert dataset pilots"
```

---

### Task 5: 完整验证并运行10条真实 PyBullet Pilot

**Files:**
- Modify: `docs/agent/PROJECT_OVERVIEW.md`
- Modify: `docs/agent/CURRENT_STATUS.md`
- Modify: `docs/worklog/WORKLOG.md`
- Generated: `outputs/dataset/expert_scaling_v1/`

**Interfaces:**
- Consumes: Tasks 1–4 的采集入口、质量检查 CLI 和 `sim_config.yaml` pilot 配置。
- Produces: 10条真实 pilot、`dataset_quality_report.json` 和更新后的阶段结论。

- [ ] **Step 1: 运行完整自动验证**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src python -m unittest discover -s tests -q
conda run -n vla_env python -m compileall -q src tests
git diff --check
```

Expected: 全部退出码为0。

- [ ] **Step 2: 审计旧数据未被修改**

在主工作区记录旧数据基线：

```bash
wc -l /home/pzk/vla_project/outputs/dataset/trajectory_expert.jsonl
wc -l /home/pzk/vla_project/outputs/dataset/episode_summary.jsonl
rg --files /home/pzk/vla_project/outputs/dataset -g '*.jpg' | wc -l
```

Expected:

```text
286
50
286
```

- [ ] **Step 3: 运行10条 pilot**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src vla-collect
```

Expected: 只创建 `outputs/dataset/expert_scaling_v1/`，完成10条 episode，不调用 VLM。

- [ ] **Step 4: 运行质量门禁**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m vla_project.simulation.evaluate_dataset
```

Expected:

```text
num_episodes=10
success_count=10
pilot_gate.passed=true
```

若失败，保留完整输出，使用 `superpowers:systematic-debugging` 诊断；不得自动改为300条。

- [ ] **Step 5: 把 pilot 证据安全保存到主工作区**

先确认主工作区目标不存在：

```bash
test ! -e /home/pzk/vla_project/outputs/dataset/expert_scaling_v1
```

再复制并比较文件清单：

```bash
cp -a outputs/dataset/expert_scaling_v1 \
  /home/pzk/vla_project/outputs/dataset/expert_scaling_v1
diff -qr outputs/dataset/expert_scaling_v1 \
  /home/pzk/vla_project/outputs/dataset/expert_scaling_v1
```

Expected: `test` 和 `diff` 退出码均为0，不覆盖任何已有版本化数据。

- [ ] **Step 6: 再次确认旧数据和 API 边界**

重新运行旧数据三个计数，必须仍为286、50、286。检查本轮命令不读取
`VLA_API_BASE_URL`、`VLA_API_KEY` 或 `VLA_MODEL_NAME`，代码中不得导入 VLM API
调用函数。

- [ ] **Step 7: 更新权威文档**

在 `PROJECT_OVERVIEW.md` 记录：

```text
expert_v1 schema、独立复位、固定 seed 和 pilot 质量门禁已经建立。
```

在 `CURRENT_STATUS.md` 记录真实 pilot 的 episode、帧数、成功率、距离/帧数统计和
质量报告路径。若 pilot 通过，下一步改为追加到至少300条；若失败，记录具体门禁字段。

在 `WORKLOG.md` 记录命令、配置、旧数据保护证据、真实指标和是否允许扩展。

- [ ] **Step 8: 最终验证并提交**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src python -m unittest discover -s tests -q
conda run -n vla_env python -m compileall -q src tests
git diff --check
```

Expected: 全部退出码为0。

Commit:

```bash
git add docs/agent/PROJECT_OVERVIEW.md \
  docs/agent/CURRENT_STATUS.md \
  docs/worklog/WORKLOG.md
git commit -m "docs: record expert dataset pilot"
```

生成的 `outputs/dataset/expert_scaling_v1/` 保留为本地实验证据，不提交 Git。
