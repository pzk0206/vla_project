# Expert 垂直俯视派生数据集 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从冻结的 `expert_scaling_v1` 确定性生成完整 `expert_topdown_v1`，保持300条轨迹和9,894帧动作/状态标签不变，并完成通用质量扫描与新视角可见性审计。

**Architecture:** 把现有可见性审计中的 expert episode 重放抽成共享模块；派生生成器在同一保存 step 先验证源斜视图和数值状态，再渲染固定448×448俯视图，最后原子发布独立数据集。通用质量扫描器增加显式数据目录参数；现有斜视审计和默认命令行为保持兼容。

**Tech Stack:** Python 3.10+、PyBullet、OpenCV、NumPy、PyYAML、标准库 `pathlib/hashlib/tempfile/shutil`、pytest/unittest、setuptools console scripts、JSON/JSONL/JPEG。

## Global Constraints

- 源数据固定为 `outputs/dataset/expert_scaling_v1`，不得修改、移动、删除或重新采集。
- 派生输出固定为 `outputs/dataset/expert_topdown_v1`；拒绝覆盖、互相嵌套、符号链接回源和 `outputs/dataset/` 外路径。
- 相机固定为448×448、眼位置 `[0.0, 0.4, 3.0]`、无随机扰动、`up_vector=[0,1,0]`、`fov=45`。
- 9维动作、指令、状态、seed、step 和终止标签逐值继承；重放浮点状态使用 `rtol=0, atol=1e-9`。
- 源 JPEG 只接受精确匹配或 `MAE <= 0.002` 且最大单通道误差 `<=3`。
- 源 manifest、配置、两个 JSONL 和全部源图片聚合 SHA-256 在生成前后必须一致。
- 派生图片保留448×448原图，不在本阶段训练、缩放、筛帧或联合多视角。
- 派生目录只读且不能成为 `vla-collect` 目标；失败不得发布半份数据集。
- 正式源码放 `src/vla_project/simulation/`，镜像测试放 `tests/simulation/`，生成证据只放 `outputs/dataset/`。
- 保留用户现有未跟踪文件 `docs/superpowers/plans/2026-07-19-agent-file-placement-rules.md`。

---

### Task 1: 抽取共享 expert 数据集重放契约

**Files:**
- Create: `src/vla_project/simulation/expert_dataset_replay.py`
- Create: `tests/simulation/test_expert_dataset_replay.py`
- Modify: `src/vla_project/simulation/audit_dataset_visibility.py`
- Modify: `tests/simulation/test_audit_dataset_visibility.py`

**Interfaces:**
- Consumes: 源 manifest、配置快照、帧 JSONL、episode 摘要和现有 `control_arm` 仿真函数。
- Produces: `ReplayValidationError`、`ReplayFrame`、`read_jsonl()`、`load_replay_inputs()`、`validate_episode_contract()`、`validate_replay_image()`、`replay_episode_frames()`。

- [ ] **Step 1: 写共享数据类型和契约的失败测试**

创建测试并锁定回调拿到的是动作后同一保存时刻：

```python
def test_replay_frame_carries_saved_step_state():
    frame = ReplayFrame(
        source_row={"episode_idx": 2, "step_idx": 24},
        robot_id=7,
        block_id=8,
        source_camera_eye=[1.0, 0.4, 1.6],
        target_joint_angles=[0.1] * 7,
        target_pos=[0.1, 0.4, 0.2],
        ee_pos=[0.0, 0.3, 0.4],
        block_pos=[0.1, 0.4, 0.05],
        distance_to_target=0.2,
    )
    assert frame.source_row["step_idx"] == 24
    assert frame.target_joint_angles == [0.1] * 7
```

把现有 `validate_episode_contract`、JPEG 精确匹配/严格容差/拒绝和 JSONL 加载测试迁到共享模块测试；原审计测试只保留对公开兼容导入的断言。

- [ ] **Step 2: 运行测试确认共享模块不存在**

Run: `pytest tests/simulation/test_expert_dataset_replay.py -v`

Expected: FAIL，包含 `ModuleNotFoundError`。

- [ ] **Step 3: 实现共享类型、读取和静态契约**

实现以下接口：

```python
@dataclass(frozen=True)
class ReplayFrame:
    source_row: dict
    robot_id: int
    block_id: int
    source_camera_eye: list[float]
    target_joint_angles: list[float]
    target_pos: list[float]
    ee_pos: list[float]
    block_pos: list[float]
    distance_to_target: float

def load_replay_inputs(dataset_dir: Path) -> tuple[dict, dict, list[dict], list[dict]]:
    dataset_dir = Path(dataset_dir)
    manifest = json.loads(
        (dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    config = yaml.safe_load(
        (dataset_dir / "config_snapshot.yaml").read_text(encoding="utf-8")
    )
    frames = read_jsonl(dataset_dir / manifest["jsonl_name"])
    summaries = read_jsonl(dataset_dir / manifest["summary_jsonl_name"])
    return manifest, config, frames, summaries
```

`replay_episode_frames()` 必须按 `random.seed(summary["random_seed"])`、home reset、初始 settle、红块生成、物体 settle、源相机采样和逐 step IK/控制顺序执行；只在保存 step 创建 `ReplayFrame` 并调用 `on_frame(frame)`，返回回调结果列表，最后始终断开 PyBullet。

- [ ] **Step 4: 写回调次数、step 顺序和异常清理测试**

用 mock 锁定：回调键等于源保存键、非保存 step 不回调、源相机不一致抛出 `replayed_camera_mismatch`、回调异常仍调用 `p.disconnect()`。

- [ ] **Step 5: 让现有可见性审计复用共享模块**

从 `audit_dataset_visibility.py` 删除重复的输入加载、episode 合同、JPEG 校验和主重放循环；保留兼容导入：

```python
from vla_project.simulation.expert_dataset_replay import (
    ReplayValidationError,
    load_replay_inputs as load_audit_inputs,
    replay_episode_frames,
    validate_episode_contract,
    validate_replay_image,
)
```

可见性 wrapper 的回调只负责源 RGB/segmentation 捕获、透明机器人参考像素、分组行构造；现有输出 schema 和统计不变。

- [ ] **Step 6: 运行共享与原审计测试**

Run: `pytest tests/simulation/test_expert_dataset_replay.py tests/simulation/test_audit_dataset_visibility.py -v`

Expected: PASS。

- [ ] **Step 7: 提交共享重放重构**

```bash
git add src/vla_project/simulation/expert_dataset_replay.py src/vla_project/simulation/audit_dataset_visibility.py tests/simulation/test_expert_dataset_replay.py tests/simulation/test_audit_dataset_visibility.py
git commit -m "refactor: share expert dataset replay"
```

### Task 2: 派生相机、schema 和逐帧数值对齐

**Files:**
- Create: `src/vla_project/simulation/render_expert_dataset_view.py`
- Create: `tests/simulation/test_render_expert_dataset_view.py`

**Interfaces:**
- Consumes: Task 1 的 `ReplayFrame` 和源配置/manifest/JSONL。
- Produces: `build_topdown_config()`、`validate_replay_values()`、`derive_manifest()`、`derive_frame_row()`、`derive_summary_row()`。

- [ ] **Step 1: 写固定俯视相机失败测试**

```python
def test_builds_frozen_vlm_topdown_camera(source_config, tmp_path):
    derived, eye = build_topdown_config(
        source_config, tmp_path / "expert_topdown_v1"
    )
    assert derived["camera"]["image_width"] == 448
    assert derived["camera"]["image_height"] == 448
    assert derived["camera"]["eye_offset_random_range"] == [0.0, 0.0]
    assert derived["camera"]["up_vector"] == [0, 1, 0]
    assert eye == [0.0, 0.4, 3.0]
    assert derived["dataset"]["schema_version"] == "expert_view_v1"
```

- [ ] **Step 2: 运行测试确认生成模块不存在**

Run: `pytest tests/simulation/test_render_expert_dataset_view.py::CameraContractTests -v`

Expected: FAIL，包含 `ModuleNotFoundError`。

- [ ] **Step 3: 实现相机合并与严格配置校验**

`build_topdown_config()` 深拷贝源配置，用 `vlm_evaluation.camera_override` 覆盖顶层
`camera`，校验所有冻结值，修改派生 `dataset.schema_version/output_dir`，其他物理和采集
字段保持相等，并返回固定 eye。

- [ ] **Step 4: 写动作和状态对齐边界测试**

```python
def test_replay_values_accept_exact_and_one_nanometer_tolerance(frame):
    row = dict(frame.source_row)
    row["ee_pos"] = [frame.ee_pos[0] + 1e-9, *frame.ee_pos[1:]]
    validate_replay_values(frame, row)

def test_replay_values_reject_misaligned_joint_target(frame):
    row = dict(frame.source_row)
    row["action"] = [9.0] + row["action"][1:]
    with pytest.raises(DerivationValidationError) as error:
        validate_replay_values(frame, row)
    assert error.value.reason == "replay_action_mismatch"
```

覆盖7关节目标、`ee_pos`、`block_pos`、`target_pos`、距离、NaN/Inf、夹爪值和每 episode
仅末帧 terminate。

- [ ] **Step 5: 实现派生 manifest、帧和摘要转换**

`derive_frame_row()` 只改变 `schema_version/image_path/camera_eye`，增加
`source_image_path/source_camera_eye`；`derive_summary_row()` 只改变 schema 和 camera，
增加 source camera。`derive_manifest()` 保留质量扫描字段并增加：

```python
{
    "schema_version": "expert_view_v1",
    "dataset_name": "expert_topdown_v1",
    "derived_read_only": True,
    "generation_method": "deterministic_state_replay",
    "view_name": "vlm_topdown",
    "image_width": 448,
    "image_height": 448,
}
```

- [ ] **Step 6: 运行 Task 2 测试并提交**

Run: `pytest tests/simulation/test_render_expert_dataset_view.py -v`

Expected: PASS。

```bash
git add src/vla_project/simulation/render_expert_dataset_view.py tests/simulation/test_render_expert_dataset_view.py
git commit -m "feat: define topdown dataset contract"
```

### Task 3: 路径保护、全源哈希和原子发布

**Files:**
- Modify: `src/vla_project/simulation/render_expert_dataset_view.py`
- Modify: `tests/simulation/test_render_expert_dataset_view.py`

**Interfaces:**
- Produces: `validate_dataset_paths()`、`aggregate_source_images_sha256()`、`atomic_publish_dataset()`、`write_failure_report()`。

- [ ] **Step 1: 写危险路径和空间不足失败测试**

参数化覆盖输出等于源、输出嵌套源内、源嵌套输出内、父目录符号链接回源、输出越出
`outputs/dataset/`；mock `shutil.disk_usage()` 的 free 小于 `2 * 1024**3`，断言稳定 reason。

- [ ] **Step 2: 运行安全测试确认函数缺失**

Run: `pytest tests/simulation/test_render_expert_dataset_view.py -k 'path or disk' -v`

Expected: FAIL，缺少路径或空间校验函数。

- [ ] **Step 3: 实现真实路径边界和空间门禁**

所有 containment 判断基于 `Path.resolve(strict=False)` 和 `relative_to()`，输出根基于
`repo_root / "outputs/dataset"`。输出已存在、发现同名 staging、空间不足均先失败；不得
自动删除任何目录。

- [ ] **Step 4: 写聚合图片哈希测试**

用两个内容不同的小文件验证排序不依赖输入行顺序，但修改路径或任一文件内容都会改变
摘要；缺图、重复键和路径逃逸源目录必须拒绝。

- [ ] **Step 5: 实现元数据及全部源图片哈希**

哈希输入必须包含相对路径长度、UTF-8路径和文件内容，按 `(episode_idx, step_idx)` 排序，
避免字符串拼接歧义。提供 `sha256_file(path)` 给四个源元数据文件复用。

- [ ] **Step 6: 写原子发布和失败证据测试**

在临时 `outputs/dataset` 中验证：成功只在 staging 完整后 `replace()`；回调中途异常不出现
目标目录；失败 JSON 包含 schema/reason/evidence；已有目标和 stale staging 都不删除。

- [ ] **Step 7: 实现写盘辅助并运行测试**

所有 `cv2.imwrite()` 必须检查返回值；完成后重新读图校验448×448。JSON/JSONL/YAML 在
staging 内写完、flush 并关闭后再发布；失败 JSON 使用临时文件原子替换。

Run: `pytest tests/simulation/test_render_expert_dataset_view.py -v`

Expected: PASS。

- [ ] **Step 8: 提交安全发布能力**

```bash
git add src/vla_project/simulation/render_expert_dataset_view.py tests/simulation/test_render_expert_dataset_view.py
git commit -m "feat: protect derived dataset publication"
```

### Task 4: 端到端派生生成器与 CLI

**Files:**
- Modify: `src/vla_project/simulation/render_expert_dataset_view.py`
- Modify: `tests/simulation/test_render_expert_dataset_view.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_package_metadata.py`

**Interfaces:**
- Produces: `run_generation(source_dataset, output_dir, repo_root=Path.cwd()) -> dict`、`main()`、console script `vla-render-expert-dataset-view`。

- [ ] **Step 1: 写小型两 episode orchestration 失败测试**

mock `load_replay_inputs()` 和 `replay_episode_frames()`，让每个 episode 产生两个
`ReplayFrame`；mock RGB 为448×448数组。断言生成4张图片、4行轨迹、2行摘要、固定 camera、
源字段等价、报告计数和：

```python
assert report["passed"] is True
assert report["observation_action_timing"] == (
    "post_single_sim_step_with_current_target"
)
```

- [ ] **Step 2: 运行 orchestration 测试确认失败**

Run: `pytest tests/simulation/test_render_expert_dataset_view.py -k run_generation -v`

Expected: FAIL，缺少 `run_generation`。

- [ ] **Step 3: 实现全量生成编排**

开始时计算四个元数据和图片聚合哈希；按 episode 顺序调用共享重放。每个保存 step：

```python
source_bgr = capture_rgb(source_config["camera"], frame.source_camera_eye)
replay_check = validate_replay_image(source_image_path, source_bgr)
validate_replay_values(frame, frame.source_row)
topdown_bgr = capture_rgb(topdown_config["camera"], topdown_eye)
```

写派生图和行，累计 exact/tolerance 数量；结束后复算源哈希、校验300/9,894固定数量、
字段等价、无孤立图片和固定相机，再写 config/manifest/JSONL/report 并发布。

- [ ] **Step 4: 写生成失败传播测试**

覆盖源 JPEG mismatch、数值 mismatch、JPEG 写入失败、帧数不是9,894、摘要数不是300、源
哈希变化；均断言没有目标目录且 sibling failure report reason 稳定。

- [ ] **Step 5: 实现 CLI 与包入口**

```python
parser.add_argument(
    "--source-dataset",
    default="outputs/dataset/expert_scaling_v1",
)
parser.add_argument(
    "--output-dir",
    default="outputs/dataset/expert_topdown_v1",
)
```

在 `pyproject.toml` 注册：

```toml
vla-render-expert-dataset-view = "vla_project.simulation.render_expert_dataset_view:main"
```

更新包元数据测试的精确入口集合为16个。

- [ ] **Step 6: 运行 Task 4 和包测试并提交**

Run: `pytest tests/simulation/test_render_expert_dataset_view.py tests/test_package_metadata.py -v`

Expected: PASS。

```bash
git add src/vla_project/simulation/render_expert_dataset_view.py tests/simulation/test_render_expert_dataset_view.py pyproject.toml tests/test_package_metadata.py
git commit -m "feat: render expert topdown dataset"
```

### Task 5: 让通用质量命令扫描任意数据集

**Files:**
- Modify: `src/vla_project/simulation/evaluate_dataset.py`
- Modify: `tests/simulation/test_evaluate_dataset.py`

**Interfaces:**
- Consumes: 原 `evaluate_dataset(dataset_dir)` 和 `write_quality_report()`。
- Produces: `parse_args(argv=None)`、`main(argv=None)`，可选 `--dataset-dir`。

- [ ] **Step 1: 写默认路径和显式覆盖失败测试**

mock `evaluate_dataset`，分别调用 `main([])` 与
`main(["--dataset-dir", "outputs/dataset/expert_topdown_v1"])`；前者断言仍读取
`sim_config.yaml` 的旧目录，后者断言不受当前 config 输出路径影响。

- [ ] **Step 2: 运行 CLI 测试确认参数不受支持**

Run: `pytest tests/simulation/test_evaluate_dataset.py -k dataset_dir -v`

Expected: FAIL，`main` 不接受 argv 或显式目录未生效。

- [ ] **Step 3: 实现向后兼容的 argparse**

`--dataset-dir` 默认 `None`；仅在缺省时加载 `CONFIG_PATH`。保持报告写入被扫描数据集目录，
失败仍以退出码1表示。

- [ ] **Step 4: 运行质量扫描测试并提交**

Run: `pytest tests/simulation/test_evaluate_dataset.py -v`

Expected: PASS。

```bash
git add src/vla_project/simulation/evaluate_dataset.py tests/simulation/test_evaluate_dataset.py
git commit -m "feat: select dataset quality input"
```

### Task 6: 全量回归验证和真实俯视数据生成

**Files:**
- Generated: `outputs/dataset/expert_topdown_v1/`

**Interfaces:**
- Consumes: Tasks 1–5 的命令。
- Produces: 完整派生数据、`view_generation_report.json`、`dataset_quality_report.json`、`visibility_audit_v1/`。

- [ ] **Step 1: 运行定向测试与完整测试**

Run: `pytest tests/simulation/test_expert_dataset_replay.py tests/simulation/test_render_expert_dataset_view.py tests/simulation/test_audit_dataset_visibility.py tests/simulation/test_evaluate_dataset.py tests/test_package_metadata.py -v`

Expected: PASS。

Run: `pytest -q`

Expected: 所有测试通过，无失败或跳过增加。

- [ ] **Step 2: 运行静态验证**

Run: `python -m compileall -q src tests`

Expected: exit 0。

Run: `git diff --check`

Expected: 无输出，exit 0。

- [ ] **Step 3: 生成真实300条俯视派生数据**

Run: `vla-render-expert-dataset-view --source-dataset outputs/dataset/expert_scaling_v1 --output-dir outputs/dataset/expert_topdown_v1`

Expected: exit 0；报告 `passed=true`、300 episodes、9,894 frames/images、固定 camera、源哈希前后一致。

- [ ] **Step 4: 执行通用质量扫描**

Run: `vla-evaluate-dataset --dataset-dir outputs/dataset/expert_topdown_v1`

Expected: exit 0；`dataset_quality_report.json` 的 `active_gate=scale`、`passed=true`、所有完整性错误为0。

- [ ] **Step 5: 执行新视角可见性审计**

Run: `vla-audit-dataset-visibility --dataset-dir outputs/dataset/expert_topdown_v1`

Expected: exit 0；9,894帧全部有标签，`replay_validation.passed=true`。只记录真实
clear/partial/severe 分布，不预设比例。

- [ ] **Step 6: 对比源与派生关键证据**

运行只读检查，断言 frame key、动作、状态和终止标签逐值相等；读取三个报告打印派生计数、
质量门禁、俯视可见性分布和源/派生非 clear 差异。若任一硬门禁失败，停止文档结论并保留
失败证据，不进入训练。

### Task 7: 更新结构、状态和实验证据

**Files:**
- Modify: `README.md`
- Modify: `docs/agent/PROJECT_STRUCTURE.md`
- Modify: `docs/agent/PROJECT_OVERVIEW.md`
- Modify: `docs/agent/CURRENT_STATUS.md`
- Modify: `docs/worklog/WORKLOG.md`

**Interfaces:**
- Consumes: Task 6 的真实报告。
- Produces: 当前架构、真实指标、恢复点和下一步行为克隆入口。

- [ ] **Step 1: 更新命令和文件职责**

README 增加派生生成、显式质量扫描和可见性审计命令；PROJECT_STRUCTURE 增加
`expert_dataset_replay.py`、`render_expert_dataset_view.py`、镜像测试、16个命令和
`expert_topdown_v1` 输出职责。

- [ ] **Step 2: 只用真实报告更新状态**

PROJECT_OVERVIEW/CURRENT_STATUS/WORKLOG 记录：源斜视数据保持不变、派生数据实际 episode/
frame/图片计数、质量门禁、俯视 clear/partial/severe、最长连续非 clear 和相对源视角差异。
下一步恢复点改为基于俯视真实分布设计行为克隆；不得提前声称模型训练完成。

- [ ] **Step 3: 运行最终验证**

Run: `pytest -q`

Expected: 全部测试通过。

Run: `python -m compileall -q src tests`

Expected: exit 0。

Run: `git diff --check`

Expected: 无输出，exit 0。

- [ ] **Step 4: 检查工作区范围并提交**

Run: `git status --short`

Expected: 只包含本计划文档更新；本地 `outputs/` 被忽略，用户未跟踪计划文件未进入提交。

```bash
git add README.md docs/agent/PROJECT_STRUCTURE.md docs/agent/PROJECT_OVERVIEW.md docs/agent/CURRENT_STATUS.md docs/worklog/WORKLOG.md
git commit -m "docs: record topdown expert dataset"
```

- [ ] **Step 5: 完成前证据复核**

读取最新 `view_generation_report.json`、`dataset_quality_report.json` 和
`visibility_audit_summary.json`，核对测试输出与 `git log --oneline` 后才能报告完成。
