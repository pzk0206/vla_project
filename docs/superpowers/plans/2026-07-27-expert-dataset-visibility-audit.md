# Expert v1 可见性审计实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对现有300条、9,894帧 `expert_v1` 数据执行逐帧确定性重放，只有重放 JPEG 与原图完全一致时才使用 PyBullet segmentation 生成红块可见率标签和遮挡分布报告。

**Architecture:** 新建一个 simulation 领域审计模块，纯函数负责 segmentation 解码、图像一致性和汇总，重放函数负责按 episode 恢复仿真并返回逐帧标签，顶层 runner 负责输入契约和原子发布。原始数据完全只读；失败只发布失败证据，不发布可信标签。

**Tech Stack:** Python 3、PyBullet、OpenCV、NumPy、PyYAML、`unittest`、JSON/JSONL

## Global Constraints

- 输入固定为 `outputs/dataset/expert_scaling_v1/` 内的 manifest、配置快照、两份 JSONL 和 JPEG。
- 仿真参数必须来自数据集内的 `config_snapshot.yaml`，不得使用当前 `sim_config.yaml` 代替。
- 不修改、删除、移动、覆盖或筛选任何原始训练图片和 JSONL。
- 重放 JPEG 重新编码解码后必须与原图逐像素完全一致，否则对应标签不可信并使整批失败。
- `severe < 0.25`、`partial < 0.75`、`clear >= 0.75`，必须复用现有 `classify_visibility()`。
- 成功时只发布 `frame_visibility.jsonl` 和 `visibility_audit_summary.json`；失败时只发布 `visibility_audit_failure.json`。
- 输出写入 `outputs/dataset/expert_scaling_v1/visibility_audit_v1/`，不得提交 Git。

---

### Task 1: 实现可见率、JPEG 一致性和汇总纯函数

**Files:**
- Create: `src/vla_project/simulation/audit_dataset_visibility.py`
- Create: `tests/simulation/test_audit_dataset_visibility.py`

**Interfaces:**
- Consumes: NumPy segmentation、OpenCV BGR 图像、已有 `classify_visibility(ratio)`。
- Produces: `count_block_pixels(segmentation, block_id) -> int`、`validate_replay_image(original_path, replay_bgr) -> dict`、`build_visibility_rows(frame_observations, reference_pixels) -> list[dict]`、`summarize_visibility(rows) -> dict`。

- [ ] **Step 1: 写 segmentation 解码和计数失败测试**

```python
class VisibilityMathTests(unittest.TestCase):
    def test_counts_block_object_id_while_ignoring_link_bits_and_background(self):
        block_id = 2
        segmentation = np.array(
            [
                [-1, block_id],
                [block_id | (3 << 24), 7],
            ],
            dtype=np.int64,
        )

        self.assertEqual(count_block_pixels(segmentation, block_id), 2)
```

- [ ] **Step 2: 运行测试确认 RED**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest \
  tests.simulation.test_audit_dataset_visibility.VisibilityMathTests.test_counts_block_object_id_while_ignoring_link_bits_and_background
```

Expected: 导入失败，因为审计模块尚不存在。

- [ ] **Step 3: 实现 object id 解码与计数**

```python
OBJECT_ID_MASK = (1 << 24) - 1


def count_block_pixels(segmentation, block_id):
    if isinstance(block_id, bool) or not isinstance(block_id, int):
        raise ValueError("block_id 必须是整数")
    object_ids = np.asarray(segmentation, dtype=np.int64) & OBJECT_ID_MASK
    return int(np.count_nonzero(object_ids == block_id))
```

- [ ] **Step 4: 运行测试确认 GREEN**

Run: Step 2 的相同命令。

Expected: 1 test passes。

- [ ] **Step 5: 写 JPEG 精确重放测试**

```python
class ReplayImageTests(unittest.TestCase):
    def test_accepts_same_image_after_default_jpeg_roundtrip(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[8:24, 8:24] = [0, 0, 255]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "frame.jpg"
            self.assertTrue(cv2.imwrite(str(path), image))

            result = validate_replay_image(path, image)

        self.assertTrue(result["replay_exact_match"])
        self.assertEqual(result["replay_pixel_mae"], 0.0)

    def test_rejects_replay_with_changed_pixels(self):
        original = np.zeros((32, 32, 3), dtype=np.uint8)
        replay = np.full((32, 32, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "frame.jpg"
            self.assertTrue(cv2.imwrite(str(path), original))

            with self.assertRaisesRegex(
                ReplayValidationError,
                "replay_image_mismatch",
            ):
                validate_replay_image(path, replay)
```

- [ ] **Step 6: 运行 JPEG 测试确认 RED**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest \
  tests.simulation.test_audit_dataset_visibility.ReplayImageTests
```

Expected: `validate_replay_image` 或 `ReplayValidationError` 不存在。

- [ ] **Step 7: 实现 JPEG 精确比较**

```python
class ReplayValidationError(RuntimeError):
    def __init__(self, reason, **evidence):
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence


def validate_replay_image(original_path, replay_bgr):
    original = cv2.imread(str(original_path))
    if original is None:
        raise ReplayValidationError(
            "unreadable_original_image",
            image_path=str(original_path),
        )
    encoded, buffer = cv2.imencode(".jpg", replay_bgr)
    if not encoded:
        raise ReplayValidationError("replay_jpeg_encode_failed")
    roundtrip = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if roundtrip is None or roundtrip.shape != original.shape:
        raise ReplayValidationError(
            "replay_image_shape_mismatch",
            expected_shape=list(original.shape),
            actual_shape=None if roundtrip is None else list(roundtrip.shape),
        )
    diff = np.abs(
        roundtrip.astype(np.int16) - original.astype(np.int16)
    )
    mae = float(np.mean(diff))
    if not np.array_equal(roundtrip, original):
        raise ReplayValidationError(
            "replay_image_mismatch",
            image_path=str(original_path),
            replay_pixel_mae=mae,
            max_pixel_error=int(diff.max()),
        )
    return {
        "replay_exact_match": True,
        "replay_pixel_mae": mae,
    }
```

- [ ] **Step 8: 运行 JPEG 测试确认 GREEN**

Run: Step 6 的相同命令。

Expected: 2 tests pass。

- [ ] **Step 9: 写可见率行构造边界测试**

```python
def observation(step_idx, pixels):
    return {
        "schema_version": "visibility_audit_v1",
        "episode_idx": 3,
        "random_seed": 1003,
        "step_idx": step_idx,
        "image_path": f"ep_3_step_{step_idx}.jpg",
        "visible_block_pixels": pixels,
        "replay_pixel_mae": 0.0,
        "replay_exact_match": True,
    }


class VisibilityRowTests(unittest.TestCase):
    def test_builds_rows_with_existing_visibility_boundaries(self):
        rows = build_visibility_rows(
            [
                observation(0, 0),
                observation(1, 25),
                observation(2, 75),
                observation(3, 100),
            ],
            reference_pixels=100,
        )

        self.assertEqual(
            [row["visibility_group"] for row in rows],
            ["severe", "partial", "clear", "clear"],
        )
        self.assertEqual(rows[1]["block_visibility_ratio"], 0.25)

    def test_rejects_nonpositive_reference_or_visible_above_reference(self):
        with self.assertRaises(ValueError):
            build_visibility_rows([observation(0, 0)], reference_pixels=0)
        with self.assertRaises(ValueError):
            build_visibility_rows([observation(0, 101)], reference_pixels=100)
```

- [ ] **Step 10: 运行行构造测试确认 RED**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest \
  tests.simulation.test_audit_dataset_visibility.VisibilityRowTests
```

Expected: `build_visibility_rows` 不存在。

- [ ] **Step 11: 实现逐帧比率和分组**

```python
def build_visibility_rows(frame_observations, reference_pixels):
    if (
        isinstance(reference_pixels, bool)
        or not isinstance(reference_pixels, int)
        or reference_pixels <= 0
    ):
        raise ValueError("reference_pixels 必须是正整数")
    rows = []
    for observation in frame_observations:
        visible = observation["visible_block_pixels"]
        if (
            isinstance(visible, bool)
            or not isinstance(visible, int)
            or not 0 <= visible <= reference_pixels
        ):
            raise ValueError("visible_block_pixels 必须位于参考范围内")
        ratio = visible / reference_pixels
        rows.append(
            {
                **observation,
                "reference_block_pixels": reference_pixels,
                "block_visibility_ratio": ratio,
                "visibility_group": classify_visibility(ratio),
            }
        )
    return rows
```

- [ ] **Step 12: 运行行构造测试确认 GREEN**

Run: Step 10 的相同命令。

Expected: 2 tests pass。

- [ ] **Step 13: 写汇总与连续遮挡测试**

```python
class VisibilitySummaryTests(unittest.TestCase):
    def test_summarizes_groups_endpoints_and_longest_nonclear_run(self):
        rows = [
            {**observation(0, 100), "block_visibility_ratio": 1.0,
             "visibility_group": "clear", "reference_block_pixels": 100},
            {**observation(24, 70), "block_visibility_ratio": 0.7,
             "visibility_group": "partial", "reference_block_pixels": 100},
            {**observation(48, 20), "block_visibility_ratio": 0.2,
             "visibility_group": "severe", "reference_block_pixels": 100},
            {**observation(72, 80), "block_visibility_ratio": 0.8,
             "visibility_group": "clear", "reference_block_pixels": 100},
        ]

        summary = summarize_visibility(rows)

        self.assertEqual(
            summary["visibility_group_counts"],
            {"clear": 2, "partial": 1, "severe": 1},
        )
        self.assertEqual(summary["episodes_with_nonclear"], 1)
        self.assertEqual(summary["episodes_with_severe"], 1)
        self.assertEqual(summary["initial_visibility_group_counts"], {"clear": 1})
        self.assertEqual(summary["terminal_visibility_group_counts"], {"clear": 1})
        self.assertEqual(summary["longest_nonclear_saved_frame_run"], 2)
        self.assertEqual(
            summary["longest_nonclear_run"],
            {"episode_idx": 3, "start_step": 24, "end_step": 48, "num_frames": 2},
        )
```

- [ ] **Step 14: 运行汇总测试确认 RED**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest \
  tests.simulation.test_audit_dataset_visibility.VisibilitySummaryTests
```

Expected: `summarize_visibility` 不存在。

- [ ] **Step 15: 实现汇总**

实现时：

```python
def longest_nonclear_run(episode_idx, episode_rows):
    longest = None
    run_start = None
    run_rows = []
    for row in episode_rows + [None]:
        if row is not None and row["visibility_group"] != "clear":
            if run_start is None:
                run_start = row["step_idx"]
            run_rows.append(row)
            continue
        if run_rows:
            candidate = {
                "episode_idx": episode_idx,
                "start_step": run_start,
                "end_step": run_rows[-1]["step_idx"],
                "num_frames": len(run_rows),
            }
            if longest is None or candidate["num_frames"] > longest["num_frames"]:
                longest = candidate
        run_start = None
        run_rows = []
    return longest


def summarize_visibility(rows):
    rows = sorted(
        rows,
        key=lambda row: (row["episode_idx"], row["step_idx"]),
    )
    if not rows:
        raise ValueError("可见性行不能为空")
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[row["episode_idx"]].append(row)

    episode_longest_runs = {
        episode_idx: longest_nonclear_run(episode_idx, episode_rows)
        for episode_idx, episode_rows in by_episode.items()
    }
    candidates = [
        run for run in episode_longest_runs.values() if run is not None
    ]
    longest = max(candidates, key=lambda run: run["num_frames"], default=None)

    ratios = [row["block_visibility_ratio"] for row in rows]
    group_counts = Counter(row["visibility_group"] for row in rows)
    initial_counts = Counter(
        episode_rows[0]["visibility_group"]
        for episode_rows in by_episode.values()
    )
    terminal_counts = Counter(
        episode_rows[-1]["visibility_group"]
        for episode_rows in by_episode.values()
    )
    initial_ratios = [
        episode_rows[0]["block_visibility_ratio"]
        for episode_rows in by_episode.values()
    ]
    terminal_ratios = [
        episode_rows[-1]["block_visibility_ratio"]
        for episode_rows in by_episode.values()
    ]
    per_episode = []
    for episode_idx, episode_rows in by_episode.items():
        episode_counts = Counter(
            row["visibility_group"] for row in episode_rows
        )
        episode_ratios = [
            row["block_visibility_ratio"] for row in episode_rows
        ]
        episode_longest = episode_longest_runs[episode_idx]
        per_episode.append(
            {
                "episode_idx": episode_idx,
                "num_frames": len(episode_rows),
                "visibility_group_counts": {
                    group: episode_counts[group]
                    for group in ("clear", "partial", "severe")
                },
                "visibility_ratio_stats": numeric_stats(
                    episode_ratios
                ),
                "initial_visibility_group": (
                    episode_rows[0]["visibility_group"]
                ),
                "terminal_visibility_group": (
                    episode_rows[-1]["visibility_group"]
                ),
                "longest_nonclear_run": episode_longest,
            }
        )
    lowest = sorted(
        rows,
        key=lambda row: row["block_visibility_ratio"],
    )[:10]
    return {
        "num_episodes": len(by_episode),
        "num_frames": len(rows),
        "replay_exact_match_count": sum(
            row["replay_exact_match"] for row in rows
        ),
        "replay_mismatch_count": 0,
        "visibility_group_counts": {
            group: group_counts[group]
            for group in ("clear", "partial", "severe")
        },
        "visibility_group_rates": {
            group: group_counts[group] / len(rows)
            for group in ("clear", "partial", "severe")
        },
        "episodes_with_nonclear": sum(
            any(row["visibility_group"] != "clear" for row in episode_rows)
            for episode_rows in by_episode.values()
        ),
        "episodes_with_severe": sum(
            any(row["visibility_group"] == "severe" for row in episode_rows)
            for episode_rows in by_episode.values()
        ),
        "initial_visibility_group_counts": dict(initial_counts),
        "terminal_visibility_group_counts": dict(terminal_counts),
        "initial_visibility_ratio_stats": numeric_stats(initial_ratios),
        "terminal_visibility_ratio_stats": numeric_stats(terminal_ratios),
        "visibility_ratio_stats": numeric_stats(ratios),
        "longest_nonclear_saved_frame_run": (
            0 if longest is None else longest["num_frames"]
        ),
        "longest_nonclear_run": longest,
        "per_episode": per_episode,
        "lowest_visibility_frames": [
            {
                "episode_idx": row["episode_idx"],
                "step_idx": row["step_idx"],
                "image_path": row["image_path"],
                "block_visibility_ratio": row["block_visibility_ratio"],
                "visibility_group": row["visibility_group"],
            }
            for row in lowest
        ],
    }
```

同文件增加：

```python
def numeric_stats(values):
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None}
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }
```

- [ ] **Step 16: 运行 Task 1 全部测试**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest tests.simulation.test_audit_dataset_visibility
```

Expected: Task 1 tests pass。

- [ ] **Step 17: 提交纯函数**

```bash
git add \
  src/vla_project/simulation/audit_dataset_visibility.py \
  tests/simulation/test_audit_dataset_visibility.py
git commit -m "feat: compute dataset visibility metrics"
```

---

### Task 2: 实现 episode 确定性重放

**Files:**
- Modify: `src/vla_project/simulation/audit_dataset_visibility.py`
- Modify: `tests/simulation/test_audit_dataset_visibility.py`

**Interfaces:**
- Consumes: 完整配置、manifest、单条 episode summary、该 episode 的 frame rows。
- Produces: `replay_episode(config, manifest, summary, frame_rows) -> list[dict]`；成功时每个输入 frame 恰好对应一个可信可见率行。

- [ ] **Step 1: 写输入契约失败测试**

```python
class ReplayContractTests(unittest.TestCase):
    def test_rejects_seed_camera_and_duplicate_step_mismatches(self):
        manifest = {"random_seed": 1000}
        summary = {"episode_idx": 3, "random_seed": 1003}
        valid = [
            {
                "episode_idx": 3,
                "random_seed": 1003,
                "step_idx": 0,
                "camera_eye": [1.0, 0.4, 1.6],
                "image_path": "ep_3_step_0.jpg",
            }
        ]
        validate_episode_contract(manifest, summary, valid)

        with self.assertRaisesRegex(ReplayValidationError, "seed_mismatch"):
            validate_episode_contract(
                manifest,
                {**summary, "random_seed": 999},
                valid,
            )
        with self.assertRaisesRegex(ReplayValidationError, "duplicate_step"):
            validate_episode_contract(manifest, summary, valid + valid)
        with self.assertRaisesRegex(ReplayValidationError, "camera_mismatch"):
            validate_episode_contract(
                manifest,
                summary,
                valid + [
                    {
                        **valid[0],
                        "step_idx": 24,
                        "camera_eye": [1.1, 0.4, 1.6],
                    }
                ],
            )
```

- [ ] **Step 2: 运行契约测试确认 RED**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest \
  tests.simulation.test_audit_dataset_visibility.ReplayContractTests
```

Expected: `validate_episode_contract` 不存在。

- [ ] **Step 3: 实现输入契约**

```python
def validate_episode_contract(manifest, summary, frame_rows):
    episode_idx = summary["episode_idx"]
    expected_seed = manifest["random_seed"] + episode_idx
    if summary["random_seed"] != expected_seed:
        raise ReplayValidationError(
            "seed_mismatch",
            episode_idx=episode_idx,
            expected_seed=expected_seed,
            actual_seed=summary["random_seed"],
        )
    steps = [row["step_idx"] for row in frame_rows]
    if len(steps) != len(set(steps)):
        raise ReplayValidationError(
            "duplicate_step",
            episode_idx=episode_idx,
        )
    if not frame_rows:
        raise ReplayValidationError(
            "missing_episode_frames",
            episode_idx=episode_idx,
        )
    camera_eyes = {
        tuple(row["camera_eye"])
        for row in frame_rows
    }
    if len(camera_eyes) != 1:
        raise ReplayValidationError(
            "camera_mismatch",
            episode_idx=episode_idx,
        )
    for row in frame_rows:
        if row["episode_idx"] != episode_idx:
            raise ReplayValidationError("episode_mismatch")
        if row["random_seed"] != expected_seed:
            raise ReplayValidationError("frame_seed_mismatch")
```

- [ ] **Step 4: 运行契约测试确认 GREEN**

Run: Step 2 的相同命令。

Expected: 1 test passes。

- [ ] **Step 5: 写小型真实重放集成测试**

测试使用临时目录和一条只运行到 step 0 的真实 DIRECT episode：

```python
class ReplayIntegrationTests(unittest.TestCase):
    def test_replays_real_saved_frame_and_recovers_segmentation(self):
        config = load_config("sim_config.yaml")
        config["connection_mode"] = "DIRECT"
        config["dataset"]["max_steps_per_episode"] = 1
        episode_idx = 0
        seed = config["dataset"]["random_seed"]

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            random.seed(seed)
            connect_physics("DIRECT")
            try:
                _, robot_id = setup_world(config)
                reset_robot_to_home(
                    robot_id,
                    config["robot"],
                    config["dataset"],
                )
                for _ in range(config["task"]["initial_settle_steps"]):
                    p.stepSimulation()
                block_id = load_block(config["task"])
                settle_object(
                    config,
                    config["task"]["initial_settle_steps"],
                )
                camera_eye = sample_camera_eye(config["camera"])
                target = get_hover_target(
                    block_id,
                    config["task"]["hover_height"],
                )
                joints = calculate_target_joints(
                    robot_id,
                    config["robot"],
                    target,
                )
                apply_joint_targets(robot_id, config["robot"], joints)
                p.stepSimulation()
                image = capture_rgb(config["camera"], camera_eye)
                image_path = dataset_dir / "ep_0_step_0.jpg"
                self.assertTrue(cv2.imwrite(str(image_path), image))
            finally:
                p.disconnect()

            rows = replay_episode(
                config,
                {"random_seed": seed},
                {"episode_idx": episode_idx, "random_seed": seed},
                [{
                    "episode_idx": episode_idx,
                    "random_seed": seed,
                    "step_idx": 0,
                    "camera_eye": camera_eye,
                    "image_path": str(image_path),
                }],
            )

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["replay_exact_match"])
        self.assertGreater(rows[0]["reference_block_pixels"], 0)
        self.assertGreater(rows[0]["visible_block_pixels"], 0)
```

- [ ] **Step 6: 运行集成测试确认 RED**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest \
  tests.simulation.test_audit_dataset_visibility.ReplayIntegrationTests
```

Expected: `replay_episode` 不存在。

- [ ] **Step 7: 实现确定性重放**

实现 `replay_episode()`：

```python
def replay_episode(config, manifest, summary, frame_rows):
    frame_rows = sorted(frame_rows, key=lambda row: row["step_idx"])
    validate_episode_contract(manifest, summary, frame_rows)
    episode_idx = summary["episode_idx"]
    random.seed(summary["random_seed"])
    connect_physics("DIRECT")
    observations = []
    try:
        _, robot_id = setup_world(config)
        reset_robot_to_home(robot_id, config["robot"], config["dataset"])
        for _ in range(config["task"]["initial_settle_steps"]):
            p.stepSimulation()
        block_id = load_block(config["task"])
        settle_object(config, config["task"]["initial_settle_steps"])
        camera_eye = sample_camera_eye(config["camera"])
        expected_eye = frame_rows[0]["camera_eye"]
        if camera_eye != expected_eye:
            raise ReplayValidationError(
                "replayed_camera_mismatch",
                episode_idx=episode_idx,
                expected_camera_eye=expected_eye,
                actual_camera_eye=camera_eye,
            )

        rows_by_step = {row["step_idx"]: row for row in frame_rows}
        for step_idx in range(frame_rows[-1]["step_idx"] + 1):
            target = get_hover_target(
                block_id,
                config["task"]["hover_height"],
            )
            joints = calculate_target_joints(
                robot_id,
                config["robot"],
                target,
            )
            apply_joint_targets(robot_id, config["robot"], joints)
            p.stepSimulation()
            if step_idx not in rows_by_step:
                continue
            source = rows_by_step[step_idx]
            replay_bgr, segmentation = capture_rgb_and_segmentation(
                config["camera"],
                camera_eye,
            )
            replay_check = validate_replay_image(
                source["image_path"],
                replay_bgr,
            )
            observations.append(
                {
                    "schema_version": "visibility_audit_v1",
                    "episode_idx": episode_idx,
                    "random_seed": summary["random_seed"],
                    "step_idx": step_idx,
                    "image_path": source["image_path"],
                    "visible_block_pixels": count_block_pixels(
                        segmentation,
                        block_id,
                    ),
                    **replay_check,
                }
            )

        p.removeBody(robot_id)
        _, reference_segmentation = capture_rgb_and_segmentation(
            config["camera"],
            camera_eye,
        )
        reference_pixels = count_block_pixels(
            reference_segmentation,
            block_id,
        )
        return build_visibility_rows(observations, reference_pixels)
    except ReplayValidationError as exc:
        exc.evidence.setdefault("episode_idx", episode_idx)
        raise
    finally:
        p.disconnect()
```

- [ ] **Step 8: 运行集成测试确认 GREEN**

Run: Step 6 的相同命令。

Expected: 1 test passes。

- [ ] **Step 9: 运行审计模块全部测试并提交**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest tests.simulation.test_audit_dataset_visibility
```

Expected: all audit tests pass。

Commit:

```bash
git add \
  src/vla_project/simulation/audit_dataset_visibility.py \
  tests/simulation/test_audit_dataset_visibility.py
git commit -m "feat: replay expert episodes for visibility"
```

---

### Task 3: 实现全数据 runner、原子输出和 CLI

**Files:**
- Modify: `src/vla_project/simulation/audit_dataset_visibility.py`
- Modify: `tests/simulation/test_audit_dataset_visibility.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_package_metadata.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 2 的 `replay_episode()` 和 Task 1 的 `summarize_visibility()`。
- Produces: `run_visibility_audit(dataset_dir, output_dir) -> dict`、CLI `vla-audit-dataset-visibility`、成功/失败输出契约。

- [ ] **Step 1: 写输入索引和失败不发布标签测试**

```python
class AuditRunnerTests(unittest.TestCase):
    def test_failure_writes_only_failure_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            output_dir = dataset_dir / "visibility_audit_v1"
            dataset_dir.mkdir()
            write_minimal_dataset_inputs(dataset_dir)

            with patch(
                "vla_project.simulation.audit_dataset_visibility.replay_episode",
                side_effect=ReplayValidationError(
                    "replay_image_mismatch",
                    episode_idx=0,
                    step_idx=0,
                ),
            ):
                with self.assertRaises(ReplayValidationError):
                    run_visibility_audit(dataset_dir, output_dir)

            self.assertTrue(
                (output_dir / "visibility_audit_failure.json").is_file()
            )
            self.assertFalse(
                (output_dir / "frame_visibility.jsonl").exists()
            )
            self.assertFalse(
                (output_dir / "visibility_audit_summary.json").exists()
            )
```

测试辅助函数使用真实文件格式：

```python
def write_minimal_dataset_inputs(dataset_dir):
    manifest = {
        "schema_version": "expert_v1",
        "random_seed": 1000,
        "jsonl_name": "trajectory_expert.jsonl",
        "summary_jsonl_name": "episode_summary.jsonl",
    }
    frame = {
        "episode_idx": 0,
        "random_seed": 1000,
        "step_idx": 0,
        "camera_eye": [1.0, 0.4, 1.6],
        "image_path": str(dataset_dir / "ep_0_step_0.jpg"),
    }
    summary = {
        "episode_idx": 0,
        "random_seed": 1000,
    }
    write_json(dataset_dir / "dataset_manifest.json", manifest)
    (dataset_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump({"connection_mode": "DIRECT"}),
        encoding="utf-8",
    )
    (dataset_dir / "trajectory_expert.jsonl").write_text(
        json.dumps(frame) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "episode_summary.jsonl").write_text(
        json.dumps(summary) + "\n",
        encoding="utf-8",
    )
    cv2.imwrite(
        frame["image_path"],
        np.zeros((32, 32, 3), dtype=np.uint8),
    )
```

mock 只替代 Task 2 已由真实集成测试覆盖的长仿真边界。

- [ ] **Step 2: 运行失败发布测试确认 RED**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest \
  tests.simulation.test_audit_dataset_visibility.AuditRunnerTests.test_failure_writes_only_failure_evidence
```

Expected: `run_visibility_audit` 不存在。

- [ ] **Step 3: 实现输入加载、失败证据和原子成功发布**

实现：

```python
def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_audit_inputs(dataset_dir):
    dataset_dir = Path(dataset_dir)
    with (dataset_dir / "dataset_manifest.json").open(
        "r", encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)
    with (dataset_dir / "config_snapshot.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        config = yaml.safe_load(handle)
    frames = read_jsonl(dataset_dir / manifest["jsonl_name"])
    summaries = read_jsonl(
        dataset_dir / manifest["summary_jsonl_name"]
    )
    return manifest, config, frames, summaries


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_visibility_audit(dataset_dir, output_dir=None):
    dataset_dir = Path(dataset_dir)
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else dataset_dir / "visibility_audit_v1"
    )
    if output_dir.exists():
        raise FileExistsError(f"审计输出目录已存在: {output_dir}")
    try:
        manifest, config, frames, summaries = load_audit_inputs(dataset_dir)
        indexed_frames = defaultdict(list)
        for row in frames:
            indexed_frames[row["episode_idx"]].append(row)
        if len({row["episode_idx"] for row in summaries}) != len(summaries):
            raise ReplayValidationError("duplicate_episode_summary")

        all_rows = []
        for summary in sorted(
            summaries,
            key=lambda row: row["episode_idx"],
        ):
            all_rows.extend(
                replay_episode(
                    config,
                    manifest,
                    summary,
                    indexed_frames[summary["episode_idx"]],
                )
            )
        if len(all_rows) != len(frames):
            raise ReplayValidationError(
                "unconsumed_frame_rows",
                expected_frames=len(frames),
                replayed_frames=len(all_rows),
            )
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=False)
        evidence = {
            "schema_version": "visibility_audit_v1",
            "passed": False,
            "reason": (
                exc.reason
                if isinstance(exc, ReplayValidationError)
                else type(exc).__name__
            ),
            "error": repr(exc),
            "evidence": (
                exc.evidence
                if isinstance(exc, ReplayValidationError)
                else {}
            ),
        }
        write_json(output_dir / "visibility_audit_failure.json", evidence)
        raise

    summary = {
        "schema_version": "visibility_audit_v1",
        "dataset_dir": str(dataset_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "replay_validation": {"passed": True},
        **summarize_visibility(all_rows),
    }
    with tempfile.TemporaryDirectory(dir=dataset_dir) as temp_dir:
        staging = Path(temp_dir) / output_dir.name
        staging.mkdir()
        with (staging / "frame_visibility.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in all_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(
            staging / "visibility_audit_summary.json",
            summary,
        )
        staging.replace(output_dir)
    return summary
```

- [ ] **Step 4: 运行失败发布测试确认 GREEN**

Run: Step 2 的相同命令。

Expected: 1 test passes。

- [ ] **Step 5: 写成功原子发布测试**

```python
def test_success_publishes_complete_rows_and_summary(self):
    replayed = [{
        **observation(0, 100),
        "reference_block_pixels": 100,
        "block_visibility_ratio": 1.0,
        "visibility_group": "clear",
    }]
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_dir = Path(temp_dir) / "dataset"
        output_dir = dataset_dir / "visibility_audit_v1"
        dataset_dir.mkdir()
        write_minimal_dataset_inputs(dataset_dir)
        with patch(
            "vla_project.simulation.audit_dataset_visibility.replay_episode",
            return_value=replayed,
        ):
            summary = run_visibility_audit(dataset_dir, output_dir)

        saved_rows = read_jsonl(
            output_dir / "frame_visibility.jsonl"
        )
        saved_summary = json.loads(
            (output_dir / "visibility_audit_summary.json").read_text(
                encoding="utf-8"
            )
        )

    self.assertEqual(saved_rows, replayed)
    self.assertTrue(summary["replay_validation"]["passed"])
    self.assertEqual(saved_summary, summary)
```

- [ ] **Step 6: 运行 runner 测试确认 GREEN**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest \
  tests.simulation.test_audit_dataset_visibility.AuditRunnerTests
```

Expected: 2 tests pass。

- [ ] **Step 7: 写 CLI 元数据失败测试**

在 `EXPECTED_SCRIPTS` 增加：

```python
"vla-audit-dataset-visibility": (
    "vla_project.simulation.audit_dataset_visibility:main"
),
```

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest tests.test_package_metadata
```

Expected: 安装分发缺少新命令。

- [ ] **Step 8: 注册 CLI 并实现 main**

在 `pyproject.toml` 增加：

```toml
vla-audit-dataset-visibility = "vla_project.simulation.audit_dataset_visibility:main"
```

模块增加：

```python
DEFAULT_DATASET_DIR = "outputs/dataset/expert_scaling_v1"


def main():
    parser = argparse.ArgumentParser(
        description="确定性重放 expert 数据并审计红块可见率"
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    summary = run_visibility_audit(
        args.dataset_dir,
        args.output_dir,
    )
    print(
        "可见性审计完成："
        f"episode={summary['num_episodes']}，"
        f"frames={summary['num_frames']}，"
        f"groups={summary['visibility_group_counts']}"
    )
```

重新安装 editable package：

```bash
conda run -n vla_env python -m pip install -e .
```

- [ ] **Step 9: 更新 README 命令和只读边界**

命令表增加：

```text
vla-audit-dataset-visibility：确定性重放专家数据并审计红块逐帧可见率
```

说明该命令：

- 使用数据集内 `config_snapshot.yaml`；
- 不修改原始数据；
- 输出到 `visibility_audit_v1/`；
- 任一重放不一致都只生成 failure JSON。

- [ ] **Step 10: 运行 Task 3 验证并提交**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest \
  tests.simulation.test_audit_dataset_visibility \
  tests.test_package_metadata
conda run -n vla_env env PYTHONPATH=src \
  python -m compileall -q src tests
git diff --check
```

Expected: selected tests、compileall、diff check pass。

Commit:

```bash
git add \
  src/vla_project/simulation/audit_dataset_visibility.py \
  tests/simulation/test_audit_dataset_visibility.py \
  pyproject.toml \
  tests/test_package_metadata.py \
  README.md
git commit -m "feat: audit expert dataset visibility"
```

---

### Task 4: 全量审计、记录结论并完成验证

**Files:**
- Generated: `outputs/dataset/expert_scaling_v1/visibility_audit_v1/`
- Modify: `docs/agent/PROJECT_OVERVIEW.md`
- Modify: `docs/agent/CURRENT_STATUS.md`
- Modify: `docs/worklog/WORKLOG.md`

**Interfaces:**
- Consumes: Task 3 的 CLI 和现有300条数据。
- Produces: 9,894条可信逐帧可见率、整批遮挡分布和下一阶段数据使用决策。

- [ ] **Step 1: 在隔离分支运行完整验证**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest discover -s tests -v
conda run -n vla_env env PYTHONPATH=src \
  python -m compileall -q src tests
git diff --check
```

Expected: all tests pass，compileall 和 diff check 退出码为0。

- [ ] **Step 2: 合并实现回 main 后核对审计输入**

Run:

```bash
conda run -n vla_env python -m pip install -e .
test ! -e outputs/dataset/expert_scaling_v1/visibility_audit_v1
conda run -n vla_env env PYTHONPATH=src python -c \
  "import json,pathlib; d=pathlib.Path('outputs/dataset/expert_scaling_v1'); f=sum(1 for x in (d/'trajectory_expert.jsonl').read_text().splitlines() if x.strip()); e=sum(1 for x in (d/'episode_summary.jsonl').read_text().splitlines() if x.strip()); r=json.loads((d/'dataset_quality_report.json').read_text()); assert (e,f)==(300,9894); assert r['passed'] is True; print({'episodes':e,'frames':f,'scale_gate':r['passed']})"
```

Expected: 输出 `episodes=300`、`frames=9894`、`scale_gate=true`。

- [ ] **Step 3: 运行全量确定性重放审计**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  vla-audit-dataset-visibility
```

Expected: 逐帧重放全部精确匹配，生成成功 JSONL 和 summary，不生成 failure JSON。

- [ ] **Step 4: 验证成功输出契约**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src python -c \
  "import json,pathlib; d=pathlib.Path('outputs/dataset/expert_scaling_v1/visibility_audit_v1'); rows=[json.loads(x) for x in (d/'frame_visibility.jsonl').read_text().splitlines() if x.strip()]; s=json.loads((d/'visibility_audit_summary.json').read_text()); assert len(rows)==9894; assert s['num_episodes']==300; assert s['num_frames']==9894; assert s['replay_exact_match_count']==9894; assert s['replay_mismatch_count']==0; assert s['replay_validation']['passed'] is True; assert sum(s['visibility_group_counts'].values())==9894; assert not (d/'visibility_audit_failure.json').exists(); print(json.dumps(s,ensure_ascii=False,indent=2))"
```

Expected: 所有断言通过并打印真实遮挡分布。

- [ ] **Step 5: 根据真实结果更新权威文档**

在 `WORKLOG.md` 记录：

- 重放精确匹配数量；
- clear/partial/severe 数量与比例；
- 有 nonclear/severe 的 episode 数；
- 初始/终止分布；
- 最长连续遮挡；
- 最低可见率样本；
- 第一版单帧训练数据是否需要隔离 severe。

同步改写 `PROJECT_OVERVIEW.md` 和 `CURRENT_STATUS.md`。不得在看到结果前预设“数据可直接
训练”；若重放失败，记录失败证据并停留在审计诊断阶段。

- [ ] **Step 6: 最终验证与提交**

Run:

```bash
conda run -n vla_env env PYTHONPATH=src \
  python -m unittest discover -s tests
conda run -n vla_env env PYTHONPATH=src \
  python -m compileall -q src tests
git diff --check
git status --short
```

Expected: 完整测试、编译和 diff 检查通过；Git 状态只包含预期文档变更和用户原有未跟踪
文件。

Commit:

```bash
git add \
  docs/agent/PROJECT_OVERVIEW.md \
  docs/agent/CURRENT_STATUS.md \
  docs/worklog/WORKLOG.md
git commit -m "docs: record expert visibility audit"
```

生成的 `visibility_audit_v1/` 作为本地实验真值证据，不提交 Git。
