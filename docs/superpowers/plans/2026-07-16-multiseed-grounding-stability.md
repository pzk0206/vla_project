# 多位置 Grounding 定位稳定性实施计划

> **供执行者使用：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，逐项实施本计划。各步骤使用复选框（`- [ ]`）跟踪进度。

**目标：** 从五个可复现的红块位置生成并评估 20 个固定相机 grounding 样本；Qwen 调用支持断点续跑，并按机械臂方向和随机种子分组汇总结果。

**架构：** 复用现有分层均衡姿态采样器，使用 seeds 42–46 和带版本号的新输出目录，保留原有四样本基准。grounding 预测增加样本 seed，付费调用通过 `sample_id` 实现断点续跑；反投影摘要增加逐方向指标和逐 seed 的像素中心抖动统计。

**技术栈：** Python 3、`unittest`、PyBullet、OpenCV、NumPy、YAML、Qwen OpenAI 兼容 API、JSON/JSONL、conda 环境 `vla_env`。

## 全局约束

- 固定正俯视相机、`448 x 448` 图片、`0.20m` XY 偏移、Qwen 模型、grounding 提示词和工作平面反投影算法均保持不变。
- 使用 seeds 42–46；每个 seed 分别生成 left、right、front、back 四个方向，总计恰好 20 个样本。
- 不得覆盖 `vlm_eval_samples_448/` 或 `grounding_qwen3_vl_flash_distance20_448_v3/`。
- `samples.jsonl` 不得包含 `block_pos` 和相机矩阵；这些真值只能保存在 `diagnostics.jsonl`。
- 本计划不添加标定偏移，也不把定位结果接入在线控制。
- 保留原 `balanced_block_position` 范围和遮挡样本；使用 PyBullet segmentation mask 把遮挡程度写入 diagnostics，不通过筛选图片掩盖问题。
- 所有 Python 测试和脚本都必须通过 `conda run -n vla_env` 执行。

---

### 任务 1：为 20 样本实验建立独立配置版本

**文件：**

- 修改：`tests/test_config_contract.py`
- 修改：`tests/test_collect_vlm_eval_samples.py`
- 修改：`sim_config.yaml`
- 修改：`.gitignore`

**接口：**

- 输入：`collect_vlm_eval_samples(config) -> tuple[Path, list[dict]]` 以及现有的 `stratified_balanced_poses` 策略。
- 输出：把 20 个样本写入 `vlm_eval_samples_448_multiseed_d020/`，并把 grounding 结果写入 `vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/` 的配置。

- [ ] **步骤 1：先收紧配置契约**

将 `test_vlm_evaluation_config_is_valid` 中旧的输出目录断言和仅检查 seed 为正数的断言替换为：

```python
self.assertEqual(
    evaluation["sample_output_dir"],
    "vlm_eval_samples_448_multiseed_d020",
)
self.assertEqual(
    evaluation["grounding_run_name"],
    "grounding_qwen3_vl_flash_distance20_448_multiseed_v4",
)
self.assertEqual(evaluation["stratified_num_seeds"], 5)
```

保留已有的采样策略、`balanced_pose_offsets_xy == [0.20]`、相机高度和 `448 x 448` 分辨率断言。

- [ ] **步骤 2：运行配置契约测试并确认红灯失败**

运行：

```bash
conda run -n vla_env python -m unittest tests.test_config_contract.ConfigContractTests.test_vlm_evaluation_config_is_valid -v
```

预期：测试失败，因为当前配置仍使用 `vlm_eval_samples_448`，并且 `stratified_num_seeds: 1`。

- [ ] **步骤 3：应用最小配置改动**

在 `sim_config.yaml` 中设置以下准确值：

```yaml
vlm_evaluation:
  sample_output_dir: "vlm_eval_samples_448_multiseed_d020"
  offline_run_name: "offline_qwen3_vl_flash_distance20_448_multiseed_v15"
  grounding_run_name: "grounding_qwen3_vl_flash_distance20_448_multiseed_v4"
  ground_then_decide_run_name: "ground_then_decide_qwen3_vl_flash_448_multiseed_v2"
  sample_strategy: "stratified_balanced_poses"
  balanced_pose_offsets_xy: [0.20]
  stratified_num_seeds: 5
```

将新生成的样本目录加入 `.gitignore`：

```gitignore
vlm_eval_samples_448_multiseed_d020/
```

- [ ] **步骤 4：加强模拟采样回归测试**

在 `test_balanced_collection_writes_matching_diagnostics` 中把 `stratified_num_seeds` 设为 `5`，然后将四样本断言替换为：

```python
self.assertEqual(len(samples), 20)
self.assertEqual(len(diagnostics), 20)
self.assertEqual(capture_sample.call_count, 20)
self.assertEqual(
    {row["random_seed"] for row in samples},
    {42, 43, 44, 45, 46},
)
self.assertEqual(
    {direction: sum(row["expected_direction"] == direction for row in samples)
     for direction in ("left", "right", "front", "back")},
    {"left": 5, "right": 5, "front": 5, "back": 5},
)
self.assertEqual(
    {row["sample_id"] for row in samples},
    {row["sample_id"] for row in diagnostics},
)
```

该回归测试不需要修改采样器生产代码即可通过，因为现有 seed 循环已经支持多个 seed。

- [ ] **步骤 5：运行专项测试并确认绿灯通过**

运行：

```bash
conda run -n vla_env python -m unittest tests.test_config_contract tests.test_collect_vlm_eval_samples -v
```

预期：测试通过，模拟的均衡采样生成 20 个唯一样本。

- [ ] **步骤 6：提交配置改动**

```bash
git add sim_config.yaml .gitignore tests/test_config_contract.py tests/test_collect_vlm_eval_samples.py
git commit -m "test: configure multiseed grounding samples"
```

---

### 任务 2：让 grounding 调用支持断点续跑并保留样本 seed

**文件：**

- 修改：`tests/test_diagnose_vlm_grounding.py`
- 修改：`diagnose_vlm_grounding.py`

**接口：**

- 输入：`samples.jsonl` 中包含 `sample_id`、`random_seed`、图片路径、指令和预期方向的记录。
- 输出：`load_existing_predictions(path) -> dict[str, dict]`，以及每个已完成 `sample_id` 对应的一条唯一 `grounding_predictions.jsonl` 记录，其中包含 `random_seed`。

- [ ] **步骤 1：编写断点续跑失败测试**

导入 `json`、`tempfile`、`Path`、`patch` 和 `cv2`，同时导入 `diagnose_grounding`，然后添加：

```python
class GroundingResumeTests(unittest.TestCase):
    def test_completed_sample_is_skipped_and_seed_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_dir = root / "samples"
            image_path = sample_dir / "images" / "sample.jpg"
            image_path.parent.mkdir(parents=True)
            self.assertTrue(
                cv2.imwrite(str(image_path), np.zeros((16, 16, 3), dtype=np.uint8))
            )
            sample = {
                "sample_id": "seed_42_d020_left",
                "image_path": str(image_path),
                "instruction": "悬停在红色积木上方",
                "expected_direction": "left",
                "random_seed": 42,
            }
            (sample_dir / "samples.jsonl").write_text(
                json.dumps(sample, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            config = {
                "vlm_evaluation": {
                    "sample_output_dir": str(sample_dir),
                    "run_output_dir": str(root / "runs"),
                    "grounding_run_name": "grounding_test",
                },
                "probe": {"api": {}},
            }
            boxes = {
                "end_effector": [100.0, 100.0, 200.0, 200.0],
                "red_block": [500.0, 500.0, 600.0, 600.0],
            }
            with patch(
                "diagnose_vlm_grounding.call_openai_compatible_api",
                return_value=(boxes, "{}"),
            ) as api_mock:
                first_dir, first_results = diagnose_grounding(config, limit=1)
                second_dir, second_results = diagnose_grounding(config, limit=1)

            self.assertEqual(api_mock.call_count, 1)
            self.assertEqual(first_dir, second_dir)
            self.assertEqual(first_results[0]["random_seed"], 42)
            self.assertEqual(second_results[0]["random_seed"], 42)
            lines = (first_dir / "grounding_predictions.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(lines), 1)
```

- [ ] **步骤 2：运行断点续跑测试并确认红灯失败**

运行：

```bash
conda run -n vla_env python -m unittest tests.test_diagnose_vlm_grounding.GroundingResumeTests -v
```

预期：测试失败，因为 API 被调用两次，并且结果中缺少 `random_seed`。

- [ ] **步骤 3：实现唯一记录加载并跳过已完成样本**

添加：

```python
def load_existing_predictions(path):
    """按 sample_id 读取已完成 grounding，拒绝重复行。"""
    path = Path(path)
    if not path.exists():
        return {}
    existing = {}
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("grounding 结果缺少非空 sample_id")
        if sample_id in existing:
            raise ValueError(f"grounding sample_id 重复: {sample_id}")
        existing[sample_id] = row
    return existing
```

在 `diagnose_grounding` 中创建 `results_path` 后，将当前的 `results = []` 初始化替换为：

```python
existing = load_existing_predictions(results_path)
```

在现有样本循环开头加入以下跳过判断：

```python
for sample in samples:
    if sample["sample_id"] in existing:
        continue
```

保留循环中现有的图片读取、API `try/except` 和结果构造逻辑，并把 seed 直接加入结果字典：

```python
result = {
    "sample_id": sample["sample_id"],
    "image_path": sample["image_path"],
    "expected_direction": sample["expected_direction"],
    "random_seed": sample["random_seed"],
    "boxes": boxes,
    "raw_response": raw_response,
    "annotated_path": str(annotated_path) if annotated_path else None,
    "latency_seconds": time.perf_counter() - started_at,
    "error_type": error_type,
    "error_message": error_message,
}
append_jsonl(results_path, result)
existing[sample["sample_id"]] = result
```

循环结束后，按照 manifest 中的顺序构造返回列表：

```python
results = [existing[sample["sample_id"]] for sample in samples]

return run_dir, results
```

- [ ] **步骤 4：运行 grounding 测试并确认绿灯通过**

运行：

```bash
conda run -n vla_env python -m unittest tests.test_diagnose_vlm_grounding -v
```

预期：测试通过，断点续跑测试中的两次执行总共只调用一次 API。

- [ ] **步骤 5：提交 grounding 断点续跑功能**

```bash
git add diagnose_vlm_grounding.py tests/test_diagnose_vlm_grounding.py
git commit -m "feat: resume grounding evaluation by sample id"
```

---

### 任务 3：按方向和 seed 汇总定位误差

**文件：**

- 修改：`tests/test_evaluate_grounding_backprojection.py`
- 修改：`evaluate_grounding_backprojection.py`

**接口：**

- 输入：包含 `expected_direction` 和 `random_seed` 的预测记录，以及通过 `sample_id` 连接的诊断真值。
- 输出：保留这些字段的逐样本结果；摘要增加 `per_direction` 和 `per_random_seed`。每组都包含有效率和误差指标，seed 分组还包含像素中心抖动。

- [ ] **步骤 1：编写分组摘要失败测试**

在 `test_summarizes_valid_errors_and_failure_types` 的有效结果字典中加入 `expected_direction` 和 `random_seed`，然后添加：

```python
def test_summarizes_by_direction_and_seed_with_pixel_jitter(self):
    results = [
        {
            "expected_direction": "left",
            "random_seed": 42,
            "box_center_pixel": [250.0, 190.0],
            "localization_error_xy": 0.01,
            "signed_error_x": -0.006,
            "signed_error_y": 0.008,
            "error_type": None,
        },
        {
            "expected_direction": "right",
            "random_seed": 42,
            "box_center_pixel": [254.0, 193.0],
            "localization_error_xy": 0.03,
            "signed_error_x": -0.018,
            "signed_error_y": 0.024,
            "error_type": None,
        },
        {
            "expected_direction": "left",
            "random_seed": 43,
            "box_center_pixel": None,
            "localization_error_xy": None,
            "signed_error_x": None,
            "signed_error_y": None,
            "error_type": "InvalidModelResponseError",
        },
    ]

    summary = summarize_results(results)

    self.assertEqual(summary["per_direction"]["left"]["num_samples"], 2)
    self.assertEqual(summary["per_direction"]["left"]["num_failed"], 1)
    self.assertAlmostEqual(
        summary["per_random_seed"]["42"]["box_center_pixel_span_x"], 4.0
    )
    self.assertAlmostEqual(
        summary["per_random_seed"]["42"]["box_center_pixel_span_y"], 3.0
    )
    self.assertAlmostEqual(
        summary["per_random_seed"]["42"]["box_center_pixel_max_distance"], 5.0
    )
```

在 `test_computes_zero_xy_error_for_matching_box_center` 的预测记录中加入以下字段，并断言评估后仍然保留：

```python
"expected_direction": "left",
"random_seed": 42,
```

```python
self.assertEqual(result["expected_direction"], "left")
self.assertEqual(result["random_seed"], 42)
```

- [ ] **步骤 2：运行分组测试并确认红灯失败**

运行：

```bash
conda run -n vla_env python -m unittest tests.test_evaluate_grounding_backprojection -v
```

预期：测试失败，因为结果记录没有保留 seed/方向，并且摘要中缺少分组字段。

- [ ] **步骤 3：在结果记录中保留分组元数据**

在 `evaluate_rows` 的 `base_result` 中加入以下字段：

```python
"expected_direction": prediction.get("expected_direction"),
"random_seed": prediction.get("random_seed"),
```

- [ ] **步骤 4：实现可复用的分组指标和像素抖动统计**

将现有整体统计提取为：

```python
def _summarize_subset(rows, include_pixel_jitter=False):
    rows = list(rows)
    valid = [row for row in rows if row["localization_error_xy"] is not None]
    errors = [row["localization_error_xy"] for row in valid]
    error_counts = Counter(row["error_type"] for row in rows if row["error_type"])
    summary = {
        "num_samples": len(rows),
        "num_valid": len(valid),
        "num_failed": len(rows) - len(valid),
        "valid_rate": len(valid) / len(rows) if rows else 0.0,
        "localization_error_xy_mean": statistics.fmean(errors) if errors else None,
        "localization_error_xy_median": statistics.median(errors) if errors else None,
        "localization_error_xy_max": max(errors) if errors else None,
        "signed_error_x_mean": (
            statistics.fmean(row["signed_error_x"] for row in valid)
            if valid else None
        ),
        "signed_error_y_mean": (
            statistics.fmean(row["signed_error_y"] for row in valid)
            if valid else None
        ),
        "error_type_counts": dict(sorted(error_counts.items())),
    }
    if include_pixel_jitter:
        pixels = [row["box_center_pixel"] for row in valid if row.get("box_center_pixel")]
        summary["box_center_pixel_span_x"] = (
            max(point[0] for point in pixels) - min(point[0] for point in pixels)
            if pixels else None
        )
        summary["box_center_pixel_span_y"] = (
            max(point[1] for point in pixels) - min(point[1] for point in pixels)
            if pixels else None
        )
        summary["box_center_pixel_max_distance"] = (
            max(
                math.dist(first, second)
                for index, first in enumerate(pixels)
                for second in pixels[index + 1:]
            ) if len(pixels) >= 2 else 0.0 if pixels else None
        )
    return summary
```

在 `summarize_results` 中构造分组输出：

```python
def summarize_results(results):
    results = list(results)
    summary = _summarize_subset(results)
    directions = sorted(
        {row["expected_direction"] for row in results if row.get("expected_direction")}
    )
    seeds = sorted(
        {row["random_seed"] for row in results if row.get("random_seed") is not None}
    )
    summary["per_direction"] = {
        direction: _summarize_subset(
            row for row in results if row.get("expected_direction") == direction
        )
        for direction in directions
    }
    summary["per_random_seed"] = {
        str(seed): _summarize_subset(
            (row for row in results if row.get("random_seed") == seed),
            include_pixel_jitter=True,
        )
        for seed in seeds
    }
    return summary
```

- [ ] **步骤 5：运行专项测试和完整测试**

运行：

```bash
conda run -n vla_env python -m unittest tests.test_evaluate_grounding_backprojection -v
conda run -n vla_env python -m unittest discover -s tests -v
```

预期：专项测试通过，完整测试套件通过且没有回归。

- [ ] **步骤 6：提交分组分析功能**

```bash
git add evaluate_grounding_backprojection.py tests/test_evaluate_grounding_backprojection.py
git commit -m "feat: summarize grounding stability by pose and seed"
```

---

### 任务 4：生成并校验 20 个本地样本

**文件：**

- 修改：`control_arm.py`
- 修改：`collect_vlm_eval_samples.py`
- 修改：`tests/test_control_arm.py`
- 修改：`tests/test_collect_vlm_eval_samples.py`
- 生成并由 Git 忽略：`vlm_eval_samples_448_multiseed_d020/images/*.jpg`
- 生成并由 Git 忽略：`vlm_eval_samples_448_multiseed_d020/samples.jsonl`
- 生成并由 Git 忽略：`vlm_eval_samples_448_multiseed_d020/diagnostics.jsonl`

**接口：**

- 输入：任务 1 的配置、PyBullet RGB 图和 segmentation mask。
- 输出：供任务 5 和任务 6 使用的 20 组图片、样本记录和诊断记录；每条 diagnostics 包含 `block_visible_pixels`、`block_reference_pixels` 和 `block_visibility_ratio`。

- [ ] **步骤 1：为 RGB 与 segmentation 联合采集编写失败测试**

在 `tests/test_control_arm.py` 中验证新函数 `capture_rgb_and_segmentation(camera_config, camera_eye)` 返回 `(BGR, segmentation)`，二者分别为 `(height, width, 3)` 和 `(height, width)`；同时保留 `capture_rgb` 的原接口。

- [ ] **步骤 2：实现联合采集并确认测试通过**

让 `capture_rgb_and_segmentation` 复用同一组 view/projection matrices；`capture_rgb` 只返回联合采集结果的第一项，避免影响现有调用方。

- [ ] **步骤 3：为可见率诊断编写失败测试**

在 `tests/test_collect_vlm_eval_samples.py` 中验证：当前 segmentation 中属于 `block_id` 的像素数为可见数；移除机械臂后参考 segmentation 中属于同一 `block_id` 的像素数为参考数；可见率等于 `visible/reference`，并且这些字段只进入 diagnostics。

- [ ] **步骤 4：实现可见率采集并确认测试通过**

均衡姿态先保存带机械臂的 RGB 和 segmentation，再移除机器人及可选 marker，使用相同相机重拍参考 segmentation。拒绝参考像素数为 0、可见数大于参考数或可见率超出 `[0, 1]` 的诊断记录。

- [ ] **步骤 5：在不调用 Qwen 的情况下重新生成样本**

运行：

```bash
conda run -n vla_env python collect_vlm_eval_samples.py
```

预期：退出码为 0，并输出 `样本数量: 20`。

- [ ] **步骤 6：校验数量、ID、seed、方向、可见率和真值隔离**

运行：

```bash
conda run -n vla_env python -c 'import json,pathlib,collections; root=pathlib.Path("vlm_eval_samples_448_multiseed_d020"); samples=[json.loads(x) for x in (root/"samples.jsonl").read_text().splitlines() if x.strip()]; diagnostics=[json.loads(x) for x in (root/"diagnostics.jsonl").read_text().splitlines() if x.strip()]; assert len(samples)==len(diagnostics)==20; assert {x["sample_id"] for x in samples}=={x["sample_id"] for x in diagnostics}; assert {x["random_seed"] for x in samples}==set(range(42,47)); assert collections.Counter(x["expected_direction"] for x in samples)==collections.Counter({"left":5,"right":5,"front":5,"back":5}); assert all("block_pos" not in x and "view_matrix" not in x for x in samples); assert all((pathlib.Path(x["image_path"])).is_file() for x in samples); print("validated", len(samples), collections.Counter(x["expected_direction"] for x in samples))'
```

预期：输出 `validated 20 Counter({'left': 5, 'right': 5, 'front': 5, 'back': 5})`。

- [ ] **步骤 7：人工检查清晰和严重遮挡图片**

打开生成图片目录中的 left 与 back 样本，确认 diagnostics 的可见率排序与画面一致。遮挡样本继续保留；只有图片损坏、红块出视野或参考 segmentation 为 0 时才停止。

---

### 任务 5：运行可断点续跑的 Qwen grounding 和反投影

**文件：**

- 生成并由 Git 忽略：`vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/grounding_predictions.jsonl`
- 生成并由 Git 忽略：`vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/annotated/*.jpg`
- 生成并由 Git 忽略：`vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/backprojection/backprojection_results.jsonl`
- 生成并由 Git 忽略：`vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/backprojection/backprojection_summary.json`

**接口：**

- 输入：任务 4 的 20 个样本和诊断真值，以及 `sim_config.yaml` 中的 API 设置和环境变量 `VLA_API_BASE_URL`、`VLA_API_KEY`、`VLA_MODEL_NAME`。
- 输出：20 条已保存的 grounding 结果和一份按方向、seed、可见程度分组的定位摘要。

- [ ] **步骤 1：在不打印密钥的情况下验证 API 配置**

运行：

```bash
conda run -n vla_env python -c 'import os; names=("VLA_API_BASE_URL","VLA_API_KEY","VLA_MODEL_NAME"); missing=[name for name in names if not os.getenv(name)]; assert not missing, f"missing env vars: {missing}"; print("API environment ready")'
```

预期：输出 `API environment ready`。禁止打印这些环境变量的实际值。

- [ ] **步骤 2：运行全部 20 次支持断点续跑的 grounding 调用**

运行：

```bash
conda run -n vla_env python diagnose_vlm_grounding.py --limit 20
```

预期：命令报告 `成功生成目标框: N/20`；每个已尝试样本都会立即保存。如果执行中断，重新运行同一命令，已完成的 `sample_id` 会被跳过。

- [ ] **步骤 3：验证已保存结果唯一且完整**

运行：

```bash
conda run -n vla_env python -c 'import json,pathlib; path=pathlib.Path("vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/grounding_predictions.jsonl"); rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]; ids=[x["sample_id"] for x in rows]; assert len(rows)==len(set(ids))==20; assert {x["random_seed"] for x in rows}==set(range(42,47)); print("unique grounding rows", len(rows), "valid", sum(x["boxes"] is not None for x in rows))'
```

预期：输出 `unique grounding rows 20 valid N`，且不存在重复 ID。

- [ ] **步骤 4：运行工作平面反投影和分组统计**

运行：

```bash
conda run -n vla_env python evaluate_grounding_backprojection.py --predictions vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/grounding_predictions.jsonl --diagnostics vlm_eval_samples_448_multiseed_d020/diagnostics.jsonl --output-dir vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/backprojection
```

预期：退出码为 0，输出的 JSON 包含 `num_samples: 20`、`per_direction`、`per_random_seed` 和 `per_visibility_group`。

- [ ] **步骤 5：在不添加补偿的情况下判断证据类型**

读取 `backprojection_summary.json` 和 `backprojection_results.jsonl`，记录：

- 整体平均值、中位数、最大值、有符号 X 均值和有符号 Y 均值；
- left/right/front/back 各方向的平均误差和失败数；
- seeds 42–46 的像素中心 X/Y 跨度和最大两两像素距离；
- 有符号误差在不同 seed 和姿态之间是否保持相同方向。
- clear、partial、severe 三个可见程度分组的有效率和定位误差。

结论规则：只有当不同位置、不同姿态下的误差方向和幅度仍然相近时，才建议后续设计标定补偿实验。否则应改进 grounding，或进行检测器/分割方法的消融实验。无论哪种结果，本轮都不修改控制器代码。

---

### 任务 6：记录已验证证据并完成回归检查

**文件：**

- 修改：`README.md`
- 修改：`docs/worklog/WORKLOG.md`
- 修改：`docs/planning/vla_robotic_study_plan.md`
- 修改：`docs/debugging/BUGLOG.md`

**接口：**

- 输入：任务 5 生成的精确 JSON 数值和逐样本证据。
- 输出：同步更新项目状态，明确区分方向级 grounding 与厘米级定位。

- [ ] **步骤 1：在工作日志和错误日志中增加实验证据章节**

使用标题 `五位置 grounding 反投影稳定性实验（2026-07-16）`。从以下 JSON 字段复制准确值，不得根据终端显示自行近似：

```text
num_samples
num_valid
num_failed
localization_error_xy_mean
localization_error_xy_median
localization_error_xy_max
signed_error_x_mean
signed_error_y_mean
per_direction
per_random_seed
```

明确说明相机始终固定，只有红块 seed 和机械臂姿态发生变化。写入依据任务 5 规则得出的结论，以及准确的输出目录。

- [ ] **步骤 2：同步 README 和学习计划的下一步表述**

在两个文件中说明：框推导方向的 `4/4` 属于方向级证据，而五位置实验衡量的是厘米级定位。继续采用“VLM grounding + 确定性几何/控制”架构；除非任务 5 的证据支持，否则不得宣称已经可以进行标定补偿。

- [ ] **步骤 3：运行最终验证**

运行：

```bash
conda run -n vla_env python -m unittest discover -s tests -v
git diff --check
git status --short
```

预期：所有测试通过，`git diff --check` 没有输出，生成的样本和运行目录仍被 Git 忽略，此时只有四个文档文件尚未提交。

- [ ] **步骤 4：提交已验证的实验文档**

```bash
git add README.md docs/worklog/WORKLOG.md docs/planning/vla_robotic_study_plan.md docs/debugging/BUGLOG.md
git commit -m "docs: record multiseed grounding stability evidence"
```

- [ ] **步骤 5：验证最终仓库状态**

运行：

```bash
git status --short --branch
git log -5 --oneline
```

预期：工作区干净，并且配置、grounding 断点续跑、分组分析和实验文档均有对应提交。除非用户明确要求发布，否则不要推送。
