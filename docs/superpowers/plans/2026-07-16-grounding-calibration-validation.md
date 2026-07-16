# Grounding 定位独立校准验证实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 冻结 seeds 42–46 清晰样本估计出的 XY 固定补偿，并在从未参与拟合的 seeds 47–51 上独立验证 clear 样本能否全部达到 3cm 定位门槛。

**Architecture:** 采样器通过独立的 `vlm_evaluation.stratified_seeds` 生成验证集，不再隐式复用 Stage 3 的 `probe_evaluation.random_seed`。新增离线脚本从旧反投影结果拟合一次补偿，再只读地应用到新反投影结果，同时保留原始坐标、补偿坐标、分组指标和明确的通过判定；在线控制器不读取该补偿。

**Tech Stack:** Python 3、PyBullet、OpenCV、PyYAML、标准库 `unittest`、JSON/JSONL、Qwen OpenAI-compatible API、Git。

## Global Constraints

- 校准集固定为 seeds 42–46 的既有反投影结果，只使用 `visibility_group == "clear"` 的 15 条记录拟合。
- 验证集固定为 seeds 47–51，每个 seed 包含 left/right/front/back 四种姿态，共 20 张图片。
- 验证阶段不得重新计算或修改补偿；校准集和验证集的 `sample_id` 必须完全不相交。
- clear 验证样本必须至少有 1 条、必须全部定位有效，并且补偿后误差必须全部 `<= 0.03m` 才能通过。
- partial/severe 仍应用相同补偿并单独报告，但不参与通过判定。
- 原始 grounding、反投影结果保持只读；补偿结果写入独立目录。
- 不把补偿写入 `stage3_probe.py`、`control_arm.py` 或任何在线机械臂控制路径。
- 所有 Python 测试和脚本使用 conda 环境 `vla_env`。
- 真实 Qwen 运行沿用 `sample_id` 断点续跑，成功或失败记录都不重复付费调用。

---

## 文件结构

- Modify: `sim_config.yaml` — 切换到验证集目录、运行名和显式 seeds 47–51。
- Modify: `collect_vlm_eval_samples.py` — 验证并优先使用显式 seed 列表。
- Modify: `tests/test_config_contract.py` — 固定验证集配置契约。
- Modify: `tests/test_collect_vlm_eval_samples.py` — 证明显式 seeds 产生 20 个唯一、均衡且配对的样本。
- Create: `validate_grounding_calibration.py` — 拟合冻结补偿、应用补偿、生成摘要和三个证据文件。
- Create: `tests/test_validate_grounding_calibration.py` — 覆盖拟合、数据隔离、无效定位、分组摘要、3cm 门槛和 CLI 输出。
- Modify: `README.md` — 更新当前阶段与独立验证结论。
- Modify: `docs/worklog/WORKLOG.md` — 记录实验方法、实际指标和工程判断。
- Modify: `docs/planning/vla_robotic_study_plan.md` — 根据是否通过更新下一步。
- Modify: `docs/debugging/BUGLOG.md` — 更新 BUG-003 的复现证据和处置边界。

---

### Task 1: 显式验证 seeds 与隔离输出目录

**Files:**
- Modify: `sim_config.yaml:116-142`
- Modify: `collect_vlm_eval_samples.py:387-513`
- Modify: `tests/test_config_contract.py:40-70`
- Modify: `tests/test_collect_vlm_eval_samples.py:135-196`

**Interfaces:**
- Consumes: `config["vlm_evaluation"]["stratified_seeds"]: list[int]`。
- Produces: `validate_stratified_seeds(value) -> list[int]`；`collect_vlm_eval_samples(config)` 使用返回列表逐 seed 采四个方向。

- [ ] **Step 1: 写配置契约和采样失败测试**

在 `tests/test_config_contract.py::test_vlm_evaluation_config_is_valid` 中把旧目录断言替换为以下精确值：

```python
self.assertEqual(
    evaluation["sample_output_dir"],
    "vlm_eval_samples_448_calibration_validation_d020",
)
self.assertEqual(
    evaluation["grounding_run_name"],
    "grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1",
)
self.assertEqual(evaluation["stratified_seeds"], [47, 48, 49, 50, 51])
self.assertEqual(len(set(evaluation["stratified_seeds"])), 5)
self.assertTrue(
    all(type(seed) is int for seed in evaluation["stratified_seeds"])
)
```

在 `tests/test_collect_vlm_eval_samples.py` 顶部导入 `validate_stratified_seeds`，并新增：

```python
class ValidateStratifiedSeedsTests(unittest.TestCase):
    def test_accepts_five_unique_integer_seeds(self):
        self.assertEqual(
            validate_stratified_seeds([47, 48, 49, 50, 51]),
            [47, 48, 49, 50, 51],
        )

    def test_rejects_empty_duplicate_boolean_and_non_integer_seeds(self):
        for value in ([], [47, 47], [True, 48], [47, "48"]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "stratified_seeds"):
                    validate_stratified_seeds(value)
```

将 `test_balanced_collection_writes_matching_diagnostics` 的配置改为：

```python
"stratified_num_seeds": 5,
"stratified_seeds": [47, 48, 49, 50, 51],
```

并把 seed 集合断言改为：

```python
self.assertEqual(
    {row["random_seed"] for row in samples},
    {47, 48, 49, 50, 51},
)
self.assertEqual(len({row["sample_id"] for row in samples}), 20)
```

- [ ] **Step 2: 运行定向测试，确认先失败**

Run:

```bash
conda run -n vla_env python -m unittest \
  tests.test_config_contract.ConfigContractTests.test_vlm_evaluation_config_is_valid \
  tests.test_collect_vlm_eval_samples.ValidateStratifiedSeedsTests \
  tests.test_collect_vlm_eval_samples.SampleDiagnosticsTests.test_balanced_collection_writes_matching_diagnostics -v
```

Expected: FAIL/ERROR，原因分别是配置仍为旧目录、`validate_stratified_seeds` 尚不存在或采样仍生成 42–46。

- [ ] **Step 3: 实现显式 seed 验证与采样选择**

在 `collect_vlm_eval_samples.py` 的采样辅助函数区新增：

```python
def validate_stratified_seeds(value):
    """返回显式分层 seeds，并拒绝空值、重复值、布尔值和非整数。"""
    if not isinstance(value, list) or not value:
        raise ValueError("vlm_evaluation.stratified_seeds 必须是非空列表")
    if any(type(seed) is not int for seed in value):
        raise ValueError("vlm_evaluation.stratified_seeds 只能包含整数")
    if len(set(value)) != len(value):
        raise ValueError("vlm_evaluation.stratified_seeds 不能重复")
    return list(value)
```

在 `collect_vlm_eval_samples()` 的分层分支中用显式列表替换 `base_seed + episode_idx`：

```python
if evaluation["sample_strategy"] == "stratified_balanced_poses":
    cases = build_stratified_balanced_cases(
        evaluation["balanced_pose_offsets_xy"]
    )
    seeds = validate_stratified_seeds(evaluation["stratified_seeds"])
else:
    cases = [
        {
            "offset_xy": evaluation["balanced_pose_offset_xy"],
            "offset_tag": None,
            "direction": direction,
        }
        for direction in BALANCED_DIRECTION_ORDER
    ]
    seeds = [
        base_seed + episode_idx
        for episode_idx in range(evaluation["offline_num_episodes"])
    ]

for seed in seeds:
    for case_index, case in enumerate(cases):
        # 保留现有逐 case 采集代码。
```

在 `sim_config.yaml` 中保留 `probe_evaluation.random_seed: 42`，只修改 VLM 项：

```yaml
  sample_output_dir: "vlm_eval_samples_448_calibration_validation_d020"
  grounding_run_name: "grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1"
  stratified_num_seeds: 5
  stratified_seeds: [47, 48, 49, 50, 51]
```

`stratified_num_seeds` 本轮保留用于兼容既有文档和配置，但分层采样的真实 seed 来源只允许是 `stratified_seeds`。

- [ ] **Step 4: 运行 Task 1 测试并确认通过**

Run:

```bash
conda run -n vla_env python -m unittest \
  tests.test_config_contract \
  tests.test_collect_vlm_eval_samples -v
```

Expected: 全部 PASS；采样 mock 明确得到 20 个样本、四方向各 5 张、seeds 47–51，samples/diagnostics ID 完全一致。

- [ ] **Step 5: 提交显式验证集配置**

```bash
git add sim_config.yaml collect_vlm_eval_samples.py \
  tests/test_config_contract.py tests/test_collect_vlm_eval_samples.py
git commit -m "feat: isolate calibration validation seeds"
```

---

### Task 2: 冻结校准参数与逐样本补偿

**Files:**
- Create: `validate_grounding_calibration.py`
- Create: `tests/test_validate_grounding_calibration.py`

**Interfaces:**
- Consumes: 校准/验证 `backprojection_results.jsonl` 行，字段包括 `sample_id`、`visibility_group`、`predicted_target_world`、`true_block_pos`、`localization_error_xy`、`signed_error_x`、`signed_error_y`。
- Produces: `fit_clear_calibration(rows, source_path) -> dict`；`apply_frozen_calibration(calibration, validation_rows) -> list[dict]`。

- [ ] **Step 1: 写冻结补偿的失败测试**

创建 `tests/test_validate_grounding_calibration.py`，先写以下数据工厂和核心测试：

```python
import math
import unittest

from validate_grounding_calibration import (
    apply_frozen_calibration,
    fit_clear_calibration,
)


def result_row(
    sample_id,
    visibility_group="clear",
    predicted=(0.18, 0.42, 0.0),
    truth=(0.20, 0.40, 0.05),
    error_type=None,
):
    if predicted is None:
        return {
            "sample_id": sample_id,
            "visibility_group": visibility_group,
            "predicted_target_world": None,
            "true_block_pos": list(truth),
            "localization_error_xy": None,
            "signed_error_x": None,
            "signed_error_y": None,
            "error_type": error_type or "MissingGroundingBoxError",
            "error_message": "red_block box 缺失",
        }
    signed_x = predicted[0] - truth[0]
    signed_y = predicted[1] - truth[1]
    return {
        "sample_id": sample_id,
        "visibility_group": visibility_group,
        "predicted_target_world": list(predicted),
        "true_block_pos": list(truth),
        "localization_error_xy": math.hypot(signed_x, signed_y),
        "signed_error_x": signed_x,
        "signed_error_y": signed_y,
        "error_type": None,
        "error_message": None,
    }


class FitClearCalibrationTests(unittest.TestCase):
    def test_fits_only_clear_rows_and_uses_opposite_signed_mean(self):
        rows = [
            result_row("seed_42_left", predicted=(0.18, 0.43, 0.0)),
            result_row("seed_43_right", predicted=(0.16, 0.41, 0.0)),
            result_row(
                "seed_44_back",
                visibility_group="severe",
                predicted=(0.50, 0.10, 0.0),
            ),
        ]

        calibration = fit_clear_calibration(rows, "calibration.jsonl")

        self.assertEqual(calibration["calibration_source"], "calibration.jsonl")
        self.assertEqual(calibration["num_clear_calibration_samples"], 2)
        self.assertEqual(
            calibration["calibration_sample_ids"],
            ["seed_42_left", "seed_43_right"],
        )
        self.assertAlmostEqual(calibration["signed_error_x_mean"], -0.03)
        self.assertAlmostEqual(calibration["signed_error_y_mean"], 0.02)
        self.assertAlmostEqual(calibration["correction_x"], 0.03)
        self.assertAlmostEqual(calibration["correction_y"], -0.02)

    def test_rejects_empty_or_invalid_clear_calibration(self):
        with self.assertRaisesRegex(ValueError, "clear 校准样本不能为空"):
            fit_clear_calibration([], "empty.jsonl")
        with self.assertRaisesRegex(ValueError, "clear 校准样本定位无效"):
            fit_clear_calibration(
                [result_row("seed_42_left", predicted=None)],
                "invalid.jsonl",
            )

    def test_rejects_missing_truth_in_calibration_input(self):
        row = result_row("seed_42_left")
        row["true_block_pos"] = None
        with self.assertRaisesRegex(ValueError, "true_block_pos 非法"):
            fit_clear_calibration([row], "invalid-truth.jsonl")


class ApplyFrozenCalibrationTests(unittest.TestCase):
    def test_applies_frozen_xy_correction_without_changing_z_or_raw_fields(self):
        calibration = {
            "calibration_source": "old.jsonl",
            "calibration_sample_ids": ["seed_42_left"],
            "correction_x": 0.03,
            "correction_y": -0.02,
        }
        validation = [
            result_row(
                "seed_47_left",
                predicted=(0.17, 0.42, 0.0),
                truth=(0.20, 0.40, 0.05),
            )
        ]

        corrected = apply_frozen_calibration(calibration, validation)

        self.assertEqual(corrected[0]["predicted_target_world"], [0.17, 0.42, 0.0])
        self.assertEqual(corrected[0]["corrected_target_world"], [0.20, 0.40, 0.0])
        self.assertAlmostEqual(corrected[0]["corrected_error_xy"], 0.0)
        self.assertAlmostEqual(corrected[0]["correction_x_applied"], 0.03)
        self.assertAlmostEqual(corrected[0]["correction_y_applied"], -0.02)

    def test_rejects_overlapping_calibration_and_validation_ids(self):
        calibration = {
            "calibration_sample_ids": ["seed_42_left"],
            "correction_x": 0.03,
            "correction_y": -0.02,
        }
        with self.assertRaisesRegex(ValueError, "sample_id 重叠"):
            apply_frozen_calibration(
                calibration,
                [result_row("seed_42_left")],
            )

    def test_preserves_failed_validation_as_failed(self):
        calibration = {
            "calibration_sample_ids": ["seed_42_left"],
            "correction_x": 0.03,
            "correction_y": -0.02,
        }
        corrected = apply_frozen_calibration(
            calibration,
            [result_row("seed_47_left", predicted=None)],
        )
        self.assertIsNone(corrected[0]["corrected_target_world"])
        self.assertIsNone(corrected[0]["corrected_error_xy"])

    def test_rejects_missing_truth_even_when_prediction_failed(self):
        calibration = {
            "calibration_sample_ids": ["seed_42_left"],
            "correction_x": 0.03,
            "correction_y": -0.02,
        }
        row = result_row("seed_47_left", predicted=None)
        row["true_block_pos"] = None
        with self.assertRaisesRegex(ValueError, "true_block_pos 非法"):
            apply_frozen_calibration(calibration, [row])
```

- [ ] **Step 2: 运行核心测试，确认先失败**

Run:

```bash
conda run -n vla_env python -m unittest \
  tests.test_validate_grounding_calibration.FitClearCalibrationTests \
  tests.test_validate_grounding_calibration.ApplyFrozenCalibrationTests -v
```

Expected: ERROR，`validate_grounding_calibration` 模块尚不存在。

- [ ] **Step 3: 实现输入验证、校准拟合和冻结应用**

创建 `validate_grounding_calibration.py`，实现以下接口和规则：

```python
"""冻结清晰 grounding 的固定偏差，并在独立反投影结果上验证。"""

import argparse
import json
import math
import statistics
from pathlib import Path


VALID_VISIBILITY_GROUPS = {"clear", "partial", "severe"}
PASS_THRESHOLD_METERS = 0.03


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _index_unique(rows, label):
    indexed = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{label} 缺少非空 sample_id")
        if sample_id in indexed:
            raise ValueError(f"{label} sample_id 重复: {sample_id}")
        group = row.get("visibility_group")
        if group not in VALID_VISIBILITY_GROUPS:
            raise ValueError(f"{label} visibility_group 非法: {sample_id}")
        indexed[sample_id] = row
    return indexed


def _finite_number(value, field_name, sample_id):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{sample_id} {field_name} 必须是有限数值")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{sample_id} {field_name} 必须是有限数值")
    return value


def _validated_truth_xy(row):
    truth = row.get("true_block_pos")
    if not isinstance(truth, list) or len(truth) < 2:
        raise ValueError(f"{row['sample_id']} true_block_pos 非法")
    return (
        _finite_number(truth[0], "true_block_pos.x", row["sample_id"]),
        _finite_number(truth[1], "true_block_pos.y", row["sample_id"]),
    )


def _validated_prediction_xyz(row):
    predicted = row.get("predicted_target_world")
    if predicted is None:
        return None
    if not isinstance(predicted, list) or len(predicted) < 3:
        raise ValueError(f"{row['sample_id']} predicted_target_world 非法")
    return [
        _finite_number(value, "predicted_target_world", row["sample_id"])
        for value in predicted[:3]
    ]


def fit_clear_calibration(rows, source_path):
    indexed = _index_unique(rows, "calibration")
    for row in indexed.values():
        _validated_truth_xy(row)
        _validated_prediction_xyz(row)
    clear_rows = [row for row in indexed.values() if row["visibility_group"] == "clear"]
    if not clear_rows:
        raise ValueError("clear 校准样本不能为空")
    for row in clear_rows:
        if row.get("localization_error_xy") is None:
            raise ValueError(f"clear 校准样本定位无效: {row['sample_id']}")
    signed_x = [
        _finite_number(row.get("signed_error_x"), "signed_error_x", row["sample_id"])
        for row in clear_rows
    ]
    signed_y = [
        _finite_number(row.get("signed_error_y"), "signed_error_y", row["sample_id"])
        for row in clear_rows
    ]
    mean_x = statistics.fmean(signed_x)
    mean_y = statistics.fmean(signed_y)
    return {
        "calibration_source": str(source_path),
        "num_clear_calibration_samples": len(clear_rows),
        "calibration_sample_ids": sorted(row["sample_id"] for row in clear_rows),
        "signed_error_x_mean": mean_x,
        "signed_error_y_mean": mean_y,
        "correction_x": -mean_x,
        "correction_y": -mean_y,
    }


def apply_frozen_calibration(calibration, validation_rows):
    validation = _index_unique(validation_rows, "validation")
    calibration_ids = set(calibration.get("calibration_sample_ids", []))
    overlap = sorted(calibration_ids & set(validation))
    if overlap:
        raise ValueError(f"校准集和验证集 sample_id 重叠: {overlap}")
    correction_x = _finite_number(calibration.get("correction_x"), "correction_x", "calibration")
    correction_y = _finite_number(calibration.get("correction_y"), "correction_y", "calibration")
    corrected_rows = []
    for row in validation.values():
        true_x, true_y = _validated_truth_xy(row)
        predicted_values = _validated_prediction_xyz(row)
        corrected = dict(row)
        corrected.update(
            {
                "correction_x_applied": correction_x,
                "correction_y_applied": correction_y,
                "corrected_target_world": None,
                "corrected_signed_error_x": None,
                "corrected_signed_error_y": None,
                "corrected_error_xy": None,
            }
        )
        if predicted_values is not None:
            corrected_x = predicted_values[0] + correction_x
            corrected_y = predicted_values[1] + correction_y
            signed_x = corrected_x - true_x
            signed_y = corrected_y - true_y
            corrected.update(
                {
                    "corrected_target_world": [corrected_x, corrected_y, predicted_values[2]],
                    "corrected_signed_error_x": signed_x,
                    "corrected_signed_error_y": signed_y,
                    "corrected_error_xy": math.hypot(signed_x, signed_y),
                }
            )
        corrected_rows.append(corrected)
    return corrected_rows
```

- [ ] **Step 4: 运行核心测试并确认通过**

Run:

```bash
conda run -n vla_env python -m unittest \
  tests.test_validate_grounding_calibration.FitClearCalibrationTests \
  tests.test_validate_grounding_calibration.ApplyFrozenCalibrationTests -v
```

Expected: 7 tests PASS。

- [ ] **Step 5: 提交冻结校准核心**

```bash
git add validate_grounding_calibration.py tests/test_validate_grounding_calibration.py
git commit -m "feat: apply frozen grounding calibration"
```

---

### Task 3: 分组摘要、通过判定与证据文件

**Files:**
- Modify: `validate_grounding_calibration.py`
- Modify: `tests/test_validate_grounding_calibration.py`

**Interfaces:**
- Consumes: `apply_frozen_calibration()` 返回的逐样本行。
- Produces: `summarize_validation(rows, calibration) -> dict`；`write_outputs(output_dir, calibration, rows, summary)`；CLI 参数 `--calibration-results`、`--validation-results`、`--output-dir`。

- [ ] **Step 1: 写摘要门槛和输出文件的失败测试**

在测试文件中新增：

```python
import json
import tempfile
from pathlib import Path

from validate_grounding_calibration import (
    summarize_validation,
    write_outputs,
)


class SummarizeValidationTests(unittest.TestCase):
    def calibration(self):
        return {
            "calibration_source": "old.jsonl",
            "num_clear_calibration_samples": 15,
            "calibration_sample_ids": ["seed_42_left"],
            "signed_error_x_mean": -0.03,
            "signed_error_y_mean": 0.02,
            "correction_x": 0.03,
            "correction_y": -0.02,
        }

    def corrected(self, rows):
        return apply_frozen_calibration(self.calibration(), rows)

    def test_passes_only_when_every_clear_sample_is_valid_and_within_3cm(self):
        rows = self.corrected(
            [
                result_row("seed_47_left", predicted=(0.17, 0.42, 0.0)),
                result_row("seed_47_right", predicted=(0.19, 0.42, 0.0)),
                result_row(
                    "seed_47_back",
                    visibility_group="severe",
                    predicted=(0.50, 0.10, 0.0),
                ),
            ]
        )
        summary = summarize_validation(rows, self.calibration())
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["num_validation_samples"], 3)
        self.assertEqual(summary["overall"]["num_samples"], 3)
        self.assertEqual(summary["per_visibility_group"]["clear"]["num_within_3cm"], 2)
        self.assertEqual(summary["per_visibility_group"]["clear"]["within_3cm_rate"], 1.0)

    def test_fails_for_clear_error_over_3cm(self):
        rows = self.corrected(
            [result_row("seed_47_left", predicted=(0.10, 0.42, 0.0))]
        )
        summary = summarize_validation(rows, self.calibration())
        self.assertFalse(summary["passed"])
        self.assertIn("clear 样本补偿后误差超过 0.03m", summary["failure_reasons"])

    def test_fails_for_missing_or_invalid_clear_samples(self):
        no_clear = self.corrected(
            [result_row("seed_47_back", visibility_group="severe")]
        )
        invalid_clear = self.corrected(
            [result_row("seed_47_left", predicted=None)]
        )
        self.assertFalse(summarize_validation(no_clear, self.calibration())["passed"])
        self.assertFalse(summarize_validation(invalid_clear, self.calibration())["passed"])

    def test_writes_three_traceable_output_files(self):
        rows = self.corrected(
            [result_row("seed_47_left", predicted=(0.17, 0.42, 0.0))]
        )
        summary = summarize_validation(rows, self.calibration())
        with tempfile.TemporaryDirectory() as temp_dir:
            write_outputs(temp_dir, self.calibration(), rows, summary)
            output = Path(temp_dir)
            self.assertTrue((output / "calibration.json").is_file())
            self.assertTrue((output / "corrected_validation_results.jsonl").is_file())
            self.assertTrue((output / "calibration_validation_summary.json").is_file())
            saved = json.loads(
                (output / "calibration_validation_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["pass_threshold_meters"], 0.03)
```

- [ ] **Step 2: 运行摘要测试，确认先失败**

Run:

```bash
conda run -n vla_env python -m unittest \
  tests.test_validate_grounding_calibration.SummarizeValidationTests -v
```

Expected: ERROR，`summarize_validation` 和 `write_outputs` 尚不存在。

- [ ] **Step 3: 实现统一分组指标与严格通过判定**

在 `validate_grounding_calibration.py` 新增：

```python
def _summarize_group(rows):
    rows = list(rows)
    valid = [row for row in rows if row.get("corrected_error_xy") is not None]
    raw_errors = [row["localization_error_xy"] for row in valid]
    corrected_errors = [row["corrected_error_xy"] for row in valid]
    within = [error for error in corrected_errors if error <= PASS_THRESHOLD_METERS]
    return {
        "num_samples": len(rows),
        "num_valid": len(valid),
        "num_failed": len(rows) - len(valid),
        "raw_error_xy_mean": statistics.fmean(raw_errors) if raw_errors else None,
        "raw_error_xy_median": statistics.median(raw_errors) if raw_errors else None,
        "raw_error_xy_max": max(raw_errors) if raw_errors else None,
        "corrected_error_xy_mean": statistics.fmean(corrected_errors) if corrected_errors else None,
        "corrected_error_xy_median": statistics.median(corrected_errors) if corrected_errors else None,
        "corrected_error_xy_max": max(corrected_errors) if corrected_errors else None,
        "num_within_3cm": len(within),
        "within_3cm_rate": len(within) / len(rows) if rows else 0.0,
    }


def summarize_validation(rows, calibration):
    rows = list(rows)
    groups = {
        group: _summarize_group(
            row for row in rows if row["visibility_group"] == group
        )
        for group in ("clear", "partial", "severe")
    }
    clear = groups["clear"]
    reasons = []
    if clear["num_samples"] == 0:
        reasons.append("验证集中没有 clear 样本")
    if clear["num_failed"]:
        reasons.append("存在定位失败的 clear 样本")
    if clear["num_valid"] and clear["num_within_3cm"] != clear["num_samples"]:
        reasons.append("clear 样本补偿后误差超过 0.03m")
    return {
        "calibration_source": calibration["calibration_source"],
        "num_clear_calibration_samples": calibration["num_clear_calibration_samples"],
        "frozen_correction": {
            "x": calibration["correction_x"],
            "y": calibration["correction_y"],
        },
        "pass_threshold_meters": PASS_THRESHOLD_METERS,
        "num_validation_samples": len(rows),
        "num_valid": sum(row.get("corrected_error_xy") is not None for row in rows),
        "num_failed": sum(row.get("corrected_error_xy") is None for row in rows),
        "overall": _summarize_group(rows),
        "per_visibility_group": groups,
        "passed": not reasons,
        "failure_reasons": reasons,
    }
```

- [ ] **Step 4: 实现三个输出文件和 CLI**

继续新增：

```python
def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(output_dir, calibration, rows, summary):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "calibration.json").open("w", encoding="utf-8") as handle:
        json.dump(calibration, handle, ensure_ascii=False, indent=2)
    write_jsonl(output_dir / "corrected_validation_results.jsonl", rows)
    with (output_dir / "calibration_validation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-results", required=True)
    parser.add_argument("--validation-results", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    calibration_rows = read_jsonl(args.calibration_results)
    validation_rows = read_jsonl(args.validation_results)
    calibration = fit_clear_calibration(calibration_rows, args.calibration_results)
    corrected = apply_frozen_calibration(calibration, validation_rows)
    summary = summarize_validation(corrected, calibration)
    write_outputs(args.output_dir, calibration, corrected, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行新模块全部测试并确认通过**

Run:

```bash
conda run -n vla_env python -m unittest \
  tests.test_validate_grounding_calibration -v
```

Expected: 全部 PASS；输出文件名、冻结补偿、clear 严格门槛和遮挡分组均有断言保护。

- [ ] **Step 6: 提交摘要和 CLI**

```bash
git add validate_grounding_calibration.py tests/test_validate_grounding_calibration.py
git commit -m "feat: report calibration validation metrics"
```

---

### Task 4: 全量自动测试与真实独立验证实验

**Files:**
- Generate: `vlm_eval_samples_448_calibration_validation_d020/`
- Generate: `vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1/`
- Generate: `vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1/backprojection/`
- Generate: `vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1/calibration_validation/`

**Interfaces:**
- Consumes: Task 1 配置、Task 2–3 CLI、既有校准结果 `vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/backprojection/backprojection_results.jsonl`。
- Produces: 20 张验证图片、20 条 diagnostics、20 条 Qwen 尝试记录、原始反投影结果和冻结补偿验证摘要。

- [ ] **Step 1: 运行完整测试套件**

Run:

```bash
conda run -n vla_env python -m unittest discover -s tests -v
```

Expected: 所有测试 PASS，且测试总数大于当前基线 85。

- [ ] **Step 2: 生成 seeds 47–51 的 20 张独立样本**

Run:

```bash
conda run -n vla_env python collect_vlm_eval_samples.py
```

Expected: `vlm_eval_samples_448_calibration_validation_d020/samples.jsonl` 与 `diagnostics.jsonl` 各 20 行，图片目录有 20 张可读 JPEG。

Run evidence check:

```bash
conda run -n vla_env python -c "import json; from pathlib import Path; root=Path('vlm_eval_samples_448_calibration_validation_d020'); samples=[json.loads(x) for x in (root/'samples.jsonl').read_text().splitlines() if x.strip()]; diagnostics=[json.loads(x) for x in (root/'diagnostics.jsonl').read_text().splitlines() if x.strip()]; images=list((root/'images').glob('*.jpg')); assert len(samples)==len(diagnostics)==len(images)==20; assert {r['random_seed'] for r in samples}=={47,48,49,50,51}; assert {r['sample_id'] for r in samples}=={r['sample_id'] for r in diagnostics}; print('samples=20 diagnostics=20 images=20 seeds=47-51')"
```

Expected: `samples=20 diagnostics=20 images=20 seeds=47-51`。

- [ ] **Step 3: 人工查看 20 张原图和可见率分布**

检查目录：

```text
/home/pzk/vla_project/vlm_eval_samples_448_calibration_validation_d020/images
```

Run:

```bash
conda run -n vla_env python -c "import json,collections; from pathlib import Path; rows=[json.loads(x) for x in Path('vlm_eval_samples_448_calibration_validation_d020/diagnostics.jsonl').read_text().splitlines() if x.strip()]; groups=collections.Counter('clear' if r['block_visibility_ratio']>=.75 else 'partial' if r['block_visibility_ratio']>=.25 else 'severe' for r in rows); print(dict(sorted(groups.items())))"
```

Expected: 总数为 20；只记录实际 clear/partial/severe 分布，不因遮挡删除或重采样图片。

- [ ] **Step 4: 对全部验证图片运行 Qwen grounding**

Run:

```bash
conda run -n vla_env python diagnose_vlm_grounding.py --limit 20
```

Expected: 运行目录为 `vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1`，`grounding_predictions.jsonl` 对 20 个 `sample_id` 各有一条成功或明确失败记录。重复执行会跳过已有 ID。

- [ ] **Step 5: 运行原始反投影评分**

Run:

```bash
conda run -n vla_env python evaluate_grounding_backprojection.py \
  --predictions vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1/grounding_predictions.jsonl \
  --diagnostics vlm_eval_samples_448_calibration_validation_d020/diagnostics.jsonl \
  --output-dir vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1/backprojection
```

Expected: 生成 20 条 `backprojection_results.jsonl` 以及含整体和可见率分组的 `backprojection_summary.json`。

- [ ] **Step 6: 应用旧校准集冻结补偿并独立判定**

Run:

```bash
conda run -n vla_env python validate_grounding_calibration.py \
  --calibration-results vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/backprojection/backprojection_results.jsonl \
  --validation-results vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1/backprojection/backprojection_results.jsonl \
  --output-dir vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1/calibration_validation
```

Expected:

```text
calibration.json
corrected_validation_results.jsonl
calibration_validation_summary.json
```

其中 `calibration.json` 必须保存 15 个 clear 校准 ID，补偿约为 `x=+0.02492227406480192m`、`y=-0.019467343494422532m`；最终是否通过只读取摘要的 `passed`，不根据主观观感改阈值。

- [ ] **Step 7: 运行证据一致性审计**

Run:

```bash
conda run -n vla_env python -c "import json; from pathlib import Path; root=Path('vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1'); calibration=json.loads((root/'calibration_validation/calibration.json').read_text()); summary=json.loads((root/'calibration_validation/calibration_validation_summary.json').read_text()); corrected=[json.loads(x) for x in (root/'calibration_validation/corrected_validation_results.jsonl').read_text().splitlines() if x.strip()]; assert calibration['num_clear_calibration_samples']==15; assert len(corrected)==20; assert summary['num_validation_samples']==20; assert not ({r['sample_id'] for r in corrected} & set(calibration['calibration_sample_ids'])); assert summary['per_visibility_group']['clear']['num_samples']>0; print(json.dumps(summary, ensure_ascii=False, indent=2))"
```

Expected: 断言全部通过并打印最终真实摘要。若 `passed` 为 false，实验仍算成功完成，但固定补偿不得接入在线控制。

- [ ] **Step 8: 确认实验证据按仓库策略保存在本地**

样本、图片、API 原始回复和 JSON/JSONL 结果继续由 `.gitignore` 排除，避免把批量生成物和可能含服务返回内容的文件强制写入 Git。Task 5 只在四份受版本控制的文档中记录指标及本地证据路径。

Run:

```bash
git check-ignore -v \
  vlm_eval_samples_448_calibration_validation_d020/samples.jsonl \
  vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1/calibration_validation/calibration_validation_summary.json
```

Expected: 两条路径分别被 `.gitignore` 的 VLM 样本规则和 `vlm_eval_runs/` 规则命中；不执行 `git add -f`。

---

### Task 5: 同步项目结论和下一步方向

**Files:**
- Modify: `README.md:13-40`
- Modify: `docs/worklog/WORKLOG.md:599-end`
- Modify: `docs/planning/vla_robotic_study_plan.md:48-60,300-305`
- Modify: `docs/debugging/BUGLOG.md:603-637`

**Interfaces:**
- Consumes: `calibration_validation_summary.json` 的真实数值与 `passed`。
- Produces: 四份文档使用同一组样本数、可见率分组、raw/corrected mean/median/max、3cm 通过率和阶段结论。

- [ ] **Step 1: 提取唯一可信的文档指标**

Run:

```bash
conda run -n vla_env python -c "import json; from pathlib import Path; p=Path('vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_calibration_validation_v1/calibration_validation/calibration_validation_summary.json'); s=json.loads(p.read_text()); c=s['per_visibility_group']['clear']; print('validation=',s['num_validation_samples']); print('valid=',s['num_valid']); print('clear=',c['num_samples']); print('raw_mean_median_max=',c['raw_error_xy_mean'],c['raw_error_xy_median'],c['raw_error_xy_max']); print('corrected_mean_median_max=',c['corrected_error_xy_mean'],c['corrected_error_xy_median'],c['corrected_error_xy_max']); print('within_3cm=',c['num_within_3cm'],c['within_3cm_rate']); print('passed=',s['passed']); print('failure_reasons=',s['failure_reasons'])"
```

Expected: 输出一组由摘要直接读取的真实指标；四份文档只允许引用这组值。

- [ ] **Step 2: 按判定结果更新 README 当前阶段**

在 `README.md` 写清：

```text
校准集：seeds 42–46，只用 clear 样本拟合一次固定 XY 补偿。
验证集：seeds 47–51，共 20 张，未参与拟合。
结果：分别列出 clear 的原始与补偿后 mean/median/max、<=3cm 数量和 passed。
边界：partial/severe 单独报告；离线真值只评分；当前仍未把补偿接入在线控制。
```

如果 `passed == true`，下一步写为“设计一次只读目标坐标到控制器的闭环 smoke test”；如果为 false，下一步写为“检查验证残差方向性，比较全局平移补偿与位置相关校准，不接入控制器”。

- [ ] **Step 3: 更新 WORKLOG、学习计划和 BUGLOG**

`WORKLOG.md` 新增一节，必须包含实验动机、冻结方式、20 张验证数据路径、原始/补偿指标、遮挡分组、是否通过和工程解释。

`vla_robotic_study_plan.md` 把“建立独立校准集与验证集”从下一步改成已完成实验，并按 `passed` 选择与 README 相同的唯一下一步。

`BUGLOG.md` 在 BUG-003 中追加独立验证证据；若失败，状态保持未解决并记录残差模式；若通过，状态改为“固定偏差在独立 clear 样本上通过离线验证，在线控制仍未验证”。

- [ ] **Step 4: 检查四份文档没有指标漂移或未完成标记**

Run:

```bash
rg -n "校准|验证集|3cm|passed|补偿|下一步" \
  README.md docs/worklog/WORKLOG.md \
  docs/planning/vla_robotic_study_plan.md docs/debugging/BUGLOG.md
rg -n "T[B]D|T[O]DO|待.定|占.位" \
  README.md docs/worklog/WORKLOG.md \
  docs/planning/vla_robotic_study_plan.md docs/debugging/BUGLOG.md
git diff --check
```

Expected: 第一条能定位四份文档中的一致结论；第二条无输出；`git diff --check` 无输出。

- [ ] **Step 5: 最终验证**

Run:

```bash
conda run -n vla_env python -m unittest discover -s tests -v
git status --short
```

Expected: 全部测试 PASS；状态只包含本任务预期的四份文档修改。

- [ ] **Step 6: 提交文档证据**

```bash
git add README.md docs/worklog/WORKLOG.md \
  docs/planning/vla_robotic_study_plan.md docs/debugging/BUGLOG.md
git commit -m "docs: analyze grounding calibration validation"
```

---

## 完成判据

- 自动测试全部通过，且新增测试覆盖显式 seeds、冻结补偿、数据泄漏、空 clear、clear 定位失败、超过 3cm 和遮挡隔离。
- seeds 47–51 的 20 张图片、20 条 diagnostics、20 条 grounding 尝试和 20 条反投影结果一一对应。
- `calibration.json` 的校准 ID 全部来自 seeds 42–46，验证 ID 全部来自 seeds 47–51，交集为空。
- clear 组是否通过完全由 `calibration_validation_summary.json` 的严格规则决定。
- 无论通过或失败，都分析真实结果并给出唯一下一步；未通过时绝不接入在线控制。
- README、WORKLOG、学习计划、BUGLOG 的数值、结论和下一步一致。
