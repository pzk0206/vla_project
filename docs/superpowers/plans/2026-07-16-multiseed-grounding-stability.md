# Multi-Seed Grounding Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and evaluate 20 fixed-camera grounding samples from five reproducible red-block positions, with resumable Qwen calls and summaries grouped by arm direction and random seed.

**Architecture:** Reuse the existing stratified balanced-pose collector with seeds 42–46 and versioned output directories so the four-sample baseline remains untouched. Extend grounding predictions with the sample seed, make paid grounding calls resumable by `sample_id`, then extend backprojection summaries with per-direction metrics and per-seed pixel-center jitter.

**Tech Stack:** Python 3, `unittest`, PyBullet, OpenCV, NumPy, YAML, Qwen OpenAI-compatible API, JSON/JSONL, conda environment `vla_env`.

## Global Constraints

- Keep the fixed top-down camera, `448 x 448` images, `0.20m` XY offset, Qwen model, grounding prompt, and plane backprojection unchanged.
- Use seeds 42–46; each seed produces left, right, front, and back, for exactly 20 samples.
- Do not overwrite `vlm_eval_samples_448/` or `grounding_qwen3_vl_flash_distance20_448_v3/`.
- Keep `block_pos` and camera matrices out of `samples.jsonl`; they remain in `diagnostics.jsonl` only.
- Do not add calibration offsets or connect localization to online control in this plan.
- Run every Python test and script through `conda run -n vla_env`.

---

### Task 1: Version the 20-sample experiment configuration

**Files:**
- Modify: `tests/test_config_contract.py`
- Modify: `tests/test_collect_vlm_eval_samples.py`
- Modify: `sim_config.yaml`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `collect_vlm_eval_samples(config) -> tuple[Path, list[dict]]` and the existing `stratified_balanced_poses` strategy.
- Produces: a config that writes 20 samples to `vlm_eval_samples_448_multiseed_d020/` and grounding results to `vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/`.

- [ ] **Step 1: Tighten the configuration contract first**

Replace the old output and positive-only seed assertions in `test_vlm_evaluation_config_is_valid` with:

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

Keep the existing assertions for strategy, `balanced_pose_offsets_xy == [0.20]`, camera height, and `448 x 448` resolution.

- [ ] **Step 2: Run the contract test and verify RED**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_config_contract.ConfigContractTests.test_vlm_evaluation_config_is_valid -v
```

Expected: FAIL because the current config still names `vlm_eval_samples_448` and has `stratified_num_seeds: 1`.

- [ ] **Step 3: Apply the minimal configuration change**

Set these exact values in `sim_config.yaml`:

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

Add the generated sample directory to `.gitignore`:

```gitignore
vlm_eval_samples_448_multiseed_d020/
```

- [ ] **Step 4: Strengthen the mocked collection regression test**

In `test_balanced_collection_writes_matching_diagnostics`, set `stratified_num_seeds` to `5`, then replace the four-sample assertion with:

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

This regression test is expected to pass without collector production changes because the existing seed loop already supports multiple seeds.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_config_contract tests.test_collect_vlm_eval_samples -v
```

Expected: PASS, with the mocked balanced collection producing 20 unique samples.

- [ ] **Step 6: Commit the configuration slice**

```bash
git add sim_config.yaml .gitignore tests/test_config_contract.py tests/test_collect_vlm_eval_samples.py
git commit -m "test: configure multiseed grounding samples"
```

---

### Task 2: Make grounding calls resumable and preserve the sample seed

**Files:**
- Modify: `tests/test_diagnose_vlm_grounding.py`
- Modify: `diagnose_vlm_grounding.py`

**Interfaces:**
- Consumes: `samples.jsonl` rows containing `sample_id`, `random_seed`, image path, instruction, and expected direction.
- Produces: `load_existing_predictions(path) -> dict[str, dict]` and one unique `grounding_predictions.jsonl` row per completed `sample_id`, including `random_seed`.

- [ ] **Step 1: Write a failing resume test**

Add imports for `json`, `tempfile`, `Path`, `patch`, and `cv2`, import `diagnose_grounding`, then add:

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

- [ ] **Step 2: Run the resume test and verify RED**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_diagnose_vlm_grounding.GroundingResumeTests -v
```

Expected: FAIL because the API is called twice and the result lacks `random_seed`.

- [ ] **Step 3: Implement unique loading and skip completed samples**

Add:

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

After `results_path` is created in `diagnose_grounding`, replace the current
`results = []` initialization with:

```python
existing = load_existing_predictions(results_path)
```

Add this guard as the first statement inside the existing sample loop:

```python
for sample in samples:
    if sample["sample_id"] in existing:
        continue
```

Keep the current image read, API `try/except`, and result construction inside
that loop. Add the seed directly to the current result dictionary:

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

After the loop, construct the return list in manifest order:

```python
results = [existing[sample["sample_id"]] for sample in samples]

return run_dir, results
```

- [ ] **Step 4: Run grounding tests and verify GREEN**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_diagnose_vlm_grounding -v
```

Expected: PASS and the resume test reports one API call across two runs.

- [ ] **Step 5: Commit resumable grounding**

```bash
git add diagnose_vlm_grounding.py tests/test_diagnose_vlm_grounding.py
git commit -m "feat: resume grounding evaluation by sample id"
```

---

### Task 3: Summarize localization error by direction and seed

**Files:**
- Modify: `tests/test_evaluate_grounding_backprojection.py`
- Modify: `evaluate_grounding_backprojection.py`

**Interfaces:**
- Consumes: prediction rows containing `expected_direction` and `random_seed`, plus diagnostics joined by `sample_id`.
- Produces: result rows preserving those fields; summary keys `per_direction` and `per_random_seed`, where every group contains validity/error metrics and seed groups also contain pixel-center jitter.

- [ ] **Step 1: Write failing grouped-summary tests**

Add `expected_direction` and `random_seed` to the valid result dictionaries in `test_summarizes_valid_errors_and_failure_types`, then add:

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

In `test_computes_zero_xy_error_for_matching_box_center`, put these fields on the prediction and assert they survive evaluation:

```python
"expected_direction": "left",
"random_seed": 42,
```

```python
self.assertEqual(result["expected_direction"], "left")
self.assertEqual(result["random_seed"], 42)
```

- [ ] **Step 2: Run the grouped tests and verify RED**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_evaluate_grounding_backprojection -v
```

Expected: FAIL because result rows do not preserve seed/direction and summaries lack group keys.

- [ ] **Step 3: Preserve grouping metadata in result rows**

Add these keys to `base_result` in `evaluate_rows`:

```python
"expected_direction": prediction.get("expected_direction"),
"random_seed": prediction.get("random_seed"),
```

- [ ] **Step 4: Implement reusable group metrics and pixel jitter**

Extract the existing overall calculations into:

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

Build grouped output in `summarize_results`:

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

- [ ] **Step 5: Run focused and full tests**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_evaluate_grounding_backprojection -v
conda run -n vla_env python -m unittest discover -s tests -v
```

Expected: focused tests PASS; the full suite passes with no regression.

- [ ] **Step 6: Commit grouped analysis**

```bash
git add evaluate_grounding_backprojection.py tests/test_evaluate_grounding_backprojection.py
git commit -m "feat: summarize grounding stability by pose and seed"
```

---

### Task 4: Generate and validate the 20 local samples

**Files:**
- Generate, ignored: `vlm_eval_samples_448_multiseed_d020/images/*.jpg`
- Generate, ignored: `vlm_eval_samples_448_multiseed_d020/samples.jsonl`
- Generate, ignored: `vlm_eval_samples_448_multiseed_d020/diagnostics.jsonl`

**Interfaces:**
- Consumes: the Task 1 config and existing PyBullet balanced-pose collector.
- Produces: exactly 20 image/sample/diagnostic records for Tasks 5 and 6.

- [ ] **Step 1: Generate samples without calling Qwen**

Run:

```bash
conda run -n vla_env python collect_vlm_eval_samples.py
```

Expected: exit 0 and `样本数量: 20`.

- [ ] **Step 2: Validate counts, IDs, seeds, directions, and truth isolation**

Run:

```bash
conda run -n vla_env python -c 'import json,pathlib,collections; root=pathlib.Path("vlm_eval_samples_448_multiseed_d020"); samples=[json.loads(x) for x in (root/"samples.jsonl").read_text().splitlines() if x.strip()]; diagnostics=[json.loads(x) for x in (root/"diagnostics.jsonl").read_text().splitlines() if x.strip()]; assert len(samples)==len(diagnostics)==20; assert {x["sample_id"] for x in samples}=={x["sample_id"] for x in diagnostics}; assert {x["random_seed"] for x in samples}==set(range(42,47)); assert collections.Counter(x["expected_direction"] for x in samples)==collections.Counter({"left":5,"right":5,"front":5,"back":5}); assert all("block_pos" not in x and "view_matrix" not in x for x in samples); assert all((pathlib.Path(x["image_path"])).is_file() for x in samples); print("validated", len(samples), collections.Counter(x["expected_direction"] for x in samples))'
```

Expected: `validated 20 Counter({'left': 5, 'right': 5, 'front': 5, 'back': 5})`.

- [ ] **Step 3: Visually inspect one image from each seed**

Open `seed_42_d020_left.jpg` through `seed_46_d020_left.jpg` from the generated images directory and confirm the red block is visible and remains in the camera field of view. Do not continue to paid calls if any target is clipped or hidden.

---

### Task 5: Run resumable Qwen grounding and backprojection

**Files:**
- Generate, ignored: `vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/grounding_predictions.jsonl`
- Generate, ignored: `vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/annotated/*.jpg`
- Generate, ignored: `vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/backprojection/backprojection_results.jsonl`
- Generate, ignored: `vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/backprojection/backprojection_summary.json`

**Interfaces:**
- Consumes: 20 samples and diagnostics from Task 4 plus API settings from `sim_config.yaml` and environment variables `VLA_API_BASE_URL`, `VLA_API_KEY`, and `VLA_MODEL_NAME`.
- Produces: 20 saved grounding outcomes and a grouped localization summary.

- [ ] **Step 1: Verify API configuration without printing secrets**

Run:

```bash
conda run -n vla_env python -c 'import os; names=("VLA_API_BASE_URL","VLA_API_KEY","VLA_MODEL_NAME"); missing=[name for name in names if not os.getenv(name)]; assert not missing, f"missing env vars: {missing}"; print("API environment ready")'
```

Expected: `API environment ready`. Never print the variable values.

- [ ] **Step 2: Run all 20 grounding calls with resume support**

Run:

```bash
conda run -n vla_env python diagnose_vlm_grounding.py --limit 20
```

Expected: the command reports `成功生成目标框: N/20`; each attempted sample is immediately saved. If interrupted, run the same command again and completed `sample_id` values are skipped.

- [ ] **Step 3: Verify unique saved outcomes**

Run:

```bash
conda run -n vla_env python -c 'import json,pathlib; path=pathlib.Path("vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/grounding_predictions.jsonl"); rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]; ids=[x["sample_id"] for x in rows]; assert len(rows)==len(set(ids))==20; assert {x["random_seed"] for x in rows}==set(range(42,47)); print("unique grounding rows", len(rows), "valid", sum(x["boxes"] is not None for x in rows))'
```

Expected: `unique grounding rows 20 valid N`, with no duplicate IDs.

- [ ] **Step 4: Run plane backprojection and grouped statistics**

Run:

```bash
conda run -n vla_env python evaluate_grounding_backprojection.py --predictions vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/grounding_predictions.jsonl --diagnostics vlm_eval_samples_448_multiseed_d020/diagnostics.jsonl --output-dir vlm_eval_runs/grounding_qwen3_vl_flash_distance20_448_multiseed_v4/backprojection
```

Expected: exit 0 and JSON containing `num_samples: 20`, `per_direction`, and `per_random_seed`.

- [ ] **Step 5: Classify the evidence without adding compensation**

Read `backprojection_summary.json` and `backprojection_results.jsonl`. Record:

- overall mean, median, maximum, signed X mean, and signed Y mean;
- left/right/front/back mean error and failure count;
- seeds 42–46 pixel-center X/Y spans and maximum pairwise pixel distance;
- whether signed errors keep the same direction across seeds and poses.

Conclusion rule: recommend a later calibration experiment only if error direction and magnitude remain similar across positions and poses. Otherwise recommend improving grounding or running a detector/segmentation ablation. Do not change controller code in either case.

---

### Task 6: Record verified evidence and complete regression checks

**Files:**
- Modify: `README.md`
- Modify: `docs/worklog/WORKLOG.md`
- Modify: `docs/planning/vla_robotic_study_plan.md`
- Modify: `docs/debugging/BUGLOG.md`

**Interfaces:**
- Consumes: the exact JSON values and per-sample evidence generated by Task 5.
- Produces: synchronized project status that distinguishes direction-level grounding from centimeter-level localization.

- [ ] **Step 1: Add one evidence section to the worklog and bug log**

Use the heading `五位置 grounding 反投影稳定性实验（2026-07-16）`. Copy exact values from these JSON keys rather than rounding from terminal text:

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

State explicitly that the camera stayed fixed and only block seed plus arm pose varied. Include the conclusion selected by Task 5's rule and the exact output directory.

- [ ] **Step 2: Synchronize README and study-plan next-step wording**

Update both files to state that `4/4` box-derived direction is direction-level evidence, while the five-position experiment measures centimeter-level localization. Keep the architecture as VLM grounding plus deterministic geometry/control; do not claim calibration is approved unless Task 5 supports it.

- [ ] **Step 3: Run final verification**

Run:

```bash
conda run -n vla_env python -m unittest discover -s tests -v
git diff --check
git status --short
```

Expected: all tests PASS, `git diff --check` prints nothing, generated sample/run directories remain ignored, and only the four documentation files are uncommitted at this point.

- [ ] **Step 4: Commit verified experiment documentation**

```bash
git add README.md docs/worklog/WORKLOG.md docs/planning/vla_robotic_study_plan.md docs/debugging/BUGLOG.md
git commit -m "docs: record multiseed grounding stability evidence"
```

- [ ] **Step 5: Verify the final repository state**

Run:

```bash
git status --short --branch
git log -5 --oneline
```

Expected: clean worktree and commits for configuration, resumable grounding, grouped analysis, and verified experiment documentation. Do not push unless the user explicitly requests publication.
