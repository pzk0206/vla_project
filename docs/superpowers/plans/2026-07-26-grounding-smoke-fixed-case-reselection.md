# Grounding Smoke Fixed Case Reselection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deterministically select one initially qualified `left`, `right`, and `front` smoke case from seeds 55–100, freeze those cases in configuration, and prove all three pass a no-API preflight-only run.

**Architecture:** Refocus the existing screening workflow on one initial observation per candidate, using only start pose error and first-frame visibility as qualification inputs. Extract the runner's existing three-case preflight into a reusable batch boundary and expose it through `--preflight-only`, then use the real no-API outputs to freeze the selected seeds and update project evidence.

**Tech Stack:** Python 3, `unittest`, `unittest.mock`, PyBullet, OpenCV, NumPy, YAML configuration, JSON/JSONL evidence.

## Global Constraints

- Candidate seeds remain exactly the inclusive range 55–100.
- Direction order remains exactly `["left", "right", "front"]`.
- Qualification is inclusive: `start_pose_error <= 0.005m` and `block_visibility_ratio >= 0.75`.
- The three selected seeds must be distinct and must be the smallest eligible seeds under direction-order selection.
- Later-frame visibility and motion target error are not candidate qualification inputs.
- Screening and `--preflight-only` must never call `call_openai_compatible_api()`.
- Do not modify grounding, backprojection, frozen XY correction, action selection, IK behavior, target hold, scoring, or API limits.
- Do not run a real online smoke test; a paid/API-backed run still requires separate explicit user approval.
- Preserve historical screening outputs; every new run uses a unique output directory.
- Preserve the user's untracked `docs/superpowers/plans/2026-07-19-agent-file-placement-rules.md`.

---

### Task 1: Replace Full-Trajectory Qualification with Initial-Observation Qualification

**Files:**
- Modify: `tests/vlm/grounding_smoke/test_screening.py`
- Modify: `src/vla_project/vlm/grounding_smoke/screening.py`
- Modify: `tests/test_config_contract.py`
- Modify: `sim_config.yaml`

**Interfaces:**
- Consumes: `config["grounding_smoke"]`, `config["grounding_smoke"]["screening"]`
- Produces: `classify_candidate(observation: dict | None, settings: dict, error: str | None = None) -> dict`
- Produces: `run_candidate(config: dict, seed: int, direction: str, candidate_dir: Path | str) -> dict`
- Preserves: `select_qualified_cases(candidate_rows: Iterable[dict], directions: Iterable[str]) -> list[dict]`
- Preserves: `run_screening(config: dict, run_name: str | None = None) -> tuple[Path, dict]`

- [ ] **Step 1: Replace trajectory classification tests with initial-boundary tests**

Before writing the tests, name the breaks they catch:

- Changing `>` to `>=` for pose error must fail the equality-boundary test.
- Changing `<` to `<=` for visibility must fail the equality-boundary test.
- Checking visibility before pose error must fail the rejection-precedence test.
- Returning a qualified result without an observation must fail the missing-observation test.

Replace `SETTINGS`, `valid_steps()`, and `CandidateClassificationTests` in
`tests/vlm/grounding_smoke/test_screening.py` with:

```python
SETTINGS = {
    "clear_visibility_threshold": 0.75,
    "max_pose_error": 0.005,
}


def valid_observation():
    return {
        "target_error_3d": 0.004,
        "block_visibility_ratio": 0.80,
    }


class CandidateClassificationTests(unittest.TestCase):
    def test_initial_observation_qualifies_at_inclusive_boundaries(self):
        observation = valid_observation()
        observation["target_error_3d"] = 0.005
        observation["block_visibility_ratio"] = 0.75

        result = classify_candidate(observation, SETTINGS)

        self.assertTrue(result["qualified"])
        self.assertEqual(result["reason"], "qualified")

    def test_start_pose_error_strictly_over_limit_is_rejected(self):
        observation = valid_observation()
        observation["target_error_3d"] = 0.005001

        result = classify_candidate(observation, SETTINGS)

        self.assertFalse(result["qualified"])
        self.assertEqual(result["reason"], "start_pose_error")

    def test_visibility_strictly_below_threshold_is_rejected(self):
        observation = valid_observation()
        observation["block_visibility_ratio"] = 0.749999

        result = classify_candidate(observation, SETTINGS)

        self.assertFalse(result["qualified"])
        self.assertEqual(result["reason"], "visibility_below_threshold")

    def test_start_pose_reason_precedes_visibility_reason(self):
        observation = {
            "target_error_3d": 0.006,
            "block_visibility_ratio": 0.50,
        }

        result = classify_candidate(observation, SETTINGS)

        self.assertEqual(result["reason"], "start_pose_error")

    def test_exception_is_rejected_with_evidence(self):
        result = classify_candidate(None, SETTINGS, error="boom")

        self.assertFalse(result["qualified"])
        self.assertEqual(result["reason"], "candidate_error")
        self.assertEqual(result["error"], "boom")

    def test_missing_observation_is_candidate_error(self):
        result = classify_candidate(None, SETTINGS)

        self.assertFalse(result["qualified"])
        self.assertEqual(result["reason"], "candidate_error")
        self.assertEqual(result["error"], "missing_initial_observation")
```

- [ ] **Step 2: Run the classification tests and verify RED**

Run:

```bash
python -m unittest \
  tests.vlm.grounding_smoke.test_screening.CandidateClassificationTests -v
```

Expected: FAIL because the existing classifier expects an iterable of five observations and checks the old full-trajectory contract.

- [ ] **Step 3: Implement the minimal initial-observation classifier**

Replace `classify_candidate()` in
`src/vla_project/vlm/grounding_smoke/screening.py` with:

```python
def classify_candidate(observation, settings, error=None):
    """按起点姿态和首帧可见率分类一个候选。"""
    if error is not None:
        return {
            "qualified": False,
            "reason": "candidate_error",
            "error": error,
        }
    if observation is None:
        return {
            "qualified": False,
            "reason": "candidate_error",
            "error": "missing_initial_observation",
        }
    if observation["target_error_3d"] > settings["max_pose_error"]:
        return {"qualified": False, "reason": "start_pose_error"}
    if (
        observation["block_visibility_ratio"]
        < settings["clear_visibility_threshold"]
    ):
        return {
            "qualified": False,
            "reason": "visibility_below_threshold",
        }
    return {"qualified": True, "reason": "qualified"}
```

- [ ] **Step 4: Run the classification tests and verify GREEN**

Run:

```bash
python -m unittest \
  tests.vlm.grounding_smoke.test_screening.CandidateClassificationTests -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Replace the motion-based candidate test with a one-frame behavior test**

Before writing the test, name the break it catches: reintroducing any call to
`direction_to_target()`, joint target calculation, joint actuation, or post-move link reads must fail because those names will not be patched and the candidate must complete with one observation.

Replace
`test_candidate_uses_four_real_control_steps_and_five_observations()` with:

```python
def test_candidate_uses_only_the_initial_observation(self):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    segmentation = np.full((20, 20), 20, dtype=np.int64)
    with tempfile.TemporaryDirectory() as temp_dir:
        config = screening_config(temp_dir)
        candidate_dir = Path(temp_dir) / "seed_055_left"
        with (
            patch(
                "vla_project.vlm.grounding_smoke.screening.connect_physics"
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.setup_world",
                return_value=(1, 10),
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.load_block",
                return_value=20,
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.settle_object"
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.get_object_position",
                return_value=[0.0, 0.4, 0.05],
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.build_balanced_ee_positions",
                return_value={"left": [0.1, 0.4, 0.2]},
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.reset_robot_to_target",
                return_value=[0.1, 0.4, 0.2],
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.sample_camera_eye",
                return_value=[0.0, 0.4, 3.0],
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.capture_rgb_and_segmentation",
                return_value=(image, segmentation),
            ) as capture,
            patch(
                "vla_project.vlm.grounding_smoke.screening.cv2.imwrite",
                return_value=True,
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.p.disconnect"
            ) as disconnect,
        ):
            result = run_candidate(config, 55, "left", candidate_dir)

    self.assertTrue(result["qualified"])
    self.assertEqual(result["reason"], "qualified")
    self.assertEqual(result["seed"], 55)
    self.assertEqual(result["direction"], "left")
    self.assertEqual(result["target_error_3d"], 0.0)
    self.assertEqual(result["block_visibility_ratio"], 1.0)
    self.assertTrue(result["image_path"].endswith("step_00.jpg"))
    capture.assert_called_once()
    disconnect.assert_called_once()
    self.assertNotIn(
        "call_openai_compatible_api",
        vla_project.vlm.grounding_smoke.screening.__dict__,
    )
```

Add this cleanup/error test:

```python
def test_candidate_error_is_recorded_and_physics_is_released(self):
    with tempfile.TemporaryDirectory() as temp_dir:
        config = screening_config(temp_dir)
        candidate_dir = Path(temp_dir) / "seed_055_left"
        with (
            patch(
                "vla_project.vlm.grounding_smoke.screening.connect_physics"
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.setup_world",
                side_effect=RuntimeError("setup failed"),
            ),
            patch(
                "vla_project.vlm.grounding_smoke.screening.p.disconnect"
            ) as disconnect,
        ):
            result = run_candidate(config, 55, "left", candidate_dir)

    self.assertFalse(result["qualified"])
    self.assertEqual(result["reason"], "candidate_error")
    self.assertIn("setup failed", result["error"])
    disconnect.assert_called_once()
```

In the existing
`test_screening_evaluates_full_cartesian_product_and_writes_summary()`,
add this behavior assertion so the saved summary freezes the rule used to
produce `selected_cases`:

```python
self.assertEqual(
    summary["selection_rule"],
    {
        "seed_range": [55, 100],
        "directions": ["left", "right", "front"],
        "max_pose_error": 0.005,
        "clear_visibility_threshold": 0.75,
        "observation_scope": "initial_only",
        "distinct_seeds": True,
        "selection": "smallest_seed_in_direction_order",
    },
)
```

- [ ] **Step 6: Run the candidate tests and verify RED**

Run:

```bash
python -m unittest \
  tests.vlm.grounding_smoke.test_screening.ScreeningRunnerTests.test_candidate_uses_only_the_initial_observation \
  tests.vlm.grounding_smoke.test_screening.ScreeningRunnerTests.test_candidate_error_is_recorded_and_physics_is_released -v
```

Expected: the one-frame test FAILS because the existing implementation attempts four movement steps; the error test may already pass and protects cleanup while refactoring.

- [ ] **Step 7: Refactor observation capture and candidate execution**

In `src/vla_project/vlm/grounding_smoke/screening.py`:

1. Remove imports used only by the old motion trajectory:

```python
apply_joint_targets
calculate_target_joints
get_link_position
direction_to_target
```

2. Rename `_capture_observation()` to `_capture_initial_observation()`, remove
`config` and `observation_step` parameters, always save `step_00.jpg`, and keep
the existing requested target, actual EE, true block, distance, visibility,
camera eye, and image-path evidence.

The return must be:

```python
return {
    "requested_target": list(requested_target),
    "actual_ee_pos": list(actual_ee),
    "target_error_3d": math.dist(actual_ee, requested_target),
    "true_block_pos": current_block,
    "true_distance_xy": math.hypot(
        current_block[0] - actual_ee[0],
        current_block[1] - actual_ee[1],
    ),
    **_compute_visibility(
        segmentation,
        block_id,
        smoke_config["visibility_reference_pixels"],
    ),
    "camera_eye": list(camera_eye),
    "image_path": str(image_path),
}
```

3. Replace the movement loop in `run_candidate()` with one call:

```python
observation = _capture_initial_observation(
    smoke,
    candidate_dir,
    requested_target,
    actual_ee,
    block_id,
    camera_config,
    camera_eye,
)
classification = classify_candidate(observation, settings)
```

Initialize `observation = None` before `try`, and return a flat record:

```python
return {
    "seed": seed,
    "direction": direction,
    **(observation or {}),
    **classification,
}
```

In the `except` branch call:

```python
classification = classify_candidate(
    observation,
    settings,
    error=repr(exc),
)
```

4. Add the frozen selection rule to the summary constructed by
`run_screening()`:

```python
"selection_rule": {
    "seed_range": list(screening["seed_range"]),
    "directions": list(screening["directions"]),
    "max_pose_error": screening["max_pose_error"],
    "clear_visibility_threshold": config["grounding_smoke"][
        "clear_visibility_threshold"
    ],
    "observation_scope": "initial_only",
    "distinct_seeds": True,
    "selection": "smallest_seed_in_direction_order",
},
```

- [ ] **Step 8: Run all screening tests and verify GREEN**

Run:

```bash
python -m unittest tests.vlm.grounding_smoke.test_screening -v
```

Expected: all screening tests PASS.

- [ ] **Step 9: Remove obsolete trajectory screening configuration under TDD**

Update `tests/test_config_contract.py` so the screening assertions are:

```python
self.assertEqual(screening["seed_range"], [55, 100])
self.assertEqual(
    screening["directions"], ["left", "right", "front"]
)
self.assertEqual(screening["max_pose_error"], 0.005)
self.assertNotIn("num_actions", screening)
self.assertNotIn("max_final_distance_xy", screening)
```

Run:

```bash
python -m unittest tests.test_config_contract.ConfigContractTests.test_grounding_smoke_config_is_safe_and_frozen -v
```

Expected: FAIL because `sim_config.yaml` still contains `num_actions` and
`max_final_distance_xy`.

Then remove those two keys from `grounding_smoke.screening` in
`sim_config.yaml`, remove the same obsolete keys from the
`screening_config()` test fixture, and revise the comments to describe
first-frame qualification.

Run:

```bash
python -m unittest \
  tests.test_config_contract.ConfigContractTests.test_grounding_smoke_config_is_safe_and_frozen \
  tests.vlm.grounding_smoke.test_screening -v
```

Expected: all selected tests PASS.

- [ ] **Step 10: Commit Task 1**

```bash
git add \
  src/vla_project/vlm/grounding_smoke/screening.py \
  tests/vlm/grounding_smoke/test_screening.py \
  tests/test_config_contract.py \
  sim_config.yaml
git commit -m "feat: screen smoke cases from initial observations"
```

---

### Task 2: Extract Batch Preflight and Add `--preflight-only`

**Files:**
- Modify: `tests/vlm/grounding_smoke/test_runner.py`
- Modify: `src/vla_project/vlm/grounding_smoke/runner.py`

**Interfaces:**
- Consumes: `preflight_smoke_case(config: dict, smoke_config: dict, case: dict) -> dict`
- Produces: `run_smoke_preflight(config: dict, run_name: str | None = None) -> tuple[Path, dict]`
- Changes: `main(argv: list[str] | None = None) -> None`
- Preserves: `run_smoke_batch(config: dict, run_name: str | None = None) -> tuple[Path, dict]`

- [ ] **Step 1: Add a failing test for the reusable successful preflight**

Add `run_smoke_preflight` to the imports from `runner`.

Before writing the test, name the break it catches: returning after the first
case, failing to persist all three records, or creating online episode evidence
must fail.

Add to `SmokeBatchContractTests`:

```python
def test_run_smoke_preflight_writes_three_qualified_cases_only(self):
    with tempfile.TemporaryDirectory() as temp_dir:
        config, _ = batch_config(temp_dir)
        rows = [
            {
                **case,
                "qualified": True,
                "rejection_reason": None,
            }
            for case in build_smoke_cases(config["grounding_smoke"])
        ]
        with patch(
            "vla_project.vlm.grounding_smoke.runner.preflight_smoke_case",
            side_effect=rows,
        ) as preflight:
            run_dir, summary = run_smoke_preflight(
                config,
                run_name="run_preflight_only",
            )

        payload = json.loads(
            (run_dir / "smoke_preflight.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(preflight.call_count, 3)
        self.assertEqual(summary, payload)
        self.assertEqual(summary["num_cases"], 3)
        self.assertEqual(summary["qualified_count"], 3)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["cases"], rows)
        self.assertFalse(
            (run_dir / "episode_summary.jsonl").exists()
        )
        self.assertFalse((run_dir / "smoke_summary.json").exists())
```

- [ ] **Step 2: Run the preflight test and verify RED**

Run:

```bash
python -m unittest \
  tests.vlm.grounding_smoke.test_runner.SmokeBatchContractTests.test_run_smoke_preflight_writes_three_qualified_cases_only -v
```

Expected: ERROR because `run_smoke_preflight` does not exist.

- [ ] **Step 3: Extract the reusable batch preflight**

Add before `run_smoke_batch()` in `runner.py`:

```python
def run_smoke_preflight(config, run_name=None):
    """只运行三个固定案例的无 API 起点预检。"""
    smoke_config = config["grounding_smoke"]
    cases = build_smoke_cases(smoke_config)
    batch_dir = make_run_dir(smoke_config["output_dir"], run_name)
    preflight_rows = [
        preflight_smoke_case(config, smoke_config, case)
        for case in cases
    ]
    summary = {
        "num_cases": len(preflight_rows),
        "qualified_count": sum(
            row["qualified"] for row in preflight_rows
        ),
        "passed": all(row["qualified"] for row in preflight_rows),
        "cases": preflight_rows,
    }
    _write_json(batch_dir / "smoke_preflight.json", summary)
    if not summary["passed"]:
        failures = ", ".join(
            f"episode={row['episode_idx']} seed={row['seed']} "
            f"direction={row['start_direction']} "
            f"reason={row['rejection_reason']}"
            for row in preflight_rows
            if not row["qualified"]
        )
        raise SmokePreflightError(
            f"smoke 初始动态预检失败: {failures}"
        )
    return batch_dir, summary
```

Change the beginning of `run_smoke_batch()` to:

```python
def run_smoke_batch(config, run_name=None):
    """运行固定三个真实 PyBullet/Qwen case，并保存批次证据。"""
    smoke_config = config["grounding_smoke"]
    cases = build_smoke_cases(smoke_config)
    batch_dir, _ = run_smoke_preflight(config, run_name=run_name)
    calibration = load_frozen_calibration(
        smoke_config["calibration_path"],
        smoke_config["expected_calibration_sample_ids"],
        smoke_config["expected_correction_x"],
        smoke_config["expected_correction_y"],
    )
```

Delete the duplicated inline preflight block from `run_smoke_batch()`.

- [ ] **Step 4: Run batch contract tests and verify GREEN**

Run:

```bash
python -m unittest \
  tests.vlm.grounding_smoke.test_runner.SmokeBatchContractTests.test_run_smoke_preflight_writes_three_qualified_cases_only \
  tests.vlm.grounding_smoke.test_runner.SmokeBatchContractTests.test_batch_preflights_all_cases_before_rejecting_without_api \
  tests.vlm.grounding_smoke.test_runner.SmokeBatchContractTests.test_batch_uses_three_isolated_cases_and_writes_summary -v
```

Expected: all 3 tests PASS. Adjust existing patches to target
`run_smoke_preflight` only if the direct `preflight_smoke_case` patches no
longer exercise the real extracted orchestration.

- [ ] **Step 5: Add a failing CLI test for `--preflight-only`**

Before writing the test, name the break it catches: ignoring
`--preflight-only` and entering `run_smoke_batch()` must fail immediately.

Import `main` from `runner`, then add:

```python
def test_cli_preflight_only_never_enters_online_batch(self):
    with tempfile.TemporaryDirectory() as temp_dir:
        config, _ = batch_config(temp_dir)
        rows = [
            {
                **case,
                "qualified": True,
                "rejection_reason": None,
            }
            for case in build_smoke_cases(config["grounding_smoke"])
        ]
        with (
            patch(
                "vla_project.vlm.grounding_smoke.runner.load_config",
                return_value=config,
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.preflight_smoke_case",
                side_effect=rows,
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.run_smoke_batch",
                side_effect=AssertionError("online batch entered"),
            ),
            patch(
                "vla_project.vlm.grounding_smoke.runner.call_openai_compatible_api",
                side_effect=AssertionError("API entered"),
            ),
        ):
            main([
                "--preflight-only",
                "--run-name",
                "run_cli_preflight",
            ])

        run_dir = Path(temp_dir) / "run_cli_preflight"
        payload = json.loads(
            (run_dir / "smoke_preflight.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["qualified_count"], 3)
        self.assertFalse(
            (run_dir / "episode_summary.jsonl").exists()
        )
        self.assertFalse((run_dir / "smoke_summary.json").exists())
```

- [ ] **Step 6: Run the CLI test and verify RED**

Run:

```bash
python -m unittest \
  tests.vlm.grounding_smoke.test_runner.SmokeBatchContractTests.test_cli_preflight_only_never_enters_online_batch -v
```

Expected: ERROR because `main()` does not accept `argv` and the parser does not
define `--preflight-only`.

- [ ] **Step 7: Implement `--preflight-only`**

Change:

```python
def main(argv=None):
```

Add the argument:

```python
parser.add_argument(
    "--preflight-only",
    action="store_true",
    help="只运行三个固定案例的无 API 起点预检",
)
args = parser.parse_args(argv)
```

Before the normal `run_smoke_batch()` branch, add:

```python
if args.preflight_only:
    batch_dir, summary = run_smoke_preflight(
        config,
        run_name=args.run_name,
    )
    print(f"run_dir={batch_dir}")
    print(
        f"qualified={summary['qualified_count']}/"
        f"{summary['num_cases']} passed={summary['passed']}"
    )
    return
```

Do not load calibration, construct episode dependencies, or inspect API
configuration in this branch.

- [ ] **Step 8: Run runner and smoke tests and verify GREEN**

Run:

```bash
python -m unittest \
  tests.vlm.grounding_smoke.test_runner.SmokeBatchContractTests.test_cli_preflight_only_never_enters_online_batch \
  tests.vlm.grounding_smoke.test_runner -v
```

Expected: all runner tests PASS.

- [ ] **Step 9: Commit Task 2**

```bash
git add \
  src/vla_project/vlm/grounding_smoke/runner.py \
  tests/vlm/grounding_smoke/test_runner.py
git commit -m "feat: add grounding smoke preflight-only mode"
```

---

### Task 3: Run the Real No-API Selection and Freeze the Selected Cases

**Files:**
- Generated: `outputs/vlm_evaluations/grounding_world_smoke_screening/run_20260726_initial_qualification_v1/`
- Modify: `sim_config.yaml`
- Modify: `tests/test_config_contract.py`
- Generated: `outputs/vlm_evaluations/grounding_world_smoke/run_20260726_fixed_cases_preflight_v1/`

**Interfaces:**
- Consumes: `vla-screen-grounding-smoke --run-name <name>`
- Consumes: `screening_summary.json["selected_cases"]`
- Produces: `grounding_smoke.seeds` ordered as `[left_seed, right_seed, front_seed]`
- Consumes: `vla-run-grounding-smoke --preflight-only --run-name <name>`
- Produces: `smoke_preflight.json` with `qualified_count == 3` and `passed == true`

- [ ] **Step 1: Run the real 138-candidate no-API screening**

Ensure no real smoke command is running. Then run:

```bash
vla-screen-grounding-smoke \
  --run-name run_20260726_initial_qualification_v1
```

Expected:

- exit code 0;
- 138 candidate records;
- three selected cases in `left`, `right`, `front` order;
- no API request or API environment dependency;
- only `step_00.jpg` exists within each successfully captured candidate directory.

If the run name already exists, choose the next deterministic suffix
`run_20260726_initial_qualification_v2`; never delete or overwrite the existing
directory.

- [ ] **Step 2: Validate the selection evidence before changing configuration**

Set `SELECTION_SUMMARY` to the actual newly created summary path and run:

```bash
python -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); s=json.loads(p.read_text()); c=s["selected_cases"]; assert s["passed"] is True; assert s["num_candidates"] == 138; assert len(c) == 3; assert [x["direction"] for x in c] == ["left", "right", "front"]; assert len({x["seed"] for x in c}) == 3; assert all(55 <= x["seed"] <= 100 for x in c); print([x["seed"] for x in c])' \
  "$SELECTION_SUMMARY"
```

Expected: prints exactly the ordered `[left_seed, right_seed, front_seed]`
selected by the saved evidence.

- [ ] **Step 3: Add a failing configuration contract for the evidence-selected cases**

Replace the old fixed seed assertion in
`tests/test_config_contract.py` with the exact three integer seeds printed in
Step 2:

```python
self.assertEqual(smoke["seeds"], [left_seed, right_seed, front_seed])
```

Here `left_seed`, `right_seed`, and `front_seed` mean the literal integers from
the actual saved `selected_cases`; write those integer literals into the test,
not variable names and not hand-selected alternatives.

Run:

```bash
python -m unittest \
  tests.test_config_contract.ConfigContractTests.test_grounding_smoke_config_is_safe_and_frozen -v
```

Expected: FAIL because `sim_config.yaml` still contains `[52, 53, 54]`.

- [ ] **Step 4: Freeze the evidence-selected cases in configuration**

Use `apply_patch` to replace only `grounding_smoke.seeds` in `sim_config.yaml`
with the exact ordered integer list from Step 2. Update the adjacent comment to
name the no-API screening run directory used as evidence. Do not change
`start_directions`.

Run:

```bash
python -m unittest \
  tests.test_config_contract.ConfigContractTests.test_grounding_smoke_config_is_safe_and_frozen \
  tests.vlm.grounding_smoke.test_runner.SmokeBatchContractTests.test_builds_exact_three_cases_without_back -v
```

If `test_builds_exact_three_cases_without_back` still hard-codes `[52, 53, 54]`,
change its fixture and expected cases to the same three literal selected seeds.

Expected: selected tests PASS and `build_smoke_cases()` preserves the
`left/right/front` mapping.

- [ ] **Step 5: Run the fixed three-case preflight-only command**

Run:

```bash
vla-run-grounding-smoke \
  --preflight-only \
  --run-name run_20260726_fixed_cases_preflight_v1
```

Expected:

- exit code 0;
- CLI reports `qualified=3/3 passed=True`;
- the run directory contains `smoke_preflight.json`;
- no episode directory, `episode_summary.jsonl`, or `smoke_summary.json`;
- no VLM API call.

If the run name already exists, use the next deterministic suffix
`run_20260726_fixed_cases_preflight_v2`; never delete or overwrite evidence.

- [ ] **Step 6: Validate the preflight evidence**

Set `PREFLIGHT_SUMMARY` to the actual newly created `smoke_preflight.json` and
run:

```bash
python -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); s=json.loads(p.read_text()); assert s["num_cases"] == 3; assert s["qualified_count"] == 3; assert s["passed"] is True; assert len(s["cases"]) == 3; assert all(x["qualified"] and x["rejection_reason"] is None for x in s["cases"]); print([(x["seed"], x["start_direction"], x["start_pose_error"], x["block_visibility_ratio"]) for x in s["cases"]])' \
  "$PREFLIGHT_SUMMARY"
```

Expected: prints the three selected cases and their passing pose/visibility
metrics.

- [ ] **Step 7: Commit Task 3**

Generated outputs remain ignored. Commit only the frozen configuration and its
contracts:

```bash
git add sim_config.yaml tests/test_config_contract.py \
  tests/vlm/grounding_smoke/test_runner.py
git commit -m "config: freeze qualified grounding smoke cases"
```

---

### Task 4: Record Evidence and Verify the Repository

**Files:**
- Modify: `docs/agent/PROJECT_OVERVIEW.md`
- Modify: `docs/agent/CURRENT_STATUS.md`
- Modify: `docs/worklog/WORKLOG.md`
- Inspect: `README.md`
- Inspect: `docs/debugging/BUGLOG.md`

**Interfaces:**
- Consumes: actual `screening_summary.json`
- Consumes: actual `smoke_preflight.json`
- Produces: current authoritative stage status and reproducible evidence links

- [ ] **Step 1: Update stable and dynamic project knowledge**

Use the literal metrics and paths from Task 3.

In `PROJECT_OVERVIEW.md`:

- replace the conclusion that zero dynamic candidates qualified as the current
  selection premise;
- retain that result as evidence that full-trajectory clear was an invalid
  prerequisite;
- add the selected fixed cases and the `3/3` initial preflight result;
- state that no second online smoke has been run.

In `CURRENT_STATUS.md`:

- update the last checked date to `2026-07-26`;
- mark fixed-case reselection and `3/3` no-API preflight complete;
- remove the obsolete unresolved items for `52-left` and `54-front`;
- set the next step to reviewing the no-API evidence and obtaining explicit
  user approval before any real smoke;
- retain autonomous stop and in-loop batch exception handling as unresolved.

In `WORKLOG.md`, add one dated entry containing:

- the frozen selection rule;
- candidate counts and reason counts;
- selected seed-direction pairs;
- screening output path;
- fixed preflight metrics for all three cases;
- preflight output path;
- explicit API call count of 0;
- the decision not to run an online smoke.

- [ ] **Step 2: Inspect README and BUGLOG for stale authoritative claims**

Run:

```bash
rg -n "52-left|53-right|54-front|55–100|55-100|0个合格|固定案例|下一步" \
  README.md docs/debugging/BUGLOG.md
```

Only update a file if it presents the obsolete fixed cases as the current next
step. Preserve historical bug evidence and clearly label it historical instead
of rewriting old results.

- [ ] **Step 3: Run focused verification**

Run:

```bash
python -m unittest \
  tests.vlm.grounding_smoke.test_screening \
  tests.vlm.grounding_smoke.test_runner \
  tests.vlm.grounding_smoke.test_targeting \
  tests.test_config_contract \
  tests.test_package_metadata -v
```

Expected: all focused tests PASS with zero failures and zero errors.

- [ ] **Step 4: Run full verification**

Run:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
git diff --check
git status --short
```

Expected:

- full suite exits 0 with zero failures and zero errors;
- `compileall` exits 0;
- `git diff --check` exits 0;
- only intended task files plus the preserved pre-existing untracked plan are
  present.

- [ ] **Step 5: Audit requirements against evidence**

Read the final `screening_summary.json` and `smoke_preflight.json` and check:

```text
[ ] 138 candidates were evaluated
[ ] selected directions are left/right/front in order
[ ] selected seeds are distinct and within 55–100
[ ] selection uses smallest eligible seed per direction after excluding used seeds
[ ] fixed preflight reports qualified_count=3 and passed=true
[ ] no episode or smoke summary exists in the preflight-only directory
[ ] no real online smoke was run
[ ] documentation records API calls=0
```

For the smallest-seed check, load `candidate_trace.jsonl`, independently filter
`qualified=true`, apply the direction-order distinct-seed rule, and compare the
result with `screening_summary.json["selected_cases"]`.

- [ ] **Step 6: Commit Task 4**

Stage only documentation actually changed:

```bash
git add \
  docs/agent/PROJECT_OVERVIEW.md \
  docs/agent/CURRENT_STATUS.md \
  docs/worklog/WORKLOG.md
git add README.md docs/debugging/BUGLOG.md
git diff --cached --stat
git commit -m "docs: record qualified grounding smoke cases"
```

If README or BUGLOG did not require changes, omit them from `git add`. Before
committing, confirm the pre-existing untracked
`docs/superpowers/plans/2026-07-19-agent-file-placement-rules.md` is not staged.

- [ ] **Step 7: Report the execution boundary**

Report:

- selected fixed cases;
- screening and preflight evidence paths;
- exact focused and full test counts;
- API calls: 0;
- real online smoke: not run;
- remaining requirement: explicit user approval before the paid/API-backed
  smoke command.
