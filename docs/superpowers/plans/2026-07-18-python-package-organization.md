# Python Package Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the repository's 11 root Python modules into an installable `src/vla_project/` package, mirror the domain structure under `tests/`, and replace direct script commands with verified console scripts.

**Architecture:** Keep every existing module body and public function intact while moving files into `simulation`, `vlm`, and `tools` subpackages. Use absolute `vla_project.*` imports, setuptools editable installation, and staged test runs so each dependency boundary is verified before the next group moves.

**Tech Stack:** Python 3.10, setuptools 82, `pyproject.toml`, `unittest`, conda environment `vla_env`, Git.

## Global Constraints

- Move and package code only; do not change control, sampling, VLM, grounding, backprojection, calibration, or migration algorithms.
- Preserve all existing function, class, constant, and generated-data schemas.
- Keep `tests/` separate from `src/` and mirror the three domain groups.
- Do not move, regenerate, delete, or rewrite `outputs/`.
- Do not call Qwen or another paid API.
- Keep `requirements.txt` as the runtime dependency source; add no new runtime dependency.
- Use `conda run -n vla_env` for Python and test commands.
- Install editable with `--no-deps --no-build-isolation` so no network access is required.
- Update live README, WORKLOG, BUGLOG, and Markdown study plan; preserve historical specs/plans and the original PDF.

---

## File Structure

- `src/vla_project/simulation/`: PyBullet camera, collection, closed-loop probe, and batch probe evaluation.
- `src/vla_project/vlm/`: sample collection, Qwen decision/grounding, backprojection, and calibration.
- `src/vla_project/tools/`: repository maintenance utilities that are not part of robot/VLM execution.
- `tests/simulation/`, `tests/vlm/`, `tests/tools/`: tests mirroring production domains.
- `tests/test_config_contract.py`: cross-domain configuration contract.
- `tests/test_package_metadata.py`: package layout and console-script contract.
- `pyproject.toml`: setuptools build metadata, src discovery, and ten console scripts.

### Task 1: Establish the src-layout package skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/vla_project/__init__.py`
- Create: `src/vla_project/simulation/__init__.py`
- Create: `src/vla_project/vlm/__init__.py`
- Create: `src/vla_project/tools/__init__.py`
- Create: `tests/test_package_metadata.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces importable empty packages `vla_project`, `vla_project.simulation`, `vla_project.vlm`, and `vla_project.tools` when `src` is on `PYTHONPATH`.
- Produces setuptools distribution name `vla-project`, version `0.1.0`, Python floor `>=3.10`.

- [ ] **Step 1: Write the failing source-layout test**

Create `tests/test_package_metadata.py`:

```python
"""保护标准 src 包结构和命令入口，防止目录整理后再次漂移。"""

import importlib
import unittest


class PackageLayoutTests(unittest.TestCase):
    def test_domain_packages_are_importable(self):
        for module_name in (
            "vla_project",
            "vla_project.simulation",
            "vla_project.vlm",
            "vla_project.tools",
        ):
            with self.subTest(module_name=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run RED before creating the package**

Run:

```bash
PYTHONPATH=src conda run -n vla_env python -m unittest tests.test_package_metadata -v
```

Expected: ERROR with `ModuleNotFoundError: No module named 'vla_project'`.

- [ ] **Step 3: Create package markers and build metadata**

Create each `__init__.py` with only a domain docstring. Create `pyproject.toml` exactly as:

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "vla-project"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]
```

Add this ignore rule beside other Python build artifacts:

```gitignore
*.egg-info/
```

- [ ] **Step 4: Run GREEN for package imports**

Run:

```bash
PYTHONPATH=src conda run -n vla_env python -m unittest tests.test_package_metadata -v
```

Expected: 1 test PASS.

- [ ] **Step 5: Commit the package skeleton**

```bash
git add pyproject.toml .gitignore src tests/test_package_metadata.py
git commit -m "build: add src package skeleton"
```

### Task 2: Move simulation modules and tests

**Files:**
- Move: `camera_geometry.py` -> `src/vla_project/simulation/camera_geometry.py`
- Move: `control_arm.py` -> `src/vla_project/simulation/control_arm.py`
- Move: `stage3_probe.py` -> `src/vla_project/simulation/stage3_probe.py`
- Move: `evaluate_probe.py` -> `src/vla_project/simulation/evaluate_probe.py`
- Move: `tests/test_camera_geometry.py` -> `tests/simulation/test_camera_geometry.py`
- Move: `tests/test_control_arm.py` -> `tests/simulation/test_control_arm.py`
- Move: `tests/test_stage3_probe.py` -> `tests/simulation/test_stage3_probe.py`
- Move: `tests/test_evaluate_probe.py` -> `tests/simulation/test_evaluate_probe.py`
- Create: `tests/__init__.py`
- Create: `tests/simulation/__init__.py`
- Modify: `tests/test_config_contract.py`
- Modify imports only: all six root VLM modules that consume simulation modules

**Interfaces:**
- Produces `vla_project.simulation.camera_geometry`, `.control_arm`, `.stage3_probe`, `.evaluate_probe` with unchanged public APIs.
- Root VLM modules temporarily import the new simulation package until Task 3 moves them.

- [ ] **Step 1: Move files without editing module bodies**

Run:

```bash
mkdir -p tests/simulation
git mv camera_geometry.py src/vla_project/simulation/camera_geometry.py
git mv control_arm.py src/vla_project/simulation/control_arm.py
git mv stage3_probe.py src/vla_project/simulation/stage3_probe.py
git mv evaluate_probe.py src/vla_project/simulation/evaluate_probe.py
git mv tests/test_camera_geometry.py tests/simulation/test_camera_geometry.py
git mv tests/test_control_arm.py tests/simulation/test_control_arm.py
git mv tests/test_stage3_probe.py tests/simulation/test_stage3_probe.py
git mv tests/test_evaluate_probe.py tests/simulation/test_evaluate_probe.py
```

Create `tests/__init__.py` and `tests/simulation/__init__.py` with test-package docstrings.

- [ ] **Step 2: Run imports to observe the expected RED**

Run:

```bash
PYTHONPATH=src conda run -n vla_env python -m unittest discover -s tests -v
```

Expected: ERROR because moved modules and tests still import `camera_geometry`, `control_arm`, `stage3_probe`, or `evaluate_probe` from the repository root.

- [ ] **Step 3: Rewrite simulation production imports**

Use these exact imports:

```python
# src/vla_project/simulation/control_arm.py
from vla_project.simulation.camera_geometry import compute_camera_matrices

# src/vla_project/simulation/stage3_probe.py
from vla_project.simulation.control_arm import (
    CONFIG_PATH,
    apply_joint_targets,
    calculate_target_joints,
    capture_rgb,
    connect_physics,
    euclidean_distance,
    get_link_position,
    get_object_position,
    load_block,
    load_config,
    sample_camera_eye,
    settle_object,
    setup_world,
)

# src/vla_project/simulation/evaluate_probe.py
from vla_project.simulation.control_arm import CONFIG_PATH, load_config
from vla_project.simulation.stage3_probe import run_probe_episode
```

Only change module paths; keep each existing imported-name list unchanged.

- [ ] **Step 4: Rewrite simulation test imports and patch targets**

Use `vla_project.simulation.*` imports. In `tests/simulation/test_control_arm.py`, bind:

```python
from vla_project.simulation import control_arm
from vla_project.simulation.control_arm import (
    calculate_target_joints,
    capture_rgb,
    determine_termination,
    next_episode_index,
)
```

Replace patch prefixes:

```text
control_arm. -> vla_project.simulation.control_arm.
stage3_probe. -> vla_project.simulation.stage3_probe.
```

Update `tests/test_config_contract.py` to:

```python
from vla_project.simulation.control_arm import load_config
```

- [ ] **Step 5: Keep still-root VLM modules working during the staged move**

Change only their simulation imports:

```text
camera_geometry -> vla_project.simulation.camera_geometry
control_arm -> vla_project.simulation.control_arm
stage3_probe -> vla_project.simulation.stage3_probe
```

Apply this to `collect_vlm_eval_samples.py`, `evaluate_vlm_decisions.py`, `diagnose_vlm_grounding.py`, `evaluate_ground_then_decide.py`, and `evaluate_grounding_backprojection.py`. `validate_grounding_calibration.py` has no simulation import.

- [ ] **Step 6: Run simulation tests and the complete staged suite**

Run:

```bash
PYTHONPATH=src conda run -n vla_env python -m unittest \
  tests.simulation.test_camera_geometry \
  tests.simulation.test_control_arm \
  tests.simulation.test_stage3_probe \
  tests.simulation.test_evaluate_probe \
  tests.test_config_contract -v
PYTHONPATH=src conda run -n vla_env python -m unittest discover -s tests -v
```

Expected: focused tests PASS; complete suite reports at least 107 tests PASS.

- [ ] **Step 7: Commit simulation organization**

```bash
git add src/vla_project/simulation tests/simulation tests/__init__.py \
  tests/test_config_contract.py collect_vlm_eval_samples.py \
  evaluate_vlm_decisions.py diagnose_vlm_grounding.py \
  evaluate_ground_then_decide.py evaluate_grounding_backprojection.py
git commit -m "refactor: package simulation modules"
```

### Task 3: Move VLM modules and tests

**Files:**
- Move six root VLM modules to `src/vla_project/vlm/` using the design mapping.
- Move six matching tests to `tests/vlm/`.
- Create: `tests/vlm/__init__.py`

**Interfaces:**
- Produces six `vla_project.vlm.*` modules with unchanged public APIs.
- Consumes packaged simulation imports from Task 2.

- [ ] **Step 1: Move the VLM files**

Run:

```bash
mkdir -p tests/vlm
git mv collect_vlm_eval_samples.py src/vla_project/vlm/collect_vlm_eval_samples.py
git mv evaluate_vlm_decisions.py src/vla_project/vlm/evaluate_vlm_decisions.py
git mv diagnose_vlm_grounding.py src/vla_project/vlm/diagnose_vlm_grounding.py
git mv evaluate_ground_then_decide.py src/vla_project/vlm/evaluate_ground_then_decide.py
git mv evaluate_grounding_backprojection.py src/vla_project/vlm/evaluate_grounding_backprojection.py
git mv validate_grounding_calibration.py src/vla_project/vlm/validate_grounding_calibration.py
git mv tests/test_collect_vlm_eval_samples.py tests/vlm/test_collect_vlm_eval_samples.py
git mv tests/test_evaluate_vlm_decisions.py tests/vlm/test_evaluate_vlm_decisions.py
git mv tests/test_diagnose_vlm_grounding.py tests/vlm/test_diagnose_vlm_grounding.py
git mv tests/test_evaluate_ground_then_decide.py tests/vlm/test_evaluate_ground_then_decide.py
git mv tests/test_evaluate_grounding_backprojection.py tests/vlm/test_evaluate_grounding_backprojection.py
git mv tests/test_validate_grounding_calibration.py tests/vlm/test_validate_grounding_calibration.py
```

Create `tests/vlm/__init__.py` with a test-package docstring.

- [ ] **Step 2: Run RED for stale root-module imports**

Run:

```bash
PYTHONPATH=src conda run -n vla_env python -m unittest discover -s tests -v
```

Expected: ERROR because VLM modules/tests still import `collect_vlm_eval_samples`, `diagnose_vlm_grounding`, or other old root modules.

- [ ] **Step 3: Rewrite VLM production imports**

Replace root imports with these package prefixes while preserving imported names:

```text
collect_vlm_eval_samples -> vla_project.vlm.collect_vlm_eval_samples
diagnose_vlm_grounding -> vla_project.vlm.diagnose_vlm_grounding
```

Simulation imports already use `vla_project.simulation.*` from Task 2. `validate_grounding_calibration.py` remains internally unchanged.

- [ ] **Step 4: Rewrite VLM test imports and patch targets**

Use `vla_project.vlm.*` for all six test imports. Replace patch prefixes:

```text
collect_vlm_eval_samples. -> vla_project.vlm.collect_vlm_eval_samples.
diagnose_vlm_grounding. -> vla_project.vlm.diagnose_vlm_grounding.
evaluate_vlm_decisions. -> vla_project.vlm.evaluate_vlm_decisions.
```

Tests importing camera helpers use `vla_project.simulation.camera_geometry`.

- [ ] **Step 5: Run VLM and complete tests**

Run:

```bash
PYTHONPATH=src conda run -n vla_env python -m unittest discover -s tests/vlm -t . -v
PYTHONPATH=src conda run -n vla_env python -m unittest discover -s tests -v
```

Expected: VLM tests PASS; complete suite reports at least 107 tests PASS without Qwen calls.

- [ ] **Step 6: Commit VLM organization**

```bash
git add src/vla_project/vlm tests/vlm
git commit -m "refactor: package VLM modules"
```

### Task 4: Move the maintenance tool and test

**Files:**
- Move: `migrate_generated_outputs.py` -> `src/vla_project/tools/migrate_generated_outputs.py`
- Move: `tests/test_migrate_generated_outputs.py` -> `tests/tools/test_migrate_generated_outputs.py`
- Create: `tests/tools/__init__.py`

**Interfaces:**
- Produces `vla_project.tools.migrate_generated_outputs` with the same constants, errors, migration functions, validation function, and `main()`.

- [ ] **Step 1: Move tool and test**

Run:

```bash
mkdir -p tests/tools
git mv migrate_generated_outputs.py src/vla_project/tools/migrate_generated_outputs.py
git mv tests/test_migrate_generated_outputs.py tests/tools/test_migrate_generated_outputs.py
```

Create `tests/tools/__init__.py` with a test-package docstring.

- [ ] **Step 2: Run RED for the stale tool import**

Run:

```bash
PYTHONPATH=src conda run -n vla_env python -m unittest tests.tools.test_migrate_generated_outputs -v
```

Expected: ERROR with `ModuleNotFoundError: No module named 'migrate_generated_outputs'`.

- [ ] **Step 3: Update the test import**

Use:

```python
from vla_project.tools.migrate_generated_outputs import (
    MIGRATIONS,
    MigrationConflictError,
    MigrationError,
    migrate_generated_outputs,
    preflight_migration,
    rewrite_path,
    validate_migrated_outputs,
)
```

- [ ] **Step 4: Run tool and complete tests**

Run:

```bash
PYTHONPATH=src conda run -n vla_env python -m unittest tests.tools.test_migrate_generated_outputs -v
PYTHONPATH=src conda run -n vla_env python -m unittest discover -s tests -v
```

Expected: 7 tool tests PASS; complete suite reports at least 107 tests PASS.

- [ ] **Step 5: Commit tool organization**

```bash
git add src/vla_project/tools tests/tools
git commit -m "refactor: package maintenance tools"
```

### Task 5: Register and verify console scripts

**Files:**
- Modify: `tests/test_package_metadata.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces ten console-script names mapped exactly to the design's ten `main()` functions.
- Produces editable distribution metadata discoverable as `vla-project`.

- [ ] **Step 1: Add the failing entry-point contract**

Extend `tests/test_package_metadata.py`:

```python
import importlib.metadata

EXPECTED_SCRIPTS = {
    "vla-collect": "vla_project.simulation.control_arm:main",
    "vla-probe": "vla_project.simulation.stage3_probe:main",
    "vla-evaluate-probe": "vla_project.simulation.evaluate_probe:main",
    "vla-collect-vlm-samples": "vla_project.vlm.collect_vlm_eval_samples:main",
    "vla-evaluate-vlm-decisions": "vla_project.vlm.evaluate_vlm_decisions:main",
    "vla-diagnose-grounding": "vla_project.vlm.diagnose_vlm_grounding:main",
    "vla-ground-then-decide": "vla_project.vlm.evaluate_ground_then_decide:main",
    "vla-evaluate-backprojection": "vla_project.vlm.evaluate_grounding_backprojection:main",
    "vla-validate-grounding-calibration": "vla_project.vlm.validate_grounding_calibration:main",
    "vla-migrate-generated-outputs": "vla_project.tools.migrate_generated_outputs:main",
}


class ConsoleScriptTests(unittest.TestCase):
    def test_distribution_registers_expected_console_scripts(self):
        distribution = importlib.metadata.distribution("vla-project")
        actual = {
            entry.name: entry.value
            for entry in distribution.entry_points
            if entry.group == "console_scripts" and entry.name.startswith("vla-")
        }
        self.assertEqual(actual, EXPECTED_SCRIPTS)

    def test_every_console_script_points_to_callable(self):
        for command, target in EXPECTED_SCRIPTS.items():
            module_name, function_name = target.split(":", maxsplit=1)
            with self.subTest(command=command):
                module = importlib.import_module(module_name)
                self.assertTrue(callable(getattr(module, function_name)))
```

- [ ] **Step 2: Install current metadata and verify RED**

Run:

```bash
conda run -n vla_env pip install -e . --no-deps --no-build-isolation
conda run -n vla_env python -m unittest tests.test_package_metadata.ConsoleScriptTests -v
```

Expected: FAIL because the installed distribution has no `vla-*` entry points.

- [ ] **Step 3: Add exact console scripts to pyproject**

Append:

```toml
[project.scripts]
vla-collect = "vla_project.simulation.control_arm:main"
vla-probe = "vla_project.simulation.stage3_probe:main"
vla-evaluate-probe = "vla_project.simulation.evaluate_probe:main"
vla-collect-vlm-samples = "vla_project.vlm.collect_vlm_eval_samples:main"
vla-evaluate-vlm-decisions = "vla_project.vlm.evaluate_vlm_decisions:main"
vla-diagnose-grounding = "vla_project.vlm.diagnose_vlm_grounding:main"
vla-ground-then-decide = "vla_project.vlm.evaluate_ground_then_decide:main"
vla-evaluate-backprojection = "vla_project.vlm.evaluate_grounding_backprojection:main"
vla-validate-grounding-calibration = "vla_project.vlm.validate_grounding_calibration:main"
vla-migrate-generated-outputs = "vla_project.tools.migrate_generated_outputs:main"
```

- [ ] **Step 4: Reinstall and run GREEN**

Run:

```bash
conda run -n vla_env pip install -e . --no-deps --no-build-isolation
conda run -n vla_env python -m unittest tests.test_package_metadata -v
```

Expected: all 3 package metadata tests PASS; all ten target modules import and expose callable `main` functions.

- [ ] **Step 5: Verify command registration without executing commands**

Run:

```bash
conda run -n vla_env python -c 'import importlib.metadata as m; d=m.distribution("vla-project"); rows=sorted((e.name,e.value) for e in d.entry_points if e.group=="console_scripts" and e.name.startswith("vla-")); assert len(rows)==10; print(*rows, sep="\n")'
```

Expected: ten `(command, module:main)` pairs. Do not invoke the commands because some start PyBullet or paid API flows.

- [ ] **Step 6: Commit package commands**

```bash
git add pyproject.toml tests/test_package_metadata.py
git commit -m "build: register VLA console scripts"
```

### Task 6: Synchronize live documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/worklog/WORKLOG.md`
- Modify: `docs/debugging/BUGLOG.md`
- Modify: `docs/planning/vla_robotic_study_plan.md`
- Inspect only: historical `docs/superpowers/specs/**`, `docs/superpowers/plans/**`, and PDF

**Interfaces:**
- Consumes final source paths and console commands from Tasks 2-5.
- Produces current documentation that no longer instructs users to run root Python files.

- [ ] **Step 1: Inventory stale live references**

Run:

```bash
rg -n '(^|[^/])(camera_geometry|collect_vlm_eval_samples|control_arm|diagnose_vlm_grounding|evaluate_ground_then_decide|evaluate_grounding_backprojection|evaluate_probe|evaluate_vlm_decisions|migrate_generated_outputs|stage3_probe|validate_grounding_calibration)\.py|python (control_arm|stage3_probe|evaluate_probe)\.py' \
  README.md docs/worklog/WORKLOG.md docs/debugging/BUGLOG.md \
  docs/planning/vla_robotic_study_plan.md
```

Expected: reports every current root-source path or old direct Python command requiring an update.

- [ ] **Step 2: Update README**

Replace the root Python listing with the approved `src/vla_project/{simulation,vlm,tools}` and mirrored `tests/` tree. Add editable installation:

```bash
conda run -n vla_env pip install -e . --no-deps --no-build-isolation
```

Replace the four common commands:

```text
python control_arm.py                         -> vla-collect
python stage3_probe.py                        -> vla-probe
conda run -n vla_env python evaluate_probe.py -> conda run -n vla_env vla-evaluate-probe
python collect_vlm_eval_samples.py            -> vla-collect-vlm-samples
```

- [ ] **Step 3: Update WORKLOG, BUGLOG, and study-plan source paths**

Map current source references to the exact new module paths. Preserve metrics, dates, configuration evidence, and conclusions. Do not change historical superpowers plans/specs or the PDF.

- [ ] **Step 4: Verify live docs and command names**

Run the Step 1 `rg` command again.

Expected: no stale current root path or old direct command remains; source filename mentions used purely as prose include their `src/vla_project/...` context.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md docs/worklog/WORKLOG.md docs/debugging/BUGLOG.md \
  docs/planning/vla_robotic_study_plan.md
git commit -m "docs: document packaged project layout"
```

### Task 7: Final regression, data invariants, and clean-root verification

**Files:**
- Verify all source, tests, package metadata, docs, and ignored outputs.

**Interfaces:**
- Consumes Tasks 1-6.
- Produces final evidence for the package reorganization.

- [ ] **Step 1: Run the complete installed-package test suite**

Run:

```bash
conda run -n vla_env python -m unittest discover -s tests -v
```

Expected: at least 109 tests PASS: original 106 plus three package metadata tests.

- [ ] **Step 2: Verify imports without PYTHONPATH overrides**

Run:

```bash
conda run -n vla_env python -c 'from vla_project.simulation.control_arm import load_config; from vla_project.simulation.stage3_probe import run_probe_episode; from vla_project.vlm.collect_vlm_eval_samples import read_jsonl; from vla_project.tools.migrate_generated_outputs import validate_migrated_outputs; print("package imports ok")'
```

Expected: prints `package imports ok`.

- [ ] **Step 3: Verify root cleanup and stale imports**

Run:

```bash
find . -maxdepth 1 -type f -name '*.py' -print
rg -n '^(from|import) (camera_geometry|collect_vlm_eval_samples|control_arm|diagnose_vlm_grounding|evaluate_ground_then_decide|evaluate_grounding_backprojection|evaluate_probe|evaluate_vlm_decisions|migrate_generated_outputs|stage3_probe|validate_grounding_calibration)' \
  src tests --glob '*.py'
```

Expected: `find` prints nothing and `rg` finds no old root-module imports.

- [ ] **Step 4: Revalidate outputs without changing them**

Run:

```bash
conda run -n vla_env python -c 'from pathlib import Path; from vla_project.tools.migrate_generated_outputs import validate_migrated_outputs; print(validate_migrated_outputs(Path.cwd()))'
```

Expected current snapshot: `files=8810`, `images=8590`, `image_bytes=74887973`, `missing_image_references=3559`.

- [ ] **Step 5: Check Git cleanliness and ignored metadata**

Run:

```bash
git check-ignore -v src/vla_project.egg-info/PKG-INFO
git check-ignore -v outputs/dataset/episode_summary.jsonl
git diff --check
git status --short
```

Expected: egg-info and outputs match `.gitignore`; diff check succeeds; status contains no uncommitted changes.

- [ ] **Step 6: Record the handoff**

Report the final test count, ten console scripts, the empty root-Python-file scan, output invariants, commit list, and explicit confirmation that no Qwen API call occurred.
