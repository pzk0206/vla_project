# Generated Output Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all main-workspace generated images and their experiment metadata under `outputs/`, and make every future generator write directly to the new categorized paths.

**Architecture:** Keep the existing config-driven writers and change their configured roots instead of adding a second path abstraction. Add one focused, tested migration utility that performs collision checks, moves the eight ignored output trees, rewrites known path prefixes inside JSON/JSONL, validates counts and image references, and rolls back moves on failure.

**Tech Stack:** Python 3, `pathlib`, `json`, `shutil`, YAML configuration, `unittest`, Git.

## Global Constraints

- Process only `/home/pzk/vla_project`; never process `.worktrees/`.
- Move images together with JSON, JSONL, annotations, traces, and summaries.
- Do not delete, compress, deduplicate, merge, or overwrite experiment data.
- Do not call Qwen or any paid API.
- Do not change control, sampling, evaluation, or backprojection behavior.
- Run Python verification through `conda run -n vla_env`.
- Treat any destination collision or unreadable JSON/JSONL as a hard failure before moving data. Preserve the exact pre-existing dangling-image-reference set; migration must not add dangling references.

---

## File Structure

- Create `migrate_generated_outputs.py`: one-time, reusable migration and integrity-check CLI; it owns old-to-new mappings and record rewriting.
- Create `tests/test_migrate_generated_outputs.py`: isolated temporary-directory tests for preflight, migration, path rewriting, and collision safety.
- Modify `tests/test_config_contract.py`: exact output-path contract for all five categories.
- Modify `sim_config.yaml`: future output roots.
- Modify `.gitignore`: ignore the single `outputs/` tree.
- Modify `control_arm.py`, `stage3_probe.py`, `tests/test_control_arm.py`: comments and user-facing text that name old roots.
- Modify `README.md`, `docs/worklog/WORKLOG.md`, `docs/debugging/BUGLOG.md`: current path references and the old-to-new directory note.
- Generate, Git-ignored: `outputs/**` by moving the eight existing output directories.

### Task 1: Lock the new configuration contract

**Files:**
- Modify: `tests/test_config_contract.py:39-83`
- Modify: `sim_config.yaml:45-48,86-89,108-119`
- Modify: `.gitignore:13-21`

**Interfaces:**
- Consumes: existing `load_config("sim_config.yaml") -> dict`.
- Produces: exact configured roots used unchanged by existing generator entry points.

- [ ] **Step 1: Write failing assertions for every new root**

Add this test to `ConfigContractTests`:

```python
def test_generated_outputs_are_grouped_under_outputs(self):
    """所有生成图片和实验记录必须进入统一 outputs 根目录。"""
    self.assertEqual(self.config["dataset"]["output_dir"], "outputs/dataset")
    self.assertEqual(self.config["probe"]["output_dir"], "outputs/probe")
    self.assertEqual(
        self.config["probe_evaluation"]["output_dir"],
        "outputs/probe_evaluations",
    )
    self.assertEqual(
        self.config["vlm_evaluation"]["sample_output_dir"],
        "outputs/vlm_samples/448_calibration_validation_d020",
    )
    self.assertEqual(
        self.config["vlm_evaluation"]["run_output_dir"],
        "outputs/vlm_evaluations",
    )
```

Also update the pre-existing exact `sample_output_dir` assertion to the same new value so the two tests do not contradict each other.

- [ ] **Step 2: Run the contract test and verify the intended failure**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_config_contract.ConfigContractTests.test_generated_outputs_are_grouped_under_outputs -v
```

Expected: FAIL showing `dataset` differs from `outputs/dataset`.

- [ ] **Step 3: Change the five configured output roots**

Use these exact YAML values:

```yaml
dataset:
  output_dir: "outputs/dataset"

probe:
  output_dir: "outputs/probe"

probe_evaluation:
  output_dir: "outputs/probe_evaluations"

vlm_evaluation:
  sample_output_dir: "outputs/vlm_samples/448_calibration_validation_d020"
  run_output_dir: "outputs/vlm_evaluations"
```

Keep every unrelated config value unchanged. Update the adjacent Chinese comments to describe the corresponding `outputs/` category.

- [ ] **Step 4: Replace individual generated-output ignores with one root rule**

Keep the section heading and replace the old eight rules with:

```gitignore
# Generated simulation outputs
outputs/
```

Do not change `.worktrees/` or any model-artifact rule.

- [ ] **Step 5: Run the contract and ignore checks**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_config_contract -v
git check-ignore -v outputs/dataset/example.jpg
git check-ignore -v outputs/vlm_evaluations/example/annotated/example.jpg
```

Expected: all config tests PASS; both example paths match `.gitignore`'s `outputs/` rule.

- [ ] **Step 6: Commit the configuration contract**

```bash
git add sim_config.yaml .gitignore tests/test_config_contract.py
git commit -m "refactor: group generated output paths"
```

### Task 2: Build a collision-safe migration utility

**Files:**
- Create: `migrate_generated_outputs.py`
- Create: `tests/test_migrate_generated_outputs.py`

**Interfaces:**
- Produces: `MIGRATIONS: tuple[tuple[str, str], ...]` ordered from the longest old VLM sample prefix to the shortest.
- Produces: `rewrite_path(value: str, mappings=MIGRATIONS) -> str`.
- Produces: `preflight_migration(project_root: Path) -> dict[str, int]`.
- Produces: `migrate_generated_outputs(project_root: Path) -> dict[str, int]`.
- Produces: `validate_migrated_outputs(project_root: Path) -> dict[str, int]`.
- Produces CLI: `python migrate_generated_outputs.py [--project-root PATH] [--apply]`; without `--apply`, it performs preflight and prints inventory only.

- [ ] **Step 1: Write failing tests for mapping, record rewriting, and movement**

Create `tests/test_migrate_generated_outputs.py` with temporary synthetic trees:

```python
import json
import tempfile
import unittest
from pathlib import Path

from migrate_generated_outputs import (
    MIGRATIONS,
    MigrationConflictError,
    migrate_generated_outputs,
    rewrite_path,
)


class GeneratedOutputMigrationTests(unittest.TestCase):
    def test_rewrite_path_uses_exact_prefix_boundaries(self):
        self.assertEqual(
            rewrite_path("vlm_eval_samples_448/images/frame.jpg"),
            "outputs/vlm_samples/448/images/frame.jpg",
        )
        self.assertEqual(rewrite_path("dataset/frame.jpg"), "outputs/dataset/frame.jpg")
        self.assertEqual(rewrite_path("dataset_backup/frame.jpg"), "dataset_backup/frame.jpg")

    def test_migration_moves_files_and_rewrites_jsonl_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for source, _ in MIGRATIONS:
                (root / source).mkdir(parents=True)
                (root / source / "marker.txt").write_text(source, encoding="utf-8")

            image = root / "vlm_eval_samples_448" / "images" / "frame.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"jpeg-placeholder")
            manifest = root / "vlm_eval_samples_448" / "samples.jsonl"
            manifest.write_text(
                json.dumps({"image_path": "vlm_eval_samples_448/images/frame.jpg"}) + "\n",
                encoding="utf-8",
            )

            summary = migrate_generated_outputs(root)

            self.assertEqual(summary["moved_directories"], len(MIGRATIONS))
            self.assertFalse((root / "vlm_eval_samples_448").exists())
            moved_manifest = root / "outputs/vlm_samples/448/samples.jsonl"
            row = json.loads(moved_manifest.read_text(encoding="utf-8"))
            self.assertEqual(row["image_path"], "outputs/vlm_samples/448/images/frame.jpg")
            self.assertTrue((root / row["image_path"]).is_file())

    def test_destination_collision_changes_nothing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for source, _ in MIGRATIONS:
                (root / source).mkdir(parents=True)
            (root / "outputs/dataset").mkdir(parents=True)

            with self.assertRaises(MigrationConflictError):
                migrate_generated_outputs(root)

            self.assertTrue(all((root / source).is_dir() for source, _ in MIGRATIONS))
```

- [ ] **Step 2: Run the tests and verify import failure**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_migrate_generated_outputs -v
```

Expected: ERROR because `migrate_generated_outputs` does not exist.

- [ ] **Step 3: Implement mappings and safe path rewriting**

Define the exact mapping tuple and boundary-aware rewriting:

```python
MIGRATIONS = (
    ("vlm_eval_samples_448_calibration_validation_d020", "outputs/vlm_samples/448_calibration_validation_d020"),
    ("vlm_eval_samples_448_multiseed_d020", "outputs/vlm_samples/448_multiseed_d020"),
    ("vlm_eval_samples_448", "outputs/vlm_samples/448"),
    ("vlm_eval_samples", "outputs/vlm_samples/default"),
    ("probe_eval_runs", "outputs/probe_evaluations"),
    ("probe_runs", "outputs/probe"),
    ("vlm_eval_runs", "outputs/vlm_evaluations"),
    ("dataset", "outputs/dataset"),
)


class MigrationError(RuntimeError):
    """生成输出迁移失败。"""


class MigrationConflictError(MigrationError):
    """目标路径已存在，迁移必须停止。"""


def rewrite_path(value, mappings=MIGRATIONS):
    for old, new in mappings:
        if value == old:
            return new
        if value.startswith(old + "/"):
            return new + value[len(old):]
    return value
```

Implement recursive JSON value rewriting for dictionaries, lists, and strings. JSON files must be emitted with `ensure_ascii=False, indent=2`; JSONL must remain one JSON object per non-empty line and end with one newline.

- [ ] **Step 4: Implement preflight, inventory, move, validation, and rollback**

The implementation must execute in this order:

```python
def migrate_generated_outputs(project_root):
    project_root = Path(project_root).resolve()
    # 1. Reject a root whose name is '.worktrees' or that is inside a '.worktrees' path.
    # 2. Require every source directory and require every destination to be absent.
    # 3. Parse every source .json and .jsonl before moving anything.
    # 4. Record total files, image files, and image-file bytes for each source.
    # 5. Create only destination parents, then Path.rename each source to destination.
    # 6. Rewrite known path prefixes in migrated .json and .jsonl files.
    # 7. Recount inventory and prove the normalized dangling-reference set is unchanged.
    # 8. If steps 5-7 fail, reverse rewritten prefixes and rename moved directories back.
    # 9. Return counts including moved_directories, files, images, and image_bytes.
```

Use image suffixes `{.png, .jpg, .jpeg, .webp, .gif}` case-insensitively. Validation resolves relative recorded paths against `project_root`; absolute paths remain absolute. Never use `shutil.rmtree`, overwrite flags, or merge-copy behavior.

- [ ] **Step 5: Implement dry-run/apply CLI**

Use `argparse` with:

```python
parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
parser.add_argument("--apply", action="store_true")
```

Without `--apply`, run the same preflight and inventory functions but do not create or move anything. Print source-to-destination mappings plus totals. With `--apply`, call `migrate_generated_outputs()` and print the returned summary.

- [ ] **Step 6: Run focused and full migration tests**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_migrate_generated_outputs -v
conda run -n vla_env python -m unittest tests.test_config_contract tests.test_migrate_generated_outputs -v
```

Expected: all tests PASS and temporary directories are removed by `TemporaryDirectory`.

- [ ] **Step 7: Commit the migration utility**

```bash
git add migrate_generated_outputs.py tests/test_migrate_generated_outputs.py
git commit -m "feat: add generated output migration utility"
```

### Task 3: Remove stale runtime path wording

**Files:**
- Modify: `control_arm.py:60-68`
- Modify: `stage3_probe.py:172-177,542-545`
- Modify: `tests/test_control_arm.py:1-6,190-193`

**Interfaces:**
- Consumes: Task 1 config paths.
- Produces: no behavior change; comments, docstrings, and terminal descriptions match actual paths.

- [ ] **Step 1: Add a source-literal regression check**

Run the check before editing:

```bash
rg -n 'dataset/|probe_runs/|probe_eval_runs/|vlm_eval_samples(_448[^/]*)?/|vlm_eval_runs/' \
  control_arm.py stage3_probe.py tests --glob '*.py'
```

Expected: matches include stale comments/docstrings in `control_arm.py`, `stage3_probe.py`, and `tests/test_control_arm.py`; the fixture string in `tests/test_collect_vlm_eval_samples.py` may remain because it tests arbitrary caller-supplied paths, not a production root.

- [ ] **Step 2: Update comments without changing executable behavior**

Use these meanings:

```text
outputs/dataset/            -> Baseline 1 生成数据
outputs/probe/              -> Stage 3 单次探测输出
outputs/probe_evaluations/  -> Stage 3 批量评估输出
```

Do not change function signatures or path construction: production code already consumes the config paths and will inherit Task 1 values.

- [ ] **Step 3: Run behavior tests and repeat the literal scan**

Run:

```bash
conda run -n vla_env python -m unittest tests.test_control_arm tests.test_stage3_probe tests.test_evaluate_probe -v
rg -n 'probe_runs/|probe_eval_runs/' control_arm.py stage3_probe.py tests --glob '*.py'
```

Expected: all tests PASS; no stale production-root matches remain. Explicit synthetic fixture paths are reviewed and documented rather than mechanically replaced.

- [ ] **Step 4: Commit runtime wording updates**

```bash
git add control_arm.py stage3_probe.py tests/test_control_arm.py
git commit -m "docs: align runtime output path wording"
```

### Task 4: Migrate the existing ignored data

**Files:**
- Execute: `migrate_generated_outputs.py`
- Move, Git-ignored: the eight old roots to `outputs/**`
- Rewrite in place, Git-ignored: migrated `*.json` and `*.jsonl`

**Interfaces:**
- Consumes: Task 2 CLI and `MIGRATIONS`.
- Produces: the exact target tree in the approved design, with working recorded image paths.

- [ ] **Step 1: Capture the pre-migration inventory and dry run**

Run:

```bash
conda run -n vla_env python migrate_generated_outputs.py
```

Expected: lists all eight source-to-destination mappings, reports exactly 8,590 main-workspace images for the current snapshot, reports no collision, and changes no files.

- [ ] **Step 2: Confirm dry run was non-mutating**

Run:

```bash
test -d dataset
test -d probe_runs
test -d probe_eval_runs
test -d vlm_eval_runs
test ! -e outputs
```

Expected: exit status 0.

- [ ] **Step 3: Apply the migration**

Run:

```bash
conda run -n vla_env python migrate_generated_outputs.py --apply
```

Expected: reports `moved_directories=8`; file count, image count, and image-byte count match preflight; no rollback message appears.

- [ ] **Step 4: Verify old roots are gone and all main-workspace images are categorized**

Run:

```bash
test ! -e dataset
test ! -e probe_runs
test ! -e probe_eval_runs
test ! -e vlm_eval_samples
test ! -e vlm_eval_samples_448
test ! -e vlm_eval_samples_448_multiseed_d020
test ! -e vlm_eval_samples_448_calibration_validation_d020
test ! -e vlm_eval_runs
find . -path './.git' -prune -o -path './.worktrees' -prune -o -type f \
  \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' -o -iname '*.gif' \) \
  ! -path './outputs/*' -print
```

Expected: all `test` commands succeed and `find` prints nothing.

- [ ] **Step 5: Validate migrated records and image counts through the utility**

Call the Task 2 validation interface directly:

```bash
conda run -n vla_env python -c 'from pathlib import Path; from migrate_generated_outputs import validate_migrated_outputs; print(validate_migrated_outputs(Path.cwd()))'
```

Expected: all JSON/JSONL parses; the normalized pre-existing dangling-reference set is unchanged; count remains exactly 8,590 images for the current snapshot.

- [ ] **Step 6: Verify Git sees no generated data**

Run:

```bash
git status --short
git check-ignore -v outputs/dataset/episode_summary.jsonl
git check-ignore -v outputs/vlm_samples/448/images/seed_42_d020_left.jpg
```

Expected: generated data is absent from `git status`; both sample paths match `outputs/` in `.gitignore`.

### Task 5: Synchronize live project documentation

**Files:**
- Modify: `README.md:60-72,105-112,150-160,250-257`
- Modify: `docs/worklog/WORKLOG.md` at every active output-path reference
- Modify: `docs/debugging/BUGLOG.md` at every migrated experiment-path reference
- Inspect only: `docs/superpowers/specs/**`, `docs/superpowers/plans/**` except this plan

**Interfaces:**
- Consumes: approved migration mapping and actual Task 4 target paths.
- Produces: current navigation points to real files; historical specs/plans retain their original commands.

- [ ] **Step 1: Scan live documentation for old roots**

Run:

```bash
rg -n 'dataset/|probe_runs/|probe_eval_runs/|vlm_eval_samples(_448[^/]*)?/|vlm_eval_runs/' \
  README.md docs/worklog/WORKLOG.md docs/debugging/BUGLOG.md
```

Expected: each match corresponds to one of the eight migrated roots.

- [ ] **Step 2: Update README structure and usage paths**

Document the category tree and update command-result descriptions to these roots:

```text
outputs/dataset/
outputs/probe/
outputs/probe_evaluations/
outputs/vlm_samples/<sample-set>/
outputs/vlm_evaluations/<run-name>/
```

Add one short note that outputs created before 2026-07-18 were moved using the mapping in the design document. Do not change experiment metrics or conclusions.

- [ ] **Step 3: Update WORKLOG and BUGLOG evidence paths**

Mechanically apply the approved eight-prefix mapping to paths that refer to files now moved in Task 4. Preserve dates, measurements, failure explanations, and historical narrative. Do not edit old `docs/superpowers/specs/**` or `docs/superpowers/plans/**` because those documents describe commands valid at their own creation time.

- [ ] **Step 4: Verify all live documentation paths exist where intended**

Run:

```bash
rg -n 'probe_runs/|probe_eval_runs/|vlm_eval_samples(_448[^/]*)?/|vlm_eval_runs/' \
  README.md docs/worklog/WORKLOG.md docs/debugging/BUGLOG.md
```

Expected: no matches. Any plain `dataset/` prose must be either the new `outputs/dataset/` path or a non-path use of the word dataset.

- [ ] **Step 5: Commit documentation synchronization**

```bash
git add README.md docs/worklog/WORKLOG.md docs/debugging/BUGLOG.md
git commit -m "docs: update generated output locations"
```

### Task 6: Full verification and no-paid-API smoke test

**Files:**
- Verify: all changed source, config, tests, docs, and ignored outputs

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: evidence that path changes are complete and future local generation uses `outputs/`.

- [ ] **Step 1: Run the full unit test suite**

Run:

```bash
conda run -n vla_env python -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] **Step 2: Run a temporary Baseline output smoke test without touching real data**

Run:

```bash
conda run -n vla_env python -c 'import os,pathlib,tempfile; from control_arm import load_config,prepare_dataset; cfg=load_config("/home/pzk/vla_project/sim_config.yaml"); tmp=tempfile.TemporaryDirectory(); os.chdir(tmp.name); dataset_dir,trajectory,summary=prepare_dataset(cfg["dataset"]); assert pathlib.Path(dataset_dir)==pathlib.Path("outputs/dataset"); assert pathlib.Path(dataset_dir).is_dir(); assert pathlib.Path(trajectory).parent==pathlib.Path(dataset_dir); assert pathlib.Path(summary).parent==pathlib.Path(dataset_dir); print(dataset_dir); os.chdir("/"); tmp.cleanup()'
```

Expected: prints `outputs/dataset`; the temporary directory is removed and no repository output is removed or overwritten.

- [ ] **Step 3: Run final stale-path and image-location scans**

Run:

```bash
rg -n '"(dataset|probe_runs|probe_eval_runs|vlm_eval_samples[^"/]*|vlm_eval_runs)"' \
  sim_config.yaml --glob '*.yaml'
find . -path './.git' -prune -o -path './.worktrees' -prune -o -type f \
  \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' -o -iname '*.gif' \) \
  ! -path './outputs/*' -print
git diff --check
git status --short
```

Expected: both stale-path/image scans print nothing; `git diff --check` succeeds; status contains only intentional changes if any remain uncommitted.

- [ ] **Step 4: Record final inventory in the handoff**

Report exact post-migration totals for files, images, and image bytes; list the five category roots; state explicitly that `.worktrees/` was untouched and no Qwen API call occurred.
