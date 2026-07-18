"""把分散的生成输出安全迁移到统一的 ``outputs/`` 目录。

默认只做预检查；显式传入 ``--apply`` 才会移动目录。迁移会同步修复
JSON/JSONL 中的旧路径，并在失败时尽力恢复原目录结构。
"""

import argparse
import json
from pathlib import Path


MIGRATIONS = (
    (
        "vlm_eval_samples_448_calibration_validation_d020",
        "outputs/vlm_samples/448_calibration_validation_d020",
    ),
    (
        "vlm_eval_samples_448_multiseed_d020",
        "outputs/vlm_samples/448_multiseed_d020",
    ),
    ("vlm_eval_samples_448", "outputs/vlm_samples/448"),
    ("vlm_eval_samples", "outputs/vlm_samples/default"),
    ("probe_eval_runs", "outputs/probe_evaluations"),
    ("probe_runs", "outputs/probe"),
    ("vlm_eval_runs", "outputs/vlm_evaluations"),
    ("dataset", "outputs/dataset"),
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


class MigrationError(RuntimeError):
    """生成输出迁移失败。"""


class MigrationConflictError(MigrationError):
    """目标路径已存在，迁移必须停止。"""


def rewrite_path(value, mappings=MIGRATIONS):
    """只替换完整目录名或目录边界后的前缀，避免误伤相似名称。"""
    for old, new in mappings:
        if value == old:
            return new
        if value.startswith(old + "/"):
            return new + value[len(old) :]
        absolute_component = "/" + old
        if value.endswith(absolute_component):
            return value[: -len(absolute_component)] + "/" + new
        absolute_prefix = absolute_component + "/"
        if absolute_prefix in value:
            return value.replace(absolute_prefix, "/" + new + "/", 1)
    return value


def _rewrite_value(value, mappings):
    """递归改写 JSON 结构中的路径字符串，保留其他值原样。"""
    if isinstance(value, dict):
        return {key: _rewrite_value(item, mappings) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_value(item, mappings) for item in value]
    if isinstance(value, str):
        return rewrite_path(value, mappings)
    return value


def _record_files(roots):
    """按稳定顺序返回目录中的 JSON 和 JSONL 文件。"""
    files = []
    for root in roots:
        if root.is_dir():
            files.extend(path for path in root.rglob("*.json") if path.is_file())
            files.extend(path for path in root.rglob("*.jsonl") if path.is_file())
    return sorted(set(files))


def _read_jsonl(path):
    """解析 JSONL，并在错误中保留文件名和行号。"""
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise MigrationError(f"无法解析 {path}:{line_number}: {exc}") from exc
    return rows


def _parse_record_file(path):
    """预解析记录文件，保证移动前就能发现损坏数据。"""
    try:
        if path.suffix == ".jsonl":
            return _read_jsonl(path)
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MigrationError(f"无法解析 {path}: {exc}") from exc


def _write_record_file(path, value):
    """通过同目录临时文件原子替换记录，避免留下半写入文件。"""
    temporary_path = path.with_name(path.name + ".migration-tmp")
    if path.suffix == ".jsonl":
        content = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in value
        )
    else:
        content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def _rewrite_record_trees(roots, mappings):
    """改写多个输出树中的所有 JSON/JSONL 字符串路径。"""
    for path in _record_files(roots):
        value = _parse_record_file(path)
        rewritten = _rewrite_value(value, mappings)
        if rewritten != value:
            _write_record_file(path, rewritten)


def _inventory(roots):
    """统计文件数、图片数和图片文件字节数。"""
    files = 0
    images = 0
    image_bytes = 0
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            files += 1
            if path.suffix.lower() in IMAGE_SUFFIXES:
                images += 1
                image_bytes += path.stat().st_size
    return {"files": files, "images": images, "image_bytes": image_bytes}


def _validate_project_root(project_root):
    """拒绝误把独立 worktree 当成主工作区。"""
    project_root = Path(project_root).resolve()
    if ".worktrees" in project_root.parts:
        raise MigrationError(f"禁止迁移 .worktrees 中的输出: {project_root}")
    if not project_root.is_dir():
        raise MigrationError(f"项目根目录不存在: {project_root}")
    return project_root


def preflight_migration(project_root):
    """在不写文件的前提下检查来源、冲突、记录格式并返回清单。"""
    project_root = _validate_project_root(project_root)
    sources = [project_root / source for source, _ in MIGRATIONS]
    destinations = [project_root / destination for _, destination in MIGRATIONS]

    missing = [str(path) for path in sources if not path.is_dir()]
    if missing:
        raise MigrationError("缺少待迁移目录: " + ", ".join(missing))

    conflicts = [str(path) for path in destinations if path.exists()]
    if conflicts:
        raise MigrationConflictError("目标路径已存在: " + ", ".join(conflicts))

    for path in _record_files(sources):
        _parse_record_file(path)

    summary = _inventory(sources)
    summary["source_directories"] = len(sources)
    summary["missing_image_references"] = len(
        _missing_image_references(project_root, sources)
    )
    return summary


def _iter_image_path_values(value):
    """找出记录中明确指向图片的路径字段。"""
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                isinstance(item, str)
                and (key.endswith("image_path") or key == "annotated_path")
            ):
                yield item
            yield from _iter_image_path_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_image_path_values(item)


def _missing_image_references(project_root, roots):
    """返回规范化后的历史悬空图片引用集合，供迁移前后精确对比。"""
    missing = set()
    for record_path in _record_files(roots):
        value = _parse_record_file(record_path)
        normalized_record = rewrite_path(
            record_path.relative_to(project_root).as_posix()
        )
        for image_path_value in _iter_image_path_values(value):
            image_path = Path(image_path_value)
            resolved = image_path if image_path.is_absolute() else project_root / image_path
            if not resolved.is_file():
                missing.add((normalized_record, rewrite_path(image_path_value)))
    return missing


def validate_migrated_outputs(project_root):
    """验证新目录和记录格式，并报告仍存在的历史悬空图片引用。"""
    project_root = _validate_project_root(project_root)
    destinations = [project_root / destination for _, destination in MIGRATIONS]
    missing = [str(path) for path in destinations if not path.is_dir()]
    if missing:
        raise MigrationError("缺少迁移后目录: " + ", ".join(missing))

    for record_path in _record_files(destinations):
        _parse_record_file(record_path)
    summary = _inventory(destinations)
    summary["missing_image_references"] = len(
        _missing_image_references(project_root, destinations)
    )
    return summary


def _remove_empty_output_parents(project_root):
    """回滚后只移除本次创建且已经为空的 outputs 父目录。"""
    outputs_root = project_root / "outputs"
    if not outputs_root.exists():
        return
    directories = sorted(
        (path for path in outputs_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in directories:
        try:
            path.rmdir()
        except OSError:
            pass
    try:
        outputs_root.rmdir()
    except OSError:
        pass


def migrate_generated_outputs(project_root):
    """执行迁移；任何移动后的失败都会触发反向改写和目录回滚。"""
    project_root = _validate_project_root(project_root)
    before = preflight_migration(project_root)
    source_paths = [project_root / source for source, _ in MIGRATIONS]
    missing_before = _missing_image_references(project_root, source_paths)
    moved = []
    destinations = [project_root / destination for _, destination in MIGRATIONS]

    try:
        for source_name, destination_name in MIGRATIONS:
            source = project_root / source_name
            destination = project_root / destination_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            moved.append((source, destination))

        _rewrite_record_trees(destinations, MIGRATIONS)
        after = validate_migrated_outputs(project_root)
        missing_after = _missing_image_references(project_root, destinations)
        if missing_after != missing_before:
            raise MigrationError(
                "迁移改变了历史悬空图片引用集合: "
                f"before={len(missing_before)}, after={len(missing_after)}"
            )
        for key in ("files", "images", "image_bytes"):
            if after[key] != before[key]:
                raise MigrationError(
                    f"迁移前后 {key} 不一致: {before[key]} != {after[key]}"
                )
    except Exception as exc:
        reverse_mappings = tuple((new, old) for old, new in MIGRATIONS)
        try:
            _rewrite_record_trees(
                [destination for _, destination in moved if destination.exists()],
                reverse_mappings,
            )
            for source, destination in reversed(moved):
                if destination.exists():
                    destination.rename(source)
            _remove_empty_output_parents(project_root)
        except Exception as rollback_exc:
            raise MigrationError(
                f"迁移失败且回滚不完整: {exc}; 回滚错误: {rollback_exc}"
            ) from rollback_exc
        if isinstance(exc, MigrationError):
            raise
        raise MigrationError(f"迁移失败，已回滚: {exc}") from exc

    result = dict(after)
    result["moved_directories"] = len(moved)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="通过预检查后实际移动目录；默认只预览。",
    )
    args = parser.parse_args()

    print("迁移映射:")
    for source, destination in MIGRATIONS:
        print(f"  {source}/ -> {destination}/")

    if args.apply:
        summary = migrate_generated_outputs(args.project_root)
        print("迁移完成:", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        summary = preflight_migration(args.project_root)
        print("预检查通过（未移动文件）:", json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
