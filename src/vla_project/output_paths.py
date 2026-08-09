"""项目管理输出目录的统一安全边界。"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path


def project_root() -> Path:
    """返回包含 ``src/`` 和 ``pyproject.toml`` 的项目根目录。"""
    return Path(__file__).resolve().parents[2]


def _resolve_from_project(value: str | Path, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve(strict=False)
    return (root / path).resolve(strict=False)


def resolve_managed_output(
    requested_path: str | Path,
    *,
    allowed_root: str | Path,
    project_root_override: str | Path | None = None,
    require_child: bool = True,
) -> Path:
    """规范化输出路径，并拒绝离开项目管理根目录的路径。"""
    if not isinstance(requested_path, (str, Path)):
        raise ValueError("invalid managed output path")
    raw = str(requested_path)
    raw_path = Path(raw)
    if (
        not raw.strip()
        or "\x00" in raw
        or raw_path in {Path("."), Path("..")}
    ):
        raise ValueError("invalid managed output path")
    if ".." in raw_path.parts:
        raise ValueError("parent traversal is not allowed")

    root = (
        Path(project_root_override).resolve()
        if project_root_override is not None
        else project_root()
    )
    managed_root = _resolve_from_project(allowed_root, root)
    try:
        managed_root.relative_to(root)
    except ValueError as exc:
        raise ValueError("managed root is outside project root") from exc

    resolved = _resolve_from_project(requested_path, root)
    try:
        relative = resolved.relative_to(managed_root)
    except ValueError as exc:
        raise ValueError("output path is outside managed root") from exc
    if require_child and relative == Path("."):
        raise ValueError("output path must be a leaf directory")
    return resolved


def validate_run_name(run_name: str) -> str:
    """要求运行名是非空的单个目录组件。"""
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name must be a non-empty directory name")
    if (
        run_name in {".", ".."}
        or "/" in run_name
        or "\\" in run_name
        or Path(run_name).is_absolute()
        or Path(run_name).name != run_name
    ):
        raise ValueError("run_name must be one directory component")
    return run_name


def publish_directory_atomically(
    staging_dir: str | Path,
    destination_dir: str | Path,
    *,
    allowed_root: str | Path,
    project_root_override: str | Path | None = None,
) -> Path:
    """原子发布 staging，并在发布失败时恢复已有目标目录。"""
    staging = resolve_managed_output(
        staging_dir,
        allowed_root=allowed_root,
        project_root_override=project_root_override,
    )
    destination = resolve_managed_output(
        destination_dir,
        allowed_root=allowed_root,
        project_root_override=project_root_override,
    )
    if staging == destination:
        raise ValueError("staging and destination must differ")
    if destination in staging.parents or staging in destination.parents:
        raise ValueError("staging and destination must not contain each other")
    if not staging.is_dir():
        raise FileNotFoundError(f"staging directory does not exist: {staging}")
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"destination is not a directory: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent,
        prefix=f".{destination.name}_backup_",
    ) as temp_dir:
        backup = Path(temp_dir) / "previous"
        had_previous = destination.exists()
        if had_previous:
            destination.replace(backup)
        try:
            staging.replace(destination)
        except Exception:
            if had_previous and backup.exists() and not destination.exists():
                backup.replace(destination)
            raise
    return destination


@contextmanager
def staged_output_directory(
    destination_dir: str | Path,
    *,
    allowed_root: str | Path,
    project_root_override: str | Path | None = None,
):
    """在目标同一文件系统生成 staging，正常退出时才发布。"""
    destination = resolve_managed_output(
        destination_dir,
        allowed_root=allowed_root,
        project_root_override=project_root_override,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent,
        prefix=f".{destination.name}_staging_",
    ) as temp_dir:
        staging = Path(temp_dir) / destination.name
        staging.mkdir()
        yield staging
        publish_directory_atomically(
            staging,
            destination,
            allowed_root=allowed_root,
            project_root_override=project_root_override,
        )
