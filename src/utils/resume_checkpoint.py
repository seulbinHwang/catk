from __future__ import annotations

from pathlib import Path
from typing import Any


def _is_empty(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def find_latest_task_checkpoint(
    *,
    log_dir: str | Path,
    task_name: str,
    checkpoint_name: str = "epoch_last.ckpt",
    exclude_output_dir: str | Path | None = None,
) -> Path | None:
    """Return the newest checkpoint for a task across Hydra run directories."""
    if _is_empty(task_name):
        raise ValueError("task_name is required for automatic resume.")
    if _is_empty(checkpoint_name):
        raise ValueError("checkpoint_name is required for automatic resume.")

    task_run_dir = Path(log_dir).expanduser() / str(task_name) / "runs"
    candidates = [
        path
        for path in task_run_dir.glob(f"*/checkpoints/{checkpoint_name}")
        if path.is_file()
    ]
    if exclude_output_dir not in (None, ""):
        output_dir = Path(str(exclude_output_dir)).expanduser()
        candidates = [
            path for path in candidates if not _is_relative_to(path, output_dir)
        ]
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, path.as_posix()),
    )


def resolve_fit_resume_ckpt_path(cfg) -> str | None:
    """Resolve the checkpoint path used by ``action=fit``.

    Explicit ``ckpt_path`` always wins. If ``resume.auto=true`` and no explicit
    checkpoint is set, the latest task-local ``epoch_last.ckpt`` is selected.
    """
    explicit_ckpt_path = cfg.get("ckpt_path")
    if not _is_empty(explicit_ckpt_path):
        return str(explicit_ckpt_path)

    resume_cfg = cfg.get("resume")
    if not resume_cfg or not bool(resume_cfg.get("auto", False)):
        return None

    resume_task_name = resume_cfg.get("task_name") or cfg.get("task_name")
    checkpoint_name = resume_cfg.get("checkpoint_name", "epoch_last.ckpt")
    latest_checkpoint = find_latest_task_checkpoint(
        log_dir=cfg.paths.log_dir,
        task_name=str(resume_task_name),
        checkpoint_name=str(checkpoint_name),
        exclude_output_dir=cfg.paths.output_dir,
    )
    if latest_checkpoint is not None:
        return latest_checkpoint.as_posix()

    if bool(resume_cfg.get("require_checkpoint", True)):
        raise FileNotFoundError(
            "resume.auto=true but no checkpoint was found under "
            f"{Path(str(cfg.paths.log_dir)) / str(resume_task_name) / 'runs'} "
            f"matching checkpoints/{checkpoint_name}."
        )
    return None
