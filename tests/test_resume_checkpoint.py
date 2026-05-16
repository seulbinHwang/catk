from __future__ import annotations

import os
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_RESUME_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "utils" / "resume_checkpoint.py"
)
_RESUME_SPEC = importlib.util.spec_from_file_location(
    "resume_checkpoint_for_test",
    _RESUME_MODULE_PATH,
)
assert _RESUME_SPEC is not None and _RESUME_SPEC.loader is not None
_RESUME_MODULE = importlib.util.module_from_spec(_RESUME_SPEC)
_RESUME_SPEC.loader.exec_module(_RESUME_MODULE)

find_latest_task_checkpoint = _RESUME_MODULE.find_latest_task_checkpoint
resolve_fit_resume_ckpt_path = _RESUME_MODULE.resolve_fit_resume_ckpt_path


class _Config(SimpleNamespace):
    def get(self, name, default=None):
        return getattr(self, name, default)


def _write_checkpoint(path: Path, mtime: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("checkpoint")
    os.utime(path, (mtime, mtime))
    return path


def test_find_latest_task_checkpoint_uses_newest_run(tmp_path: Path) -> None:
    old_ckpt = _write_checkpoint(
        tmp_path / "logs" / "smart_pretrain" / "runs" / "old" / "checkpoints" / "epoch_last.ckpt",
        mtime=100,
    )
    new_ckpt = _write_checkpoint(
        tmp_path / "logs" / "smart_pretrain" / "runs" / "new" / "checkpoints" / "epoch_last.ckpt",
        mtime=200,
    )

    actual = find_latest_task_checkpoint(
        log_dir=tmp_path / "logs",
        task_name="smart_pretrain",
        checkpoint_name="epoch_last.ckpt",
    )

    assert actual == new_ckpt
    assert actual != old_ckpt


def test_resolve_fit_resume_ckpt_path_prefers_explicit_ckpt(tmp_path: Path) -> None:
    explicit_ckpt = tmp_path / "manual.ckpt"
    cfg = _Config(
        ckpt_path=explicit_ckpt.as_posix(),
        task_name="smart_pretrain",
        paths=SimpleNamespace(
            log_dir=(tmp_path / "logs").as_posix(),
            output_dir=(tmp_path / "logs" / "smart_pretrain" / "runs" / "current").as_posix(),
        ),
        resume={"auto": True},
    )

    assert resolve_fit_resume_ckpt_path(cfg) == explicit_ckpt.as_posix()


def test_resolve_fit_resume_ckpt_path_auto_uses_latest_previous_run(tmp_path: Path) -> None:
    previous_ckpt = _write_checkpoint(
        tmp_path / "logs" / "smart_pretrain" / "runs" / "previous" / "checkpoints" / "epoch_last.ckpt",
        mtime=100,
    )
    current_ckpt = _write_checkpoint(
        tmp_path / "logs" / "smart_pretrain" / "runs" / "current" / "checkpoints" / "epoch_last.ckpt",
        mtime=200,
    )
    cfg = _Config(
        ckpt_path=None,
        task_name="smart_pretrain",
        paths=SimpleNamespace(
            log_dir=(tmp_path / "logs").as_posix(),
            output_dir=(tmp_path / "logs" / "smart_pretrain" / "runs" / "current").as_posix(),
        ),
        resume={"auto": True, "checkpoint_name": "epoch_last.ckpt"},
    )

    actual = resolve_fit_resume_ckpt_path(cfg)

    assert actual == previous_ckpt.as_posix()
    assert actual != current_ckpt.as_posix()


def test_resolve_fit_resume_ckpt_path_raises_when_required_missing(tmp_path: Path) -> None:
    cfg = _Config(
        ckpt_path=None,
        task_name="smart_pretrain",
        paths=SimpleNamespace(
            log_dir=(tmp_path / "logs").as_posix(),
            output_dir=(tmp_path / "logs" / "smart_pretrain" / "runs" / "current").as_posix(),
        ),
        resume={"auto": True, "require_checkpoint": True},
    )

    with pytest.raises(FileNotFoundError):
        resolve_fit_resume_ckpt_path(cfg)
