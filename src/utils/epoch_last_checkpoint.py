from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import wandb
from lightning import Callback, LightningModule, Trainer

_CHECK_VAL_INTERVAL_UNSET = object()


class EpochLastCheckpointCallback(Callback):
    """Save a single latest-epoch checkpoint and optionally upload it to W&B."""

    def __init__(
        self,
        dirpath: str,
        filename: str = "epoch_last.ckpt",
        save_weights_only: bool = False,
        upload_to_wandb: bool = True,
        artifact_name: str | None = None,
    ) -> None:
        self.dirpath = Path(dirpath)
        self.filename = filename if filename.endswith(".ckpt") else f"{filename}.ckpt"
        self.save_weights_only = save_weights_only
        self.upload_to_wandb = upload_to_wandb
        self.artifact_name = artifact_name
        self._last_saved_epoch: int | None = None
        self._pending_validation = False
        self._pending_validation_epoch: int | None = None
        self._resume_validation_pending = False
        self._resume_validation_epoch: int | None = None
        self._resume_check_val_every_n_epoch: Any = _CHECK_VAL_INTERVAL_UNSET

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return None if value is None else int(value)

    @staticmethod
    def _get_wandb_logger(trainer: Trainer):
        for logger in trainer.loggers:
            if logger.__class__.__name__ == "WandbLogger":
                return logger
        return None

    @staticmethod
    def _wandb_artifact_upload_enabled() -> bool:
        wandb_mode = os.getenv("WANDB_MODE", "").strip().lower()
        wandb_disabled = os.getenv("WANDB_DISABLED", "").strip().lower()
        if wandb_mode in {"offline", "dryrun", "disabled"}:
            return False
        if wandb_disabled in {"true", "1", "yes"}:
            return False
        if wandb.run is None:
            return False

        run_mode = str(getattr(getattr(wandb.run, "settings", None), "mode", "")).strip().lower()
        return run_mode not in {"offline", "dryrun", "disabled"}

    def _checkpoint_path(self) -> Path:
        return self.dirpath / self.filename

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_saved_epoch": self._last_saved_epoch,
            "pending_validation": self._pending_validation,
            "pending_validation_epoch": self._pending_validation_epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._last_saved_epoch = self._optional_int(state_dict.get("last_saved_epoch"))
        self._pending_validation = bool(state_dict.get("pending_validation", False))
        self._pending_validation_epoch = self._optional_int(
            state_dict.get("pending_validation_epoch")
        )
        self._resume_validation_pending = self._pending_validation
        self._resume_validation_epoch = self._pending_validation_epoch
        self._resume_check_val_every_n_epoch = _CHECK_VAL_INTERVAL_UNSET

    def _save_checkpoint(self, trainer: Trainer, *, pending_validation: bool) -> None:
        self._pending_validation = pending_validation
        self._pending_validation_epoch = (
            int(trainer.current_epoch) if pending_validation else None
        )
        self.dirpath.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self._checkpoint_path()
        trainer.save_checkpoint(checkpoint_path, weights_only=self.save_weights_only)
        self._last_saved_epoch = int(trainer.current_epoch)

        if not self.upload_to_wandb or not trainer.is_global_zero:
            return

        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is None or not self._wandb_artifact_upload_enabled():
            return

        artifact = wandb.Artifact(
            name=self.artifact_name or f"epoch-last-{wandb_logger.experiment.id}",
            type="model",
            metadata={
                "epoch": int(trainer.current_epoch + 1),
                "global_step": int(trainer.global_step),
                "save_weights_only": bool(self.save_weights_only),
                "original_filename": checkpoint_path.name,
            },
        )
        artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)
        wandb_logger.experiment.log_artifact(artifact, aliases=["latest", "epoch_last"])

    def _already_saved_for_epoch(self, trainer: Trainer) -> bool:
        return self._last_saved_epoch == int(trainer.current_epoch)

    @staticmethod
    def _has_validation_batches(trainer: Trainer) -> bool:
        num_val_batches = getattr(trainer, "num_val_batches", None)
        if isinstance(num_val_batches, list):
            return sum(num_val_batches) > 0
        if isinstance(num_val_batches, (int, float)):
            return not math.isinf(num_val_batches) and num_val_batches > 0

        limit_val_batches = getattr(trainer, "limit_val_batches", None)
        if isinstance(limit_val_batches, (int, float)):
            return limit_val_batches > 0

        return True

    @staticmethod
    def _should_run_validation_after_epoch(trainer: Trainer) -> bool:
        check_val_every_n_epoch = getattr(trainer, "check_val_every_n_epoch", 1)
        if check_val_every_n_epoch is None:
            return False
        if int(check_val_every_n_epoch) <= 0:
            return False

        val_check_interval = getattr(trainer, "val_check_interval", 1.0)
        if val_check_interval != 1.0:
            return False

        if not EpochLastCheckpointCallback._has_validation_batches(trainer):
            return False

        return (int(trainer.current_epoch) + 1) % int(check_val_every_n_epoch) == 0

    def _restore_forced_validation_interval(self, trainer: Trainer) -> None:
        if self._resume_check_val_every_n_epoch is _CHECK_VAL_INTERVAL_UNSET:
            return

        trainer.check_val_every_n_epoch = self._resume_check_val_every_n_epoch
        self._resume_check_val_every_n_epoch = _CHECK_VAL_INTERVAL_UNSET

    def _clear_resume_validation_state(self, trainer: Trainer) -> None:
        self._restore_forced_validation_interval(trainer)
        self._resume_validation_pending = False
        self._resume_validation_epoch = None

    @staticmethod
    def _is_last_train_batch(trainer: Trainer, batch_idx: int) -> bool:
        fit_loop = getattr(trainer, "fit_loop", None)
        epoch_loop = getattr(fit_loop, "epoch_loop", None)
        batch_progress = getattr(epoch_loop, "batch_progress", None)
        if batch_progress is not None and bool(getattr(batch_progress, "is_last_batch", False)):
            return True

        num_training_batches = getattr(trainer, "num_training_batches", None)
        if isinstance(num_training_batches, (int, float)) and not math.isinf(num_training_batches):
            return batch_idx + 1 >= int(num_training_batches)

        return False

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch
        if trainer.sanity_checking or self._already_saved_for_epoch(trainer):
            return

        # Save before fit-time validation begins so epoch_last.ckpt stays resumable
        # at the latest training state even when validation runs before epoch end hooks.
        if self._is_last_train_batch(trainer, batch_idx) and self._should_run_validation_after_epoch(
            trainer
        ):
            self._save_checkpoint(trainer, pending_validation=True)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if trainer.sanity_checking or not self._resume_validation_pending:
            return

        if self._resume_validation_epoch is not None and int(trainer.current_epoch) != int(
            self._resume_validation_epoch
        ):
            self._clear_resume_validation_state(trainer)
            return

        if self._resume_check_val_every_n_epoch is _CHECK_VAL_INTERVAL_UNSET:
            self._resume_check_val_every_n_epoch = trainer.check_val_every_n_epoch
        trainer.check_val_every_n_epoch = 1

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        self._restore_forced_validation_interval(trainer)
        if trainer.sanity_checking or self._already_saved_for_epoch(trainer):
            return

        self._save_checkpoint(trainer, pending_validation=False)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if trainer.sanity_checking:
            return

        should_persist_completed_validation = self._pending_validation and (
            self._pending_validation_epoch is None
            or int(trainer.current_epoch) == int(self._pending_validation_epoch)
        )
        self._pending_validation = False
        self._pending_validation_epoch = None
        self._clear_resume_validation_state(trainer)
        if should_persist_completed_validation:
            self._save_checkpoint(trainer, pending_validation=False)
