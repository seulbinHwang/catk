from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import wandb
from lightning import Callback, LightningModule, Trainer


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

    def _save_checkpoint(self, trainer: Trainer) -> None:
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
        if self._is_last_train_batch(trainer, batch_idx):
            self._save_checkpoint(trainer)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if trainer.sanity_checking or self._already_saved_for_epoch(trainer):
            return

        self._save_checkpoint(trainer)
