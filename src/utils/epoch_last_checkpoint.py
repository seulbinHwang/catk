from __future__ import annotations

import os
from pathlib import Path

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

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if trainer.sanity_checking:
            return

        self.dirpath.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self._checkpoint_path()
        trainer.save_checkpoint(checkpoint_path, weights_only=self.save_weights_only)

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
