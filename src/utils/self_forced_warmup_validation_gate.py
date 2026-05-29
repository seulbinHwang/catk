from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from lightning import Callback, LightningModule, Trainer


class SelfForcedWarmupValidationGateCallback(Callback):
    """Disable fit-time validation while self-forced estimator warmup is active."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = bool(verbose)
        self._original_should_check_val_epoch: Callable[[], bool] | None = None
        self._last_logged_global_step: int | None = None

    @staticmethod
    def _is_warmup_active(pl_module: LightningModule) -> bool:
        is_active = getattr(pl_module, "_is_self_forced_estimator_warmup_active", None)
        if not callable(is_active):
            return False
        return bool(is_active())

    def _log_skip_once_per_step(self, trainer: Trainer) -> None:
        if not self.verbose or not trainer.is_global_zero:
            return
        global_step = int(getattr(trainer, "global_step", 0))
        if self._last_logged_global_step == global_step:
            return
        self._last_logged_global_step = global_step
        print(
            "[SelfForcedWarmupValidationGateCallback] "
            f"skip validation during estimator warmup at global_step={global_step}",
            flush=True,
        )

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch_loop = getattr(getattr(trainer, "fit_loop", None), "epoch_loop", None)
        if epoch_loop is None or not hasattr(epoch_loop, "_should_check_val_epoch"):
            raise RuntimeError("Could not locate Lightning training epoch loop validation hook.")

        if self._original_should_check_val_epoch is not None:
            return

        original_should_check_val_epoch = epoch_loop._should_check_val_epoch
        self._original_should_check_val_epoch = original_should_check_val_epoch

        def gated_should_check_val_epoch(*args: Any, **kwargs: Any) -> bool:
            if not bool(original_should_check_val_epoch(*args, **kwargs)):
                return False
            if self._is_warmup_active(pl_module):
                self._log_skip_once_per_step(trainer)
                return False
            return True

        epoch_loop._should_check_val_epoch = gated_should_check_val_epoch

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if self._original_should_check_val_epoch is None:
            return
        epoch_loop = getattr(getattr(trainer, "fit_loop", None), "epoch_loop", None)
        if epoch_loop is not None:
            epoch_loop._should_check_val_epoch = self._original_should_check_val_epoch
        self._original_should_check_val_epoch = None


class SelfForcedWarmupCheckpointCallback(Callback):
    """Save a reusable checkpoint when estimator-only warmup finishes."""

    def __init__(
        self,
        dirpath: str | None = None,
        filename: str = "self_forced_after_fake_warmup.ckpt",
        verbose: bool = True,
    ) -> None:
        self.dirpath = dirpath
        self.filename = filename if filename.endswith(".ckpt") else f"{filename}.ckpt"
        self.verbose = bool(verbose)
        self._saved = False

    @staticmethod
    def _get_int_attr(obj: object, name: str, default: int = 0) -> int:
        return int(getattr(obj, name, default))

    def _should_save_after_epoch(self, trainer: Trainer, pl_module: LightningModule) -> bool:
        warmup_epochs = self._get_int_attr(pl_module, "self_forced_estimator_warmup_epochs", 0)
        if warmup_epochs <= 0:
            return False
        start_epoch = self._get_int_attr(pl_module, "self_forced_start_epoch", 0)
        return int(trainer.current_epoch) + 1 == start_epoch + warmup_epochs

    def _checkpoint_path(self, trainer: Trainer) -> Path:
        if self.dirpath:
            checkpoint_dir = Path(self.dirpath)
        else:
            checkpoint_dir = Path(str(trainer.default_root_dir)) / "checkpoints"
        return checkpoint_dir / self.filename

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking or self._saved:
            return
        if not self._should_save_after_epoch(trainer, pl_module):
            return

        checkpoint_path = self._checkpoint_path(trainer)
        if trainer.is_global_zero:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(checkpoint_path), weights_only=False)
            if self.verbose:
                print(
                    "[SelfForcedWarmupCheckpointCallback] "
                    f"saved reusable warmup checkpoint: {checkpoint_path}",
                    flush=True,
                )
        trainer.strategy.barrier("self_forced_warmup_checkpoint_saved")
        self._saved = True
