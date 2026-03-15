from __future__ import annotations

import time
from typing import List

import torch
import torch.distributed as dist
import wandb
from lightning import Callback, LightningModule, Trainer


class WandbRuntimeMetricsCallback(Callback):
    """Logs the worst peak CUDA reserved-memory percentage during training."""

    def __init__(self, log_every_n_steps: int = 20) -> None:
        self.log_every_n_steps = max(1, log_every_n_steps)
        self._epoch_values: List[float] = []
        self._accumulated_runtime_sec = 0.0
        self._fit_start_time: float | None = None
        self._progress_points: List[list[float]] = []

    @staticmethod
    def _get_cuda_device(pl_module: LightningModule) -> torch.device | None:
        device = pl_module.device
        if not torch.cuda.is_available() or device.type != "cuda":
            return None
        return device

    @staticmethod
    def _reduce_max(value: float, device: torch.device) -> float:
        reduced = torch.tensor(value, device=device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
        return float(reduced.item())

    @staticmethod
    def _log_metrics(trainer: Trainer, metrics: dict[str, float], step: int) -> None:
        for logger in trainer.loggers:
            logger.log_metrics(metrics, step=step)

    @staticmethod
    def _get_wandb_logger(trainer: Trainer):
        for logger in trainer.loggers:
            if logger.__class__.__name__ == "WandbLogger":
                return logger
        return None

    def _runtime_seconds(self) -> float:
        if self._fit_start_time is None:
            return self._accumulated_runtime_sec
        return self._accumulated_runtime_sec + (time.monotonic() - self._fit_start_time)

    def state_dict(self) -> dict[str, object]:
        return {
            "accumulated_runtime_sec": self._runtime_seconds(),
            "progress_points": self._progress_points,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self._accumulated_runtime_sec = float(state_dict.get("accumulated_runtime_sec", 0.0))
        self._progress_points = [list(point) for point in state_dict.get("progress_points", [])]

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer, pl_module
        self._fit_start_time = time.monotonic()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer, pl_module
        self._accumulated_runtime_sec = self._runtime_seconds()
        self._fit_start_time = None

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if pl_module.global_rank == 0:
            self._epoch_values = []

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        del trainer, batch, batch_idx

        device = self._get_cuda_device(pl_module)
        if device is None:
            return

        torch.cuda.reset_peak_memory_stats(device)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx

        if not trainer.loggers:
            return

        device = self._get_cuda_device(pl_module)
        if device is None:
            return

        total_memory = torch.cuda.get_device_properties(device).total_memory
        peak_reserved = torch.cuda.max_memory_reserved(device)
        local_peak_reserved_pct = 100.0 * peak_reserved / total_memory
        worst_peak_reserved_pct = self._reduce_max(local_peak_reserved_pct, device)

        if pl_module.global_rank != 0:
            return

        self._epoch_values.append(worst_peak_reserved_pct)

        if trainer.global_step > 0 and trainer.global_step % self.log_every_n_steps == 0:
            self._log_metrics(
                trainer,
                {"worst_peak_reserved_pct": worst_peak_reserved_pct},
                step=trainer.global_step,
            )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if pl_module.global_rank != 0 or not trainer.loggers or not self._epoch_values:
            return

        values = torch.tensor(self._epoch_values, dtype=torch.float32)
        self._log_metrics(
            trainer,
            {
                "worst_peak_reserved_pct_epoch_max": float(values.max().item()),
            },
            step=trainer.global_step,
        )

        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is None or trainer.max_epochs is None or trainer.max_epochs <= 0:
            return

        elapsed_training_hours = self._runtime_seconds() / 3600.0
        epoch_progress_pct = min(100.0, 100.0 * (pl_module.current_epoch + 1) / trainer.max_epochs)
        self._progress_points.append([elapsed_training_hours, epoch_progress_pct])
        table = wandb.Table(
            columns=["elapsed_training_hours", "epoch_progress_pct"],
            data=self._progress_points,
        )
        wandb_logger.experiment.log(
            {
                "training_progress_vs_runtime": wandb.plot.line(
                    table,
                    "elapsed_training_hours",
                    "epoch_progress_pct",
                    title="Training Progress vs Runtime",
                )
            },
            step=trainer.global_step,
        )
