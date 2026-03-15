from __future__ import annotations

from typing import List

import torch
import torch.distributed as dist
from lightning import Callback, LightningModule, Trainer


class WandbRuntimeMetricsCallback(Callback):
    """Logs the worst peak CUDA reserved-memory percentage during training."""

    def __init__(self, log_every_n_steps: int = 20) -> None:
        self.log_every_n_steps = max(1, log_every_n_steps)
        self._epoch_values: List[float] = []

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
                "worst_peak_reserved_pct_epoch_p99": float(torch.quantile(values, 0.99).item()),
                "worst_peak_reserved_pct_epoch_min": float(values.min().item()),
            },
            step=trainer.global_step,
        )
