from __future__ import annotations

from numbers import Number
from typing import Dict, Optional

import torch
from lightning import Callback, LightningModule, Trainer


class WandbRuntimeMetricsCallback(Callback):
    """Logs one representative GPU and basic train setup metrics."""

    def __init__(self) -> None:
        self._last_logged_step = -1

    @staticmethod
    def _log_metrics(trainer: Trainer, metrics: Dict[str, float], step: int) -> None:
        if not metrics or not trainer.loggers:
            return
        for logger in trainer.loggers:
            logger.log_metrics(metrics, step=step)

    @staticmethod
    def _resolve_accumulate_grad_batches(trainer: Trainer) -> Optional[float]:
        value = trainer.accumulate_grad_batches
        if isinstance(value, Number):
            return float(value)
        if isinstance(value, dict):
            resolved = None
            for epoch, epoch_value in sorted(value.items()):
                if trainer.current_epoch >= int(epoch):
                    resolved = epoch_value
            if isinstance(resolved, Number):
                return float(resolved)
        return None

    @staticmethod
    def _collect_gpu_metrics(device: torch.device) -> Dict[str, float]:
        if device.type != "cuda":
            return {}

        device_index = device.index if device.index is not None else torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        allocated_bytes = torch.cuda.memory_allocated(device_index)
        reserved_bytes = torch.cuda.memory_reserved(device_index)
        total_bytes = float(total_bytes)
        gib = float(1024**3)

        return {
            "System/GPU Memory Allocated (%)": 100.0 * (1.0 - free_bytes / total_bytes),
            "System/Process GPU Memory Allocated (%)": 100.0
            * allocated_bytes
            / total_bytes,
            "System/Process GPU Memory Reserved (%)": 100.0 * reserved_bytes / total_bytes,
            "System/Process GPU Memory Allocated (GiB)": allocated_bytes / gib,
            "System/Process GPU Memory Reserved (GiB)": reserved_bytes / gib,
        }

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        metrics: Dict[str, float] = {}
        batch_size = getattr(trainer.datamodule, "train_batch_size", None)
        if isinstance(batch_size, Number):
            metrics["train_setup/batch_size"] = float(batch_size)

        accumulate_grad_batches = self._resolve_accumulate_grad_batches(trainer)
        if accumulate_grad_batches is not None:
            metrics["train_setup/accumulate_grad_batches"] = accumulate_grad_batches

        self._log_metrics(trainer, metrics, step=trainer.global_step)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if not trainer.is_global_zero or not torch.cuda.is_available():
            return

        step = trainer.global_step
        if step <= 0 or step == self._last_logged_step:
            return
        if step % max(1, trainer.log_every_n_steps) != 0:
            return

        metrics = self._collect_gpu_metrics(pl_module.device)
        self._log_metrics(trainer, metrics, step=step)
        self._last_logged_step = step
