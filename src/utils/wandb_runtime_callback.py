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
        self._epoch_train_start_time: float | None = None
        self._epoch_validation_sec = 0.0
        self._validation_start_time: float | None = None

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
    def _lightning_log_step(trainer: Trainer, *, epoch_end: bool = False) -> int:
        """Match Lightning's internal logger step so W&B history stays monotonic."""
        step = trainer.fit_loop.epoch_loop._batches_that_stepped
        if epoch_end:
            step -= 1
        return max(step, 0)

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
        self._fit_start_time = time.monotonic()
        if pl_module.global_rank != 0 or not trainer.loggers:
            return

        datamodule = getattr(trainer, "datamodule", None)
        per_device_batch_size = getattr(datamodule, "train_batch_size", None)
        val_batch_size = getattr(datamodule, "val_batch_size", None)
        num_workers = getattr(datamodule, "num_workers", None)
        if per_device_batch_size is None:
            return

        train_setup_metrics = {
            "train_setup/global_batch_size": int(per_device_batch_size) * int(trainer.world_size),
        }
        if val_batch_size is not None:
            train_setup_metrics["train_setup/val_batch_size"] = int(val_batch_size)
        if num_workers is not None:
            train_setup_metrics["train_setup/num_workers"] = int(num_workers)

        n_rollout_closed_val = getattr(pl_module, "n_rollout_closed_val", None)
        if n_rollout_closed_val is not None:
            train_setup_metrics["train_setup/n_rollout_closed_val"] = int(n_rollout_closed_val)

        self._log_metrics(
            trainer,
            train_setup_metrics,
            step=self._lightning_log_step(trainer),
        )

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer, pl_module
        self._accumulated_runtime_sec = self._runtime_seconds()
        self._fit_start_time = None
        self._epoch_train_start_time = None
        self._epoch_validation_sec = 0.0
        self._validation_start_time = None

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer
        if pl_module.global_rank == 0:
            self._epoch_values = []
            self._epoch_train_start_time = time.monotonic()
            self._epoch_validation_sec = 0.0
            self._validation_start_time = None

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
        del outputs, batch

        if not trainer.loggers:
            return

        device = self._get_cuda_device(pl_module)
        if device is None:
            return

        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated = torch.cuda.memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        local_allocated_pct = 100.0 * allocated / total_memory
        local_peak_reserved_pct = 100.0 * peak_reserved / total_memory
        max_allocated_pct = self._reduce_max(local_allocated_pct, device)
        worst_peak_reserved_pct = self._reduce_max(local_peak_reserved_pct, device)

        if pl_module.global_rank != 0:
            return

        self._epoch_values.append(worst_peak_reserved_pct)

        if trainer.fit_loop._should_accumulate() and trainer.lightning_module.automatic_optimization:
            return

        log_step = self._lightning_log_step(trainer)
        if (log_step + 1) % self.log_every_n_steps == 0:
            self._log_metrics(
                trainer,
                {
                    "System/GPU Memory Allocated (%)": max_allocated_pct,
                    "worst_peak_reserved_pct": worst_peak_reserved_pct,
                    "train/epoch_progress_pct": min(
                        100.0,
                        100.0 * float(batch_idx + 1) / max(float(trainer.num_training_batches), 1.0),
                    ),
                },
                step=log_step,
            )

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if self._fit_start_time is None or trainer.sanity_checking or trainer.global_rank != 0:
            return
        self._validation_start_time = time.monotonic()

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if self._validation_start_time is None or self._fit_start_time is None:
            return
        if trainer.sanity_checking or trainer.global_rank != 0 or not trainer.loggers:
            self._validation_start_time = None
            return

        validation_minutes = (time.monotonic() - self._validation_start_time) / 60.0
        self._validation_start_time = None
        self._epoch_validation_sec += validation_minutes * 60.0
        self._log_metrics(
            trainer,
            {
                "time/validation_minutes": validation_minutes,
            },
            step=self._lightning_log_step(trainer),
        )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if pl_module.global_rank != 0 or not trainer.loggers:
            return

        epoch_metrics: dict[str, float] = {
            "train/epoch_progress_pct": 100.0,
        }
        if self._epoch_train_start_time is not None:
            train_minutes = max(
                time.monotonic() - self._epoch_train_start_time - self._epoch_validation_sec,
                0.0,
            ) / 60.0
            epoch_metrics["time/train_epoch_minutes"] = train_minutes

        if self._epoch_values:
            values = torch.tensor(self._epoch_values, dtype=torch.float32)
            epoch_metrics["worst_peak_reserved_pct_epoch_max"] = float(values.max().item())

        self._log_metrics(
            trainer,
            epoch_metrics,
            step=self._lightning_log_step(trainer, epoch_end=True),
        )

        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is None or trainer.max_epochs is None or trainer.max_epochs <= 0:
            return

        elapsed_training_hours = self._runtime_seconds() / 3600.0
        epoch_progress_pct = min(100.0, 100.0 * (pl_module.current_epoch + 1) / trainer.max_epochs)
        self._progress_points.append([elapsed_training_hours, epoch_progress_pct])
        elapsed_hours = [point[0] for point in self._progress_points]
        epoch_progress = [point[1] for point in self._progress_points]
        wandb_logger.experiment.log(
            {
                "training_progress_vs_runtime": wandb.plot.line_series(
                    xs=elapsed_hours,
                    ys=[epoch_progress],
                    keys=["epoch_progress_pct"],
                    title="Training Progress vs Runtime",
                    xname="elapsed_training_hours",
                    split_table=True,
                )
            }
        )
