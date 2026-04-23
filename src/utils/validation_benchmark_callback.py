"""Validation-loop benchmark callback for sim_agents submission export sweeps.

Measures steady-state seconds-per-validation-batch and rank-max peak
allocated VRAM. Emits a single `[VALBENCH]` line at teardown so the sweep
driver can grep it out.

Intentionally decoupled from wandb / metric logic so the bench runs do not
pay for any of that overhead.
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist
from lightning import Callback, LightningModule, Trainer


class ValidationBenchmarkCallback(Callback):
    """Measure steady-state val batch seconds and peak allocated VRAM."""

    def __init__(
        self,
        warmup_batches: int = 1,
    ) -> None:
        super().__init__()
        self.warmup_batches = warmup_batches
        self._count = 0
        self._timed_batches = 0
        self._timed_start: float | None = None
        self._timed_end: float | None = None

    def on_validation_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        torch.cuda.reset_peak_memory_stats()
        self._count = 0
        self._timed_batches = 0
        self._timed_start = None
        self._timed_end = None

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        torch.cuda.synchronize()
        self._count += 1
        if self._count == self.warmup_batches + 1:
            self._timed_start = time.perf_counter()
            self._timed_batches = 0
        elif self._count > self.warmup_batches + 1:
            self._timed_end = time.perf_counter()
            self._timed_batches += 1

    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        if stage not in ("validate", "fit"):
            return
        if self._timed_start is None or self._timed_end is None:
            return
        elapsed = self._timed_end - self._timed_start
        if elapsed <= 0 or self._timed_batches <= 0:
            return

        per_gpu_batch = trainer.datamodule.val_batch_size
        num_ranks = trainer.world_size
        samples_per_batch = per_gpu_batch * num_ranks
        total_samples_per_s = samples_per_batch * self._timed_batches / elapsed
        batch_time_ms = 1000.0 * elapsed / self._timed_batches
        peak_vram_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_vram_all = torch.tensor([peak_vram_mib], device="cuda")
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(peak_vram_all, op=dist.ReduceOp.MAX)
        peak_vram_max_mib = float(peak_vram_all.item())

        if trainer.is_global_zero:
            print(
                f"[VALBENCH] val_bs={per_gpu_batch} ranks={num_ranks} "
                f"timed_batches={self._timed_batches} "
                f"batch_ms={batch_time_ms:.1f} "
                f"total_samples_s={total_samples_per_s:.2f} "
                f"peak_vram_mib={peak_vram_max_mib:.0f}",
                flush=True,
            )
