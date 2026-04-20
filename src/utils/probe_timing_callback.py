from __future__ import annotations

import time

import torch
from lightning import Callback, LightningModule, Trainer


class ProbeTimingCallback(Callback):
    """Prints per-step wall-clock time and peak memory to stdout for batch-size probing."""

    def __init__(self, skip_warmup_steps: int = 3) -> None:
        self.skip_warmup_steps = skip_warmup_steps
        self._step_start_time: float | None = None
        self._step_times: list[float] = []
        self._peak_reserved_pct: list[float] = []

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        del trainer, batch, batch_idx
        if torch.cuda.is_available() and pl_module.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pl_module.device)
            torch.cuda.synchronize(pl_module.device)
        self._step_start_time = time.monotonic()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        del outputs, batch
        if self._step_start_time is None:
            return
        if torch.cuda.is_available() and pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)
        elapsed = time.monotonic() - self._step_start_time
        self._step_start_time = None

        peak_reserved_pct = 0.0
        peak_reserved_gb = 0.0
        if torch.cuda.is_available() and pl_module.device.type == "cuda":
            device = pl_module.device
            total_memory = torch.cuda.get_device_properties(device).total_memory
            peak_reserved = torch.cuda.max_memory_reserved(device)
            peak_reserved_pct = 100.0 * peak_reserved / total_memory
            peak_reserved_gb = peak_reserved / (1024 ** 3)

        if batch_idx >= self.skip_warmup_steps:
            self._step_times.append(elapsed)
            self._peak_reserved_pct.append(peak_reserved_pct)

        if pl_module.global_rank == 0:
            print(
                f"[PROBE] step={batch_idx} sec_per_step={elapsed:.4f} "
                f"peak_reserved_pct={peak_reserved_pct:.2f} "
                f"peak_reserved_gb={peak_reserved_gb:.2f}",
                flush=True,
            )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer
        if pl_module.global_rank != 0:
            return
        if not self._step_times:
            print("[PROBE_SUMMARY] no measurements collected", flush=True)
            return
        times = torch.tensor(self._step_times, dtype=torch.float64)
        mems = torch.tensor(self._peak_reserved_pct, dtype=torch.float64)
        print(
            "[PROBE_SUMMARY] "
            f"n={len(self._step_times)} "
            f"sec_per_step_mean={float(times.mean()):.4f} "
            f"sec_per_step_median={float(times.median()):.4f} "
            f"sec_per_step_min={float(times.min()):.4f} "
            f"sec_per_step_max={float(times.max()):.4f} "
            f"peak_reserved_pct_max={float(mems.max()):.2f} "
            f"peak_reserved_pct_mean={float(mems.mean()):.2f}",
            flush=True,
        )
