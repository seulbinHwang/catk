from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType
from typing import Any, Callable

import torch
from lightning import Callback, Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch import nn


class SelfForcedProfileCallback(Callback):
    """Opt-in profiler for self-forced DMD train-step phases.

    The callback only wraps methods with timers. It does not change tensors,
    losses, optimizer steps, validation, or checkpointing.
    """

    def __init__(
        self,
        output_dir: str,
        warmup_steps: int = 3,
        active_steps: int = 12,
        profile_all_ranks: bool = True,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.warmup_steps = int(warmup_steps)
        self.active_steps = int(active_steps)
        self.profile_all_ranks = bool(profile_all_ranks)
        self._seen_train_batches = 0
        self._active_batches = 0
        self._active = False
        self._device: torch.device | None = None
        self._events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self._step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._wall_totals: dict[str, float] = defaultdict(float)
        self._wall_counts: dict[str, int] = defaultdict(int)
        self._params: dict[str, int] = {}
        self._wrapped: list[tuple[object, str, object]] = []
        self._phase_stack: list[str] = []

    @staticmethod
    def _unique_param_count(*modules: nn.Module | None) -> int:
        seen: set[int] = set()
        total = 0
        for module in modules:
            if module is None:
                continue
            for param in module.parameters(recurse=True):
                ident = id(param)
                if ident in seen:
                    continue
                seen.add(ident)
                total += param.numel()
        return int(total)

    def _record_pre(self, label: str) -> None:
        if not self._active or self._device is None or self._device.type != "cuda":
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self._events[label].append((start, end))

    def _record_post(self, label: str) -> None:
        if not self._active or self._device is None or self._device.type != "cuda":
            return
        events = self._events.get(label)
        if events:
            events[-1][1].record()

    def _current_phase_label(self, base: str) -> str:
        if self._phase_stack:
            return f"{self._phase_stack[-1]}.{base}"
        return f"self_forced.generator_update.{base}"

    def _wrap_method(
        self,
        obj: object | None,
        method_name: str,
        label: str,
        *,
        params: int = 0,
        phase: str | None = None,
        label_fn: Callable[[object, tuple[object, ...], dict[str, object]], str] | None = None,
    ) -> None:
        if obj is None or not hasattr(obj, method_name):
            return
        original = getattr(obj, method_name)
        self._params.setdefault(label, int(params))

        def wrapped(instance, *args, **kwargs):
            active_label = label_fn(instance, args, kwargs) if label_fn is not None else label
            self._params.setdefault(active_label, int(params))
            wall_start = time.perf_counter() if self._active else None
            if phase is not None:
                self._phase_stack.append(phase)
            self._record_pre(active_label)
            try:
                return original(*args, **kwargs)
            finally:
                self._record_post(active_label)
                if phase is not None:
                    self._phase_stack.pop()
                if wall_start is not None:
                    self._wall_totals[active_label] += (time.perf_counter() - wall_start) * 1000.0
                    self._wall_counts[active_label] += 1

        setattr(obj, method_name, MethodType(wrapped, obj))
        self._wrapped.append((obj, method_name, original))

    def _restore(self) -> None:
        for obj, method_name, original in self._wrapped:
            setattr(obj, method_name, original)
        self._wrapped.clear()

    def on_fit_start(self, trainer: Trainer, pl_module: nn.Module) -> None:
        if not torch.cuda.is_available():
            rank_zero_info("[self_forced_profile] CUDA is unavailable; disabled.")
            return
        if not bool(getattr(pl_module, "self_forced_enabled", False)):
            rank_zero_info("[self_forced_profile] self_forced is disabled; disabled.")
            return
        self._device = pl_module.device
        encoder = getattr(pl_module, "encoder", None)
        teacher = getattr(pl_module, "self_forced_target_teacher", None)
        estimator = getattr(pl_module, "self_forced_generated_estimator", None)

        self._params["self_forced.online_generator"] = self._unique_param_count(encoder)
        self._params["self_forced.target_teacher"] = self._unique_param_count(teacher)
        self._params["self_forced.generated_estimator"] = self._unique_param_count(estimator)

        self._wrap_method(
            pl_module,
            "_training_step_self_forced",
            "self_forced.training_step.total",
            params=self._params["self_forced.online_generator"],
        )
        self._wrap_method(pl_module, "_build_eval_tokenized_inputs", "self_forced.eval_tokenize")
        self._wrap_method(
            pl_module,
            "_run_self_forced_rollout",
            "self_forced.rollout.total",
            params=self._params["self_forced.online_generator"],
            phase="self_forced.rollout",
        )
        self._wrap_method(pl_module, "_pack_self_forced_committed_rollout", "self_forced.pack_committed_control")
        self._wrap_method(pl_module, "_sync_distributed_bool_any", "self_forced.ddp_bool_any_sync")
        self._wrap_method(
            pl_module,
            "_update_generated_path_flow_estimator",
            "self_forced.estimator_update.total",
            params=self._params["self_forced.generated_estimator"],
            phase="self_forced.estimator_update",
        )
        self._wrap_method(
            pl_module,
            "_compute_self_forced_distribution_matching_loss",
            "self_forced.generator_dmd_loss.total",
            phase="self_forced.generator_dmd_loss",
        )
        self._wrap_method(pl_module, "_build_self_forced_active_control_mask", "self_forced.generator_dmd_loss.active_control_mask")
        self._wrap_method(
            pl_module,
            "_compute_self_forced_direction",
            "self_forced.generator_dmd_loss.direction",
            phase="self_forced.generator_dmd_loss.direction",
        )
        self._wrap_method(pl_module, "_sample_flow_state_from_clean", "self_forced.flow_ode_sample")
        self._wrap_method(pl_module, "_sample_self_forced_guidance_flow_state", "self_forced.guidance_flow_ode_sample")
        self._wrap_method(pl_module, "_predict_self_forced_teacher_estimator_clean_paths", "self_forced.predict_teacher_estimator_clean_paths")
        self._wrap_method(pl_module, "_clear_self_forced_auxiliary_gradients", "self_forced.clear_auxiliary_gradients")
        self._wrap_method(pl_module, "_clear_self_forced_generator_gradients", "self_forced.clear_generator_gradients")
        self._wrap_method(pl_module, "_assert_self_forced_generator_update_isolated", "self_forced.assert_generator_isolated")
        self._wrap_method(pl_module, "_assert_self_forced_estimator_update_isolated", "self_forced.assert_estimator_isolated")

        def clean_estimate_label(_instance: object, args: tuple[object, ...], kwargs: dict[str, object]) -> str:
            decoder = kwargs.get("decoder")
            if decoder is None and args:
                decoder = args[0]
            if decoder is teacher:
                return "self_forced.target_teacher.path_flow_clean_estimate"
            if decoder is estimator:
                return "self_forced.generated_estimator.path_flow_clean_estimate"
            if decoder is encoder:
                return "self_forced.online_generator.path_flow_clean_estimate"
            return "self_forced.unknown_decoder.path_flow_clean_estimate"

        self._wrap_method(
            pl_module,
            "_predict_path_flow_clean_estimate",
            "self_forced.path_flow_clean_estimate",
            label_fn=clean_estimate_label,
        )

        def backward_label(_instance: object, _args: tuple[object, ...], _kwargs: dict[str, object]) -> str:
            return self._current_phase_label("backward")

        def opt_step_label(_instance: object, _args: tuple[object, ...], _kwargs: dict[str, object]) -> str:
            return self._current_phase_label("optimizer_step")

        self._wrap_method(pl_module, "_manual_backward_without_autocast", "self_forced.backward", label_fn=backward_label)
        self._wrap_method(pl_module, "_clip_and_step_with_optional_scaler", "self_forced.optimizer_step", label_fn=opt_step_label)

        if encoder is not None:
            self._wrap_method(encoder, "encode_map", "self_forced.rollout.online_encode_map")
            self._wrap_method(encoder, "prepare_training_rollout_cache", "self_forced.rollout.prepare_training_cache")
            self._wrap_method(encoder, "training_rollout_from_cache", "self_forced.rollout.training_rollout_from_cache")
        for prefix, decoder in (
            ("target_teacher", teacher),
            ("generated_estimator", estimator),
        ):
            if decoder is None:
                continue
            self._wrap_method(decoder, "encode_map", f"self_forced.{prefix}.encode_map")
            self._wrap_method(decoder, "path_flow_velocity_for_anchor0", f"self_forced.{prefix}.path_flow_velocity_for_anchor0")

        rank_zero_info(
            "[self_forced_profile] enabled "
            f"warmup_steps={self.warmup_steps} active_steps={self.active_steps} "
            f"profile_all_ranks={self.profile_all_ranks}"
        )

    def on_train_batch_start(self, trainer: Trainer, pl_module: nn.Module, batch: Any, batch_idx: int) -> None:
        self._active = (
            self._device is not None
            and self._seen_train_batches >= self.warmup_steps
            and self._active_batches < self.active_steps
        )
        self._events = defaultdict(list)
        if self._active and self._device is not None and self._device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._step_events.append((start, end))

    def on_train_batch_end(self, trainer: Trainer, pl_module: nn.Module, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self._active and self._device is not None and self._device.type == "cuda":
            self._step_events[-1][1].record()
            torch.cuda.synchronize(self._device)
            for label, pairs in self._events.items():
                for start, end in pairs:
                    try:
                        elapsed_ms = float(start.elapsed_time(end))
                    except RuntimeError:
                        continue
                    self._totals[label] += elapsed_ms
                    self._counts[label] += 1
            self._active_batches += 1
            if self._active_batches >= self.active_steps:
                self._dump_summary(trainer)
                trainer.should_stop = True
        self._seen_train_batches += 1
        self._active = False

    def on_fit_end(self, trainer: Trainer, pl_module: nn.Module) -> None:
        if self._active_batches and not trainer.should_stop:
            self._dump_summary(trainer)
        self._restore()

    def _row(self, label: str, active_batches: int) -> dict[str, object]:
        cuda_ms = self._totals.get(label, 0.0) / max(1, active_batches)
        wall_ms = self._wall_totals.get(label, 0.0) / max(1, active_batches)
        params = int(self._params.get(label, 0))
        return {
            "label": label,
            "params": params,
            "cuda_ms_per_batch": cuda_ms,
            "wall_ms_per_batch": wall_ms,
            "calls_per_batch": self._counts.get(label, 0) / max(1, active_batches),
            "wall_calls_per_batch": self._wall_counts.get(label, 0) / max(1, active_batches),
            "cuda_ms_per_million_params": (cuda_ms / (params / 1_000_000.0)) if params > 0 else None,
        }

    def _dump_summary(self, trainer: Trainer) -> None:
        if not self.profile_all_ranks and not trainer.is_global_zero:
            return
        if not self._step_events:
            return
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        active_batches = max(1, self._active_batches)
        step_ms = sum(start.elapsed_time(end) for start, end in self._step_events) / active_batches
        labels = sorted(set(self._params) | set(self._totals) | set(self._wall_totals))
        rows = [self._row(label, active_batches) for label in labels]
        payload = {
            "rank": int(trainer.global_rank),
            "world_size": int(trainer.world_size),
            "warmup_steps": self.warmup_steps,
            "active_batches": self._active_batches,
            "step_ms_per_batch": step_ms,
            "rows": sorted(rows, key=lambda item: max(float(item["cuda_ms_per_batch"]), float(item["wall_ms_per_batch"])), reverse=True),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output = self.output_dir / f"self_forced_profile_rank{int(trainer.global_rank):02d}.json"
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if trainer.is_global_zero:
            rank_zero_info(f"[self_forced_profile] wrote {output}")
