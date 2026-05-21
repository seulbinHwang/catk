from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from types import MethodType
from typing import Callable, Iterable

import torch
from lightning import Callback, Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch import nn


class ModuleProfileCallback(Callback):
    """Opt-in CUDA-event profiler for Flow pretrain module timing.

    This callback is intentionally enabled only through profiling Hydra
    overrides. It does not change training tensors, losses, optimizer steps, or
    validation behavior; it only records CUDA elapsed time around coarse module
    regions for a short run.
    """

    def __init__(
        self,
        output_dir: str,
        warmup_steps: int = 5,
        active_steps: int = 20,
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
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._wrapped_methods: list[tuple[object, str, object]] = []
        self._batch_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self._batch_backward_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self._step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._totals: dict[str, float] = defaultdict(float)
        self._backward_totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._backward_counts: dict[str, int] = defaultdict(int)
        self._params: dict[str, int] = {}

    @staticmethod
    def _unique_param_count(modules: Iterable[nn.Module]) -> int:
        seen: set[int] = set()
        total = 0
        for module in modules:
            for param in module.parameters(recurse=True):
                ident = id(param)
                if ident in seen:
                    continue
                seen.add(ident)
                total += param.numel()
        return int(total)

    def _record_pre(self, label: str, *, backward: bool = False) -> None:
        if not self._active or self._device is None or self._device.type != "cuda":
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        target = self._batch_backward_events if backward else self._batch_events
        target[label].append((start, end))

    def _record_post(self, label: str, *, backward: bool = False) -> None:
        if not self._active or self._device is None or self._device.type != "cuda":
            return
        target = self._batch_backward_events if backward else self._batch_events
        events = target.get(label)
        if events:
            events[-1][1].record()

    def _add_module_hooks(self, label: str, modules: Iterable[nn.Module]) -> None:
        module_list = list(modules)
        self._params[label] = self._unique_param_count(module_list)
        for module in module_list:
            self._handles.append(
                module.register_forward_pre_hook(
                    lambda _module, _inputs, name=label: self._record_pre(name)
                )
            )
            self._handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, _outputs, name=label: self._record_post(name)
                )
            )
            self._handles.append(
                module.register_full_backward_pre_hook(
                    lambda _module, _grad_outputs, name=label: self._record_pre(name, backward=True)
                )
            )
            self._handles.append(
                module.register_full_backward_hook(
                    lambda _module, _grad_inputs, _grad_outputs, name=label: self._record_post(name, backward=True)
                )
            )

    def _wrap_method(
        self,
        obj: object,
        method_name: str,
        label: str,
        params: int = 0,
    ) -> None:
        original = getattr(obj, method_name)
        self._params[label] = int(params)

        def wrapped(instance, *args, **kwargs):
            self._record_pre(label)
            try:
                return original(*args, **kwargs)
            finally:
                self._record_post(label)

        setattr(obj, method_name, MethodType(wrapped, obj))
        self._wrapped_methods.append((obj, method_name, original))

    def _restore_wrapped_methods(self) -> None:
        for obj, method_name, original in self._wrapped_methods:
            setattr(obj, method_name, original)
        self._wrapped_methods.clear()

    def on_fit_start(self, trainer: Trainer, pl_module: nn.Module) -> None:
        if not torch.cuda.is_available():
            rank_zero_info("[module_profile] CUDA is unavailable; profiler disabled.")
            return
        self._device = pl_module.device
        encoder = pl_module.encoder
        map_encoder = encoder.map_encoder
        agent_context = encoder.agent_encoder
        flow_decoder = agent_context.flow_decoder

        self._add_module_hooks("token_processor", [pl_module.token_processor])
        self._add_module_hooks("map_encoder.total", [map_encoder])
        self._add_module_hooks("map_encoder.relation_embedding", [map_encoder.r_pt2pt_emb])
        self._add_module_hooks("map_encoder.token_embedding", [map_encoder.token_emb])
        self._add_module_hooks("map_encoder.type_embedding", [map_encoder.type_pt_emb])
        self._add_module_hooks("map_encoder.polygon_type_embedding", [map_encoder.polygon_type_emb])
        self._add_module_hooks("map_encoder.light_embedding", [map_encoder.light_pl_emb])
        self._add_module_hooks("map_encoder.pt2pt_attention.all_layers", list(map_encoder.pt2pt_layers))

        self._add_module_hooks("agent_context.temporal_attention.all_layers", list(agent_context.t_attn_layers))
        self._add_module_hooks("agent_context.map2agent_attention.all_layers", list(agent_context.pt2a_attn_layers))
        self._add_module_hooks("agent_context.agent2agent_attention.all_layers", list(agent_context.a2a_attn_layers))
        self._add_module_hooks("agent_context.agent_token_embedding_modules", [
            agent_context.type_a_emb,
            agent_context.shape_emb,
            agent_context.x_a_emb,
            agent_context.token_emb_veh,
            agent_context.token_emb_ped,
            agent_context.token_emb_cyc,
            agent_context.fusion_emb,
        ])
        self._add_module_hooks("flow_decoder.context_projector", [flow_decoder.context_projector])
        self._add_module_hooks("flow_decoder.noisy_future_encoder", [flow_decoder.noisy_future_encoder])
        self._add_module_hooks("flow_decoder.chunk_mixer.all_layers", list(flow_decoder.chunk_mixers))
        self._add_module_hooks("flow_decoder.step_refiner", [flow_decoder.step_refiner])
        self._add_module_hooks("flow_decoder.velocity_head", [flow_decoder.velocity_head])

        self._wrap_method(
            agent_context,
            "agent_token_embedding",
            "agent_context.agent_token_embedding",
            params=self._params["agent_context.agent_token_embedding_modules"],
        )
        self._wrap_method(
            agent_context,
            "build_temporal_edge",
            "agent_context.temporal_edge_build",
            params=self._unique_param_count([agent_context.r_t_emb]),
        )
        self._wrap_method(
            agent_context,
            "build_map2agent_edge",
            "agent_context.map2agent_edge_build",
            params=self._unique_param_count([agent_context.r_pt2a_emb]),
        )
        self._wrap_method(
            agent_context,
            "build_interaction_edge",
            "agent_context.agent2agent_edge_build",
            params=self._unique_param_count([agent_context.r_a2a_emb]),
        )
        self._wrap_method(agent_context.flow_ode, "sample", "flow_ode.sample", params=0)
        self._wrap_method(agent_context, "_to_pose_metric_norm", "flow_decoder.pose_metric_conversion", params=0)

        rank_zero_info(
            "[module_profile] enabled "
            f"warmup_steps={self.warmup_steps} active_steps={self.active_steps} "
            f"profile_all_ranks={self.profile_all_ranks}"
        )

    def on_train_batch_start(self, trainer: Trainer, pl_module: nn.Module, batch, batch_idx: int) -> None:
        self._active = (
            self._device is not None
            and self._seen_train_batches >= self.warmup_steps
            and self._active_batches < self.active_steps
        )
        self._batch_events = defaultdict(list)
        self._batch_backward_events = defaultdict(list)
        if self._active and self._device is not None and self._device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._step_events.append((start, end))

    def on_train_batch_end(self, trainer: Trainer, pl_module: nn.Module, outputs, batch, batch_idx: int) -> None:
        if self._active and self._device is not None and self._device.type == "cuda":
            self._step_events[-1][1].record()
            torch.cuda.synchronize(self._device)
            for label, pairs in self._batch_events.items():
                for start, end in pairs:
                    try:
                        elapsed_ms = float(start.elapsed_time(end))
                    except RuntimeError:
                        continue
                    self._totals[label] += elapsed_ms
                    self._counts[label] += 1
            for label, pairs in self._batch_backward_events.items():
                for start, end in pairs:
                    try:
                        elapsed_ms = float(start.elapsed_time(end))
                    except RuntimeError:
                        continue
                    self._backward_totals[label] += elapsed_ms
                    self._backward_counts[label] += 1
            self._active_batches += 1
            if self._active_batches >= self.active_steps:
                self._dump_summary(trainer)
                trainer.should_stop = True
        self._seen_train_batches += 1
        self._active = False

    def on_fit_end(self, trainer: Trainer, pl_module: nn.Module) -> None:
        if self._active_batches and not trainer.should_stop:
            self._dump_summary(trainer)
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._restore_wrapped_methods()

    def _row(self, label: str, active_batches: int) -> dict[str, object]:
        forward = self._totals.get(label, 0.0) / max(1, active_batches)
        backward = self._backward_totals.get(label, 0.0) / max(1, active_batches)
        total = forward + backward
        params = int(self._params.get(label, 0))
        return {
            "label": label,
            "params": params,
            "forward_ms_per_batch": forward,
            "backward_hook_ms_per_batch": backward,
            "total_profiled_ms_per_batch": total,
            "forward_calls": self._counts.get(label, 0),
            "backward_hook_calls": self._backward_counts.get(label, 0),
            "ms_per_million_params": (total / (params / 1_000_000.0)) if params > 0 else None,
        }

    def _dump_summary(self, trainer: Trainer) -> None:
        if not self.profile_all_ranks and not trainer.is_global_zero:
            return
        if not self._step_events:
            return
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        active_batches = max(1, self._active_batches)
        total_step_ms = sum(start.elapsed_time(end) for start, end in self._step_events)
        labels = sorted(set(self._params) | set(self._totals) | set(self._backward_totals))
        rows = [self._row(label, active_batches) for label in labels]
        payload = {
            "rank": int(trainer.global_rank),
            "world_size": int(trainer.world_size),
            "warmup_steps": self.warmup_steps,
            "active_batches": self._active_batches,
            "step_ms_per_batch": total_step_ms / active_batches,
            "rows": sorted(rows, key=lambda item: item["total_profiled_ms_per_batch"], reverse=True),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output = self.output_dir / f"module_profile_rank{int(trainer.global_rank):02d}.json"
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if trainer.is_global_zero:
            rank_zero_info(f"[module_profile] wrote {output}")
