from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
from lightning import Callback, Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch import nn


class ModuleTimeProfilerCallback(Callback):
    """Measure coarse SMART module forward/backward CUDA time during training.

    The callback is intentionally opt-in through Hydra. It registers hooks only
    for the profiling run and stops training after ``profile_steps`` measured
    batches. Timings are per-rank; rank zero writes the JSON summary.
    """

    def __init__(
        self,
        warmup_steps: int = 10,
        profile_steps: int = 30,
        output_json: str | None = None,
    ) -> None:
        super().__init__()
        self.warmup_steps = int(warmup_steps)
        self.profile_steps = int(profile_steps)
        self.output_json = output_json
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._active = False
        self._seen_train_batches = 0
        self._profiled_batches = 0
        self._batch_events: dict[tuple[str, str], list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self._step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._totals_ms: dict[tuple[str, str], float] = defaultdict(float)
        self._counts: dict[tuple[str, str], int] = defaultdict(int)
        self._params_by_category: dict[str, int] = {}
        self._device: torch.device | None = None

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

    @staticmethod
    def _param_id_set(modules: Iterable[nn.Module]) -> set[int]:
        ids: set[int] = set()
        for module in modules:
            for param in module.parameters(recurse=True):
                ids.add(id(param))
        return ids

    @staticmethod
    def _param_count_excluding(module: nn.Module, excluded_modules: Iterable[nn.Module]) -> int:
        excluded = ModuleTimeProfilerCallback._param_id_set(excluded_modules)
        seen: set[int] = set()
        total = 0
        for param in module.parameters(recurse=True):
            ident = id(param)
            if ident in seen or ident in excluded:
                continue
            seen.add(ident)
            total += param.numel()
        return int(total)

    def _record_pre(self, category: str, kind: str) -> None:
        if not self._active or self._device is None or self._device.type != "cuda":
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self._batch_events[(category, kind)].append((start, end))

    def _record_post(self, category: str, kind: str) -> None:
        if not self._active or self._device is None or self._device.type != "cuda":
            return
        events = self._batch_events.get((category, kind))
        if not events:
            return
        events[-1][1].record()

    def _add_hooks(self, category: str, modules: Iterable[nn.Module]) -> None:
        for module in modules:
            self._handles.append(
                module.register_forward_pre_hook(
                    lambda _module, _inputs, cat=category: self._record_pre(cat, "forward")
                )
            )
            self._handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, _outputs, cat=category: self._record_post(cat, "forward")
                )
            )
            self._handles.append(
                module.register_full_backward_pre_hook(
                    lambda _module, _grad_outputs, cat=category: self._record_pre(cat, "backward")
                )
            )
            self._handles.append(
                module.register_full_backward_hook(
                    lambda _module, _grad_inputs, _grad_outputs, cat=category: self._record_post(cat, "backward")
                )
            )

    def on_fit_start(self, trainer: Trainer, pl_module: nn.Module) -> None:
        if not torch.cuda.is_available():
            rank_zero_info("[module_profiler] CUDA is unavailable; profiler disabled.")
            return

        self._device = pl_module.device
        token_processor = pl_module.token_processor
        encoder = pl_module.encoder
        map_encoder = encoder.map_encoder
        agent_encoder = encoder.agent_encoder

        map_attn = list(map_encoder.pt2pt_layers)
        map_embedding_modules = [
            map_encoder.type_pt_emb,
            map_encoder.polygon_type_emb,
            map_encoder.light_pl_emb,
            map_encoder.r_pt2pt_emb,
            map_encoder.token_emb,
        ]
        t_attn = list(agent_encoder.t_attn_layers)
        pt2a_attn = list(agent_encoder.pt2a_attn_layers)
        a2a_attn = list(agent_encoder.a2a_attn_layers)
        head = agent_encoder.token_predict_head
        agent_embedding_modules = [
            agent_encoder.type_a_emb,
            agent_encoder.shape_emb,
            agent_encoder.x_a_emb,
            agent_encoder.r_t_emb,
            agent_encoder.r_pt2a_emb,
            agent_encoder.r_a2a_emb,
            agent_encoder.token_emb_veh,
            agent_encoder.token_emb_ped,
            agent_encoder.token_emb_cyc,
            agent_encoder.fusion_emb,
        ]

        self._add_hooks("token_processor", [token_processor])
        self._add_hooks("map_encoder_total", [map_encoder])
        self._add_hooks("map_embedding_modules", map_embedding_modules)
        self._add_hooks("map_pt2pt_attention", map_attn)
        self._add_hooks("agent_encoder_total", [agent_encoder])
        self._add_hooks("agent_embedding_modules", agent_embedding_modules)
        self._add_hooks("agent_temporal_attention", t_attn)
        self._add_hooks("agent_map2agent_attention", pt2a_attn)
        self._add_hooks("agent_agent2agent_attention", a2a_attn)
        self._add_hooks("token_predict_head", [head])
        self._add_hooks("training_loss", [pl_module.training_loss])

        self._params_by_category = {
            "token_processor": self._unique_param_count([token_processor]),
            "map_encoder_total": self._unique_param_count([map_encoder]),
            "map_pt2pt_attention": self._unique_param_count(map_attn),
            "map_embedding_edge_build": self._unique_param_count(map_embedding_modules),
            "agent_encoder_total": self._unique_param_count([agent_encoder]),
            "agent_temporal_attention": self._unique_param_count(t_attn),
            "agent_map2agent_attention": self._unique_param_count(pt2a_attn),
            "agent_agent2agent_attention": self._unique_param_count(a2a_attn),
            "token_predict_head": self._unique_param_count([head]),
            "agent_embedding_edge_build": self._unique_param_count(agent_embedding_modules),
            "training_loss": self._unique_param_count([pl_module.training_loss]),
            "optimizer_ddp_logging_other": 0,
        }
        rank_zero_info(
            "[module_profiler] enabled "
            f"warmup_steps={self.warmup_steps} profile_steps={self.profile_steps}"
        )

    def on_train_batch_start(self, trainer: Trainer, pl_module: nn.Module, batch, batch_idx: int) -> None:
        self._active = (
            self._handles
            and self._seen_train_batches >= self.warmup_steps
            and self._profiled_batches < self.profile_steps
        )
        self._batch_events = defaultdict(list)
        if self._active and self._device is not None and self._device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._step_events.append((start, end))

    def on_train_batch_end(self, trainer: Trainer, pl_module: nn.Module, outputs, batch, batch_idx: int) -> None:
        if self._active and self._device is not None and self._device.type == "cuda":
            self._step_events[-1][1].record()
            torch.cuda.synchronize(self._device)
            for key, pairs in self._batch_events.items():
                for start, end in pairs:
                    self._totals_ms[key] += float(start.elapsed_time(end))
                    self._counts[key] += 1
            self._profiled_batches += 1
            if self._profiled_batches >= self.profile_steps:
                self._dump_summary(trainer)
                trainer.should_stop = True
        self._seen_train_batches += 1
        self._active = False

    def on_fit_end(self, trainer: Trainer, pl_module: nn.Module) -> None:
        if self._profiled_batches and not trainer.should_stop:
            self._dump_summary(trainer)
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _category_ms(self, category: str, kind: str) -> float:
        return float(self._totals_ms.get((category, kind), 0.0))

    def _derived_rows(self, total_step_ms: float) -> list[dict[str, float | int | str]]:
        raw = {
            category: {
                "forward_ms": self._category_ms(category, "forward"),
                "backward_ms": self._category_ms(category, "backward"),
            }
            for category in [
                "token_processor",
                "map_encoder_total",
                "map_embedding_modules",
                "map_pt2pt_attention",
                "agent_encoder_total",
                "agent_embedding_modules",
                "agent_temporal_attention",
                "agent_map2agent_attention",
                "agent_agent2agent_attention",
                "token_predict_head",
                "training_loss",
            ]
        }

        def row(name: str, forward_ms: float, backward_ms: float, params: int) -> dict[str, float | int | str]:
            total_ms = max(0.0, forward_ms) + max(0.0, backward_ms)
            share = total_ms / total_step_ms if total_step_ms > 0 else 0.0
            return {
                "name": name,
                "forward_ms": max(0.0, forward_ms),
                "backward_ms": max(0.0, backward_ms),
                "total_ms": total_ms,
                "time_share": share,
                "params": int(params),
                "share_per_mparam": (share / (params / 1_000_000.0)) if params > 0 else None,
            }

        map_other_fwd = raw["map_encoder_total"]["forward_ms"] - raw["map_pt2pt_attention"]["forward_ms"]
        map_other_bwd = raw["map_embedding_modules"]["backward_ms"]

        agent_children = [
            "agent_temporal_attention",
            "agent_map2agent_attention",
            "agent_agent2agent_attention",
            "token_predict_head",
        ]
        agent_other_fwd = raw["agent_encoder_total"]["forward_ms"] - sum(
            raw[name]["forward_ms"] for name in agent_children
        )
        agent_other_bwd = raw["agent_embedding_modules"]["backward_ms"]

        rows = [
            row(
                "token_processor",
                raw["token_processor"]["forward_ms"],
                raw["token_processor"]["backward_ms"],
                self._params_by_category["token_processor"],
            ),
            row(
                "map_embedding_edge_build",
                map_other_fwd,
                map_other_bwd,
                self._params_by_category["map_embedding_edge_build"],
            ),
            row(
                "map_pt2pt_attention",
                raw["map_pt2pt_attention"]["forward_ms"],
                raw["map_pt2pt_attention"]["backward_ms"],
                self._params_by_category["map_pt2pt_attention"],
            ),
            row(
                "agent_embedding_edge_build",
                agent_other_fwd,
                agent_other_bwd,
                self._params_by_category["agent_embedding_edge_build"],
            ),
            row(
                "agent_temporal_attention",
                raw["agent_temporal_attention"]["forward_ms"],
                raw["agent_temporal_attention"]["backward_ms"],
                self._params_by_category["agent_temporal_attention"],
            ),
            row(
                "agent_map2agent_attention",
                raw["agent_map2agent_attention"]["forward_ms"],
                raw["agent_map2agent_attention"]["backward_ms"],
                self._params_by_category["agent_map2agent_attention"],
            ),
            row(
                "agent_agent2agent_attention",
                raw["agent_agent2agent_attention"]["forward_ms"],
                raw["agent_agent2agent_attention"]["backward_ms"],
                self._params_by_category["agent_agent2agent_attention"],
            ),
            row(
                "token_predict_head",
                raw["token_predict_head"]["forward_ms"],
                raw["token_predict_head"]["backward_ms"],
                self._params_by_category["token_predict_head"],
            ),
            row(
                "training_loss",
                raw["training_loss"]["forward_ms"],
                raw["training_loss"]["backward_ms"],
                self._params_by_category["training_loss"],
            ),
        ]
        accounted = sum(item["total_ms"] for item in rows)
        other = row(
            "optimizer_ddp_logging_other",
            0.0,
            max(0.0, total_step_ms - accounted),
            0,
        )
        rows.append(other)
        return rows

    def _dump_summary(self, trainer: Trainer) -> None:
        if not trainer.is_global_zero:
            return
        if not self._step_events:
            return
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        total_step_ms = sum(start.elapsed_time(end) for start, end in self._step_events)
        rows = self._derived_rows(total_step_ms=total_step_ms)
        payload = {
            "warmup_steps": self.warmup_steps,
            "profile_steps": self._profiled_batches,
            "total_step_ms": total_step_ms,
            "mean_step_ms": total_step_ms / max(1, self._profiled_batches),
            "rows": sorted(rows, key=lambda item: item["total_ms"], reverse=True),
            "params_by_category": self._params_by_category,
            "raw_totals_ms": {
                f"{category}/{kind}": value for (category, kind), value in self._totals_ms.items()
            },
            "raw_counts": {
                f"{category}/{kind}": value for (category, kind), value in self._counts.items()
            },
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        rank_zero_info("[module_profiler] summary:\n" + text)
        if self.output_json:
            output = Path(self.output_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text + "\n", encoding="utf-8")
