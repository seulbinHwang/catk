from __future__ import annotations

import gc
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import hydra
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor
from waymo_open_dataset.utils.sim_agents import submission_specs

from src.mdg.geometry import global_to_local_xy, wrap_angle
from src.mdg.modules import MDGBackbone
from src.smart.metrics import (
    SimAgentsMetrics,
    SimAgentsSubmission,
    WOSACDistributionMetrics,
    log_and_reset_wosac_distribution_metric,
    minADE,
)
from src.smart.metrics.wosac_distribution_metrics import update_wosac_distribution_metric_from_batch
from src.utils.sim_agents_utils import get_scenario_id_int_tensor, get_scenario_rollouts
from src.utils.vis_waymo import VisWaymo


class MDG(LightningModule):
    @staticmethod
    def _required_sim_agents_rollout_count() -> int:
        config = submission_specs.get_submission_config(submission_specs.ChallengeType.SIM_AGENTS)
        return int(config.n_rollouts)

    @staticmethod
    def _check_sim_agents_submission_rollout_count(is_active: bool, n_rollout_closed_val: int) -> None:
        if not is_active:
            return
        expected = MDG._required_sim_agents_rollout_count()
        if int(n_rollout_closed_val) != expected:
            raise ValueError(
                f"Sim Agents submission requires n_rollout_closed_val={expected}, "
                f"got {n_rollout_closed_val}."
            )

    def __init__(self, model_config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.num_historical_steps = int(model_config.backbone.history_steps)
        self.num_future_steps = int(model_config.backbone.future_steps)
        self.num_noise_levels = int(model_config.backbone.num_noise_levels)
        self.lr = float(model_config.lr)
        self.weight_decay = float(model_config.weight_decay)
        self.lr_warmup_steps = int(model_config.lr_warmup_steps)
        self.lr_decay_step = int(model_config.lr_decay_step)
        self.lr_decay_factor = float(model_config.lr_decay_factor)
        self.denoising_loss_weight = float(model_config.denoising_loss_weight)
        self.aux_loss_weight = float(model_config.aux_loss_weight)
        self.kinematic_chunk_filter = bool(getattr(model_config, "kinematic_chunk_filter", True))
        self.kinematic_max_step_displacement_m = float(
            getattr(model_config, "kinematic_max_step_displacement_m", 5.0)
        )
        self.n_rollout_closed_val = int(model_config.n_rollout_closed_val)
        self.rollout_chunk_size = int(model_config.rollout_chunk_size)
        self.replanning_interval = int(model_config.replanning_interval)
        self.closed_loop_denoising_steps = int(getattr(model_config, "closed_loop_denoising_steps", 5))
        self.closed_loop_denoising_schedule = str(
            getattr(model_config, "closed_loop_denoising_schedule", "temporal")
        ).lower()
        self.closed_loop_reuse_actions = bool(getattr(model_config, "closed_loop_reuse_actions", False))
        self.closed_loop_reuse_alpha = tuple(
            float(value)
            for value in getattr(model_config, "closed_loop_reuse_alpha", (0.70, 0.60, 0.50, 0.01))
        )
        self.validation_closed_seed = int(model_config.validation_closed_seed)
        self.val_closed_loop = bool(model_config.val_closed_loop)
        self.n_batch_sim_agents_metric = int(model_config.n_batch_sim_agents_metric)
        self.scorer_scene_num = getattr(model_config, "scorer_scene_num", None)
        self._scorer_scene_num_last_key: tuple[int, int, int] | None = None
        self._scorer_val_limit_last_key: tuple[int, int | float, int] | None = None
        self._fit_time_original_limit_val_batches: int | float | None = None
        self._fit_time_checkpoint_only_validation_enabled = False
        self.n_vis_batch = int(model_config.n_vis_batch)
        self.n_vis_scenario = int(model_config.n_vis_scenario)
        self.n_vis_rollout = int(model_config.n_vis_rollout)
        self.delete_local_videos_after_wandb_upload = bool(
            getattr(model_config, "delete_local_videos_after_wandb_upload", True)
        )

        self.backbone = MDGBackbone(**model_config.backbone)
        self.sim_agents_metrics = SimAgentsMetrics("val_closed")
        self.sim_agents_submission = SimAgentsSubmission(**model_config.sim_agents_submission)
        self._check_sim_agents_submission_rollout_count(
            bool(self.sim_agents_submission.is_active),
            self.n_rollout_closed_val,
        )
        self.minADE = minADE()
        type_scale = getattr(model_config, "wosac_distribution_type_scale", None)
        cpd_reference = getattr(model_config, "wosac_cpd_reference", None)
        self.wosac_distribution_metrics = WOSACDistributionMetrics(
            "val_closed",
            cpd_reference=cpd_reference,
            type_scale=type_scale,
        )
        self.test_wosac_distribution_metrics = WOSACDistributionMetrics(
            "test",
            cpd_reference=cpd_reference,
            type_scale=type_scale,
        )
        self.closed_loop_metric_name = "val_closed/sim_agents_2025/realism_meta_metric"
        self.val_closed_minade_name = (
            "val_closed/sim_agents_2025/minADE_best_of_n_rollout_closed_val"
        )
        try:
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        except ValueError:
            output_dir = "."
        self.video_dir = Path(output_dir) / "videos"
        if self.closed_loop_denoising_steps < 1:
            raise ValueError(
                "model_config.closed_loop_denoising_steps must be >= 1, "
                f"got {self.closed_loop_denoising_steps}."
            )
        if self.closed_loop_denoising_schedule not in {"global", "temporal"}:
            raise ValueError(
                "model_config.closed_loop_denoising_schedule must be 'global' or 'temporal', "
                f"got {self.closed_loop_denoising_schedule!r}."
            )
        if self.closed_loop_reuse_actions:
            if len(self.closed_loop_reuse_alpha) != 4:
                raise ValueError(
                    "model_config.closed_loop_reuse_alpha must contain four alpha values "
                    "for near/mid/far/tail action chunks."
                )
            for value in self.closed_loop_reuse_alpha:
                if not (0.0 < value <= 1.0):
                    raise ValueError(
                        "model_config.closed_loop_reuse_alpha values must be in (0, 1], "
                        f"got {self.closed_loop_reuse_alpha}."
                    )
            if self.replanning_interval % self.backbone.action_chunk != 0:
                raise ValueError(
                    "closed-loop action reuse requires replanning_interval to be divisible by "
                    f"action_chunk, got replanning_interval={self.replanning_interval}, "
                    f"action_chunk={self.backbone.action_chunk}."
                )
        if self.kinematic_chunk_filter and self.kinematic_max_step_displacement_m <= 0.0:
            raise ValueError(
                "model_config.kinematic_max_step_displacement_m must be positive when "
                f"kinematic_chunk_filter is enabled, got {self.kinematic_max_step_displacement_m}."
            )

    def _should_enable_fit_time_checkpoint_only_validation(self) -> bool:
        return (
            self.val_closed_loop
            and not self.sim_agents_submission.is_active
            and int(self.n_batch_sim_agents_metric) > 0
        )

    def _resolve_val_batch_size(self) -> int | None:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return None
        val_batch_size = getattr(datamodule, "val_batch_size", None)
        if not isinstance(val_batch_size, int) or val_batch_size <= 0:
            return None
        return int(val_batch_size)

    def _apply_scorer_scene_num_overrides(self) -> None:
        scorer_scene_num = self.scorer_scene_num
        if scorer_scene_num is None:
            return
        try:
            scorer_scene_num = int(scorer_scene_num)
        except (TypeError, ValueError):
            return
        if scorer_scene_num <= 0:
            return

        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        world_size = int(getattr(trainer, "world_size", 1) or 1)
        if world_size <= 0:
            world_size = 1
        val_batch_size = self._resolve_val_batch_size()
        if val_batch_size is None:
            return

        per_rank_scenes = math.ceil(scorer_scene_num / world_size)
        self.n_batch_sim_agents_metric = max(1, math.ceil(per_rank_scenes / val_batch_size))

        current_key = (int(scorer_scene_num), int(world_size), int(val_batch_size))
        if self._scorer_scene_num_last_key == current_key:
            return
        self._scorer_scene_num_last_key = current_key
        if getattr(trainer, "is_global_zero", True):
            print(
                "[scorer_scene_num] Fast WOSAC sim_agents_2025 scorer batch count set to "
                f"n_batch_sim_agents_metric={self.n_batch_sim_agents_metric} "
                f"(requested_scenes={scorer_scene_num}, world_size={world_size}, "
                f"val_batch_size={val_batch_size}).",
                flush=True,
            )

    def _estimate_val_batches_per_rank(self) -> int | None:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return None
        val_dataset = getattr(datamodule, "val_dataset", None)
        if val_dataset is None:
            return None
        try:
            dataset_len = int(len(val_dataset))
        except (TypeError, ValueError):
            return None
        if dataset_len <= 0:
            return None
        val_batch_size = self._resolve_val_batch_size()
        if val_batch_size is None:
            return None

        world_size = int(getattr(trainer, "world_size", 1) or 1)
        if world_size <= 0:
            world_size = 1
        global_rank = int(getattr(trainer, "global_rank", 0) or 0)
        shard_size, remainder = divmod(dataset_len, world_size)
        rank_samples = shard_size + int(global_rank < remainder)
        return max(1, math.ceil(rank_samples / val_batch_size))

    def _ensure_validation_limit_reaches_scorer_batches(self) -> None:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        target_batches = int(self.n_batch_sim_agents_metric)
        if target_batches <= 0:
            return

        limit_val_batches = getattr(trainer, "limit_val_batches", None)
        if isinstance(limit_val_batches, bool) or limit_val_batches is None:
            return

        resolved_batches: int | None = None
        if isinstance(limit_val_batches, int):
            if limit_val_batches <= 0:
                return
            resolved_batches = int(limit_val_batches)
        elif isinstance(limit_val_batches, float):
            if limit_val_batches <= 0.0 or limit_val_batches >= 1.0:
                return
            total_batches = self._estimate_val_batches_per_rank()
            if total_batches is None:
                return
            resolved_batches = int(total_batches * limit_val_batches)
        else:
            return

        if resolved_batches >= target_batches:
            return

        old_limit = limit_val_batches
        trainer.limit_val_batches = target_batches
        current_key = (target_batches, old_limit, resolved_batches)
        if self._scorer_val_limit_last_key == current_key:
            return
        self._scorer_val_limit_last_key = current_key
        if getattr(trainer, "is_global_zero", True):
            print(
                "[scorer_scene_num] trainer.limit_val_batches increased "
                f"from {old_limit} to {target_batches} for Fast WOSAC scoring "
                f"(resolved_val_batches={resolved_batches}).",
                flush=True,
            )

    def _configure_fast_wosac_validation_scope(self) -> None:
        self._apply_scorer_scene_num_overrides()
        self._ensure_validation_limit_reaches_scorer_batches()

    def _apply_fit_time_validation_batch_limit(self) -> None:
        if not self._should_enable_fit_time_checkpoint_only_validation():
            self._fit_time_checkpoint_only_validation_enabled = False
            return
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        if self._fit_time_original_limit_val_batches is None:
            self._fit_time_original_limit_val_batches = trainer.limit_val_batches
        trainer.limit_val_batches = int(self.n_batch_sim_agents_metric)
        self._fit_time_checkpoint_only_validation_enabled = True

    def _restore_fit_time_validation_batch_limit(self) -> None:
        trainer = getattr(self, "trainer", None)
        if trainer is not None and self._fit_time_original_limit_val_batches is not None:
            trainer.limit_val_batches = self._fit_time_original_limit_val_batches
        self._fit_time_original_limit_val_batches = None
        self._fit_time_checkpoint_only_validation_enabled = False

    def on_fit_start(self) -> None:
        self._configure_fast_wosac_validation_scope()
        self._apply_fit_time_validation_batch_limit()

    def on_validation_start(self) -> None:
        self._configure_fast_wosac_validation_scope()

    def setup(self, stage: str) -> None:
        if stage in {"fit", "validate"}:
            self._configure_fast_wosac_validation_scope()

    def on_fit_end(self) -> None:
        self._restore_fit_time_validation_batch_limit()

    def transfer_batch_to_device(self, batch: Dict[str, Any], device: torch.device, dataloader_idx: int) -> Dict[str, Any]:
        out = {}
        for key, value in batch.items():
            out[key] = value.to(device) if isinstance(value, Tensor) else value
        return out

    def _alpha_schedule(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        noisy = torch.linspace(0.99, 0.01, self.num_noise_levels, device=device, dtype=dtype)
        clean = torch.ones(1, device=device, dtype=dtype)
        return torch.cat((clean, noisy), dim=0)

    def _alpha_from_mask_level(self, mask_level: Tensor, dtype: torch.dtype) -> Tensor:
        alpha_schedule = self._alpha_schedule(mask_level.device, dtype)
        if not torch.is_floating_point(mask_level):
            return alpha_schedule[mask_level]

        max_index = self.num_noise_levels
        level = mask_level.to(dtype=dtype).clamp(0.0, float(max_index))
        lower = torch.floor(level).long()
        upper = torch.ceil(level).long()
        weight = level - lower.to(dtype=dtype)
        lower_alpha = alpha_schedule[lower]
        upper_alpha = alpha_schedule[upper]
        return torch.lerp(lower_alpha, upper_alpha, weight)

    def _mask_level_from_alpha(self, alpha: Tensor) -> Tensor:
        alpha_schedule = self._alpha_schedule(alpha.device, alpha.dtype)
        alpha_clean = alpha_schedule[0]
        alpha_weak = alpha_schedule[1]
        alpha_full = alpha_schedule[-1]
        alpha = torch.clamp(alpha, min=alpha_full, max=alpha_clean)
        clean_span = (alpha_clean - alpha_weak).clamp_min(torch.finfo(alpha.dtype).eps)
        noisy_span = (alpha_weak - alpha_full).clamp_min(torch.finfo(alpha.dtype).eps)
        clean_level = (alpha_clean - alpha) / clean_span
        noisy_level = 1.0 + (alpha_weak - alpha) / noisy_span * float(self.num_noise_levels - 1)
        return torch.where(alpha >= alpha_weak, clean_level, noisy_level).clamp(0.0, float(self.num_noise_levels))

    def _distributed_world_rank(self) -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_world_size()), int(torch.distributed.get_rank())
        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            return int(getattr(trainer, "world_size", 1) or 1), int(getattr(trainer, "global_rank", 0) or 0)
        return 1, 0

    def _masking_deltas(self, batch_size: int, device: torch.device, batch_idx: int | None = None) -> Tensor:
        world_size, rank = self._distributed_world_rank()
        global_batch_size = int(batch_size) * int(world_size)
        if global_batch_size <= 1:
            return torch.zeros(batch_size, device=device)

        step = int(getattr(self, "global_step", 0))
        epoch = int(getattr(self, "current_epoch", 0))
        batch_offset = 0 if batch_idx is None else int(batch_idx)
        seed = (
            1_729
            + 1_000_003 * step
            + 10_007 * epoch
            + 101 * batch_offset
        ) % (2**63 - 1)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        bins = torch.randperm(global_batch_size, device=device, generator=generator)
        local_bins = bins[rank * batch_size : (rank + 1) * batch_size]
        return local_bins.to(dtype=torch.float32) / float(global_batch_size - 1)

    def _sample_mask_levels(self, batch: Dict[str, Tensor], batch_idx: int | None = None) -> Tensor:
        valid = batch["agent_valid"]
        bsz, num_agents = valid.shape
        action_steps = self.backbone.action_steps
        device = valid.device
        max_level = float(self.num_noise_levels)
        deltas = self._masking_deltas(bsz, device=device, batch_idx=batch_idx)
        mask = torch.ones((bsz, num_agents, action_steps), dtype=torch.float32, device=device)
        for batch_index in range(bsz):
            delta = float(deltas[batch_index].item())
            if torch.rand((), device=device) < 0.5:
                full_count = int(round(delta * action_steps))
                if full_count > 0:
                    mask[batch_index, :, action_steps - full_count :] = max_level
                remaining = action_steps - full_count
                if remaining > 0:
                    denom = float(max(remaining - 1, 1))
                    progressive_max = 1.0 + 3.0 * torch.arange(remaining, device=device, dtype=torch.float32) / denom
                    random_levels = 1.0 + torch.rand((num_agents, remaining), device=device) * (
                        progressive_max.view(1, remaining) - 1.0
                    )
                    random_levels = torch.cummax(random_levels, dim=1).values
                    mask[batch_index, :, :remaining] = random_levels
            else:
                valid_indices = torch.where(valid[batch_index])[0]
                num_full = int(round(delta * int(valid_indices.numel())))
                if num_full > 0:
                    perm = valid_indices[torch.randperm(int(valid_indices.numel()), device=device)]
                    mask[batch_index, perm[:num_full], :] = max_level
                    rest = perm[num_full:]
                else:
                    rest = valid_indices
                if rest.numel() > 0:
                    low_levels = 1.0 + torch.rand((int(rest.numel()),), device=device) * 3.0
                    mask[batch_index, rest, :] = low_levels.view(-1, 1)
            mask[batch_index, ~valid[batch_index], :] = 1.0
        return mask

    def _apply_noise(
        self,
        clean_action: Tensor,
        mask_level: Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        alpha = self._alpha_from_mask_level(mask_level, clean_action.dtype).unsqueeze(-1)
        noise = torch.randn(
            clean_action.shape,
            device=clean_action.device,
            dtype=clean_action.dtype,
            generator=generator,
        )
        return torch.sqrt(alpha) * clean_action + torch.sqrt(1.0 - alpha) * noise

    def _closed_loop_mask_schedule(self, device: torch.device, action_steps: int | None = None) -> Tensor:
        max_level = float(self.num_noise_levels)
        if self.closed_loop_denoising_schedule == "global":
            if self.closed_loop_denoising_steps == 1:
                return torch.tensor([max_level], dtype=torch.float32, device=device)
            schedule = torch.linspace(
                max_level,
                1.0,
                self.closed_loop_denoising_steps,
                device=device,
                dtype=torch.float32,
            )
            schedule[0] = max_level
            schedule[-1] = 1.0
            return schedule

        if action_steps is None:
            raise ValueError("action_steps is required for temporal closed-loop denoising schedule.")
        if self.closed_loop_denoising_steps == 1:
            return torch.full((1, int(action_steps)), max_level, dtype=torch.float32, device=device)

        schedule = torch.linspace(
            max_level,
            1.0,
            self.closed_loop_denoising_steps,
            device=device,
            dtype=torch.float32,
        )
        schedule[0] = max_level
        time_band = torch.div(
            torch.arange(int(action_steps), device=device) * int(self.num_noise_levels - 1),
            int(action_steps),
            rounding_mode="floor",
        ).to(dtype=torch.float32)
        return torch.clamp(schedule[:, None] + time_band[None, :], max=max_level)

    @staticmethod
    def _expand_closed_loop_mask(mask_template: Tensor, reference_mask: Tensor) -> Tensor:
        if mask_template.ndim == 0:
            dtype = mask_template.dtype if torch.is_floating_point(mask_template) else reference_mask.dtype
            return torch.full(reference_mask.shape, mask_template.item(), dtype=dtype, device=reference_mask.device)
        if mask_template.ndim == 1:
            dtype = mask_template.dtype if torch.is_floating_point(mask_template) else reference_mask.dtype
            return mask_template.to(device=reference_mask.device, dtype=dtype).view(1, 1, -1).expand(reference_mask.shape)
        raise ValueError(f"Closed-loop mask template must be scalar or [Ta], got shape {tuple(mask_template.shape)}.")

    def _closed_loop_reuse_mask_template(
        self,
        device: torch.device,
        dtype: torch.dtype,
        action_steps: int,
    ) -> Tensor:
        shift_chunks = self.replanning_interval // self.backbone.action_chunk
        shift_chunks = max(1, min(int(shift_chunks), int(action_steps)))
        tail_start = max(0, int(action_steps) - shift_chunks)
        mid_split = (shift_chunks + tail_start) // 2
        alpha_values = torch.tensor(self.closed_loop_reuse_alpha, device=device, dtype=dtype)
        mask_values = self._mask_level_from_alpha(alpha_values)
        template = torch.empty(int(action_steps), device=device, dtype=dtype)
        template[:shift_chunks] = mask_values[0]
        template[shift_chunks:mid_split] = mask_values[1]
        template[mid_split:tail_start] = mask_values[2]
        template[tail_start:] = mask_values[3]
        return template

    def _reuse_shifted_action(self, previous_action: Tensor) -> Tensor:
        shift_chunks = self.replanning_interval // self.backbone.action_chunk
        if shift_chunks <= 0:
            return previous_action
        action_steps = int(previous_action.shape[2])
        shifted = torch.zeros_like(previous_action)
        if shift_chunks < action_steps:
            shifted[:, :, : action_steps - shift_chunks] = previous_action[:, :, shift_chunks:]
        return shifted

    @staticmethod
    def _apply_reuse_mask_schedule(mask_schedule: Tensor, reuse_mask_template: Tensor | None) -> Tensor:
        if reuse_mask_template is None:
            return mask_schedule
        if mask_schedule.ndim == 1:
            mask_schedule = mask_schedule[:, None].expand(-1, reuse_mask_template.shape[0])
        return torch.minimum(mask_schedule, reuse_mask_template.view(1, -1).to(mask_schedule.dtype))

    def _future_valid(self, batch: Dict[str, Tensor]) -> Tensor:
        return (
            batch["agent_valid"].unsqueeze(-1)
            & batch["agent_valid_mask"][:, :, self.num_historical_steps : self.num_historical_steps + self.num_future_steps]
        )

    def _chunk_valid(self, batch: Dict[str, Tensor]) -> Tensor:
        future_valid = self._future_valid(batch)
        return future_valid.reshape(
            *future_valid.shape[:2],
            self.backbone.action_steps,
            self.backbone.action_chunk,
        ).all(dim=-1)

    def _kinematic_chunk_valid(
        self,
        batch: Dict[str, Tensor],
        chunk_valid: Tensor | None = None,
    ) -> Tensor:
        if chunk_valid is None:
            chunk_valid = self._chunk_valid(batch)
        if not self.kinematic_chunk_filter:
            return chunk_valid

        current_pos = batch["agent_position"][:, :, self.num_historical_steps - 1 : self.num_historical_steps, :2]
        future_pos = batch["agent_position"][
            :, :, self.num_historical_steps : self.num_historical_steps + self.num_future_steps, :2
        ]
        traj_pos = torch.cat((current_pos, future_pos), dim=2)

        current_valid = (
            batch["agent_valid"].unsqueeze(-1)
            & batch["agent_valid_mask"][:, :, self.num_historical_steps - 1 : self.num_historical_steps]
        )
        traj_valid = torch.cat((current_valid, self._future_valid(batch)), dim=2)
        step_pair_valid = traj_valid[:, :, 1:] & traj_valid[:, :, :-1]
        step_displacement = torch.linalg.norm(traj_pos[:, :, 1:] - traj_pos[:, :, :-1], dim=-1)
        sane_step = (~step_pair_valid) | (step_displacement <= self.kinematic_max_step_displacement_m)
        sane_chunk = sane_step.reshape(
            *sane_step.shape[:2],
            self.backbone.action_steps,
            self.backbone.action_chunk,
        ).all(dim=-1)
        return chunk_valid & sane_chunk

    def _chunk_state_loss(
        self,
        pred_chunk_state: Tensor,
        clean_chunk_state: Tensor,
        batch: Dict[str, Tensor],
        chunk_valid: Tensor | None = None,
    ) -> Tensor:
        if chunk_valid is None:
            chunk_valid = self._chunk_valid(batch)
        valid_state = chunk_valid.unsqueeze(-1)
        pred_chunk_state = torch.where(valid_state, pred_chunk_state, torch.zeros_like(pred_chunk_state))
        clean_chunk_state = torch.where(valid_state, clean_chunk_state, torch.zeros_like(clean_chunk_state))
        loss = F.mse_loss(pred_chunk_state, clean_chunk_state, reduction="none").sum(dim=-1)
        return (loss * chunk_valid.to(dtype=loss.dtype)).sum() / chunk_valid.sum().clamp_min(1).to(dtype=loss.dtype)

    def _auxiliary_loss(self, aux: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        future_pos = batch["agent_position"][:, :, self.num_historical_steps :, :2]
        future_heading = batch["agent_heading"][:, :, self.num_historical_steps :]
        current_pos = batch["agent_position"][:, :, self.num_historical_steps - 1, :2]
        current_heading = batch["agent_heading"][:, :, self.num_historical_steps - 1]
        local_pos = global_to_local_xy(future_pos, current_pos, current_heading)
        local_heading = wrap_angle(future_heading - current_heading.unsqueeze(-1))
        target = torch.cat((local_pos, local_heading.unsqueeze(-1)), dim=-1)
        valid = self._future_valid(batch)
        target = torch.where(valid.unsqueeze(-1), target, torch.zeros_like(target))
        target_expanded = target.unsqueeze(2).expand_as(aux)
        l2_score = torch.linalg.norm(aux[..., :2] - target_expanded[..., :2], dim=-1)
        l2_score = (l2_score * valid.unsqueeze(2).to(dtype=l2_score.dtype)).sum(dim=-1)
        denom = valid.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=l2_score.dtype)
        best_mode = (l2_score / denom).argmin(dim=-1)

        per_mode = F.smooth_l1_loss(
            aux,
            target_expanded,
            reduction="none",
        ).sum(dim=-1)
        per_mode = (per_mode * valid.unsqueeze(2).to(dtype=per_mode.dtype)).sum(dim=-1)
        per_mode = per_mode / denom.to(dtype=per_mode.dtype)
        best = per_mode.gather(-1, best_mode.unsqueeze(-1)).squeeze(-1)
        agent_valid = batch["agent_valid"]
        return (best * agent_valid.to(dtype=best.dtype)).sum() / agent_valid.sum().clamp_min(1).to(dtype=best.dtype)

    def _training_forward(self, batch: Dict[str, Tensor], batch_idx: int | None = None) -> Dict[str, Tensor]:
        clean_action, clean_chunk_state = self.backbone.clean_actions_and_chunk_state_from_batch(batch)
        raw_chunk_valid = self._chunk_valid(batch)
        chunk_valid = self._kinematic_chunk_valid(batch, raw_chunk_valid)
        valid_action = chunk_valid.unsqueeze(-1)
        clean_action = torch.where(valid_action, clean_action, torch.zeros_like(clean_action))
        clean_chunk_state = torch.where(valid_action, clean_chunk_state, torch.zeros_like(clean_chunk_state))
        mask_level = self._sample_mask_levels(batch, batch_idx=batch_idx)
        noised_action = self._apply_noise(clean_action, mask_level)
        _, _, _, _, pred_chunk_state, scene, aux = self.backbone.denoise_actions(
            batch,
            noised_action,
            mask_level,
            future_valid=chunk_valid,
        )
        if aux is None:
            raise RuntimeError("Auxiliary predictor output is required during training.")
        state_loss = self._chunk_state_loss(pred_chunk_state, clean_chunk_state, batch, chunk_valid=chunk_valid)
        aux_loss = self._auxiliary_loss(aux, batch)
        total = (
            self.denoising_loss_weight * state_loss
            + self.aux_loss_weight * aux_loss
        )
        return {
            "loss": total,
            "state_loss": state_loss.detach(),
            "aux_loss": aux_loss.detach(),
            "valid_chunk_ratio": chunk_valid.to(dtype=state_loss.dtype).mean().detach(),
            "kinematic_invalid_chunk_ratio": (
                ((raw_chunk_valid & ~chunk_valid).sum().to(dtype=state_loss.dtype))
                / raw_chunk_valid.sum().clamp_min(1).to(dtype=state_loss.dtype)
            ).detach(),
        }

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        out = self._training_forward(batch, batch_idx=batch_idx)
        self.log("train/loss", out["loss"], on_step=True, prog_bar=True, batch_size=1)
        self.log("train/state_loss", out["state_loss"], on_step=True, batch_size=1)
        self.log("train/aux_loss", out["aux_loss"], on_step=True, batch_size=1)
        self.log("train/valid_chunk_ratio", out["valid_chunk_ratio"], on_step=True, batch_size=1)
        self.log(
            "train/kinematic_invalid_chunk_ratio",
            out["kinematic_invalid_chunk_ratio"],
            on_step=True,
            batch_size=1,
        )
        return out["loss"]

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        out = self._training_forward(batch, batch_idx=batch_idx)
        self.log("val/loss", out["loss"], on_epoch=True, sync_dist=True, batch_size=1)
        if not self.val_closed_loop:
            return
        pred = self.generate_closed_loop_rollouts(batch, self.n_rollout_closed_val)
        flat = self._flatten_rollouts(batch, pred)
        self._update_distribution_metric(self.wosac_distribution_metrics, batch, flat, include_gt=True)
        self.minADE.update(
            pred=flat["pred_traj"],
            target=flat["agent_position"][:, self.num_historical_steps :, :2],
            target_valid=flat["agent_valid_mask"][:, self.num_historical_steps :],
        )
        scenario_rollouts = None
        if self.sim_agents_submission.is_active:
            self.sim_agents_submission.update(
                scenario_id=batch["scenario_id"],
                agent_id=flat["agent_id"],
                agent_batch=flat["agent_batch"],
                pred_traj=flat["pred_traj"],
                pred_z=flat["pred_z"],
                pred_head=flat["pred_head"],
            )
            scenario_rollouts = self.sim_agents_submission.aggregate_current_batch()
        elif batch_idx < self.n_batch_sim_agents_metric:
            self.sim_agents_metrics.update_from_prediction_tensors(
                scenario_files=batch["tfrecord_path"],
                agent_id=flat["agent_id"],
                agent_batch=flat["agent_batch"],
                pred_traj=flat["pred_traj"],
                pred_z=flat["pred_z"],
                pred_head=flat["pred_head"],
            )
        self._maybe_visualize(batch, flat, scenario_rollouts, batch_idx)

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        pred = self.generate_closed_loop_rollouts(batch, self.n_rollout_closed_val)
        flat = self._flatten_rollouts(batch, pred)
        self._update_distribution_metric(self.test_wosac_distribution_metrics, batch, flat, include_gt=False)
        if self.sim_agents_submission.is_active:
            self.sim_agents_submission.update(
                scenario_id=batch["scenario_id"],
                agent_id=flat["agent_id"],
                agent_batch=flat["agent_batch"],
                pred_traj=flat["pred_traj"],
                pred_z=flat["pred_z"],
                pred_head=flat["pred_head"],
            )
            self.sim_agents_submission.aggregate_current_batch()

    def _make_rollout_seed(self, scenario_id: str, rollout_idx: int) -> int:
        payload = f"{self.validation_closed_seed}:{scenario_id}:{rollout_idx}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, "little", signed=False) & 0x7FFF_FFFF_FFFF_FFFF

    def _clone_batch_for_rollout(self, batch: Dict[str, Tensor], repeat_count: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, Tensor):
                repeat_pattern = (repeat_count,) + (1,) * value.dim()
                out[key] = value.unsqueeze(0).repeat(repeat_pattern).flatten(0, 1).clone()
            else:
                repeated = []
                for _ in range(repeat_count):
                    repeated.extend(value)
                out[key] = repeated
        return out

    @torch.no_grad()
    def generate_closed_loop_rollouts(self, batch: Dict[str, Tensor], num_rollouts: int) -> Dict[str, Tensor]:
        rollout_chunks: List[Dict[str, Tensor]] = []
        for start in range(0, int(num_rollouts), self.rollout_chunk_size):
            chunk_rollouts = min(self.rollout_chunk_size, int(num_rollouts) - start)
            rollout_batch = self._clone_batch_for_rollout(batch, chunk_rollouts)
            bsz = batch["agent_position"].shape[0]
            output_pos = torch.zeros(
                bsz * chunk_rollouts,
                batch["agent_position"].shape[1],
                self.num_future_steps,
                2,
                device=batch["agent_position"].device,
                dtype=batch["agent_position"].dtype,
            )
            output_heading = torch.zeros(
                bsz * chunk_rollouts,
                batch["agent_position"].shape[1],
                self.num_future_steps,
                device=batch["agent_position"].device,
                dtype=batch["agent_position"].dtype,
            )
            output_speed = torch.zeros_like(output_heading)
            previous_pred_action: Tensor | None = None

            for segment_start in range(0, self.num_future_steps, self.replanning_interval):
                seeds = []
                for rollout_index in range(start, start + chunk_rollouts):
                    for scenario_id in batch["scenario_id"]:
                        seeds.append(self._make_rollout_seed(str(scenario_id), rollout_index + segment_start * 997))
                generator = torch.Generator(device=batch["agent_position"].device)
                generator.manual_seed(int(sum(seeds) % (2**63 - 1)))
                noised_action, mask = self.backbone.full_noise_sample(rollout_batch, generator=generator)
                reuse_mask_template: Tensor | None = None
                if self.closed_loop_reuse_actions and previous_pred_action is not None:
                    reuse_clean_action = self._reuse_shifted_action(previous_pred_action)
                    reuse_mask_template = self._closed_loop_reuse_mask_template(
                        device=noised_action.device,
                        dtype=noised_action.dtype,
                        action_steps=int(noised_action.shape[2]),
                    )
                    mask = self._expand_closed_loop_mask(reuse_mask_template, mask)
                    noised_action = self._apply_noise(reuse_clean_action, mask, generator=generator)
                mask_schedule = self._closed_loop_mask_schedule(
                    noised_action.device,
                    action_steps=int(noised_action.shape[2]),
                )
                mask_schedule = self._apply_reuse_mask_schedule(mask_schedule, reuse_mask_template)
                scene = self.backbone.scene_encoder(rollout_batch)
                pred_pos: Tensor | None = None
                pred_heading: Tensor | None = None
                pred_speed: Tensor | None = None
                pred_action: Tensor | None = None
                for step_index, mask_template in enumerate(mask_schedule):
                    mask = self._expand_closed_loop_mask(mask_template, mask)
                    pred_action, pred_pos, pred_heading, pred_speed, _, _, _ = self.backbone.denoise_actions(
                        rollout_batch,
                        noised_action,
                        mask,
                        scene=scene,
                        compute_aux=False,
                    )
                    if step_index + 1 < len(mask_schedule):
                        next_mask = self._expand_closed_loop_mask(mask_schedule[step_index + 1], mask)
                        noised_action = self._apply_noise(pred_action, next_mask, generator=generator)
                        mask = next_mask
                if pred_pos is None or pred_heading is None or pred_speed is None:
                    raise RuntimeError("Closed-loop denoising produced no prediction.")
                if pred_action is None:
                    raise RuntimeError("Closed-loop denoising produced no action prediction.")
                previous_pred_action = pred_action.detach()
                commit = min(self.replanning_interval, self.num_future_steps - segment_start)
                output_pos[:, :, segment_start : segment_start + commit] = pred_pos[:, :, :commit]
                output_heading[:, :, segment_start : segment_start + commit] = pred_heading[:, :, :commit]
                output_speed[:, :, segment_start : segment_start + commit] = pred_speed[:, :, :commit]
                self._append_history(rollout_batch, pred_pos[:, :, :commit], pred_heading[:, :, :commit], pred_speed[:, :, :commit])

            rollout_chunks.append(
                {
                    "pred_pos": output_pos.reshape(chunk_rollouts, bsz, *output_pos.shape[1:]).permute(1, 2, 0, 3, 4),
                    "pred_heading": output_heading.reshape(chunk_rollouts, bsz, *output_heading.shape[1:]).permute(1, 2, 0, 3),
                    "pred_speed": output_speed.reshape(chunk_rollouts, bsz, *output_speed.shape[1:]).permute(1, 2, 0, 3),
                }
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            "pred_pos": torch.cat([chunk["pred_pos"] for chunk in rollout_chunks], dim=2),
            "pred_heading": torch.cat([chunk["pred_heading"] for chunk in rollout_chunks], dim=2),
            "pred_speed": torch.cat([chunk["pred_speed"] for chunk in rollout_chunks], dim=2),
        }

    def _append_history(self, batch: Dict[str, Tensor], pos: Tensor, heading: Tensor, speed: Tensor) -> None:
        commit = int(pos.shape[2])
        current_z = batch["agent_position"][:, :, self.num_historical_steps - 1 : self.num_historical_steps, 2:3]
        pos3 = torch.cat((pos, current_z.expand(-1, -1, commit, -1)), dim=-1)
        velocity = torch.stack((torch.cos(heading) * speed, torch.sin(heading) * speed), dim=-1)
        valid = batch["agent_valid"].unsqueeze(-1).expand(-1, -1, commit)
        batch["agent_position"][:, :, : self.num_historical_steps] = torch.cat(
            (batch["agent_position"][:, :, commit : self.num_historical_steps], pos3),
            dim=2,
        )
        batch["agent_heading"][:, :, : self.num_historical_steps] = torch.cat(
            (batch["agent_heading"][:, :, commit : self.num_historical_steps], heading),
            dim=2,
        )
        batch["agent_velocity"][:, :, : self.num_historical_steps] = torch.cat(
            (batch["agent_velocity"][:, :, commit : self.num_historical_steps], velocity),
            dim=2,
        )
        batch["agent_valid_mask"][:, :, : self.num_historical_steps] = torch.cat(
            (batch["agent_valid_mask"][:, :, commit : self.num_historical_steps], valid),
            dim=2,
        )

    def _flatten_rollouts(self, batch: Dict[str, Tensor], pred: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pred_traj_parts = []
        pred_head_parts = []
        pred_z_parts = []
        id_parts = []
        type_parts = []
        position_parts = []
        valid_parts = []
        batch_parts = []
        for batch_index in range(batch["agent_valid"].shape[0]):
            mask = batch["agent_valid"][batch_index]
            count = int(mask.sum().item())
            if count == 0:
                continue
            pred_traj_parts.append(pred["pred_pos"][batch_index, mask])
            pred_head_parts.append(pred["pred_heading"][batch_index, mask])
            z = batch["agent_position"][batch_index, mask, self.num_historical_steps - 1, 2]
            pred_z_parts.append(z[:, None, None].expand(-1, pred["pred_pos"].shape[2], self.num_future_steps))
            id_parts.append(batch["agent_id"][batch_index, mask])
            type_parts.append(batch["agent_type"][batch_index, mask])
            position_parts.append(batch["agent_position"][batch_index, mask])
            valid_parts.append(batch["agent_valid_mask"][batch_index, mask])
            batch_parts.append(torch.full((count,), batch_index, dtype=torch.long, device=mask.device))
        return {
            "pred_traj": torch.cat(pred_traj_parts, dim=0),
            "pred_head": torch.cat(pred_head_parts, dim=0),
            "pred_z": torch.cat(pred_z_parts, dim=0),
            "agent_id": torch.cat(id_parts, dim=0),
            "agent_type": torch.cat(type_parts, dim=0),
            "agent_position": torch.cat(position_parts, dim=0),
            "agent_valid_mask": torch.cat(valid_parts, dim=0),
            "agent_batch": torch.cat(batch_parts, dim=0),
        }

    def _update_distribution_metric(
        self,
        metric: WOSACDistributionMetrics,
        batch: Dict[str, Tensor],
        flat: Dict[str, Tensor],
        include_gt: bool,
    ) -> None:
        data = {
            "agent": {
                "position": flat["agent_position"],
                "valid_mask": flat["agent_valid_mask"],
                "type": flat["agent_type"],
                "batch": flat["agent_batch"],
            }
        }
        update_wosac_distribution_metric_from_batch(
            metric=metric,
            data=data,
            pred_traj=flat["pred_traj"],
            num_historical_steps=self.num_historical_steps,
            include_gt=include_gt,
        )

    def _get_video_logger(self):
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            for logger in getattr(trainer, "loggers", []) or []:
                if hasattr(logger, "log_video"):
                    return logger
        logger = getattr(self, "logger", None)
        return logger if hasattr(logger, "log_video") else None

    def _maybe_visualize(
        self,
        batch: Dict[str, Any],
        flat: Dict[str, Tensor],
        scenario_rollouts: Any,
        batch_idx: int,
    ) -> None:
        if self.global_rank != 0 or batch_idx >= self.n_vis_batch:
            return
        if not batch.get("tfrecord_path"):
            return
        if scenario_rollouts is None:
            scenario_rollouts = get_scenario_rollouts(
                scenario_id=get_scenario_id_int_tensor(batch["scenario_id"], flat["pred_traj"].device),
                agent_id=flat["agent_id"],
                agent_batch=flat["agent_batch"],
                pred_traj=flat["pred_traj"],
                pred_z=flat["pred_z"],
                pred_head=flat["pred_head"],
            )
        video_logger = self._get_video_logger()
        for scenario_index in range(min(self.n_vis_scenario, len(batch["tfrecord_path"]))):
            if batch["tfrecord_path"][scenario_index] is None:
                continue
            vis = VisWaymo(
                scenario_path=batch["tfrecord_path"][scenario_index],
                save_dir=self.video_dir / f"batch_{batch_idx:02d}-scenario_{scenario_index:02d}",
            )
            vis.save_video_scenario_rollout(scenario_rollouts[scenario_index], self.n_vis_rollout)
            if video_logger is not None:
                for path in vis.video_paths:
                    video_logger.log_video("/".join(path.split("/")[-3:]), [path])

    def on_validation_epoch_end(self) -> None:
        distribution_metrics = log_and_reset_wosac_distribution_metric(self.wosac_distribution_metrics)
        if self.sim_agents_submission.is_active:
            self.sim_agents_submission.save_sub_file()
            return
        epoch_metrics = {}
        if self.val_closed_loop:
            if self.n_batch_sim_agents_metric > 0:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    metric_state = self.sim_agents_metrics.get_state_tensor(device=self.device)
                    torch.distributed.all_reduce(metric_state)
                    epoch_metrics.update(self.sim_agents_metrics.compute_from_state_tensor(metric_state))
                else:
                    epoch_metrics.update(self.sim_agents_metrics.compute())
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                minade_state = torch.stack(
                    [
                        self.minADE.sum.detach().to(device=self.device),
                        self.minADE.count.detach().to(device=self.device),
                    ]
                )
                torch.distributed.all_reduce(minade_state)
                minade_value = minade_state[0] / minade_state[1].clamp_min(1.0)
            else:
                minade_value = self.minADE.sum / self.minADE.count.clamp_min(1.0)
            epoch_metrics[self.val_closed_minade_name] = minade_value
            epoch_metrics.update(distribution_metrics)
            if self.closed_loop_metric_name in epoch_metrics:
                self.log(
                    self.closed_loop_metric_name,
                    epoch_metrics[self.closed_loop_metric_name],
                    on_epoch=True,
                    sync_dist=True,
                )
            if self.global_rank == 0 and self.logger is not None and hasattr(self.logger, "log_metrics"):
                self.logger.log_metrics(epoch_metrics)
            if self.n_batch_sim_agents_metric > 0:
                self.sim_agents_metrics.reset()
            self.minADE.reset()

    def on_test_epoch_end(self) -> None:
        distribution_metrics = log_and_reset_wosac_distribution_metric(self.test_wosac_distribution_metrics)
        if self.global_rank == 0 and distribution_metrics and self.logger is not None and hasattr(self.logger, "log_metrics"):
            self.logger.log_metrics(distribution_metrics)
        if self.sim_agents_submission.is_active:
            self.sim_agents_submission.save_sub_file()

    def _lr_scale(self, step: int) -> float:
        step = int(step)
        if self.lr_warmup_steps > 0 and step < self.lr_warmup_steps:
            return float(step + 1) / float(self.lr_warmup_steps)
        if self.lr_decay_step <= 0:
            return 1.0
        decay_count = max(0, step - self.lr_warmup_steps) // self.lr_decay_step
        return self.lr_decay_factor ** decay_count

    def _set_optimizer_lr(self, optimizer: torch.optim.Optimizer, step: int) -> None:
        lr = self.lr * self._lr_scale(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None) -> None:
        self._set_optimizer_lr(optimizer, int(self.global_step))
        optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr * self._lr_scale(0),
            weight_decay=self.weight_decay,
        )
        return optimizer
