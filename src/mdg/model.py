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
        self.action_loss_weight = float(model_config.action_loss_weight)
        self.n_rollout_closed_val = int(model_config.n_rollout_closed_val)
        self.rollout_chunk_size = int(model_config.rollout_chunk_size)
        self.replanning_interval = int(model_config.replanning_interval)
        self.closed_loop_denoising_steps = int(getattr(model_config, "closed_loop_denoising_steps", 1))
        self.validation_closed_seed = int(model_config.validation_closed_seed)
        self.val_closed_loop = bool(model_config.val_closed_loop)
        self.n_batch_sim_agents_metric = int(model_config.n_batch_sim_agents_metric)
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

    def transfer_batch_to_device(self, batch: Dict[str, Any], device: torch.device, dataloader_idx: int) -> Dict[str, Any]:
        out = {}
        for key, value in batch.items():
            out[key] = value.to(device) if isinstance(value, Tensor) else value
        return out

    def _alpha_schedule(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.linspace(0.99, 0.01, self.num_noise_levels, device=device, dtype=dtype)

    def _sample_mask_levels(self, batch: Dict[str, Tensor]) -> Tensor:
        valid = batch["agent_valid"]
        bsz, num_agents = valid.shape
        action_steps = self.backbone.action_steps
        device = valid.device
        max_level = self.num_noise_levels - 1
        deltas = torch.linspace(0.0, 1.0, bsz, device=device)
        if bsz > 1:
            deltas = deltas[torch.randperm(bsz, device=device)]
        mask = torch.zeros((bsz, num_agents, action_steps), dtype=torch.long, device=device)
        for batch_index in range(bsz):
            delta = float(deltas[batch_index].item())
            if torch.rand((), device=device) < 0.5:
                full_count = int(round(delta * action_steps))
                if full_count > 0:
                    mask[batch_index, :, action_steps - full_count :] = max_level
                remaining = action_steps - full_count
                if remaining > 0:
                    progressive_max = torch.linspace(0, max_level - 1, remaining, device=device).round().long()
                    random_levels = torch.floor(
                        torch.rand(remaining, device=device) * (progressive_max + 1).clamp_min(1)
                    ).long()
                    random_levels = torch.cummax(random_levels, dim=0).values
                    mask[batch_index, :, :remaining] = random_levels.view(1, remaining)
            else:
                valid_indices = torch.where(valid[batch_index])[0]
                num_full = int(round(delta * int(valid_indices.numel())))
                if num_full > 0:
                    perm = valid_indices[torch.randperm(int(valid_indices.numel()), device=device)]
                    mask[batch_index, perm[:num_full], :] = max_level
                    rest = perm[num_full:]
                else:
                    rest = valid_indices
                if rest.numel() > 0 and max_level > 0:
                    low_levels = torch.randint(0, max_level, (int(rest.numel()),), device=device)
                    mask[batch_index, rest, :] = low_levels.view(-1, 1)
            mask[batch_index, ~valid[batch_index], :] = 0
        return mask

    def _apply_noise(
        self,
        clean_action: Tensor,
        mask_level: Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        alpha = self._alpha_schedule(clean_action.device, clean_action.dtype)[mask_level].unsqueeze(-1)
        noise = torch.randn(
            clean_action.shape,
            device=clean_action.device,
            dtype=clean_action.dtype,
            generator=generator,
        )
        return torch.sqrt(alpha) * clean_action + torch.sqrt(1.0 - alpha) * noise

    def _closed_loop_mask_schedule(self, device: torch.device) -> Tensor:
        max_level = self.num_noise_levels - 1
        if self.closed_loop_denoising_steps == 1:
            return torch.tensor([max_level], dtype=torch.long, device=device)
        schedule = torch.linspace(
            max_level,
            0,
            self.closed_loop_denoising_steps,
            device=device,
        ).round().long()
        schedule[0] = max_level
        schedule[-1] = 0
        return schedule

    def _future_valid(self, batch: Dict[str, Tensor]) -> Tensor:
        return (
            batch["agent_valid"].unsqueeze(-1)
            & batch["agent_valid_mask"][:, :, self.num_historical_steps : self.num_historical_steps + self.num_future_steps]
        )

    def _trajectory_state_loss(
        self,
        pred_pos: Tensor,
        pred_heading: Tensor,
        pred_speed: Tensor,
        batch: Dict[str, Tensor],
    ) -> Tensor:
        future_valid = self._future_valid(batch)
        target_pos = batch["agent_position"][:, :, self.num_historical_steps :, :2]
        target_heading = batch["agent_heading"][:, :, self.num_historical_steps :]
        target_speed = torch.linalg.norm(
            batch["agent_velocity"][:, :, self.num_historical_steps :, :2],
            dim=-1,
        )
        pos_loss = F.mse_loss(pred_pos, target_pos, reduction="none").sum(dim=-1)
        pred_heading_vec = torch.stack((torch.cos(pred_heading), torch.sin(pred_heading)), dim=-1)
        target_heading_vec = torch.stack((torch.cos(target_heading), torch.sin(target_heading)), dim=-1)
        heading_loss = F.mse_loss(pred_heading_vec, target_heading_vec, reduction="none").sum(dim=-1)
        speed_loss = F.mse_loss(pred_speed, target_speed, reduction="none")
        loss = pos_loss + heading_loss + speed_loss
        return (loss * future_valid.to(dtype=loss.dtype)).sum() / future_valid.sum().clamp_min(1).to(dtype=loss.dtype)

    def _auxiliary_loss(self, aux: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        future_pos = batch["agent_position"][:, :, self.num_historical_steps :, :2]
        future_heading = batch["agent_heading"][:, :, self.num_historical_steps :]
        current_pos = batch["agent_position"][:, :, self.num_historical_steps - 1, :2]
        current_heading = batch["agent_heading"][:, :, self.num_historical_steps - 1]
        local_pos = global_to_local_xy(future_pos, current_pos, current_heading)
        local_heading = wrap_angle(future_heading - current_heading.unsqueeze(-1))
        target = torch.cat((local_pos, local_heading.unsqueeze(-1)), dim=-1)
        valid = self._future_valid(batch)
        per_mode = F.smooth_l1_loss(
            aux,
            target.unsqueeze(2).expand_as(aux),
            reduction="none",
        ).sum(dim=-1)
        per_mode = (per_mode * valid.unsqueeze(2).to(dtype=per_mode.dtype)).sum(dim=-1)
        denom = valid.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=per_mode.dtype)
        per_mode = per_mode / denom
        best = per_mode.min(dim=-1).values
        agent_valid = batch["agent_valid"]
        return (best * agent_valid.to(dtype=best.dtype)).sum() / agent_valid.sum().clamp_min(1).to(dtype=best.dtype)

    def _training_forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        clean_action = self.backbone.clean_actions_from_batch(batch)
        mask_level = self._sample_mask_levels(batch)
        noised_action = self._apply_noise(clean_action, mask_level)
        pred_action, pred_pos, pred_heading, pred_speed, scene, aux = self.backbone.denoise_actions(
            batch,
            noised_action,
            mask_level,
        )
        if aux is None:
            raise RuntimeError("Auxiliary predictor output is required during training.")
        state_loss = self._trajectory_state_loss(pred_pos, pred_heading, pred_speed, batch)
        if self.action_loss_weight > 0.0:
            future_valid = self._future_valid(batch)
            action_valid = future_valid.reshape(
                *future_valid.shape[:2],
                self.backbone.action_steps,
                self.backbone.action_chunk,
            ).all(dim=-1)
            action_loss = F.smooth_l1_loss(pred_action, clean_action, reduction="none").sum(dim=-1)
            action_loss = (action_loss * action_valid.to(dtype=action_loss.dtype)).sum() / action_valid.sum().clamp_min(1).to(dtype=action_loss.dtype)
        else:
            action_loss = state_loss.new_zeros(())
        aux_loss = self._auxiliary_loss(aux, batch)
        total = (
            self.denoising_loss_weight * state_loss
            + self.action_loss_weight * action_loss
            + self.aux_loss_weight * aux_loss
        )
        return {
            "loss": total,
            "state_loss": state_loss.detach(),
            "action_loss": action_loss.detach(),
            "aux_loss": aux_loss.detach(),
        }

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        out = self._training_forward(batch)
        self.log("train/loss", out["loss"], on_step=True, prog_bar=True, batch_size=1)
        self.log("train/state_loss", out["state_loss"], on_step=True, batch_size=1)
        self.log("train/action_loss", out["action_loss"], on_step=True, batch_size=1)
        self.log("train/aux_loss", out["aux_loss"], on_step=True, batch_size=1)
        return out["loss"]

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        out = self._training_forward(batch)
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
        mask_schedule = [int(level) for level in self._closed_loop_mask_schedule(torch.device("cpu")).tolist()]
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

            for segment_start in range(0, self.num_future_steps, self.replanning_interval):
                seeds = []
                for rollout_index in range(start, start + chunk_rollouts):
                    for scenario_id in batch["scenario_id"]:
                        seeds.append(self._make_rollout_seed(str(scenario_id), rollout_index + segment_start * 997))
                generator = torch.Generator(device=batch["agent_position"].device)
                generator.manual_seed(int(sum(seeds) % (2**63 - 1)))
                noised_action, mask = self.backbone.full_noise_sample(rollout_batch, generator=generator)
                scene = self.backbone.scene_encoder(rollout_batch)
                pred_pos: Tensor | None = None
                pred_heading: Tensor | None = None
                pred_speed: Tensor | None = None
                for step_index, noise_level in enumerate(mask_schedule):
                    if step_index > 0:
                        mask = torch.full_like(mask, noise_level)
                    pred_action, pred_pos, pred_heading, pred_speed, _, _ = self.backbone.denoise_actions(
                        rollout_batch,
                        noised_action,
                        mask,
                        scene=scene,
                        compute_aux=False,
                    )
                    if step_index + 1 < len(mask_schedule):
                        next_level = mask_schedule[step_index + 1]
                        next_mask = torch.full_like(mask, next_level)
                        noised_action = self._apply_noise(pred_action, next_mask, generator=generator)
                        mask = next_mask
                if pred_pos is None or pred_heading is None or pred_speed is None:
                    raise RuntimeError("Closed-loop denoising produced no prediction.")
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
