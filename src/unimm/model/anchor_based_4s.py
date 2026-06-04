from __future__ import annotations

import gc
import hashlib
import math
from pathlib import Path
from typing import Dict, Sequence

import hydra
import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torch.distributions import Categorical
from torch.optim.lr_scheduler import LambdaLR
from waymo_open_dataset.utils.sim_agents import submission_specs

from src.smart.metrics import (
    SimAgentsMetrics,
    SimAgentsSubmission,
    WOSACDistributionMetrics,
    log_and_reset_wosac_distribution_metric,
    minADE,
    update_wosac_distribution_metric_from_model,
)
from src.utils.sim_agents_utils import get_scenario_id_int_tensor, get_scenario_rollouts
from src.utils.vis_waymo import VisWaymo
from src.unimm.anchors import (
    AGENT_TYPE_NAMES,
    AnchorSpec,
    gather_anchors_by_type,
    load_anchor_file,
)
from src.unimm.losses import unimm_candidate_set_ce_loss, unimm_top_m_mixture_nll_loss
from src.unimm.modules import UniMMAnchorBasedNetwork
from src.unimm.processor import UniMMProcessor


class UniMMAnchorBased4s(LightningModule):
    """UniMM Anchor-Based-4s.

    This module keeps the repository's WOSAC evaluation/submission utilities but
    replaces SMART next-token prediction with the anchor-based continuous mixture
    model described in arXiv:2501.17015.
    """

    @staticmethod
    def _required_sim_agents_rollout_count() -> int:
        submission_config = submission_specs.get_submission_config(
            submission_specs.ChallengeType.SIM_AGENTS
        )
        return int(submission_config.n_rollouts)

    @staticmethod
    def _check_sim_agents_submission_rollout_count(is_active: bool, n_rollout_closed_val: int) -> None:
        if not is_active:
            return
        expected_rollouts = UniMMAnchorBased4s._required_sim_agents_rollout_count()
        if int(n_rollout_closed_val) != expected_rollouts:
            raise ValueError(
                "Sim Agents submission export requires "
                f"n_rollout_closed_val={expected_rollouts}, got {n_rollout_closed_val}."
            )

    def __init__(self, model_config: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = float(model_config.lr)
        self.weight_decay = float(model_config.weight_decay)
        self.lr_warmup_steps = int(model_config.lr_warmup_steps)
        self.lr_total_steps = int(model_config.lr_total_steps)
        self.num_historical_steps = int(model_config.num_historical_steps)
        self.log_epoch = -1
        self.val_open_loop = bool(model_config.val_open_loop)
        self.val_closed_loop = bool(model_config.val_closed_loop)
        self.n_rollout_closed_val = int(model_config.n_rollout_closed_val)
        self._check_sim_agents_submission_rollout_count(
            is_active=bool(model_config.sim_agents_submission.is_active),
            n_rollout_closed_val=self.n_rollout_closed_val,
        )

        anchors, threshold, _ = load_anchor_file(model_config.anchor_file)
        self.spec = AnchorSpec(
            num_anchors=int(anchors.shape[1]),
            num_future_steps=int(anchors.shape[2]),
            num_prediction_steps=int(model_config.prediction_horizon_steps),
            num_commit_steps=int(model_config.commit_steps),
            num_match_steps=int(model_config.match_steps),
        )
        if self.spec.num_prediction_steps > anchors.shape[2]:
            raise ValueError(
                "prediction_horizon_steps cannot exceed anchor horizon, "
                f"got {self.spec.num_prediction_steps} and {anchors.shape[2]}"
            )
        if self.spec.num_future_steps % self.spec.num_commit_steps != 0:
            raise ValueError(
                "anchor horizon must be divisible by commit_steps for fixed-interval "
                f"closed-loop simulation, got {self.spec.num_future_steps} and "
                f"{self.spec.num_commit_steps}"
            )
        if self.spec.num_prediction_steps < self.spec.num_commit_steps:
            raise ValueError(
                "prediction_horizon_steps must cover at least one committed update, "
                f"got {self.spec.num_prediction_steps} and {self.spec.num_commit_steps}"
            )
        self.register_buffer("anchors_by_type", anchors, persistent=False)
        if threshold is None:
            threshold = torch.full((3,), float("inf"), dtype=torch.float32)
        self.register_buffer("posterior_error_threshold", threshold, persistent=False)

        self.processor = UniMMProcessor(
            prediction_horizon_steps=self.spec.num_prediction_steps,
            commit_steps=self.spec.num_commit_steps,
            match_steps=self.spec.num_match_steps,
            first_context_step=int(model_config.first_context_step),
            last_train_context_step=int(model_config.last_train_context_step),
            anchor_heading_weight=float(model_config.anchor_heading_weight),
            anchor_match_chunk_size=int(model_config.anchor_match_chunk_size),
            positive_tie_break_horizon_steps=getattr(
                model_config,
                "positive_tie_break_horizon_steps",
                None,
            ),
            positive_tie_break_tolerance=float(
                getattr(model_config, "positive_tie_break_tolerance", 0.0)
            ),
            positive_top_m=int(getattr(model_config, "positive_top_m", 8)),
        )
        self.network = UniMMAnchorBasedNetwork(
            hidden_dim=int(model_config.decoder.hidden_dim),
            num_anchors=self.spec.num_anchors,
            num_prediction_steps=self.spec.num_prediction_steps,
            pl2pl_radius=float(model_config.decoder.pl2pl_radius),
            pl2a_radius=float(model_config.decoder.pl2a_radius),
            a2a_radius=float(model_config.decoder.a2a_radius),
            time_span=int(model_config.decoder.time_span),
            num_freq_bands=int(model_config.decoder.num_freq_bands),
            num_map_layers=int(model_config.decoder.num_map_layers),
            num_agent_layers=int(model_config.decoder.num_agent_layers),
            num_heads=int(model_config.decoder.num_heads),
            head_dim=int(model_config.decoder.head_dim),
            dropout=float(model_config.decoder.dropout),
            min_laplace_scale=float(model_config.decoder.min_laplace_scale),
            min_von_mises_concentration=float(model_config.decoder.min_von_mises_concentration),
            max_von_mises_concentration=float(
                getattr(model_config.decoder, "max_von_mises_concentration", 100.0)
            ),
        )

        self.use_closed_loop_training = bool(model_config.use_closed_loop_training)
        self.loss_weights = model_config.loss_weights
        self.mixture_loss_weight = float(getattr(self.loss_weights, "mixture", 1.0))
        self.aux_ce_loss_weight = float(getattr(self.loss_weights, "aux_ce", 0.2))
        self.inference_temperature = float(model_config.inference_temperature)
        self.inference_top_k = int(getattr(model_config, "inference_top_k", 0))
        self.inference_top_p = float(getattr(model_config, "inference_top_p", 1.0))
        if self.inference_top_k < 0:
            raise ValueError(
                f"inference_top_k must be non-negative, got {self.inference_top_k}."
            )
        if not (math.isfinite(self.inference_top_p) and 0.0 < self.inference_top_p <= 1.0):
            raise ValueError(
                "inference_top_p must be finite and in the interval (0, 1], "
                f"got {self.inference_top_p}."
            )
        self.validation_closed_seed = int(model_config.validation_closed_seed)
        self.n_batch_sim_agents_metric = int(model_config.n_batch_sim_agents_metric)
        self.scorer_scene_num = getattr(model_config, "scorer_scene_num", None)
        self._scorer_scene_num_last_key: tuple[int, int, int] | None = None
        self.n_vis_batch = int(model_config.n_vis_batch)
        self.n_vis_scenario = int(model_config.n_vis_scenario)
        self.n_vis_rollout = int(model_config.n_vis_rollout)
        self.delete_local_videos_after_wandb_upload = bool(
            getattr(model_config, "delete_local_videos_after_wandb_upload", True)
        )

        self.minADE = minADE()
        self.sim_agents_metrics = SimAgentsMetrics("val_closed")
        self.sim_agents_submission = SimAgentsSubmission(**model_config.sim_agents_submission)
        self.wosac_distribution_metrics = WOSACDistributionMetrics(
            "val_closed",
            cpd_reference=getattr(model_config, "wosac_cpd_reference", None),
            type_scale=getattr(model_config, "wosac_distribution_type_scale", None),
        )
        self.test_wosac_distribution_metrics = WOSACDistributionMetrics(
            "test",
            cpd_reference=getattr(model_config, "wosac_cpd_reference", None),
            type_scale=getattr(model_config, "wosac_distribution_type_scale", None),
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
        if self.sim_agents_submission.is_active:
            return

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
        self.n_batch_sim_agents_metric = max(
            1,
            math.ceil(per_rank_scenes / val_batch_size),
        )

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

    def on_fit_start(self) -> None:
        self._apply_scorer_scene_num_overrides()

    def on_validation_start(self) -> None:
        self._apply_scorer_scene_num_overrides()

    @staticmethod
    def _repeat_tensor_on_first_dim(tensor: torch.Tensor, repeat_count: int) -> torch.Tensor:
        if repeat_count == 1:
            return tensor
        repeat_pattern = (repeat_count,) + (1,) * tensor.dim()
        return tensor.unsqueeze(0).repeat(repeat_pattern).flatten(0, 1).contiguous()

    def _context_embeddings(
        self,
        tokenized_map: Dict[str, torch.Tensor],
        tokenized_agent: Dict[str, torch.Tensor],
        context_indices: torch.Tensor,
    ) -> torch.Tensor:
        embedding_seq = self.network.encode(tokenized_map, tokenized_agent)
        return embedding_seq[:, context_indices]

    def _candidate_prediction_anchors(
        self,
        agent_type: torch.Tensor,
        z_candidates: torch.Tensor,
    ) -> torch.Tensor:
        n_agent, n_context, n_candidate = z_candidates.shape
        row_type = agent_type[:, None, None].expand(-1, n_context, n_candidate)
        selected = gather_anchors_by_type(
            self.anchors_by_type[:, :, : self.spec.num_prediction_steps],
            row_type,
            z_candidates,
        )
        return selected.view(n_agent, n_context, n_candidate, self.spec.num_prediction_steps, 3)

    def _forward_loss(self, data, use_closed_loop: bool) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch = self.processor.build_training_batch(
            data=data,
            anchors_by_type=self.anchors_by_type,
            posterior_threshold=self.posterior_error_threshold,
            use_closed_loop=use_closed_loop,
        )
        context_embedding = self._context_embeddings(
            batch.tokenized_map,
            batch.tokenized_agent,
            batch.context_indices,
        )
        candidate_anchor = self._candidate_prediction_anchors(
            batch.tokenized_agent["type"],
            batch.z_candidates,
        )
        pred = self.network.motion_decoder(context_embedding, candidate_anchor)

        train_mask = data["agent"]["train_mask"] if "train_mask" in data["agent"] else None
        target_valid = batch.target_valid
        if train_mask is not None:
            target_valid = target_valid & train_mask[:, None, None]

        mixture_loss, candidate_nll = unimm_top_m_mixture_nll_loss(
            pred,
            pred["logits"],
            batch.z_candidates,
            batch.target_local,
            target_valid,
        )
        aux_ce_loss = unimm_candidate_set_ce_loss(
            pred["logits"],
            batch.z_candidates,
            target_valid,
            match_steps=self.spec.num_match_steps,
        )
        total_loss = self.mixture_loss_weight * mixture_loss + self.aux_ce_loss_weight * aux_ce_loss
        if not torch.isfinite(total_loss):
            raise RuntimeError(
                "UniMM training loss became non-finite "
                f"(loss={float(total_loss.detach().cpu())}, "
                f"mixture={float(mixture_loss.detach().cpu())}, "
                f"aux_ce={float(aux_ce_loss.detach().cpu())})."
            )
        z_star_valid = target_valid[..., : self.spec.num_match_steps].any(dim=-1)
        aux_mixture_ratio = aux_ce_loss.detach() / mixture_loss.detach().abs().clamp_min(1e-6)
        candidate_error_valid = batch.z_candidate_error[z_star_valid]
        candidate_nll_valid = candidate_nll[z_star_valid]
        logs = {
            "loss": total_loss,
            "loss_mixture": mixture_loss.detach(),
            "loss_aux_ce": aux_ce_loss.detach(),
            "aux_mixture_ratio": aux_mixture_ratio,
            "z_star_error": batch.z_star_error[z_star_valid].mean().detach()
            if bool(z_star_valid.any())
            else batch.z_star_error.sum().detach() * 0.0,
            "top_m_error": candidate_error_valid.mean().detach()
            if bool(z_star_valid.any())
            else batch.z_candidate_error.sum().detach() * 0.0,
            "top_m_nll": candidate_nll_valid.mean().detach()
            if bool(z_star_valid.any())
            else candidate_nll.sum().detach() * 0.0,
        }
        posterior_stats = batch.posterior_stats
        for stat_key in (
            "accept_rate",
            "error_mean",
            "error_p50",
            "error_p90",
            "error_p95",
            "error_over_threshold",
        ):
            if stat_key in posterior_stats:
                logs[f"posterior_{stat_key}"] = posterior_stats[stat_key].detach()

        type_rates = posterior_stats.get("accept_rate_by_type")
        if type_rates is not None:
            for type_idx, type_name in enumerate(AGENT_TYPE_NAMES):
                if type_idx < int(type_rates.numel()):
                    logs[f"posterior_accept_rate_{type_name}"] = type_rates[type_idx].detach()

        context_rates = posterior_stats.get("accept_rate_by_context")
        context_steps = posterior_stats.get("context_raw_steps")
        if context_rates is not None and context_steps is not None:
            for idx in range(int(context_rates.numel())):
                raw_step = int(context_steps[idx].item())
                logs[f"posterior_accept_rate_ctx_{raw_step}"] = context_rates[idx].detach()
        return total_loss, logs

    def training_step(self, data, batch_idx):
        loss, logs = self._forward_loss(data, use_closed_loop=self.use_closed_loop_training)
        for key, value in logs.items():
            self.log(f"train/{key}", value, on_step=True, batch_size=1)
        return loss

    def validation_step(self, data, batch_idx):
        if self.val_open_loop:
            loss, logs = self._forward_loss(data, use_closed_loop=False)
            self.log("val_open/loss", loss, on_epoch=True, sync_dist=True, batch_size=1)
            self.log(
                "val_open/loss_mixture",
                logs["loss_mixture"],
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "val_open/loss_aux_ce",
                logs["loss_aux_ce"],
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "val_open/aux_mixture_ratio",
                logs["aux_mixture_ratio"],
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )

        if self.val_closed_loop:
            pred_traj, pred_z, pred_head = self._run_closed_loop_rollouts(
                data=data,
                scenario_ids=data["scenario_id"],
            )
            update_wosac_distribution_metric_from_model(
                metric=self.wosac_distribution_metrics,
                model=self,
                data=data,
                pred_traj=pred_traj,
                include_gt=True,
            )

            scenario_rollouts = None
            if self.sim_agents_submission.is_active:
                self.sim_agents_submission.update(
                    scenario_id=data["scenario_id"],
                    agent_id=data["agent"]["id"],
                    agent_batch=data["agent"]["batch"],
                    pred_traj=pred_traj,
                    pred_z=pred_z,
                    pred_head=pred_head,
                )
                scenario_rollouts = self.sim_agents_submission.aggregate_current_batch()
            else:
                self.minADE.update(
                    pred=pred_traj,
                    target=data["agent"]["position"][
                        :, self.num_historical_steps :, : pred_traj.shape[-1]
                    ],
                    target_valid=data["agent"]["valid_mask"][:, self.num_historical_steps :],
                )
                if batch_idx < self.n_batch_sim_agents_metric:
                    self.sim_agents_metrics.update_from_prediction_tensors(
                        scenario_files=data["tfrecord_path"],
                        agent_id=data["agent"]["id"],
                        agent_batch=data["agent"]["batch"],
                        pred_traj=pred_traj,
                        pred_z=pred_z,
                        pred_head=pred_head,
                    )

            self._maybe_log_rollout_videos(data, batch_idx, pred_traj, pred_z, pred_head, scenario_rollouts)

    def _log_metrics_to_logger(self, metrics: Dict[str, object]) -> None:
        logger = getattr(self, "logger", None)
        if logger is not None and hasattr(logger, "log_metrics"):
            logger.log_metrics(metrics)

    def _get_video_logger(self):
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            for logger in getattr(trainer, "loggers", []) or []:
                if hasattr(logger, "log_video"):
                    return logger
        logger = getattr(self, "logger", None)
        if hasattr(logger, "log_video"):
            return logger
        return None

    def _cleanup_local_video(self, video_path: str) -> None:
        video_file = Path(video_path)
        try:
            video_file.resolve().relative_to(self.video_dir.resolve())
        except ValueError:
            return
        video_file.unlink(missing_ok=True)
        current_dir = video_file.parent
        while current_dir != self.video_dir.parent:
            try:
                current_dir.rmdir()
            except OSError:
                break
            current_dir = current_dir.parent

    def _maybe_log_rollout_videos(
        self,
        data,
        batch_idx: int,
        pred_traj: torch.Tensor,
        pred_z: torch.Tensor,
        pred_head: torch.Tensor,
        scenario_rollouts,
    ) -> None:
        if self.global_rank != 0 or batch_idx >= self.n_vis_batch:
            return
        video_logger = self._get_video_logger()
        if scenario_rollouts is None:
            device = pred_traj.device
            scenario_rollouts = get_scenario_rollouts(
                scenario_id=get_scenario_id_int_tensor(data["scenario_id"], device),
                agent_id=data["agent"]["id"],
                agent_batch=data["agent"]["batch"],
                pred_traj=pred_traj,
                pred_z=pred_z,
                pred_head=pred_head,
            )
        if scenario_rollouts is None:
            return
        for scenario_index in range(min(self.n_vis_scenario, len(data["scenario_id"]))):
            vis = VisWaymo(
                scenario_path=data["tfrecord_path"][scenario_index],
                save_dir=self.video_dir / f"batch_{batch_idx:02d}-scenario_{scenario_index:02d}",
            )
            vis.save_video_scenario_rollout(scenario_rollouts[scenario_index], self.n_vis_rollout)
            for path in vis.video_paths:
                if video_logger is not None:
                    video_logger.log_video("/".join(path.split("/")[-3:]), [path])
                    if self.delete_local_videos_after_wandb_upload:
                        self._cleanup_local_video(path)

    def on_validation_epoch_end(self):
        if not self.val_closed_loop:
            return
        distribution_metrics = log_and_reset_wosac_distribution_metric(
            self.wosac_distribution_metrics
        )
        if not self.sim_agents_submission.is_active:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                reduced_state = self.sim_agents_metrics.get_state_tensor(device=self.device)
                torch.distributed.all_reduce(reduced_state)
                sim_metrics = self.sim_agents_metrics.compute_from_state_tensor(reduced_state)
                minade_state = torch.stack(
                    [
                        self.minADE.sum.detach().to(device=self.device),
                        self.minADE.count.detach().to(device=self.device),
                    ]
                )
                torch.distributed.all_reduce(minade_state)
                minade_value = minade_state[0] / minade_state[1].clamp_min(1e-6)
            else:
                sim_metrics = self.sim_agents_metrics.compute()
                minade_value = self.minADE.sum / self.minADE.count.clamp_min(1e-6)
            closed_loop_metric = sim_metrics[self.closed_loop_metric_name]
            sim_metrics[self.val_closed_minade_name] = minade_value
            sim_metrics.update(distribution_metrics)
            self.log(
                self.closed_loop_metric_name,
                closed_loop_metric,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
            if self.global_rank == 0:
                sim_metrics["epoch"] = self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                self._log_metrics_to_logger(sim_metrics)
            self.sim_agents_metrics.reset()
            self.minADE.reset()
        else:
            if self.global_rank == 0 and distribution_metrics:
                distribution_metrics["epoch"] = self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                self._log_metrics_to_logger(distribution_metrics)
            self.sim_agents_submission.save_sub_file()

    def _make_closed_loop_seed(self, scenario_id: str, rollout_idx: int) -> int:
        payload = f"{self.validation_closed_seed}:{scenario_id}:{int(rollout_idx)}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False) & 0x7FFF_FFFF_FFFF_FFFF

    def _make_closed_loop_generators(
        self,
        scenario_ids: Sequence[str],
        rollout_idx: int,
        device: torch.device,
    ) -> dict[int, torch.Generator]:
        if len(scenario_ids) == 0:
            scenario_ids = ("0",)
        generators: dict[int, torch.Generator] = {}
        for scenario_index, scenario_id in enumerate(scenario_ids):
            generator = torch.Generator(device=device)
            generator.manual_seed(self._make_closed_loop_seed(str(scenario_id), rollout_idx))
            generators[int(scenario_index)] = generator
        return generators

    def _sample_component(
        self,
        logits: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if self.inference_temperature <= 0:
            return logits.argmax(dim=-1)
        scaled_logits = (logits / self.inference_temperature).float()
        candidate_indices: torch.Tensor | None = None
        if 0 < self.inference_top_k < scaled_logits.shape[-1]:
            scaled_logits, candidate_indices = torch.topk(
                scaled_logits,
                k=self.inference_top_k,
                dim=-1,
            )
        if 0.0 < self.inference_top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            remove_mask = sorted_probs.cumsum(dim=-1) > self.inference_top_p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
            probs = torch.softmax(sorted_logits, dim=-1)
            sampled = torch.multinomial(
                probs,
                num_samples=1,
                replacement=True,
                generator=generator,
            ).squeeze(-1)
            sampled = sorted_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            if candidate_indices is not None:
                return candidate_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            return sampled
        probs = torch.softmax(scaled_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1, replacement=True, generator=generator).squeeze(-1)
        if candidate_indices is not None:
            return candidate_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        return sampled

    def _sample_component_by_agent_batch(
        self,
        logits: torch.Tensor,
        agent_batch: torch.Tensor,
        generators_by_batch: dict[int, torch.Generator],
    ) -> torch.Tensor:
        if agent_batch.shape != logits.shape[:-1]:
            raise ValueError(
                "agent_batch must match logits without the component dimension, "
                f"got agent_batch={tuple(agent_batch.shape)} and logits={tuple(logits.shape)}."
            )

        sampled = torch.empty(logits.shape[:-1], dtype=torch.long, device=logits.device)
        batch_ids = torch.unique(agent_batch.detach().cpu()).tolist()
        for batch_id in batch_ids:
            batch_id = int(batch_id)
            generator = generators_by_batch.get(batch_id)
            if generator is None:
                raise ValueError(
                    f"Missing scenario generator for agent batch index {batch_id}. "
                    f"Available indices: {sorted(generators_by_batch)}."
                )
            mask = agent_batch == batch_id
            sampled[mask] = self._sample_component(logits[mask], generator)
        return sampled

    def _predict_one_step(
        self,
        tokenized_map: Dict[str, torch.Tensor],
        tokenized_agent: Dict[str, torch.Tensor],
        current_pos: torch.Tensor,
        current_head: torch.Tensor,
        generator: torch.Generator | dict[int, torch.Generator] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding_seq = self.network.encode(tokenized_map, tokenized_agent)
        embedding_now = embedding_seq[:, -1]
        logits = self.network.motion_decoder.scorer(embedding_now)
        if isinstance(generator, dict):
            z = self._sample_component_by_agent_batch(
                logits=logits,
                agent_batch=tokenized_agent["batch"],
                generators_by_batch=generator,
            )
        else:
            z = self._sample_component(logits, generator)
        selected_anchor = gather_anchors_by_type(
            self.anchors_by_type[:, :, : self.spec.num_prediction_steps],
            tokenized_agent["type"],
            z,
        )
        pred = self.network.motion_decoder.decode_selected(embedding_now, selected_anchor)
        pred_pos, pred_head = self.processor.local_prediction_to_global(
            mean_pos=pred["mean_pos"],
            mean_head=pred["mean_head"],
            ref_pos=current_pos,
            ref_head=current_head,
        )
        return pred_pos, pred_head, z

    @torch.no_grad()
    def _run_one_rollout(
        self,
        data,
        rollout_idx: int,
        scenario_ids: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenized_map, tokenized_agent, current_pos, current_head, current_valid = (
            self.processor.build_rollout_seed(data)
        )
        pred_traj = current_pos.new_zeros(
            (current_pos.shape[0], self.spec.num_future_steps, 2)
        )
        pred_head = current_head.new_zeros((current_head.shape[0], self.spec.num_future_steps))
        pred_z_step = current_head.new_zeros((current_head.shape[0],), dtype=torch.long)
        pred_z = data["agent"]["position"][:, self.num_historical_steps - 1, 2].new_zeros(
            (current_head.shape[0], self.spec.num_future_steps)
        )

        if scenario_ids is None:
            scenario_ids = data["scenario_id"]
        generators = self._make_closed_loop_generators(
            scenario_ids=scenario_ids,
            rollout_idx=rollout_idx,
            device=current_pos.device,
        )

        for rollout_step in range(self.spec.num_future_steps // self.spec.num_commit_steps):
            pred_pos_4s, pred_head_4s, z = self._predict_one_step(
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
                current_pos=current_pos,
                current_head=current_head,
                generator=generators,
            )
            sl = slice(
                rollout_step * self.spec.num_commit_steps,
                (rollout_step + 1) * self.spec.num_commit_steps,
            )
            pred_traj[:, sl] = pred_pos_4s[:, : self.spec.num_commit_steps]
            pred_head[:, sl] = pred_head_4s[:, : self.spec.num_commit_steps]
            pred_z[:, sl] = data["agent"]["position"][
                :, self.num_historical_steps - 1, 2
            ].unsqueeze(1)
            pred_z_step = z
            current_pos = pred_pos_4s[:, self.spec.num_commit_steps - 1]
            current_head = pred_head_4s[:, self.spec.num_commit_steps - 1]
            tokenized_agent = self.processor.append_rollout_state(
                tokenized_agent,
                next_pos=current_pos,
                next_head=current_head,
                next_valid=current_valid,
                next_tracklet_pos=pred_pos_4s[:, : self.spec.num_commit_steps],
                next_tracklet_head=pred_head_4s[:, : self.spec.num_commit_steps],
                next_tracklet_valid=current_valid[:, None].expand(-1, self.spec.num_commit_steps),
            )
        _ = pred_z_step
        return pred_traj, pred_z, pred_head

    @staticmethod
    def _is_cuda_out_of_memory(error: RuntimeError) -> bool:
        message = str(error).lower()
        return any(
            pattern in message
            for pattern in ("out of memory", "cuda error: out of memory", "cublas_status_alloc_failed")
        )

    @staticmethod
    def _cleanup_after_rollout_oom() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _run_closed_loop_rollouts(
        self,
        data,
        scenario_ids: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_traj_list = []
        pred_z_list = []
        pred_head_list = []
        for rollout_idx in range(self.n_rollout_closed_val):
            try:
                pred_traj, pred_z, pred_head = self._run_one_rollout(
                    data,
                    rollout_idx,
                    scenario_ids=scenario_ids,
                )
            except RuntimeError as error:
                if not self._is_cuda_out_of_memory(error):
                    raise
                self._cleanup_after_rollout_oom()
                raise
            pred_traj_list.append(pred_traj)
            pred_z_list.append(pred_z)
            pred_head_list.append(pred_head)
        return (
            torch.stack(pred_traj_list, dim=1),
            torch.stack(pred_z_list, dim=1),
            torch.stack(pred_head_list, dim=1),
        )

    def test_step(self, data, batch_idx):
        pred_traj, pred_z, pred_head = self._run_closed_loop_rollouts(
            data=data,
            scenario_ids=data["scenario_id"],
        )
        update_wosac_distribution_metric_from_model(
            metric=self.test_wosac_distribution_metrics,
            model=self,
            data=data,
            pred_traj=pred_traj,
            include_gt=False,
        )
        if self.sim_agents_submission.is_active:
            self.sim_agents_submission.update(
                scenario_id=data["scenario_id"],
                agent_id=data["agent"]["id"],
                agent_batch=data["agent"]["batch"],
                pred_traj=pred_traj,
                pred_z=pred_z,
                pred_head=pred_head,
            )
            self.sim_agents_submission.aggregate_current_batch()

    def on_test_epoch_end(self):
        distribution_metrics = log_and_reset_wosac_distribution_metric(
            self.test_wosac_distribution_metrics
        )
        if self.global_rank == 0 and distribution_metrics:
            self._log_metrics_to_logger(distribution_metrics)
        if self.sim_agents_submission.is_active:
            self.sim_agents_submission.save_sub_file()

    def _lr_multiplier(self, current_epoch: int) -> float:
        """Epoch-wise linear warmup followed by cosine-to-zero decay."""
        current_epoch = max(int(current_epoch), 0)
        warmup_steps = max(int(self.lr_warmup_steps), 0)
        total_steps = max(int(self.lr_total_steps), 1)
        if warmup_steps > 0 and current_epoch < warmup_steps:
            return float(current_epoch + 1) / float(warmup_steps)

        decay_start = warmup_steps if warmup_steps > 0 else 0
        decay_steps = max(total_steps - decay_start, 1)
        progress = min(1.0, max(0.0, (current_epoch - decay_start) / decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        def lr_lambda(current_epoch):
            return self._lr_multiplier(current_epoch)

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
