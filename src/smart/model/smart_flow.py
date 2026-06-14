from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import hydra
import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

from src.smart.metrics import (
    SimAgentsMetrics,
    SimAgentsSubmission,
    WOSACDistributionMetrics,
    log_and_reset_wosac_distribution_metric,
    minADE,
    update_wosac_distribution_metric_from_model,
)
from src.smart.metrics.flow_metrics import (
    WeightedMeanMetric,
    ade_future,
    fde_future,
    flow_matching_loss,
    yaw_ade_future,
    yaw_fde_future,
)
from src.smart.modules.self_forced_path_flow import (
    build_anchor0_normalized_committed_control,
    build_anchor0_normalized_committed_path,
    get_anchor0_valid_mask,
)
from src.smart.modules.self_forced_dmd_guidance import (
    active_control_dmd_surrogate_loss,
    build_active_control_mask,
    build_clean_dmd_direction,
    compute_self_forced_dmd_injection_scale,
    normalize_pose_heading_vector,
)
from src.smart.modules.self_forced_sid_loss import compute_clean_sid_loss
from src.smart.modules.self_forced_update_separation import (
    assert_no_module_gradients,
    clear_module_gradients,
    detach_tensor_tree,
    module_gradients_disabled,
)
from src.smart.modules.self_forced_estimator_warmup import (
    is_self_forced_estimator_warmup_epoch,
    resolve_self_forced_estimator_warmup_epochs,
    should_compute_anchor_flow_matching_loss,
    should_run_self_forced_validation_after_epoch,
)
from src.smart.modules.self_forced_trainable_range import (
    apply_self_forced_unfrozen_range,
    resolve_self_forced_unfrozen_range,
)
from src.smart.modules.kinematic_control import CYCLIST_TYPE_ID
from src.smart.modules.smart_flow_decoder import SMARTFlowDecoder
from src.smart.tokens.flow_token_processor import FlowTokenProcessor
from src.smart.utils.finetune import set_model_for_finetuning
from src.smart.utils.flow_horizon import format_flow_horizon_tag
from src.utils.vis_waymo import VisWaymo
from src.utils.sim_agents_utils import get_scenario_id_int_tensor, get_scenario_rollouts


_TOKEN_PROCESSOR_DECODER_SHARED_KEYS = (
    "use_kinematic_control_flow",
    "use_holonomic_model_only",
    "use_rolling_supervision",
    "control_pos_scale_m",
    "control_vehicle_no_slip_point_ratio",
    "control_cyclist_no_slip_point_ratio",
    "control_vehicle_yaw_scale_rad",
    "control_pedestrian_yaw_scale_rad",
    "control_cyclist_yaw_scale_rad",
)


def _values_match(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-12)
    return left == right


def _build_decoder_config_from_token_processor(decoder_config: Any, token_processor: FlowTokenProcessor) -> Dict[str, Any]:
    """token_processor와 decoder가 공유하는 control-space 설정을 한 곳에서 고정합니다."""
    synced_config = dict(decoder_config)
    for key in _TOKEN_PROCESSOR_DECODER_SHARED_KEYS:
        token_value = getattr(token_processor, key)
        if key in synced_config:
            decoder_value = decoder_config[key]
            if not _values_match(decoder_value, token_value):
                raise ValueError(
                    f"model_config.decoder.{key} must match "
                    f"model_config.token_processor.{key}. "
                    "Set the token_processor value only; decoder uses it as the single source of truth. "
                    f"got decoder={decoder_value!r}, token_processor={token_value!r}."
                )
        synced_config[key] = token_value
    return synced_config


class SMARTFlow(LightningModule):

    def __init__(self, model_config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = model_config.lr
        self.lr_warmup_steps = model_config.lr_warmup_steps
        self.lr_total_steps = model_config.lr_total_steps
        self.lr_min_ratio = model_config.lr_min_ratio
        self.num_historical_steps = model_config.decoder.num_historical_steps
        self.flow_window_steps = int(getattr(model_config.decoder, "flow_window_steps", 20))
        self.flow_horizon_tag = format_flow_horizon_tag(self.flow_window_steps)
        self.log_epoch = -1
        self.val_open_loop = model_config.val_open_loop
        self.val_closed_loop = model_config.val_closed_loop
        self.train_open_loop_metrics = bool(
            getattr(model_config, "train_open_loop_metrics", True)
        )
        self.skip_empty_open_loop_optimizer_guard = bool(
            getattr(model_config, "skip_empty_open_loop_optimizer_guard", False)
        )
        self.token_processor = FlowTokenProcessor(**model_config.token_processor)
        self.use_kinematic_control_flow = bool(self.token_processor.use_kinematic_control_flow)
        decoder_config = _build_decoder_config_from_token_processor(
            decoder_config=model_config.decoder,
            token_processor=self.token_processor,
        )

        self.encoder = SMARTFlowDecoder(
            **decoder_config,
            n_token_agent=self.token_processor.n_token_agent,
        )
        if self.flow_window_steps != int(self.token_processor.flow_window_steps):
            raise ValueError(
                "decoder.flow_window_steps and token_processor.flow_window_steps must match, "
                f"got {self.flow_window_steps} and {int(self.token_processor.flow_window_steps)}."
            )
        set_model_for_finetuning(self.encoder, model_config.finetune)

        self.minADE = minADE()
        self.minADE_predict = minADE()
        self.sim_agents_metrics = SimAgentsMetrics(
            "val_closed",
            max_workers=model_config.sim_agents_metric_workers,
        )
        self.sim_agents_submission = SimAgentsSubmission(**model_config.sim_agents_submission)
        wosac_cpd_reference = getattr(model_config, "wosac_cpd_reference", None)
        wosac_distribution_type_scale = getattr(
            model_config,
            "wosac_distribution_type_scale",
            None,
        )
        self.wosac_distribution_metrics = WOSACDistributionMetrics(
            prefix="val_closed",
            cpd_reference=wosac_cpd_reference,
            type_scale=wosac_distribution_type_scale,
        )
        self.test_wosac_distribution_metrics = WOSACDistributionMetrics(
            prefix="test",
            cpd_reference=wosac_cpd_reference,
            type_scale=wosac_distribution_type_scale,
        )

        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self.closed_loop_metric_name = "val_closed/sim_agents_2025/realism_meta_metric"
        self.val_closed_minade_name = (
            f"val_closed/sim_agents_2025/minADE_best_of_n_rollout_closed_val"
        )
        self.validation_open_seed = int(model_config.validation_open_seed)
        self.validation_closed_seed = int(model_config.validation_closed_seed)
        self.n_vis_batch = model_config.n_vis_batch
        self.n_vis_scenario = model_config.n_vis_scenario
        self.n_vis_rollout = model_config.n_vis_rollout
        self.vis_ghost_gt = bool(getattr(model_config, "vis_ghost_gt", True))
        self.vis_flow_2s_preview = bool(
            getattr(
                model_config,
                "vis_flow_preview",
                getattr(model_config, "vis_flow_2s_preview", False),
            )
        )
        self.delete_local_videos_after_wandb_upload = model_config.delete_local_videos_after_wandb_upload
        self.n_batch_sim_agents_metric = model_config.n_batch_sim_agents_metric
        self.scorer_scene_num = getattr(model_config, "scorer_scene_num", None)
        self._scorer_scene_num_last_key: tuple[int, int, int] | None = None
        self._scorer_val_limit_last_key: tuple[int, int | float, int] | None = None
        self._fit_time_original_limit_val_batches: int | float | None = None
        self._fit_time_checkpoint_only_validation_enabled = False
        self.open_metric_names = {
            "ade": f"ADE{self.flow_horizon_tag}",
            "fde": f"FDE{self.flow_horizon_tag}",
            "yaw_ade": f"yaw_ADE{self.flow_horizon_tag}",
            "yaw_fde": f"yaw_FDE{self.flow_horizon_tag}",
        }
        self.train_open_metric_names = {
            "ade": self.open_metric_names["ade"],
            "fde": self.open_metric_names["fde"],
            "yaw_ade": f"ADEyaw{self.flow_horizon_tag}",
            "yaw_fde": f"FDEyaw{self.flow_horizon_tag}",
        }
        self._train_open_epoch_log_names = (
            "train/loss",
            "train/loss_fm",
            f"train/{self.train_open_metric_names['ade']}",
            f"train/{self.train_open_metric_names['fde']}",
            f"train/{self.train_open_metric_names['yaw_ade']}",
            f"train/{self.train_open_metric_names['yaw_fde']}",
        )
        self.register_buffer(
            "_train_open_epoch_metric_sums",
            torch.zeros(len(self._train_open_epoch_log_names), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_train_open_epoch_metric_counts",
            torch.zeros(len(self._train_open_epoch_log_names), dtype=torch.float32),
            persistent=False,
        )

        self.video_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.video_dir = Path(self.video_dir) / "videos"

        self.validation_rollout_sampling = model_config.validation_rollout_sampling

        self.self_forced_config = getattr(model_config, "self_forced", None)
        self.self_forced_enabled = bool(
            self.self_forced_config is not None
            and getattr(self.self_forced_config, "enabled", False)
        )
        self.self_forced_start_epoch = (
            int(getattr(self.self_forced_config, "start_epoch", 0))
            if self.self_forced_config is not None
            else 0
        )
        self.self_forced_weight = (
            float(getattr(self.self_forced_config, "weight", 1.0))
            if self.self_forced_config is not None
            else 0.0
        )
        self.self_forced_direction_normalizer_eps = (
            float(getattr(self.self_forced_config, "clean_dmd_normalizer_eps", 0.05))
            if self.self_forced_config is not None
            else 0.05
        )
        self.self_forced_dmd_beta = (
            float(getattr(self.self_forced_config, "beta", 1.0))
            if self.self_forced_config is not None
            else 1.0
        )
        if (
            not math.isfinite(self.self_forced_dmd_beta)
            or self.self_forced_dmd_beta <= 0.0
            or self.self_forced_dmd_beta > 1.0
        ):
            raise ValueError(
                "self_forced.beta must be finite and in the interval (0, 1], "
                f"got {self.self_forced_dmd_beta!r}."
            )
        self.self_forced_distribution_matching_objective = (
            str(getattr(self.self_forced_config, "distribution_matching_objective", "dmd")).lower()
            if self.self_forced_config is not None
            else "dmd"
        )
        if self.self_forced_distribution_matching_objective not in {"dmd", "sid"}:
            raise ValueError(
                "self_forced.distribution_matching_objective must be 'dmd' or 'sid', "
                f"got {self.self_forced_distribution_matching_objective}."
            )
        self.self_forced_project_dmd_to_pose_space = (
            bool(getattr(self.self_forced_config, "project_dmd_to_pose_space", False))
            if self.self_forced_config is not None
            else False
        )
        self.self_forced_dmd_use_stable_scale_filter = (
            bool(getattr(self.self_forced_config, "dmd_use_stable_scale_filter", True))
            if self.self_forced_config is not None
            else True
        )
        self.self_forced_dmd_stable_scale_scope = (
            str(getattr(self.self_forced_config, "dmd_stable_scale_scope", "agent")).lower()
            if self.self_forced_config is not None
            else "agent"
        )
        if self.self_forced_dmd_stable_scale_scope not in {"agent", "type", "scene"}:
            raise ValueError(
                "self_forced.dmd_stable_scale_scope must be one of "
                "'agent', 'type', or 'scene', "
                f"got {self.self_forced_dmd_stable_scale_scope!r}."
            )
        self.self_forced_dmd_use_teacher_alignment_filter = (
            bool(getattr(self.self_forced_config, "dmd_use_teacher_alignment_filter", False))
            if self.self_forced_config is not None
            else False
        )
        self.self_forced_dmd_use_trust_region_filter = (
            bool(getattr(self.self_forced_config, "dmd_use_trust_region_filter", False))
            if self.self_forced_config is not None
            else False
        )
        self.self_forced_dmd_use_injection_ramp = (
            bool(getattr(self.self_forced_config, "dmd_use_injection_ramp", False))
            if self.self_forced_config is not None
            else False
        )
        self.self_forced_sid_alpha = (
            float(getattr(self.self_forced_config, "sid_alpha", 1.0))
            if self.self_forced_config is not None
            else 1.0
        )
        self.self_forced_sid_normalizer_eps = (
            float(
                getattr(
                    self.self_forced_config,
                    "sid_normalizer_eps",
                    1.0e-3,
                )
            )
            if self.self_forced_config is not None
            else 1.0e-3
        )
        self.self_forced_detach_block_transition = (
            bool(getattr(self.self_forced_config, "detach_block_transition", False))
            if self.self_forced_config is not None
            else False
        )
        # Stop-motion gating is disabled for both inference and self-forced
        # training rollouts in this branch, regardless of config overrides.
        self.self_forced_use_stop_motion = False
        self.self_forced_guidance_tau_low = (
            float(getattr(self.self_forced_config, "clean_dmd_tau_low", 0.02))
            if self.self_forced_config is not None
            else 0.02
        )
        self.self_forced_guidance_tau_high = (
            float(getattr(self.self_forced_config, "clean_dmd_tau_high", 0.98))
            if self.self_forced_config is not None
            else 0.98
        )
        self.self_forced_anchor_weight = (
            float(getattr(self.self_forced_config, "anchor_weight", 0.05))
            if self.self_forced_config is not None
            else 0.0
        )
        self.self_forced_use_anchor_fm_loss = (
            bool(getattr(self.self_forced_config, "use_anchor_flow_matching_loss", True))
            if self.self_forced_config is not None
            else True
        )
        self.self_forced_use_distribution_matching_loss = (
            bool(getattr(self.self_forced_config, "use_distribution_matching_loss", True))
            if self.self_forced_config is not None
            else True
        )
        self.self_forced_estimator_updates_per_step = (
            max(1, int(getattr(self.self_forced_config, "estimator_updates_per_step", 1)))
            if self.self_forced_config is not None
            else 1
        )
        self.self_forced_generated_estimator_lr = (
            float(getattr(self.self_forced_config, "generated_estimator_lr", self.lr))
            if self.self_forced_config is not None
            else self.lr
        )
        self.self_forced_cache_frozen_map_features = (
            bool(getattr(self.self_forced_config, "cache_frozen_map_features", True))
            if self.self_forced_config is not None
            else False
        )
        self.self_forced_estimator_warmup_epochs = (
            resolve_self_forced_estimator_warmup_epochs(self.self_forced_config)
        )
        self.self_forced_generated_estimator_init_path = (
            str(getattr(self.self_forced_config, "generated_estimator_init_path", "") or "")
            if self.self_forced_config is not None
            else ""
        )
        self.self_forced_generated_estimator_init_strict = (
            bool(getattr(self.self_forced_config, "generated_estimator_init_strict", True))
            if self.self_forced_config is not None
            else True
        )
        self.self_forced_generated_estimator_skip_warmup_on_load = (
            bool(getattr(self.self_forced_config, "generated_estimator_skip_warmup_on_load", True))
            if self.self_forced_config is not None
            else True
        )
        self.self_forced_generated_estimator_bank_snapshot_path = (
            str(getattr(self.self_forced_config, "generated_estimator_bank_snapshot_path", "") or "")
            if self.self_forced_config is not None
            else ""
        )
        self.self_forced_generated_estimator_bank_target_warmup_epochs = (
            int(getattr(self.self_forced_config, "generated_estimator_bank_target_warmup_epochs", 0) or 0)
            if self.self_forced_config is not None
            else 0
        )
        self.self_forced_generated_estimator_bank_loaded_warmup_epochs = (
            int(getattr(self.self_forced_config, "generated_estimator_bank_loaded_warmup_epochs", 0) or 0)
            if self.self_forced_config is not None
            else 0
        )
        self.self_forced_generated_estimator_bank_upload_artifact = (
            str(getattr(self.self_forced_config, "generated_estimator_bank_upload_artifact", "") or "")
            if self.self_forced_config is not None
            else ""
        )
        self.self_forced_generated_estimator_bank_upload_on_warmup_end = (
            bool(getattr(self.self_forced_config, "generated_estimator_bank_upload_on_warmup_end", False))
            if self.self_forced_config is not None
            else False
        )
        self._self_forced_generated_estimator_bank_loaded = False
        self._self_forced_generated_estimator_bank_snapshot_saved = False
        self.self_forced_initialize_aux_on_fit_start = (
            bool(getattr(self.self_forced_config, "initialize_aux_from_generator_on_fit_start", True))
            if self.self_forced_config is not None
            else True
        )
        self.self_forced_unfrozen_range = resolve_self_forced_unfrozen_range(
            self.self_forced_config,
        )
        self.self_forced_gradient_clip_val = (
            float(getattr(self.self_forced_config, "gradient_clip_val", 1.0))
            if self.self_forced_config is not None
            else 1.0
        )
        self.self_forced_ema_weight = (
            float(getattr(self.self_forced_config, "ema_weight", 0.99))
            if self.self_forced_config is not None
            else 0.99
        )
        self.self_forced_ema_start_step = (
            max(0, int(getattr(self.self_forced_config, "ema_start_step", 50)))
            if self.self_forced_config is not None
            else 50
        )
        self.self_forced_sampling = (
            getattr(self.self_forced_config, "sampling", self.validation_rollout_sampling)
            if self.self_forced_config is not None
            else self.validation_rollout_sampling
        )
        self.self_forced_target_teacher = None
        self.self_forced_generated_estimator = None
        self.self_forced_generator_ema = None
        self._self_forced_aux_loaded_from_checkpoint = False
        self._self_forced_generator_ema_loaded_from_checkpoint = False
        self._self_forced_backward_context: Dict[str, Tensor] | None = None
        self._self_forced_original_check_val_every_n_epoch: int | None = None
        self._self_forced_validation_schedule_captured = False
        self._automatic_open_loop_has_target_since_step = False
        self._automatic_open_loop_has_target_pending: list[tuple[Tensor, Any | None]] = []
        self._skip_next_automatic_optimizer_step = False
        if self.self_forced_enabled:
            if not (0.0 <= self.self_forced_ema_weight < 1.0):
                raise ValueError(
                    "self_forced.ema_weight must be in [0, 1), "
                    f"got {self.self_forced_ema_weight}."
                )
            self.automatic_optimization = False
            self.strict_loading = False
            self.self_forced_target_teacher = copy.deepcopy(self.encoder)
            self.self_forced_target_teacher.requires_grad_(False)
            self.self_forced_generated_estimator = copy.deepcopy(self.encoder)
            self.self_forced_generator_ema = copy.deepcopy(self.encoder)
            self.self_forced_generator_ema.requires_grad_(False)
            self.self_forced_generator_ema.eval()
            self.register_buffer(
                "self_forced_generator_update_count",
                torch.zeros((), dtype=torch.long),
                persistent=True,
            )
            self.register_buffer(
                "self_forced_generator_ema_ready",
                torch.zeros((), dtype=torch.bool),
                persistent=True,
            )
        self._apply_self_forced_unfrozen_range()

        self.val_open_epoch_metrics = nn.ModuleDict(
            {
                self.open_metric_names["ade"]: WeightedMeanMetric(),
                self.open_metric_names["fde"]: WeightedMeanMetric(),
                self.open_metric_names["yaw_ade"]: WeightedMeanMetric(),
                self.open_metric_names["yaw_fde"]: WeightedMeanMetric(),
            }
        )

    def _should_enable_fit_time_checkpoint_only_validation(self) -> bool:
        """학습 중 validation을 체크포인트 점수 전용으로 줄일지 판단합니다.

        Returns:
            bool:
                아래 조건을 모두 만족하면 ``True`` 를 돌려줍니다.
                1) closed-loop validation을 사용함
                2) open-loop validation을 같이 쓰지 않음
                3) submission 저장 모드가 아님
                4) Fast WOSAC 점수에 사용할 batch 개수가 1 이상임
        """
        return (
            self.val_closed_loop
            and not self.val_open_loop
            and not self.sim_agents_submission.is_active
            and int(self.n_batch_sim_agents_metric) > 0
        )

    def _resolve_val_batch_size(self) -> int | None:
        """현재 trainer datamodule의 validation batch size를 안전하게 읽습니다."""
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
        """GPU 수와 validation batch size에 맞춰 scorer batch 수를 자동 조정합니다.

        ``scorer_scene_num`` 이 양의 정수이면 전역 기준으로 그 정도의 scene을
        Fast WOSAC scorer에 넣을 수 있도록 ``n_batch_sim_agents_metric`` 을 per-rank
        batch 수로 덮어씁니다. 별도의 scenario-level cap은 두지 않습니다.
        """
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
        n_batch_override = max(1, math.ceil(per_rank_scenes / val_batch_size))
        self.n_batch_sim_agents_metric = int(n_batch_override)

        current_key = (int(scorer_scene_num), int(world_size), int(val_batch_size))
        if self._scorer_scene_num_last_key == current_key:
            return
        self._scorer_scene_num_last_key = current_key
        if getattr(trainer, "is_global_zero", True):
            print(
                "[scorer_scene_num] Fast WOSAC sim_agents_2025 scorer batch 수를 "
                f"n_batch_sim_agents_metric={self.n_batch_sim_agents_metric} 으로 설정합니다 "
                f"(requested_scenes={scorer_scene_num}, world_size={world_size}, "
                f"val_batch_size={val_batch_size}).",
                flush=True,
            )

    def _estimate_val_batches_per_rank(self) -> int | None:
        """현재 rank에서 실행 가능한 validation batch 수를 추정합니다."""
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
        """Fast WOSAC scorer가 요청 scene 수까지 도달하도록 val loop cap을 보정합니다."""
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        try:
            target_batches = int(self.n_batch_sim_agents_metric)
        except (TypeError, ValueError):
            return
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
            if limit_val_batches <= 0.0:
                return
            if limit_val_batches >= 1.0:
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
                "[scorer_scene_num] Fast WOSAC scorer가 요청 scene 수까지 평가하도록 "
                f"trainer.limit_val_batches를 {old_limit}에서 {target_batches}로 늘립니다 "
                f"(기존 resolved_val_batches={resolved_batches}).",
                flush=True,
            )

    def _configure_fast_wosac_validation_scope(self) -> None:
        """scorer scene 수와 validation loop cap을 함께 정렬합니다."""
        self._apply_scorer_scene_num_overrides()
        self._ensure_validation_limit_reaches_scorer_batches()

    def _apply_fit_time_validation_batch_limit(self) -> None:
        """학습 중 validation에서 앞쪽 일부 batch만 돌도록 trainer 값을 바꿉니다.

        이 함수는 학습 시작 시 한 번 호출됩니다.
        사용자가 넘긴 config 파일은 그대로 두고, 실행 중 trainer 객체의
        validation batch 제한만 잠깐 바꿉니다.

        Returns:
            None
        """
        if not self._should_enable_fit_time_checkpoint_only_validation():
            self._fit_time_checkpoint_only_validation_enabled = False
            return

        if self.trainer is None:
            return

        if self._fit_time_original_limit_val_batches is None:
            self._fit_time_original_limit_val_batches = self.trainer.limit_val_batches

        target_batches = int(self.n_batch_sim_agents_metric)
        self.trainer.limit_val_batches = target_batches
        self._fit_time_checkpoint_only_validation_enabled = True

    def _restore_fit_time_validation_batch_limit(self) -> None:
        """학습이 끝나면 trainer의 validation 제한 값을 원래대로 돌립니다.

        Returns:
            None
        """
        if self.trainer is None:
            self._fit_time_checkpoint_only_validation_enabled = False
            return

        if self._fit_time_original_limit_val_batches is not None:
            self.trainer.limit_val_batches = self._fit_time_original_limit_val_batches

        self._fit_time_original_limit_val_batches = None
        self._fit_time_checkpoint_only_validation_enabled = False

    def _capture_self_forced_validation_interval(self) -> None:
        """self-forced warmup이 trainer validation 주기를 바꾸기 전 원래 값을 저장합니다."""
        if self.trainer is None:
            return
        if self._self_forced_validation_schedule_captured:
            return
        self._self_forced_original_check_val_every_n_epoch = self.trainer.check_val_every_n_epoch
        self._self_forced_validation_schedule_captured = True

    def _restore_self_forced_validation_interval(self) -> None:
        """fit 종료 시 trainer의 epoch validation 주기를 원래 값으로 복원합니다."""
        if self.trainer is not None and self._self_forced_validation_schedule_captured:
            self.trainer.check_val_every_n_epoch = (
                self._self_forced_original_check_val_every_n_epoch
            )
        self._self_forced_original_check_val_every_n_epoch = None
        self._self_forced_validation_schedule_captured = False

    def _self_forced_skip_validation_interval_for_current_epoch(self) -> int:
        """현재 epoch 끝 validation이 실행되지 않게 하는 임시 interval을 반환합니다."""
        return int(self.current_epoch) + 2

    def _apply_self_forced_validation_schedule_for_current_epoch(self) -> None:
        """estimator warmup 이후부터 validation 주기를 다시 세도록 trainer 값을 조정합니다."""
        if self.trainer is None:
            return
        if not self.self_forced_enabled:
            return
        if not self.self_forced_use_distribution_matching_loss:
            return
        if int(self.self_forced_estimator_warmup_epochs) <= 0:
            return

        self._capture_self_forced_validation_interval()
        original_interval = self._self_forced_original_check_val_every_n_epoch
        if original_interval is None:
            return
        check_interval = int(original_interval)
        if check_interval <= 0:
            return

        current_epoch = int(self.current_epoch)
        if current_epoch < int(self.self_forced_start_epoch):
            self.trainer.check_val_every_n_epoch = check_interval
            return

        should_validate = should_run_self_forced_validation_after_epoch(
            current_epoch=current_epoch,
            self_forced_start_epoch=int(self.self_forced_start_epoch),
            estimator_warmup_epochs=int(self.self_forced_estimator_warmup_epochs),
            check_val_every_n_epoch=check_interval,
        )
        if should_validate:
            self.trainer.check_val_every_n_epoch = 1
        else:
            self.trainer.check_val_every_n_epoch = (
                self._self_forced_skip_validation_interval_for_current_epoch()
            )

    def _should_compute_closed_loop_minade(self) -> bool:
        """현재 validation에서 closed-loop minADE를 계산할지 판단합니다.

        학습 중 빠른 validation에서는 checkpoint 선택에 쓰는 Fast WOSAC 점수만
        남기고 minADE 계산은 끕니다.

        Returns:
            bool:
                minADE를 계산해야 하면 ``True`` 입니다.
        """
        return not self._fit_time_checkpoint_only_validation_enabled

    def _get_video_logger(self):
        if self.trainer is not None:
            for logger in getattr(self.trainer, "loggers", []):
                if hasattr(logger, "log_video"):
                    return logger
        if hasattr(self.logger, "log_video"):
            return self.logger
        return None

    def _cleanup_local_video(self, video_path: str) -> None:
        video_file = Path(video_path)
        if video_file.exists():
            video_file.unlink()

        current_dir = video_file.parent
        while current_dir != self.video_dir.parent:
            try:
                current_dir.rmdir()
            except OSError:
                break
            current_dir = current_dir.parent

    def _get_scenario_flow_preview(
        self,
        agent_id: Tensor,
        agent_batch: Tensor,
        scenario_index: int,
        flow_preview: Dict[str, Tensor] | None,
    ) -> Dict[str, object] | None:
        if flow_preview is None:
            return None

        scenario_mask = agent_batch == scenario_index
        if not scenario_mask.any():
            return None

        return {
            "object_id": agent_id[scenario_mask].detach().cpu().numpy(),
            "traj": flow_preview["traj"][scenario_mask].detach().cpu().numpy(),
            "valid": flow_preview["valid"][scenario_mask].detach().cpu().numpy(),
        }

    def _build_open_loop_metric_dict(
        self,
        pred_clean_norm: Tensor,
        target_clean_norm: Tensor,
        valid_mask: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        """open-loop 위치와 방향 오차를 유효한 미래 step 기준으로 계산합니다.

        Args:
            pred_clean_norm: 모델이 만든 정규화된 미래입니다.
                shape은 ``[n_valid_anchor, flow_window_steps, 4]`` 입니다.
            target_clean_norm: 정답 정규화 미래입니다.
                shape은 ``[n_valid_anchor, flow_window_steps, 4]`` 입니다.
            valid_mask: 지표 계산에 포함할 미래 step입니다.
                shape은 ``[n_valid_anchor, flow_window_steps]`` 입니다.
                값이 없으면 전체 step을 사용합니다.

        Returns:
            Dict[str, Tensor]:
                meter 단위 위치 오차와 degree 단위 방향 오차를 담은 사전입니다.
        """
        metric_mask = valid_mask.detach() if valid_mask is not None else None
        with torch.no_grad():
            return {
                self.open_metric_names["ade"]: ade_future(
                    pred_clean_norm.detach(),
                    target_clean_norm.detach(),
                    valid_mask=metric_mask,
                ),
                self.open_metric_names["fde"]: fde_future(
                    pred_clean_norm.detach(),
                    target_clean_norm.detach(),
                    valid_mask=metric_mask,
                ),
                self.open_metric_names["yaw_ade"]: yaw_ade_future(
                    pred_clean_norm.detach(),
                    target_clean_norm.detach(),
                    valid_mask=metric_mask,
                ),
                self.open_metric_names["yaw_fde"]: yaw_fde_future(
                    pred_clean_norm.detach(),
                    target_clean_norm.detach(),
                    valid_mask=metric_mask,
                ),
            }

    @staticmethod
    def _has_open_loop_loss_targets(pred_dict: Dict[str, Tensor]) -> bool:
        """open-loop FM loss에 실제로 들어갈 미래 target이 있는지 확인합니다."""
        pred_norm = pred_dict["flow_pred_norm"]
        target_norm = pred_dict["flow_target_norm"]
        if pred_norm.numel() == 0 or target_norm.numel() == 0:
            return False
        loss_mask = pred_dict.get("flow_loss_mask")
        if loss_mask is None:
            return True
        return bool(loss_mask.to(device=pred_norm.device, dtype=torch.bool).any().item())

    def _build_trainable_connected_zero_loss(self, module: nn.Module | None = None) -> Tensor:
        """trainable parameter graph에 연결된 scalar 0 loss를 만듭니다."""
        zero_loss: Tensor | None = None
        parameter_source = module if module is not None else self
        for param in parameter_source.parameters():
            if not param.requires_grad:
                continue
            term = param.sum() * 0.0
            zero_loss = term if zero_loss is None else zero_loss + term
        if zero_loss is None:
            return torch.zeros((), device=self.device, requires_grad=True)
        return zero_loss

    @staticmethod
    def _first_parameter_device(module: nn.Module) -> torch.device:
        """module 안 첫 parameter device를 반환합니다."""
        for param in module.parameters():
            return param.device
        return torch.device("cpu")

    def _optimizer_parameter_device(self, optimizer) -> torch.device:
        """optimizer가 관리하는 첫 parameter device를 반환합니다."""
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        for group in getattr(raw_optimizer, "param_groups", []):
            for param in group.get("params", []):
                return param.device
        return self._first_parameter_device(self)

    @staticmethod
    def _distributed_available_and_initialized() -> bool:
        """torch.distributed all-reduce를 사용할 수 있는지 확인합니다."""
        distributed = getattr(torch, "distributed", None)
        return bool(
            distributed is not None
            and distributed.is_available()
            and distributed.is_initialized()
        )

    def _sync_distributed_bool_any(
        self,
        value: bool,
        *,
        device: torch.device | None = None,
    ) -> bool:
        """DDP 전체 rank 중 하나라도 True인지 동기화해 반환합니다."""
        sync_device = device if device is not None else self._first_parameter_device(self)
        flag = torch.tensor(int(bool(value)), device=sync_device, dtype=torch.long)
        if self._distributed_available_and_initialized():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        return bool(flag.item())

    def _start_distributed_bool_any(
        self,
        value: bool,
        *,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Any | None]:
        """DDP any bool sync를 시작하고, 가능하면 backward와 겹치도록 async work를 반환합니다."""
        sync_device = device if device is not None else self._first_parameter_device(self)
        flag = torch.tensor(int(bool(value)), device=sync_device, dtype=torch.long)
        if self._distributed_available_and_initialized():
            work = torch.distributed.all_reduce(
                flag,
                op=torch.distributed.ReduceOp.MAX,
                async_op=True,
            )
            return flag, work
        return flag, None

    @staticmethod
    def _finish_distributed_bool_any(pending: tuple[Tensor, Any | None]) -> bool:
        """_start_distributed_bool_any 결과를 기다린 뒤 Python bool로 반환합니다."""
        flag, work = pending
        if work is not None:
            work.wait()
        return bool(flag.item())

    def _accumulate_open_loop_train_epoch_metrics(
        self,
        *,
        total_loss: Tensor,
        fm_loss: Tensor,
        open_metric_dict: Dict[str, Tensor],
        sample_count: int,
    ) -> None:
        """Open-loop train metric을 epoch 말 global 평균용으로 local 누적합니다.

        Train step마다 logging metric 전체를 DDP 동기화하면 작은 collective가
        매 batch 발생합니다. 학습 loss/backward 경로는 그대로 두고, detached scalar
        값만 buffer에 누적한 뒤 epoch 끝에서 한 번만 동기화합니다.
        """
        weight = float(max(int(sample_count), 0))
        if weight <= 0.0:
            return
        values = [
            total_loss.detach(),
            fm_loss.detach(),
        ]
        if open_metric_dict:
            values.extend(
                [
                    open_metric_dict[self.open_metric_names["ade"]].detach(),
                    open_metric_dict[self.open_metric_names["fde"]].detach(),
                    open_metric_dict[self.open_metric_names["yaw_ade"]].detach(),
                    open_metric_dict[self.open_metric_names["yaw_fde"]].detach(),
                ]
            )
        else:
            zero = total_loss.detach().new_zeros(())
            values.extend([zero, zero, zero, zero])
        value_tensor = torch.stack(
            [
                value.to(
                    device=self._train_open_epoch_metric_sums.device,
                    dtype=torch.float32,
                ).reshape(())
                for value in values
            ]
        )
        weight_tensor = value_tensor.new_full(value_tensor.shape, weight)
        if not open_metric_dict:
            weight_tensor[2:] = 0.0
        self._train_open_epoch_metric_sums += value_tensor * weight_tensor
        self._train_open_epoch_metric_counts += weight_tensor

    def _reset_open_loop_train_epoch_metrics(self) -> None:
        self._train_open_epoch_metric_sums.zero_()
        self._train_open_epoch_metric_counts.zero_()

    def _compute_and_reset_open_loop_train_epoch_metrics(self) -> Dict[str, Tensor]:
        """누적 train metric을 DDP 전체에서 합산한 뒤 epoch 평균으로 반환합니다."""
        packed = torch.cat(
            [
                self._train_open_epoch_metric_sums.detach().clone(),
                self._train_open_epoch_metric_counts.detach().clone(),
            ],
            dim=0,
        )
        self._reset_open_loop_train_epoch_metrics()
        if self._distributed_available_and_initialized():
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)

        n_metric = len(self._train_open_epoch_log_names)
        sums = packed[:n_metric]
        counts = packed[n_metric:]
        metrics: Dict[str, Tensor] = {}
        for idx, name in enumerate(self._train_open_epoch_log_names):
            count = counts[idx]
            if bool((count > 0).item()):
                metrics[name] = sums[idx] / count.clamp_min(1.0)
        return metrics

    def _log_open_loop_train_epoch_metrics(self) -> None:
        """W&B에는 step별 global sync 없이 epoch 말 train metric만 정확히 남깁니다."""
        if self.self_forced_enabled and self.current_epoch >= self.self_forced_start_epoch:
            # Self-forced epochs do not use the open-loop pretrain metric
            # accumulator for optimization. Avoid an extra epoch-boundary DDP
            # collective here; Lightning handles the self-forced train logs.
            self._reset_open_loop_train_epoch_metrics()
            return
        metrics = self._compute_and_reset_open_loop_train_epoch_metrics()
        if not metrics or not self.trainer.is_global_zero:
            return
        loggers = getattr(self.trainer, "loggers", None)
        if loggers is None:
            logger = getattr(self.trainer, "logger", None)
            loggers = [logger] if logger is not None else []
        metrics_to_log = {
            name: float(value.detach().cpu().item())
            for name, value in metrics.items()
        }
        for logger in loggers:
            if logger is not None:
                logger.log_metrics(metrics_to_log, step=self.global_step)

    def _open_loop_denoise_metrics(
        self,
        pred_dict: Dict[str, Tensor],
        zero_loss_module: nn.Module | None = None,
    ) -> tuple[Tensor, Dict[str, Tensor], int, bool]:
        """잡음 제거 방식 검증 점수와 유효 표본 수를 계산합니다.

        Args:
            pred_dict: flow decoder가 낸 출력 사전입니다.
                ``flow_pred_norm`` 과 ``flow_target_norm`` 의 shape은
                ``[n_valid_anchor, flow_window_steps, 4]`` 입니다.
                ``flow_loss_mask`` 가 있으면 shape은
                ``[n_valid_anchor, flow_window_steps]`` 입니다.
            zero_loss_module: 유효 target이 없을 때 0 loss를 연결할 trainable
                parameter 소스입니다. 값이 없으면 flow generator에 연결합니다.

        Returns:
            tuple[Tensor, Dict[str, Tensor], int, bool]:
                flow matching loss, meter/degree 단위 지표 사전,
                유효 anchor 개수, 그리고 loss에 실제 target이 있는지 여부입니다.
        """
        loss_mask = pred_dict.get("flow_loss_mask")
        has_loss_targets = self._has_open_loop_loss_targets(pred_dict)
        if has_loss_targets:
            loss = flow_matching_loss(
                pred_dict["flow_pred_norm"],
                pred_dict["flow_target_norm"],
                valid_mask=loss_mask,
            )
        else:
            loss = self._build_trainable_connected_zero_loss(zero_loss_module or self.encoder)
        if not self.train_open_loop_metrics and self.training:
            sample_count = int(pred_dict["flow_clean_norm"].shape[0])
            return loss, {}, sample_count, has_loss_targets
        metric_pred_clean_norm = pred_dict.get(
            "flow_pred_clean_metric_norm",
            pred_dict["flow_pred_clean_norm"],
        )
        metric_target_clean_norm = pred_dict.get(
            "flow_clean_metric_norm",
            pred_dict["flow_clean_norm"],
        )
        metric_dict = self._build_open_loop_metric_dict(
            pred_clean_norm=metric_pred_clean_norm,
            target_clean_norm=metric_target_clean_norm,
            valid_mask=loss_mask,
        )
        sample_count = int(pred_dict["flow_clean_norm"].shape[0])
        return loss, metric_dict, sample_count, has_loss_targets

    def _update_weighted_validation_metrics(
        self,
        metric_store: nn.ModuleDict,
        metric_dict: Dict[str, Tensor],
        sample_count: int,
    ) -> None:
        """batch 평균을 유효 표본 수로 가중해 epoch 누적 상태에 반영합니다.

        Args:
            metric_store: ``WeightedMeanMetric`` 들을 담은 저장소입니다.
            metric_dict: 이번 batch에서 계산한 스칼라 지표 사전입니다.
            sample_count: 이번 batch에서 실제로 채점된 anchor 개수입니다.
        """
        for metric_name, metric_value in metric_dict.items():
            metric_store[metric_name].update(metric_value.detach(), sample_count)

    def _compute_and_reset_validation_metrics(
        self,
        prefix: str,
        metric_store: nn.ModuleDict,
    ) -> Dict[str, Tensor]:
        """누적된 validation 지표를 계산한 뒤 다음 epoch를 위해 초기화합니다.

        Args:
            prefix: 로그 이름 앞부분입니다.
            metric_store: ``WeightedMeanMetric`` 들을 담은 저장소입니다.

        Returns:
            Dict[str, Tensor]: ``prefix/metric_name`` 형태의 최종 스칼라 지표 사전입니다.
        """
        computed_metrics: Dict[str, Tensor] = {}
        for metric_name, metric in metric_store.items():
            computed_metrics[f"{prefix}/{metric_name}"] = metric.compute()
            metric.reset()
        return computed_metrics

    def _get_validation_open_seed(self, batch_idx: int) -> int:
        """배치 순서가 같으면 매 epoch 같은 open 샘플이 나오도록 seed를 만듭니다.

        Args:
            batch_idx: 현재 validation batch 순번입니다.

        Returns:
            int: 이번 batch에서 사용할 고정 seed입니다.
        """
        return self.validation_open_seed + int(batch_idx)

    def _make_closed_loop_seed(self, scenario_id: str, rollout_idx: int) -> int:
        """시나리오 문자열과 rollout 번호를 섞어 어디서 돌려도 같은 seed를 만듭니다.

        Args:
            scenario_id: Waymo 시나리오 문자열입니다.
            rollout_idx: 같은 시나리오 안 rollout 번호입니다.

        Returns:
            int: 0 이상 63비트 범위의 고정 seed입니다.
        """
        seed_payload = (
            f"{self.validation_closed_seed}:{scenario_id}:{int(rollout_idx)}".encode("utf-8")
        )
        digest = hashlib.blake2b(seed_payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False) & 0x7FFF_FFFF_FFFF_FFFF

    def _get_closed_loop_scenario_seeds(
        self,
        scenario_ids: Sequence[str],
        rollout_idx: int,
        device: torch.device,
    ) -> Tensor:
        """배치 안 각 시나리오용 closed-loop seed를 만듭니다.

        Args:
            scenario_ids: 현재 batch의 시나리오 문자열 목록입니다.
                길이는 ``[n_scenario]`` 입니다.
            rollout_idx: 같은 시나리오 안 rollout 번호입니다.
            device: seed 텐서를 올릴 장치입니다.

        Returns:
            Tensor:
                시나리오별 고정 seed입니다.
                shape은 ``[n_scenario]`` 입니다.
        """
        seed_rollout_idx, _ = self._get_closed_loop_antithetic_base_and_sign(rollout_idx)
        scenario_seeds = [
            self._make_closed_loop_seed(scenario_id=scenario_id, rollout_idx=seed_rollout_idx)
            for scenario_id in scenario_ids
        ]
        return torch.tensor(scenario_seeds, dtype=torch.long, device=device)

    def _use_closed_loop_antithetic_pairs(self) -> bool:
        """validation closed-loop rollout에서 antithetic noise pair를 쓸지 반환합니다."""
        return bool(getattr(self.validation_rollout_sampling, "antithetic_pairs", False))

    def _use_closed_loop_stratified_gaussian_noise(self) -> bool:
        """validation closed-loop rollout에서 stratified Gaussian noise를 쓸지 반환합니다."""
        return bool(
            getattr(self.validation_rollout_sampling, "stratified_gaussian_noise", False)
        )

    def _closed_loop_stratified_noise_num_strata(self) -> int:
        """stratified Gaussian base rollout bin 개수를 반환합니다."""
        if not self._use_closed_loop_stratified_gaussian_noise():
            return 0
        n_rollout = int(self.n_rollout_closed_val)
        if not self._use_closed_loop_antithetic_pairs():
            raise ValueError(
                "validation_rollout_sampling.stratified_gaussian_noise=true requires "
                "validation_rollout_sampling.antithetic_pairs=true."
            )
        if n_rollout % 2 != 0:
            raise ValueError(
                "validation_rollout_sampling.stratified_gaussian_noise=true requires an "
                f"even n_rollout_closed_val, got {n_rollout}."
            )
        return n_rollout // 2

    def _get_closed_loop_antithetic_base_and_sign(self, rollout_idx: int) -> tuple[int, float]:
        """rollout 번호를 antithetic pair용 base 번호와 noise 부호로 바꿉니다."""
        rollout_idx = int(rollout_idx)
        if not self._use_closed_loop_antithetic_pairs():
            return rollout_idx, 1.0

        n_rollout = int(self.n_rollout_closed_val)
        if n_rollout % 2 != 0:
            raise ValueError(
                "validation_rollout_sampling.antithetic_pairs=true requires an even "
                f"n_rollout_closed_val, got {n_rollout}."
            )
        pair_offset = n_rollout // 2
        if rollout_idx < pair_offset:
            return rollout_idx, 1.0
        return rollout_idx - pair_offset, -1.0

    def _get_closed_loop_scenario_noise_signs(
        self,
        scenario_ids: Sequence[str],
        rollout_idx: int,
        device: torch.device,
    ) -> Tensor:
        """배치 안 각 시나리오용 closed-loop noise 부호를 만듭니다."""
        _, noise_sign = self._get_closed_loop_antithetic_base_and_sign(rollout_idx)
        return torch.full(
            (len(scenario_ids),),
            float(noise_sign),
            dtype=torch.float32,
            device=device,
        )

    def _make_closed_loop_stratification_seed(self, scenario_id: str) -> int:
        """scenario별 stratified noise bin permutation seed를 만듭니다."""
        seed_payload = (
            f"{self.validation_closed_seed}:{scenario_id}:stratified_gaussian_noise".encode(
                "utf-8"
            )
        )
        digest = hashlib.blake2b(seed_payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False) & 0x7FFF_FFFF_FFFF_FFFF

    def _get_closed_loop_scenario_stratification_seeds(
        self,
        scenario_ids: Sequence[str],
        device: torch.device,
    ) -> Tensor:
        """배치 안 각 시나리오용 stratified noise permutation seed를 만듭니다."""
        scenario_seeds = [
            self._make_closed_loop_stratification_seed(scenario_id=scenario_id)
            for scenario_id in scenario_ids
        ]
        return torch.tensor(scenario_seeds, dtype=torch.long, device=device)

    def _build_closed_loop_seed_table(
        self,
        scenario_ids: Sequence[str],
        rollout_indices: Sequence[int],
        device: torch.device,
    ) -> Tensor:
        """여러 rollout의 scenario seed를 한 번에 모읍니다.

        Args:
            scenario_ids: 현재 batch의 시나리오 문자열 목록입니다.
                길이는 ``[n_scenario]`` 입니다.
            rollout_indices: 이번에 함께 돌릴 rollout 번호 목록입니다.
                길이는 ``[n_rollout_chunk]`` 입니다.
            device: seed 텐서를 올릴 장치입니다.

        Returns:
            Tensor:
                rollout별, scenario별 고정 seed 표입니다.
                shape은 ``[n_rollout_chunk, n_scenario]`` 입니다.
        """
        seed_rows = [
            self._get_closed_loop_scenario_seeds(
                scenario_ids=scenario_ids,
                rollout_idx=rollout_idx,
                device=device,
            )
            for rollout_idx in rollout_indices
        ]
        if len(seed_rows) == 0:
            return torch.zeros((0, len(scenario_ids)), dtype=torch.long, device=device)
        return torch.stack(seed_rows, dim=0)

    def _build_closed_loop_noise_sign_table(
        self,
        scenario_ids: Sequence[str],
        rollout_indices: Sequence[int],
        device: torch.device,
    ) -> Tensor | None:
        """여러 rollout의 scenario별 noise 부호를 한 번에 모읍니다."""
        if not self._use_closed_loop_antithetic_pairs():
            return None
        sign_rows = [
            self._get_closed_loop_scenario_noise_signs(
                scenario_ids=scenario_ids,
                rollout_idx=rollout_idx,
                device=device,
            )
            for rollout_idx in rollout_indices
        ]
        if len(sign_rows) == 0:
            return torch.zeros((0, len(scenario_ids)), dtype=torch.float32, device=device)
        return torch.stack(sign_rows, dim=0)

    def _build_closed_loop_noise_strata_table(
        self,
        scenario_ids: Sequence[str],
        rollout_indices: Sequence[int],
        device: torch.device,
    ) -> Tensor | None:
        """여러 rollout의 scenario별 stratified noise bin offset을 모읍니다."""
        num_strata = self._closed_loop_stratified_noise_num_strata()
        if num_strata <= 0:
            return None
        stratum_rows = []
        for rollout_idx in rollout_indices:
            base_idx, _ = self._get_closed_loop_antithetic_base_and_sign(int(rollout_idx))
            if base_idx < 0 or base_idx >= num_strata:
                raise ValueError(
                    f"stratified Gaussian base rollout index must be in [0, {num_strata}), "
                    f"got {base_idx} for rollout_idx={rollout_idx}."
                )
            stratum_rows.append(
                torch.full(
                    (len(scenario_ids),),
                    int(base_idx),
                    dtype=torch.long,
                    device=device,
                )
            )
        if len(stratum_rows) == 0:
            return torch.zeros((0, len(scenario_ids)), dtype=torch.long, device=device)
        return torch.stack(stratum_rows, dim=0)

    def _build_closed_loop_stratification_seed_table(
        self,
        scenario_ids: Sequence[str],
        rollout_indices: Sequence[int],
        device: torch.device,
    ) -> Tensor | None:
        """여러 rollout의 scenario별 stratified noise permutation seed를 모읍니다."""
        if not self._use_closed_loop_stratified_gaussian_noise():
            return None
        scenario_seed_row = self._get_closed_loop_scenario_stratification_seeds(
            scenario_ids=scenario_ids,
            device=device,
        )
        if len(rollout_indices) == 0:
            return torch.zeros((0, len(scenario_ids)), dtype=torch.long, device=device)
        return scenario_seed_row.unsqueeze(0).repeat(len(rollout_indices), 1)

    def _repeat_tensor_on_first_dim(self, tensor: Tensor, repeat_count: int) -> Tensor:
        """첫 번째 축을 rollout 수만큼 반복합니다.

        Args:
            tensor: 원본 텐서입니다. shape은 ``[n_item, ...]`` 입니다.
            repeat_count: 반복 횟수입니다.

        Returns:
            Tensor:
                첫 번째 축만 늘어난 텐서입니다.
                shape은 ``[repeat_count * n_item, ...]`` 입니다.
        """
        if repeat_count == 1:
            return tensor
        repeat_pattern = (repeat_count,) + (1,) * tensor.dim()
        return tensor.unsqueeze(0).repeat(repeat_pattern).flatten(0, 1).contiguous()

    def _expand_batch_index_for_rollouts(
        self,
        batch_index: Tensor,
        repeat_count: int,
        num_graphs: int,
    ) -> Tensor:
        """rollout마다 다른 장면 번호를 갖도록 batch 번호를 벌립니다.

        Args:
            batch_index: 원본 장면 번호입니다. shape은 ``[n_item]`` 입니다.
            repeat_count: 반복할 rollout 개수입니다.
            num_graphs: 원본 batch 안 장면 개수입니다.

        Returns:
            Tensor:
                rollout 축까지 붙은 새 장면 번호입니다.
                shape은 ``[repeat_count * n_item]`` 입니다.
        """
        if repeat_count == 1:
            return batch_index
        rollout_offsets = (
            torch.arange(repeat_count, device=batch_index.device, dtype=batch_index.dtype)
            * int(num_graphs)
        )
        expanded_batch = batch_index.unsqueeze(0).repeat(repeat_count, 1)
        expanded_batch = expanded_batch + rollout_offsets.unsqueeze(1)
        return expanded_batch.reshape(-1).contiguous()

    def _build_parallel_rollout_map_feature(
        self,
        map_feature: Dict[str, Tensor],
        repeat_count: int,
        num_graphs: int,
    ) -> Dict[str, Tensor]:
        """지도 특징을 rollout 병렬 실행용 큰 batch로 펼칩니다.

        Args:
            map_feature: 지도 인코더 출력입니다.
                ``pt_token`` 과 ``position`` 은 ``[n_map_token, ...]`` 이고,
                ``batch`` 는 ``[n_map_token]`` 입니다.
            repeat_count: 이번에 동시에 돌릴 rollout 개수입니다.
            num_graphs: 원본 batch 안 장면 개수입니다.

        Returns:
            Dict[str, Tensor]:
                rollout까지 펼친 지도 특징입니다.
                지도 토큰 축은 ``[repeat_count * n_map_token, ...]`` 입니다.
        """
        if repeat_count == 1:
            return map_feature
        expanded_map_feature = {
            "pt_token": self._repeat_tensor_on_first_dim(map_feature["pt_token"], repeat_count),
            "position": self._repeat_tensor_on_first_dim(map_feature["position"], repeat_count),
            "orientation": self._repeat_tensor_on_first_dim(
                map_feature["orientation"], repeat_count
            ),
            "batch": self._expand_batch_index_for_rollouts(
                map_feature["batch"],
                repeat_count=repeat_count,
                num_graphs=num_graphs,
            ),
        }
        if "light_type" in map_feature:
            expanded_map_feature["light_type"] = self._repeat_tensor_on_first_dim(
                map_feature["light_type"],
                repeat_count,
            )
        return expanded_map_feature

    def _build_parallel_rollout_tokenized_agent(
        self,
        tokenized_agent: Dict[str, Tensor],
        repeat_count: int,
        num_graphs: int,
    ) -> Dict[str, Tensor]:
        """rollout 병렬 실행에 필요한 agent 입력만 늘려서 만듭니다.

        Args:
            tokenized_agent: 평가용 agent 토큰 사전입니다.
                agent 축 텐서는 대체로 ``[n_agent, ...]`` 입니다.
            repeat_count: 이번에 동시에 돌릴 rollout 개수입니다.
            num_graphs: 원본 batch 안 장면 개수입니다.

        Returns:
            Dict[str, Tensor]:
                rollout까지 펼친 입력 사전입니다.
                agent 축 텐서는 ``[repeat_count * n_agent, ...]`` 입니다.
        """
        if repeat_count == 1:
            return tokenized_agent

        runtime_tokenized_agent = {
            "batch": self._expand_batch_index_for_rollouts(
                tokenized_agent["batch"],
                repeat_count=repeat_count,
                num_graphs=num_graphs,
            ),
            "type": self._repeat_tensor_on_first_dim(tokenized_agent["type"], repeat_count),
            "token_agent_shape": self._repeat_tensor_on_first_dim(
                tokenized_agent["token_agent_shape"], repeat_count
            ),
            "token_bank_all_veh": tokenized_agent["token_bank_all_veh"],
            "token_bank_all_ped": tokenized_agent["token_bank_all_ped"],
            "token_bank_all_cyc": tokenized_agent["token_bank_all_cyc"],
            "gt_pos_raw": self._repeat_tensor_on_first_dim(tokenized_agent["gt_pos_raw"], repeat_count),
            "gt_head_raw": self._repeat_tensor_on_first_dim(
                tokenized_agent["gt_head_raw"], repeat_count
            ),
            "gt_valid_raw": self._repeat_tensor_on_first_dim(
                tokenized_agent["gt_valid_raw"], repeat_count
            ),
            "gt_pos": self._repeat_tensor_on_first_dim(tokenized_agent["gt_pos"], repeat_count),
            "gt_heading": self._repeat_tensor_on_first_dim(
                tokenized_agent["gt_heading"], repeat_count
            ),
            "valid_mask": self._repeat_tensor_on_first_dim(
                tokenized_agent["valid_mask"], repeat_count
            ),
            "gt_z_raw": self._repeat_tensor_on_first_dim(tokenized_agent["gt_z_raw"], repeat_count),
        }

        if "shape" in tokenized_agent:
            runtime_tokenized_agent["shape"] = self._repeat_tensor_on_first_dim(
                tokenized_agent["shape"],
                repeat_count,
            )

        return runtime_tokenized_agent

    def _build_parallel_rollout_cache(
        self,
        rollout_cache: Dict[str, object],
        repeat_count: int,
    ) -> Dict[str, object]:
        """rollout cache의 agent 축 상태를 rollout 수만큼 펼칩니다.

        Args:
            rollout_cache: ``prepare_inference_cache`` 가 만든 원본 캐시입니다.
                agent 축 상태 텐서는 ``[n_agent, ...]`` 입니다.
            repeat_count: 이번에 동시에 돌릴 rollout 개수입니다.

        Returns:
            Dict[str, object]:
                rollout 병렬 실행용 큰 캐시입니다.
                agent 축 상태 텐서는 ``[repeat_count * n_agent, ...]`` 입니다.
        """
        if repeat_count == 1:
            return rollout_cache

        categorical_embs = rollout_cache["categorical_embs"]
        if isinstance(categorical_embs, tuple):
            expanded_categorical_embs = tuple(
                self._repeat_tensor_on_first_dim(emb, repeat_count) if torch.is_tensor(emb) else emb
                for emb in categorical_embs
            )
        else:
            expanded_categorical_embs = [
                self._repeat_tensor_on_first_dim(emb, repeat_count) if torch.is_tensor(emb) else emb
                for emb in categorical_embs
            ]

        feat_a_t_dict = rollout_cache["feat_a_t_dict"]
        expanded_feat_a_t_dict = {
            layer_idx: self._repeat_tensor_on_first_dim(layer_value, repeat_count)
            for layer_idx, layer_value in feat_a_t_dict.items()
        }

        expanded_cache = {
            "n_agent": int(rollout_cache["n_agent"]) * repeat_count,
            "n_step_future_10hz": int(rollout_cache["n_step_future_10hz"]),
            "n_step_future_2hz": int(rollout_cache["n_step_future_2hz"]),
            "max_context_steps": int(rollout_cache["max_context_steps"]),
            "pos_window": self._repeat_tensor_on_first_dim(rollout_cache["pos_window"], repeat_count),
            "head_window": self._repeat_tensor_on_first_dim(rollout_cache["head_window"], repeat_count),
            "head_vector_window": self._repeat_tensor_on_first_dim(
                rollout_cache["head_vector_window"], repeat_count
            ),
            "valid_window": self._repeat_tensor_on_first_dim(
                rollout_cache["valid_window"], repeat_count
            ),
            "pred_idx_window": self._repeat_tensor_on_first_dim(
                rollout_cache["pred_idx_window"], repeat_count
            ),
            "feat_a": self._repeat_tensor_on_first_dim(rollout_cache["feat_a"], repeat_count),
            "agent_token_emb": self._repeat_tensor_on_first_dim(
                rollout_cache["agent_token_emb"], repeat_count
            ),
            "agent_token_emb_veh": rollout_cache["agent_token_emb_veh"],
            "agent_token_emb_ped": rollout_cache["agent_token_emb_ped"],
            "agent_token_emb_cyc": rollout_cache["agent_token_emb_cyc"],
            "veh_mask": self._repeat_tensor_on_first_dim(rollout_cache["veh_mask"], repeat_count),
            "ped_mask": self._repeat_tensor_on_first_dim(rollout_cache["ped_mask"], repeat_count),
            "cyc_mask": self._repeat_tensor_on_first_dim(rollout_cache["cyc_mask"], repeat_count),
            "categorical_embs": expanded_categorical_embs,
            "feat_a_now": self._repeat_tensor_on_first_dim(
                rollout_cache["feat_a_now"], repeat_count
            ),
            "feat_a_t_dict": expanded_feat_a_t_dict,
        }
        for key in [
            "exec_pos_history_10hz",
            "exec_head_history_10hz",
            "exec_valid_history_10hz",
            "exec_pos_pair_10hz",
            "exec_head_pair_10hz",
            "exec_valid_pair_10hz",
        ]:
            if key in rollout_cache:
                expanded_cache[key] = self._repeat_tensor_on_first_dim(
                    rollout_cache[key],
                    repeat_count,
                )
        return expanded_cache

    def _reshape_parallel_rollout_prediction(
        self,
        pred_tensor: Tensor,
        repeat_count: int,
        num_agent: int,
    ) -> Tensor:
        """병렬 rollout 출력을 기존 metric shape로 되돌립니다.

        Args:
            pred_tensor: rollout 축을 agent 축에 붙여서 만든 출력입니다.
                shape은 ``[repeat_count * n_agent, ...]`` 입니다.
            repeat_count: 이번 chunk의 rollout 개수입니다.
            num_agent: 원래 batch의 agent 개수입니다.

        Returns:
            Tensor:
                rollout 축이 다시 분리된 출력입니다.
                shape은 ``[n_agent, repeat_count, ...]`` 입니다.
        """
        pred_tensor = pred_tensor.reshape(repeat_count, num_agent, *pred_tensor.shape[1:])
        permute_order = (1, 0) + tuple(range(2, pred_tensor.dim()))
        return pred_tensor.permute(*permute_order).contiguous()

    def _run_parallel_rollout_chunk(
        self,
        rollout_encoder: SMARTFlowDecoder,
        data,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        rollout_cache: Dict[str, object],
        rollout_indices: Sequence[int],
        return_flow_2s_preview: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Dict[str, Tensor] | None]:
        """주어진 rollout 번호 묶음을 한 번의 큰 batch로 실행합니다.

        Args:
            rollout_encoder: rollout을 실행할 Generator입니다.
            data: dataloader가 준 원본 batch입니다.
            tokenized_agent: 평가용 agent 토큰 사전입니다.
                agent 축 텐서는 ``[n_agent, ...]`` 입니다.
            map_feature: 한 번 인코딩한 지도 특징입니다.
                지도 토큰 축 텐서는 ``[n_map_token, ...]`` 입니다.
            rollout_cache: 원본 closed-loop cache 입니다.
            rollout_indices: 이번에 한꺼번에 돌릴 rollout 번호 목록입니다.
                길이는 ``[n_rollout_chunk]`` 입니다.

        Returns:
            tuple[Tensor, Tensor, Tensor, Dict[str, Tensor] | None]:
                위치, 높이, 방향 예측입니다.
                shape은 각각 ``[n_agent, n_rollout_chunk, 80, 2]``,
                ``[n_agent, n_rollout_chunk, 80]``,
                ``[n_agent, n_rollout_chunk, 80]`` 입니다.
                마지막 값은 선택적 2초 preview 사전입니다.
        """
        chunk_size = int(len(rollout_indices))
        scenario_device = tokenized_agent["batch"].device
        if chunk_size == 1:
            scenario_sampling_seeds = self._get_closed_loop_scenario_seeds(
                scenario_ids=data["scenario_id"],
                rollout_idx=int(rollout_indices[0]),
                device=scenario_device,
            )
            scenario_sampling_signs = self._build_closed_loop_noise_sign_table(
                scenario_ids=data["scenario_id"],
                rollout_indices=rollout_indices,
                device=scenario_device,
            )
            if scenario_sampling_signs is not None:
                scenario_sampling_signs = scenario_sampling_signs.reshape(-1).contiguous()
            scenario_sampling_strata = self._build_closed_loop_noise_strata_table(
                scenario_ids=data["scenario_id"],
                rollout_indices=rollout_indices,
                device=scenario_device,
            )
            if scenario_sampling_strata is not None:
                scenario_sampling_strata = scenario_sampling_strata.reshape(-1).contiguous()
            scenario_sampling_stratification_seeds = (
                self._build_closed_loop_stratification_seed_table(
                    scenario_ids=data["scenario_id"],
                    rollout_indices=rollout_indices,
                    device=scenario_device,
                )
            )
            if scenario_sampling_stratification_seeds is not None:
                scenario_sampling_stratification_seeds = (
                    scenario_sampling_stratification_seeds.reshape(-1).contiguous()
                )
            pred = rollout_encoder.rollout_from_cache(
                rollout_cache=rollout_cache,
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
                sampling_scheme=self.validation_rollout_sampling,
                scenario_sampling_seeds=scenario_sampling_seeds,
                scenario_sampling_signs=scenario_sampling_signs,
                scenario_sampling_strata=scenario_sampling_strata,
                scenario_sampling_stratification_seeds=scenario_sampling_stratification_seeds,
                scenario_sampling_num_strata=self._closed_loop_stratified_noise_num_strata(),
                return_flow_2s_preview=return_flow_2s_preview,
            )
            flow_preview = None
            if return_flow_2s_preview:
                flow_preview = {
                    "traj": pred["pred_flow_preview_traj"].unsqueeze(1),
                    "valid": pred["pred_flow_preview_valid"].unsqueeze(1),
                }
            return (
                pred["pred_traj_10hz"].unsqueeze(1),
                pred["pred_z_10hz"].unsqueeze(1),
                pred["pred_head_10hz"].unsqueeze(1),
                flow_preview,
            )

        num_agent = int(tokenized_agent["batch"].shape[0])
        num_graphs = len(data["scenario_id"])
        scenario_seed_table = self._build_closed_loop_seed_table(
            scenario_ids=data["scenario_id"],
            rollout_indices=rollout_indices,
            device=scenario_device,
        )
        scenario_sign_table = self._build_closed_loop_noise_sign_table(
            scenario_ids=data["scenario_id"],
            rollout_indices=rollout_indices,
            device=scenario_device,
        )
        scenario_strata_table = self._build_closed_loop_noise_strata_table(
            scenario_ids=data["scenario_id"],
            rollout_indices=rollout_indices,
            device=scenario_device,
        )
        scenario_stratification_seed_table = self._build_closed_loop_stratification_seed_table(
            scenario_ids=data["scenario_id"],
            rollout_indices=rollout_indices,
            device=scenario_device,
        )
        expanded_tokenized_agent = self._build_parallel_rollout_tokenized_agent(
            tokenized_agent=tokenized_agent,
            repeat_count=chunk_size,
            num_graphs=num_graphs,
        )
        expanded_map_feature = self._build_parallel_rollout_map_feature(
            map_feature=map_feature,
            repeat_count=chunk_size,
            num_graphs=num_graphs,
        )
        expanded_rollout_cache = self._build_parallel_rollout_cache(
            rollout_cache=rollout_cache,
            repeat_count=chunk_size,
        )
        pred = rollout_encoder.rollout_from_cache(
            rollout_cache=expanded_rollout_cache,
            tokenized_agent=expanded_tokenized_agent,
            map_feature=expanded_map_feature,
            sampling_scheme=self.validation_rollout_sampling,
            scenario_sampling_seeds=scenario_seed_table.reshape(-1).contiguous(),
            scenario_sampling_signs=(
                scenario_sign_table.reshape(-1).contiguous()
                if scenario_sign_table is not None
                else None
            ),
            scenario_sampling_strata=(
                scenario_strata_table.reshape(-1).contiguous()
                if scenario_strata_table is not None
                else None
            ),
            scenario_sampling_stratification_seeds=(
                scenario_stratification_seed_table.reshape(-1).contiguous()
                if scenario_stratification_seed_table is not None
                else None
            ),
            scenario_sampling_num_strata=self._closed_loop_stratified_noise_num_strata(),
            return_flow_2s_preview=return_flow_2s_preview,
        )
        flow_preview = None
        if return_flow_2s_preview:
            flow_preview = {
                "traj": self._reshape_parallel_rollout_prediction(
                    pred["pred_flow_preview_traj"],
                    repeat_count=chunk_size,
                    num_agent=num_agent,
                ),
                "valid": self._reshape_parallel_rollout_prediction(
                    pred["pred_flow_preview_valid"],
                    repeat_count=chunk_size,
                    num_agent=num_agent,
                ),
            }
        return (
            self._reshape_parallel_rollout_prediction(
                pred["pred_traj_10hz"],
                repeat_count=chunk_size,
                num_agent=num_agent,
            ),
            self._reshape_parallel_rollout_prediction(
                pred["pred_z_10hz"],
                repeat_count=chunk_size,
                num_agent=num_agent,
            ),
            self._reshape_parallel_rollout_prediction(
                pred["pred_head_10hz"],
                repeat_count=chunk_size,
                num_agent=num_agent,
            ),
            flow_preview,
        )

    def _build_rollout_chunk_size_candidates(self) -> list[int]:
        """한 번에 같이 돌릴 rollout 개수 후보를 큰 값부터 만듭니다.

        Returns:
            list[int]:
                가장 공격적인 값부터 안전한 값까지의 후보 목록입니다.
                예를 들면 ``8 -> [8, 4, 2, 1]`` 입니다.
        """
        chunk_sizes: list[int] = []
        current = max(1, int(self.n_rollout_closed_val))
        while True:
            if current not in chunk_sizes:
                chunk_sizes.append(current)
            if current == 1:
                break
            current = max(1, math.ceil(current / 2))
        return chunk_sizes

    def _is_cuda_out_of_memory(self, error: RuntimeError) -> bool:
        """CUDA 메모리 부족 예외인지 문자열로 판별합니다.

        Args:
            error: rollout 실행 중 잡은 예외입니다.

        Returns:
            bool:
                메모리 부족으로 보는 게 맞으면 ``True`` 입니다.
        """
        error_message = str(error).lower()
        oom_patterns = (
            "out of memory",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
        )
        return any(pattern in error_message for pattern in oom_patterns)

    def _cleanup_after_rollout_oom(self) -> None:
        """병렬 rollout 시도 실패 뒤 남은 임시 메모리를 정리합니다.

        Returns:
            None
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_closed_loop_rollouts(
        self,
        rollout_encoder: SMARTFlowDecoder,
        data,
        tokenized_agent,
        map_feature: Dict[str, Tensor],
        return_flow_2s_preview: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Dict[str, Tensor] | None]:
        """한 batch의 모든 closed-loop rollout을 가능한 크게 묶어 생성합니다.

        기본은 모든 rollout을 한 번에 큰 batch로 처리합니다.
        다만 메모리가 부족하면 자동으로 묶음 크기를 절반 정도씩 줄여
        같은 결과 shape을 유지한 채 다시 시도합니다.

        Args:
            rollout_encoder: rollout을 실행할 Generator입니다. EMA가 준비된 validation/test에서는
                EMA Generator가 들어오고, 그 전에는 online Generator가 들어옵니다.
            data: dataloader가 준 원본 batch입니다.
            tokenized_agent: 평가용 agent 토큰 사전입니다.
            map_feature: 한 번 인코딩한 지도 특징입니다.

        Returns:
            tuple[Tensor, Tensor, Tensor, Dict[str, Tensor] | None]:
                위치, 높이, 방향 예측입니다.
                shape은 각각 ``[n_agent, n_rollout, 80, 2]``,
                ``[n_agent, n_rollout, 80]``,
                ``[n_agent, n_rollout, 80]`` 입니다.
                마지막 값은 선택적 2초 preview 사전입니다.
        """
        rollout_cache = rollout_encoder.prepare_inference_cache(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
        )
        rollout_indices = list(range(int(self.n_rollout_closed_val)))
        last_oom_error: RuntimeError | None = None

        for chunk_size in self._build_rollout_chunk_size_candidates():
            pred_traj_chunks: list[Tensor] = []
            pred_z_chunks: list[Tensor] = []
            pred_head_chunks: list[Tensor] = []
            flow_preview_traj_chunks: list[Tensor] = []
            flow_preview_valid_chunks: list[Tensor] = []
            try:
                for chunk_start in range(0, len(rollout_indices), chunk_size):
                    chunk_rollout_indices = rollout_indices[chunk_start : chunk_start + chunk_size]
                    chunk_pred_traj, chunk_pred_z, chunk_pred_head, chunk_flow_preview = self._run_parallel_rollout_chunk(
                        rollout_encoder=rollout_encoder,
                        data=data,
                        tokenized_agent=tokenized_agent,
                        map_feature=map_feature,
                        rollout_cache=rollout_cache,
                        rollout_indices=chunk_rollout_indices,
                        return_flow_2s_preview=return_flow_2s_preview,
                    )
                    pred_traj_chunks.append(chunk_pred_traj)
                    pred_z_chunks.append(chunk_pred_z)
                    pred_head_chunks.append(chunk_pred_head)
                    if return_flow_2s_preview and chunk_flow_preview is not None:
                        flow_preview_traj_chunks.append(chunk_flow_preview["traj"])
                        flow_preview_valid_chunks.append(chunk_flow_preview["valid"])
                flow_preview = None
                if return_flow_2s_preview:
                    flow_preview = {
                        "traj": torch.cat(flow_preview_traj_chunks, dim=1),
                        "valid": torch.cat(flow_preview_valid_chunks, dim=1),
                    }
                return (
                    torch.cat(pred_traj_chunks, dim=1),
                    torch.cat(pred_z_chunks, dim=1),
                    torch.cat(pred_head_chunks, dim=1),
                    flow_preview,
                )
            except RuntimeError as error:
                if (not self._is_cuda_out_of_memory(error)) or chunk_size == 1:
                    raise
                last_oom_error = error
                del pred_traj_chunks, pred_z_chunks, pred_head_chunks
                del flow_preview_traj_chunks, flow_preview_valid_chunks
                self._cleanup_after_rollout_oom()
                continue

        if last_oom_error is not None:
            raise last_oom_error
        raise RuntimeError("closed-loop rollout 실행 중 알 수 없는 오류가 발생했습니다.")

    def _update_closed_loop_metric_states(
        self,
        data,
        batch_idx: int,
        pred_traj: Tensor,
        pred_z: Tensor,
        pred_head: Tensor,
    ) -> object:
        scenario_rollouts = None
        if self._should_compute_closed_loop_minade():
            self.minADE.update(
                pred=pred_traj,
                target=data["agent"]["position"][:, self.num_historical_steps :, : pred_traj.shape[-1]],
                target_valid=data["agent"]["valid_mask"][:, self.num_historical_steps :],
            )
            predict_mask = data["agent"]["role"][:, 2]  # tracks_to_predict
            if predict_mask.any():
                target_valid_predict = (
                    data["agent"]["valid_mask"][:, self.num_historical_steps :]
                    & predict_mask.unsqueeze(1)
                )
                self.minADE_predict.update(
                    pred=pred_traj,
                    target=data["agent"]["position"][:, self.num_historical_steps :, : pred_traj.shape[-1]],
                    target_valid=target_valid_predict,
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
        if batch_idx < self.n_vis_batch:
            device = pred_traj.device
            scenario_rollouts = get_scenario_rollouts(
                scenario_id=get_scenario_id_int_tensor(data["scenario_id"], device),
                agent_id=data["agent"]["id"],
                agent_batch=data["agent"]["batch"],
                pred_traj=pred_traj,
                pred_z=pred_z,
                pred_head=pred_head,
            )
        return scenario_rollouts

    def _is_self_forced_estimator_warmup_active(self) -> bool:
        """현재 epoch에서 generated estimator만 먼저 적응시킬지 판단합니다."""
        if not self.self_forced_use_distribution_matching_loss:
            return False
        return is_self_forced_estimator_warmup_epoch(
            current_epoch=int(self.current_epoch),
            self_forced_start_epoch=int(self.self_forced_start_epoch),
            estimator_warmup_epochs=int(self.self_forced_estimator_warmup_epochs),
        )

    def _finish_self_forced_estimator_warmup_step(
        self,
        estimator_loss: Tensor | None,
    ) -> Tensor:
        """warmup step을 마무리하고 generator update 없이 반환합니다."""
        self._clear_self_forced_generator_gradients()
        self._clear_self_forced_backward_context()
        if estimator_loss is None:
            detached_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        else:
            detached_loss = estimator_loss.detach().float()

        self.log(
            "train/loss",
            detached_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        self.log(
            "train/self_forced_estimator_warmup/active",
            torch.ones((), device=self.device, dtype=torch.float32),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        self.log(
            "train/self_forced_estimator_warmup/estimator_loss",
            detached_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        return detached_loss

    def _is_self_forced_active(self) -> bool:
        """현재 epoch에서 self-forced NPFM을 사용할지 판단합니다.

        Returns:
            bool: 설정이 켜져 있고 시작 epoch에 도달했으면 ``True`` 입니다.
        """
        return bool(
            self.self_forced_enabled
            and int(self.current_epoch) >= int(self.self_forced_start_epoch)
            and self.self_forced_target_teacher is not None
            and self.self_forced_generated_estimator is not None
        )


    def _apply_self_forced_unfrozen_range(self) -> None:
        """self-forcing에서 학습할 generator / estimator 범위를 적용합니다.

        Returns:
            None

        설명:
            ``except_map_encoder`` 는 기존 ``freeze_map_encoder=true`` 와 같은 의도입니다.
            ``middle`` 은 마지막 flow decoder와 생성부 바로 앞의 마지막 agent 문맥 블록만 엽니다.
            ``full_flow_decoder`` 는 마지막 궤적 생성부만 엽니다.
        """
        if not self.self_forced_enabled:
            return

        apply_self_forced_unfrozen_range(
            self.encoder,
            self.self_forced_unfrozen_range,
        )
        if self.self_forced_generated_estimator is not None:
            apply_self_forced_unfrozen_range(
                self.self_forced_generated_estimator,
                self.self_forced_unfrozen_range,
            )
        if self.self_forced_target_teacher is not None:
            self.self_forced_target_teacher.requires_grad_(False)
        if self.self_forced_generator_ema is not None:
            self.self_forced_generator_ema.requires_grad_(False)

    def _set_self_forced_auxiliary_modes(self) -> None:
        """self-forced 보조 모델의 기본 eval/frozen 상태를 정돈합니다.

        Returns:
            None
        """
        if self.self_forced_target_teacher is None or self.self_forced_generated_estimator is None:
            return
        self.self_forced_target_teacher.requires_grad_(False)
        self.self_forced_target_teacher.eval()
        self.self_forced_generated_estimator.requires_grad_(True)
        self.self_forced_generated_estimator.eval()
        if self.self_forced_generator_ema is not None:
            self.self_forced_generator_ema.requires_grad_(False)
            self.self_forced_generator_ema.eval()
        self._apply_self_forced_unfrozen_range()

    def _copy_online_generator_to_ema(self) -> None:
        """현재 online Generator weight를 EMA Generator에 그대로 복사합니다."""
        if self.self_forced_generator_ema is None:
            return
        self.self_forced_generator_ema.load_state_dict(self.encoder.state_dict())
        self.self_forced_generator_ema.requires_grad_(False)
        self.self_forced_generator_ema.eval()

    def _prepare_self_forced_generator_ema(self) -> None:
        """fit 시작 시 EMA Generator 상태를 checkpoint 상황에 맞게 정돈합니다."""
        if not self.self_forced_enabled or self.self_forced_generator_ema is None:
            return
        if not self._self_forced_generator_ema_loaded_from_checkpoint:
            self._copy_online_generator_to_ema()
            self.self_forced_generator_update_count.zero_()
            self.self_forced_generator_ema_ready.fill_(False)
            return
        self.self_forced_generator_ema.requires_grad_(False)
        self.self_forced_generator_ema.eval()

    def _is_self_forced_generator_ema_ready(self) -> bool:
        """EMA Generator를 eval/test에 사용할 수 있는지 확인합니다."""
        return bool(
            self.self_forced_enabled
            and self.self_forced_generator_ema is not None
            and hasattr(self, "self_forced_generator_ema_ready")
            and bool(self.self_forced_generator_ema_ready.item())
        )

    def _get_eval_generator(self) -> SMARTFlowDecoder:
        """validation/test에서 사용할 Generator를 반환합니다."""
        if self._is_self_forced_generator_ema_ready():
            return self.self_forced_generator_ema
        return self.encoder

    @torch.no_grad()
    def _update_self_forced_generator_ema_after_step(self) -> None:
        """Generator optimizer step 직후 EMA Generator를 갱신합니다."""
        if not self.self_forced_enabled or self.self_forced_generator_ema is None:
            return
        self.self_forced_generator_update_count.add_(1)
        if int(self.self_forced_generator_update_count.item()) < int(self.self_forced_ema_start_step):
            return
        if not bool(self.self_forced_generator_ema_ready.item()):
            self._copy_online_generator_to_ema()
            self.self_forced_generator_ema_ready.fill_(True)
            return

        ema_weight = float(self.self_forced_ema_weight)
        online_state = self.encoder.state_dict()
        ema_state = self.self_forced_generator_ema.state_dict()
        for name, ema_value in ema_state.items():
            online_value = online_state[name].detach().to(device=ema_value.device)
            if torch.is_floating_point(ema_value):
                ema_value.mul_(ema_weight).add_(
                    online_value.to(dtype=ema_value.dtype),
                    alpha=1.0 - ema_weight,
                )
            else:
                ema_value.copy_(online_value.to(dtype=ema_value.dtype))
        self.self_forced_generator_ema.eval()

    @staticmethod
    def _switch_module_to_eval_preserving_modes(module: nn.Module) -> Dict[nn.Module, bool]:
        """autograd는 유지한 채 module을 eval mode로 바꾸고 기존 mode를 기록합니다.

        Args:
            module: eval mode로 잠깐 전환할 모듈입니다.

        Returns:
            Dict[nn.Module, bool]: 각 하위 모듈의 기존 ``training`` 플래그입니다.
        """
        training_modes = {submodule: submodule.training for submodule in module.modules()}
        module.eval()
        return training_modes

    @staticmethod
    def _restore_module_training_modes(training_modes: Dict[nn.Module, bool]) -> None:
        """저장해둔 train/eval mode를 하위 모듈별로 복원합니다.

        Args:
            training_modes: ``_switch_module_to_eval_preserving_modes`` 의 반환값입니다.

        Returns:
            None
        """
        for module, was_training in training_modes.items():
            module.train(was_training)

    def _sync_self_forced_auxiliary_models(self) -> None:
        """Generator weight를 frozen teacher와 generated estimator의 시작점으로 복사합니다.

        설명:
            PDF의 Step 2와 Step 4.1을 코드로 옮긴 함수입니다. 학습 시작 시점에는
            checkpoint가 이미 ``self.encoder`` 로 로드된 뒤이므로, 그 weight를 그대로
            ``F_rho`` 와 ``F_psi`` 의 초기 weight로 씁니다. ``F_rho`` 는 이후 고정하고,
            ``F_psi`` 는 generated self-rollout으로만 online 업데이트합니다.
            단, self-forced checkpoint에서 resume하는 경우에는 checkpoint 안의
            ``F_rho`` / ``F_psi`` state를 보존해야 하므로 재복사하지 않습니다.

        Returns:
            None
        """
        if not self.self_forced_enabled:
            return
        if self.self_forced_target_teacher is None or self.self_forced_generated_estimator is None:
            return
        if self._self_forced_aux_loaded_from_checkpoint:
            self._set_self_forced_auxiliary_modes()
            return
        if not self.self_forced_initialize_aux_on_fit_start:
            return

        encoder_state = self.encoder.state_dict()
        self.self_forced_target_teacher.load_state_dict(encoder_state)
        self.self_forced_generated_estimator.load_state_dict(encoder_state)
        self._set_self_forced_auxiliary_modes()

    @staticmethod
    def _extract_self_forced_generated_estimator_state_dict(
        checkpoint: object,
    ) -> Dict[str, Tensor]:
        """generated estimator state만 다양한 checkpoint 포맷에서 꺼냅니다."""
        if not isinstance(checkpoint, dict):
            raise TypeError(
                "generated-estimator checkpoint must be a dict saved by torch.save()."
            )
        raw_state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(raw_state, dict):
            raise TypeError("generated-estimator checkpoint state_dict must be a dict.")

        prefix = "self_forced_generated_estimator."
        if any(isinstance(key, str) and key.startswith(prefix) for key in raw_state.keys()):
            return {
                key[len(prefix) :]: value
                for key, value in raw_state.items()
                if isinstance(key, str) and key.startswith(prefix)
            }

        return {
            key: value
            for key, value in raw_state.items()
            if isinstance(key, str) and torch.is_tensor(value)
        }

    def _load_self_forced_generated_estimator_bank(self) -> None:
        """W&B bank 등에서 받은 generated estimator state를 보조 모델에 주입합니다."""
        if not self.self_forced_enabled:
            return
        if self.self_forced_generated_estimator is None:
            return
        if not self.self_forced_generated_estimator_init_path:
            return

        init_path = Path(self.self_forced_generated_estimator_init_path)
        if not init_path.is_file():
            raise FileNotFoundError(
                "self_forced.generated_estimator_init_path does not exist: "
                f"{init_path}"
            )

        checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
        state_dict = self._extract_self_forced_generated_estimator_state_dict(checkpoint)
        incompatible = self.self_forced_generated_estimator.load_state_dict(
            state_dict,
            strict=bool(self.self_forced_generated_estimator_init_strict),
        )
        if not self.self_forced_generated_estimator_init_strict:
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])
            if missing or unexpected:
                print(
                    "[self-forced-estimator-bank] non-strict load "
                    f"missing={missing} unexpected={unexpected}"
                )
        self._self_forced_generated_estimator_bank_loaded = True
        if self.self_forced_generated_estimator_skip_warmup_on_load:
            self.self_forced_estimator_warmup_epochs = 0
        self._set_self_forced_auxiliary_modes()
        print(
            "[self-forced-estimator-bank] loaded generated estimator from "
            f"{init_path}; skip_warmup={self.self_forced_generated_estimator_skip_warmup_on_load}"
        )

    def _build_self_forced_generated_estimator_bank_metadata(self) -> Dict[str, Any]:
        """저장/업로드할 generated estimator bank metadata를 구성합니다."""
        target_warmup_epochs = int(
            self.self_forced_generated_estimator_bank_target_warmup_epochs
            or self.self_forced_estimator_warmup_epochs
        )
        return {
            "format_version": 1,
            "state_dict_kind": "self_forced_generated_estimator",
            "state_dict_prefix": "",
            "source_epoch": int(self.current_epoch),
            "global_step": int(getattr(self, "global_step", 0)),
            "self_forced_start_epoch": int(self.self_forced_start_epoch),
            "estimator_warmup_epochs": target_warmup_epochs,
            "remaining_warmup_epochs": int(self.self_forced_estimator_warmup_epochs),
            "loaded_warmup_epochs": int(self.self_forced_generated_estimator_bank_loaded_warmup_epochs),
            "lr": float(self.lr),
            "generated_estimator_lr": float(self.self_forced_generated_estimator_lr),
            "unfrozen_range": str(self.self_forced_unfrozen_range),
            "control_flow": bool(self.use_kinematic_control_flow),
            "flow_window_steps": int(self.flow_window_steps),
        }

    def _save_self_forced_generated_estimator_bank_snapshot(self) -> Path | None:
        """warmup 종료 시점의 generated estimator만 별도 파일로 저장합니다."""
        if not self.self_forced_enabled:
            return None
        if self.self_forced_generated_estimator is None:
            return None
        if self._self_forced_generated_estimator_bank_snapshot_saved:
            return None
        if not self.self_forced_generated_estimator_bank_snapshot_path:
            return None
        if self.trainer is not None and not self.trainer.is_global_zero:
            return None

        snapshot_path = Path(self.self_forced_generated_estimator_bank_snapshot_path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in self.self_forced_generated_estimator.state_dict().items()
            },
            "metadata": self._build_self_forced_generated_estimator_bank_metadata(),
        }
        torch.save(payload, snapshot_path)
        metadata_path = snapshot_path.with_suffix(snapshot_path.suffix + ".metadata.json")
        metadata_path.write_text(
            json.dumps(payload["metadata"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._self_forced_generated_estimator_bank_snapshot_saved = True
        print(f"[self-forced-estimator-bank] saved snapshot: {snapshot_path}")
        return snapshot_path

    def _upload_self_forced_generated_estimator_bank_snapshot(self, snapshot_path: Path) -> None:
        """현재 W&B run에 generated estimator bank artifact를 업로드합니다."""
        if not self.self_forced_generated_estimator_bank_upload_on_warmup_end:
            return
        artifact_name = self.self_forced_generated_estimator_bank_upload_artifact
        if not artifact_name:
            return
        logger = getattr(self, "logger", None)
        experiment = getattr(logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "log_artifact"):
            print(
                "[self-forced-estimator-bank] skip artifact upload: "
                "current logger does not expose W&B log_artifact()."
            )
            return

        try:
            import wandb
        except Exception as exc:  # pragma: no cover - depends on runtime env.
            print(f"[self-forced-estimator-bank] skip artifact upload: {exc}")
            return

        metadata = self._build_self_forced_generated_estimator_bank_metadata()
        artifact = wandb.Artifact(
            name=artifact_name,
            type="generated_estimator_bank",
            metadata=metadata,
        )
        artifact.add_file(snapshot_path.as_posix(), name=snapshot_path.name)
        metadata_path = snapshot_path.with_suffix(snapshot_path.suffix + ".metadata.json")
        if metadata_path.is_file():
            artifact.add_file(metadata_path.as_posix(), name=metadata_path.name)
        aliases = [
            "latest",
            f"warmup{int(metadata['estimator_warmup_epochs'])}",
            f"lr{float(metadata['generated_estimator_lr']):.0e}",
        ]
        experiment.log_artifact(artifact, aliases=aliases)
        print(
            "[self-forced-estimator-bank] logged W&B artifact "
            f"{artifact_name} aliases={aliases}"
        )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """self-forced resume 여부를 기록합니다.

        Args:
            checkpoint: Lightning checkpoint dictionary입니다.

        Returns:
            None
        """
        self._assert_motion_missingness_checkpoint_compatible(checkpoint)
        state_dict = checkpoint.get("state_dict", {})
        has_target_teacher = any(
            key.startswith("self_forced_target_teacher.") for key in state_dict
        )
        has_generated_estimator = any(
            key.startswith("self_forced_generated_estimator.") for key in state_dict
        )
        has_generator_ema = any(
            key.startswith("self_forced_generator_ema.") for key in state_dict
        )
        self._self_forced_aux_loaded_from_checkpoint = bool(
            self.self_forced_enabled and has_target_teacher and has_generated_estimator
        )
        self._self_forced_generator_ema_loaded_from_checkpoint = bool(
            self.self_forced_enabled and has_generator_ema
        )

    def _assert_motion_missingness_checkpoint_compatible(self, checkpoint: Dict[str, Any]) -> None:
        """motion missingness 입력 차원과 맞지 않는 예전 checkpoint를 명확히 거부합니다."""
        state_dict = checkpoint.get("state_dict", {})
        if not isinstance(state_dict, dict):
            return
        current_state = self.state_dict()
        guarded_keys = [
            "encoder.agent_encoder.x_a_emb.freqs.weight",
            "encoder.agent_encoder.r_a2a_emb.freqs.weight",
        ]
        mismatches: list[str] = []
        for key in guarded_keys:
            checkpoint_value = state_dict.get(key)
            current_value = current_state.get(key)
            if checkpoint_value is None or current_value is None:
                continue
            if tuple(checkpoint_value.shape) != tuple(current_value.shape):
                mismatches.append(
                    f"{key}: checkpoint={tuple(checkpoint_value.shape)}, "
                    f"current={tuple(current_value.shape)}"
                )
        if mismatches:
            raise RuntimeError(
                "Motion Missingness Feature changes flow context input dimensions and "
                "requires a fresh pretrain checkpoint. Incompatible checkpoint tensors: "
                + "; ".join(mismatches)
            )

    def _manual_backward_without_autocast(self, loss: Tensor) -> None:
        """manual optimization의 backward만 autocast 밖에서 실행합니다.

        Args:
            loss: backward를 수행할 scalar loss입니다.

        Returns:
            None

        설명:
            ``loss.float()`` 으로 fp32 캐스팅을 유지합니다. ``precision='16-mixed'`` 인
            경우 Lightning의 precision plugin이 ``manual_backward`` 안에서
            ``GradScaler.scale`` 을 적용하므로, 이후 step은
            ``_clip_and_step_with_optional_scaler`` 를 통해 unscale → clip → step → update
            순서를 지킵니다.
        """
        with torch.autocast(device_type=loss.device.type, enabled=False):
            self.manual_backward(loss.float())

    def _get_amp_grad_scaler(self) -> Any | None:
        """fp16 mixed precision에서 Lightning이 만든 GradScaler를 가져옵니다.

        Returns:
            Any | None: ``precision='16-mixed'`` 일 때 ``torch.amp.GradScaler``,
            그 외(``bf16-mixed`` / ``32-true``)에는 ``None``.

        설명:
            manual optimization은 Lightning의 ``optimizer_step`` 경로를 사용하지 않으므로
            scaler의 unscale/step/update를 우리가 직접 호출해야 합니다.
        """
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None
        plugin = getattr(trainer, "precision_plugin", None)
        if plugin is None:
            return None
        return getattr(plugin, "scaler", None)

    def _clip_and_step_with_optional_scaler(
        self,
        optimizer,
        *,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str = "norm",
    ) -> None:
        """unscale → clip → step → update 순서로 fp16-safe하게 step을 수행합니다.

        Args:
            optimizer: step 대상 optimizer.
            gradient_clip_val: gradient clip threshold. ``None`` 이면 clipping 생략합니다.
            gradient_clip_algorithm: clip 알고리즘 ("norm" 또는 "value").

        Returns:
            None.

        설명:
            ``GradScaler`` 가 활성이면 ``scaler.unscale_`` 으로 gradient를 정상 스케일로
            돌린 뒤 clip을 적용하고, ``scaler.step`` 으로 inf/NaN을 자동 감지·skip하며
            ``scaler.update`` 로 scale factor를 갱신합니다. scaler가 없으면 평문 경로로
            동일한 의미를 유지합니다.
        """
        scaler = self._get_amp_grad_scaler()
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        if scaler is not None:
            scaler.unscale_(raw_optimizer)
        if gradient_clip_val is not None:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm,
            )
        if scaler is not None:
            scaler.step(raw_optimizer)
            scaler.update()
        else:
            optimizer.step()

    def _clear_self_forced_auxiliary_gradients(self) -> None:
        """self-forcing 보조 모델의 gradient를 비웁니다.

        Args:
            없음.

        Returns:
            None.

        설명:
            target teacher와 generated estimator는 Generator update에서 평가자 역할만 해야
            합니다. update 경계마다 두 보조 모델의 gradient를 지워서 이전 단계의 값이 다음
            검사에 섞이지 않게 합니다.
        """
        if not self.self_forced_enabled:
            return
        clear_module_gradients(self.self_forced_target_teacher)
        clear_module_gradients(self.self_forced_generated_estimator)

    def _clear_self_forced_generator_gradients(self) -> None:
        """online Generator의 gradient를 비웁니다.

        Args:
            없음.

        Returns:
            None.

        설명:
            generated estimator update는 detached rollout만 학습해야 하므로 Generator에
            gradient가 남아 있으면 안 됩니다. update가 끝난 뒤와 estimator backward 직전에
            Generator gradient를 비웁니다.
        """
        if not self.self_forced_enabled:
            return
        clear_module_gradients(self.encoder)

    def _prepare_self_forced_generator_backward_boundary(self) -> None:
        """Generator backward 직전에 보조 모델 gradient를 초기화합니다.

        Args:
            없음.

        Returns:
            None.

        설명:
            Generator loss backward 뒤에 생긴 gradient만 검사하기 위해, backward 직전에
            target teacher와 generated estimator의 이전 gradient를 모두 지웁니다.
        """
        self._clear_self_forced_auxiliary_gradients()

    def _prepare_self_forced_estimator_backward_boundary(self) -> None:
        """generated estimator backward 직전에 Generator gradient를 초기화합니다.

        Args:
            없음.

        Returns:
            None.

        설명:
            estimator loss backward 뒤에 Generator gradient가 새로 생겼는지만 확인하기 위해,
            backward 직전에 online Generator와 target teacher의 gradient를 지웁니다.
        """
        self._clear_self_forced_generator_gradients()
        clear_module_gradients(self.self_forced_target_teacher)

    def _assert_self_forced_generator_update_isolated(self) -> None:
        """Generator update가 보조 모델을 학습하지 않았는지 검사합니다.

        Args:
            없음.

        Returns:
            None.

        Raises:
            RuntimeError: target teacher나 generated estimator에 gradient가 생기면 발생합니다.

        설명:
            clean-DMD 방향은 Generator를 움직이는 고정 목표여야 합니다. 이 검사에 실패하면
            Generator loss graph 안에서 보조 모델이 함께 학습되고 있다는 뜻입니다.
        """
        if not self.self_forced_enabled:
            return
        assert_no_module_gradients(self.self_forced_target_teacher, "self_forced_target_teacher", "generator update")
        assert_no_module_gradients(self.self_forced_generated_estimator, "self_forced_generated_estimator", "generator update")

    def _assert_self_forced_estimator_update_isolated(self) -> None:
        """generated estimator update가 Generator를 학습하지 않았는지 검사합니다.

        Args:
            없음.

        Returns:
            None.

        Raises:
            RuntimeError: online Generator나 target teacher에 gradient가 생기면 발생합니다.

        설명:
            generated estimator는 현재 Generator가 만든 detached closed-loop path를 설명하는
            모델입니다. 이 update에서 Generator에 gradient가 생기면 DMD의 분리 원칙이 깨집니다.
        """
        if not self.self_forced_enabled:
            return
        assert_no_module_gradients(self.encoder, "online Generator", "generated-estimator update")
        assert_no_module_gradients(self.self_forced_target_teacher, "self_forced_target_teacher", "generated-estimator update")

    def _set_token_processor_training_mode(self, is_training: bool) -> None:
        """token processor의 train/eval 상태를 안전하게 바꿉니다.

        Args:
            is_training: ``True`` 면 train mode, ``False`` 면 eval mode로 둡니다.

        Returns:
            None
        """
        if is_training:
            self.token_processor.train()
        else:
            self.token_processor.eval()

    def _build_eval_tokenized_inputs(self, data) -> tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """self-rollout 학습에 사용할 평가 모드 token을 만듭니다.

        설명:
            self-forced rollout은 실제 inference와 같은 agent selection과 0.5초 commit/update
            규칙을 써야 합니다. 그래서 open-loop anchor 학습과 별도로 token processor를
            잠깐 eval mode로 바꿔 평가용 token을 만든 뒤, 원래 mode로 되돌립니다.

        Args:
            data: 학습 batch입니다.

        Returns:
            tuple[Dict[str, Tensor], Dict[str, Tensor]]: map token과 agent token입니다.
        """
        was_training = self.token_processor.training
        self._set_token_processor_training_mode(False)
        tokenized_map, tokenized_agent = self.token_processor(data)
        self._set_token_processor_training_mode(was_training)
        return tokenized_map, tokenized_agent

    def _get_self_forced_rollout_steps_2hz(self) -> int:
        """flow_window_steps에 맞춘 0.5초 commit block 수를 계산합니다.

        Returns:
            int: ``flow_window_steps / 5`` 로 얻은 N초 self-rollout block 수입니다.
        """
        if self.flow_window_steps % 5 != 0:
            raise ValueError(
                "self-forced NPFM assumes flow_window_steps is divisible by 5, "
                f"got {self.flow_window_steps}."
            )
        return max(1, int(self.flow_window_steps // 5))

    def _sample_flow_state_from_clean(self, clean_path_norm: Tensor):
        """현재 Generator의 flow path 규칙으로 전체 tau 구간의 noisy path를 만듭니다.

        Args:
            clean_path_norm: clean path입니다. shape은 ``[n_agent_valid, F_win, 4]`` 입니다.

        Returns:
            FlowSample: ``x_t``, ``target``, ``tau`` 를 담은 flow sample입니다.
                tau는 rollout을 만들 때 사용한 random terminal step과 무관하게
                flow ODE의 기본 전체 구간에서 새로 뽑힙니다.
        """
        return self.encoder.agent_encoder.flow_ode.sample(
            clean_path_norm,
            target_type="velocity",
        )

    def _can_cache_self_forced_map_feature(self, decoder: SMARTFlowDecoder) -> bool:
        """self-forced step 안에서 decoder map feature를 재사용해도 되는지 확인합니다."""
        if not self.self_forced_cache_frozen_map_features:
            return False
        map_encoder = getattr(decoder, "map_encoder", None)
        if map_encoder is None:
            return False
        return not any(parameter.requires_grad for parameter in map_encoder.parameters())

    def _encode_self_forced_map_feature(
        self,
        decoder: SMARTFlowDecoder,
        tokenized_map: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """frozen map encoder 출력을 self-forced step cache용으로 만듭니다."""
        with torch.no_grad():
            map_feature = decoder.encode_map(tokenized_map)
        return detach_tensor_tree(map_feature)

    def _predict_path_flow_clean_estimate(
        self,
        decoder: SMARTFlowDecoder,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        noisy_path_norm: Tensor,
        tau: Tensor,
        anchor_mask: Tensor,
        map_feature: Dict[str, Tensor] | None = None,
    ) -> Dict[str, Tensor]:
        """주어진 decoder가 noisy N초 path를 어떻게 clean path로 보는지 계산합니다.

        Args:
            decoder: ``F_rho`` 또는 ``F_psi`` 역할의 decoder입니다.
            tokenized_map: 평가 모드 map token 사전입니다.
            tokenized_agent: 평가 모드 agent token 사전입니다.
            noisy_path_norm: noisy path입니다. shape은 ``[n_valid_agent, F_win, 4]`` 입니다.
            tau: flow interpolation time입니다. shape은 ``[n_valid_agent]`` 입니다.
            anchor_mask: 첫 anchor에서 사용할 agent mask입니다. shape은 ``[n_agent]`` 입니다.
            map_feature: 이미 계산한 지도 특징입니다. 값이 있으면 ``tokenized_map`` 으로
                다시 map encoder를 호출하지 않습니다.

        Returns:
            Dict[str, Tensor]: ``velocity`` 와 ``clean`` 을 담은 사전입니다.
        """
        if map_feature is None:
            map_feature = decoder.encode_map(tokenized_map)
        return decoder.path_flow_velocity_for_anchor0(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            path_noisy_norm=noisy_path_norm,
            tau=tau,
            anchor_mask=anchor_mask,
        )

    def _build_self_forced_zero_metrics(self, reference: Tensor) -> Dict[str, Tensor]:
        """self-forced logging에 필요한 0 metric 사전을 만듭니다.

        Args:
            reference: device와 dtype을 맞출 기준 텐서입니다.

        Returns:
            Dict[str, Tensor]: self-forced loss 관련 0 scalar 사전입니다.
        """
        zero = reference.new_zeros(())
        metric_dict = {
            "sf_loss": zero,
            "gen_estimator_loss": zero,
            "anchor_loss": zero,
            "total_loss": zero,
        }
        return metric_dict

    def _run_self_forced_rollout(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """실제 inference와 같은 규칙으로 N초 committed self-rollout을 만듭니다.

        Args:
            tokenized_map: 평가 모드 map token 사전입니다.
            tokenized_agent: 평가 모드 agent token 사전입니다.

        Returns:
            Dict[str, Tensor]: closed-loop rollout 결과입니다. ``pred_traj_10hz`` 와
            ``pred_head_10hz`` 는 실제로 commit된 N초 rollout입니다. random-s 학습이 켜져
            있으면 DDP 전체 rank가 공유한 ``s`` 와 tau 구간도 함께 들어갑니다.
        """
        encoder_modes = self._switch_module_to_eval_preserving_modes(self.encoder)
        try:
            map_feature = self.encoder.encode_map(tokenized_map)
            rollout_cache = self.encoder.prepare_training_rollout_cache(tokenized_agent, map_feature)
            return self.encoder.training_rollout_from_cache(
                rollout_cache=rollout_cache,
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
                sampling_scheme=self.self_forced_sampling,
                rollout_steps_2hz=self._get_self_forced_rollout_steps_2hz(),
                self_forced_epoch=int(self.current_epoch),
                detach_block_transition=self.self_forced_detach_block_transition,
                use_stop_motion=self.self_forced_use_stop_motion,
            )
        finally:
            self._restore_module_training_modes(encoder_modes)

    def _pack_self_forced_committed_rollout(
        self,
        rollout: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """committed rollout을 첫 anchor 기준 packed N초 flow state로 변환합니다.

        Args:
            rollout: ``_run_self_forced_rollout`` 의 출력입니다.
            tokenized_agent: 평가 모드 agent token 사전입니다.

        Returns:
            tuple[Tensor, Tensor]: packed flow state와 agent mask입니다.
                pose-space shape은 ``[n_valid_agent, F_win, 4]`` 이고,
                control-space shape은 ``[n_valid_agent, F_win, 3]`` 이며,
                mask shape은 ``[n_agent]`` 입니다.

        Notes:
            random terminal N은 self-rollout을 어디에서 끊을지만 정합니다.
            이후 generated estimator 학습과 generator update의 noising tau는
            여기서 전달하지 않습니다.
        """
        anchor_mask = get_anchor0_valid_mask(tokenized_agent)
        committed_path_norm = build_anchor0_normalized_committed_path(
            pred_traj_10hz=rollout["pred_traj_10hz"],
            pred_head_10hz=rollout["pred_head_10hz"],
            tokenized_agent=tokenized_agent,
            flow_window_steps=self.flow_window_steps,
        )
        packed_path_norm = committed_path_norm[anchor_mask]
        if self.use_kinematic_control_flow:
            packed_path_norm = build_anchor0_normalized_committed_control(
                committed_path_norm=packed_path_norm,
                tokenized_agent=tokenized_agent,
                anchor_mask=anchor_mask,
                pos_scale_m=self.encoder.agent_encoder.control_pos_scale_m,
                vehicle_yaw_scale_rad=self.encoder.agent_encoder.control_vehicle_yaw_scale_rad,
                pedestrian_yaw_scale_rad=self.encoder.agent_encoder.control_pedestrian_yaw_scale_rad,
                cyclist_yaw_scale_rad=self.encoder.agent_encoder.control_cyclist_yaw_scale_rad,
                use_holonomic_model_only=self.encoder.agent_encoder.use_holonomic_model_only,
                use_rolling_supervision=self.encoder.agent_encoder.use_rolling_supervision,
                vehicle_no_slip_point_ratio=self.encoder.agent_encoder.control_vehicle_no_slip_point_ratio,
                cyclist_no_slip_point_ratio=self.encoder.agent_encoder.control_cyclist_no_slip_point_ratio,
            )
        return packed_path_norm, anchor_mask

    def _update_generated_path_flow_estimator(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        committed_path_norm: Tensor,
        anchor_mask: Tensor,
        *,
        has_committed_path_global: bool | None = None,
    ) -> Tensor:
        """detached self-rollout으로 generated estimator F_psi를 online 업데이트합니다.

        Args:
            tokenized_map: 평가 모드 map token 사전입니다.
            tokenized_agent: 평가 모드 agent token 사전입니다.
            committed_path_norm: Generator가 실제로 실행한 N초 self-forced flow state입니다.
                pose-space에서는 ``[n_valid_agent, F_win, 4]`` 이고,
                control-space에서는 ``[n_valid_agent, F_win, 3]`` 입니다.
            anchor_mask: 첫 anchor에서 사용할 agent mask입니다.
                shape은 ``[n_agent]`` 입니다.
            has_committed_path_global: DDP 전체 rank 기준으로 self-forced path가 하나라도
                있는지입니다. 값이 없으면 이 함수 안에서 동기화합니다.

        Returns:
            Tensor: 마지막 estimator update의 flow matching loss입니다.

        Notes:
            noising tau는 random terminal N과 독립적으로 전체 tau 구간에서 샘플링합니다.
        """
        if self.self_forced_generated_estimator is None:
            raise RuntimeError("self_forced_generated_estimator is not initialized.")
        if self.self_forced_target_teacher is None:
            raise RuntimeError("self_forced_target_teacher is not initialized.")

        optimizer = self.optimizers()[1]
        last_loss = committed_path_norm.new_zeros(())
        has_committed_path_local = committed_path_norm.numel() > 0
        if has_committed_path_global is None:
            has_committed_path_global = self._sync_distributed_bool_any(
                has_committed_path_local,
                device=committed_path_norm.device,
            )
        if not has_committed_path_global:
            return last_loss.detach()

        clean_path = committed_path_norm.detach().clone()
        estimator_tokenized_map = detach_tensor_tree(tokenized_map)
        estimator_tokenized_agent = detach_tensor_tree(tokenized_agent)
        estimator_anchor_mask = anchor_mask.detach()

        self.toggle_optimizer(optimizer)
        self.self_forced_target_teacher.eval()
        self.self_forced_generated_estimator.train()
        try:
            with module_gradients_disabled(self.encoder, self.self_forced_target_teacher):
                estimator_map_feature = None
                if has_committed_path_local and self._can_cache_self_forced_map_feature(
                    self.self_forced_generated_estimator,
                ):
                    estimator_map_feature = self._encode_self_forced_map_feature(
                        decoder=self.self_forced_generated_estimator,
                        tokenized_map=estimator_tokenized_map,
                    )
                for _ in range(self.self_forced_estimator_updates_per_step):
                    optimizer.zero_grad(set_to_none=True)
                    self._prepare_self_forced_estimator_backward_boundary()
                    if has_committed_path_local:
                        with torch.no_grad():
                            flow_sample = self.self_forced_generated_estimator.agent_encoder.flow_ode.sample(
                                clean_path,
                                target_type="velocity",
                            )
                        noisy_path_norm = flow_sample.x_t.detach()
                        tau = flow_sample.tau.detach()
                        flow_target = flow_sample.target.detach()
                        pred_dict = self._predict_path_flow_clean_estimate(
                            decoder=self.self_forced_generated_estimator,
                            tokenized_map=estimator_tokenized_map,
                            tokenized_agent=estimator_tokenized_agent,
                            noisy_path_norm=noisy_path_norm,
                            tau=tau,
                            anchor_mask=estimator_anchor_mask,
                            map_feature=estimator_map_feature,
                        )
                        last_loss = flow_matching_loss(pred_dict["velocity"], flow_target)
                    else:
                        last_loss = self._build_trainable_connected_zero_loss(
                            self.self_forced_generated_estimator,
                        )
                    self._manual_backward_without_autocast(last_loss)
                    self._assert_self_forced_estimator_update_isolated()
                    self._clip_and_step_with_optional_scaler(
                        optimizer,
                        gradient_clip_val=self.self_forced_gradient_clip_val,
                        gradient_clip_algorithm="norm",
                    )
                    self._clear_self_forced_auxiliary_gradients()
                    self._clear_self_forced_generator_gradients()
        finally:
            self._clear_self_forced_auxiliary_gradients()
            self._clear_self_forced_generator_gradients()
            self.untoggle_optimizer(optimizer)
            self._set_self_forced_auxiliary_modes()
        return last_loss.detach()

    def _compute_self_forced_direction(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        committed_path_norm: Tensor,
        anchor_mask: Tensor,
        active_control_mask: Tensor | None = None,
        dmd_injection_scale: float | Tensor = 1.0,
    ) -> Tensor:
        """clean-DMD 방향을 고정된 평가자 출력으로 계산합니다.

        Args:
            tokenized_map: map token 사전입니다.
            tokenized_agent: agent token 사전입니다.
            committed_path_norm: Generator가 closed-loop로 실제 실행한 self-forced flow state입니다.
                pose-space에서는 ``[n_valid_agent, flow_window_steps, 4]`` 이고,
                control-space에서는 ``[n_valid_agent, flow_window_steps, 3]`` 입니다.
            anchor_mask: 첫 anchor 기준으로 유효한 agent mask입니다.
                shape은 ``[n_agent]`` 입니다.
            active_control_mask: DMD에 사용할 active 축 mask입니다. shape은
                ``[n_valid_agent, 1, flow_dim]`` 입니다.
            dmd_injection_scale: detached target에 주입할 DMD 방향 계수입니다. pose-projected
                DMD에서는 pose target을 만든 뒤 control target으로 되돌리는 데 사용합니다.

        Returns:
            Tensor: 현재 committed path에 더할 정규화된 DMD 방향입니다.
            shape은 ``committed_path_norm`` 과 같습니다.

        설명:
            Generator update에서 target teacher와 generated estimator는 학습 대상이 아닙니다.
            두 모델은 같은 noisy path를 보고 clean path 추정을 내는 평가자로만 쓰입니다.
            그래서 모든 보조 모델 호출은 ``no_grad``로 감싸고, 최종 방향도 detach합니다.
        """
        if self.self_forced_target_teacher is None or self.self_forced_generated_estimator is None:
            raise RuntimeError("self-forced auxiliary models are not initialized.")

        stable_scale_group_index = self._build_self_forced_dmd_stable_scale_group_index(
            tokenized_agent=tokenized_agent,
            anchor_mask=anchor_mask,
            device=committed_path_norm.device,
        )
        self.self_forced_target_teacher.eval()
        self.self_forced_generated_estimator.eval()
        self._clear_self_forced_auxiliary_gradients()
        with torch.no_grad():
            clean_for_guidance = committed_path_norm.detach()
            flow_sample = self.encoder.agent_encoder.flow_ode.sample(
                clean_for_guidance,
                target_type="velocity",
                tau_low=self.self_forced_guidance_tau_low,
                tau_high=self.self_forced_guidance_tau_high,
            )
            target_pred = self._predict_path_flow_clean_estimate(
                decoder=self.self_forced_target_teacher,
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
                noisy_path_norm=flow_sample.x_t,
                tau=flow_sample.tau,
                anchor_mask=anchor_mask,
            )
            generated_pred = self._predict_path_flow_clean_estimate(
                decoder=self.self_forced_generated_estimator,
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
                noisy_path_norm=flow_sample.x_t,
                tau=flow_sample.tau,
                anchor_mask=anchor_mask,
            )
            if self._should_project_self_forced_dmd_to_pose_space(committed_path_norm):
                path_delta = self._compute_pose_projected_self_forced_control_delta(
                    tokenized_agent=tokenized_agent,
                    committed_control_norm=clean_for_guidance,
                    target_clean_control_norm=target_pred["clean"],
                    generated_clean_control_norm=generated_pred["clean"],
                    anchor_mask=anchor_mask,
                    stable_scale_group_index=stable_scale_group_index,
                    dmd_injection_scale=dmd_injection_scale,
                )
            else:
                path_delta = build_clean_dmd_direction(
                    committed_path_norm=clean_for_guidance,
                    target_clean_norm=target_pred["clean"],
                    generated_clean_norm=generated_pred["clean"],
                    active_mask=active_control_mask,
                    stable_scale_group_index=stable_scale_group_index,
                    normalizer_eps=self.self_forced_direction_normalizer_eps,
                    beta=self.self_forced_dmd_beta,
                    use_stable_scale_filter=self.self_forced_dmd_use_stable_scale_filter,
                    use_teacher_alignment_filter=self.self_forced_dmd_use_teacher_alignment_filter,
                    use_trust_region_filter=self.self_forced_dmd_use_trust_region_filter,
                )

        self._assert_self_forced_generator_update_isolated()
        return path_delta.to(dtype=committed_path_norm.dtype).detach()

    def _should_project_self_forced_dmd_to_pose_space(self, committed_path_norm: Tensor) -> bool:
        """현재 self-forced DMD를 pose-space에서 판단해야 하는지 확인합니다."""
        return (
            bool(self.self_forced_project_dmd_to_pose_space)
            and bool(self.use_kinematic_control_flow)
            and committed_path_norm.ndim == 3
            and int(committed_path_norm.shape[-1]) == 3
        )

    def _get_anchor0_agent_type_and_length(
        self,
        tokenized_agent: Dict[str, Tensor],
        anchor_mask: Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor | None]:
        """첫 anchor에 남은 agent의 type/length metadata를 가져옵니다."""
        if anchor_mask.ndim != 1:
            raise ValueError(f"anchor_mask must have shape [n_agent], got {tuple(anchor_mask.shape)}.")
        if "type" not in tokenized_agent:
            raise KeyError("tokenized_agent must contain type for self-forced control metadata.")
        agent_type = tokenized_agent["type"][anchor_mask].to(device=device)
        agent_length = (
            tokenized_agent["shape"][anchor_mask, 0].to(device=device, dtype=dtype)
            if "shape" in tokenized_agent
            else None
        )
        return agent_type, agent_length

    def _build_self_forced_dmd_stable_scale_group_index(
        self,
        tokenized_agent: Dict[str, Tensor],
        anchor_mask: Tensor,
        *,
        device: torch.device,
    ) -> Tensor | None:
        """config에 따라 DMD stable scale을 공유할 agent group id를 만듭니다."""
        scope = str(self.self_forced_dmd_stable_scale_scope)
        if scope == "agent":
            return None
        if anchor_mask.ndim != 1:
            raise ValueError(f"anchor_mask must have shape [n_agent], got {tuple(anchor_mask.shape)}.")
        if "batch" not in tokenized_agent:
            raise KeyError("tokenized_agent must contain batch for grouped DMD stable scale.")
        selected_batch = tokenized_agent["batch"][anchor_mask].to(device=device, dtype=torch.long)
        if scope == "scene":
            return selected_batch
        if scope == "type":
            if "type" not in tokenized_agent:
                raise KeyError("tokenized_agent must contain type for type-grouped DMD stable scale.")
            selected_type = tokenized_agent["type"][anchor_mask].to(device=device, dtype=torch.long)
            return selected_batch * int(CYCLIST_TYPE_ID + 1) + selected_type
        raise ValueError(
            "self_forced.dmd_stable_scale_scope must be one of 'agent', 'type', or 'scene', "
            f"got {scope!r}."
        )

    def _control_norm_to_self_forced_pose_norm(
        self,
        control_norm: Tensor,
        tokenized_agent: Dict[str, Tensor],
        anchor_mask: Tensor,
    ) -> Tensor:
        """self-forced control state를 closed-loop metric과 같은 pose-space 표현으로 복원합니다."""
        agent_type, agent_length = self._get_anchor0_agent_type_and_length(
            tokenized_agent,
            anchor_mask,
            device=control_norm.device,
            dtype=control_norm.dtype,
        )
        if agent_type.shape[0] != control_norm.shape[0]:
            raise ValueError(
                "anchor_mask selected agent count must match control batch: "
                f"got {agent_type.shape[0]} and {control_norm.shape[0]}."
            )
        return self.encoder.flow_norm_to_pose_metric_norm(
            value=control_norm,
            agent_type=agent_type,
            agent_length=agent_length,
        )

    def _pose_norm_to_self_forced_control_norm(
        self,
        pose_norm: Tensor,
        tokenized_agent: Dict[str, Tensor],
        anchor_mask: Tensor,
    ) -> Tensor:
        """pose-space DMD target을 기존 rolling control target으로 되돌립니다."""
        return build_anchor0_normalized_committed_control(
            committed_path_norm=pose_norm,
            tokenized_agent=tokenized_agent,
            anchor_mask=anchor_mask,
            pos_scale_m=self.encoder.agent_encoder.control_pos_scale_m,
            vehicle_yaw_scale_rad=self.encoder.agent_encoder.control_vehicle_yaw_scale_rad,
            pedestrian_yaw_scale_rad=self.encoder.agent_encoder.control_pedestrian_yaw_scale_rad,
            cyclist_yaw_scale_rad=self.encoder.agent_encoder.control_cyclist_yaw_scale_rad,
            use_holonomic_model_only=self.encoder.agent_encoder.use_holonomic_model_only,
            use_rolling_supervision=self.encoder.agent_encoder.use_rolling_supervision,
            vehicle_no_slip_point_ratio=self.encoder.agent_encoder.control_vehicle_no_slip_point_ratio,
            cyclist_no_slip_point_ratio=self.encoder.agent_encoder.control_cyclist_no_slip_point_ratio,
        )

    def _compute_pose_projected_self_forced_control_delta(
        self,
        *,
        tokenized_agent: Dict[str, Tensor],
        committed_control_norm: Tensor,
        target_clean_control_norm: Tensor,
        generated_clean_control_norm: Tensor,
        anchor_mask: Tensor,
        stable_scale_group_index: Tensor | None,
        dmd_injection_scale: float | Tensor,
    ) -> Tensor:
        """pose-space에서 DMD target을 만들고 rolling control delta로 되돌립니다."""
        committed_pose_norm = self._control_norm_to_self_forced_pose_norm(
            committed_control_norm,
            tokenized_agent,
            anchor_mask,
        )
        target_pose_norm = self._control_norm_to_self_forced_pose_norm(
            target_clean_control_norm,
            tokenized_agent,
            anchor_mask,
        )
        generated_pose_norm = self._control_norm_to_self_forced_pose_norm(
            generated_clean_control_norm,
            tokenized_agent,
            anchor_mask,
        )
        pose_delta = build_clean_dmd_direction(
            committed_path_norm=committed_pose_norm,
            target_clean_norm=target_pose_norm,
            generated_clean_norm=generated_pose_norm,
            active_mask=None,
            stable_scale_group_index=stable_scale_group_index,
            normalizer_eps=self.self_forced_direction_normalizer_eps,
            beta=self.self_forced_dmd_beta,
            use_stable_scale_filter=self.self_forced_dmd_use_stable_scale_filter,
            use_teacher_alignment_filter=self.self_forced_dmd_use_teacher_alignment_filter,
            use_trust_region_filter=self.self_forced_dmd_use_trust_region_filter,
        )
        if isinstance(dmd_injection_scale, Tensor):
            injection_scale = dmd_injection_scale.to(
                device=pose_delta.device,
                dtype=pose_delta.dtype,
            )
        else:
            injection_scale = torch.as_tensor(
                float(dmd_injection_scale),
                device=pose_delta.device,
                dtype=pose_delta.dtype,
            )
        pose_target_norm = normalize_pose_heading_vector(
            committed_pose_norm + pose_delta * injection_scale,
        )
        control_target_norm = self._pose_norm_to_self_forced_control_norm(
            pose_target_norm,
            tokenized_agent,
            anchor_mask,
        )
        if tuple(control_target_norm.shape) != tuple(committed_control_norm.shape):
            raise ValueError(
                "pose-projected DMD control target shape must match committed control shape: "
                f"target={tuple(control_target_norm.shape)}, committed={tuple(committed_control_norm.shape)}."
            )
        return control_target_norm - committed_control_norm

    def _build_self_forced_active_control_mask(
        self,
        tokenized_agent: Dict[str, Tensor],
        committed_path_norm: Tensor,
        anchor_mask: Tensor,
    ) -> Tensor:
        """self-forced DMD가 실행 가능한 control 축에만 작동하도록 mask를 만듭니다."""
        if anchor_mask.ndim != 1:
            raise ValueError(f"anchor_mask must have shape [n_agent], got {tuple(anchor_mask.shape)}.")
        if "type" not in tokenized_agent:
            raise KeyError("tokenized_agent must contain type for self-forced active-control DMD.")
        agent_type = tokenized_agent["type"][anchor_mask].to(device=committed_path_norm.device)
        if agent_type.shape[0] != committed_path_norm.shape[0]:
            raise ValueError(
                "anchor_mask selected agent count must match committed_path_norm batch: "
                f"got {agent_type.shape[0]} and {committed_path_norm.shape[0]}."
            )
        return build_active_control_mask(
            agent_type=agent_type,
            flow_dim=int(committed_path_norm.shape[-1]),
            device=committed_path_norm.device,
            dtype=committed_path_norm.dtype,
            use_kinematic_control_flow=bool(self.use_kinematic_control_flow),
            use_holonomic_model_only=bool(self.encoder.agent_encoder.use_holonomic_model_only),
        )


    def _sample_self_forced_guidance_flow_state(self, clean_path_norm: Tensor):
        """SiD/DMD teacher query에 쓸 noisy path를 샘플링합니다.

        Args:
            clean_path_norm: Generator가 만든 clean flow state입니다.
                pose-space에서는 ``[n_valid_agent, flow_window_steps, 4]`` 이고,
                control-space에서는 ``[n_valid_agent, flow_window_steps, 3]`` 입니다.

        Returns:
            object: ``x_t`` 와 ``tau`` 를 가진 flow sample입니다.
        """
        try:
            return self.encoder.agent_encoder.flow_ode.sample(
                clean_path_norm,
                target_type="velocity",
                tau_low=self.self_forced_guidance_tau_low,
                tau_high=self.self_forced_guidance_tau_high,
            )
        except TypeError:
            return self.encoder.agent_encoder.flow_ode.sample(
                clean_path_norm,
                target_type="velocity",
            )

    def _predict_self_forced_teacher_estimator_clean_paths(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        committed_path_norm: Tensor,
        anchor_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """같은 noisy path에서 teacher와 generated estimator의 clean 예측을 구합니다.

        Args:
            tokenized_map: 평가 모드 map token 사전입니다.
            tokenized_agent: 평가 모드 agent token 사전입니다.
            committed_path_norm: Generator가 실제로 실행한 self-forced flow state입니다.
                pose-space에서는 ``[n_valid_agent, flow_window_steps, 4]`` 이고,
                control-space에서는 ``[n_valid_agent, flow_window_steps, 3]`` 입니다.
            anchor_mask: 첫 anchor에서 사용할 agent mask입니다.
                shape은 ``[n_agent]`` 입니다.

        Returns:
            tuple[Tensor, Tensor]: ``target_clean_norm`` 과 ``generated_clean_norm`` 입니다.
                각 shape은 ``committed_path_norm`` 과 같습니다.
        """
        if self.self_forced_target_teacher is None or self.self_forced_generated_estimator is None:
            raise RuntimeError("self-forced auxiliary models are not initialized.")

        self.self_forced_target_teacher.eval()
        self.self_forced_generated_estimator.eval()
        if hasattr(self, "_clear_self_forced_auxiliary_gradients"):
            self._clear_self_forced_auxiliary_gradients()

        with torch.no_grad():
            clean_for_guidance = committed_path_norm.detach()
            flow_sample = self._sample_self_forced_guidance_flow_state(clean_for_guidance)
            target_pred = self._predict_path_flow_clean_estimate(
                decoder=self.self_forced_target_teacher,
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
                noisy_path_norm=flow_sample.x_t,
                tau=flow_sample.tau,
                anchor_mask=anchor_mask,
            )
            generated_pred = self._predict_path_flow_clean_estimate(
                decoder=self.self_forced_generated_estimator,
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
                noisy_path_norm=flow_sample.x_t,
                tau=flow_sample.tau,
                anchor_mask=anchor_mask,
            )

        if hasattr(self, "_assert_self_forced_generator_update_isolated"):
            self._assert_self_forced_generator_update_isolated()
        return target_pred["clean"].detach(), generated_pred["clean"].detach()

    def _compute_self_forced_sid_loss(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        committed_path_norm: Tensor,
        anchor_mask: Tensor,
    ) -> Tensor:
        """Self-forced rollout path에 SiD-lite loss를 계산합니다.

        Args:
            tokenized_map: 평가 모드 map token 사전입니다.
            tokenized_agent: 평가 모드 agent token 사전입니다.
            committed_path_norm: Generator가 실제로 실행한 self-forced flow state ``X`` 입니다.
                pose-space에서는 ``[n_valid_agent, flow_window_steps, 4]`` 이고,
                control-space에서는 ``[n_valid_agent, flow_window_steps, 3]`` 입니다.
            anchor_mask: 첫 anchor에서 사용할 agent mask입니다.
                shape은 ``[n_agent]`` 입니다.

        Returns:
            Tensor: scalar SiD-lite loss입니다. shape은 ``[]`` 입니다.
        """
        target_clean_norm, generated_clean_norm = self._predict_self_forced_teacher_estimator_clean_paths(
            tokenized_map=tokenized_map,
            tokenized_agent=tokenized_agent,
            committed_path_norm=committed_path_norm,
            anchor_mask=anchor_mask,
        )
        self._set_self_forced_backward_context(
            committed_path_norm=committed_path_norm,
            target_clean_norm=target_clean_norm,
            generated_clean_norm=generated_clean_norm,
        )
        return compute_clean_sid_loss(
            committed_path_norm=committed_path_norm,
            target_clean_norm=target_clean_norm,
            generated_clean_norm=generated_clean_norm,
            sid_alpha=self.self_forced_sid_alpha,
            normalizer_eps=self.self_forced_sid_normalizer_eps,
        )

    def _compute_self_forced_distribution_matching_loss(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        committed_path_norm: Tensor,
        anchor_mask: Tensor,
    ) -> Tensor:
        """설정에 따라 DMD-style 또는 SiD-style generator loss를 계산합니다.

        Args:
            tokenized_map: 평가 모드 map token 사전입니다.
            tokenized_agent: 평가 모드 agent token 사전입니다.
            committed_path_norm: Generator가 실제로 실행한 self-forced flow state입니다.
                pose-space에서는 ``[n_valid_agent, flow_window_steps, 4]`` 이고,
                control-space에서는 ``[n_valid_agent, flow_window_steps, 3]`` 입니다.
            anchor_mask: 첫 anchor에서 사용할 agent mask입니다.
                shape은 ``[n_agent]`` 입니다.

        Returns:
            Tensor: scalar 분포 맞춤 loss입니다. shape은 ``[]`` 입니다.
        """
        if self.self_forced_distribution_matching_objective == "sid":
            return self._compute_self_forced_sid_loss(
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
                committed_path_norm=committed_path_norm,
                anchor_mask=anchor_mask,
            )

        active_control_mask = self._build_self_forced_active_control_mask(
            tokenized_agent=tokenized_agent,
            committed_path_norm=committed_path_norm,
            anchor_mask=anchor_mask,
        )
        dmd_start_epoch = int(self.self_forced_start_epoch) + int(self.self_forced_estimator_warmup_epochs)
        dmd_injection_scale = compute_self_forced_dmd_injection_scale(
            current_epoch=int(self.current_epoch),
            dmd_start_epoch=dmd_start_epoch,
            use_ramp=self.self_forced_dmd_use_injection_ramp,
        )
        pose_projected_dmd = self._should_project_self_forced_dmd_to_pose_space(committed_path_norm)
        path_delta = self._compute_self_forced_direction(
            tokenized_map=tokenized_map,
            tokenized_agent=tokenized_agent,
            committed_path_norm=committed_path_norm,
            anchor_mask=anchor_mask,
            active_control_mask=active_control_mask,
            dmd_injection_scale=dmd_injection_scale,
        )
        surrogate_injection_scale = 1.0 if pose_projected_dmd else dmd_injection_scale
        sf_loss, target_path_norm = active_control_dmd_surrogate_loss(
            committed_path_norm=committed_path_norm,
            dmd_direction=path_delta,
            active_mask=active_control_mask,
            dmd_injection_scale=surrogate_injection_scale,
        )
        self._set_self_forced_backward_context(
            committed_path_norm=committed_path_norm,
            path_delta=path_delta,
            target_path_norm=target_path_norm,
            active_control_mask=active_control_mask,
            dmd_injection_scale=committed_path_norm.new_tensor(dmd_injection_scale),
            pose_projected_dmd=committed_path_norm.new_tensor(float(pose_projected_dmd)),
        )
        return sf_loss

    def on_fit_start(self) -> None:
        """학습 시작 전에 빠른 closed-loop validation 모드를 켭니다.

        Lightning은 ``on_fit_start`` 를 sanity check 전에 호출합니다.
        그래서 여기서 validation batch 개수를 줄이면 학습 전 sanity check와
        학습 중 validation 둘 다 같은 빠른 규칙을 사용하게 됩니다.

        Returns:
            None
        """
        self._configure_fast_wosac_validation_scope()
        self._apply_fit_time_validation_batch_limit()
        if (
            self.self_forced_enabled
            and self.self_forced_use_distribution_matching_loss
            and int(self.self_forced_estimator_warmup_epochs) > 0
        ):
            self._capture_self_forced_validation_interval()
        self._sync_self_forced_auxiliary_models()
        self._load_self_forced_generated_estimator_bank()
        self._prepare_self_forced_generator_ema()

    def on_validation_start(self) -> None:
        """validation 시작 직전에 scorer batch 수 자동 조정을 다시 시도합니다."""
        self._configure_fast_wosac_validation_scope()

    def setup(self, stage: str) -> None:
        """validation dataloader cap이 scorer scene 수보다 작지 않도록 미리 맞춥니다."""
        if stage in {"fit", "validate"}:
            self._configure_fast_wosac_validation_scope()

    def on_fit_end(self) -> None:
        """학습이 끝나면 임시로 바꾼 validation 제한 값을 정리합니다.

        Returns:
            None
        """
        self._restore_fit_time_validation_batch_limit()
        self._restore_self_forced_validation_interval()

    @staticmethod
    def _summarize_nonfinite_tensor(tensor: Tensor) -> str:
        """non-finite tensor의 요약 문자열을 만듭니다."""
        detached = tensor.detach()
        finite_mask = torch.isfinite(detached)
        nonfinite_count = int((~finite_mask).sum().item())
        finite_abs_max = float(detached[finite_mask].abs().max().item()) if finite_mask.any() else float("nan")
        return (
            f"shape={tuple(detached.shape)}, dtype={detached.dtype}, "
            f"nonfinite_count={nonfinite_count}, finite_abs_max={finite_abs_max}"
        )

    def _set_self_forced_backward_context(self, **tensors: Tensor) -> None:
        """Store SF tensors for error-only diagnostics without scanning them."""
        self._self_forced_backward_context = {
            name: tensor.detach() for name, tensor in tensors.items()
        }

    def _clear_self_forced_backward_context(self) -> None:
        self._self_forced_backward_context = None

    def _format_self_forced_backward_context(self) -> str:
        if not self._self_forced_backward_context:
            return ""
        summaries = [
            f"{name}=({self._summarize_nonfinite_tensor(tensor)})"
            for name, tensor in self._self_forced_backward_context.items()
        ]
        return " Self-forced backward context: " + "; ".join(summaries)

    def _training_step_manual_open_loop(self, data, batch_idx):
        """self-forced 시작 전 epoch에서 기존 open-loop loss를 manual optimizer로 학습합니다.

        Args:
            data: 학습용 장면 batch입니다.
            batch_idx: 현재 batch 번호입니다.

        Returns:
            Tensor: logging용 detached 총 loss입니다.
        """
        tokenized_map, tokenized_agent = self.token_processor(data)
        pred = self.encoder(
            tokenized_map,
            tokenized_agent,
            anchor_mask_key="flow_train_mask",
            compute_metric_outputs=self.train_open_loop_metrics,
        )
        fm_loss, open_metric_dict, sample_count, has_open_loop_targets = self._open_loop_denoise_metrics(
            pred,
            zero_loss_module=self.encoder,
        )
        total_loss = fm_loss
        self._accumulate_open_loop_train_epoch_metrics(
            total_loss=total_loss,
            fm_loss=fm_loss,
            open_metric_dict=open_metric_dict,
            sample_count=sample_count,
        )
        has_open_loop_targets_pending = self._start_distributed_bool_any(
            has_open_loop_targets,
            device=total_loss.device,
        )

        generator_optimizer = self.optimizers()[0]
        self.toggle_optimizer(generator_optimizer)
        try:
            generator_optimizer.zero_grad(set_to_none=True)
            self._prepare_self_forced_generator_backward_boundary()
            self._manual_backward_without_autocast(total_loss)
            self._assert_self_forced_generator_update_isolated()
            has_open_loop_targets_global = self._finish_distributed_bool_any(
                has_open_loop_targets_pending
            )
            if has_open_loop_targets_global:
                self._clip_and_step_with_optional_scaler(
                    generator_optimizer,
                    gradient_clip_val=self.self_forced_gradient_clip_val,
                    gradient_clip_algorithm="norm",
                )
                self._update_self_forced_generator_ema_after_step()
        finally:
            self._clear_self_forced_generator_gradients()
            self.untoggle_optimizer(generator_optimizer)

        return total_loss.detach()

    def _training_step_self_forced_anchor_only(
        self,
        *,
        fm_loss: Tensor | None,
        open_metric_dict: Dict[str, Tensor] | None,
        has_anchor_fm_targets: bool,
    ) -> Tensor:
        """self-forced 보조 objective를 끄고 anchor FM loss만으로 Generator를 업데이트합니다."""
        if fm_loss is None:
            anchor_loss = self._build_trainable_connected_zero_loss(self.encoder)
            has_anchor_fm_targets_global = False
        else:
            anchor_loss = fm_loss
            has_anchor_fm_targets_global = self._sync_distributed_bool_any(
                has_anchor_fm_targets,
                device=fm_loss.device,
            )
        total_loss = self.self_forced_anchor_weight * anchor_loss
        should_step = bool(
            fm_loss is not None
            and has_anchor_fm_targets_global
            and float(self.self_forced_anchor_weight) != 0.0
        )

        generator_optimizer = self.optimizers()[0]
        self.toggle_optimizer(generator_optimizer)
        try:
            generator_optimizer.zero_grad(set_to_none=True)
            self._prepare_self_forced_generator_backward_boundary()
            self._manual_backward_without_autocast(total_loss)
            self._assert_self_forced_generator_update_isolated()
            if should_step:
                self._clip_and_step_with_optional_scaler(
                    generator_optimizer,
                    gradient_clip_val=self.self_forced_gradient_clip_val,
                    gradient_clip_algorithm="norm",
                )
                self._update_self_forced_generator_ema_after_step()
        finally:
            self._clear_self_forced_generator_gradients()
            self._clear_self_forced_backward_context()
            self.untoggle_optimizer(generator_optimizer)

        zero_metric = total_loss.detach().new_zeros(())
        self.log("train/loss", total_loss.detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        if fm_loss is not None:
            self.log("train/loss_fm", fm_loss.detach(), on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_npfm_loss", zero_metric, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_generated_estimator_loss", zero_metric, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_anchor_fm_enabled", float(self.self_forced_use_anchor_fm_loss), on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_anchor_loss", anchor_loss.detach(), on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_anchor_weight", float(self.self_forced_anchor_weight), on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_distribution_matching_enabled", 0.0, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_pose_projected_dmd", 0.0, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        if open_metric_dict:
            self.log(
                f"train/{self.train_open_metric_names['ade']}",
                open_metric_dict[self.open_metric_names["ade"]].detach(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                f"train/{self.train_open_metric_names['fde']}",
                open_metric_dict[self.open_metric_names["fde"]].detach(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                f"train/{self.train_open_metric_names['yaw_ade']}",
                open_metric_dict[self.open_metric_names["yaw_ade"]].detach(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                f"train/{self.train_open_metric_names['yaw_fde']}",
                open_metric_dict[self.open_metric_names["yaw_fde"]].detach(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
        return total_loss.detach()

    def _training_step_self_forced(self, data, batch_idx):
        """PDF Step 3~10에 해당하는 self-forced NPFM 학습 step입니다.

        Args:
            data: 학습용 장면 batch입니다.
            batch_idx: 현재 batch 번호입니다.

        Returns:
            Tensor: logging용 detached 총 loss입니다.
        """
        fm_loss = None
        open_metric_dict = None
        has_anchor_fm_targets = False
        is_estimator_warmup_active = self._is_self_forced_estimator_warmup_active()
        if should_compute_anchor_flow_matching_loss(
            use_anchor_flow_matching_loss=self.self_forced_use_anchor_fm_loss,
            is_estimator_warmup_active=is_estimator_warmup_active,
        ):
            tokenized_map_train, tokenized_agent_train = self.token_processor(data)
            pred = self.encoder(
                tokenized_map_train,
                tokenized_agent_train,
                anchor_mask_key="flow_train_mask",
                compute_metric_outputs=self.train_open_loop_metrics,
            )
            fm_loss, open_metric_dict, _, has_anchor_fm_targets = self._open_loop_denoise_metrics(
                pred,
                zero_loss_module=self.encoder,
            )

        if not self.self_forced_use_distribution_matching_loss:
            return self._training_step_self_forced_anchor_only(
                fm_loss=fm_loss,
                open_metric_dict=open_metric_dict,
                has_anchor_fm_targets=has_anchor_fm_targets,
            )

        tokenized_map_eval, tokenized_agent_eval = self._build_eval_tokenized_inputs(data)
        if is_estimator_warmup_active:
            with torch.no_grad():
                rollout = self._run_self_forced_rollout(tokenized_map_eval, tokenized_agent_eval)
        else:
            rollout = self._run_self_forced_rollout(tokenized_map_eval, tokenized_agent_eval)
        committed_path_norm, anchor_mask = self._pack_self_forced_committed_rollout(
            rollout=rollout,
            tokenized_agent=tokenized_agent_eval,
        )
        has_committed_path_local = committed_path_norm.numel() > 0
        has_committed_path_global = self._sync_distributed_bool_any(
            has_committed_path_local,
            device=committed_path_norm.device,
        )
        has_anchor_fm_targets_global = False
        if fm_loss is not None:
            has_anchor_fm_targets_global = self._sync_distributed_bool_any(
                has_anchor_fm_targets,
                device=fm_loss.device,
            )

        if not has_committed_path_global:
            if is_estimator_warmup_active:
                return self._finish_self_forced_estimator_warmup_step(None)
            if fm_loss is None or not has_anchor_fm_targets_global:
                zero_loss = (
                    fm_loss
                    if fm_loss is not None
                    else self._build_trainable_connected_zero_loss(self.encoder)
                )
                generator_optimizer = self.optimizers()[0]
                self.toggle_optimizer(generator_optimizer)
                try:
                    generator_optimizer.zero_grad(set_to_none=True)
                    self._prepare_self_forced_generator_backward_boundary()
                    self._manual_backward_without_autocast(zero_loss)
                    self._assert_self_forced_generator_update_isolated()
                finally:
                    self._clear_self_forced_generator_gradients()
                    self.untoggle_optimizer(generator_optimizer)
                self.log("train/loss", zero_loss.detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
                if fm_loss is not None:
                    self.log(
                        "train/loss_fm",
                        fm_loss.detach(),
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=1,
                    )
                self.log("train/sf_anchor_fm_enabled", 0.0, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
                self.log("train/sf_anchor_weight", float(self.self_forced_anchor_weight), on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
                self.log(
                    "train/sf_pose_projected_dmd",
                    float(self.self_forced_project_dmd_to_pose_space and self.use_kinematic_control_flow),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=1,
                )
                return zero_loss.detach()
            generator_optimizer = self.optimizers()[0]
            self.toggle_optimizer(generator_optimizer)
            generator_optimizer.zero_grad(set_to_none=True)
            self._prepare_self_forced_generator_backward_boundary()
            try:
                self._manual_backward_without_autocast(fm_loss)
                self._assert_self_forced_generator_update_isolated()
                if has_anchor_fm_targets_global:
                    self._clip_and_step_with_optional_scaler(generator_optimizer)
                    self._update_self_forced_generator_ema_after_step()
            finally:
                self._clear_self_forced_generator_gradients()
                self.untoggle_optimizer(generator_optimizer)
            self.log("train/loss", fm_loss.detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
            self.log("train/loss_fm", fm_loss.detach(), on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
            return fm_loss.detach()

        gen_estimator_loss = self._update_generated_path_flow_estimator(
            tokenized_map=tokenized_map_eval,
            tokenized_agent=tokenized_agent_eval,
            committed_path_norm=committed_path_norm,
            anchor_mask=anchor_mask,
            has_committed_path_global=has_committed_path_global,
        )
        if is_estimator_warmup_active:
            return self._finish_self_forced_estimator_warmup_step(gen_estimator_loss)
        if has_committed_path_local:
            sf_loss = self._compute_self_forced_distribution_matching_loss(
                tokenized_map=tokenized_map_eval,
                tokenized_agent=tokenized_agent_eval,
                committed_path_norm=committed_path_norm,
                anchor_mask=anchor_mask,
            )
        else:
            sf_loss = self._build_trainable_connected_zero_loss(self.encoder)
        anchor_loss = (
            fm_loss
            if fm_loss is not None
            else committed_path_norm.new_zeros(())
        )
        total_loss = (
            self.self_forced_weight * sf_loss
            + self.self_forced_anchor_weight * anchor_loss
        )
        if not torch.isfinite(total_loss):
            context = self._format_self_forced_backward_context()
            self._clear_self_forced_backward_context()
            raise RuntimeError(
                "Non-finite self-forced total_loss detected: "
                f"{self._summarize_nonfinite_tensor(total_loss)}"
                f"{context}"
            )

        generator_optimizer = self.optimizers()[0]
        try:
            self.toggle_optimizer(generator_optimizer)
            try:
                generator_optimizer.zero_grad(set_to_none=True)
                self._prepare_self_forced_generator_backward_boundary()
                self._manual_backward_without_autocast(total_loss)
                self._assert_self_forced_generator_update_isolated()
                self._clip_and_step_with_optional_scaler(
                    generator_optimizer,
                    gradient_clip_val=self.self_forced_gradient_clip_val,
                    gradient_clip_algorithm="norm",
                )
                self._update_self_forced_generator_ema_after_step()
                self._clear_self_forced_generator_gradients()
            finally:
                self.untoggle_optimizer(generator_optimizer)
        finally:
            self._clear_self_forced_backward_context()

        self.log("train/loss", total_loss.detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        if fm_loss is not None:
            self.log("train/loss_fm", fm_loss.detach(), on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_npfm_loss", sf_loss.detach(), on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_generated_estimator_loss", gen_estimator_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_anchor_fm_enabled", float(self.self_forced_use_anchor_fm_loss), on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_anchor_loss", anchor_loss.detach(), on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_anchor_weight", float(self.self_forced_anchor_weight), on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_distribution_matching_enabled", 1.0, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log(
            "train/sf_pose_projected_dmd",
            float(self._should_project_self_forced_dmd_to_pose_space(committed_path_norm)),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        if "sf_terminal_s_by_scenario" in rollout:
            self.log(
                "train/sf_terminal_s_mean",
                rollout["sf_terminal_s_by_scenario"].float().mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "train/sf_terminal_k_mean",
                rollout["sf_terminal_step_by_scenario"].float().mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
        if open_metric_dict:
            self.log(
                f"train/{self.train_open_metric_names['ade']}",
                open_metric_dict[self.open_metric_names["ade"]].detach(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                f"train/{self.train_open_metric_names['fde']}",
                open_metric_dict[self.open_metric_names["fde"]].detach(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                f"train/{self.train_open_metric_names['yaw_ade']}",
                open_metric_dict[self.open_metric_names["yaw_ade"]].detach(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                f"train/{self.train_open_metric_names['yaw_fde']}",
                open_metric_dict[self.open_metric_names["yaw_fde"]].detach(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
        return total_loss.detach()

    def training_step(self, data, batch_idx):
        """한 batch의 Flow Matching loss를 계산합니다.

        Args:
            data: 학습용 장면 배치입니다.
            batch_idx: 현재 batch 번호입니다.

        Returns:
            Tensor: 최종 학습 loss입니다.
        """
        if self.self_forced_enabled:
            if self._is_self_forced_active():
                return self._training_step_self_forced(data=data, batch_idx=batch_idx)
            return self._training_step_manual_open_loop(data=data, batch_idx=batch_idx)
        tokenized_map, tokenized_agent = self.token_processor(data)
        """ pred
flow_pred_norm [n_valid_anchor, 20, 4]
flow_target_norm [n_valid_anchor, 20, 4]
    -> flow_pred_norm / flow_target_norm 을 비교해 FM loss 계산
flow_pred_clean_norm [n_valid_anchor, 20, 4] -> 속도 예측을 clean trajectory 공간으로 복원한 값
flow_clean_norm [n_valid_anchor, 20, 4]
    -> 정답 궤적 (flow_pred_clean_norm / flow_clean_norm 릴 비교해서 ADE/FDE/yaw error 계산)
        """
        pred = self.encoder(
            tokenized_map,
            tokenized_agent,
            anchor_mask_key="flow_train_mask",
            compute_metric_outputs=self.train_open_loop_metrics,
        )
        """
fm_loss: 
    Tensor shape []
open_metric_dict: 
    Dict[str, Tensor]
        """
        fm_loss, open_metric_dict, sample_count, has_open_loop_targets = self._open_loop_denoise_metrics(
            pred,
            zero_loss_module=self,
        )
        total_loss = fm_loss
        if not torch.isfinite(fm_loss):
            raise RuntimeError(f"Non-finite fm_loss detected: {self._summarize_nonfinite_tensor(fm_loss)}")
        if not torch.isfinite(total_loss):
            raise RuntimeError(
                "Non-finite total_loss detected: "
                f"{self._summarize_nonfinite_tensor(total_loss)}"
            )

        self._accumulate_open_loop_train_epoch_metrics(
            total_loss=total_loss,
            fm_loss=fm_loss,
            open_metric_dict=open_metric_dict,
            sample_count=sample_count,
        )
        self._record_automatic_open_loop_targets(
            has_open_loop_targets=has_open_loop_targets,
            loss=total_loss,
        )
        return total_loss

    def _record_automatic_open_loop_targets(
        self,
        *,
        has_open_loop_targets: bool,
        loss: Tensor,
    ) -> None:
        """Record automatic-optimization target coverage, unless pre-verified."""

        if self.skip_empty_open_loop_optimizer_guard:
            if not has_open_loop_targets:
                raise RuntimeError(
                    "skip_empty_open_loop_optimizer_guard=true requires every local "
                    "open-loop train batch to contain at least one loss target. "
                    "Run scripts/verify_flow_target_coverage.py on the selected "
                    "train split, or set the guard flag back to false."
                )
            return
        has_open_loop_targets_pending = self._start_distributed_bool_any(
            has_open_loop_targets,
            device=loss.device,
        )
        self._automatic_open_loop_has_target_pending.append(has_open_loop_targets_pending)

    def on_before_optimizer_step(self, optimizer) -> None:
        """DDP 전체에 target이 없는 automatic optimization step의 업데이트를 막습니다."""
        if not bool(getattr(self, "automatic_optimization", True)):
            return
        if bool(getattr(self, "skip_empty_open_loop_optimizer_guard", False)):
            self._automatic_open_loop_has_target_pending.clear()
            self._automatic_open_loop_has_target_since_step = False
            self._skip_next_automatic_optimizer_step = False
            return
        has_open_loop_targets_global = bool(self._automatic_open_loop_has_target_since_step)
        if self._automatic_open_loop_has_target_pending:
            has_open_loop_targets_global = any(
                self._finish_distributed_bool_any(pending)
                for pending in self._automatic_open_loop_has_target_pending
            )
            self._automatic_open_loop_has_target_pending.clear()
        self._skip_next_automatic_optimizer_step = not has_open_loop_targets_global
        if not has_open_loop_targets_global:
            optimizer.zero_grad(set_to_none=True)
        self._automatic_open_loop_has_target_since_step = False
        self._skip_next_automatic_optimizer_step = False

    def on_after_backward(self) -> None:
        """Backward 이후 추가 gradient scan을 수행하지 않습니다.

        Loss/parameter non-finite fail-fast는 forward 경로에 남기고, 매 step 모든
        gradient를 순회하던 debug-only 검사는 제거해 pretrain step latency를 줄입니다.
        """
        return

    def validation_step(self, data, batch_idx):
        eval_generator = self._get_eval_generator()
        tokenized_map, tokenized_agent = self.token_processor(data)
        map_feature = None
        if self.val_open_loop or self.val_closed_loop:
            map_feature = eval_generator.encode_map(tokenized_map)

        if self.val_open_loop:
            denoise_pred = eval_generator.forward_from_map_feature(
                map_feature=map_feature,
                tokenized_agent=tokenized_agent,
                anchor_mask_key="flow_eval_mask",
            )
            open_sample_count = int(denoise_pred["flow_clean_norm"].shape[0])
            open_pred_clean_norm = eval_generator.sample_open_loop_future(
                anchor_hidden=denoise_pred["anchor_hidden"],
                anchor_mask=denoise_pred["anchor_mask"],
                sampling_scheme=self.validation_rollout_sampling,
                sampling_seed=self._get_validation_open_seed(batch_idx),
            )
            open_pred_metric_norm = eval_generator.flow_norm_to_pose_metric_norm(
                value=open_pred_clean_norm,
                agent_type=denoise_pred.get("flow_metric_agent_type"),
                agent_length=denoise_pred.get("flow_metric_agent_length"),
            )
            open_target_metric_norm = denoise_pred.get(
                "flow_clean_metric_norm",
                denoise_pred["flow_clean_norm"],
            )
            open_metric_dict = self._build_open_loop_metric_dict(
                pred_clean_norm=open_pred_metric_norm,
                target_clean_norm=open_target_metric_norm,
            )
            self._update_weighted_validation_metrics(
                metric_store=self.val_open_epoch_metrics,
                metric_dict=open_metric_dict,
                sample_count=open_sample_count,
            )

        if self.val_closed_loop:
            return_flow_2s_preview = self.vis_flow_2s_preview and batch_idx < self.n_vis_batch
            pred_traj, pred_z, pred_head, flow_preview = self._run_closed_loop_rollouts(
                rollout_encoder=eval_generator,
                data=data,
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
                return_flow_2s_preview=return_flow_2s_preview,
            )
            update_wosac_distribution_metric_from_model(
                model=self,
                data=data,
                pred_traj=pred_traj,
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
                scenario_rollouts = self._update_closed_loop_metric_states(
                    data=data,
                    batch_idx=batch_idx,
                    pred_traj=pred_traj,
                    pred_z=pred_z,
                    pred_head=pred_head,
                )

            if self.global_rank == 0 and batch_idx < self.n_vis_batch and scenario_rollouts is not None:
                video_logger = self._get_video_logger()
                for scen_idx in range(self.n_vis_scenario):
                    vis = VisWaymo(
                        scenario_path=data["tfrecord_path"][scen_idx],
                        save_dir=self.video_dir / f"batch_{batch_idx:02d}-scenario_{scen_idx:02d}",
                        vis_ghost_gt=self.vis_ghost_gt,
                        vis_flow_preview=self.vis_flow_2s_preview,
                        flow_preview_commit_steps=self.encoder.agent_encoder.shift,
                    )
                    vis.save_video_scenario_rollout(
                        scenario_rollouts[scen_idx],
                        self.n_vis_rollout,
                        flow_preview=self._get_scenario_flow_preview(
                            agent_id=data["agent"]["id"],
                            agent_batch=data["agent"]["batch"],
                            scenario_index=scen_idx,
                            flow_preview=flow_preview,
                        ),
                    )
                    for video_path in vis.video_paths:
                        if video_logger is not None:
                            video_logger.log_video("/".join(video_path.split("/")[-3:]), [video_path])
                            if self.delete_local_videos_after_wandb_upload:
                                self._cleanup_local_video(video_path)

    def on_validation_epoch_end(self):
        log_and_reset_wosac_distribution_metric(
            model=self,
            metric=self.wosac_distribution_metrics,
        )
        if self.val_open_loop:
            epoch_open_metrics = self._compute_and_reset_validation_metrics(
                prefix="val_open",
                metric_store=self.val_open_epoch_metrics,
            )
            for metric_name, metric_value in epoch_open_metrics.items():
                self.log(metric_name, metric_value, on_step=False, on_epoch=True, sync_dist=True)

        if self.val_closed_loop:
            if not self.sim_agents_submission.is_active:
                self.sim_agents_metrics._drain_completed_futures(wait=True, drain_all=True)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    reduced_metric_state = self.sim_agents_metrics.get_state_tensor(device=self.device)
                    torch.distributed.all_reduce(reduced_metric_state)
                    epoch_sim_agents_metrics = self.sim_agents_metrics.compute_from_state_tensor(
                        reduced_metric_state
                    )
                    minade_value: Tensor | None = None
                    if self._should_compute_closed_loop_minade():
                        reduced_minade_state = torch.stack(
                            [
                                self.minADE.sum.detach().to(device=self.device),
                                self.minADE.count.detach().to(device=self.device),
                            ]
                        )
                        torch.distributed.all_reduce(reduced_minade_state)
                        minade_value = reduced_minade_state[0] / reduced_minade_state[1].clamp_min(1e-6)
                else:
                    epoch_sim_agents_metrics = self.sim_agents_metrics.compute()
                    minade_value = None
                    if self._should_compute_closed_loop_minade():
                        minade_value = self.minADE.compute()
                        if self.minADE_predict.count > 0:
                            minade_predict_value = self.minADE_predict.compute()
                            epoch_sim_agents_metrics[
                                "val_closed/sim_agents_2025/minADE_tracks_to_predict"
                            ] = minade_predict_value
                closed_loop_metric = epoch_sim_agents_metrics[self.closed_loop_metric_name]
                if self.global_rank == 0 and minade_value is not None:
                    epoch_sim_agents_metrics[self.val_closed_minade_name] = minade_value
                self.log(
                    self.closed_loop_metric_name,
                    closed_loop_metric,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
                if self.global_rank == 0 and self.logger is not None:
                    epoch_sim_agents_metrics["epoch"] = (
                        self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                    )
                    self.logger.log_metrics(epoch_sim_agents_metrics)
                self.sim_agents_metrics.reset()
                self.minADE.reset()
                self.minADE_predict.reset()
            if self.sim_agents_submission.is_active:
                self.sim_agents_submission.save_sub_file()

    def configure_optimizers(self):
        def lr_lambda(_current_step):
            current_step = self.current_epoch + 1
            if current_step < self.lr_warmup_steps:
                return self.lr_min_ratio + (1.0 - self.lr_min_ratio) * current_step / max(self.lr_warmup_steps, 1)
            return self.lr_min_ratio + 0.5 * (1.0 - self.lr_min_ratio) * (
                1.0
                + math.cos(
                    math.pi * min(
                        1.0,
                        (current_step - self.lr_warmup_steps) / max(self.lr_total_steps - self.lr_warmup_steps, 1),
                    )
                )
            )

        if self.self_forced_enabled:
            self._apply_self_forced_unfrozen_range()
            generator_params = [param for param in self.encoder.parameters() if param.requires_grad]
            if not generator_params:
                raise RuntimeError("No trainable generator parameters found for self-forced optimization.")
            generator_optimizer = torch.optim.AdamW(generator_params, lr=self.lr)
            if self.self_forced_generated_estimator is None:
                raise RuntimeError("self_forced_generated_estimator is not initialized.")
            estimator_params = [
                param for param in self.self_forced_generated_estimator.parameters() if param.requires_grad
            ]
            if not estimator_params:
                raise RuntimeError("No trainable generated-estimator parameters found.")
            generated_estimator_optimizer = torch.optim.AdamW(
                estimator_params,
                lr=self.self_forced_generated_estimator_lr,
            )
            return [generator_optimizer, generated_estimator_optimizer]

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]

    def on_train_epoch_start(self) -> None:
        """새 epoch의 open-loop train metric accumulator를 초기화합니다."""
        self._reset_open_loop_train_epoch_metrics()
        self._automatic_open_loop_has_target_pending.clear()
        self._apply_self_forced_validation_schedule_for_current_epoch()

    def on_train_epoch_end(self) -> None:
        """self-forced manual optimization에서 scheduler가 있으면 epoch마다 한 번 진행합니다.

        Returns:
            None
        """
        self._log_open_loop_train_epoch_metrics()
        if not self.self_forced_enabled:
            return
        if (
            int(self.self_forced_estimator_warmup_epochs) > 0
            and int(self.current_epoch)
            == int(self.self_forced_start_epoch) + int(self.self_forced_estimator_warmup_epochs) - 1
        ):
            snapshot_path = self._save_self_forced_generated_estimator_bank_snapshot()
            if snapshot_path is not None:
                self._upload_self_forced_generated_estimator_bank_snapshot(snapshot_path)
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return
        if isinstance(schedulers, (list, tuple)):
            if len(schedulers) == 0:
                return
            scheduler = schedulers[0]
        else:
            scheduler = schedulers
        scheduler.step()

    def test_step(self, data, batch_idx):
        eval_generator = self._get_eval_generator()
        tokenized_map, tokenized_agent = self.token_processor(data)
        map_feature = eval_generator.encode_map(tokenized_map)
        pred_traj, pred_z, pred_head, _ = self._run_closed_loop_rollouts(
            rollout_encoder=eval_generator,
            data=data,
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
        )
        update_wosac_distribution_metric_from_model(
            model=self,
            data=data,
            pred_traj=pred_traj,
        )

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
        log_and_reset_wosac_distribution_metric(
            model=self,
            metric=self.test_wosac_distribution_metrics,
        )
        self.sim_agents_submission.save_sub_file()
