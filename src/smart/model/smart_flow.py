from __future__ import annotations

import copy
import gc
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import hydra
import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

from src.smart.metrics import SimAgentsMetrics, SimAgentsSubmission, minADE
from src.smart.metrics.flow_metrics import (
    WeightedMeanMetric,
    ade_future,
    fde_future,
    flow_matching_loss,
    yaw_ade_future,
    yaw_fde_future,
)
from src.smart.modules.draft_physics import (
    DRAFT_PHYSICS_ACTUAL_UNIT_KEYS,
    DRAFT_PHYSICS_COMPONENT_KEYS,
    DraftPhysicsRegularizer,
)
from src.smart.modules.self_forced_path_flow import (
    build_anchor0_normalized_committed_path,
    build_anchor0_physics_inputs,
    get_anchor0_valid_mask,
    masked_mean_square_loss,
)
from src.smart.modules.self_forced_dmd_guidance import build_clean_dmd_direction
from src.smart.modules.self_forced_sid_loss import compute_clean_sid_loss
from src.smart.modules.self_forced_update_separation import (
    assert_no_module_gradients,
    clear_module_gradients,
)
from src.smart.modules.self_forced_trainable_range import (
    apply_self_forced_unfrozen_range,
    resolve_self_forced_unfrozen_range,
)
from src.smart.modules.smart_flow_decoder import SMARTFlowDecoder
from src.smart.tokens.flow_token_processor import FlowTokenProcessor
from src.smart.utils.finetune import set_model_for_finetuning
from src.smart.utils.flow_horizon import format_flow_horizon_tag
from src.utils.vis_waymo import VisWaymo
from src.utils.sim_agents_utils import get_scenario_id_int_tensor, get_scenario_rollouts


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
        self.token_processor = FlowTokenProcessor(**model_config.token_processor)

        self.encoder = SMARTFlowDecoder(
            **model_config.decoder,
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

        self.video_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.video_dir = Path(self.video_dir) / "videos"

        self.validation_rollout_sampling = model_config.validation_rollout_sampling

        draft_config = getattr(model_config, "draft", None)
        self.draft_enabled = bool(draft_config is not None and getattr(draft_config, "enabled", False))
        self.draft_sampling = getattr(draft_config, "sampling", None)
        self.draft_start_epoch = int(getattr(draft_config, "start_epoch", 0)) if draft_config is not None else 0
        self.draft_ramp_epochs = int(getattr(draft_config, "ramp_epochs", 1)) if draft_config is not None else 1
        self.draft_max_weight = float(getattr(draft_config, "max_weight", 0.0)) if draft_config is not None else 0.0
        self.draft_physics_force_fp32 = False

        if self.draft_enabled:
            draft_physics = getattr(draft_config, "physics")
            self.draft_physics_force_fp32 = bool(getattr(draft_physics, "force_fp32", True))
            self.draft_regularizer = DraftPhysicsRegularizer(
                dt=float(getattr(draft_physics, "dt", 0.1)),
                pos_scale_m=float(getattr(draft_physics, "pos_scale_m", 20.0)),
                speed_floor_mps=float(getattr(draft_physics, "speed_floor_mps", 0.5)),
                vehicle_v_max_mps=float(getattr(draft_physics, "vehicle_v_max_mps", 35.0)),
                vehicle_a_max_mps2=float(getattr(draft_physics, "vehicle_a_max_mps2", 8.0)),
                vehicle_lat_accel_max_mps2=float(
                    getattr(draft_physics, "vehicle_lat_accel_max_mps2", 4.2)
                ),
                bicycle_v_max_mps=float(getattr(draft_physics, "bicycle_v_max_mps", 22.0)),
                bicycle_a_max_mps2=float(getattr(draft_physics, "bicycle_a_max_mps2", 5.5)),
                bicycle_lat_accel_max_mps2=float(
                    getattr(draft_physics, "bicycle_lat_accel_max_mps2", 4.4)
                ),
                pedestrian_v_max_mps=float(getattr(draft_physics, "pedestrian_v_max_mps", 5.0)),
                pedestrian_a_max_mps2=float(getattr(draft_physics, "pedestrian_a_max_mps2", 4.7)),
                vehicle_wheelbase_scale=float(
                    getattr(draft_physics, "vehicle_wheelbase_scale", 0.60)
                ),
                bicycle_wheelbase_scale=float(
                    getattr(draft_physics, "bicycle_wheelbase_scale", 0.85)
                ),
                vehicle_steer_max_rad=float(getattr(draft_physics, "vehicle_steer_max_rad", 0.55)),
                bicycle_steer_max_rad=float(getattr(draft_physics, "bicycle_steer_max_rad", 1.00)),
                vehicle_steer_rate_max_radps=float(
                    getattr(draft_physics, "vehicle_steer_rate_max_radps", 0.8)
                ),
                bicycle_steer_rate_max_radps=float(
                    getattr(draft_physics, "bicycle_steer_rate_max_radps", 1.5)
                ),
                soft_weight=float(
                    getattr(
                        draft_physics,
                        "soft_weight",
                        getattr(
                            draft_physics,
                            "vehicle_soft_weight",
                            getattr(
                                draft_physics,
                                "bicycle_soft_weight",
                                getattr(draft_physics, "pedestrian_soft_weight", 0.25),
                            ),
                        ),
                    )
                ),
                compare_softness_to_gt=bool(getattr(draft_physics, "compare_softness_to_gt", True)),
                pedestrian_heading_weight=float(
                    getattr(draft_physics, "pedestrian_heading_weight", 0.05)
                ),
                pedestrian_heading_speed_threshold_mps=float(
                    getattr(draft_physics, "pedestrian_heading_speed_threshold_mps", 0.5)
                ),
                eps=float(getattr(draft_physics, "eps", 1e-6)),
            )
        else:
            self.draft_regularizer = None

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
        self.self_forced_path_step_size = (
            float(getattr(self.self_forced_config, "path_step_size", 0.05))
            if self.self_forced_config is not None
            else 0.05
        )
        self.self_forced_direction_normalizer_eps = (
            float(getattr(self.self_forced_config, "clean_dmd_normalizer_eps", 1.0e-3))
            if self.self_forced_config is not None
            else 1.0e-3
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
                    self.self_forced_direction_normalizer_eps,
                )
            )
            if self.self_forced_config is not None
            else self.self_forced_direction_normalizer_eps
        )
        self.self_forced_detach_block_transition = (
            bool(getattr(self.self_forced_config, "detach_block_transition", False))
            if self.self_forced_config is not None
            else False
        )
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
        self.self_forced_estimator_updates_per_step = (
            max(1, int(getattr(self.self_forced_config, "estimator_updates_per_step", 1)))
            if self.self_forced_config is not None
            else 1
        )
        self.self_forced_estimator_lr = self.lr / float(
            self.self_forced_estimator_updates_per_step
        )
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
        self.self_forced_use_physics = (
            bool(getattr(self.self_forced_config, "use_control_space_physics_regularization", False))
            if self.self_forced_config is not None
            else False
        )
        self.self_forced_physics_weight = (
            float(getattr(self.self_forced_config, "physics_weight", 0.0))
            if self.self_forced_config is not None
            else 0.0
        )
        self.self_forced_physics_force_fp32 = False
        self.self_forced_target_teacher = None
        self.self_forced_generated_estimator = None
        self.self_forced_generator_ema = None
        self._self_forced_aux_loaded_from_checkpoint = False
        self._self_forced_generator_ema_loaded_from_checkpoint = False
        self._self_forced_backward_context: Dict[str, Tensor] | None = None
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
            physics_config = getattr(
                self.self_forced_config,
                "physics",
                getattr(draft_config, "physics", None),
            )
            if self.self_forced_use_physics and physics_config is not None:
                self.self_forced_physics_force_fp32 = bool(getattr(physics_config, "force_fp32", True))
                self.self_forced_regularizer = DraftPhysicsRegularizer(
                    dt=float(getattr(physics_config, "dt", 0.1)),
                    pos_scale_m=float(getattr(physics_config, "pos_scale_m", 20.0)),
                    speed_floor_mps=float(getattr(physics_config, "speed_floor_mps", 0.5)),
                    vehicle_v_max_mps=float(getattr(physics_config, "vehicle_v_max_mps", 35.0)),
                    vehicle_a_max_mps2=float(getattr(physics_config, "vehicle_a_max_mps2", 8.0)),
                    vehicle_lat_accel_max_mps2=float(
                        getattr(physics_config, "vehicle_lat_accel_max_mps2", 4.2)
                    ),
                    bicycle_v_max_mps=float(getattr(physics_config, "bicycle_v_max_mps", 22.0)),
                    bicycle_a_max_mps2=float(getattr(physics_config, "bicycle_a_max_mps2", 5.5)),
                    bicycle_lat_accel_max_mps2=float(
                        getattr(physics_config, "bicycle_lat_accel_max_mps2", 4.4)
                    ),
                    pedestrian_v_max_mps=float(getattr(physics_config, "pedestrian_v_max_mps", 5.0)),
                    pedestrian_a_max_mps2=float(getattr(physics_config, "pedestrian_a_max_mps2", 4.7)),
                    vehicle_wheelbase_scale=float(getattr(physics_config, "vehicle_wheelbase_scale", 0.60)),
                    bicycle_wheelbase_scale=float(getattr(physics_config, "bicycle_wheelbase_scale", 0.85)),
                    vehicle_steer_max_rad=float(getattr(physics_config, "vehicle_steer_max_rad", 0.55)),
                    bicycle_steer_max_rad=float(getattr(physics_config, "bicycle_steer_max_rad", 1.00)),
                    vehicle_steer_rate_max_radps=float(
                        getattr(physics_config, "vehicle_steer_rate_max_radps", 0.8)
                    ),
                    bicycle_steer_rate_max_radps=float(
                        getattr(physics_config, "bicycle_steer_rate_max_radps", 1.5)
                    ),
                    soft_weight=float(
                        getattr(
                            physics_config,
                            "soft_weight",
                            getattr(
                                physics_config,
                                "vehicle_soft_weight",
                                getattr(
                                    physics_config,
                                    "bicycle_soft_weight",
                                    getattr(physics_config, "pedestrian_soft_weight", 0.25),
                                ),
                            ),
                        )
                    ),
                    compare_softness_to_gt=bool(getattr(physics_config, "compare_softness_to_gt", False)),
                    pedestrian_heading_weight=float(getattr(physics_config, "pedestrian_heading_weight", 0.05)),
                    pedestrian_heading_speed_threshold_mps=float(
                        getattr(physics_config, "pedestrian_heading_speed_threshold_mps", 0.5)
                    ),
                    eps=float(getattr(physics_config, "eps", 1e-6)),
                )
            else:
                self.self_forced_regularizer = None
        else:
            self.self_forced_regularizer = None
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
                4) official 점수에 사용할 batch 개수가 1 이상임
        """
        return (
            self.val_closed_loop
            and not self.val_open_loop
            and not self.sim_agents_submission.is_active
            and int(self.n_batch_sim_agents_metric) > 0
        )

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

    def _should_compute_closed_loop_minade(self) -> bool:
        """현재 validation에서 closed-loop minADE를 계산할지 판단합니다.

        학습 중 빠른 validation에서는 checkpoint 선택에 쓰는 official 점수만
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

    def _open_loop_denoise_metrics(
        self,
        pred_dict: Dict[str, Tensor],
    ) -> tuple[Tensor, Dict[str, Tensor], int]:
        """잡음 제거 방식 검증 점수와 유효 표본 수를 계산합니다.

        Args:
            pred_dict: flow decoder가 낸 출력 사전입니다.
                ``flow_pred_norm`` 과 ``flow_target_norm`` 의 shape은
                ``[n_valid_anchor, flow_window_steps, 4]`` 입니다.
                ``flow_loss_mask`` 가 있으면 shape은
                ``[n_valid_anchor, flow_window_steps]`` 입니다.

        Returns:
            tuple[Tensor, Dict[str, Tensor], int]:
                flow matching loss, meter/degree 단위 지표 사전,
                그리고 유효 anchor 개수입니다.
        """
        loss_mask = pred_dict.get("flow_loss_mask")
        loss = flow_matching_loss(
            pred_dict["flow_pred_norm"],
            pred_dict["flow_target_norm"],
            valid_mask=loss_mask,
        )
        metric_dict = self._build_open_loop_metric_dict(
            pred_clean_norm=pred_dict["flow_pred_clean_norm"],
            target_clean_norm=pred_dict["flow_clean_norm"],
            valid_mask=loss_mask,
        )
        sample_count = int(pred_dict["flow_clean_norm"].shape[0])
        return loss, metric_dict, sample_count

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
        scenario_seeds = [
            self._make_closed_loop_seed(scenario_id=scenario_id, rollout_idx=rollout_idx)
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
        return {
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
            pred = rollout_encoder.rollout_from_cache(
                rollout_cache=rollout_cache,
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
                sampling_scheme=self.validation_rollout_sampling,
                scenario_sampling_seeds=scenario_sampling_seeds,
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
            ``full_flow_decoder`` 는 draft fine-tuning의 ``train_full_flow_decoder_only=true`` 처럼
            마지막 궤적 생성부만 엽니다.
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

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """self-forced resume 여부를 기록합니다.

        Args:
            checkpoint: Lightning checkpoint dictionary입니다.

        Returns:
            None
        """
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

    def _manual_backward_without_autocast(self, loss: Tensor) -> None:
        """manual optimization의 backward만 autocast 밖에서 실행합니다.

        Args:
            loss: backward를 수행할 scalar loss입니다.

        Returns:
            None
        """
        with torch.autocast(device_type=loss.device.type, enabled=False):
            self.manual_backward(loss.float())

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

    def _predict_path_flow_clean_estimate(
        self,
        decoder: SMARTFlowDecoder,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        noisy_path_norm: Tensor,
        tau: Tensor,
        anchor_mask: Tensor,
    ) -> Dict[str, Tensor]:
        """주어진 decoder가 noisy N초 path를 어떻게 clean path로 보는지 계산합니다.

        Args:
            decoder: ``F_rho`` 또는 ``F_psi`` 역할의 decoder입니다.
            tokenized_map: 평가 모드 map token 사전입니다.
            tokenized_agent: 평가 모드 agent token 사전입니다.
            noisy_path_norm: noisy path입니다. shape은 ``[n_valid_agent, F_win, 4]`` 입니다.
            tau: flow interpolation time입니다. shape은 ``[n_valid_agent]`` 입니다.
            anchor_mask: 첫 anchor에서 사용할 agent mask입니다. shape은 ``[n_agent]`` 입니다.

        Returns:
            Dict[str, Tensor]: ``velocity`` 와 ``clean`` 을 담은 사전입니다.
        """
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
            "physics_loss": zero,
            "anchor_loss": zero,
            "total_loss": zero,
        }
        metric_dict.update(self._build_zero_draft_metrics(reference))
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
            )
        finally:
            self._restore_module_training_modes(encoder_modes)

    def _pack_self_forced_committed_rollout(
        self,
        rollout: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """committed rollout을 첫 anchor 기준 packed N초 path로 변환합니다.

        Args:
            rollout: ``_run_self_forced_rollout`` 의 출력입니다.
            tokenized_agent: 평가 모드 agent token 사전입니다.

        Returns:
            tuple[Tensor, Tensor]: packed path와 agent mask입니다.
                packed path shape은 ``[n_valid_agent, F_win, 4]`` 이고,
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
        return committed_path_norm[anchor_mask], anchor_mask

    def _update_generated_path_flow_estimator(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        committed_path_norm: Tensor,
        anchor_mask: Tensor,
    ) -> Tensor:
        """detached self-rollout으로 generated estimator F_psi를 online 업데이트합니다.

        Args:
            tokenized_map: 평가 모드 map token 사전입니다.
            tokenized_agent: 평가 모드 agent token 사전입니다.
            committed_path_norm: Generator가 실제로 실행한 N초 path입니다.
                shape은 ``[n_valid_agent, F_win, 4]`` 입니다.
            anchor_mask: 첫 anchor에서 사용할 agent mask입니다.
                shape은 ``[n_agent]`` 입니다.

        Returns:
            Tensor: 마지막 estimator update의 flow matching loss입니다.

        Notes:
            noising tau는 random terminal N과 독립적으로 전체 tau 구간에서 샘플링합니다.
        """
        if self.self_forced_generated_estimator is None:
            raise RuntimeError("self_forced_generated_estimator is not initialized.")

        optimizer = self.optimizers()[1]
        last_loss = committed_path_norm.new_zeros(())

        self.toggle_optimizer(optimizer)
        self.self_forced_target_teacher.eval()
        self.self_forced_generated_estimator.train()
        try:
            for _ in range(self.self_forced_estimator_updates_per_step):
                optimizer.zero_grad(set_to_none=True)
                self._prepare_self_forced_estimator_backward_boundary()
                clean_path = committed_path_norm.detach()
                flow_sample = self.self_forced_generated_estimator.agent_encoder.flow_ode.sample(
                    clean_path,
                    target_type="velocity",
                )
                pred_dict = self._predict_path_flow_clean_estimate(
                    decoder=self.self_forced_generated_estimator,
                    tokenized_map=tokenized_map,
                    tokenized_agent=tokenized_agent,
                    noisy_path_norm=flow_sample.x_t,
                    tau=flow_sample.tau,
                    anchor_mask=anchor_mask,
                )
                last_loss = flow_matching_loss(pred_dict["velocity"], flow_sample.target)
                self._manual_backward_without_autocast(last_loss)
                self._assert_self_forced_estimator_update_isolated()
                self.clip_gradients(
                    optimizer,
                    gradient_clip_val=self.self_forced_gradient_clip_val,
                    gradient_clip_algorithm="norm",
                )
                optimizer.step()
                self._clear_self_forced_auxiliary_gradients()
                self._clear_self_forced_generator_gradients()
        finally:
            self.untoggle_optimizer(optimizer)
            self._set_self_forced_auxiliary_modes()
        return last_loss.detach()

    def _compute_self_forced_direction(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        committed_path_norm: Tensor,
        anchor_mask: Tensor,
    ) -> Tensor:
        """clean-DMD 방향을 고정된 평가자 출력으로 계산합니다.

        Args:
            tokenized_map: map token 사전입니다.
            tokenized_agent: agent token 사전입니다.
            committed_path_norm: Generator가 closed-loop로 실제 실행한 path입니다.
                shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
            anchor_mask: 첫 anchor 기준으로 유효한 agent mask입니다.
                shape은 ``[n_agent]`` 입니다.

        Returns:
            Tensor: 현재 committed path에 더할 정규화된 DMD 방향입니다.
            shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.

        설명:
            Generator update에서 target teacher와 generated estimator는 학습 대상이 아닙니다.
            두 모델은 같은 noisy path를 보고 clean path 추정을 내는 평가자로만 쓰입니다.
            그래서 모든 보조 모델 호출은 ``no_grad``로 감싸고, 최종 방향도 detach합니다.
        """
        if self.self_forced_target_teacher is None or self.self_forced_generated_estimator is None:
            raise RuntimeError("self-forced auxiliary models are not initialized.")

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
            path_delta = build_clean_dmd_direction(
                committed_path_norm=clean_for_guidance,
                target_clean_norm=target_pred["clean"],
                generated_clean_norm=generated_pred["clean"],
                normalizer_eps=self.self_forced_direction_normalizer_eps,
            )

        self._assert_self_forced_generator_update_isolated()
        return path_delta.to(dtype=committed_path_norm.dtype).detach()


    def _sample_self_forced_guidance_flow_state(self, clean_path_norm: Tensor):
        """SiD/DMD teacher query에 쓸 noisy path를 샘플링합니다.

        Args:
            clean_path_norm: Generator가 만든 clean path입니다.
                shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.

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
            committed_path_norm: Generator가 실제로 실행한 path입니다.
                shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
            anchor_mask: 첫 anchor에서 사용할 agent mask입니다.
                shape은 ``[n_agent]`` 입니다.

        Returns:
            tuple[Tensor, Tensor]: ``target_clean_norm`` 과 ``generated_clean_norm`` 입니다.
                각 shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
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
            committed_path_norm: Generator가 실제로 실행한 path ``X`` 입니다.
                shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
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
            committed_path_norm: Generator가 실제로 실행한 path입니다.
                shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
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

        path_delta = self._compute_self_forced_direction(
            tokenized_map=tokenized_map,
            tokenized_agent=tokenized_agent,
            committed_path_norm=committed_path_norm,
            anchor_mask=anchor_mask,
        )
        target_path_norm = (committed_path_norm + self.self_forced_path_step_size * path_delta).detach()
        self._set_self_forced_backward_context(
            committed_path_norm=committed_path_norm,
            path_delta=path_delta,
            target_path_norm=target_path_norm,
        )
        return masked_mean_square_loss(committed_path_norm, target_path_norm)

    def _compute_self_forced_physics_loss(
        self,
        committed_path_norm: Tensor,
        tokenized_agent: Dict[str, Tensor],
        anchor_mask: Tensor,
    ) -> Dict[str, Tensor]:
        """실제로 실행된 committed N초 self-rollout에만 physics loss를 겁니다.

        Args:
            committed_path_norm: packed committed rollout입니다. shape은
                ``[n_valid_agent, F_win, 4]`` 입니다.
            tokenized_agent: 평가 모드 agent token 사전입니다.
            anchor_mask: 첫 anchor에서 사용할 agent mask입니다. shape은 ``[n_agent]`` 입니다.

        Returns:
            Dict[str, Tensor]: physics loss와 세부 항입니다.
        """
        if (
            not self.self_forced_use_physics
            or self.self_forced_regularizer is None
            or committed_path_norm.numel() == 0
        ):
            return self._build_zero_draft_metrics(committed_path_norm)

        physics_inputs = build_anchor0_physics_inputs(
            tokenized_agent=tokenized_agent,
            anchor_mask=anchor_mask,
        )
        if not self.self_forced_physics_force_fp32:
            physics_dict = self.self_forced_regularizer(
                pred_future_norm=committed_path_norm,
                target_future_norm=committed_path_norm.detach(),
                packed_agent_type=physics_inputs["agent_type"],
                packed_agent_length=physics_inputs["agent_length"],
                packed_prev_control=physics_inputs["prev_control"],
                packed_prev_control_valid=physics_inputs["prev_control_valid"],
            )
            if not all(torch.isfinite(value).all() for value in physics_dict.values()):
                return self._build_zero_draft_metrics(committed_path_norm)
            return physics_dict

        with torch.autocast(device_type=committed_path_norm.device.type, enabled=False):
            physics_dict = self.self_forced_regularizer(
                pred_future_norm=committed_path_norm.float(),
                target_future_norm=committed_path_norm.detach().float(),
                packed_agent_type=physics_inputs["agent_type"],
                packed_agent_length=physics_inputs["agent_length"].float(),
                packed_prev_control=physics_inputs["prev_control"].float(),
                packed_prev_control_valid=physics_inputs["prev_control_valid"],
            )
        if not all(torch.isfinite(value).all() for value in physics_dict.values()):
            return self._build_zero_draft_metrics(committed_path_norm)
        return physics_dict

    def on_fit_start(self) -> None:
        """학습 시작 전에 빠른 closed-loop validation 모드를 켭니다.

        Lightning은 ``on_fit_start`` 를 sanity check 전에 호출합니다.
        그래서 여기서 validation batch 개수를 줄이면 학습 전 sanity check와
        학습 중 validation 둘 다 같은 빠른 규칙을 사용하게 됩니다.

        Returns:
            None
        """
        self._apply_fit_time_validation_batch_limit()
        self._sync_self_forced_auxiliary_models()
        self._prepare_self_forced_generator_ema()

    def on_fit_end(self) -> None:
        """학습이 끝나면 임시로 바꾼 validation 제한 값을 정리합니다.

        Returns:
            None
        """
        self._restore_fit_time_validation_batch_limit()


    def _get_draft_loss_weight(self) -> float:
        """현재 epoch에서 사용할 DRaFT physics 가중치를 계산합니다.

        Returns:
            float:
                warm-up 이전이면 ``0.0`` 이고,
                그 뒤에는 설정한 최대값까지 선형으로 올라갑니다.
        """
        if not self.draft_enabled or self.draft_max_weight <= 0.0:
            return 0.0

        current_epoch = int(self.current_epoch)
        if current_epoch < self.draft_start_epoch:
            return 0.0

        if self.draft_ramp_epochs <= 1:
            return self.draft_max_weight

        progress = (current_epoch - self.draft_start_epoch + 1) / float(self.draft_ramp_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return self.draft_max_weight * progress

    def _build_zero_draft_metrics(self, reference: Tensor) -> Dict[str, Tensor]:
        """DRaFT logging에 필요한 0 metric 사전을 만듭니다."""
        zero = reference.new_zeros(())
        metric_dict = {
            "loss": zero,
            "raw_pred_loss": zero,
        }
        for key in DRAFT_PHYSICS_COMPONENT_KEYS:
            metric_dict[key] = zero
        for key in DRAFT_PHYSICS_ACTUAL_UNIT_KEYS:
            metric_dict[key] = zero
            metric_dict[f"pred_{key}"] = zero
            metric_dict[f"gt_{key}"] = zero
        return metric_dict

    def _find_first_nonfinite_parameter(self) -> tuple[str, Tensor] | None:
        """처음 발견한 non-finite trainable parameter를 반환합니다."""
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if not torch.isfinite(param).all():
                return name, param
        return None

    def _find_first_nonfinite_gradient(self) -> tuple[str, Tensor] | None:
        """처음 발견한 non-finite gradient를 반환합니다."""
        for name, param in self.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return name, param.grad
        return None

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

    def _sample_draft_eval_future(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        """DRaFT physics target trajectory를 inference와 같은 eval mode에서 생성합니다."""
        encoder_modes = self._switch_module_to_eval_preserving_modes(self.encoder)
        try:
            draft_context = self.encoder.build_anchor_context(
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
                anchor_mask_key="flow_train_mask",
            )
            pred_sample_norm = self.encoder.sample_open_loop_future(
                anchor_hidden=draft_context["anchor_hidden"],
                anchor_mask=draft_context["anchor_mask"],
                sampling_scheme=self.draft_sampling,
            )
        finally:
            self._restore_module_training_modes(encoder_modes)
        return pred_sample_norm, draft_context

    def _compute_draft_training_loss(
        self,
        pred_dict: Dict[str, Tensor],
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """실제 샘플러를 돌린 최종 미래에 physics loss를 계산합니다.

        Args:
            pred_dict: flow decoder 출력 사전입니다.
                ``anchor_hidden`` 은 ``[n_agent, 13, hidden_dim]`` 이고,
                ``flow_clean_norm`` 은 ``[n_valid_anchor, 20, 4]`` 입니다.
            tokenized_map: 학습용 맵 토큰 사전입니다.
            tokenized_agent: 학습용 에이전트 토큰 사전입니다.
                DRaFT용 packed 메타데이터가 들어 있어야 합니다.

        Returns:
            Dict[str, Tensor]:
                총 physics loss와 세부 항을 담은 사전입니다.
        """
        if (
            not self.draft_enabled
            or self.draft_regularizer is None
            or pred_dict["flow_clean_norm"].numel() == 0
        ):
            return self._build_zero_draft_metrics(pred_dict["flow_clean_norm"])

        # pred_sample_norm : [n_valid_anchor, 20, 4]
        pred_sample_norm, draft_pred = self._sample_draft_eval_future(
            tokenized_map=tokenized_map,
            tokenized_agent=tokenized_agent,
        )
        draft_target_norm = draft_pred["flow_clean_norm"]
        draft_loss_mask = draft_pred.get("flow_loss_mask")
        if not torch.isfinite(pred_sample_norm).all():
            return self._build_zero_draft_metrics(draft_target_norm)

        if pred_sample_norm.shape[0] != tokenized_agent["flow_train_agent_type"].shape[0]:
            raise ValueError(
                "DRaFT 샘플 개수와 packed anchor 메타데이터 개수가 다릅니다. "
                f"got {pred_sample_norm.shape[0]} and {tokenized_agent['flow_train_agent_type'].shape[0]}"
            )

        if not self.draft_physics_force_fp32:
            physics_dict = self.draft_regularizer(
                pred_future_norm=pred_sample_norm,
                target_future_norm=draft_target_norm,
                packed_agent_type=tokenized_agent["flow_train_agent_type"],
                packed_agent_length=tokenized_agent["flow_train_agent_length"],
                packed_prev_control=tokenized_agent["flow_train_prev_control"],
                packed_prev_control_valid=tokenized_agent["flow_train_prev_control_valid"],
                future_valid_mask=draft_loss_mask,
            )
            if not all(torch.isfinite(value).all() for value in physics_dict.values()):
                return self._build_zero_draft_metrics(draft_target_norm)
            return physics_dict

        # Keep the threshold-heavy physics penalty in fp32 even when the trainer
        # runs with bf16 autocast, while preserving gradients to pred_sample_norm.
        with torch.autocast(device_type=pred_sample_norm.device.type, enabled=False):
            physics_dict = self.draft_regularizer(
                pred_future_norm=pred_sample_norm.float(),
                target_future_norm=draft_target_norm.float(),
                packed_agent_type=tokenized_agent["flow_train_agent_type"],
                packed_agent_length=tokenized_agent["flow_train_agent_length"].float(),
                packed_prev_control=tokenized_agent["flow_train_prev_control"].float(),
                packed_prev_control_valid=tokenized_agent["flow_train_prev_control_valid"],
                future_valid_mask=draft_loss_mask,
            )
        if not all(torch.isfinite(value).all() for value in physics_dict.values()):
            return self._build_zero_draft_metrics(draft_target_norm)
        return physics_dict

    def _log_draft_training_metrics(
        self,
        draft_weight: float,
        physics_dict: Dict[str, Tensor],
    ) -> None:
        """DRaFT fine-tuning용 학습 로그를 기록합니다.

        Args:
            draft_weight: 현재 batch에 적용한 physics loss 가중치입니다.
            physics_dict: physics loss 계산 결과 사전입니다.

        Returns:
            None
        """
        self.log(
            "train/draft_weight",
            float(draft_weight),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        self.log(
            "train/loss_phys",
            physics_dict["loss"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        self.log(
            "train/loss_if",
            physics_dict["loss"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        self.log(
            "train/loss_phys_raw",
            physics_dict["raw_pred_loss"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        self.log(
            "train/loss_if_raw",
            physics_dict["raw_pred_loss"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        for metric_name in DRAFT_PHYSICS_COMPONENT_KEYS:
            self.log(
                f"draft_component/{metric_name}",
                physics_dict[metric_name],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
        for metric_name in DRAFT_PHYSICS_ACTUAL_UNIT_KEYS:
            self.log(
                f"draft_actual_pred/{metric_name}",
                physics_dict[f"pred_{metric_name}"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                f"draft_actual_gt/{metric_name}",
                physics_dict[f"gt_{metric_name}"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )

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
        )
        fm_loss, open_metric_dict, _ = self._open_loop_denoise_metrics(pred)
        draft_weight = self._get_draft_loss_weight()
        physics_dict = self._build_zero_draft_metrics(fm_loss)
        total_loss = fm_loss
        if draft_weight > 0.0:
            physics_dict = self._compute_draft_training_loss(
                pred_dict=pred,
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
            )
            total_loss = total_loss + draft_weight * 0.005 * physics_dict["loss"]

        generator_optimizer = self.optimizers()[0]
        self.toggle_optimizer(generator_optimizer)
        generator_optimizer.zero_grad(set_to_none=True)
        self._prepare_self_forced_generator_backward_boundary()
        self._manual_backward_without_autocast(total_loss)
        self._assert_self_forced_generator_update_isolated()
        self.clip_gradients(
            generator_optimizer,
            gradient_clip_val=self.self_forced_gradient_clip_val,
            gradient_clip_algorithm="norm",
        )
        generator_optimizer.step()
        self._update_self_forced_generator_ema_after_step()
        self._clear_self_forced_generator_gradients()
        self.untoggle_optimizer(generator_optimizer)

        self.log("train/loss", total_loss.detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/loss_fm", fm_loss.detach(), on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
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
        if self.draft_enabled:
            self._log_draft_training_metrics(
                draft_weight=draft_weight,
                physics_dict=physics_dict,
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
        if self.self_forced_use_anchor_fm_loss:
            tokenized_map_train, tokenized_agent_train = self.token_processor(data)
            pred = self.encoder(
                tokenized_map_train,
                tokenized_agent_train,
                anchor_mask_key="flow_train_mask",
            )
            fm_loss, open_metric_dict, _ = self._open_loop_denoise_metrics(pred)

        tokenized_map_eval, tokenized_agent_eval = self._build_eval_tokenized_inputs(data)
        rollout = self._run_self_forced_rollout(tokenized_map_eval, tokenized_agent_eval)
        committed_path_norm, anchor_mask = self._pack_self_forced_committed_rollout(
            rollout=rollout,
            tokenized_agent=tokenized_agent_eval,
        )
        if committed_path_norm.numel() == 0:
            if fm_loss is None:
                # DDP requires backward on every rank to participate in the
                # gradient all-reduce. Walk Generator parameters with a
                # zero-coefficient sum so autograd traverses the param graph
                # and DDP completes its all-reduce; skip optimizer.step()
                # because the gradients are deterministically zero.
                zero_loss = sum(
                    p.sum() for p in self.encoder.parameters() if p.requires_grad
                ) * 0.0
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
                self.log("train/sf_anchor_fm_enabled", 0.0, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
                self.log("train/sf_anchor_weight", float(self.self_forced_anchor_weight), on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
                return zero_loss.detach()
            generator_optimizer = self.optimizers()[0]
            self.toggle_optimizer(generator_optimizer)
            generator_optimizer.zero_grad(set_to_none=True)
            self._prepare_self_forced_generator_backward_boundary()
            self._manual_backward_without_autocast(fm_loss)
            self._assert_self_forced_generator_update_isolated()
            generator_optimizer.step()
            self._update_self_forced_generator_ema_after_step()
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
        )
        sf_loss = self._compute_self_forced_distribution_matching_loss(
            tokenized_map=tokenized_map_eval,
            tokenized_agent=tokenized_agent_eval,
            committed_path_norm=committed_path_norm,
            anchor_mask=anchor_mask,
        )
        physics_dict = self._compute_self_forced_physics_loss(
            committed_path_norm=committed_path_norm,
            tokenized_agent=tokenized_agent_eval,
            anchor_mask=anchor_mask,
        )
        anchor_loss = (
            fm_loss
            if fm_loss is not None
            else committed_path_norm.new_zeros(())
        )
        total_loss = (
            self.self_forced_weight * sf_loss
            + self.self_forced_anchor_weight * anchor_loss
            + self.self_forced_physics_weight * physics_dict["loss"]
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
                self.clip_gradients(
                    generator_optimizer,
                    gradient_clip_val=self.self_forced_gradient_clip_val,
                    gradient_clip_algorithm="norm",
                )
                generator_optimizer.step()
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
        self.log("train/sf_physics_loss", physics_dict["loss"].detach(), on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_anchor_fm_enabled", float(self.self_forced_use_anchor_fm_loss), on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_anchor_loss", anchor_loss.detach(), on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_anchor_weight", float(self.self_forced_anchor_weight), on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_physics_weight", float(self.self_forced_physics_weight), on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
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
        if open_metric_dict is not None:
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
        if self.self_forced_use_physics:
            self._log_draft_training_metrics(
                draft_weight=float(self.self_forced_physics_weight),
                physics_dict=physics_dict,
            )
        return total_loss.detach()

    def training_step(self, data, batch_idx):
        """한 batch의 FM loss와 DRaFT physics loss를 함께 계산합니다.

        Args:
            data: 학습용 장면 배치입니다.
            batch_idx: 현재 batch 번호입니다.

        Returns:
            Tensor: 최종 학습 loss입니다.
        """
        bad_param = self._find_first_nonfinite_parameter()
        if bad_param is not None:
            bad_name, bad_tensor = bad_param
            raise RuntimeError(
                "Detected non-finite trainable parameter before forward pass: "
                f"{bad_name} ({self._summarize_nonfinite_tensor(bad_tensor)})"
            )
        if self.self_forced_enabled:
            if self._is_self_forced_active():
                return self._training_step_self_forced(data=data, batch_idx=batch_idx)
            return self._training_step_manual_open_loop(data=data, batch_idx=batch_idx)
        """ tokenized_agent
flow_train_agent_type [n_valid_anchor]
flow_train_agent_length [n_valid_anchor]
flow_train_prev_control [n_valid_anchor, 3]
flow_train_prev_control_valid [n_valid_anchor]

        """
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
        )
        """
fm_loss: 
    Tensor shape []
open_metric_dict: 
    Dict[str, Tensor]
        """
        fm_loss, open_metric_dict, _ = self._open_loop_denoise_metrics(pred)

        draft_weight = self._get_draft_loss_weight()
        """ physics_dict : Dict[str, Tensor] # 모든 값은 scalar tensor

        loss, raw_pred_loss

        vehicle_hard, vehicle_soft, vehicle_total
        bicycle_hard, bicycle_soft, bicycle_total
        pedestrian_hard, pedestrian_soft, pedestrian_head, pedestrian_total

        pred_speed_excess_mps, pred_accel_excess_mps2,
        pred_steer_excess_deg, pred_steer_rate_excess_degps,
        pred_lat_accel_excess_mps2, pred_heading_error_deg
        """
        physics_dict = self._build_zero_draft_metrics(fm_loss)
        total_loss = fm_loss
        if draft_weight > 0.0:
            physics_dict = self._compute_draft_training_loss(
                pred_dict=pred,
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
            )
            total_loss = total_loss + draft_weight * 0.005 * physics_dict["loss"]
        if not torch.isfinite(fm_loss):
            raise RuntimeError(f"Non-finite fm_loss detected: {self._summarize_nonfinite_tensor(fm_loss)}")
        if not torch.isfinite(total_loss):
            raise RuntimeError(
                "Non-finite total_loss detected: "
                f"{self._summarize_nonfinite_tensor(total_loss)}"
            )

        self.log("train/loss", total_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/loss_fm", fm_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log(
            f"train/{self.train_open_metric_names['ade']}",
            open_metric_dict[self.open_metric_names["ade"]],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        self.log(
            f"train/{self.train_open_metric_names['fde']}",
            open_metric_dict[self.open_metric_names["fde"]],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        self.log(
            f"train/{self.train_open_metric_names['yaw_ade']}",
            open_metric_dict[self.open_metric_names["yaw_ade"]],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        self.log(
            f"train/{self.train_open_metric_names['yaw_fde']}",
            open_metric_dict[self.open_metric_names["yaw_fde"]],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        if self.draft_enabled:
            self._log_draft_training_metrics(
                draft_weight=draft_weight,
                physics_dict=physics_dict,
            )
        return total_loss

    def on_after_backward(self) -> None:
        """역전파 직후 non-finite gradient를 fail-fast로 잡습니다."""
        bad_grad = self._find_first_nonfinite_gradient()
        if bad_grad is None:
            return
        bad_name, bad_tensor = bad_grad
        raise RuntimeError(
            "Detected non-finite gradient after backward: "
            f"{bad_name} ({self._summarize_nonfinite_tensor(bad_tensor)})"
            f"{self._format_self_forced_backward_context()}"
        )

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
            open_metric_dict = self._build_open_loop_metric_dict(
                pred_clean_norm=open_pred_clean_norm,
                target_clean_norm=denoise_pred["flow_clean_norm"],
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
                lr=self.self_forced_estimator_lr,
            )
            return [generator_optimizer, generated_estimator_optimizer]

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]

    def on_train_epoch_end(self) -> None:
        """self-forced manual optimization에서 scheduler가 있으면 epoch마다 한 번 진행합니다.

        Returns:
            None
        """
        if not self.self_forced_enabled:
            return
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
        self.sim_agents_submission.save_sub_file()
