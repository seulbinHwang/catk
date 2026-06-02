from __future__ import annotations

import copy
import gc
import hashlib
import math
import torch.nn.functional as F
from contextlib import nullcontext
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
    masked_mean_square_loss,
)
from src.smart.modules.self_forced_update_separation import (
    assert_no_module_gradients,
    clear_module_gradients,
    detach_tensor_tree,
    module_gradients_disabled,
    temporarily_clear_module_gradients,
)
from src.smart.modules.self_forced_estimator_warmup import (
    is_self_forced_estimator_warmup_epoch,
    is_self_forced_estimator_warmup_step,
    resolve_self_forced_estimator_warmup_epochs,
    resolve_self_forced_estimator_warmup_steps,
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
        self.weight_decay = float(getattr(model_config, "weight_decay", 0.01))
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

        # ── OCSC (Open-Closed Self-Consistency) fine-tuning mode ──────────
        # finetune.mode == "ocsc_ft" 이고 enabled=True 면 training_step 이
        # self_forced 대신 _run_flow_ocsc_ft_step 으로 분기.
        # 별도 ref_flow_decoder (frozen pretrained flow_decoder deepcopy) 만 둠.
        self.finetune_config = model_config.finetune
        self.ref_flow_decoder: nn.Module | None = None
        self._ref_flow_decoder_loaded_from_ckpt = False
        ocsc_use_ref = bool(
            self._is_ocsc_ft_enabled()
            and getattr(self.finetune_config, "ocsc_use_pretrained_ref", True)
        )
        if ocsc_use_ref:
            self.ref_flow_decoder = copy.deepcopy(self.encoder.agent_encoder.flow_decoder)
            for p in self.ref_flow_decoder.parameters():
                p.requires_grad_(False)

        # velocity_head_only: set_model_for_finetuning 이 step_refiner 도 unfreeze 했으면
        # 다시 freeze 해서 velocity_head 만 trainable (OCSC clean 의 flow_velocity_head_only 정합).
        if self._is_ocsc_ft_enabled() and bool(
            getattr(self.finetune_config, "velocity_head_only", False)
        ):
            step_refiner = getattr(self.encoder.agent_encoder.flow_decoder, "step_refiner", None)
            if step_refiner is not None:
                for p in step_refiner.parameters():
                    p.requires_grad_(False)

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
        # Self-Forcing entropy knob (constant β; β annealing은 추후 작업).
        # β=1.0 → 기존 동작 그대로. β<1 → fake 항 1/β 배 (entropy↑). β>1 → sharpening.
        self.self_forced_dmd_beta = (
            float(getattr(self.self_forced_config, "dmd_beta", 1.0))
            if self.self_forced_config is not None
            else 1.0
        )
        if not (self.self_forced_dmd_beta > 0.0):
            raise ValueError(
                f"self_forced.dmd_beta must be > 0, got {self.self_forced_dmd_beta}."
            )
        # OCSC self_forcing_dmd 정합: Self-Forcing abs-mean normalizer (|pred_x0_real|) on/off.
        self.self_forced_dmd_normalize = (
            bool(getattr(self.self_forced_config, "dmd_normalize", True))
            if self.self_forced_config is not None
            else True
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
        self.self_forced_estimator_updates_per_step = (
            max(1, int(getattr(self.self_forced_config, "estimator_updates_per_step", 1)))
            if self.self_forced_config is not None
            else 1
        )
        # OCSC dmd_n_rollouts 정합 — 같은 anchor 0 에서 N rollout 으로 DMD direction variance↓.
        # 구현은 batch-replicate: scenario 를 N 장 복제해 단일 rollout 한 번에 N noise tape 처리.
        # walltime ≈ 1× (sequential 대비), VRAM 은 ~N 배.  critic / DMD direction 도 같은 packed
        # dim 에서 작동하므로 별도 분기 없이 평균에 포함됨.  default 1 = 기존 동작.
        self.self_forced_n_rollouts = (
            max(1, int(getattr(self.self_forced_config, "n_rollouts", 1)))
            if self.self_forced_config is not None
            else 1
        )
        # OCSC dmd_anchor_stride 정합 — 한 closed-loop rollout 안에서 stride 간격으로 여러
        # anchor 잡아 각각 DMD/SiD step.  학습 신호 N anchors 배.  default = 1 anchor (anchor 0
        # 만), behavior-preserving.  rollout 길이가 자동으로 (n_anchors-1)*stride + window/shift
        # 만큼 늘어남.
        self.self_forced_n_anchors = (
            max(1, int(getattr(self.self_forced_config, "n_anchors", 1)))
            if self.self_forced_config is not None
            else 1
        )
        self.self_forced_anchor_stride = (
            max(1, int(getattr(self.self_forced_config, "anchor_stride", 1)))
            if self.self_forced_config is not None
            else 1
        )
        # token anchor index 범위 가드: j 번째 anchor 의 token index = j*stride 가 anchor
        # grid(FLOW_TRAIN_ANCHOR_COUNT=16 / FLOW_CONTEXT_TOKEN_COUNT=18)를 넘으면 안 됨.
        # path_flow_velocity_for_anchor_k 는 ctx_hidden_pack 길이 >= 2+token 을 요구 → token<=16,
        # flow_eval_mask 는 16 anchors → token<=15.  보수적으로 (n_anchors-1)*stride<=15.
        if self.self_forced_enabled:
            _max_token_anchor = (int(self.self_forced_n_anchors) - 1) * int(self.self_forced_anchor_stride)
            if _max_token_anchor > 15:
                raise ValueError(
                    "self_forced (n_anchors-1)*anchor_stride must be <= 15 "
                    f"(anchor grid 한계), got n_anchors={int(self.self_forced_n_anchors)}, "
                    f"anchor_stride={int(self.self_forced_anchor_stride)} → max token anchor "
                    f"index {_max_token_anchor}.  n_anchors 또는 anchor_stride 를 줄이세요."
                )
        # anchor 추출(🅐): OCSC GT-grounded per-anchor rollout 으로 정식 구현됨.
        # _training_step_self_forced 가 anchor 마다 GT current 에서 출발하는 별도 rollout 을
        # 돌리고(_build_self_forced_anchor_rollout_tokens), pack 은 anchor_grounded=True 로
        # window 0 을 anchor k GT frame 기준 추출한다.  (구 🅑 의 단일 rollout 다중 window
        # 추출 = self-rollout drift 버그는 제거.)  gt_pos 는 full-episode coarse 라 슬라이스
        # 가능하고, exec fine-history 는 rollout_full_*_10hz 로 anchor current 에서 재생성.
        # critic(generated estimator) LR.  config에 estimator_lr이 양수로 명시되어 있으면
        # 그 절대값을 그대로 사용 (generator LR 과 동일하게 두고 싶을 때 유용).  값이 없거나
        # ``null`` / ``<= 0`` 이면 기존 비례 관계 ``lr / estimator_updates_per_step`` 을 그대로
        # 적용 — Self-Forcing 의 dfake_gen_update_ratio=5 관습과 일관.
        _estimator_lr_override = (
            getattr(self.self_forced_config, "estimator_lr", None)
            if self.self_forced_config is not None
            else None
        )
        if _estimator_lr_override is not None and float(_estimator_lr_override) > 0.0:
            self.self_forced_estimator_lr = float(_estimator_lr_override)
        else:
            self.self_forced_estimator_lr = self.lr / float(
                self.self_forced_estimator_updates_per_step
            )
        self.self_forced_estimator_warmup_epochs = (
            resolve_self_forced_estimator_warmup_epochs(self.self_forced_config)
        )
        self.self_forced_estimator_warmup_steps = (
            resolve_self_forced_estimator_warmup_steps(self.self_forced_config)
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
        self.self_forced_target_teacher = None
        self.self_forced_generated_estimator = None
        self.self_forced_generator_ema = None
        self._self_forced_aux_loaded_from_checkpoint = False
        self._self_forced_generator_ema_loaded_from_checkpoint = False
        self._self_forced_backward_context: Dict[str, Tensor] | None = None
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
        batch 수로 덮어씁니다. 평가는 batch 단위로 잘리므로 실제 scene 수는
        ``world_size * val_batch_size`` 의 배수가 됩니다.
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

        global_val_batch_size = world_size * val_batch_size
        n_batch_override = max(1, math.ceil(scorer_scene_num / global_val_batch_size))
        effective_scene_num = n_batch_override * global_val_batch_size
        self.n_batch_sim_agents_metric = int(n_batch_override)

        current_key = (
            int(scorer_scene_num),
            int(world_size),
            int(val_batch_size),
            int(effective_scene_num),
        )
        if self._scorer_scene_num_last_key == current_key:
            return
        self._scorer_scene_num_last_key = current_key
        if getattr(trainer, "is_global_zero", True):
            print(
                "[scorer_scene_num] Fast WOSAC sim_agents_2025 scorer batch 수를 "
                f"n_batch_sim_agents_metric={self.n_batch_sim_agents_metric} 으로 설정합니다 "
                f"(requested_scenes={scorer_scene_num}, world_size={world_size}, "
                f"val_batch_size={val_batch_size}, global_val_batch_size={global_val_batch_size}, "
                f"effective_scenes={effective_scene_num}).",
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
            open_metric_dict[self.open_metric_names["ade"]].detach(),
            open_metric_dict[self.open_metric_names["fde"]].detach(),
            open_metric_dict[self.open_metric_names["yaw_ade"]].detach(),
            open_metric_dict[self.open_metric_names["yaw_fde"]].detach(),
        ]
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

    def _is_self_forced_estimator_warmup_active(self) -> bool:
        """현재 epoch 또는 global step 이 generated estimator 사전 적응 구간인지 판단합니다.

        epoch 기반 warmup (``estimator_warmup_epochs``) 과 step 기반 warmup
        (``estimator_warmup_steps``) 중 하나라도 활성이면 warmup 으로 봅니다.
        둘 다 0 이면 즉시 generator+critic 동시 학습으로 들어갑니다.
        """
        epoch_active = is_self_forced_estimator_warmup_epoch(
            current_epoch=int(self.current_epoch),
            self_forced_start_epoch=int(self.self_forced_start_epoch),
            estimator_warmup_epochs=int(self.self_forced_estimator_warmup_epochs),
        )
        step_active = is_self_forced_estimator_warmup_step(
            global_step=int(self.global_step),
            estimator_warmup_steps=int(self.self_forced_estimator_warmup_steps),
        )
        return epoch_active or step_active

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
            self._set_self_forced_auxiliary_modes()
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

    def _manual_backward_without_autocast(self, loss: Tensor, retain_graph: bool = False) -> None:
        """manual optimization 의 backward 만 autocast 밖에서 실행합니다.

        Args:
            loss: backward 를 수행할 scalar loss.
            retain_graph: ``True`` 면 backward 후 graph 를 유지합니다.
                multi-anchor 학습 시 같은 rollout 결과를 여러 anchor 가 share 하므로
                마지막 anchor 전까지 ``True`` 로 두어야 graph free 시점이 안 겹칩니다.

        설명:
            ``loss.float()`` 으로 fp32 캐스팅을 유지합니다. ``precision='16-mixed'`` 인 경우
            Lightning 의 precision plugin 이 ``manual_backward`` 안에서 ``GradScaler.scale`` 을
            적용하므로, 이후 step 은 ``_clip_and_step_with_optional_scaler`` 를 통해
            unscale → clip → step → update 순서를 지킵니다.
        """
        with torch.autocast(device_type=loss.device.type, enabled=False):
            self.manual_backward(loss.float(), retain_graph=retain_graph)

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

    # vocab-shared / packed-eval 텐서는 replicate 대상이 아님.  첫 dim 이 n_agent 와
    # 우연히 같아도 잘못 늘리지 않게 명시 skip-set 으로 보호.
    _SELF_FORCED_REPLICATE_SKIP_KEYS = frozenset(
        {
            # vocab buffers (shape [vocab_size, ...])
            "trajectory_token_veh",
            "trajectory_token_ped",
            "trajectory_token_cyc",
            "token_bank_all_veh",
            "token_bank_all_ped",
            "token_bank_all_cyc",
            "token_traj_src",
            # packed eval-only tensors (self-forced critic/DMD 에서 미사용)
            "flow_eval_clean_norm",
            "flow_eval_clean_metric_norm",
            "flow_eval_agent_type",
            "flow_eval_agent_length",
            "flow_train_clean_norm",
            "flow_train_clean_metric_norm",
            "flow_train_loss_mask",
            "flow_train_agent_type",
            "flow_train_agent_length",
        }
    )

    def _replicate_token_dict_along_first_dim(
        self,
        token_dict: Dict[str, Tensor],
        first_dim_size: int,
        num_graphs: int,
        repeat_count: int,
    ) -> Dict[str, Tensor]:
        """첫 dim 이 ``first_dim_size`` 인 모든 tensor 를 ``repeat_count`` 배로 복제합니다.

        ``batch`` 는 ``+ i*num_graphs`` 로 offset, ``num_graphs`` 는 ``× repeat_count`` 로
        scale, vocab-shared / packed eval 텐서는 그대로 통과시킵니다.
        """
        if repeat_count == 1:
            return token_dict
        replicated: Dict[str, Tensor] = {}
        for key, value in token_dict.items():
            if key == "num_graphs":
                replicated[key] = int(value) * repeat_count
                continue
            if key == "batch":
                replicated[key] = self._expand_batch_index_for_rollouts(
                    value,
                    repeat_count=repeat_count,
                    num_graphs=num_graphs,
                )
                continue
            if key in self._SELF_FORCED_REPLICATE_SKIP_KEYS:
                replicated[key] = value
                continue
            if torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] == first_dim_size:
                replicated[key] = self._repeat_tensor_on_first_dim(value, repeat_count)
                continue
            replicated[key] = value
        return replicated

    def _build_self_forced_replicated_tokens(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        repeat_count: int,
    ) -> tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """self-forced rollout 을 N 배 batch 로 묶기 위해 두 dict 를 복제합니다.

        Args:
            tokenized_map: ``_build_eval_tokenized_inputs`` 가 만든 map token.
            tokenized_agent: ``_build_eval_tokenized_inputs`` 가 만든 agent token.
            repeat_count: rollout 복제 수 (``self_forced_n_rollouts`` 값).

        Returns:
            tuple: ``(replicated_tokenized_map, replicated_tokenized_agent)``.
            ``repeat_count == 1`` 이면 입력을 그대로 돌려줍니다.

        설명:
            scenario 를 N 장 복제해서 한 번의 closed-loop rollout 으로 N 개의 noise tape
            샘플을 동시에 굴리기 위함입니다.  agent-dim / map-dim tensor 는 첫 dim 으로
            repeat 하고, ``batch`` 는 ``+ i*num_graphs`` 로 offset 해서 PyG 가 N 배 큰
            장면 묶음으로 인식하도록 만듭니다.  vocab buffer 와 packed eval-only tensor
            는 변경하지 않습니다 (self-forced critic/DMD pass 에서 사용되지 않음).
        """
        if repeat_count <= 1:
            return tokenized_map, tokenized_agent
        if "batch" not in tokenized_agent or "num_graphs" not in tokenized_agent:
            raise KeyError(
                "tokenized_agent must contain 'batch' and 'num_graphs' for self-forced "
                "rollout replication."
            )
        if "batch" not in tokenized_map:
            raise KeyError(
                "tokenized_map must contain 'batch' for self-forced rollout replication."
            )
        num_graphs = int(tokenized_agent["num_graphs"])
        n_agent = int(tokenized_agent["batch"].shape[0])
        n_pl = int(tokenized_map["batch"].shape[0])
        replicated_agent = self._replicate_token_dict_along_first_dim(
            token_dict=tokenized_agent,
            first_dim_size=n_agent,
            num_graphs=num_graphs,
            repeat_count=repeat_count,
        )
        replicated_map = self._replicate_token_dict_along_first_dim(
            token_dict=tokenized_map,
            first_dim_size=n_pl,
            num_graphs=num_graphs,
            repeat_count=repeat_count,
        )
        return replicated_map, replicated_agent

    @staticmethod
    def _build_anchor_fine_exec_history(
        pos10: Tensor,
        head10: Tensor,
        valid10: Tensor,
        current_raw_step: int,
        shift: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """anchor 의 current_raw_step(10Hz) 기준 최근 ``shift+1`` 개 fine exec-history.

        base ``TokenProcessor._build_rollout_init_fine_history`` 를 임의 current_raw_step
        으로 일반화 (OCSC GT-grounded per-anchor 용).  raw step
        ``[current_raw_step-shift : current_raw_step+1]`` 을 쓰고 길이가 모자라면 앞을 반복.
        VR4: 마지막 step 이 anchor current.
        """
        n_step = int(pos10.shape[1])
        current_raw_step = min(int(current_raw_step), n_step - 1)
        start = max(current_raw_step - int(shift), 0)
        pos_h = pos10[:, start : current_raw_step + 1].contiguous()
        head_h = head10[:, start : current_raw_step + 1].contiguous()
        valid_h = valid10[:, start : current_raw_step + 1].contiguous()
        history_len = int(shift) + 1
        if pos_h.shape[1] < history_len:
            pad = history_len - pos_h.shape[1]
            pos_h = torch.cat([pos_h[:, :1].expand(-1, pad, -1), pos_h], dim=1)
            head_h = torch.cat([head_h[:, :1].expand(-1, pad), head_h], dim=1)
            valid_h = torch.cat([valid_h[:, :1].expand(-1, pad), valid_h], dim=1)
        else:
            pos_h = pos_h[:, -history_len:].contiguous()
            head_h = head_h[:, -history_len:].contiguous()
            valid_h = valid_h[:, -history_len:].contiguous()
        return pos_h, head_h, valid_h

    @staticmethod
    def _anchor_rollout_tokens_static(
        tokenized_agent: Dict[str, Tensor],
        anchor_idx: int,
        shift: int,
    ) -> Dict[str, Tensor]:
        """anchor ``anchor_idx`` 의 GT-grounded rollout 입력 토큰을 만든다 (🅐).

        coarse 키(gt_pos/gt_heading/valid_mask/gt_idx)를 ``[:, anchor_idx:]`` 로 슬라이스
        (VR3: cache 의 ``[:, :step_current_2hz]`` 가 anchor k history window 가 됨) 하고,
        exec fine-history 를 anchor current_raw_step=``shift*(anchor_idx+2)`` 기준으로
        rollout_full_*_10hz 에서 재생성 (VR4).  anchor_idx==0 이면 입력 그대로 (VR6).
        score 평가용 ctx_* 키는 슬라이스하지 않는다 (anchor_idx 로 인덱싱).
        """
        k = int(anchor_idx)
        if k == 0:
            return tokenized_agent
        out: Dict[str, Tensor] = dict(tokenized_agent)
        for key in ("gt_pos", "gt_heading", "valid_mask", "gt_idx"):
            v = tokenized_agent.get(key)
            if torch.is_tensor(v) and v.dim() >= 2 and int(v.shape[1]) > k:
                out[key] = v[:, k:].contiguous()
        fine_keys = (
            "rollout_full_pos_10hz",
            "rollout_full_head_10hz",
            "rollout_full_valid_10hz",
        )
        if all(fk in tokenized_agent for fk in fine_keys):
            cur = int(shift) * (k + 2)
            ph, hh, vh = SMARTFlow._build_anchor_fine_exec_history(
                tokenized_agent["rollout_full_pos_10hz"],
                tokenized_agent["rollout_full_head_10hz"],
                tokenized_agent["rollout_full_valid_10hz"],
                current_raw_step=cur,
                shift=int(shift),
            )
            out["rollout_init_fine_pos_history"] = ph
            out["rollout_init_fine_head_history"] = hh
            out["rollout_init_fine_valid_history"] = vh
            out["rollout_init_fine_pos_pair"] = ph[:, -2:].contiguous()
            out["rollout_init_fine_head_pair"] = hh[:, -2:].contiguous()
            out["rollout_init_fine_valid_pair"] = vh[:, -2:].contiguous()
        for fk in fine_keys:
            out.pop(fk, None)
        return out

    def _build_self_forced_anchor_rollout_tokens(
        self,
        tokenized_agent: Dict[str, Tensor],
        anchor_idx: int,
    ) -> Dict[str, Tensor]:
        """``_anchor_rollout_tokens_static`` 의 instance wrapper (shift 주입)."""
        return self._anchor_rollout_tokens_static(
            tokenized_agent=tokenized_agent,
            anchor_idx=int(anchor_idx),
            shift=int(self.encoder.agent_encoder.shift),
        )

    def _get_self_forced_rollout_steps_2hz(self) -> int:
        """0.5 초 commit block 수.

        OCSC GT-grounded per-anchor(🅐) 에서는 anchor 마다 별도 rollout 을 window 길이만큼만
        돌리므로 항상 ``flow_window_steps / 5`` (= 1 window).  (구 🅑 의 단일 rollout 다중
        window 추출용 ``(n_anchors-1)*stride`` 확장은 제거.)
        """
        if self.flow_window_steps % 5 != 0:
            raise ValueError(
                "self-forced NPFM assumes flow_window_steps is divisible by 5, "
                f"got {self.flow_window_steps}."
            )
        window_2hz = int(self.flow_window_steps // 5)
        return max(1, window_2hz)

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
        anchor_idx: int = 0,
    ) -> Dict[str, Tensor]:
        """주어진 decoder 가 noisy path 를 어떻게 clean 으로 보는지 계산합니다.

        Args:
            decoder: ``F_rho`` (teacher) 또는 ``F_psi`` (critic) 역할의 decoder.
            tokenized_map: 평가 모드 map token 사전.
            tokenized_agent: 평가 모드 agent token 사전.
            noisy_path_norm: noisy path ``[n_valid_agent, F_win, flow_state_dim]``.
            tau: flow interpolation time ``[n_valid_agent]``.
            anchor_mask: 사용할 anchor 의 agent mask ``[n_agent]``.
            anchor_idx: 사용할 anchor index (>= 0, default 0).

        Returns:
            ``{"velocity": ..., "clean": ...}``.
        """
        map_feature = decoder.encode_map(tokenized_map)
        return decoder.path_flow_velocity_for_anchor_k(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            path_noisy_norm=noisy_path_norm,
            tau=tau,
            anchor_mask=anchor_mask,
            anchor_idx=int(anchor_idx),
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
        anchor_idx: int = 0,
        anchor_grounded: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """committed rollout 을 ``anchor_idx`` 번째 anchor 기준 packed flow state 로 변환합니다.

        ``anchor_grounded=True`` (OCSC GT-grounded per-anchor, 🅐): ``rollout`` 이 이미 anchor
        ``anchor_idx`` 의 GT current 에서 출발한 별도 rollout 이라고 보고 window 0 을 추출,
        frame 은 anchor ``anchor_idx`` 의 GT current(ctx_sampled_pos[:, 1+k]) 로 정규화한다.
        ``tokenized_agent`` 은 원본(슬라이스 전) eval 토큰이어야 ctx_sampled / flow_eval_mask
        가 anchor k 를 올바르게 가리킨다.

        Returns:
            packed_path_norm: downstream default — use_kinematic_control_flow=True 면
                control-space 3-dim, 아니면 pose-space 4-dim.  DMD/SiD critic / estimator 가 사용.
            packed_path_pose_norm: 항상 pose-space 4-dim ``[x/20, y/20, cos, sin]``.
                metric / diagnostics 에서 pose-space 값이 필요할 때 사용합니다.
            anchor_mask: anchor 별 agent mask.
        """
        from src.smart.modules.self_forced_path_flow import (
            build_anchor_k_normalized_committed_path,
            get_anchor_k_valid_mask,
        )
        anchor_mask = get_anchor_k_valid_mask(tokenized_agent, anchor_idx=int(anchor_idx))
        committed_path_norm_full = build_anchor_k_normalized_committed_path(
            pred_traj_10hz=rollout["pred_traj_10hz"],
            pred_head_10hz=rollout["pred_head_10hz"],
            tokenized_agent=tokenized_agent,
            flow_window_steps=self.flow_window_steps,
            anchor_idx=int(anchor_idx),
            anchor_stride_2hz=int(self.self_forced_anchor_stride),
            shift=int(self.encoder.agent_encoder.shift),
            rollout_is_anchor_grounded=bool(anchor_grounded),
        )
        packed_path_pose_norm = committed_path_norm_full[anchor_mask]
        if self.use_kinematic_control_flow:
            packed_path_norm = build_anchor0_normalized_committed_control(
                committed_path_norm=packed_path_pose_norm,
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
        else:
            packed_path_norm = packed_path_pose_norm
        return packed_path_norm, packed_path_pose_norm, anchor_mask

    def _update_generated_path_flow_estimator(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        committed_path_norm: Tensor,
        anchor_mask: Tensor,
        *,
        has_committed_path_global: bool | None = None,
        anchor_idx: int = 0,
        preserve_generator_gradients: bool = False,
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
            anchor_idx: generated estimator가 볼 self-forced anchor index입니다.
            preserve_generator_gradients: 이미 누적된 Generator gradient를 보존한 채
                estimator update 분리성만 검사할지입니다. multi-anchor non-warmup에서는
                이전 anchor의 Generator gradient가 이미 누적되어 있으므로 True로 둡니다.

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

        generator_gradient_context = (
            temporarily_clear_module_gradients(self.encoder)
            if preserve_generator_gradients
            else nullcontext()
        )
        with generator_gradient_context:
            self.toggle_optimizer(optimizer)
            self.self_forced_target_teacher.eval()
            self.self_forced_generated_estimator.train()
            try:
                with module_gradients_disabled(self.encoder, self.self_forced_target_teacher):
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
                                anchor_idx=int(anchor_idx),
                            )
                            last_loss = flow_matching_loss(pred_dict["velocity"], flow_target)
                        else:
                            last_loss = self._build_trainable_connected_zero_loss(
                                self.self_forced_generated_estimator,
                            )
                        # anchor k 의 valid agent 가 critic forward 동안 사라지거나 (anchor_mask
                        # 가 모두 False) flow_matching_loss 가 empty path 분기로 빠지면
                        # last_loss 가 grad-free leaf 가 된다.  backward 직전에 grad 가 있는지
                        # 확인하고, 없으면 critic param 에 연결된 trainable zero loss 로 fallback
                        # (DDP 동기화를 위해 모든 rank 가 동일한 backward graph 를 가져야 함).
                        if not last_loss.requires_grad:
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

    def _compute_self_forced_distribution_matching_loss(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        committed_path_norm: Tensor,
        anchor_mask: Tensor,
        anchor_idx: int = 0,
        committed_path_pose_norm: Tensor | None = None,  # legacy; unused after OCSC port
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
            committed_path_pose_norm: legacy unused (GT BC 가 별도 OCSC mode 로 옮겨감).

        Returns:
            Tensor: scalar 분포 맞춤 loss입니다. shape은 ``[]`` 입니다.
        """
        # ── OCSC_clean self_forcing_dmd 정합 — Self-Forcing DMD synthetic gradient ──
        # OCSC `_run_flow_dmd_ft_step` 의 generator loss 를 현재 inference primitive
        # (`_predict_path_flow_clean_estimate`, `flow_ode`) 로 그대로 재현한다.
        #   τ ~ U(eps, 1) (full range),  x_t = σ_τ·ε + τ·x_gen.detach()
        #   pred_x0_{r,f} = β·x_t + σ_τ·v_{r,f}        (v = real/fake score velocity)
        #   g = (1/β)·pred_x0_fake − pred_x0_real
        #   g_n = g / normalizer,  normalizer = |pred_x0_real|.mean(path) (OCSC 정합)
        #   target = (x_gen − g_n).detach(),  L_gen = 0.5·MSE(x_gen, target)
        # β=1 vanilla / β<1 diversity↑ / β>1 sharpening.  (구 path_step_size·guidance-τ·
        # |committed−real| normalizer 는 OCSC 와 불일치라 제거.)
        #
        # ★ conditioning 공유 (OCSC cond_d 정합): OCSC 는 generator 인코더의 anchor hidden
        # cond_d 1개를 real/fake score 가 공유하고 flow_decoder 가중치만 다르게 평가한다.
        # 본 포트는 score 평가에서 teacher/estimator 가 각자 인코더로 anchor_hidden 을
        # 재인코딩하지만, flow_dmd recipe 의 unfrozen_range=full_flow_decoder 로 generator/
        # teacher/estimator 의 인코더가 모두 frozen-identical(=pretrained) 로 고정되므로
        # 재인코딩 결과가 byte-identical → 동일 x_t·동일 conditioning 에서 flow_decoder 만
        # 다른 OCSC 평가와 행동적으로 동치다.  (인코더를 학습시키는 scope 로 바꾸면 이
        # 등가성이 깨지므로, 그때는 명시적 cond_d 주입이 필요하다.)
        if str(self.self_forced_distribution_matching_objective) != "dmd":
            raise ValueError(
                "renew 포트 이후에는 OCSC self_forcing_dmd objective 만 지원합니다; "
                f"got distribution_matching_objective={self.self_forced_distribution_matching_objective!r}."
            )
        if self.self_forced_target_teacher is None or self.self_forced_generated_estimator is None:
            raise RuntimeError("self-forced auxiliary models are not initialized.")

        flow_ode = self.encoder.agent_encoder.flow_ode
        beta = float(self._dmd_effective_beta())
        inv_beta = 1.0 / beta
        use_normalize = bool(self.self_forced_dmd_normalize)

        x_gen = committed_path_norm  # generator graph (grad ON)
        self.self_forced_target_teacher.eval()
        self.self_forced_generated_estimator.eval()
        self._clear_self_forced_auxiliary_gradients()
        with torch.no_grad():
            x_gen_d = x_gen.detach()
            tau = torch.rand(
                x_gen_d.shape[0], device=x_gen_d.device, dtype=x_gen_d.dtype
            ) * (1.0 - float(flow_ode.eps)) + float(flow_ode.eps)
            noise = torch.randn_like(x_gen_d)
            view_tau = tau.view(-1, 1, 1)
            view_sigma = flow_ode._sigma_t(tau).view(-1, 1, 1)
            x_t = view_sigma * noise + view_tau * x_gen_d

            real_pred = self._predict_path_flow_clean_estimate(
                decoder=self.self_forced_target_teacher,
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
                noisy_path_norm=x_t,
                tau=tau,
                anchor_mask=anchor_mask,
                anchor_idx=int(anchor_idx),
            )
            fake_pred = self._predict_path_flow_clean_estimate(
                decoder=self.self_forced_generated_estimator,
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
                noisy_path_norm=x_t,
                tau=tau,
                anchor_mask=anchor_mask,
                anchor_idx=int(anchor_idx),
            )
            beta_path = float(flow_ode._beta())
            pred_x0_real = beta_path * x_t + view_sigma * real_pred["velocity"].to(dtype=x_t.dtype)
            pred_x0_fake = beta_path * x_t + view_sigma * fake_pred["velocity"].to(dtype=x_t.dtype)
            g_dmd = inv_beta * pred_x0_fake - pred_x0_real
            if use_normalize:
                normalizer = pred_x0_real.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
                g_n = g_dmd / normalizer
            else:
                g_n = g_dmd
            g_n = torch.nan_to_num(g_n, nan=0.0, posinf=0.0, neginf=0.0)

        target = (x_gen - g_n.to(dtype=x_gen.dtype)).detach()
        self._set_self_forced_backward_context(
            committed_path_norm=x_gen,
            dmd_direction=g_n,
            target_path_norm=target,
        )
        return 0.5 * F.mse_loss(x_gen, target, reduction="mean")

    def _dmd_effective_beta(self) -> float:
        """현재 step 의 effective DMD β (OCSC β annealing 정합).

        warmup 동안 β=1.0 유지 → anneal 동안 1.0 → ``dmd_beta`` linear ramp →
        이후 ``dmd_beta`` 고정.  warmup/anneal step 이 0 이면 상수 β(``dmd_beta``).
        """
        beta_final = float(self.self_forced_dmd_beta)
        cfg = self.self_forced_config
        beta_warmup = int(getattr(cfg, "dmd_beta_warmup_steps", 0) or 0) if cfg is not None else 0
        beta_anneal = int(getattr(cfg, "dmd_beta_anneal_steps", 0) or 0) if cfg is not None else 0
        if beta_final == 1.0 or beta_anneal <= 0:
            beta = beta_final
        else:
            stepped = 0
            _trainer = getattr(self, "trainer", None)
            _fit_loop = getattr(_trainer, "fit_loop", None) if _trainer is not None else None
            _epoch_loop = getattr(_fit_loop, "epoch_loop", None) if _fit_loop is not None else None
            if _epoch_loop is not None:
                stepped = int(getattr(_epoch_loop, "_batches_that_stepped", 0) or 0)
            if stepped < beta_warmup:
                beta = 1.0
            elif stepped < beta_warmup + beta_anneal:
                progress = (stepped - beta_warmup) / float(beta_anneal)
                beta = 1.0 + (beta_final - 1.0) * progress
            else:
                beta = beta_final
        if not (beta > 0.0):
            raise ValueError(f"dmd_beta_effective must be > 0, got {beta} (target={beta_final}).")
        return beta

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
        self._sync_ocsc_ref_decoder_from_main_if_needed()
        self._sync_self_forced_auxiliary_models()
        self._prepare_self_forced_generator_ema()

    def _sync_ocsc_ref_decoder_from_main_if_needed(self) -> None:
        """Fresh OCSC finetune에서 frozen ref decoder를 ckpt-loaded main decoder와 맞춥니다."""
        if self.ref_flow_decoder is None or not self._is_ocsc_ft_enabled():
            return
        if self._ref_flow_decoder_loaded_from_ckpt:
            return
        self.ref_flow_decoder.load_state_dict(
            self.encoder.agent_encoder.flow_decoder.state_dict(),
            strict=True,
        )
        for p in self.ref_flow_decoder.parameters():
            p.requires_grad_(False)
        self.ref_flow_decoder.eval()
        self._ref_flow_decoder_loaded_from_ckpt = True

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
        if self.self_forced_use_anchor_fm_loss:
            tokenized_map_train, tokenized_agent_train = self.token_processor(data)
            pred = self.encoder(
                tokenized_map_train,
                tokenized_agent_train,
                anchor_mask_key="flow_train_mask",
            )
            fm_loss, open_metric_dict, _, has_anchor_fm_targets = self._open_loop_denoise_metrics(
                pred,
                zero_loss_module=self.encoder,
            )

        tokenized_map_eval, tokenized_agent_eval = self._build_eval_tokenized_inputs(data)
        # ── Batched n_rollouts: scenario 를 N 장 복제해서 한 번의 closed-loop rollout 으로
        #    N 개의 noise tape 샘플을 동시에 굴린다.  agent dim 이 N 배가 되어 packed
        #    committed_path_norm 도 자동으로 N 배 — critic / DMD direction 호출은 무수정.
        #    sequential N forward 와 비교해 walltime ≈ 1×, variance 1/N.
        n_rollouts = max(1, int(self.self_forced_n_rollouts))
        if n_rollouts > 1:
            tokenized_map_eval, tokenized_agent_eval = self._build_self_forced_replicated_tokens(
                tokenized_map=tokenized_map_eval,
                tokenized_agent=tokenized_agent_eval,
                repeat_count=n_rollouts,
            )
        # Warmup 상태를 batch 진입 시점에 한 번만 평가해 캐시. estimator optimizer step 사이에
        # self.global_step 이 증가하면서 한 batch 안의 첫 호출과 마지막 호출 결과가 갈리는
        # race 를 막아야 한다. race 가 발생하면 첫 호출에서 rollout 을 torch.no_grad() 로
        # 돌고 두 번째 호출에선 warmup off 로 판정되어 generator backward 로 진입,
        # committed_path_norm 에 graph 가 없는 상태에서 backward 가 시도되어
        # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        # 가 난다.
        in_estimator_warmup = self._is_self_forced_estimator_warmup_active()
        has_anchor_fm_targets_global = False
        if fm_loss is not None:
            has_anchor_fm_targets_global = self._sync_distributed_bool_any(
                has_anchor_fm_targets,
                device=fm_loss.device,
            )

        # ── Single batched rollout + anchor loop 만 남김.
        #    rollout 축은 이미 agent dim 에 packed 되어 있어 sf_loss/critic 평균에 자동 포함.
        #    denom 은 n_anchors 로만 나누고 (n_rollouts 는 batch 안에서 평균됨).
        rollout_results = []  # (rollout dict, committed_path_norm, anchor_mask, has_committed_local)
        per_rollout_gen_est_losses = []
        per_rollout_sf_losses = []
        per_rollout_total_losses = []
        any_committed_global = False
        last_anchor_loss_val: Tensor | None = None

        # Do not Lightning-toggle the generator optimizer here.  This step also
        # updates the generated estimator, and Lightning optimizer toggles are
        # not stack-safe: nesting the estimator toggle inside a generator toggle
        # can leave estimator params disabled after warmup.  Isolation is enforced
        # by detached estimator inputs plus the gradient assertions below.
        generator_optimizer = self.optimizers()[0] if not in_estimator_warmup else None
        if generator_optimizer is not None:
            generator_optimizer.zero_grad(set_to_none=True)
            self._prepare_self_forced_generator_backward_boundary()

        n_anchors = max(1, int(self.self_forced_n_anchors))
        denom = float(n_anchors)
        try:
            for anchor_idx in range(n_anchors):
                # anchor_stride: anchor 간격(coarse 2Hz step).  j 번째 anchor 의 token anchor
                # index = j*stride → GT coarse step (j*stride)+1.  stride=4 면 2초 간격
                # (10Hz shift*(j*stride+2)).  stride=1 이면 연속(0.5초).
                token_anchor_idx = int(anchor_idx) * int(self.self_forced_anchor_stride)
                # OCSC GT-grounded per-anchor(🅐): anchor 마다 GT current 에서 출발하는 별도
                # rollout.  anchor 입력 토큰 = coarse 키 [:, k:] 슬라이스 + fine-history 를
                # current_raw_step=shift*(k+2) 로 재생성 (token_anchor_idx=0 은 원본 그대로).
                anchor_rollout_tokens = self._build_self_forced_anchor_rollout_tokens(
                    tokenized_agent_eval, token_anchor_idx
                )
                if in_estimator_warmup:
                    with torch.no_grad():
                        rollout = self._run_self_forced_rollout(tokenized_map_eval, anchor_rollout_tokens)
                else:
                    rollout = self._run_self_forced_rollout(tokenized_map_eval, anchor_rollout_tokens)
                # pack 은 원본 tokenized_agent_eval 사용 (ctx_sampled frame + flow_eval_mask 가
                # token_anchor_idx 의 GT 기준).  anchor_grounded=True → rollout window 0 추출.
                committed_path_norm, committed_path_pose_norm, anchor_mask = (
                    self._pack_self_forced_committed_rollout(
                        rollout=rollout,
                        tokenized_agent=tokenized_agent_eval,
                        anchor_idx=token_anchor_idx,
                        anchor_grounded=True,
                    )
                )
                has_committed_local = committed_path_norm.numel() > 0
                has_committed_global = self._sync_distributed_bool_any(
                    has_committed_local,
                    device=committed_path_norm.device,
                )
                any_committed_global = any_committed_global or has_committed_global
                rollout_results.append((rollout, committed_path_norm, anchor_mask, has_committed_local))

                # critic update — 매 anchor 마다.  agent dim 이 N×B 이므로 critic 도 동일 데이터 1 step 으로 봄.
                gen_estimator_loss = self._update_generated_path_flow_estimator(
                    tokenized_map=tokenized_map_eval,
                    tokenized_agent=tokenized_agent_eval,
                    committed_path_norm=committed_path_norm,
                    anchor_mask=anchor_mask,
                    has_committed_path_global=has_committed_global,
                    anchor_idx=token_anchor_idx,
                    preserve_generator_gradients=(not in_estimator_warmup and anchor_idx > 0),
                )
                per_rollout_gen_est_losses.append(gen_estimator_loss)

                if in_estimator_warmup or not has_committed_global:
                    # warmup 중이거나 valid anchor 없는 batch — generator backward skip.
                    continue

                # DMD/SiD direction backward — grad accumulate / n_anchors.
                if has_committed_local:
                    sf_loss_i = self._compute_self_forced_distribution_matching_loss(
                        tokenized_map=tokenized_map_eval,
                        tokenized_agent=tokenized_agent_eval,
                        committed_path_norm=committed_path_norm,
                        committed_path_pose_norm=committed_path_pose_norm,
                        anchor_mask=anchor_mask,
                        anchor_idx=token_anchor_idx,
                    )
                else:
                    sf_loss_i = self._build_trainable_connected_zero_loss(self.encoder)
                anchor_loss_i = (
                    fm_loss
                    if fm_loss is not None
                    else committed_path_norm.new_zeros(())
                )
                last_anchor_loss_val = anchor_loss_i
                total_loss_i = (
                    self.self_forced_weight * sf_loss_i
                    + self.self_forced_anchor_weight * anchor_loss_i
                ) / denom
                if not torch.isfinite(total_loss_i):
                    context = self._format_self_forced_backward_context()
                    self._clear_self_forced_backward_context()
                    raise RuntimeError(
                        "Non-finite self-forced total_loss detected: "
                        f"{self._summarize_nonfinite_tensor(total_loss_i)}"
                        f"{context}"
                    )
                # anchor k 의 forward 가 empty path 분기로 빠지면 sf_loss_i 가 grad-free
                # leaf 가 되어 backward 시 "no grad_fn" RuntimeError 가 난다.
                # DDP 동기화 위해 항상 generator param 에 연결된 trainable zero loss 로 fallback.
                if not total_loss_i.requires_grad:
                    total_loss_i = self._build_trainable_connected_zero_loss(self.encoder)
                # multi-anchor 학습 시 같은 rollout 결과를 anchor 들이 share 하므로
                # 마지막 anchor 전까지 graph 를 유지해야 두 번째 anchor backward 가
                # 끊긴 graph 로 들어가는 race 를 막을 수 있다.
                is_last_anchor_in_rollout = anchor_idx == n_anchors - 1
                self._manual_backward_without_autocast(
                    total_loss_i,
                    retain_graph=not is_last_anchor_in_rollout,
                )
                self._assert_self_forced_generator_update_isolated()
                per_rollout_sf_losses.append(sf_loss_i.detach())
                per_rollout_total_losses.append(total_loss_i.detach() * denom)
                self._clear_self_forced_backward_context()

            # ── warmup-only batch: gen optimizer step 생략하고 warmup 종료 처리.
            if in_estimator_warmup:
                avg_gen_est = (
                    torch.stack(
                        [v.detach().float() for v in per_rollout_gen_est_losses if v is not None]
                    ).mean()
                    if any(v is not None for v in per_rollout_gen_est_losses)
                    else None
                )
                return self._finish_self_forced_estimator_warmup_step(avg_gen_est)

            # ── valid anchor 가 한 번도 없었던 batch — 기존 zero-loss / fm-only 경로.
            if not any_committed_global:
                self._clear_self_forced_generator_gradients()
                if fm_loss is None or not has_anchor_fm_targets_global:
                    zero_loss = (
                        fm_loss
                        if fm_loss is not None
                        else self._build_trainable_connected_zero_loss(self.encoder)
                    )
                    self._prepare_self_forced_generator_backward_boundary()
                    self._manual_backward_without_autocast(zero_loss)
                    self._assert_self_forced_generator_update_isolated()
                    self._clear_self_forced_generator_gradients()
                    self.log("train/loss", zero_loss.detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
                    if fm_loss is not None:
                        self.log("train/loss_fm", fm_loss.detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
                    self.log("train/sf_anchor_fm_enabled", 0.0, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
                    self.log("train/sf_anchor_weight", float(self.self_forced_anchor_weight), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
                    return zero_loss.detach()
                self._prepare_self_forced_generator_backward_boundary()
                self._manual_backward_without_autocast(fm_loss)
                self._assert_self_forced_generator_update_isolated()
                if has_anchor_fm_targets_global:
                    self._clip_and_step_with_optional_scaler(generator_optimizer)
                    self._update_self_forced_generator_ema_after_step()
                self._clear_self_forced_generator_gradients()
                self.log("train/loss", fm_loss.detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
                self.log("train/loss_fm", fm_loss.detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
                return fm_loss.detach()

            # ── 정상 multi-rollout step: 누적된 grad 로 generator optimizer.step().
            self._clip_and_step_with_optional_scaler(
                generator_optimizer,
                gradient_clip_val=self.self_forced_gradient_clip_val,
                gradient_clip_algorithm="norm",
            )
            self._update_self_forced_generator_ema_after_step()
            self._clear_self_forced_generator_gradients()
        finally:
            self._clear_self_forced_backward_context()

        # ── Logging 변수: rollout 평균.
        rollout = rollout_results[-1][0]
        committed_path_norm = rollout_results[-1][1]
        sf_loss = (
            torch.stack(per_rollout_sf_losses).mean()
            if per_rollout_sf_losses
            else self._build_trainable_connected_zero_loss(self.encoder).detach()
        )
        gen_estimator_loss = (
            torch.stack(
                [v.detach().float() for v in per_rollout_gen_est_losses if v is not None]
            ).mean()
            if any(v is not None for v in per_rollout_gen_est_losses)
            else torch.zeros((), device=self.device, dtype=torch.float32)
        )
        anchor_loss = (
            last_anchor_loss_val
            if last_anchor_loss_val is not None
            else committed_path_norm.new_zeros(())
        )
        total_loss = (
            torch.stack(per_rollout_total_losses).mean()
            if per_rollout_total_losses
            else (self.self_forced_weight * sf_loss + self.self_forced_anchor_weight * anchor_loss).detach()
        )

        # Sweep 시 step 별 loss 추이를 봐야 하므로 on_step=True 로 통일.  (이전 on_step=False
        # 였던 키는 epoch end 까지 wandb 에 안 찍혀서 매 200 step val 직전엔 generator/critic
        # loss 가 보이지 않는 문제가 있었음.)
        self.log("train/loss", total_loss.detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        if fm_loss is not None:
            self.log("train/loss_fm", fm_loss.detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_npfm_loss", sf_loss.detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_generated_estimator_loss", gen_estimator_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        if self.self_forced_distribution_matching_objective == "dmd":
            self.log("train/sf_dmd_beta", float(self.self_forced_dmd_beta), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_anchor_fm_enabled", float(self.self_forced_use_anchor_fm_loss), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_anchor_loss", anchor_loss.detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/sf_anchor_weight", float(self.self_forced_anchor_weight), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
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
        return total_loss.detach()

    def _is_ocsc_ft_enabled(self) -> bool:
        """OCSC (Open-Closed Self-Consistency) fine-tune mode 활성 여부.

        finetune.enabled=True 이고 finetune.mode == 'ocsc_ft' 일 때 True.
        self_forced 와 격리되어 training_step 이 별도 분기.
        """
        ft = getattr(self, "finetune_config", None)
        if ft is None:
            return False
        if not bool(getattr(ft, "enabled", False)):
            return False
        return str(getattr(ft, "mode", "none")).lower() == "ocsc_ft"

    # ──────────────────────────────────────────────────────────────────────
    # OCSC (Open-Closed Self-Consistency) finetune mode — single anchor (0)
    #   알고리즘 (OCSC_clean 정합 단순화 버전, single anchor 0):
    #     1. eval-mode tokenize + encode_map (no_grad).
    #     2. prepare_inference_cache → active_mask, active_hidden, current_pos/head.
    #     3. M open-loop samples (no_grad, ref_flow_decoder swap):
    #          ol_norms[m] = _sample_open_loop_future_from_hidden(active_hidden, seed=m)
    #          shape [n_active, 20, 4] = [x/20, y/20, cos, sin] in anchor-0 local frame.
    #     4. G closed-loop rollouts (with grad), per g:
    #          rollout_g = encoder.agent_encoder.training_rollout_from_cache(seed_g)
    #          → pred_traj_10hz / pred_head_10hz, world frame
    #          → transform to anchor-0 local + normalize ⇒ cl_norm_g shape [n_active, 20, 4]
    #     5. nearest match (ocsc_ol_nearest_match=True) — per CL g, per agent:
    #          m* = argmin_m mean_L2(cl_norm_g[a], ol_norm[m][a])
    #        gt_target=True 면 OL pool 대신 GT 1 sample (M=1, no nearest).
    #     6. paired L2:  pos_w * L2(pos) + head_w * L2(cos,sin), mean over agents/time.
    #     7. backward (Lightning automatic_optimization).
    # ──────────────────────────────────────────────────────────────────────

    def _ocsc_anchor0_origin(
        self,
        rollout_cache: Dict[str, object],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """anchor 0 = history end 의 active_mask / current_pos / current_head 를 뽑습니다."""
        active_mask = rollout_cache["valid_window"][:, -1]
        current_pos = rollout_cache["pos_window"][:, -1]
        current_head = rollout_cache["head_window"][:, -1]
        return active_mask, current_pos, current_head

    def _ocsc_world_traj_to_anchor0_pose_norm(
        self,
        pred_pos_global: Tensor,    # [n_active, T, 2]
        pred_head_global: Tensor,   # [n_active, T]
        current_pos: Tensor,        # [n_active, 2]
        current_head: Tensor,       # [n_active]
        pos_scale_m: float = 20.0,
    ) -> Tensor:
        """closed-loop rollout 의 global pose 를 anchor-0 local pose-norm 4-dim 으로 변환.

        Returns:
            Tensor ``[n_active, T, 4]`` = ``[x/20, y/20, cos, sin]``.
        """
        from src.smart.utils.rollout import transform_to_local
        pos_local, head_local = transform_to_local(
            pos_global=pred_pos_global,
            head_global=pred_head_global,
            pos_now=current_pos,
            head_now=current_head,
        )
        return torch.stack(
            [
                pos_local[..., 0] / float(pos_scale_m),
                pos_local[..., 1] / float(pos_scale_m),
                head_local.cos(),
                head_local.sin(),
            ],
            dim=-1,
        )

    def _ocsc_build_gt_target_norm(
        self,
        data,
        active_mask: Tensor,
        current_pos: Tensor,
        current_head: Tensor,
        window_steps_10hz: int,
    ) -> tuple[Tensor, Tensor] | None:
        """GT (raw 10Hz) future pose 를 anchor-0 local pose-norm 4-dim 으로.

        anchor 0 = history end (10Hz step index = num_historical_steps - 1 = 10).
        future 는 step 11 .. 11+window_steps_10hz-1.
        invalid future 는 loss 에서 mask 처리합니다.
        """
        try:
            agent_data = data["agent"]
            pos_full = agent_data["position"][..., :2]  # [N, T_seq, 2]
            head_full = agent_data["heading"]            # [N, T_seq]
            valid_full = agent_data["valid_mask"]        # [N, T_seq]
        except Exception:
            return None
        num_hist_10hz = int(self.encoder.agent_encoder.num_historical_steps)
        gt_start = num_hist_10hz
        gt_end = gt_start + int(window_steps_10hz)
        if pos_full.shape[1] < gt_end:
            return None
        gt_pos = pos_full[active_mask, gt_start:gt_end]
        gt_head = head_full[active_mask, gt_start:gt_end]
        gt_valid = valid_full[active_mask, gt_start:gt_end]
        if not bool(gt_valid.any()):
            return None
        gt_norm = self._ocsc_world_traj_to_anchor0_pose_norm(
            pred_pos_global=gt_pos,
            pred_head_global=gt_head,
            current_pos=current_pos,
            current_head=current_head,
        )
        return gt_norm, gt_valid

    def _ocsc_paired_pose_loss(
        self,
        cl_norm: Tensor,   # [n_active, T, C]  C ∈ {3, 4}
        target_norm: Tensor,  # [n_active, T, C]
        pos_w: float,
        head_w: float,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        """anchor-0 local 정규화 텐서의 paired L2.

        C=3 (control-space): pos=Δs/Δn (정규화), head=Δyaw (정규화) 1 ch.
        C=4 (pose-space):    pos=Δx/Δy (/20),    head=Δcos/Δsin 2 ch.
        """
        diff = cl_norm.float() - target_norm.float().detach()
        if diff.shape[-1] != target_norm.shape[-1]:
            raise RuntimeError(
                f"OCSC paired loss dim mismatch: cl={tuple(cl_norm.shape)} "
                f"target={tuple(target_norm.shape)}."
            )
        if valid_mask is not None:
            valid = valid_mask[..., : diff.shape[-2]].to(device=diff.device, dtype=diff.dtype)
            if not bool(valid.any()):
                return cl_norm.sum() * 0.0
            mask = valid.unsqueeze(-1)
            denom = mask.sum().clamp(min=1.0)
            pos_loss = (diff[..., :2].square() * mask).sum() / denom
        else:
            pos_loss = diff[..., :2].square().mean()
        head_dim_end = min(4, diff.shape[-1])
        if head_dim_end > 2:
            if valid_mask is not None:
                head_loss = (diff[..., 2:head_dim_end].square() * mask).sum() / denom
            else:
                head_loss = diff[..., 2:head_dim_end].square().mean()
            return pos_w * pos_loss + head_w * head_loss
        return pos_w * pos_loss

    def _ocsc_sample_open_loop_with_ref(
        self,
        active_hidden: Tensor,
        active_agent_type: Tensor,
        active_agent_length: Tensor | None,
        m: int,
        sampling_scheme,
        seed_base: int,
        output_space: str = "pose",
    ) -> Tensor:
        """ref_flow_decoder 로 1 개 OL sample 생성 후 pose 4-dim 으로 변환 (no_grad).

        student agent_encoder 의 flow_decoder 를 잠시 ref 로 swap 후 호출.
        use_kinematic_control_flow=True 면 native output 이 control 3-dim 이므로
        ``control_norm_to_pose_norm`` 으로 anchor 0 origin forward kinematic 누적해
        pose 4-dim ``[x/20, y/20, cos, sin]`` 로 변환.
        """
        from src.smart.modules.kinematic_control import control_norm_to_pose_norm
        agent_enc = self.encoder.agent_encoder
        orig_fd = agent_enc.flow_decoder
        if self.ref_flow_decoder is not None:
            self.ref_flow_decoder.eval()
            agent_enc.flow_decoder = self.ref_flow_decoder
        try:
            ol_raw = agent_enc._sample_open_loop_future_from_hidden(
                anchor_hidden_valid=active_hidden,
                sampling_scheme=sampling_scheme,
                sampling_seed=int(seed_base) + int(m),
            )
        finally:
            agent_enc.flow_decoder = orig_fd
        output_space = str(output_space).lower()
        if output_space == "control":
            if not self.use_kinematic_control_flow:
                raise ValueError("OCSC control-space matching requires use_kinematic_control_flow=True.")
            return ol_raw
        # native dim: control 3-dim (use_kinematic_control_flow=True) or pose 4-dim (False).
        if self.use_kinematic_control_flow:
            ol_pose_norm = control_norm_to_pose_norm(
                control_norm=ol_raw,
                agent_type=active_agent_type,
                agent_length=active_agent_length,
                pos_scale_m=agent_enc.control_pos_scale_m,
                vehicle_yaw_scale_rad=agent_enc.control_vehicle_yaw_scale_rad,
                pedestrian_yaw_scale_rad=agent_enc.control_pedestrian_yaw_scale_rad,
                cyclist_yaw_scale_rad=agent_enc.control_cyclist_yaw_scale_rad,
                use_holonomic_model_only=agent_enc.use_holonomic_model_only,
                vehicle_no_slip_point_ratio=agent_enc.control_vehicle_no_slip_point_ratio,
                cyclist_no_slip_point_ratio=agent_enc.control_cyclist_no_slip_point_ratio,
            )  # [n_active, 20, 4]
            return ol_pose_norm
        return ol_raw  # already pose 4-dim

    def _ocsc_sample_open_loop_with_ref_batched(
        self,
        active_hidden: Tensor,
        active_agent_type: Tensor,
        active_agent_length: Tensor | None,
        tokenized_agent: Dict[str, Tensor],
        active_mask: Tensor,
        scenario_ids: Sequence[str],
        sample_count: int,
        sampling_scheme,
        output_space: str = "pose",
    ) -> Tensor:
        """ref_flow_decoder OL samples를 sample 축으로 묶어 한 번에 생성합니다.

        OCSC-clean은 OL target과 CL rollout의 첫 2초 noise tape을 같은
        scenario seed에서 뽑습니다.  그래야 M>=G nearest-match에서도 첫 G개
        OL 후보가 대응 CL rollout과 같은 stochastic branch를 공유합니다.
        """
        from src.smart.modules.kinematic_control import control_norm_to_pose_norm

        agent_enc = self.encoder.agent_encoder
        sample_count = int(sample_count)
        output_space = str(output_space).lower()
        if sample_count < 1:
            raise ValueError(f"sample_count must be positive, got {sample_count}.")
        if output_space not in {"pose", "control"}:
            raise ValueError(f"Unsupported OCSC output_space={output_space!r}.")
        if output_space == "control" and not self.use_kinematic_control_flow:
            raise ValueError("OCSC control-space matching requires use_kinematic_control_flow=True.")

        n_active = int(active_hidden.shape[0])
        if n_active == 0:
            last_dim = 3 if output_space == "control" else 4
            return active_hidden.new_zeros((sample_count, 0, self.flow_window_steps, last_dim))

        x_init_norm = self._ocsc_build_open_loop_x_init_stack(
            agent_enc=agent_enc,
            tokenized_agent=tokenized_agent,
            active_mask=active_mask,
            scenario_ids=scenario_ids,
            sample_count=sample_count,
            sampling_scheme=sampling_scheme,
            dtype=active_hidden.dtype,
        ).reshape(sample_count * n_active, self.flow_window_steps, agent_enc.flow_state_dim)
        repeated_hidden = (
            active_hidden.unsqueeze(0)
            .expand(sample_count, *active_hidden.shape)
            .reshape(sample_count * n_active, *active_hidden.shape[1:])
            .contiguous()
        )
        flow_sample_steps = int(getattr(
            sampling_scheme,
            "sample_steps",
            agent_enc.flow_ode.solver_steps,
        ))
        flow_sample_method = getattr(
            sampling_scheme,
            "sample_method",
            agent_enc.flow_ode.solver_method,
        )
        backprop_last_k = getattr(sampling_scheme, "backprop_last_k", None)

        orig_fd = agent_enc.flow_decoder
        if self.ref_flow_decoder is not None:
            self.ref_flow_decoder.eval()
            agent_enc.flow_decoder = self.ref_flow_decoder
        try:
            ol_raw = agent_enc.flow_ode.generate(
                x_init=x_init_norm,
                model_fn=lambda x_t, tau: agent_enc.flow_decoder(repeated_hidden, x_t, tau),
                steps=flow_sample_steps,
                method=flow_sample_method,
                backprop_last_k=backprop_last_k,
            )
        finally:
            agent_enc.flow_decoder = orig_fd

        if output_space == "control":
            return ol_raw.reshape(sample_count, n_active, self.flow_window_steps, ol_raw.shape[-1])
        if self.use_kinematic_control_flow:
            repeated_agent_type = (
                active_agent_type.unsqueeze(0)
                .expand(sample_count, n_active)
                .reshape(sample_count * n_active)
                .contiguous()
            )
            repeated_agent_length = None
            if active_agent_length is not None:
                repeated_agent_length = (
                    active_agent_length.unsqueeze(0)
                    .expand(sample_count, n_active)
                    .reshape(sample_count * n_active)
                    .contiguous()
                )
            ol_pose = control_norm_to_pose_norm(
                control_norm=ol_raw,
                agent_type=repeated_agent_type,
                agent_length=repeated_agent_length,
                pos_scale_m=agent_enc.control_pos_scale_m,
                vehicle_yaw_scale_rad=agent_enc.control_vehicle_yaw_scale_rad,
                pedestrian_yaw_scale_rad=agent_enc.control_pedestrian_yaw_scale_rad,
                cyclist_yaw_scale_rad=agent_enc.control_cyclist_yaw_scale_rad,
                use_holonomic_model_only=agent_enc.use_holonomic_model_only,
                vehicle_no_slip_point_ratio=agent_enc.control_vehicle_no_slip_point_ratio,
                cyclist_no_slip_point_ratio=agent_enc.control_cyclist_no_slip_point_ratio,
            )
            return ol_pose.reshape(sample_count, n_active, self.flow_window_steps, 4)
        return ol_raw.reshape(sample_count, n_active, self.flow_window_steps, ol_raw.shape[-1])

    def _ocsc_build_open_loop_x_init_stack(
        self,
        *,
        agent_enc,
        tokenized_agent: Dict[str, Tensor],
        active_mask: Tensor,
        scenario_ids: Sequence[str],
        sample_count: int,
        sampling_scheme,
        dtype: torch.dtype,
    ) -> Tensor:
        """Build OL initial noise from the same scenario seeds used by CL rollout."""
        sample_count = int(sample_count)
        if sample_count < 1:
            raise ValueError(f"sample_count must be positive, got {sample_count}.")
        n_active = int(active_mask.sum().item())
        if n_active == 0:
            return active_mask.new_zeros(
                (sample_count, 0, self.flow_window_steps, agent_enc.flow_state_dim),
                dtype=dtype,
            )

        agent_batch = tokenized_agent["batch"]
        scenario_seed_table = self._build_closed_loop_seed_table(
            scenario_ids=scenario_ids,
            rollout_indices=range(sample_count),
            device=agent_batch.device,
        )
        x_init_chunks = []
        for m in range(sample_count):
            noise_tape = agent_enc._build_rollout_noise_tape(
                num_agent=int(agent_batch.shape[0]),
                tape_steps=int(self.flow_window_steps),
                device=active_mask.device,
                dtype=dtype,
                sampling_scheme=sampling_scheme,
                scenario_sampling_seeds=scenario_seed_table[m],
                agent_batch=agent_batch,
            )
            x_init_chunks.append(noise_tape[active_mask, : self.flow_window_steps].contiguous())
        return torch.stack(x_init_chunks, dim=0).contiguous()

    def _ocsc_run_closed_loop_rollouts_batched(
        self,
        data,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        rollout_cache: Dict[str, object],
        active_mask: Tensor,
        current_pos_active: Tensor,
        current_head_active: Tensor,
        sampling_scheme,
        rollout_count: int,
        window_steps_10hz: int,
        rollout_steps_2hz: int,
        match_space: str = "pose",
    ) -> Tensor:
        """OCSC closed-loop rollouts를 rollout 축으로 묶어 한 번에 실행합니다."""
        agent_enc = self.encoder.agent_encoder
        match_space = str(match_space).lower()
        if match_space not in {"pose", "control"}:
            raise ValueError(f"Unsupported OCSC match_space={match_space!r}.")
        if match_space == "control" and not self.use_kinematic_control_flow:
            raise ValueError("OCSC control-space matching requires use_kinematic_control_flow=True.")
        rollout_count = int(rollout_count)
        if rollout_count < 1:
            raise ValueError(f"rollout_count must be positive, got {rollout_count}.")

        num_agent = int(tokenized_agent["batch"].shape[0])
        num_graphs = len(data["scenario_id"])
        scenario_seed_table = self._build_closed_loop_seed_table(
            scenario_ids=data["scenario_id"],
            rollout_indices=range(rollout_count),
            device=tokenized_agent["batch"].device,
        )
        expanded_tokenized_agent = self._build_parallel_rollout_tokenized_agent(
            tokenized_agent=tokenized_agent,
            repeat_count=rollout_count,
            num_graphs=num_graphs,
        )
        expanded_map_feature = self._build_parallel_rollout_map_feature(
            map_feature=map_feature,
            repeat_count=rollout_count,
            num_graphs=num_graphs,
        )
        expanded_rollout_cache = self._build_parallel_rollout_cache(
            rollout_cache=rollout_cache,
            repeat_count=rollout_count,
        )
        rollout = agent_enc.training_rollout_from_cache(
            rollout_cache=expanded_rollout_cache,
            tokenized_agent=expanded_tokenized_agent,
            map_feature=expanded_map_feature,
            sampling_scheme=sampling_scheme,
            scenario_sampling_seeds=scenario_seed_table.reshape(-1).contiguous(),
            rollout_steps_2hz=rollout_steps_2hz,
            return_committed_control=(match_space == "control"),
        )

        n_active = int(active_mask.sum().item())
        if match_space == "control":
            control = rollout["pred_control_10hz"].reshape(
                rollout_count,
                num_agent,
                *rollout["pred_control_10hz"].shape[1:],
            )[:, active_mask, :window_steps_10hz]
            return control.contiguous()

        traj = rollout["pred_traj_10hz"].reshape(
            rollout_count,
            num_agent,
            *rollout["pred_traj_10hz"].shape[1:],
        )[:, active_mask, :window_steps_10hz]
        head = rollout["pred_head_10hz"].reshape(
            rollout_count,
            num_agent,
            *rollout["pred_head_10hz"].shape[1:],
        )[:, active_mask, :window_steps_10hz]

        traj_flat = traj.contiguous().reshape(rollout_count * n_active, window_steps_10hz, 2)
        head_flat = head.contiguous().reshape(rollout_count * n_active, window_steps_10hz)
        pos_flat = (
            current_pos_active.unsqueeze(0)
            .expand(rollout_count, n_active, 2)
            .reshape(rollout_count * n_active, 2)
            .contiguous()
        )
        head_now_flat = (
            current_head_active.unsqueeze(0)
            .expand(rollout_count, n_active)
            .reshape(rollout_count * n_active)
            .contiguous()
        )
        cl_norm_flat = self._ocsc_world_traj_to_anchor0_pose_norm(
            pred_pos_global=traj_flat,
            pred_head_global=head_flat,
            current_pos=pos_flat,
            current_head=head_now_flat,
        )
        return cl_norm_flat.reshape(rollout_count, n_active, window_steps_10hz, 4)

    def _run_flow_ocsc_ft_step(self, data, batch_idx) -> Tensor:
        """OCSC (Open-Closed Self-Consistency) finetune training step.

        single anchor (anchor 0 = history end) 버전.  manual_optimization 사용 안 함
        (Lightning automatic_optimization=True 가정).
        """
        ft = self.finetune_config
        G = int(getattr(ft, "ocsc_n_rollouts", 4))
        M_raw = int(getattr(ft, "ocsc_n_ol_rollouts", -1))
        M = G if M_raw <= 0 else max(1, M_raw)
        nearest_match = bool(getattr(ft, "ocsc_ol_nearest_match", True))
        use_gt_target = bool(getattr(ft, "ocsc_gt_target", False))
        match_space = str(getattr(ft, "ocsc_match_space", "pose")).lower()
        if match_space not in {"pose", "control"}:
            raise ValueError(f"Unsupported ocsc_match_space={match_space!r}; expected pose or control.")
        if match_space == "control":
            if not self.use_kinematic_control_flow:
                raise ValueError("ocsc_match_space=control requires use_kinematic_control_flow=True.")
            if use_gt_target:
                raise ValueError("ocsc_match_space=control currently supports OL-ref targets only.")
        pos_w = float(getattr(ft, "ocsc_position_weight", 1.0))
        head_w = float(getattr(ft, "ocsc_heading_weight", 0.01))
        loss_window_raw = int(getattr(ft, "ocsc_loss_window_steps", -1))
        loss_window_steps_10hz = (
            int(self.flow_window_steps)
            if loss_window_raw <= 0
            else min(int(loss_window_raw), int(self.flow_window_steps))
        )
        if loss_window_steps_10hz <= 0:
            raise ValueError(
                "ocsc_loss_window_steps must be positive or -1, "
                f"got {loss_window_raw}."
            )
        loss_stride_raw = int(getattr(ft, "ocsc_loss_temporal_stride", -1))

        if use_gt_target and nearest_match:
            nearest_match = False  # GT 1 개 target 이면 nearest 의미 없음

        # 1. eval-mode tokenize
        tokenized_map, tokenized_agent = self._build_eval_tokenized_inputs(data)

        agent_enc = self.encoder.agent_encoder

        # 2. encode_map (no_grad — map encoder frozen 전제)
        with torch.no_grad():
            map_feature = self.encoder.encode_map(tokenized_map)

        # 3. prepare_inference_cache + anchor-0 origin (no_grad — student feature 는 ref/CL 공유)
        with torch.no_grad():
            rollout_cache = agent_enc.prepare_inference_cache(
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
            )
        active_mask, current_pos, current_head = self._ocsc_anchor0_origin(rollout_cache)
        if bool(getattr(ft, "ocsc_strict_active_mask", False)):
            try:
                valid_full = data["agent"]["valid_mask"]
                num_hist_10hz = int(agent_enc.num_historical_steps)
                future_start = num_hist_10hz
                future_end = future_start + int(loss_window_steps_10hz)
                if future_end <= int(valid_full.shape[1]):
                    active_mask = active_mask & valid_full[:, future_start:future_end].all(dim=1)
                else:
                    active_mask = active_mask & torch.zeros_like(active_mask, dtype=torch.bool)
            except Exception:
                active_mask = active_mask & torch.zeros_like(active_mask, dtype=torch.bool)
        if not bool(active_mask.any()):
            return self._build_trainable_connected_zero_loss(self.encoder)
        current_pos_active = current_pos[active_mask]
        current_head_active = current_head[active_mask]
        active_hidden = rollout_cache["feat_a_now"][active_mask]
        active_agent_type = tokenized_agent["type"][active_mask]
        active_agent_length = (
            tokenized_agent["shape"][active_mask, 0]
            if "shape" in tokenized_agent
            else None
        )

        sampling_scheme = self.validation_rollout_sampling

        # 4. M open-loop targets (no_grad, ref_flow_decoder swap)
        target_stack: Tensor
        gt_valid: Tensor | None = None
        if use_gt_target:
            gt_target = self._ocsc_build_gt_target_norm(
                data=data,
                active_mask=active_mask,
                current_pos=current_pos_active,
                current_head=current_head_active,
                window_steps_10hz=loss_window_steps_10hz,
            )
            if gt_target is None:
                return self._build_trainable_connected_zero_loss(self.encoder)
            gt_norm, gt_valid = gt_target
            target_stack = gt_norm.detach().unsqueeze(0)
        else:
            with torch.no_grad():
                target_stack = self._ocsc_sample_open_loop_with_ref_batched(
                    active_hidden=active_hidden,
                    active_agent_type=active_agent_type,
                    active_agent_length=active_agent_length,
                    tokenized_agent=tokenized_agent,
                    active_mask=active_mask,
                    scenario_ids=data["scenario_id"],
                    sample_count=M,
                    sampling_scheme=sampling_scheme,
                    output_space=match_space,
                ).detach()
            if target_stack.shape[2] > loss_window_steps_10hz:
                target_stack = target_stack[:, :, :loss_window_steps_10hz].contiguous()

        # 5. G closed-loop rollouts (with grad), returned in the configured match space.
        #    pose: world frame pose -> anchor-0 local `[x/20, y/20, cos, sin]`.
        #    control: raw normalized committed `[delta_s, delta_n, delta_yaw]`.
        win_10hz = int(loss_window_steps_10hz)
        rollout_steps_2hz = max(1, math.ceil(win_10hz / int(agent_enc.shift)))
        cl_stack = self._ocsc_run_closed_loop_rollouts_batched(
            data=data,
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            rollout_cache=rollout_cache,
            active_mask=active_mask,
            current_pos_active=current_pos_active,
            current_head_active=current_head_active,
            sampling_scheme=sampling_scheme,
            rollout_count=G,
            window_steps_10hz=win_10hz,
            rollout_steps_2hz=rollout_steps_2hz,
            match_space=match_space,
        )  # pose: [G, n_active, T, 4], control: [G, n_active, T, 3]

        # 5b. Temporal loss sampling.  Default(-1) preserves the historical 2 Hz
        # coarse endpoints.  Set stride=1 to match all 10 Hz steps.
        shift = int(agent_enc.shift)
        loss_temporal_stride = shift if loss_stride_raw <= 0 else max(1, int(loss_stride_raw))
        if loss_temporal_stride > 1 and cl_stack.shape[2] >= loss_temporal_stride:
            cl_stack = cl_stack[:, :, loss_temporal_stride - 1::loss_temporal_stride, :]
            target_stack = target_stack[:, :, loss_temporal_stride - 1::loss_temporal_stride, :]
            if gt_valid is not None:
                gt_valid = gt_valid[:, loss_temporal_stride - 1::loss_temporal_stride]

        # 6. paired loss (nearest_match or 단순 mean)
        if use_gt_target:
            # GT 1 개 target — 모든 CL g 가 GT 와 paired L2 → 평균.
            target = target_stack[0]
            total_loss = sum(
                self._ocsc_paired_pose_loss(cl_g, target, pos_w, head_w, valid_mask=gt_valid)
                for cl_g in cl_stack
            ) / float(G)
        elif nearest_match:
            # per CL g, batch-flat argmin OL pool → paired L2 with one scene-level target.
            losses = []
            for g, cl_g in enumerate(cl_stack):
                with torch.no_grad():
                    flat_diff = target_stack - cl_g.detach().unsqueeze(0)
                    flat_sq = flat_diff.float().square()
                    d_pos = flat_sq[..., :2].sum(dim=(1, 2, 3))
                    if flat_sq.shape[-1] > 2:
                        d_head = flat_sq[..., 2:].sum(dim=(1, 2, 3))
                    else:
                        d_head = torch.zeros_like(d_pos)
                    nearest_m = (pos_w * d_pos + head_w * d_head).argmin()
                chosen = target_stack[int(nearest_m)]
                losses.append(self._ocsc_paired_pose_loss(cl_g, chosen, pos_w, head_w))
            total_loss = sum(losses) / float(G)
        else:
            # M == G, paired index (g, g) — 기본 paired L2 (ablation).
            if M != G:
                raise ValueError(
                    f"ocsc_ft: ocsc_ol_nearest_match=False requires M==G, got M={M}, G={G}."
                )
            total_loss = sum(
                self._ocsc_paired_pose_loss(cl_g, target_stack[g], pos_w, head_w)
                for g, cl_g in enumerate(cl_stack)
            ) / float(G)

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                "Non-finite ocsc_ft loss: "
                f"{self._summarize_nonfinite_tensor(total_loss)}"
            )

        self.log(
            "train/ocsc_ft/loss",
            total_loss.detach(),
            on_step=True,
            on_epoch=False,
            batch_size=1,
        )
        return total_loss

    def training_step(self, data, batch_idx):
        """한 batch의 Flow Matching loss를 계산합니다.

        Args:
            data: 학습용 장면 배치입니다.
            batch_idx: 현재 batch 번호입니다.

        Returns:
            Tensor: 최종 학습 loss입니다.
        """
        if self._is_ocsc_ft_enabled():
            return self._run_flow_ocsc_ft_step(data=data, batch_idx=batch_idx)
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
        has_open_loop_targets_pending = self._start_distributed_bool_any(
            has_open_loop_targets,
            device=total_loss.device,
        )
        self._automatic_open_loop_has_target_pending.append(has_open_loop_targets_pending)
        return total_loss

    def on_before_optimizer_step(self, optimizer) -> None:
        """DDP 전체에 target이 없는 automatic optimization step의 업데이트를 막습니다."""
        if not bool(getattr(self, "automatic_optimization", True)):
            return
        if self._is_ocsc_ft_enabled():
            return
        has_open_loop_targets_global = bool(self._automatic_open_loop_has_target_since_step)
        has_open_loop_target_pending = getattr(
            self,
            "_automatic_open_loop_has_target_pending",
            None,
        )
        if has_open_loop_target_pending is None:
            has_open_loop_targets_global = self._sync_distributed_bool_any(
                has_open_loop_targets_global,
            )
        elif has_open_loop_target_pending:
            has_open_loop_targets_global = any(
                self._finish_distributed_bool_any(pending)
                for pending in has_open_loop_target_pending
            )
            has_open_loop_target_pending.clear()
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
            generator_optimizer = torch.optim.AdamW(
                generator_params,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
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
                weight_decay=self.weight_decay,
            )
            return [generator_optimizer, generated_estimator_optimizer]

        trainable_params = [param for param in self.parameters() if param.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found for optimization.")
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]

    def on_train_epoch_start(self) -> None:
        """새 epoch의 open-loop train metric accumulator를 초기화합니다."""
        self._reset_open_loop_train_epoch_metrics()
        self._automatic_open_loop_has_target_pending.clear()

    def on_train_epoch_end(self) -> None:
        """self-forced manual optimization에서 scheduler가 있으면 epoch마다 한 번 진행합니다.

        Returns:
            None
        """
        self._log_open_loop_train_epoch_metrics()
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

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        self._ref_flow_decoder_loaded_from_ckpt = any(
            str(key).startswith("ref_flow_decoder.") for key in state_dict.keys()
        )
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)

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
