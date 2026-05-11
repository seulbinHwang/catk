from __future__ import annotations

import contextlib
import gc
import hashlib
import math
import os
from pathlib import Path
from typing import Dict, Sequence, Tuple

import hydra
import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

from src.smart.metrics import (
    HardSimAgentsMetrics,
    SimAgentsMetrics,
    SimAgentsSubmission,
    WOSACDistributionMetrics,
    log_and_reset_wosac_distribution_metric,
    minADE,
    update_wosac_distribution_metric_from_model,
)
from src.smart.metrics.flow_metrics import (
    WeightedMeanMetric,
    ade_2s,
    fde_2s,
    flow_matching_loss,
    yaw_ade_2s,
    yaw_fde_2s,
)
from src.smart.metrics.mmd_consistency_loss import (
    mmd_from_stacked,
    mmd_precompute_sigma_sq,
    mmd_per_rollout_proxy,
)
from src.smart.modules.flow_adjoint_matching import AdjointMatchingLoss, SmoothControlProjector
from src.smart.modules.flow_kinematic_projection import KinematicProjection
from src.smart.modules.flow_reward import KinematicProjectionReward
from src.smart.utils.geometry import wrap_angle
from src.smart.utils.rollout import transform_to_local
from src.smart.modules.flow_projected_generation import ProjectedFlowGenerator
from src.smart.modules.flow_terminal_cost_final_step import TerminalCostFinalStepLoss
from src.smart.modules.smart_flow_decoder import SMARTFlowDecoder
from src.smart.tokens.flow_token_processor import FlowTokenProcessor
from src.smart.utils.finetune import FinetuneConfig, set_model_for_finetuning
from src.utils.pylogger import RankedLogger
from src.utils.vis_waymo import VisWaymo
from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts


#Surrogate metrics
from waymo_open_dataset.protos import scenario_pb2, sim_agents_metrics_pb2
from src.smart.metrics.wosac_metric_features_torch.metric_features_torch_differentiable import (
    PredictedSimTrajectories,
    compute_metric_features_from_predicted_sim_trajectories,
    compute_metric_features_batched_scenes,
)
from src.smart.metrics.wosac_metametric_pytorch_differentiable import (
    compute_wosac_metametric_soft,
    compute_wosac_metametric_soft_batched,
    WosacMetametricSoftResult,
)
from src.smart.metrics.wosac_metric_features_torch.surrogate import SurrogateConfig

log = RankedLogger(__name__, rank_zero_only=True)

# ── Per-process caches (survive across Lightning training steps) ──────────────
# Scenario protos: eliminates TFRecord disk I/O + proto parse on repeat steps.
# 256 scenarios × ~2 MB/proto ≈ 512 MB peak RAM — acceptable for a training machine.
_SCENARIO_PROTO_CACHE: dict = {}
_SCENARIO_PROTO_CACHE_MAX: int = 256

# log_feat_dict: eliminates compute_metric_features (TF-based) on repeat steps.
# Stored as CPU tensors. ~100 KB/scenario × 2048 ≈ 200 MB peak.
_LOG_FEAT_DICT_CACHE: dict = {}
_LOG_FEAT_DICT_CACHE_MAX: int = 2048


def _slice_log_feat_dict_to_pred_horizon(
    log_feat_dict: dict[str, Tensor],
    t_horizon: int,
) -> dict[str, Tensor]:
    """GT log metric features를 예측 궤적 길이(10Hz ``T``)에 맞게 잘라 soft RMM log/sim 정합을 맞춥니다."""
    out: dict[str, Tensor] = {}
    for k, v in log_feat_dict.items():
        if isinstance(v, Tensor) and v.ndim >= 3 and v.shape[-1] > t_horizon:
            out[k] = v[..., :t_horizon]
        else:
            out[k] = v
    return out


class SMARTFlow(LightningModule):

    automatic_optimization = False

    def __init__(self, model_config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = model_config.lr
        self.lr_warmup_steps = int(model_config.lr_warmup_steps)
        self.lr_total_steps = int(model_config.lr_total_steps)
        self.lr_min_ratio = float(model_config.lr_min_ratio)
        self.weight_decay = float(getattr(model_config, "weight_decay", 0.01))
        self.lr_scheduler_unit = str(getattr(model_config, "lr_scheduler_unit", "epoch"))
        if self.lr_scheduler_unit not in {"epoch", "step"}:
            raise ValueError(f"Unsupported lr_scheduler_unit: {self.lr_scheduler_unit}")
        self.num_historical_steps = model_config.decoder.num_historical_steps
        self.log_epoch = -1
        self.val_open_loop = model_config.val_open_loop
        self.val_closed_loop = model_config.val_closed_loop
        self.token_processor = FlowTokenProcessor(**model_config.token_processor)

        self.encoder = SMARTFlowDecoder(
            **model_config.decoder,
            n_token_agent=self.token_processor.n_token_agent,
        )
        self.finetune_config: FinetuneConfig = set_model_for_finetuning(
            self.encoder,
            model_config.finetune,
        )
        self.adjoint_matching_loss = None
        self.terminal_cost_final_step_loss: TerminalCostFinalStepLoss | None = None
        self.kinematic_reward_fn: KinematicProjectionReward | None = None
        self.ref_flow_decoder: nn.Module | None = None
        self._dpo_debug_path: str | None = None
        self.dice_critic: nn.Module | None = None
        if self.finetune_config.enabled:
            if self.finetune_config.mode == "adjoint_matching":
                self.adjoint_matching_loss = AdjointMatchingLoss(
                    rollout_steps=self.finetune_config.rollout_steps,
                    rollout_noise_scale=self.finetune_config.rollout_noise_scale,
                    feasible_weight=self.finetune_config.feasible_weight,
                    smooth_deadzone_epsilon=self.finetune_config.smooth_deadzone_epsilon,
                    smooth_deadzone_tau=self.finetune_config.smooth_deadzone_tau,
                )
            elif self.finetune_config.mode in {
                "terminal_cost_final_step",
                "terminal_cost_full_grad",
            }:
                self.terminal_cost_final_step_loss = TerminalCostFinalStepLoss(
                    rollout_steps=self.finetune_config.rollout_steps,
                    rollout_noise_scale=self.finetune_config.rollout_noise_scale,
                    feasible_weight=self.finetune_config.feasible_weight,
                    smooth_deadzone_epsilon=self.finetune_config.smooth_deadzone_epsilon,
                    smooth_deadzone_tau=self.finetune_config.smooth_deadzone_tau,
                    flow_reg_lambda=self.finetune_config.flow_reg_lambda,
                )
            elif self.finetune_config.mode == "kinematic_reward_ft":
                # KinematicProjectionReward는 plain callable — TerminalCostFinalStepLoss가
                # BPTT ODE 인프라를 제공하고 forward_reward_grad로 reward를 연결합니다.
                # SmoothControlProjector를 만들지만 forward_feasibility_with_bc는 호출 안 합니다.
                self.terminal_cost_final_step_loss = TerminalCostFinalStepLoss(
                    rollout_steps=self.finetune_config.rollout_steps,
                    rollout_noise_scale=self.finetune_config.rollout_noise_scale,
                    feasible_weight=self.finetune_config.feasible_weight,
                    smooth_deadzone_epsilon=self.finetune_config.smooth_deadzone_epsilon,
                    smooth_deadzone_tau=self.finetune_config.smooth_deadzone_tau,
                    flow_reg_lambda=self.finetune_config.flow_reg_lambda,
                )
                # kinematic_reward_fn은 kinematic_projector가 설정된 후 (아래) 초기화됩니다.
            elif self.finetune_config.mode == "kinematic_proj_ft":
                # ODE → KinematicProjection → FM target; 별도 loss 모듈 불필요.
                pass
            elif self.finetune_config.mode == "rmm_bptt_ft":
                pass
            elif self.finetune_config.mode == "ocsc_ft":
                pass
            elif self.finetune_config.mode == "ref_nll_ft":
                pass
            else:
                raise ValueError(f"Unsupported finetune mode: {self.finetune_config.mode}")

        self.minADE = minADE()
        if bool(getattr(model_config, "wosac_torch_compile", False)):
            os.environ["WOSAC_TORCH_COMPILE"] = "1"

        _validation_metric = str(getattr(model_config, "validation_metric", "real")).lower()
        if _validation_metric == "hard":
            self.sim_agents_metrics = HardSimAgentsMetrics("val_closed")
        else:
            self.sim_agents_metrics = SimAgentsMetrics(
                "val_closed",
                max_workers=model_config.sim_agents_metric_workers,
            )
        self.sim_agents_submission = SimAgentsSubmission(**model_config.sim_agents_submission)

        wosac_cpd_reference = getattr(model_config, "wosac_cpd_reference", None)
        self.wosac_distribution_metrics = WOSACDistributionMetrics(
            prefix="val_closed",
            cpd_reference=wosac_cpd_reference,
        )
        self.test_wosac_distribution_metrics = WOSACDistributionMetrics(
            prefix="test",
            cpd_reference=wosac_cpd_reference,
        )

        # OCSC: per-step HardRMM 모니터링용 인-프로세스 metric 객체 (current + ref)
        _is_ocsc = self.finetune_config.enabled and self.finetune_config.mode == "ocsc_ft"
        if _is_ocsc and bool(getattr(self.finetune_config, "ocsc_eval_hard_rmm", True)):
            self._ocsc_train_hard_rmm: HardSimAgentsMetrics | None = HardSimAgentsMetrics("train_ocsc")
            self._ocsc_train_hard_rmm_ref: HardSimAgentsMetrics | None = HardSimAgentsMetrics("train_ocsc_ref")
        else:
            self._ocsc_train_hard_rmm = None
            self._ocsc_train_hard_rmm_ref = None

        # pretrained ref model Δ RMM 모니터링 플래그 (train / val 독립)
        _is_bptt = self.finetune_config.enabled and self.finetune_config.mode == "rmm_bptt_ft"
        self._ref_train_enabled: bool = (
            _is_bptt and bool(getattr(self.finetune_config, "rmm_bptt_ref_train", False))
        )
        _ref_val_on = (
            _is_bptt
            and bool(getattr(self.finetune_config, "rmm_bptt_ref_val", False))
            and bool(getattr(model_config, "val_closed_loop", True))
        )
        self._ref_val_enabled: bool = _ref_val_on
        if _ref_val_on:
            if _validation_metric == "hard":
                self.ref_sim_agents_metrics: HardSimAgentsMetrics | SimAgentsMetrics | None = HardSimAgentsMetrics("val_ref")
            else:
                self.ref_sim_agents_metrics = SimAgentsMetrics(
                    "val_ref",
                    max_workers=model_config.sim_agents_metric_workers,
                )
        else:
            self.ref_sim_agents_metrics = None

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
        self.delete_local_videos_after_wandb_upload = model_config.delete_local_videos_after_wandb_upload
        self.n_batch_sim_agents_metric = model_config.n_batch_sim_agents_metric
        self._fit_time_original_limit_val_batches: int | float | None = None
        self._fit_time_checkpoint_only_validation_enabled = False

        # EMA reward whitening buffers for rmm_bptt_ft.
        # Normalise loss = -(rmm - ema_mean) / (ema_std + eps) so gradient scale
        # stays consistent across scenarios with very different RMM baselines.
        if (
            getattr(model_config, "finetune", None) is not None
            and str(getattr(model_config.finetune, "mode", "")) == "rmm_bptt_ft"
        ):
            self.register_buffer("_rmm_ema_mean", torch.tensor(0.5))
            self._rmm_ema_initialized = False

        self.video_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.video_dir = Path(self.video_dir) / "videos"

        self.eval_sampling_noise = model_config.eval_sampling_noise

        # Projected Diffusion generation (inference-time feasibility projection)
        proj_cfg = getattr(model_config, "projected_generation", None)
        self.projected_generator: ProjectedFlowGenerator | None = None
        if proj_cfg is not None and getattr(proj_cfg, "enabled", False):
            _projector = SmoothControlProjector(
                feasible_weight=self.finetune_config.feasible_weight,
                smooth_deadzone_epsilon=self.finetune_config.smooth_deadzone_epsilon,
                smooth_deadzone_tau=self.finetune_config.smooth_deadzone_tau,
            )
            self.projected_generator = ProjectedFlowGenerator(
                projector=_projector,
                n_proj_steps=int(getattr(proj_cfg, "n_proj_steps", 3)),
                proj_lr=float(getattr(proj_cfg, "proj_lr", 0.01)),
            )
            self.val_projected_epoch_metrics = nn.ModuleDict(
                {
                    "proj_ADE2s": WeightedMeanMetric(),
                    "proj_FDE2s": WeightedMeanMetric(),
                    "proj_yaw_ADE2s": WeightedMeanMetric(),
                    "proj_yaw_FDE2s": WeightedMeanMetric(),
                }
            )

        # Final projection: ODE 완료 후 마지막 한 번만 kinematic projection 적용 (PPR final-step only 버전)
        final_proj_cfg = getattr(model_config, "final_projection", None)
        kin_cfg = getattr(model_config, "kinematic_projection", None)

        def _kin_proj_from_cfg() -> KinematicProjection:
            """kinematic_projection 블록이 있으면 그 하이퍼파라미터를 쓰고, 없으면 기본값."""
            def _ka(attr: str, default):
                return getattr(kin_cfg, attr, default) if kin_cfg is not None else default

            return KinematicProjection(
                coord_scale=20.0,
                dt=0.1,
                wheelbase=float(_ka("wheelbase", 2.7)),
                delta_max=float(_ka("delta_max", 0.52)),
                a_max=float(_ka("a_max", 4.0)),
                d_max=float(_ka("d_max", 8.0)),
                delta_rate_max=float(_ka("delta_rate_max", 0.6)),
                ped_a_max=float(_ka("ped_a_max", 2.0)),
                eps=float(_ka("eps", 1e-6)),
                use_lqr=bool(_ka("use_lqr", True)),
                lqr_q_xy=float(_ka("lqr_q_xy", 2.0)),
                lqr_q_yaw=float(_ka("lqr_q_yaw", 2.0)),
                lqr_q_v=float(_ka("lqr_q_v", 0.5)),
                lqr_q_delta=float(_ka("lqr_q_delta", 0.2)),
                lqr_r_a=float(_ka("lqr_r_a", 0.2)),
                lqr_r_delta_rate=float(_ka("lqr_r_delta_rate", 0.2)),
                lqr_qf_scale=float(_ka("lqr_qf_scale", 2.0)),
            )

        self._final_proj_kin_projector: KinematicProjection | None = None
        self.final_proj_generator: ProjectedFlowGenerator | None = None  # deprecated, no longer used
        if final_proj_cfg is not None and getattr(final_proj_cfg, "enabled", False):
            self._final_proj_kin_projector = _kin_proj_from_cfg()
            self.val_final_proj_epoch_metrics = nn.ModuleDict(
                {
                    "final_proj_ADE2s": WeightedMeanMetric(),
                    "final_proj_FDE2s": WeightedMeanMetric(),
                    "final_proj_yaw_ADE2s": WeightedMeanMetric(),
                    "final_proj_yaw_FDE2s": WeightedMeanMetric(),
                }
            )

        self.val_open_epoch_metrics = nn.ModuleDict(
            {
                "ADE2s": WeightedMeanMetric(),
                "FDE2s": WeightedMeanMetric(),
                "yaw_ADE2s": WeightedMeanMetric(),
                "yaw_FDE2s": WeightedMeanMetric(),
            }
        )

        if kin_cfg is not None and getattr(kin_cfg, "enabled", False):
            _kin_proj = _kin_proj_from_cfg()
            # attach to the agent encoder so both open-loop and closed-loop paths use it
            self.encoder.agent_encoder.kinematic_projector = _kin_proj
            self.encoder.agent_encoder.use_predict_project_renoise = bool(
                getattr(kin_cfg, "predict_project_renoise", False)
            )
            _ppr_steps = getattr(kin_cfg, "ppr_steps", None)
            self.encoder.agent_encoder.ppr_steps = int(_ppr_steps) if _ppr_steps is not None else None
            # Kinematic post-processing option (closed-loop, separate from PPR)
            _pp_cfg = getattr(model_config, "kinematic_postproc", None)
            self.encoder.agent_encoder.use_kinematic_postproc = bool(
                getattr(_pp_cfg, "enabled", False)
            ) if _pp_cfg is not None else False
            self.val_kinematic_proj_epoch_metrics = nn.ModuleDict(
                {
                    "kin_ADE2s": WeightedMeanMetric(),
                    "kin_FDE2s": WeightedMeanMetric(),
                    "kin_yaw_ADE2s": WeightedMeanMetric(),
                    "kin_yaw_FDE2s": WeightedMeanMetric(),
                }
            )
            # kinematic_reward_ft / dice_ft(reward_enabled): kinematic_projector가
            # 이제 설정됐으므로 reward fn 초기화
            _needs_kin_reward = (
                self.finetune_config.enabled
                and self.kinematic_reward_fn is None
                and (
                    self.finetune_config.mode == "kinematic_reward_ft"
                    or (
                        self.finetune_config.mode == "dice_ft"
                        and self.finetune_config.dice_reward_enabled
                    )
                )
            )
            if _needs_kin_reward:
                self.kinematic_reward_fn = KinematicProjectionReward(
                    kinematic_projector=self.encoder.agent_encoder.kinematic_projector,
                    huber_beta=self.finetune_config.reward_huber_beta,
                )
        elif (
            self.finetune_config.enabled
            and self.finetune_config.mode == "kinematic_reward_ft"
        ):
            raise ValueError(
                "kinematic_reward_ft requires kinematic_projection.enabled=True"
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

    def _build_open_loop_metric_dict(
        self,
        pred_clean_norm: Tensor,
        target_clean_norm: Tensor,
    ) -> Dict[str, Tensor]:
        """2초 open-loop 위치와 방향 오차를 계산합니다.

        Args:
            pred_clean_norm: 모델이 만든 정규화된 미래입니다.
                shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            target_clean_norm: 정답 정규화 미래입니다.
                shape은 ``[n_valid_anchor, 20, 4]`` 입니다.

        Returns:
            Dict[str, Tensor]:
                meter 단위 위치 오차와 degree 단위 방향 오차를 담은 사전입니다.
        """
        with torch.no_grad():
            return {
                "ADE2s": ade_2s(
                    pred_clean_norm.detach(),
                    target_clean_norm.detach(),
                ),
                "FDE2s": fde_2s(
                    pred_clean_norm.detach(),
                    target_clean_norm.detach(),
                ),
                "yaw_ADE2s": yaw_ade_2s(
                    pred_clean_norm.detach(),
                    target_clean_norm.detach(),
                ),
                "yaw_FDE2s": yaw_fde_2s(
                    pred_clean_norm.detach(),
                    target_clean_norm.detach(),
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
                ``[n_valid_anchor, 20, 4]`` 입니다.

        Returns:
            tuple[Tensor, Dict[str, Tensor], int]:
                flow matching loss, meter/degree 단위 지표 사전,
                그리고 유효 anchor 개수입니다.
        """
        loss = flow_matching_loss(pred_dict["flow_pred_norm"], pred_dict["flow_target_norm"])
        metric_dict = self._build_open_loop_metric_dict(
            pred_clean_norm=pred_dict["flow_pred_clean_norm"],
            target_clean_norm=pred_dict["flow_clean_norm"],
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

        return {
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
        data,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        rollout_cache: Dict[str, object],
        rollout_indices: Sequence[int],
        return_anchor_hidden: bool = False,
        full_grad: bool = False,
        max_steps: int | None = None,
        warm_coarse_steps: int = 0,
        share_noise_across_time: bool = False,
        noise_tape_override: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        """주어진 rollout 번호 묶음을 한 번의 큰 batch로 실행합니다.

        Args:
            data: dataloader가 준 원본 batch입니다.
            tokenized_agent: 평가용 agent 토큰 사전입니다.
                agent 축 텐서는 ``[n_agent, ...]`` 입니다.
            map_feature: 한 번 인코딩한 지도 특징입니다.
                지도 토큰 축 텐서는 ``[n_map_token, ...]`` 입니다.
            rollout_cache: 원본 closed-loop cache 입니다.
            rollout_indices: 이번에 한꺼번에 돌릴 rollout 번호 목록입니다.
                길이는 ``[n_rollout_chunk]`` 입니다.

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                위치, 높이, 방향 예측입니다.
                shape은 각각 ``[n_agent, n_rollout_chunk, 80, 2]``,
                ``[n_agent, n_rollout_chunk, 80]``,
                ``[n_agent, n_rollout_chunk, 80]`` 입니다.
        """
        chunk_size = int(len(rollout_indices))
        scenario_device = tokenized_agent["batch"].device
        if chunk_size == 1:
            scenario_sampling_seeds = self._get_closed_loop_scenario_seeds(
                scenario_ids=data["scenario_id"],
                rollout_idx=int(rollout_indices[0]),
                device=scenario_device,
            )
            if full_grad:
                pred = self.encoder.rollout_from_cache(
                    rollout_cache=rollout_cache,
                    tokenized_agent=tokenized_agent,
                    map_feature=map_feature,
                    sampling_noise=self.eval_sampling_noise,
                    scenario_sampling_seeds=scenario_sampling_seeds,
                    max_steps=max_steps,
                    warm_coarse_steps=warm_coarse_steps,
                    share_noise_across_time=share_noise_across_time,
                    noise_tape_override=noise_tape_override,
                )
            else:
                pred = self.encoder.rollout_from_cache_no_grad(
                    rollout_cache=rollout_cache,
                    tokenized_agent=tokenized_agent,
                    map_feature=map_feature,
                    sampling_noise=self.eval_sampling_noise,
                    scenario_sampling_seeds=scenario_sampling_seeds,
                )
            base_ret = (
                pred["pred_traj_10hz"].unsqueeze(1),
                pred["pred_z_10hz"].unsqueeze(1),
                pred["pred_head_10hz"].unsqueeze(1),
            )
            if not return_anchor_hidden:
                return base_ret
            return base_ret + (pred["anchor_hidden_2hz"].unsqueeze(1),)

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
        if full_grad:
            pred = self.encoder.rollout_from_cache(
                rollout_cache=expanded_rollout_cache,
                tokenized_agent=expanded_tokenized_agent,
                map_feature=expanded_map_feature,
                sampling_noise=self.eval_sampling_noise,
                scenario_sampling_seeds=scenario_seed_table.reshape(-1).contiguous(),
                max_steps=max_steps,
                warm_coarse_steps=warm_coarse_steps,
                share_noise_across_time=share_noise_across_time,
            )
        else:
            pred = self.encoder.rollout_from_cache_no_grad(
                rollout_cache=expanded_rollout_cache,
                tokenized_agent=expanded_tokenized_agent,
                map_feature=expanded_map_feature,
                sampling_noise=self.eval_sampling_noise,
                scenario_sampling_seeds=scenario_seed_table.reshape(-1).contiguous(),
            )
        base_ret = (
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
        )
        if not return_anchor_hidden:
            return base_ret
        return base_ret + (
            self._reshape_parallel_rollout_prediction(
                pred["anchor_hidden_2hz"],
                repeat_count=chunk_size,
                num_agent=num_agent,
            ),
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
        data,
        tokenized_agent,
        map_feature: Dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """한 batch의 모든 closed-loop rollout을 가능한 크게 묶어 생성합니다.

        기본은 모든 rollout을 한 번에 큰 batch로 처리합니다.
        다만 메모리가 부족하면 자동으로 묶음 크기를 절반 정도씩 줄여
        같은 결과 shape을 유지한 채 다시 시도합니다.

        Args:
            data: dataloader가 준 원본 batch입니다.
            tokenized_agent: 평가용 agent 토큰 사전입니다.
            map_feature: 한 번 인코딩한 지도 특징입니다.

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                위치, 높이, 방향 예측입니다.
                shape은 각각 ``[n_agent, n_rollout, 80, 2]``,
                ``[n_agent, n_rollout, 80]``,
                ``[n_agent, n_rollout, 80]`` 입니다.
        """
        rollout_cache = self.encoder.prepare_inference_cache(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
        )
        rollout_indices = list(range(int(self.n_rollout_closed_val)))
        last_oom_error: RuntimeError | None = None

        for chunk_size in self._build_rollout_chunk_size_candidates():
            pred_traj_chunks: list[Tensor] = []
            pred_z_chunks: list[Tensor] = []
            pred_head_chunks: list[Tensor] = []
            try:
                for chunk_start in range(0, len(rollout_indices), chunk_size):
                    chunk_rollout_indices = rollout_indices[chunk_start : chunk_start + chunk_size]
                    chunk_pred_traj, chunk_pred_z, chunk_pred_head = self._run_parallel_rollout_chunk(
                        data=data,
                        tokenized_agent=tokenized_agent,
                        map_feature=map_feature,
                        rollout_cache=rollout_cache,
                        rollout_indices=chunk_rollout_indices,
                    )
                    pred_traj_chunks.append(chunk_pred_traj)
                    pred_z_chunks.append(chunk_pred_z)
                    pred_head_chunks.append(chunk_pred_head)
                return (
                    torch.cat(pred_traj_chunks, dim=1),
                    torch.cat(pred_z_chunks, dim=1),
                    torch.cat(pred_head_chunks, dim=1),
                )
            except RuntimeError as error:
                if (not self._is_cuda_out_of_memory(error)) or chunk_size == 1:
                    raise
                last_oom_error = error
                del pred_traj_chunks, pred_z_chunks, pred_head_chunks
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

    def on_fit_start(self) -> None:
        """학습 시작 전에 빠른 closed-loop validation 모드를 켭니다.

        Lightning은 ``on_fit_start`` 를 sanity check 전에 호출합니다.
        그래서 여기서 validation batch 개수를 줄이면 학습 전 sanity check와
        학습 중 validation 둘 다 같은 빠른 규칙을 사용하게 됩니다.

        Returns:
            None
        """
        self._apply_fit_time_validation_batch_limit()

    def on_fit_end(self) -> None:
        """학습이 끝나면 임시로 바꾼 validation 제한 값을 정리합니다.

        Returns:
            None
        """
        self._restore_fit_time_validation_batch_limit()

    # ──────────────────────────────────────────────────────────────────────
    # Kinematic init helper (open-loop val / final-proj val / kinematic_proj_ft 공용)
    # ──────────────────────────────────────────────────────────────────────

    def _compute_kinematic_init(
        self,
        tokenized_agent: Dict,
        anchor_mask_tensor,
        kp,
    ):
        """ctx_sampled_pos/heading 기반 v_init, delta_init 계산.

        closed-loop chunk 초기화와 동일한 coarse-dt 변위 공식.

        Args:
            tokenized_agent: token processor 출력 사전.
            anchor_mask_tensor: [n_agent, n_anchor] bool, 유효 anchor 마스크.
            kp: KinematicProjection 인스턴스 (wheelbase, delta_max, dt 참조용).

        Returns:
            (v_init [n_valid], delta_init [n_valid]) or (None, None)
        """
        if (
            anchor_mask_tensor is None
            or not anchor_mask_tensor.any()
            or "ctx_sampled_pos" not in tokenized_agent
            or "ctx_sampled_heading" not in tokenized_agent
        ):
            return None, None

        ctx_pos = tokenized_agent["ctx_sampled_pos"]       # [n_agent, 14, 2]
        ctx_head = tokenized_agent["ctx_sampled_heading"]  # [n_agent, 14]
        _coarse_dt = float(self.encoder.agent_encoder.shift) * float(kp.dt)
        packed_v: list[Tensor] = []
        packed_d: list[Tensor] = []

        for anchor_idx in range(anchor_mask_tensor.shape[1]):
            mask_i = anchor_mask_tensor[:, anchor_idx]
            if not bool(mask_i.any()):
                continue
            dp = ctx_pos[:, anchor_idx + 1] - ctx_pos[:, anchor_idx]
            v_i = dp[mask_i].norm(dim=-1) / _coarse_dt
            packed_v.append(v_i)
            dtheta = wrap_angle(ctx_head[:, anchor_idx + 1] - ctx_head[:, anchor_idx])
            _v_c = v_i.clamp_min(1e-6)
            _kappa = dtheta[mask_i] / (_v_c * _coarse_dt + 1e-6)
            delta_i = torch.atan(kp.wheelbase * _kappa).clamp(-kp.delta_max, kp.delta_max)
            packed_d.append(delta_i)

        if not packed_v:
            return None, None
        return torch.cat(packed_v, dim=0), torch.cat(packed_d, dim=0)

    # ──────────────────────────────────────────────────────────────────────
    # Fine-tuning mode checks
    # ──────────────────────────────────────────────────────────────────────

    def _is_adjoint_matching_enabled(self) -> bool:
        """현재 학습이 Adjoint Matching 분기인지 확인합니다.

        Returns:
            bool: residual head만 학습하는 fine-tuning 단계면 ``True`` 입니다.
        """
        return bool(self.finetune_config.enabled and self.adjoint_matching_loss is not None)

    def _is_terminal_cost_final_step_enabled(self) -> bool:
        """현재 학습이 terminal_cost 기반 마지막 step gradient 분기인지 확인합니다."""
        return bool(
            self.finetune_config.enabled and self.terminal_cost_final_step_loss is not None
        )

    def _is_kinematic_proj_ft_enabled(self) -> bool:
        """kinematic_proj_ft 분기인지 확인합니다."""
        return bool(
            self.finetune_config.enabled
            and self.finetune_config.mode == "kinematic_proj_ft"
            and self.encoder.agent_encoder.kinematic_projector is not None
        )

    def _is_kinematic_reward_ft_enabled(self) -> bool:
        """kinematic_reward_ft 분기인지 확인합니다."""
        return bool(
            self.finetune_config.enabled
            and self.finetune_config.mode == "kinematic_reward_ft"
            and self.kinematic_reward_fn is not None
            and self.terminal_cost_final_step_loss is not None
        )

    def _is_rmm_bptt_ft_enabled(self) -> bool:
        return bool(
            self.finetune_config.enabled
            and self.finetune_config.mode == "rmm_bptt_ft"
        )

    def _is_ocsc_ft_enabled(self) -> bool:
        return bool(
            self.finetune_config.enabled
            and self.finetune_config.mode == "ocsc_ft"
        )

    def _is_ref_nll_ft_enabled(self) -> bool:
        return bool(
            self.finetune_config.enabled
            and self.finetune_config.mode == "ref_nll_ft"
        )

    def on_train_start(self) -> None:
        _needs_ref = (
            self.finetune_config.enabled
            and self.ref_flow_decoder is None
            and self.finetune_config.mode == "rmm_bptt_ft"
            and (
                self.finetune_config.rmm_bptt_use_ref_model
                or self._ref_train_enabled
                or self._ref_val_enabled
            )
        )
        # OCSC: ref_flow_decoder 를 항상 생성 (open-loop target 및 delta HardRMM 모니터링 공용)
        # ocsc_use_pretrained_ref=False 여도 delta 계산을 위해 frozen ref 가 필요하다.
        _needs_ref_ocsc = (
            self.finetune_config.enabled
            and self.ref_flow_decoder is None
            and self.finetune_config.mode == "ocsc_ft"
        )
        # ref_nll_ft: frozen reference flow decoder for likelihood reward
        _needs_ref_nll = (
            self.finetune_config.enabled
            and self.ref_flow_decoder is None
            and self.finetune_config.mode == "ref_nll_ft"
        )
        if _needs_ref or _needs_ref_ocsc or _needs_ref_nll:
            from copy import deepcopy
            flow_decoder = self.encoder.agent_encoder.flow_decoder
            self.ref_flow_decoder = deepcopy(flow_decoder)
            for p in self.ref_flow_decoder.parameters():
                p.requires_grad_(False)
            print(f"[{self.finetune_config.mode}] frozen reference model created from pretrained checkpoint.")

        # rmm_bptt_ft / ocsc_ft: BPTT backward through ODE steps can produce NaN/Inf
        # gradients (exploding Jacobian, numerical instability). Register nan_to_num
        # hooks on trainable parameters so any NaN/Inf gradient is zeroed out.
        if self._is_rmm_bptt_ft_enabled() or self._is_ocsc_ft_enabled() or self._is_ref_nll_ft_enabled():
            # NaN → 0, Inf → finite large value (not 0) so the optimizer still
            # sees the direction even under mild overflow.
            n_hooked = 0
            for p in self.parameters():
                if p.requires_grad:
                    p.register_hook(
                        lambda g: torch.nan_to_num(g, nan=0.0, posinf=1e4, neginf=-1e4)
                    )
                    n_hooked += 1
            log.info(f"[{self.finetune_config.mode}] registered nan_to_num grad hooks on {n_hooked} trainable params")

    def _world_traj_to_flow_norm(
        self,
        pred_traj: Tensor,   # [n, 20, 2]  world XY at 10Hz
        pred_head: Tensor,   # [n, 20]     world heading
        current_pos: Tensor, # [n, 2]      reference position
        current_head: Tensor, # [n]        reference heading
    ) -> Tensor:             # [n, 20, 4]  normalized [x/20, y/20, cos, sin]
        """Convert world-coordinate 20-step trajectory to normalized flow-space."""
        pos_local, head_local = transform_to_local(
            pos_global=pred_traj,
            head_global=pred_head,
            pos_now=current_pos,
            head_now=current_head,
        )
        return torch.stack(
            [
                pos_local[..., 0] / 20.0,
                pos_local[..., 1] / 20.0,
                head_local.cos(),
                head_local.sin(),
            ],
            dim=-1,
        )



    def _compute_soft_rmm(
        self,
        scenario: scenario_pb2.Scenario,
        x:    Tensor,  # [A, 80]
        y:    Tensor,
        z:    Tensor,
        head: Tensor,
        agent_ids: Tensor,
        valid: Tensor,
        log_feat_dict: dict,
        config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
        debug: bool = False,
    ) -> WosacMetametricSoftResult:
        pred = PredictedSimTrajectories(
            object_id=agent_ids.cpu(),
            center_x=x, center_y=y, center_z=z, heading=head, valid=valid,
        )
        SURROGATE = SurrogateConfig(
            collision_temperature=0.15,
            offroad_temperature=0.15,
            red_light_crossing_temperature=0.05,
        )
        sim_feat = compute_metric_features_from_predicted_sim_trajectories(
            scenario=scenario, pred=pred, surrogate=SURROGATE,
        )
        sim_feat_dict = sim_feat.as_dict()

        if debug:
            # 극값 로깅: gradient explosion 원인 후보 탐지
            with torch.no_grad():
                def _stat(t: Tensor, name: str) -> str:
                    v = t.detach().float()
                    return f"{name}=[{v.min():.3f},{v.max():.3f}]"
                log.warning(
                    "[soft_rmm_feat_debug] "
                    + " | ".join([
                        _stat(sim_feat_dict["linear_speed"],    "lin_spd"),
                        _stat(sim_feat_dict["angular_speed"],   "ang_spd"),
                        _stat(sim_feat_dict["distance_to_nearest_object"], "dno"),
                        _stat(sim_feat_dict["distance_to_road_edge"],      "d_road"),
                        _stat(sim_feat_dict["collision_per_step"].float(),  "coll"),
                        _stat(sim_feat_dict["offroad_per_step"].float(),    "offrd"),
                    ])
                )

        return compute_wosac_metametric_soft(
            config=config,
            log_features=log_feat_dict,
            sim_features=sim_feat_dict,
            debug=debug,
        )

    def _compute_rmm_group(
        self,
        data: dict,
        agent_ids: Tensor,    # [n_agents]
        agent_batch: Tensor,  # [n_agents]
        pred_traj: Tensor,    # [n_agents, G, 80, 2]
        pred_z: Tensor,       # [n_agents, G, 80]
        pred_head: Tensor,    # [n_agents, G, 80]
    ) -> Tensor:              # [n_scenarios, G]
        """Compute RMM for each of G rollouts, for each scenario.

        Returns:
            Float Tensor ``[n_scenarios, G]``. Returns zeros if tfrecord_path unavailable.
        """
        import multiprocessing as mp
        from src.smart.metrics import _sim_agents_worker, SimAgentsMetrics

        scenario_files = data.get("tfrecord_path", None)
        G = pred_traj.shape[1]
        n_scenarios = int(agent_batch.max().item()) + 1 if agent_batch.numel() > 0 else 0

        if scenario_files is None or n_scenarios == 0:
            return torch.zeros(n_scenarios, G)

        agent_batch_cpu = agent_batch.cpu()
        sizes = [int((agent_batch_cpu == i).sum()) for i in range(n_scenarios)]
        ids_list = agent_ids.cpu().split(sizes)
        traj_list = pred_traj.cpu().split(sizes)
        z_list = pred_z.cpu().split(sizes)
        head_list = pred_head.cpu().split(sizes)

        config_bytes = SimAgentsMetrics._load_config_bytes()

        # Build args: scenario × rollout (interleaved as sc0_r0, sc0_r1, ..., sc1_r0, ...)
        args_all = []
        for i in range(n_scenarios):
            ids_np = ids_list[i].numpy()
            t_np = traj_list[i].numpy()   # [n_i, G, 80, 2]
            z_np = z_list[i].numpy()
            h_np = head_list[i].numpy()
            for g in range(G):
                args_all.append((config_bytes, scenario_files[i], ids_np,
                                 t_np[:, g:g+1, :, :], z_np[:, g:g+1, :], h_np[:, g:g+1, :]))

        try:
            mp.set_start_method("forkserver", force=True)
        except RuntimeError:
            pass

        import os
        n_pool = min(len(args_all), max(1, (os.cpu_count() or 8) // 4))
        with mp.Pool(processes=n_pool) as pool:
            results = pool.starmap(_sim_agents_worker, args_all)
            pool.close()
            pool.join()

        # Reshape: results are [sc0_r0, sc0_r1, ..., sc1_r0, ...] → [n_scenarios, G]
        meta_vals = [r["metametric"] for r in results]
        rmm = torch.tensor(meta_vals, dtype=torch.float32).reshape(n_scenarios, G)
        return rmm

    def _compute_rmm_bptt_gt_fm_loss(
        self,
        map_feature: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
    ) -> Tensor | None:
        """GT 정규화 궤적에 대한 flow-matching MSE (velocity_head에만 gradient).

        ``kinematic_proj_ft`` 의 ``flow_reg_lambda`` BC 항과 동일한 경로:
        ``flow_train_clean_norm`` + ``flow_ode.sample(..., target_type='velocity')``.
        """
        gt_clean = tokenized_agent.get("flow_train_clean_norm")
        if gt_clean is None or gt_clean.numel() == 0:
            return None
        with torch.no_grad():
            _, _, anchor_hidden_valid = self.encoder.encode_anchor_context_from_map_feature(
                map_feature=map_feature,
                tokenized_agent=tokenized_agent,
                anchor_mask_key="flow_train_mask",
            )
        if anchor_hidden_valid.numel() == 0:
            return None
        anchor_hidden = anchor_hidden_valid.detach().to(dtype=torch.float32)
        flow_ode = self.encoder.agent_encoder.flow_ode
        flow_decoder = self.encoder.agent_encoder.flow_decoder
        gt_sample = flow_ode.sample(gt_clean.to(dtype=torch.float32), target_type="velocity")
        gt_pred = flow_decoder(anchor_hidden, gt_sample.x_t, gt_sample.tau)
        fm = flow_matching_loss(gt_pred, gt_sample.target)
        if not torch.isfinite(fm).all():
            log.warning("[rmm_bptt_ft] non-finite GT FM loss; skipping")
            return None
        return fm

    # ─────────────────────────────────────────────────────────────────────────
    # Ref-NLL fine-tuning
    # ─────────────────────────────────────────────────────────────────────────

    def _run_ref_nll_ft_step(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        data: dict | None = None,
    ) -> dict:
        """Reference-NLL fine-tuning step (closed-loop → open-loop likelihood).

        알고리즘:
          1. Closed-loop rollout (BPTT): G 번 AR rollout. coarse step 마다 0.5s 만 commit
             하면서 ``pred_max_steps`` 만큼 (= ``pred_max_steps × shift`` fine step) 굴려
             world-frame 의 2s 궤적 ``pred_traj_10hz`` 을 모은다.
          2. 모은 closed-loop 2s 궤적을 **초기 pose** (rollout 시작 시점의 pos/head) 의
             local frame 으로 변환해 ``x₁`` (= [n_active, 20, 4]) 을 만든다.
          3. Frozen ref_flow_decoder (open-loop pretrained) 의 backward ODE + Hutchinson 으로
             ``log p_ref(τ_2s | initial_anchor)`` 와 ``∂ log p_ref / ∂ x₁`` 계산.
          4. Straight-through loss: ``L = -mean(∂ log p_ref / ∂ x₁ · x₁)``
             → gradient = -(∂ log p_ref / ∂ x₁) 가 x₁ → pred_traj_10hz → flow_ode → θ
             로 BPTT 역전파. 즉, **closed-loop AR joint likelihood 를 open-loop
             p(τ|initial) 로 끌어올리는 covariate-shift 보정 fine-tuning** 이다.
          5. (선택) GT FM regularization.

        주의: open-loop ref 는 horizon 20 fine step (= 2s, 10Hz) 에 학습돼 있으므로
        ``pred_max_steps × shift == 20`` 이어야 한다. 다르면 경고 후 가능한 prefix 만 사용.
        """
        from src.smart.modules.flow_likelihood import backward_ode_log_prob_and_grad

        G = int(getattr(self.finetune_config, "ref_nll_n_rollouts", 2))
        pred_max_steps_raw = int(getattr(self.finetune_config, "ref_nll_pred_max_steps", 4))
        pred_max_steps: int | None = pred_max_steps_raw if pred_max_steps_raw > 0 else None
        n_hutch = int(getattr(self.finetune_config, "ref_nll_n_hutch_samples", 1))
        use_full_div_grad = bool(getattr(self.finetune_config, "ref_nll_use_full_div_grad", False))
        fm_reg_lambda = float(getattr(self.finetune_config, "ref_nll_fm_reg_lambda", 0.0))
        loss_scale = float(getattr(self.finetune_config, "ref_nll_loss_scale", 1.0))
        use_adjoint = bool(getattr(self.finetune_config, "bptt_use_adjoint", False))
        warm_coarse = int(getattr(self.finetune_config, "bptt_warm_coarse_steps", 0))
        _last_n_solver = int(getattr(self.finetune_config, "bptt_last_n_solver_steps", 0))
        _grad_clip = float(getattr(self.finetune_config, "bptt_grad_clip_traj", 1.0))

        if data is None:
            raise ValueError("ref_nll_ft requires `data` dict with scenario metadata.")
        if "scenario_id" not in data:
            raise KeyError("ref_nll_ft requires data['scenario_id'].")

        # ── 1. Encode map (no_grad) ──────────────────────────────────────────
        with torch.no_grad():
            map_feature = self.encoder.encode_map(tokenized_map)

        # ── 2. Build rollout cache (no_grad) ─────────────────────────────────
        with torch.no_grad():
            rollout_cache = self.encoder.agent_encoder.prepare_inference_cache(
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
            )

        _agent_enc = self.encoder.agent_encoder
        flow_ode = _agent_enc.flow_ode
        # fresh sample 생성 시 ODE backward 방식 설정
        flow_ode.use_adjoint_for_bptt = use_adjoint
        flow_ode.last_n_grad_solver_steps = (
            min(_last_n_solver, flow_ode.solver_steps) if _last_n_solver > 0 else 0
        )

        # Open-loop ref decoder 는 step_embed(20) + view(num_chunks, chunk_size) 로
        # x_t_norm shape [batch, 20, 4] 가 강제된다. 그러므로 closed-loop fine step 길이
        # 도 정확히 20 이어야 한다 (= pred_max_steps × shift).
        _shift = int(_agent_enc.shift)
        _open_loop_horizon = 20
        pred_max_steps_eff = (
            _open_loop_horizon // _shift if pred_max_steps is None else int(pred_max_steps)
        )
        n_fine = pred_max_steps_eff * _shift
        if n_fine != _open_loop_horizon:
            raise ValueError(
                f"[ref_nll_ft] pred_max_steps×shift={n_fine} but open-loop ref requires "
                f"exactly {_open_loop_horizon} fine steps. Set ref_nll_pred_max_steps="
                f"{_open_loop_horizon // _shift} (= {_open_loop_horizon // _shift * 0.5}s)."
            )

        # 초기 pose (rollout 시작 시점) — closed-loop 2s 궤적을 이 frame 으로 normalize 한다.
        initial_pos = rollout_cache["pos_window"][:, -1].detach().clone()    # [n_agent, 2]
        initial_head = rollout_cache["head_window"][:, -1].detach().clone()  # [n_agent]
        initial_active_mask = rollout_cache["valid_window"][:, -1].clone()   # [n_agent]

        def _make_grad_clip_hook(max_norm: float):
            def _hook(g: Tensor) -> Tensor:
                n = g.norm()
                return g if n <= max_norm else g * (max_norm / (n + 1e-6))
            return _hook

        total_loss_accum: Tensor | float = 0.0
        total_log_p_accum: float = 0.0
        n_valid_terms: int = 0

        # ── 3. G rollout (WITH grad, BPTT) ───────────────────────────────────
        for g in range(G):
            seeds_g = self._get_closed_loop_scenario_seeds(
                scenario_ids=data["scenario_id"],
                rollout_idx=g,
                device=tokenized_agent["batch"].device,
            )

            # Closed-loop rollout WITH gradient.
            # 0.5s 씩 commit 하며 pred_max_steps × 0.5s 만큼 world-frame 으로 굴린다.
            pred = self.encoder.rollout_from_cache(
                rollout_cache=rollout_cache,
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
                sampling_noise=self.eval_sampling_noise,
                scenario_sampling_seeds=seeds_g,
                max_steps=pred_max_steps,
                warm_coarse_steps=warm_coarse,
                return_per_step_x1=False,
            )

            # Closed-loop 2s 궤적 (world frame, 10Hz) — gradient 가 흐른다.
            pred_traj_10hz: Tensor = pred["pred_traj_10hz"]   # [n_agent, n_fine_total, 2]
            pred_head_10hz: Tensor = pred["pred_head_10hz"]   # [n_agent, n_fine_total]
            pred_valid: Tensor = pred["pred_valid"]           # history + coarse steps

            # rollout 끝까지 활성이었던 agent 만 사용. rollout 은 한 번 inactive 가 되면
            # 이후 next_valid 도 False 로 전파하므로 마지막 step 만 검사하면 충분하다.
            all_active = initial_active_mask & pred_valid[:, -1]
            if not bool(all_active.any()):
                continue

            traj_active = pred_traj_10hz[:, :n_fine][all_active]    # [n_active, n_fine, 2]
            head_active = pred_head_10hz[:, :n_fine][all_active]    # [n_active, n_fine]
            ipos_active = initial_pos[all_active]
            ihead_active = initial_head[all_active]

            # World-frame 2s 궤적 → 초기 frame 의 normalized [x/20, y/20, cosΔh, sinΔh].
            x1 = self._world_traj_to_flow_norm(
                pred_traj=traj_active,
                pred_head=head_active,
                current_pos=ipos_active,
                current_head=ihead_active,
            ).float()

            if x1.requires_grad and _grad_clip > 0:
                x1.register_hook(_make_grad_clip_hook(_grad_clip))

            # 초기 anchor (rollout t=0) 만 condition 으로 사용 — open-loop 와 동일.
            anchor_hidden_t0: Tensor = pred["anchor_hidden_2hz"][:, 0][all_active].detach().float()

            def _v_ref_fn(x: Tensor, tau: Tensor, _ah: Tensor = anchor_hidden_t0) -> Tensor:
                return self.ref_flow_decoder(_ah, x.float(), tau.float())

            log_p_g, grad_x1_g = backward_ode_log_prob_and_grad(
                x1=x1,
                v_fn=_v_ref_fn,
                steps=flow_ode.solver_steps,
                eps_t=flow_ode.eps,
                n_hutch=n_hutch,
                use_full_div_grad=use_full_div_grad,
            )

            if not torch.isfinite(log_p_g).all():
                log.warning(f"[ref_nll_ft] non-finite log_p at rollout {g}; skipping")
                continue

            total_log_p_accum += float(log_p_g.mean().item())
            n_valid_terms += 1

            # warm_coarse 가 전체 rollout 을 덮으면 x1 에 grad 가 없어 backward 불가 →
            # log_p 모니터링만 하고 loss accumulation 은 건너뛴다.
            if not x1.requires_grad:
                continue

            # Straight-through loss: L_g = -(grad_log_p · x₁)
            # gradient: x₁ → pred_traj_10hz → flow_ode[t] → θ  (BPTT through closed-loop)
            loss_g = -(
                (grad_x1_g.detach() * x1).flatten(1).sum(1).mean()
            ) * loss_scale / G

            total_loss_accum = total_loss_accum + loss_g

        # ── 5. GT FM regularization (선택) ────────────────────────────────────
        if fm_reg_lambda > 0.0:
            fm_loss = self._compute_rmm_bptt_gt_fm_loss(map_feature, tokenized_agent)
            if fm_loss is not None:
                total_loss_accum = total_loss_accum + fm_reg_lambda * fm_loss

        mean_log_p = total_log_p_accum / max(1, n_valid_terms)

        if isinstance(total_loss_accum, Tensor):
            if not torch.isfinite(total_loss_accum):
                log.warning("[ref_nll_ft] non-finite total loss; skipping backward")
                return {
                    "loss": total_loss_accum.detach(),
                    "train/ref_nll_log_p": mean_log_p,
                    "train/ref_nll_n_terms": float(n_valid_terms),
                }
            self.manual_backward(total_loss_accum)
            return {
                "loss": total_loss_accum.detach(),
                "train/ref_nll_log_p": mean_log_p,
                "train/ref_nll_n_terms": float(n_valid_terms),
            }
        else:
            # no valid terms — return zero loss
            dummy = torch.zeros(1, device=tokenized_agent["batch"].device, requires_grad=True)
            self.manual_backward(dummy * 0.0)
            return {
                "loss": torch.zeros(1),
                "train/ref_nll_log_p": 0.0,
                "train/ref_nll_n_terms": 0.0,
            }

    # ─────────────────────────────────────────────────────────────────────────
    # OCSC (Open-Closed Self-Consistency) fine-tuning
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_ocsc_train_hard_rmm(
        self,
        scenario_files: list,
        agent_ids: Tensor,
        agent_batch: Tensor,
        traj_list: list,   # G tensors of [n_agents, 80, 2]  ← 반드시 8초(80 step)
        z_list: list,      # G tensors of [n_agents, 80]
        head_list: list,   # G tensors of [n_agents, 80]
        metric: "HardSimAgentsMetrics | None" = None,
    ) -> float | None:
        """G개의 8초 detached 궤적으로 HardRMM(WOSAC official 기준)을 계산합니다.

        Args:
            metric: 사용할 HardSimAgentsMetrics 인스턴스.
                None 이면 self._ocsc_train_hard_rmm 을 사용합니다 (current model).

        Returns:
            float | None: 계산된 HardRMM 값. 실패 시 None.
        """
        _metric = metric if metric is not None else self._ocsc_train_hard_rmm
        if _metric is None or len(traj_list) == 0:
            return None
        try:
            # Stack into [n_agents, G, T, x]
            pred_traj = torch.stack(traj_list, dim=1)    # [n_agents, G, 80, 2]
            pred_z    = torch.stack(z_list, dim=1)       # [n_agents, G, 80]
            pred_head = torch.stack(head_list, dim=1)    # [n_agents, G, 80]

            with torch.no_grad():
                _metric.update_from_prediction_tensors(
                    scenario_files=list(scenario_files),
                    agent_id=agent_ids,
                    agent_batch=agent_batch,
                    pred_traj=pred_traj,
                    pred_z=pred_z,
                    pred_head=pred_head,
                )
            result_dict = _metric.compute()
            _metric.reset()
            # _metric_key는 prefix에서 파생되므로 current/ref 모두 정확히 일치
            key = getattr(_metric, "_metric_key", "train_ocsc/sim_agents_2025/realism_meta_metric")
            val = result_dict.get(key)
            if val is not None:
                return float(val.item() if isinstance(val, Tensor) else val)
        except Exception as exc:
            log.warning(f"[ocsc_ft] HardRMM computation failed: {exc}")
        return None

    def _run_flow_ocsc_ft_step(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        data: dict | None = None,
    ) -> dict:
        """Open-Closed Self-Consistency (OCSC) fine-tuning step.

        알고리즘:
          1. Open-loop target (no_grad, G번): flow ODE 직접 호출 (autoregressive 없음).
             rollout_cache["feat_a_now"][active_mask] 을 context 로 사용.
             → normalized local frame [n_active, 20, 4] (x/20, y/20, cos_h, sin_h)
          2. Closed-loop predictions (with grad, G번): autoregressive rollout.
             → world frame [n_agents, G, T, 2] → _world_traj_to_flow_norm 으로 변환
          3. Consistency loss: mean_g(L2(closed_norm_g[active], open_g.detach()))
          4. HardRMM 모니터링 (optional, configurable interval).

        BPTT tricks:
          - bptt_sequential_rollouts: G rollout 순차 backward (메모리 절감)
          - bptt_use_adjoint: ODE gradient checkpoint
          - bptt_warm_coarse_steps / bptt_last_n_coarse_steps: sliding-window BPTT
          - bptt_last_n_solver_steps: ODE solver 마지막 N step gradient
          - bptt_grad_clip_traj: closed-loop traj gradient L2 norm clip
        """
        G = int(getattr(self.finetune_config, "ocsc_n_rollouts", 2))
        # ocsc_n_ol_rollouts=-1 (default): M = G (paired L2/MMD).
        # ocsc_n_ol_rollouts=1: single OL sample broadcast → 모든 CL rollout 이 동일 OL 과 비교.
        # ocsc_n_ol_rollouts=M (>G) + ocsc_ol_nearest_match=True: 각 CL g 에 대해 M 개 OL 중
        #   per-anchor flat L2 거리 최소를 골라 paired L2 target 으로 사용.
        _g_ol_raw = int(getattr(self.finetune_config, "ocsc_n_ol_rollouts", -1))
        M_ol = G if _g_ol_raw <= 0 else max(1, _g_ol_raw)
        _nearest_match = bool(getattr(self.finetune_config, "ocsc_ol_nearest_match", False))
        if _nearest_match and M_ol < G:
            log.warning(
                f"[ocsc_ft] ocsc_ol_nearest_match=True 인데 M={M_ol} < G={G}: nearest_match 비활성."
            )
            _nearest_match = False
        # _shared_ol: M < G 일 때 모든 CL 이 ol_norms[0] 과 비교 (broadcast).
        # nearest_match 일 때는 M >= G 강제이므로 _shared_ol 은 항상 False.
        _shared_ol = (M_ol < G) and (not _nearest_match)
        # G_ol: backward-compatible alias for M_ol when M_ol <= G (legacy code paths).
        G_ol = min(M_ol, G)
        pred_max_steps_raw = int(getattr(self.finetune_config, "ocsc_pred_max_steps", 4))
        pred_max_steps: int | None = pred_max_steps_raw if pred_max_steps_raw > 0 else None
        loss_type = str(getattr(self.finetune_config, "ocsc_loss_type", "l2"))
        # ocsc_use_mmd=True: proper MMD² (self-term 포함, mode collapse 방지)
        # ocsc_use_mmd=False: 기존 paired L2 mean (비교/ablation 용)
        use_mmd = bool(getattr(self.finetune_config, "ocsc_use_mmd", True))
        if _shared_ol and use_mmd:
            log.warning(
                f"[ocsc_ft] M={M_ol} < G={G}: forcing use_mmd=False "
                f"(single OL sample → no distribution to match)"
            )
            use_mmd = False
        if _nearest_match and use_mmd:
            log.warning(
                f"[ocsc_ft] ocsc_ol_nearest_match=True: forcing use_mmd=False (paired L2 with argmin target)"
            )
            use_mmd = False
        # ocsc_gt_target=True: open-loop sample 대신 GT 궤적을 target으로 사용.
        # CL 예측을 2Hz로 다운샘플 후 GT(2Hz)와 비교.
        use_gt_target = bool(getattr(self.finetune_config, "ocsc_gt_target", False))
        # GT resolution: "2hz" (default, 기존) | "10hz" (raw fine 10Hz GT, no downsample).
        gt_resolution = str(getattr(self.finetune_config, "ocsc_gt_resolution", "2hz")).lower()
        if gt_resolution not in ("2hz", "10hz"):
            log.warning(
                f"[ocsc_ft] unknown ocsc_gt_resolution={gt_resolution!r}, falling back to '2hz'."
            )
            gt_resolution = "2hz"
        gt_is_10hz = (gt_resolution == "10hz")
        # nearest_match candidate pool 에 GT 1 개 포함 (always raw 10Hz GT).
        _nearest_include_gt = bool(getattr(self.finetune_config, "ocsc_nearest_include_gt", False))
        if _nearest_include_gt and not _nearest_match:
            log.warning(
                "[ocsc_ft] ocsc_nearest_include_gt=True 인데 nearest_match 비활성: include_gt 자동 비활성."
            )
            _nearest_include_gt = False
        if _nearest_include_gt and use_gt_target:
            log.warning(
                "[ocsc_ft] ocsc_nearest_include_gt=True + ocsc_gt_target=True 중복: "
                "use_gt_target 만 활성 (include_gt 무시)."
            )
            _nearest_include_gt = False
        heading_w = float(getattr(self.finetune_config, "ocsc_heading_weight", 0.0))
        pos_w = float(getattr(self.finetune_config, "ocsc_position_weight", 1.0))
        rel_disp_w = float(getattr(self.finetune_config, "ocsc_rel_disp_weight", 0.0))
        # GT FM regularization: MMD만 줄일 때 velocity_head가 GT에서 drift하는 것을 방지.
        # 각 anchor에서 active_hidden으로 GT 궤적에 대한 FM loss를 계산해 함께 backward.
        fm_reg_lambda = float(self.finetune_config.ocsc_fm_reg_lambda)
        sequential = bool(getattr(self.finetune_config, "bptt_sequential_rollouts", False))
        if _nearest_match and sequential:
            log.warning(
                "[ocsc_ft] nearest_match + sequential 조합은 미구현 — sequential=False 강제."
            )
            sequential = False
        use_adjoint = bool(getattr(self.finetune_config, "bptt_use_adjoint", False))
        warm_coarse = int(getattr(self.finetune_config, "bptt_warm_coarse_steps", 0))
        _last_coarse_only = bool(getattr(self.finetune_config, "bptt_last_coarse_only", False))
        if _last_coarse_only and pred_max_steps is not None and pred_max_steps > 1:
            warm_coarse = pred_max_steps - 1
        _grad_clip  = float(getattr(self.finetune_config, "bptt_grad_clip_traj", 1.0))
        _last_n_solver = int(getattr(self.finetune_config, "bptt_last_n_solver_steps", 0))
        _last_n_coarse = int(getattr(self.finetune_config, "bptt_last_n_coarse_steps", 0))
        eval_hard_rmm = bool(getattr(self.finetune_config, "ocsc_eval_hard_rmm", True))
        eval_hard_rmm_interval = max(1, int(getattr(self.finetune_config, "ocsc_eval_hard_rmm_interval", 1)))
        _shift = int(getattr(self.encoder.agent_encoder, "shift", 5))
        # consistency 구간을 실제 grad가 살아있는 10Hz suffix로 제한할지 여부.
        # bptt_last_coarse_only=true면 warm_coarse=pred_max_steps-1 이므로 마지막 coarse step만 남긴다.
        _consistency_tail_10hz_steps: int | None = None
        _consistency_tail_2hz_steps: int | None = None
        if _last_coarse_only and pred_max_steps is not None and pred_max_steps > 0:
            _grad_coarse = max(0, int(pred_max_steps) - int(warm_coarse))
            _grad_coarse = max(1, _grad_coarse)
            _consistency_tail_10hz_steps = _grad_coarse * _shift
            _consistency_tail_2hz_steps = _grad_coarse

        # ── 데이터 검증 ──────────────────────────────────────────────────────
        if data is None:
            raise ValueError("ocsc_ft requires `data` dict with scenario metadata.")
        if "tfrecord_path" not in data or "scenario_id" not in data:
            raise KeyError("ocsc_ft requires data['tfrecord_path'] and data['scenario_id'].")
        agent_ids = None
        try:
            agent_ids = data["agent"]["id"]
        except Exception:
            pass
        if agent_ids is None:
            try:
                agent_ids = data["id"]
            except Exception:
                pass
        if agent_ids is None:
            raise KeyError("ocsc_ft requires agent object ids: data['agent']['id'] (or data['id']).")
        if int(agent_ids.shape[0]) != int(tokenized_agent["batch"].shape[0]):
            raise ValueError("agent id count mismatch")

        tfrecord_paths = data["tfrecord_path"]
        agent_batch    = tokenized_agent["batch"]

        # ── 1. Encode map (no_grad; encoder frozen) ──────────────────────────
        with torch.no_grad():
            map_feature = self.encoder.encode_map(tokenized_map)

        # ── 2. Build rollout cache (no_grad) ─────────────────────────────────
        with torch.no_grad():
            rollout_cache = self.encoder.agent_encoder.prepare_inference_cache(
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
            )

        # ── Anchor index selection (strided, uniform coverage) ───────────────
        # anchor_idx=k: GT step k를 "현재 시점"으로 하고, 이후 pred_steps 예측.
        # stride=N → 매 N번째 step만 anchor로 사용 (메모리/품질 트레이드오프).
        # stride=1 이면 가능한 모든 위치, stride=4 이면 0,4,8,12 (14→4개).
        step_current_2hz = int(rollout_cache["valid_window"].shape[1])
        total_2hz_steps = int(tokenized_agent["gt_pos"].shape[1])
        pred_steps = pred_max_steps_raw if pred_max_steps_raw > 0 else 4
        anchor_stride = max(1, int(getattr(self.finetune_config, "ocsc_anchor_stride", 4)))

        valid_anchor_end = max(1, total_2hz_steps - pred_steps)
        all_anchor_indices = list(range(0, valid_anchor_end, anchor_stride))

        # bptt_last_n_coarse_steps → effective warm_coarse
        _n_coarse_pred = pred_max_steps_raw if pred_max_steps_raw > 0 else 16
        if _last_n_coarse > 0:
            _last_n_coarse = min(_last_n_coarse, _n_coarse_pred)
            warm_coarse = max(warm_coarse, _n_coarse_pred - _last_n_coarse)
            log.info(
                f"[ocsc_ft] bptt_last_n_coarse_steps={_last_n_coarse}: "
                f"effective warm_coarse={warm_coarse} "
                f"(gradient on last {_last_n_coarse}/{_n_coarse_pred} coarse steps)"
            )

        # ── norm-clip hook helper ─────────────────────────────────────────────
        def _make_norm_clip_hook(max_norm: float):
            def _hook(g: Tensor) -> Tensor:
                g = torch.nan_to_num(g, nan=0.0, posinf=max_norm, neginf=-max_norm)
                g_norm = g.norm()
                if g_norm > max_norm:
                    g = g * (max_norm / g_norm)
                return g
            return _hook

        # ── consistency loss in normalized 4-channel space ───────────────────
        # Both pred_norm and tgt_norm: [..., T, 4] = [x/20, y/20, cos_h, sin_h]
        def _consistency_loss(pred_norm: Tensor, tgt_norm: Tensor) -> Tensor:
            T = min(pred_norm.shape[-2], tgt_norm.shape[-2])
            p = pred_norm[..., :T, :]
            t = tgt_norm[..., :T, :].detach()
            if loss_type == "smooth_l1":
                pos_loss = F.smooth_l1_loss(p[..., :2], t[..., :2], reduction="mean")
            elif loss_type == "l1":
                pos_loss = F.l1_loss(p[..., :2], t[..., :2], reduction="mean")
            else:  # default: l2
                pos_loss = F.mse_loss(p[..., :2], t[..., :2], reduction="mean")
            total = pos_w * pos_loss
            if rel_disp_w > 0.0 and T >= 2:
                # 상대변위(delta x/y) 정렬: 절대 위치 복귀보다 이동 패턴 일치에 직접적인 신호.
                disp_p = p[..., 1:, :2] - p[..., :-1, :2]
                disp_t = t[..., 1:, :2] - t[..., :-1, :2]
                if loss_type == "smooth_l1":
                    rel_disp_loss = F.smooth_l1_loss(disp_p, disp_t, reduction="mean")
                elif loss_type == "l1":
                    rel_disp_loss = F.l1_loss(disp_p, disp_t, reduction="mean")
                else:
                    rel_disp_loss = F.mse_loss(disp_p, disp_t, reduction="mean")
                total = total + rel_disp_w * rel_disp_loss
            if heading_w > 0.0:
                head_loss = F.mse_loss(p[..., 2:], t[..., 2:], reduction="mean")
                total = total + heading_w * head_loss
            return total

        def _slice_consistency_suffix(x: Tensor) -> Tensor:
            if _consistency_tail_10hz_steps is None:
                return x
            _tail = max(1, min(int(_consistency_tail_10hz_steps), int(x.shape[-2])))
            return x[..., -_tail:, :]

        def _slice_consistency_suffix_2hz(x: Tensor) -> Tensor:
            """GT target mode 전용: 2Hz 텐서 [..., T, C]의 tail 슬라이스."""
            if _consistency_tail_2hz_steps is None:
                return x
            _tail = max(1, min(int(_consistency_tail_2hz_steps), int(x.shape[-2])))
            return x[..., -_tail:, :]

        def _slice_valid_suffix_2hz(x: Tensor) -> Tensor:
            """GT valid mask [n, T] (2D)의 tail 슬라이스."""
            if _consistency_tail_2hz_steps is None:
                return x
            _tail = max(1, min(int(_consistency_tail_2hz_steps), int(x.shape[-1])))
            return x[..., -_tail:]

        def _consistency_loss_gt(
            pred_norm: Tensor,  # [n, T, 4]  (2Hz CL, gradient 있음)
            tgt_norm: Tensor,   # [n, T, 4]  (2Hz GT, detached)
            tgt_valid: Tensor,  # [n, T]     (GT 유효 마스크)
        ) -> Tensor:
            """GT target 전용: 유효한 GT step만 사용하는 masked consistency loss."""
            T = min(pred_norm.shape[-2], tgt_norm.shape[-2])
            p = pred_norm[..., :T, :]
            t = tgt_norm[..., :T, :].detach()
            valid = tgt_valid[..., :T]            # [n, T]
            if not valid.any():
                return p.sum() * 0.0
            mask = valid.unsqueeze(-1).float()    # [n, T, 1]
            n_valid = mask.sum().clamp(min=1.0)
            if loss_type == "smooth_l1":
                pos_loss = (F.smooth_l1_loss(p[..., :2], t[..., :2], reduction="none") * mask).sum() / n_valid
            elif loss_type == "l1":
                pos_loss = (F.l1_loss(p[..., :2], t[..., :2], reduction="none") * mask).sum() / n_valid
            else:
                pos_loss = (F.mse_loss(p[..., :2], t[..., :2], reduction="none") * mask).sum() / n_valid
            total = pos_w * pos_loss
            if rel_disp_w > 0.0 and T >= 2:
                pair_valid = (valid[..., 1:] & valid[..., :-1]).unsqueeze(-1).float()  # [n, T-1, 1]
                n_pair = pair_valid.sum().clamp(min=1.0)
                disp_p = p[..., 1:, :2] - p[..., :-1, :2]
                disp_t = t[..., 1:, :2] - t[..., :-1, :2]
                if loss_type == "smooth_l1":
                    rd_loss = (F.smooth_l1_loss(disp_p, disp_t, reduction="none") * pair_valid).sum() / n_pair
                elif loss_type == "l1":
                    rd_loss = (F.l1_loss(disp_p, disp_t, reduction="none") * pair_valid).sum() / n_pair
                else:
                    rd_loss = (F.mse_loss(disp_p, disp_t, reduction="none") * pair_valid).sum() / n_pair
                total = total + rel_disp_w * rd_loss
            if heading_w > 0.0:
                head_loss = (F.mse_loss(p[..., 2:], t[..., 2:], reduction="none") * mask).sum() / n_valid
                total = total + heading_w * head_loss
            return total

        # ── world → normalized frame helper ──────────────────────────────────
        def _cl_to_norm(cl_xy: Tensor, cl_head: Tensor, current_pos_active: Tensor, current_head_active: Tensor) -> Tensor:
            return self._world_traj_to_flow_norm(
                pred_traj=cl_xy,
                pred_head=cl_head,
                current_pos=current_pos_active,
                current_head=current_head_active,
            )

        def _cl_downsample_to_2hz(cl_xy: Tensor, cl_head: Tensor, T_target: int) -> tuple[Tensor, Tensor]:
            """10Hz CL 예측을 2Hz로 다운샘플: 각 coarse step의 마지막 fine-step 위치를 사용."""
            cl_xy_2hz = cl_xy[:, _shift - 1 :: _shift, :][:, :T_target]     # [n, T_2hz, 2]
            cl_head_2hz = cl_head[:, _shift - 1 :: _shift][:, :T_target]    # [n, T_2hz]
            return cl_xy_2hz, cl_head_2hz

        # ── 3+4. Anchor-sequential loop: OL → CL → loss → backward → free ────
        # 한 anchor씩 처리 후 즉시 backward하여 모든 anchor의 캐시/OL/CL을 동시에
        # 메모리에 올리지 않는다. 피크 메모리 = O(G), anchor 수에 무관.
        _use_ref = (
            bool(getattr(self.finetune_config, "ocsc_use_pretrained_ref", False))
            and self.ref_flow_decoder is not None
        )
        _agent_enc = self.encoder.agent_encoder
        _orig_fd = _agent_enc.flow_decoder

        flow_ode = _agent_enc.flow_ode
        flow_ode.use_adjoint_for_bptt = use_adjoint
        flow_ode.last_n_grad_solver_steps = (
            min(_last_n_solver, flow_ode.solver_steps) if _last_n_solver > 0 else 0
        )
        if _last_n_solver > 0:
            log.info(
                f"[ocsc_ft] bptt_last_n_solver_steps={flow_ode.last_n_grad_solver_steps}/{flow_ode.solver_steps}: "
                f"velocity detach on first {flow_ode.solver_steps - flow_ode.last_n_grad_solver_steps} solver steps."
            )

        total_loss_accum = 0.0
        fm_reg_accum = 0.0
        n_valid_anchors = 0
        n_anchors_total = max(1, len(all_anchor_indices))
        _diag_pred_traj = None   # 마지막 anchor CL traj (variance 진단용)
        _diag_ol_norms: list[Tensor] = []
        _diag_active_mask: Tensor | None = None
        _seq_keys = {"gt_pos", "gt_heading", "valid_mask", "gt_idx"}

        try:
            for anchor_idx in all_anchor_indices:
                # ── 3a. Build anchor tokenized_agent (slice views, no copy) ──
                hist_start = max(0, anchor_idx + 1 - step_current_2hz)
                tokenized_agent_anchor: dict[str, Tensor] = {}
                for key, value in tokenized_agent.items():
                    if (
                        key in _seq_keys
                        and torch.is_tensor(value)
                        and value.dim() >= 2
                    ):
                        tokenized_agent_anchor[key] = value[:, hist_start : anchor_idx + 1]
                    else:
                        tokenized_agent_anchor[key] = value

                # ── 3b. Build anchor rollout cache ────────────────────────────
                with torch.no_grad():
                    rollout_cache_anchor = _agent_enc.prepare_inference_cache(
                        tokenized_agent=tokenized_agent_anchor,
                        map_feature=map_feature,
                    )
                active_mask = rollout_cache_anchor["valid_window"][:, -1]
                if not bool(active_mask.any()):
                    del rollout_cache_anchor
                    continue

                current_pos_active = rollout_cache_anchor["pos_window"][:, -1][active_mask]
                current_head_active = rollout_cache_anchor["head_window"][:, -1][active_mask]
                active_hidden = rollout_cache_anchor["feat_a_now"][active_mask]

                # ── 3c. Target samples: GT or Open-loop ───────────────────────
                _n_agent_full = int(tokenized_agent_anchor["batch"].shape[0])
                _n_step_10hz = int(rollout_cache_anchor["n_step_future_10hz"])
                _sample_win = 20
                _tape_steps = _n_step_10hz + _sample_win - _agent_enc.shift
                shared_tapes: list[Tensor] = []  # noise_tape_g per rollout [n_agent, tape_steps, 4]

                # GT target 모드: GT 궤적을 target으로 사용.
                # resolution 토글에 따라 2Hz (기존, tokenized_agent["gt_pos"]) 또는
                # 10Hz raw (data["agent"]["position"]) GT 점을 anchor frame으로 정규화.
                gt_norm_anchor: Tensor | None = None
                gt_valid_anchor: Tensor | None = None
                # nearest_include_gt 전용: candidate pool 추가 GT (항상 raw 10Hz).
                # use_gt_target=False 분기에서 활성 시 별도 set.
                gt_norm_anchor_inc: Tensor | None = None
                gt_valid_anchor_inc: Tensor | None = None
                if use_gt_target:
                    if gt_is_10hz:
                        # raw 10Hz GT: anchor (2Hz idx) 의 10Hz 시점 = (anchor+1)*shift,
                        # GT 는 그 직후 fine 점들 (anchor +0.1s 부터).
                        _T_gt = (pred_max_steps_raw if pred_max_steps_raw > 0 else 4) * _shift
                        _anchor_now_10hz = (anchor_idx + 1) * _shift
                        _gt_start = _anchor_now_10hz + 1
                        _gt_end = _gt_start + _T_gt
                        _gt_pos  = data["agent"]["position"][active_mask, _gt_start:_gt_end, :2]
                        _gt_head = data["agent"]["heading"][active_mask, _gt_start:_gt_end]
                        _gt_valid = data["agent"]["valid_mask"][active_mask, _gt_start:_gt_end]
                    else:
                        # 2Hz GT (기존): tokenized_agent["gt_pos"] 의 2Hz slice.
                        _T_gt = pred_max_steps_raw if pred_max_steps_raw > 0 else 4
                        _gt_start = anchor_idx + 1
                        _gt_end = _gt_start + _T_gt
                        _gt_pos  = tokenized_agent["gt_pos"][active_mask, _gt_start:_gt_end, :]     # [n_active, T_gt, 2]
                        _gt_head = tokenized_agent["gt_heading"][active_mask, _gt_start:_gt_end]     # [n_active, T_gt]
                        _gt_valid = tokenized_agent["valid_mask"][active_mask, _gt_start:_gt_end]    # [n_active, T_gt]
                    # 실제 사용 가능한 GT step 수 (시퀀스 끝에서 잘릴 수 있음)
                    _T_gt_actual = _gt_pos.shape[1]
                    if _T_gt_actual == 0 or not _gt_valid.any():
                        del rollout_cache_anchor
                        continue
                    gt_norm_anchor = _cl_to_norm(
                        _gt_pos, _gt_head, current_pos_active, current_head_active,
                    ).detach()   # [n_active, T_gt_actual, 4]
                    gt_valid_anchor = _gt_valid                                                   # [n_active, T_gt_actual]
                    ol_norms: list[Tensor] = []
                    # noise tape은 CL rollout을 위해 여전히 필요 (G개)
                    for g in range(G):
                        _seeds_g = self._get_closed_loop_scenario_seeds(
                            scenario_ids=data["scenario_id"],
                            rollout_idx=g,
                            device=active_hidden.device,
                        )
                        tape_g = _agent_enc._build_rollout_noise_tape(
                            num_agent=_n_agent_full,
                            tape_steps=_tape_steps,
                            device=active_hidden.device,
                            dtype=active_hidden.dtype,
                            sampling_noise=self.eval_sampling_noise,
                            scenario_sampling_seeds=_seeds_g,
                            agent_batch=tokenized_agent_anchor["batch"],
                            share_noise_across_time=False,
                        )
                        shared_tapes.append(tape_g)
                else:
                    # 기존 open-loop sample 생성
                    # g별 per-scenario seed로 전체 noise tape 생성 → OL과 CL이 같은 tape 공유.
                    # OL-g: tape_g[active_mask, :20, :]  (fine-step 별 independent, CL step-0과 동일)
                    # CL-g: tape_g 전체 (coarse step t에서 tape[t*shift : t*shift+20] 사용)
                    # → OL 2초 horizon 내 위치는 pairwise 매칭, 그 밖은 독립 random.
                    if _use_ref:
                        _agent_enc.flow_decoder = self.ref_flow_decoder
                    with torch.no_grad():
                        ol_norms: list[Tensor] = []
                        # shared_tapes 는 CL rollout 용으로 항상 G 개. ol_norms 는 M_ol 개 sample.
                        # M_ol < G (_shared_ol): G_ol=1 broadcast.
                        # M_ol = G: 기존 paired.
                        # M_ol > G + nearest_match: 추가 (M_ol - G) 개 OL-only sample 생성 (CL 과 noise 공유 안 함).
                        for g in range(G):
                            _seeds_g = self._get_closed_loop_scenario_seeds(
                                scenario_ids=data["scenario_id"],
                                rollout_idx=g,
                                device=active_hidden.device,
                            )
                            tape_g = _agent_enc._build_rollout_noise_tape(
                                num_agent=_n_agent_full,
                                tape_steps=_tape_steps,
                                device=active_hidden.device,
                                dtype=active_hidden.dtype,
                                sampling_noise=self.eval_sampling_noise,
                                scenario_sampling_seeds=_seeds_g,
                                agent_batch=tokenized_agent_anchor["batch"],
                                share_noise_across_time=False,
                            )  # [n_agent, tape_steps, 4]
                            shared_tapes.append(tape_g)
                            if g < G_ol:
                                x_init_ol = tape_g[active_mask, :_sample_win, :].clone()  # [n_active, 20, 4]
                                ol_norms.append(_agent_enc._sample_open_loop_future_from_hidden(
                                    anchor_hidden_valid=active_hidden,
                                    sampling_noise=self.eval_sampling_noise,
                                    x_init_override=x_init_ol,
                                ))
                        # OL-only extra samples (M_ol > G): CL 과 noise 공유 안 하고 별도 seed.
                        for m in range(G, M_ol):
                            _seeds_m = self._get_closed_loop_scenario_seeds(
                                scenario_ids=data["scenario_id"],
                                rollout_idx=m,
                                device=active_hidden.device,
                            )
                            tape_m = _agent_enc._build_rollout_noise_tape(
                                num_agent=_n_agent_full,
                                tape_steps=_tape_steps,
                                device=active_hidden.device,
                                dtype=active_hidden.dtype,
                                sampling_noise=self.eval_sampling_noise,
                                scenario_sampling_seeds=_seeds_m,
                                agent_batch=tokenized_agent_anchor["batch"],
                                share_noise_across_time=False,
                            )
                            x_init_ol = tape_m[active_mask, :_sample_win, :].clone()
                            ol_norms.append(_agent_enc._sample_open_loop_future_from_hidden(
                                anchor_hidden_valid=active_hidden,
                                sampling_noise=self.eval_sampling_noise,
                                x_init_override=x_init_ol,
                            ))
                            del tape_m
                    if _use_ref:
                        _agent_enc.flow_decoder = _orig_fd

                    # nearest_include_gt: candidate pool 에 raw 10Hz GT 1 개 추가.
                    if _nearest_include_gt:
                        _T_gt_inc = (pred_max_steps_raw if pred_max_steps_raw > 0 else 4) * _shift
                        _anchor_now_10hz_inc = (anchor_idx + 1) * _shift
                        _gt_start_inc = _anchor_now_10hz_inc + 1
                        _gt_end_inc = _gt_start_inc + _T_gt_inc
                        _gt_pos_inc  = data["agent"]["position"][active_mask, _gt_start_inc:_gt_end_inc, :2]
                        _gt_head_inc = data["agent"]["heading"][active_mask, _gt_start_inc:_gt_end_inc]
                        _gt_valid_inc = data["agent"]["valid_mask"][active_mask, _gt_start_inc:_gt_end_inc]
                        if _gt_pos_inc.shape[1] > 0 and bool(_gt_valid_inc.any()):
                            gt_norm_anchor_inc = _cl_to_norm(
                                _gt_pos_inc, _gt_head_inc, current_pos_active, current_head_active,
                            ).detach()
                            gt_valid_anchor_inc = _gt_valid_inc

                # ── 4. Closed-loop rollout + loss ─────────────────────────────
                _T_gt = int(gt_norm_anchor.shape[1]) if use_gt_target else 0

                if sequential and G > 1:
                    # 2-pass sequential: peak memory O(1 graph), MMD gradient = exact.
                    #
                    # Pass 1 (no_grad): G CL rollouts → detached cl_norms for kernel reference.
                    # Pass 2 (with_grad): re-run each rollout g, compute per-rollout MMD proxy,
                    #   call .backward() immediately, free graph. Memory stays O(1 graph).
                    #
                    # Gradient identity (why detaching cl_j≠g is safe):
                    #   ∂k(cl_g, detach(cl_j))/∂cl_g == ∂k(cl_g, cl_j)/∂cl_g
                    # so ∂proxy_g/∂θ == ∂MMD²/∂cl_g · ∂cl_g/∂θ  (exact contribution of rollout g).
                    # Summing over g: exact ∂MMD²/∂θ.
                    _do_seq_mmd = use_mmd and G >= 2
                    cl_norms_det: list[Tensor] = []
                    sigma_sq_seq: Tensor | None = None

                    if _do_seq_mmd:
                        # Pass 1 ─────────────────────────────────────────────
                        with torch.no_grad():
                            for g in range(G):
                                _traj_d, _, _head_d, _ = self._run_parallel_rollout_chunk(
                                    data=data,
                                    tokenized_agent=tokenized_agent_anchor,
                                    map_feature=map_feature,
                                    rollout_cache=rollout_cache_anchor,
                                    rollout_indices=[g],
                                    return_anchor_hidden=True,
                                    full_grad=True,   # no_grad 컨텍스트 안이라 gradient 없음; max_steps 인수 전달에 필요
                                    max_steps=pred_max_steps,
                                    warm_coarse_steps=warm_coarse,
                                    noise_tape_override=shared_tapes[g],
                                )
                                _T_d = _traj_d.shape[-2]
                                if use_gt_target:
                                    _xy_d, _hd_d = _cl_downsample_to_2hz(
                                        _traj_d[active_mask, 0, :_T_d, :],
                                        _head_d[active_mask, 0, :_T_d],
                                        _T_gt,
                                    )
                                    _cl_norm_det = _cl_to_norm(_xy_d, _hd_d, current_pos_active, current_head_active)
                                    _cl_norm_det = _slice_consistency_suffix_2hz(_cl_norm_det)
                                else:
                                    _cl_norm_det = _cl_to_norm(
                                        _traj_d[active_mask, 0, :_T_d, :],
                                        _head_d[active_mask, 0, :_T_d],
                                        current_pos_active, current_head_active,
                                    )
                                    _cl_norm_det = _slice_consistency_suffix(_cl_norm_det)
                                cl_norms_det.append(_cl_norm_det)
                                del _traj_d, _head_d

                        if use_gt_target:
                            _gt_slice = _slice_consistency_suffix_2hz(gt_norm_anchor)
                            _ol_ref_list = [_gt_slice] * G
                            sigma_sq_seq = mmd_precompute_sigma_sq(
                                _ol_ref_list, cl_norms_det,
                                pos_weight=pos_w, heading_weight=heading_w,
                            )
                            # Log detached MMD from pass-1 vs GT
                            _T_log = min(cl_norms_det[0].shape[-2], _gt_slice.shape[-2])
                            with torch.no_grad():
                                _gt_stack = _gt_slice.unsqueeze(0).expand(G, -1, -1, -1)[:, :, :_T_log, :]
                                _mmd_log = mmd_from_stacked(
                                    torch.stack(cl_norms_det, dim=0)[:, :, :_T_log, :],
                                    _gt_stack,
                                    pos_weight=pos_w, heading_weight=heading_w,
                                )
                            total_loss_accum += _mmd_log.item()
                            del _mmd_log, _gt_stack
                        else:
                            _ol_det = [_slice_consistency_suffix(o.detach()) for o in ol_norms]
                            _ol_ref_list = _ol_det
                            sigma_sq_seq = mmd_precompute_sigma_sq(
                                _ol_det, cl_norms_det,
                                pos_weight=pos_w, heading_weight=heading_w,
                            )
                            # Log MMD value from detached pass-1 samples (consistent with parallel mode)
                            _T_log = min(cl_norms_det[0].shape[-2], _ol_det[0].shape[-2])
                            with torch.no_grad():
                                _mmd_log = mmd_from_stacked(
                                    torch.stack(cl_norms_det, dim=0)[:, :, :_T_log, :],
                                    torch.stack(_ol_det,      dim=0)[:, :, :_T_log, :],
                                    pos_weight=pos_w, heading_weight=heading_w,
                                )
                            total_loss_accum += _mmd_log.item()
                            del _mmd_log

                    # Pass 2 ─────────────────────────────────────────────────
                    for g in range(G):
                        pred_traj_g, pred_z_g, pred_head_g, _ = self._run_parallel_rollout_chunk(
                            data=data,
                            tokenized_agent=tokenized_agent_anchor,
                            map_feature=map_feature,
                            rollout_cache=rollout_cache_anchor,
                            rollout_indices=[g],
                            return_anchor_hidden=True,
                            full_grad=True,
                            max_steps=pred_max_steps,
                            warm_coarse_steps=warm_coarse,
                            noise_tape_override=shared_tapes[g],
                        )
                        if pred_traj_g.requires_grad and _grad_clip > 0:
                            pred_traj_g.register_hook(_make_norm_clip_hook(_grad_clip))
                        T_cl = pred_traj_g.shape[-2]
                        cl_xy_g = pred_traj_g[active_mask, 0, :T_cl, :]
                        cl_head_g = pred_head_g[active_mask, 0, :T_cl]

                        if use_gt_target:
                            _xy_2hz, _hd_2hz = _cl_downsample_to_2hz(cl_xy_g, cl_head_g, _T_gt)
                            cl_norm_g = _cl_to_norm(_xy_2hz, _hd_2hz, current_pos_active, current_head_active)
                            cl_norm_g = _slice_consistency_suffix_2hz(cl_norm_g)
                            _gt_slice_pass2 = _slice_consistency_suffix_2hz(gt_norm_anchor)
                            _gt_valid_slice = _slice_valid_suffix_2hz(gt_valid_anchor)
                        else:
                            cl_norm_g = _cl_to_norm(cl_xy_g, cl_head_g, current_pos_active, current_head_active)
                            cl_norm_g = _slice_consistency_suffix(cl_norm_g)

                        if _do_seq_mmd:
                            # (proxy_g / n_anchors).backward() summed over g = ∂(mean_anchor MMD²)/∂θ
                            # GT target: ol_norms_ref = [gt_slice] * G → kco = k(cl_g, GT) exactly.
                            proxy_g = mmd_per_rollout_proxy(
                                cl_norm_g=cl_norm_g,
                                cl_norms_ref=cl_norms_det,
                                ol_norms_ref=_ol_ref_list,
                                sigma_sqs=sigma_sq_seq,
                                pos_weight=pos_w,
                                heading_weight=heading_w,
                            )
                            (proxy_g / n_anchors_total).backward()
                            del proxy_g
                        elif use_gt_target:
                            loss_g = _consistency_loss_gt(cl_norm_g, _gt_slice_pass2, _gt_valid_slice)
                            total_loss_accum += loss_g.item()
                            (loss_g / (n_anchors_total * G)).backward()
                            del loss_g
                        else:
                            _ol_idx = 0 if _shared_ol else g
                            loss_g = _consistency_loss(
                                cl_norm_g,
                                _slice_consistency_suffix(ol_norms[_ol_idx]),
                            )
                            total_loss_accum += loss_g.item()
                            (loss_g / (n_anchors_total * G)).backward()
                            del loss_g

                        del pred_traj_g, pred_z_g, pred_head_g, cl_xy_g, cl_head_g, cl_norm_g
                else:
                    # G rollout 병렬 (MMD 사용 가능)
                    pred_traj_all, pred_z_all, pred_head_all, _ = self._run_parallel_rollout_chunk(
                        data=data,
                        tokenized_agent=tokenized_agent_anchor,
                        map_feature=map_feature,
                        rollout_cache=rollout_cache_anchor,
                        rollout_indices=list(range(G)),
                        return_anchor_hidden=True,
                        full_grad=True,
                        max_steps=pred_max_steps,
                        warm_coarse_steps=warm_coarse,
                    )
                    if pred_traj_all.requires_grad and _grad_clip > 0:
                        pred_traj_all.register_hook(_make_norm_clip_hook(_grad_clip))
                    T_cl = pred_traj_all.shape[-2]
                    cl_norms: list[Tensor] = []
                    for g in range(G):
                        if use_gt_target and not gt_is_10hz:
                            # 2Hz GT mode: CL 을 2Hz 로 다운샘플
                            _xy_2hz, _hd_2hz = _cl_downsample_to_2hz(
                                pred_traj_all[active_mask, g, :T_cl, :],
                                pred_head_all[active_mask, g, :T_cl],
                                _T_gt,
                            )
                            cl_norms.append(_cl_to_norm(_xy_2hz, _hd_2hz, current_pos_active, current_head_active))
                        else:
                            # OL mode 또는 10Hz GT mode: CL 은 native fine 10Hz
                            cl_norms.append(_cl_to_norm(
                                pred_traj_all[active_mask, g, :T_cl, :],
                                pred_head_all[active_mask, g, :T_cl],
                                current_pos_active, current_head_active,
                            ))

                    if use_gt_target:
                        # GT mode: 2Hz 면 _slice_consistency_suffix_2hz, 10Hz 면 fine suffix.
                        _cl_slice_fn = _slice_consistency_suffix if gt_is_10hz else _slice_consistency_suffix_2hz
                        _gt_slice = _cl_slice_fn(gt_norm_anchor)
                        # valid mask: 10Hz mode 에선 _consistency_tail_10hz_steps 기준이 맞지만,
                        # last_coarse_only=False 인 경우 둘 다 None → identity. 가장 흔한 경로.
                        _gt_valid_slice = _slice_valid_suffix_2hz(gt_valid_anchor)
                        if use_mmd and G >= 2:
                            T_min = min(cl_norms[0].shape[-2], _gt_slice.shape[-2])
                            cl_stack = torch.stack(
                                [_cl_slice_fn(c) for c in cl_norms], dim=0
                            )[:, :, :T_min, :]
                            # GT를 G번 반복해 ol_stack으로 사용: koo=1 (constant, no grad)
                            gt_stack = _gt_slice.unsqueeze(0).expand(G, -1, -1, -1)[:, :, :T_min, :].detach()
                            anchor_loss = mmd_from_stacked(
                                cl_stack, gt_stack,
                                pos_weight=pos_w, heading_weight=heading_w,
                            )
                        else:
                            anchor_loss = torch.stack([
                                _consistency_loss_gt(
                                    _cl_slice_fn(cl_norms[g]),
                                    _gt_slice,
                                    _gt_valid_slice,
                                )
                                for g in range(G)
                            ]).mean()
                    elif use_mmd and G >= 2:
                        T_min = min(T_cl, ol_norms[0].shape[-2])
                        cl_stack = torch.stack(cl_norms, dim=0)[:, :, :T_min, :]
                        ol_stack = torch.stack(ol_norms, dim=0)[:, :, :T_min, :].detach()
                        cl_stack = _slice_consistency_suffix(cl_stack)
                        ol_stack = _slice_consistency_suffix(ol_stack)
                        anchor_loss = mmd_from_stacked(
                            cl_stack, ol_stack,
                            pos_weight=pos_w, heading_weight=heading_w,
                        )
                    else:
                        cl_sliced_pl = [_slice_consistency_suffix(cl_norms[g]) for g in range(G)]
                        if _nearest_match:
                            # 각 CL g 에 대해 M_ol 개 OL (+ optionally 1 GT) 중
                            # per-anchor flat L2 거리 최소를 선택.
                            ol_sliced_pl = [_slice_consistency_suffix(ol_norms[m]) for m in range(M_ol)]
                            T_min_nm = min(cl_sliced_pl[0].shape[-2], ol_sliced_pl[0].shape[-2])
                            _use_gt_cand = (
                                _nearest_include_gt
                                and gt_norm_anchor_inc is not None
                                and gt_valid_anchor_inc is not None
                            )
                            if _use_gt_cand:
                                gt_sliced_nm = _slice_consistency_suffix(gt_norm_anchor_inc)
                                T_min_nm = min(T_min_nm, gt_sliced_nm.shape[-2])
                            with torch.no_grad():
                                cl_stk_nm = torch.stack(
                                    [c[:, :T_min_nm, :] for c in cl_sliced_pl], dim=0
                                ).detach()  # [G, N_active, T, F]
                                ol_stk_nm = torch.stack(
                                    [o[:, :T_min_nm, :] for o in ol_sliced_pl], dim=0
                                )  # [M, N_active, T, F]
                                # [G, M] flat L2² distance
                                _d2_ol = ((cl_stk_nm.unsqueeze(1) - ol_stk_nm.unsqueeze(0)) ** 2).flatten(2).sum(-1)
                                if _use_gt_cand:
                                    gt_stk = gt_sliced_nm[:, :T_min_nm, :]  # [N_active, T, F]
                                    gt_mask = gt_valid_anchor_inc[:, :T_min_nm].float().unsqueeze(-1)  # [N, T, 1]
                                    # GT 거리: invalid step 의 squared diff 는 0 (mask), 전체 sum.
                                    _d2_gt = (
                                        (cl_stk_nm - gt_stk.unsqueeze(0)) ** 2
                                        * gt_mask.unsqueeze(0)
                                    ).flatten(1).sum(-1, keepdim=True)  # [G, 1]
                                    _d2_nm = torch.cat([_d2_ol, _d2_gt], dim=1)  # [G, M+1]
                                else:
                                    _d2_nm = _d2_ol
                                m_star_nm = _d2_nm.argmin(dim=1).tolist()
                            del cl_stk_nm, ol_stk_nm, _d2_nm
                            if _use_gt_cand:
                                losses_nm = []
                                gt_valid_for_loss = gt_valid_anchor_inc[:, :T_min_nm]
                                for g in range(G):
                                    _m_idx = int(m_star_nm[g])
                                    if _m_idx == M_ol:
                                        losses_nm.append(_consistency_loss_gt(
                                            cl_sliced_pl[g][..., :T_min_nm, :],
                                            gt_sliced_nm[..., :T_min_nm, :],
                                            gt_valid_for_loss,
                                        ))
                                    else:
                                        losses_nm.append(_consistency_loss(
                                            cl_sliced_pl[g], ol_sliced_pl[_m_idx],
                                        ))
                                anchor_loss = torch.stack(losses_nm).mean()
                            else:
                                anchor_loss = torch.stack([
                                    _consistency_loss(cl_sliced_pl[g], ol_sliced_pl[int(m_star_nm[g])])
                                    for g in range(G)
                                ]).mean()
                        else:
                            anchor_loss = torch.stack([
                                _consistency_loss(
                                    cl_sliced_pl[g],
                                    _slice_consistency_suffix(ol_norms[0 if _shared_ol else g]),
                                )
                                for g in range(G)
                            ]).mean()
                    total_loss_accum += anchor_loss.item()
                    (anchor_loss / n_anchors_total).backward()

                    # 진단용: 마지막 anchor의 CL/OL 보존 (variance logging; GT mode에서는 OL skip)
                    _diag_pred_traj = pred_traj_all.detach()
                    if not use_gt_target:
                        _diag_ol_norms = [o.detach() for o in ol_norms]
                    _diag_active_mask = active_mask

                    del pred_traj_all, pred_z_all, pred_head_all, cl_norms, anchor_loss

                n_valid_anchors += 1
                del rollout_cache_anchor, active_hidden, current_pos_active, current_head_active
                if not use_gt_target:
                    del ol_norms

        finally:
            flow_ode.use_adjoint_for_bptt = False
            flow_ode.last_n_grad_solver_steps = 0
            if _use_ref:
                _agent_enc.flow_decoder = _orig_fd

        # ── GT FM regularization (batch-level, anchor loop 이후) ─────────────
        # flow_decoder는 T=20을 기대하므로 flow_train_clean_norm(T=20)을 사용.
        # velocity_head가 GT에서 drift하지 않도록 consistency loss와 독립적으로 backward.
        if fm_reg_lambda > 0.0 and n_valid_anchors > 0:
            fm_val = self._compute_rmm_bptt_gt_fm_loss(map_feature, tokenized_agent)
            if fm_val is not None and torch.isfinite(fm_val):
                (fm_reg_lambda * fm_val).backward()
                fm_reg_accum = fm_val.item()

        if n_valid_anchors == 0:
            _ddp_dummy = sum(p.sum() * 0.0 for p in self.parameters() if p.requires_grad)
            return {"loss": _ddp_dummy}

        mean_loss = torch.tensor(
            total_loss_accum / n_valid_anchors,
            dtype=torch.float32, device=agent_batch.device,
        )
        log.info(
            f"[ocsc] step={int(getattr(self,'global_step',0))} "
            f"consistency_loss={mean_loss.item():.4f} n_anchors={n_valid_anchors}"
            + (f" fm_reg={fm_reg_accum:.4f}" if fm_reg_lambda > 0.0 else "")
        )
        ret = {
            "train/consistency_loss": mean_loss,
        }
        if fm_reg_lambda > 0.0:
            ret["train/fm_reg_loss"] = torch.tensor(
                fm_reg_accum, dtype=torch.float32, device=agent_batch.device,
            )

        # Mode collapse 진단 (마지막 anchor, no extra compute)
        if _diag_pred_traj is not None and _diag_pred_traj.shape[1] >= 2 and _diag_active_mask is not None:
            with torch.no_grad():
                _cl_var = _diag_pred_traj[_diag_active_mask].var(dim=1).mean()
                ret["train/traj_var_cl"] = _cl_var
            if len(_diag_ol_norms) >= 2:
                with torch.no_grad():
                    _ol_var = torch.stack(_diag_ol_norms, dim=0).var(dim=0).mean()
                    ret["train/traj_var_ol"] = _ol_var

        # ── 5. Hard RMM monitoring (WOSAC official 8초, no_grad, configurable interval) ──
        # 훈련 rollout 은 2초(max_steps=4)여서 WOSAC 기준에 맞지 않는다.
        # 별도로 no_grad full rollout(max_steps=None → 16 coarse step = 8초)을 수행한다.
        _global_step = int(getattr(self, "global_step", 0))
        if (
            eval_hard_rmm
            and self._ocsc_train_hard_rmm is not None
            and (_global_step % eval_hard_rmm_interval == 0)
        ):
            # ── 5a. Current model 8초 rollout ─────────────────────────────────
            with torch.no_grad():
                rmm_traj_all, rmm_z_all, rmm_head_all, _ = self._run_parallel_rollout_chunk(
                    data=data,
                    tokenized_agent=tokenized_agent,
                    map_feature=map_feature,
                    rollout_cache=rollout_cache,
                    rollout_indices=list(range(G)),
                    return_anchor_hidden=True,
                    full_grad=False,
                    max_steps=None,   # full 16 coarse step = 8초 = 80 timestep
                )
            # rmm_traj_all: [n_agents, G, 80, 2]
            _rmm_traj_list = [rmm_traj_all[:, g] for g in range(G)]
            _rmm_z_list    = [rmm_z_all[:, g]    for g in range(G)]
            _rmm_head_list = [rmm_head_all[:, g] for g in range(G)]

            hard_rmm_val = self._compute_ocsc_train_hard_rmm(
                scenario_files=list(tfrecord_paths),
                agent_ids=agent_ids,
                agent_batch=agent_batch,
                traj_list=_rmm_traj_list,
                z_list=_rmm_z_list,
                head_list=_rmm_head_list,
                metric=self._ocsc_train_hard_rmm,
            )
            if hard_rmm_val is not None:
                ret["train/hard_rmm"] = torch.tensor(
                    hard_rmm_val, dtype=torch.float32, device=agent_batch.device
                )
                log.info(f"[ocsc] step={_global_step} hard_rmm={hard_rmm_val:.4f}")

            # ── 5b. Reference model 8초 rollout (delta 계산) ──────────────────
            if self.ref_flow_decoder is not None and self._ocsc_train_hard_rmm_ref is not None:
                _agent_enc.flow_decoder = self.ref_flow_decoder
                try:
                    with torch.no_grad():
                        rmm_ref_traj, rmm_ref_z, rmm_ref_head, _ = self._run_parallel_rollout_chunk(
                            data=data,
                            tokenized_agent=tokenized_agent,
                            map_feature=map_feature,
                            rollout_cache=rollout_cache,
                            rollout_indices=list(range(G)),
                            return_anchor_hidden=True,
                            full_grad=False,
                            max_steps=None,   # 8초
                        )
                finally:
                    _agent_enc.flow_decoder = _orig_fd  # current model 복원

                _rmm_ref_traj_list = [rmm_ref_traj[:, g] for g in range(G)]
                _rmm_ref_z_list    = [rmm_ref_z[:, g]    for g in range(G)]
                _rmm_ref_head_list = [rmm_ref_head[:, g] for g in range(G)]

                hard_rmm_ref_val = self._compute_ocsc_train_hard_rmm(
                    scenario_files=list(tfrecord_paths),
                    agent_ids=agent_ids,
                    agent_batch=agent_batch,
                    traj_list=_rmm_ref_traj_list,
                    z_list=_rmm_ref_z_list,
                    head_list=_rmm_ref_head_list,
                    metric=self._ocsc_train_hard_rmm_ref,
                )
                if hard_rmm_ref_val is not None:
                    ret["train/hard_rmm_ref"] = torch.tensor(
                        hard_rmm_ref_val, dtype=torch.float32, device=agent_batch.device
                    )
                    if hard_rmm_val is not None:
                        delta = hard_rmm_val - hard_rmm_ref_val
                        ret["train/hard_rmm_delta"] = torch.tensor(
                            delta, dtype=torch.float32, device=agent_batch.device
                        )
                        log.info(
                            f"[ocsc] step={_global_step} "
                            f"hard_rmm={hard_rmm_val:.4f} ref={hard_rmm_ref_val:.4f} delta={delta:+.4f}"
                        )

        # DDP: 모든 trainable param을 dummy graph에 연결해 bucket reducer가 정상 작동하도록 함.
        # no_sync 컨텍스트 내에서 .backward()로 누적한 grad를 training_step에서
        # manual_backward(_ddp_dummy)로 최종 all-reduce 한 번에 동기화.
        _ddp_dummy = sum(p.sum() * 0.0 for p in self.parameters() if p.requires_grad)
        ret["loss"] = _ddp_dummy
        return ret

    def _run_flow_bptt_ft_step(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        data: dict | None = None,
    ) -> dict:
        """Flow-BPTT fine-tuning step.

        Closed-loop coarse rollout 후 soft RMM으로 미분 가능한 점수를 내고 gradient ascent 합니다.

        메모리 절감 옵션:
          - ``bptt_sequential_rollouts=True`` (기본): G rollout 을 1개씩 순차 실행 후 각각 즉시
            backward 해 computation graph 를 즉시 해제. 피크 메모리 ≈ G 배 절감.
          - ``bptt_warm_coarse_steps=N``: 앞 N coarse step 을 no_grad + detach (sliding BPTT).
          - ``bptt_use_adjoint=True``: ODE velocity head 내부 checkpoint (activation 절감).
          - ``bptt_max_coarse_steps=K``: coarse step 수 상한 (짧을수록 graph 작아짐).

        GT 정규화(선택):
          - ``flow_reg_lambda>0`` 이면 ``flow_train_clean_norm`` 기반 velocity FM MSE 를
            RMM loss 에 더합니다 (``kinematic_proj_ft`` 와 동일 키).
        """
        G = int(getattr(self.finetune_config, "bptt_n_rollouts", 1))
        _bmc_raw = getattr(self.finetune_config, "bptt_max_coarse_steps", None)
        if _bmc_raw is None:
            bptt_max_coarse_steps: int | None = None
        else:
            _n = int(_bmc_raw)
            bptt_max_coarse_steps = None if _n <= 0 else _n
        use_adjoint = bool(getattr(self.finetune_config, "bptt_use_adjoint", False))
        sequential = bool(getattr(self.finetune_config, "bptt_sequential_rollouts", True))
        warm_coarse = int(getattr(self.finetune_config, "bptt_warm_coarse_steps", 0))
        _grad_clip_traj = float(getattr(self.finetune_config, "bptt_grad_clip_traj", 1.0))
        _dbg_enabled = bool(getattr(self.finetune_config, "bptt_debug", False))
        _flow_reg_lambda = float(getattr(self.finetune_config, "flow_reg_lambda", 0.0))
        _last_n_coarse = int(getattr(self.finetune_config, "bptt_last_n_coarse_steps", 0))
        _last_n_solver = int(getattr(self.finetune_config, "bptt_last_n_solver_steps", 0))

        # ── 1. Encode map (no_grad; encoder frozen) ─────────────────────────
        with torch.no_grad():
            map_feature = self.encoder.encode_map(tokenized_map)

        # ── 2. Build rollout cache (no_grad; cache 초기화는 gradient 불필요) ─
        with torch.no_grad():
            rollout_cache = self.encoder.agent_encoder.prepare_inference_cache(
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
            )

        agent_batch = tokenized_agent["batch"]   # [n_agents] scenario index

        # ── 3. 데이터 검증 ──────────────────────────────────────────────────
        if data is None:
            raise ValueError("flow_bptt_ft requires `data` dict with scenario metadata.")
        if "tfrecord_path" not in data or "scenario_id" not in data:
            raise KeyError("flow_bptt_ft requires data['tfrecord_path'] and data['scenario_id'].")
        # IMPORTANT: soft RMM 의 object_id 는 rollout agent 순서와 정확히 일치해야 합니다.
        agent_ids = None
        try:
            agent_ids = data["agent"]["id"]
        except Exception:
            agent_ids = None
        if agent_ids is None:
            try:
                agent_ids = data["id"]
            except Exception:
                agent_ids = None
        if agent_ids is None:
            raise KeyError(
                "flow_bptt_ft requires exact agent object ids: data['agent']['id'] (or data['id'])."
            )
        if int(agent_ids.shape[0]) != int(tokenized_agent["batch"].shape[0]):
            raise ValueError(
                "agent id count mismatch: "
                f"ids={int(agent_ids.shape[0])} vs rollout_agents={int(tokenized_agent['batch'].shape[0])}"
            )

        scenario_ids: Sequence[str] = data["scenario_id"]
        tfrecord_paths: Sequence[str] = data["tfrecord_path"]
        n_scenarios = len(scenario_ids)
        if len(tfrecord_paths) != n_scenarios:
            raise ValueError(
                f"tfrecord_path length {len(tfrecord_paths)} != scenario_id length {n_scenarios}"
            )

        # ── Waymo metric config (once per process) ───────────────────────────
        if not hasattr(self, "_soft_rmm_waymo_cfg") or self._soft_rmm_waymo_cfg is None:
            from google.protobuf import text_format
            import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as wm_metrics
            from pathlib import Path as _Path

            config_path = _Path(wm_metrics.__file__).parent / "challenge_2025_sim_agents_config.textproto"
            cfg = sim_agents_metrics_pb2.SimAgentMetricsConfig()
            with open(config_path) as f:
                text_format.Parse(f.read(), cfg)
            self._soft_rmm_waymo_cfg = cfg
        cfg = self._soft_rmm_waymo_cfg

        from waymo_open_dataset.utils.sim_agents import submission_specs
        from src.smart.metrics.wosac_metric_features_torch.metric_features_torch import (
            compute_metric_features,
            scenario_to_joint_scene,
            _cache_get_or_build,
        )
        import tensorflow as tf

        _sim_agents_challenge = submission_specs.ChallengeType.SIM_AGENTS

        # ── 4. TFRecord & log-feature 사전 로드 (모든 rollout 에서 공유) ─────
        # 예측 지평선 T_hor 사전 계산 (coarse_steps × shift = 10Hz steps)
        _shift = int(getattr(self.encoder.agent_encoder, "shift", 5))
        _n_coarse = bptt_max_coarse_steps if bptt_max_coarse_steps is not None else 16
        _t_hor_pre = _n_coarse * _shift

        # bptt_last_n_coarse_steps: "마지막 N coarse step에만 gradient" 편의 파라미터.
        # warm_coarse 를 max(warm_coarse, n_coarse - last_n) 으로 override 해
        # 앞 구간을 no_grad + detach 처리합니다.
        if _last_n_coarse > 0:
            _last_n_coarse = min(_last_n_coarse, _n_coarse)
            warm_coarse = max(warm_coarse, _n_coarse - _last_n_coarse)
            log.info(
                f"[rmm_bptt] bptt_last_n_coarse_steps={_last_n_coarse}: "
                f"effective warm_coarse_steps={warm_coarse} "
                f"(gradient on last {_last_n_coarse}/{_n_coarse} coarse steps)"
            )

        _sc_scenarios: list = []
        _sc_log_feat_dicts: list = []
        _sc_masks: list = []
        _sc_agent_ids: list = []
        for sc_idx in range(n_scenarios):
            sc_mask = (agent_batch == sc_idx)
            _sc_masks.append(sc_mask)
            if not bool(sc_mask.any()):
                _sc_scenarios.append(None)
                _sc_log_feat_dicts.append(None)
                _sc_agent_ids.append(None)
                continue

            batch_scenario_id = str(scenario_ids[sc_idx])

            # ── scenario proto cache ──────────────────────────────────────────
            scenario = _SCENARIO_PROTO_CACHE.get(batch_scenario_id)
            if scenario is None:
                scenario = scenario_pb2.Scenario()
                for tfdata in tf.data.TFRecordDataset([tfrecord_paths[sc_idx]], compression_type=""):
                    scenario.ParseFromString(bytes(tfdata.numpy()))
                    break
                tfrecord_scenario_id = str(getattr(scenario, "scenario_id", ""))
                if tfrecord_scenario_id != batch_scenario_id:
                    raise ValueError(
                        "scenario_id mismatch between dataloader metadata and TFRecord content: "
                        f"batch='{batch_scenario_id}' vs tfrecord='{tfrecord_scenario_id}'. "
                        f"path='{tfrecord_paths[sc_idx]}'"
                    )
                _SCENARIO_PROTO_CACHE[batch_scenario_id] = scenario
                if len(_SCENARIO_PROTO_CACHE) > _SCENARIO_PROTO_CACHE_MAX:
                    _SCENARIO_PROTO_CACHE.pop(next(iter(_SCENARIO_PROTO_CACHE)))

            # ── log_feat_dict cache ───────────────────────────────────────────
            _lf_key = f"{batch_scenario_id}_{_t_hor_pre}"
            _lf_cpu = _LOG_FEAT_DICT_CACHE.get(_lf_key)
            if _lf_cpu is None:
                with torch.no_grad():
                    log_joint = scenario_to_joint_scene(scenario, _sim_agents_challenge)
                    log_feat = compute_metric_features(
                        scenario, log_joint,
                        challenge_type=_sim_agents_challenge, use_log_validity=True,
                    )
                    log_feat_dict = {
                        k: v.to(device=agent_batch.device)
                        for k, v in log_feat.as_dict().items()
                    }
                    log_feat_dict = _slice_log_feat_dict_to_pred_horizon(log_feat_dict, _t_hor_pre)
                _lf_cpu = {k: v.cpu() for k, v in log_feat_dict.items()}
                _LOG_FEAT_DICT_CACHE[_lf_key] = _lf_cpu
                if len(_LOG_FEAT_DICT_CACHE) > _LOG_FEAT_DICT_CACHE_MAX:
                    _LOG_FEAT_DICT_CACHE.pop(next(iter(_LOG_FEAT_DICT_CACHE)))
            else:
                log_feat_dict = {k: v.to(device=agent_batch.device) for k, v in _lf_cpu.items()}

            # ── static scenario cache (road edges, lanes, logged traj) ────────
            _cache_get_or_build(scenario)
            _sc_scenarios.append(scenario)
            _sc_log_feat_dicts.append(log_feat_dict)
            _sc_agent_ids.append(agent_ids[sc_mask])

        # ── Pre-compute valid scenario indices (shared by 5a and 5b) ────────
        _valid_sc_idx = [i for i in range(n_scenarios) if _sc_scenarios[i] is not None]
        _n_valid = len(_valid_sc_idx)
        _valid_scenarios    = [_sc_scenarios[i]     for i in _valid_sc_idx]
        _valid_agent_ids    = [_sc_agent_ids[i]     for i in _valid_sc_idx]
        _valid_log_feat_dicts = [_sc_log_feat_dicts[i] for i in _valid_sc_idx]
        _valid_sc_masks     = [_sc_masks[i]         for i in _valid_sc_idx]

        _SURROGATE = SurrogateConfig(
            collision_temperature=0.15,
            offroad_temperature=0.15,
            red_light_crossing_temperature=0.05,
        )

        # ── norm-clip hook helper ────────────────────────────────────────────
        def _make_norm_clip_hook(max_norm: float):
            def _hook(g: Tensor) -> Tensor:
                g = torch.nan_to_num(g, nan=0.0, posinf=max_norm, neginf=-max_norm)
                g_norm = g.norm()
                if g_norm > max_norm:
                    g = g * (max_norm / g_norm)
                return g
            return _hook

        flow_ode = self.encoder.agent_encoder.flow_ode

        # ── 5a. 순차 rollout 모드 (G > 1 이고 sequential=True) ───────────────
        # 한 번에 1개 rollout 만 그래프를 가지므로 피크 메모리 ≈ G 배 절감.
        # Lightning 에 반환하는 dummy loss (=0) 에 대한 backward 는 grad=0 이므로 무해.
        if sequential and G > 1:
            total_rmm_accum = 0.0
            total_count_accum = 0

            flow_ode.use_adjoint_for_bptt = use_adjoint
            flow_ode.last_n_grad_solver_steps = min(_last_n_solver, flow_ode.solver_steps) if _last_n_solver > 0 else 0
            if _last_n_solver > 0:
                log.info(f"[rmm_bptt] bptt_last_n_solver_steps={flow_ode.last_n_grad_solver_steps}/{flow_ode.solver_steps}: "
                         f"velocity detach on first {flow_ode.solver_steps - flow_ode.last_n_grad_solver_steps} solver steps.")
            try:
                for g in range(G):
                    pred_traj_g, pred_z_g, pred_head_g, _ = self._run_parallel_rollout_chunk(
                        data=data,
                        tokenized_agent=tokenized_agent,
                        map_feature=map_feature,
                        rollout_cache=rollout_cache,
                        rollout_indices=[g],
                        return_anchor_hidden=True,
                        full_grad=True,
                        max_steps=bptt_max_coarse_steps,
                        warm_coarse_steps=warm_coarse,
                    )
                    # pred_traj_g: [n_agents, 1, T, 2]

                    if pred_traj_g.requires_grad and _grad_clip_traj > 0:
                        pred_traj_g.register_hook(_make_norm_clip_hook(_grad_clip_traj))
                    if pred_head_g.requires_grad and _grad_clip_traj > 0:
                        pred_head_g.register_hook(_make_norm_clip_hook(_grad_clip_traj))

                    if _n_valid > 0:
                        # Batched soft-RMM across all valid scenes for this rollout
                        _preds_g = [
                            PredictedSimTrajectories(
                                object_id=_valid_agent_ids[j].cpu(),
                                center_x=pred_traj_g[_valid_sc_masks[j], 0, :, 0],
                                center_y=pred_traj_g[_valid_sc_masks[j], 0, :, 1],
                                center_z=pred_z_g[_valid_sc_masks[j], 0, :],
                                heading=pred_head_g[_valid_sc_masks[j], 0, :],
                                valid=pred_traj_g.new_ones(
                                    int(_valid_sc_masks[j].sum()), pred_traj_g.shape[2],
                                    dtype=torch.bool,
                                ),
                            )
                            for j in range(_n_valid)
                        ]
                        _feat_list = compute_metric_features_batched_scenes(
                            scenarios=_valid_scenarios, preds=_preds_g, surrogate=_SURROGATE,
                        )
                        _rmm_g_vec = compute_wosac_metametric_soft_batched(
                            config=cfg,
                            log_features_list=_valid_log_feat_dicts,
                            sim_features_list=[f.as_dict() for f in _feat_list],
                            debug=(_dbg_enabled and g == 0),
                        )  # (_n_valid,)
                        _finite_g = torch.isfinite(_rmm_g_vec)
                        count_g = int(_finite_g.sum().item())
                        if count_g > 0:
                            _safe_g = torch.where(_finite_g, _rmm_g_vec, torch.zeros_like(_rmm_g_vec))
                            rmm_g = _safe_g.sum() / float(count_g)
                        else:
                            count_g = 0
                    else:
                        count_g = 0

                    if count_g > 0:
                        rmm_g_val = rmm_g.detach().item()
                        total_rmm_accum += rmm_g_val
                        total_count_accum += 1
                        if math.isfinite(rmm_g_val):
                            partial_loss = -rmm_g / G
                            partial_loss.backward()
                        else:
                            log.warning(f"[rmm_bptt_ft] non-finite rmm at g={g}, skipping backward.")

                    del pred_traj_g, pred_z_g, pred_head_g
            finally:
                flow_ode.use_adjoint_for_bptt = False
                flow_ode.last_n_grad_solver_steps = 0

            if total_count_accum == 0:
                return {"loss": sum(p.sum() * 0.0 for p in self.parameters() if p.requires_grad)}

            mean_rmm = torch.tensor(
                total_rmm_accum / total_count_accum,
                dtype=torch.float32, device=agent_batch.device,
            )
            _step = int(getattr(self, "global_step", 0))
            # EMA for monitoring only (not used in loss)
            _ema_mom = 0.98
            if hasattr(self, "_rmm_ema_mean"):
                if not self._rmm_ema_initialized:
                    self._rmm_ema_mean.fill_(mean_rmm.item())
                    self._rmm_ema_initialized = True
                else:
                    self._rmm_ema_mean = _ema_mom * self._rmm_ema_mean + (1 - _ema_mom) * mean_rmm.detach()
            _ema_log = self._rmm_ema_mean.detach() if hasattr(self, "_rmm_ema_mean") else mean_rmm.detach()
            log.info(f"[rmm] step={_step} rmm_soft={mean_rmm.item():.4f} ema={_ema_log.item():.4f}")

            fm_bc_det: Tensor | None = None
            if _flow_reg_lambda > 0:
                _fm = self._compute_rmm_bptt_gt_fm_loss(map_feature, tokenized_agent)
                if _fm is not None:
                    (_flow_reg_lambda * _fm).backward()
                    fm_bc_det = _fm.detach()

            # sequential 은 loss 가 dummy(0) 이라 train/loss 로그가 의미 없음. 모니터링용 합산 스칼라.
            train_combined = (-mean_rmm).detach()
            if fm_bc_det is not None:
                train_combined = train_combined + _flow_reg_lambda * fm_bc_det

            # DDP: 모든 trainable param 을 dummy loss graph 에 연결해야 bucket reducer 가 정상 작동.
            # manual backward 로 grad 는 이미 누적됐으므로 backward() 추가 기여는 0.
            _ddp_dummy = sum(p.sum() * 0.0 for p in self.parameters() if p.requires_grad)
            seq_ret = {
                "loss": _ddp_dummy,   # grads already accumulated via manual backward
                "train/rmm_soft": mean_rmm,
                "train/rmm_loss": -mean_rmm,
                "train/combined_loss": train_combined,
                "train/rmm_ema": _ema_log,
                "train/rmm_n_scenarios": torch.tensor(float(total_count_accum), device=mean_rmm.device),
            }
            if fm_bc_det is not None:
                seq_ret["train/fm_bc_loss"] = fm_bc_det
            # sequential mode: ref soft RMM (train Δ 모니터링)
            if self._ref_train_enabled and self.ref_flow_decoder is not None:
                _orig_fd = self.encoder.agent_encoder.flow_decoder
                self.encoder.agent_encoder.flow_decoder = self.ref_flow_decoder
                try:
                    with torch.no_grad():
                        ref_traj_s, ref_z_s, ref_head_s, _ = self._run_parallel_rollout_chunk(
                            data=data,
                            tokenized_agent=tokenized_agent,
                            map_feature=map_feature,
                            rollout_cache=rollout_cache,
                            rollout_indices=list(range(G)),
                            return_anchor_hidden=True,
                            full_grad=True,   # no_grad context 내에서 max_steps 적용
                            max_steps=bptt_max_coarse_steps,
                            warm_coarse_steps=warm_coarse,
                        )
                finally:
                    self.encoder.agent_encoder.flow_decoder = _orig_fd
                if _n_valid > 0:
                    ref_s_total = 0.0
                    with torch.no_grad():
                        for g in range(G):
                            _ref_preds_g = [
                                PredictedSimTrajectories(
                                    object_id=_valid_agent_ids[j].cpu(),
                                    center_x=ref_traj_s[_valid_sc_masks[j], g, :, 0],
                                    center_y=ref_traj_s[_valid_sc_masks[j], g, :, 1],
                                    center_z=ref_z_s[_valid_sc_masks[j], g, :],
                                    heading=ref_head_s[_valid_sc_masks[j], g, :],
                                    valid=ref_traj_s.new_ones(
                                        int(_valid_sc_masks[j].sum()), ref_traj_s.shape[2],
                                        dtype=torch.bool,
                                    ),
                                )
                                for j in range(_n_valid)
                            ]
                            _ref_feat_list = compute_metric_features_batched_scenes(
                                scenarios=_valid_scenarios, preds=_ref_preds_g, surrogate=_SURROGATE,
                            )
                            _ref_rmm_vec = compute_wosac_metametric_soft_batched(
                                config=cfg,
                                log_features_list=_valid_log_feat_dicts,
                                sim_features_list=[f.as_dict() for f in _ref_feat_list],
                            )
                            ref_s_total += _ref_rmm_vec.sum().item() / (G * _n_valid)
                    seq_ref_rmm = torch.tensor(ref_s_total, dtype=torch.float32, device=mean_rmm.device)
                    seq_ret["train/rmm_ref"] = seq_ref_rmm
                    seq_ret["train/rmm_delta"] = mean_rmm.detach() - seq_ref_rmm
            return seq_ret

        # ── 5b. 병렬 rollout 모드 (G=1 또는 sequential=False) ────────────────
        flow_ode.use_adjoint_for_bptt = use_adjoint
        flow_ode.last_n_grad_solver_steps = min(_last_n_solver, flow_ode.solver_steps) if _last_n_solver > 0 else 0
        if _last_n_solver > 0:
            log.info(f"[rmm_bptt] bptt_last_n_solver_steps={flow_ode.last_n_grad_solver_steps}/{flow_ode.solver_steps}: "
                     f"velocity detach on first {flow_ode.solver_steps - flow_ode.last_n_grad_solver_steps} solver steps.")
        try:
            pred_traj, pred_z, pred_head_traj, _ = (
                self._run_parallel_rollout_chunk(
                    data=data,
                    tokenized_agent=tokenized_agent,
                    map_feature=map_feature,
                    rollout_cache=rollout_cache,
                    rollout_indices=list(range(G)),
                    return_anchor_hidden=True,
                    full_grad=True,
                    max_steps=bptt_max_coarse_steps,
                    warm_coarse_steps=warm_coarse,
                )
            )
        finally:
            flow_ode.use_adjoint_for_bptt = False
            flow_ode.last_n_grad_solver_steps = 0

        if pred_traj.requires_grad and _grad_clip_traj > 0:
            pred_traj.register_hook(_make_norm_clip_hook(_grad_clip_traj))
        if pred_head_traj.requires_grad and _grad_clip_traj > 0:
            pred_head_traj.register_hook(_make_norm_clip_hook(_grad_clip_traj))

        if _n_valid == 0:
            return {"loss": sum(p.sum() * 0.0 for p in self.parameters() if p.requires_grad)}

        # Batched soft-RMM: compute DNO+TTC once per rollout across all valid scenes
        _rmm_by_g = []  # [G] each (n_valid,)
        for g in range(G):
            _preds_g = [
                PredictedSimTrajectories(
                    object_id=_valid_agent_ids[j].cpu(),
                    center_x=pred_traj[_valid_sc_masks[j], g, :, 0],
                    center_y=pred_traj[_valid_sc_masks[j], g, :, 1],
                    center_z=pred_z[_valid_sc_masks[j], g, :],
                    heading=pred_head_traj[_valid_sc_masks[j], g, :],
                    valid=pred_traj.new_ones(
                        int(_valid_sc_masks[j].sum()), pred_traj.shape[2],
                        dtype=torch.bool,
                    ),
                )
                for j in range(_n_valid)
            ]
            _feat_list = compute_metric_features_batched_scenes(
                scenarios=_valid_scenarios, preds=_preds_g, surrogate=_SURROGATE,
            )
            _rmm_g = compute_wosac_metametric_soft_batched(
                config=cfg,
                log_features_list=_valid_log_feat_dicts,
                sim_features_list=[f.as_dict() for f in _feat_list],
                debug=(_dbg_enabled and g == 0),
            )  # (_n_valid,)
            _rmm_by_g.append(_rmm_g)

        rmm_matrix = torch.stack(_rmm_by_g, dim=1)  # (_n_valid, G)
        rmm_per_scene = rmm_matrix.mean(dim=1)       # (_n_valid,) — mean over rollouts

        finite_mask = torch.isfinite(rmm_per_scene)
        total_count = int(finite_mask.sum().item())

        if total_count == 0:
            return {"loss": sum(p.sum() * 0.0 for p in self.parameters() if p.requires_grad)}

        safe_rmm_per_scene = torch.where(finite_mask, rmm_per_scene, torch.zeros_like(rmm_per_scene))
        mean_rmm = safe_rmm_per_scene.sum() / float(total_count)

        _step = int(getattr(self, "global_step", 0))
        # EMA for monitoring only (not used in loss)
        _ema_mom = 0.98
        if hasattr(self, "_rmm_ema_mean"):
            if not self._rmm_ema_initialized:
                self._rmm_ema_mean.fill_(mean_rmm.item())
                self._rmm_ema_initialized = True
            else:
                self._rmm_ema_mean = _ema_mom * self._rmm_ema_mean + (1 - _ema_mom) * mean_rmm.detach()
        _ema_log = self._rmm_ema_mean.detach() if hasattr(self, "_rmm_ema_mean") else mean_rmm.detach()
        log.info(f"[rmm] step={_step} rmm_soft={mean_rmm.item():.4f} ema={_ema_log.item():.4f}")

        # ── pretrained ref soft RMM (train 단계 Δ 모니터링) ────────────────
        # finetuned 와 동일한 G개 rollout (no_grad → gradient graph 에 영향 없음).
        # rollout_indices=list(range(G)) → 동일 hash seed → noise 완전 정합.
        ref_rmm_log: Tensor | None = None
        if self._ref_train_enabled and self.ref_flow_decoder is not None:
            _orig_fd = self.encoder.agent_encoder.flow_decoder
            self.encoder.agent_encoder.flow_decoder = self.ref_flow_decoder
            try:
                with torch.no_grad():
                    # full_grad=True: rollout_from_cache_no_grad は max_steps を
                    # 受け付けないため常に 80 step を返す → log_feat_dict の長さ不一致.
                    # torch.no_grad() 内なので gradient は生成されず安全.
                    # warm_coarse_steps も finetuned と揃えて完全同条件にする.
                    ref_traj, ref_z, ref_head, _ = self._run_parallel_rollout_chunk(
                        data=data,
                        tokenized_agent=tokenized_agent,
                        map_feature=map_feature,
                        rollout_cache=rollout_cache,
                        rollout_indices=list(range(G)),
                        return_anchor_hidden=True,
                        full_grad=True,
                        max_steps=bptt_max_coarse_steps,
                        warm_coarse_steps=warm_coarse,
                    )
            finally:
                self.encoder.agent_encoder.flow_decoder = _orig_fd

            if _n_valid > 0:
                ref_total = 0.0
                with torch.no_grad():
                    for g in range(G):
                        _ref_preds_g = [
                            PredictedSimTrajectories(
                                object_id=_valid_agent_ids[j].cpu(),
                                center_x=ref_traj[_valid_sc_masks[j], g, :, 0],
                                center_y=ref_traj[_valid_sc_masks[j], g, :, 1],
                                center_z=ref_z[_valid_sc_masks[j], g, :],
                                heading=ref_head[_valid_sc_masks[j], g, :],
                                valid=ref_traj.new_ones(
                                    int(_valid_sc_masks[j].sum()), ref_traj.shape[2],
                                    dtype=torch.bool,
                                ),
                            )
                            for j in range(_n_valid)
                        ]
                        _ref_feat_list = compute_metric_features_batched_scenes(
                            scenarios=_valid_scenarios, preds=_ref_preds_g, surrogate=_SURROGATE,
                        )
                        _ref_rmm_vec = compute_wosac_metametric_soft_batched(
                            config=cfg,
                            log_features_list=_valid_log_feat_dicts,
                            sim_features_list=[f.as_dict() for f in _ref_feat_list],
                        )
                        ref_total += _ref_rmm_vec.sum().item() / (G * _n_valid)
                ref_rmm_log = torch.tensor(ref_total, dtype=torch.float32, device=mean_rmm.device)

        loss = -mean_rmm
        fm_bc_det: Tensor | None = None
        if _flow_reg_lambda > 0:
            _fm = self._compute_rmm_bptt_gt_fm_loss(map_feature, tokenized_agent)
            if _fm is not None:
                loss = loss + _flow_reg_lambda * _fm
                fm_bc_det = _fm.detach()

        ret = {
            "loss": loss,
            "train/rmm_soft": mean_rmm.detach(),
            "train/rmm_loss": (-mean_rmm).detach(),
            "train/combined_loss": loss.detach(),
            "train/rmm_ema": _ema_log,
            "train/rmm_n_scenarios": torch.tensor(float(total_count), device=mean_rmm.device),
        }
        if fm_bc_det is not None:
            ret["train/fm_bc_loss"] = fm_bc_det
        if ref_rmm_log is not None:
            ret["train/rmm_ref"] = ref_rmm_log
            ret["train/rmm_delta"] = mean_rmm.detach() - ref_rmm_log
        return ret

    def _run_kinematic_reward_ft_step(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
    ) -> dict:
        """ODE full-grad rollout → KinematicProjection reward → reward gradient.

        Steps:
          1. encoder (no_grad) → anchor_hidden_valid
          2. ctx_sampled_pos/heading → v_init, delta_init
          3. ODE full-BPTT rollout → y_hat  (gradient flows through all steps)
          4. KinematicProjectionReward(y_hat) → Huber(y_hat, y_proj.detach())
          5. (optional) BC regularisation on GT via flow_reg_lambda
        """
        kp = self.encoder.agent_encoder.kinematic_projector
        flow_ode = self.encoder.agent_encoder.flow_ode
        flow_decoder = self.encoder.agent_encoder.flow_decoder

        # 1. Encode context (no_grad; encoder is frozen)
        with torch.no_grad():
            map_feature = self.encoder.encode_map(tokenized_map)
            _, _, anchor_hidden_valid = self.encoder.encode_anchor_context_from_map_feature(
                map_feature=map_feature,
                tokenized_agent=tokenized_agent,
                anchor_mask_key="flow_train_mask",
            )

        if anchor_hidden_valid.numel() == 0:
            dummy = next(p for p in self.parameters() if p.requires_grad)
            return {"loss": dummy.sum() * 0.0}

        anchor_hidden = anchor_hidden_valid.detach().to(dtype=torch.float32)

        # 2. v_init / delta_init (closed-loop chunk 초기화와 동일 공식)
        anchor_mask_tensor = tokenized_agent.get("flow_train_mask")
        v_init, delta_init = self._compute_kinematic_init(tokenized_agent, anchor_mask_tensor, kp)
        agent_type = tokenized_agent["flow_train_agent_type"]

        # 3–4. Full-BPTT ODE + KinematicProjection reward (soft Huber loss)
        gt_clean_norm = tokenized_agent.get("flow_train_clean_norm")
        _dev = anchor_hidden.device
        result = self.terminal_cost_final_step_loss.forward_reward_grad(
            flow_decoder=flow_decoder,
            flow_ode=flow_ode,
            anchor_hidden_valid=anchor_hidden,
            reward_fn=self.kinematic_reward_fn,
            gt_clean_norm=gt_clean_norm.to(dtype=torch.float32, device=_dev) if gt_clean_norm is not None else None,
            # reward_fn kwargs:
            agent_type=agent_type.to(_dev),
            v_init=v_init.to(_dev) if v_init is not None else None,
            delta_init=delta_init.to(_dev) if delta_init is not None else None,
        )

        log_dict = {
            "train/reward_loss": result.terminal_cost,
            "train/projection_gap": result.projection_gap,
            "train/v_init_mean": v_init.mean().item() if v_init is not None else 0.0,
        }
        if result.flow_reg_loss is not None:
            log_dict["train/bc_loss"] = result.flow_reg_loss
        log_dict["train/loss"] = result.loss.detach()
        return {"loss": result.loss, **log_dict}

    def _run_kinematic_proj_ft_step(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
    ) -> dict:
        """ODE 생성 → KinematicProjection → projected trajectory를 FM target으로 fine-tuning.

        Steps:
          1. encoder (no_grad) → anchor_hidden_valid
          2. ctx_sampled_pos/heading → v_init, delta_init  (closed-loop init과 동일)
          3. ODE generate (no_grad, current policy) → y_hat
          4. KinematicProjection(y_hat, v_init, delta_init) → y_proj  (no_grad)
          5. flow_matching_loss(flow_decoder(x_t), target)  with y_proj as clean target
          6. (optional) BC regularization on GT with flow_reg_lambda
        """
        kp = self.encoder.agent_encoder.kinematic_projector
        flow_ode = self.encoder.agent_encoder.flow_ode
        flow_decoder = self.encoder.agent_encoder.flow_decoder

        # 1. Encode context (no_grad; encoder is frozen in finetune mode)
        with torch.no_grad():
            map_feature = self.encoder.encode_map(tokenized_map)
            _, _, anchor_hidden_valid = self.encoder.encode_anchor_context_from_map_feature(
                map_feature=map_feature,
                tokenized_agent=tokenized_agent,
                anchor_mask_key="flow_train_mask",
            )

        if anchor_hidden_valid.numel() == 0:
            dummy = next(p for p in self.parameters() if p.requires_grad)
            return {"loss": dummy.sum() * 0.0}

        anchor_hidden = anchor_hidden_valid.detach().to(dtype=torch.float32)

        # 2. v_init / delta_init (closed-loop chunk 초기화와 동일 공식)
        anchor_mask_tensor = tokenized_agent.get("flow_train_mask")
        v_init, delta_init = self._compute_kinematic_init(tokenized_agent, anchor_mask_tensor, kp)

        # 3. ODE generate with current (frozen) policy
        n_anchor = anchor_hidden.shape[0]
        noise_scale = float(getattr(self.finetune_config, "rollout_noise_scale", 1.0))
        x_init = torch.randn(n_anchor, 20, 4, device=anchor_hidden.device, dtype=torch.float32)
        x_init = x_init * noise_scale

        def _model_fn(x_t: Tensor, tau: Tensor) -> Tensor:
            with torch.no_grad():
                return flow_decoder(anchor_hidden, x_t, tau)

        with torch.no_grad():
            y_hat = flow_ode.generate(x_init=x_init, model_fn=_model_fn)

            # 4. KinematicProjection → projected target
            agent_type = tokenized_agent["flow_train_agent_type"]
            _dev = y_hat.device
            y_proj = kp(
                y_hat,
                agent_type.to(_dev),
                v_init=v_init.to(_dev) if v_init is not None else None,
                delta_init=delta_init.to(_dev) if delta_init is not None else None,
            )

        # 5. Flow matching loss on projected target (gradient through flow_decoder only)
        y_proj_fp32 = y_proj.to(dtype=torch.float32)
        proj_sample = flow_ode.sample(y_proj_fp32, target_type="velocity")
        proj_pred = flow_decoder(anchor_hidden, proj_sample.x_t, proj_sample.tau)
        loss = flow_matching_loss(proj_pred, proj_sample.target)
        log_dict = {
            "train/proj_ft_loss": loss.detach(),
            "train/v_init_mean": v_init.mean().item() if v_init is not None else 0.0,
        }

        # 6. (optional) BC regularization on GT
        if self.finetune_config.flow_reg_lambda > 0:
            gt_clean = tokenized_agent.get("flow_train_clean_norm")
            if gt_clean is not None:
                gt_sample = flow_ode.sample(gt_clean.to(dtype=torch.float32), target_type="velocity")
                gt_pred = flow_decoder(anchor_hidden, gt_sample.x_t, gt_sample.tau)
                bc_loss = flow_matching_loss(gt_pred, gt_sample.target)
                loss = loss + self.finetune_config.flow_reg_lambda * bc_loss
                log_dict["train/bc_loss"] = bc_loss.detach()

        log_dict["train/loss"] = loss.detach()
        return {"loss": loss, **log_dict}

    def _run_adjoint_matching_training_step(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
    ):
        """Frozen base 문맥으로 Adjoint Matching loss를 계산합니다.

        Args:
            tokenized_map: 지도 토큰 사전입니다.
            tokenized_agent: agent 토큰 사전입니다.

        Returns:
            AdjointMatchingResult: loss와 logging용 스칼라 묶음입니다.
        """
        device_type = self.device.type if self.device.type else "cpu"
        # Adjoint loss는 작은 tau 분모와 autograd.grad를 같이 써서 mixed precision에 민감합니다.
        with torch.autocast(device_type=device_type, enabled=False):
            with torch.no_grad():
                map_feature = self.encoder.encode_map(tokenized_map)

                _, _, anchor_hidden_valid = self.encoder.encode_anchor_context_from_map_feature(
                    map_feature=map_feature,
                    tokenized_agent=tokenized_agent,
                    anchor_mask_key="flow_train_mask",
                )
            """
            - ``anchor_hidden_valid``: 유효 anchor만 모은 문맥입니다.
                shape은 ``[n_valid_anchor, hidden_dim]`` 입니다.
            - flow_train_agent_type :  [n_valid_anchor] 
                vehicle / pedestrian / cyclist를 구분하는 용도
            - flow_train_current_control : [n_valid_anchor, 3]
                - “anchor 직전 0.1초 동안의 현재 운동 상태를 body frame으로 표현한 값”
                - 정규화된 값도 아니다.
            - flow_train_current_control_valid : [n_valid_anchor]
                - “방금 만든 current_control을 실제로 믿을 수 있는가”를 나타내는 bool 마스크
                - raw_step-1과 raw_step이 둘 다 valid일 때만 True
                - “현재 운동과의 연속성 제약을 적용할지 여부”
            """
            return self.adjoint_matching_loss(
                flow_decoder=self.encoder.agent_encoder.flow_decoder,
                flow_ode=self.encoder.agent_encoder.flow_ode,
                anchor_hidden_valid=anchor_hidden_valid.detach().to(dtype=torch.float32),
                agent_type=tokenized_agent["flow_train_agent_type"],
                current_control=tokenized_agent["flow_train_current_control"].to(dtype=torch.float32),
                current_control_valid=tokenized_agent["flow_train_current_control_valid"],
            )


    def training_step(self, data, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()

        tokenized_map, tokenized_agent = self.token_processor(data)

        if self._is_rmm_bptt_ft_enabled():
            result = self._run_flow_bptt_ft_step(tokenized_map, tokenized_agent, data)
            for k, v in result.items():
                if k != "loss" and isinstance(v, (Tensor, float)):
                    self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
            self.log("train/loss", result["loss"].detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)

        elif self._is_ocsc_ft_enabled():
            # DDP multi-GPU: 모든 .backward() 호출을 no_sync 컨텍스트 안에서 실행해
            # per-backward all-reduce를 막고, 루프 종료 후 manual_backward(_ddp_dummy)로
            # all-reduce를 딱 1회만 트리거. anchor 수가 GPU마다 달라도 deadlock 없음.
            _ddp_model = getattr(getattr(self, "trainer", None) and self.trainer.strategy, "model", None)
            _no_sync_ctx = (
                _ddp_model.no_sync()
                if _ddp_model is not None and hasattr(_ddp_model, "no_sync")
                else contextlib.nullcontext()
            )
            with _no_sync_ctx:
                diag = self._run_flow_ocsc_ft_step(tokenized_map, tokenized_agent, data)

            # 최종 DDP gradient all-reduce (grad는 이미 누적됨, dummy기여=0)
            if "loss" in diag:
                self.manual_backward(diag["loss"])

            for k, v in diag.items():
                if k == "loss":
                    continue  # dummy tensor — metric 아님
                if isinstance(v, (Tensor, float)):
                    self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
            if "train/consistency_loss" in diag:
                _fm_reg = diag.get("train/fm_reg_loss", 0.0)
                _fm_reg_lambda = float(self.finetune_config.ocsc_fm_reg_lambda)
                _total = diag["train/consistency_loss"] + _fm_reg_lambda * (
                    _fm_reg if isinstance(_fm_reg, Tensor) else torch.tensor(_fm_reg, device=diag["train/consistency_loss"].device)
                )
                self.log("train/loss", _total, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)

        elif self._is_ref_nll_ft_enabled():
            result = self._run_ref_nll_ft_step(tokenized_map, tokenized_agent, data)
            for k, v in result.items():
                if k != "loss" and isinstance(v, (Tensor, float)):
                    self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
            self.log("train/loss", result["loss"].detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)

        elif self._is_kinematic_proj_ft_enabled():
            result = self._run_kinematic_proj_ft_step(tokenized_map, tokenized_agent)
            self.manual_backward(result["loss"])
            for k, v in result.items():
                if k != "loss" and isinstance(v, (Tensor, float)):
                    self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
            self.log("train/loss", result["loss"].detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)

        elif self._is_kinematic_reward_ft_enabled():
            result = self._run_kinematic_reward_ft_step(tokenized_map, tokenized_agent)
            self.manual_backward(result["loss"])
            for k, v in result.items():
                if k != "loss" and isinstance(v, (Tensor, float)):
                    self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
            self.log("train/loss", result["loss"].detach(), on_step=True, on_epoch=True, sync_dist=True, batch_size=1)

        elif self._is_terminal_cost_final_step_enabled():
            with torch.no_grad():
                map_feature = self.encoder.encode_map(tokenized_map)
                _, _, anchor_hidden_valid = self.encoder.encode_anchor_context_from_map_feature(
                    map_feature=map_feature,
                    tokenized_agent=tokenized_agent,
                    anchor_mask_key="flow_train_mask",
                )
            anchor_hidden_fp32 = anchor_hidden_valid.detach().to(dtype=torch.float32)
            gt_clean_norm = tokenized_agent["flow_train_clean_norm"].to(dtype=torch.float32)

            if self.finetune_config.mode == "terminal_cost_full_grad":
                result = self.terminal_cost_final_step_loss.forward_feasibility_with_bc(
                    flow_decoder=self.encoder.agent_encoder.flow_decoder,
                    flow_ode=self.encoder.agent_encoder.flow_ode,
                    anchor_hidden_valid=anchor_hidden_fp32,
                    gt_clean_norm=gt_clean_norm,
                    agent_type=tokenized_agent["flow_train_agent_type"],
                    current_control=tokenized_agent["flow_train_current_control"].to(dtype=torch.float32),
                    current_control_valid=tokenized_agent["flow_train_current_control_valid"],
                )
                self.log("train/loss", result.loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
                self.log("train/feasibility_cost", result.terminal_cost, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
                self.log("train/projection_gap", result.projection_gap, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
                if result.flow_reg_loss is not None:
                    self.log("train/bc_loss", result.flow_reg_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
            else:
                result = self.terminal_cost_final_step_loss.forward_l2(
                    flow_decoder=self.encoder.agent_encoder.flow_decoder,
                    flow_ode=self.encoder.agent_encoder.flow_ode,
                    anchor_hidden_valid=anchor_hidden_fp32,
                    gt_clean_norm=gt_clean_norm,
                )
                self.log("train/loss", result.loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
                self.log("train/l2_loss", result.terminal_cost, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
            self.manual_backward(result.loss)

        elif self._is_adjoint_matching_enabled():
            am_result = self._run_adjoint_matching_training_step(
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
            )
            self.log("train/loss", am_result.loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
            self.log("train/terminal_cost", am_result.terminal_cost, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
            self.log("train/projection_gap", am_result.projection_gap, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
            self.log("train/residual_norm", am_result.residual_norm, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
            self.manual_backward(am_result.loss)

        else:
            pred = self.encoder(
                tokenized_map,
                tokenized_agent,
                anchor_mask_key="flow_train_mask",
            )
            loss, open_metric_dict, _ = self._open_loop_denoise_metrics(pred)
            self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
            self.log("train/ADE2s", open_metric_dict["ADE2s"], on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
            self.log("train/FDE2s", open_metric_dict["FDE2s"], on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
            self.log("train/ADEyaw2s", open_metric_dict["yaw_ADE2s"], on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
            self.log("train/FDEyaw2s", open_metric_dict["yaw_FDE2s"], on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
            self.manual_backward(loss)

        _clip_val = float(getattr(self.finetune_config, "gradient_clip_val", 0.0) or 0.0)
        if _clip_val > 0:
            self.clip_gradients(opt, gradient_clip_val=_clip_val)

        opt.step()

    def _projected_generation_val_step(
        self,
        tokenized_map: Dict,
        tokenized_agent: Dict,
        batch_idx: int,
    ) -> None:
        """Projected Diffusion ODE로 open-loop trajectory를 생성하고 ADE/FDE를 기록합니다.

        매 ODE step 후 kinematic feasibility gap에 대해 gradient descent를 수행합니다.
        """
        # train/val에 따라 mask key 결정 (train: flow_train_*, val: flow_eval_*)
        if "flow_eval_mask" in tokenized_agent:
            anchor_mask_key = "flow_eval_mask"
            clean_norm_key = "flow_eval_clean_norm"
            agent_type_key = "flow_eval_agent_type"
            ctrl_key = "flow_eval_current_control"
            ctrl_valid_key = "flow_eval_current_control_valid"
        elif "flow_train_mask" in tokenized_agent:
            anchor_mask_key = "flow_train_mask"
            clean_norm_key = "flow_train_clean_norm"
            agent_type_key = "flow_train_agent_type"
            ctrl_key = "flow_train_current_control"
            ctrl_valid_key = "flow_train_current_control_valid"
        else:
            return

        with torch.no_grad():
            map_feature = self.encoder.encode_map(tokenized_map)
            _, _, anchor_hidden_valid = self.encoder.encode_anchor_context_from_map_feature(
                map_feature=map_feature,
                tokenized_agent=tokenized_agent,
                anchor_mask_key=anchor_mask_key,
            )

        if anchor_hidden_valid.numel() == 0:
            return

        anchor_hidden_valid = anchor_hidden_valid.to(dtype=torch.float32)
        flow_decoder = self.encoder.agent_encoder.flow_decoder
        flow_ode = self.encoder.agent_encoder.flow_ode

        x_init = torch.randn(
            anchor_hidden_valid.shape[0], 20, 4,
            device=anchor_hidden_valid.device,
            dtype=torch.float32,
        )

        def model_fn(x_t: Tensor, tau: Tensor) -> Tensor:
            with torch.no_grad():
                return flow_decoder(anchor_hidden_valid, x_t, tau)

        pred_clean_norm = self.projected_generator.generate(
            flow_ode=flow_ode,
            model_fn=model_fn,
            x_init=x_init,
            agent_type=tokenized_agent[agent_type_key],
            current_control=tokenized_agent.get(ctrl_key),
            current_control_valid=tokenized_agent.get(ctrl_valid_key),
            steps=16,
        )

        target_clean_norm = tokenized_agent[clean_norm_key].to(
            device=pred_clean_norm.device, dtype=pred_clean_norm.dtype
        )
        proj_metric_dict = self._build_open_loop_metric_dict(
            pred_clean_norm=pred_clean_norm,
            target_clean_norm=target_clean_norm,
        )
        # key 앞에 proj_ prefix 붙여 저장
        proj_metric_dict = {f"proj_{k}": v for k, v in proj_metric_dict.items()}
        self._update_weighted_validation_metrics(
            metric_store=self.val_projected_epoch_metrics,
            metric_dict=proj_metric_dict,
            sample_count=int(target_clean_norm.shape[0]),
        )

    def _final_projection_val_step(
        self,
        tokenized_map: Dict,
        tokenized_agent: Dict,
    ) -> None:
        """표준 ODE 생성 후 마지막에 한 번만 kinematic projection 적용하고 ADE/FDE를 기록합니다.

        PPR과 달리 매 step이 아닌 ODE 완료 후 최종 결과에만 KinematicProjection을 적용합니다.
        """
        if "flow_eval_mask" in tokenized_agent:
            anchor_mask_key = "flow_eval_mask"
            clean_norm_key = "flow_eval_clean_norm"
            agent_type_key = "flow_eval_agent_type"
            ctrl_key = "flow_eval_current_control"
            ctrl_valid_key = "flow_eval_current_control_valid"
        elif "flow_train_mask" in tokenized_agent:
            anchor_mask_key = "flow_train_mask"
            clean_norm_key = "flow_train_clean_norm"
            agent_type_key = "flow_train_agent_type"
            ctrl_key = "flow_train_current_control"
            ctrl_valid_key = "flow_train_current_control_valid"
        else:
            return

        with torch.no_grad():
            map_feature = self.encoder.encode_map(tokenized_map)
            _, _, anchor_hidden_valid = self.encoder.encode_anchor_context_from_map_feature(
                map_feature=map_feature,
                tokenized_agent=tokenized_agent,
                anchor_mask_key=anchor_mask_key,
            )

        if anchor_hidden_valid.numel() == 0:
            return

        anchor_hidden_valid = anchor_hidden_valid.to(dtype=torch.float32)
        flow_decoder = self.encoder.agent_encoder.flow_decoder
        flow_ode = self.encoder.agent_encoder.flow_ode

        x_init = torch.randn(
            anchor_hidden_valid.shape[0], 20, 4,
            device=anchor_hidden_valid.device,
            dtype=torch.float32,
        )

        def model_fn(x_t: Tensor, tau: Tensor) -> Tensor:
            with torch.no_grad():
                return flow_decoder(anchor_hidden_valid, x_t, tau)

        # v_init / delta_init: _compute_kinematic_init 헬퍼 사용
        anchor_mask_tensor = tokenized_agent.get(anchor_mask_key)
        _kp_fp = self._final_proj_kin_projector
        v_init, delta_init = self._compute_kinematic_init(tokenized_agent, anchor_mask_tensor, _kp_fp)
        # ctx 정보가 없으면 current_control fallback
        if v_init is None:
            current_control = tokenized_agent.get(ctrl_key)
            current_control_valid = tokenized_agent.get(ctrl_valid_key)
            if current_control is not None:
                v_init = current_control[..., :2].norm(dim=-1)
                if current_control_valid is not None:
                    v_init = v_init.masked_fill(~current_control_valid.to(x_init.device), 0.0)
                if delta_init is None and _kp_fp is not None:
                    omega = current_control[..., 2]
                    _v = v_init.clamp_min(1e-6)
                    delta_init = torch.atan(_kp_fp.wheelbase * (omega / _v)).clamp(-_kp_fp.delta_max, _kp_fp.delta_max)
                    if current_control_valid is not None:
                        delta_init = delta_init.masked_fill(~current_control_valid.to(x_init.device), 0.0)

        agent_type = tokenized_agent[agent_type_key]

        print(
            f"[final_proj] n_agents={anchor_hidden_valid.shape[0]} "
            f"v_init={'None' if v_init is None else f'mean={v_init.mean():.3f}'}"
        )

        with torch.no_grad():
            # 1. 표준 ODE (per-step projection 없음)
            pred_clean_norm = flow_ode.generate(x_init=x_init, model_fn=model_fn)
            _pre_disp = (pred_clean_norm[..., :2] * 20.0).norm(dim=-1).mean().item()
            print(f"[final_proj] pre-proj  mean_disp={_pre_disp:.4f}m")

            # 2. 마지막 한 번만 kinematic projection
            _dev = pred_clean_norm.device
            pred_clean_norm = self._final_proj_kin_projector(
                pred_clean_norm,
                agent_type.to(_dev),
                v_init=v_init.to(_dev) if v_init is not None else None,
                delta_init=delta_init.to(_dev) if delta_init is not None else None,
            )
            _post_disp = (pred_clean_norm[..., :2] * 20.0).norm(dim=-1).mean().item()
            print(f"[final_proj] post-proj mean_disp={_post_disp:.4f}m  delta={_post_disp - _pre_disp:+.4f}m")

        target_clean_norm = tokenized_agent[clean_norm_key].to(
            device=pred_clean_norm.device, dtype=pred_clean_norm.dtype
        )
        fp_metric_dict = self._build_open_loop_metric_dict(
            pred_clean_norm=pred_clean_norm,
            target_clean_norm=target_clean_norm,
        )
        fp_metric_dict = {f"final_proj_{k}": v for k, v in fp_metric_dict.items()}
        self._update_weighted_validation_metrics(
            metric_store=self.val_final_proj_epoch_metrics,
            metric_dict=fp_metric_dict,
            sample_count=int(target_clean_norm.shape[0]),
        )

    def validation_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        map_feature = None
        if self.val_open_loop or self.val_closed_loop:
            map_feature = self.encoder.encode_map(tokenized_map)

        if self.val_open_loop:
            denoise_pred = self.encoder.forward_from_map_feature(
                map_feature=map_feature,
                tokenized_agent=tokenized_agent,
                anchor_mask_key="flow_eval_mask",
            )
            open_sample_count = int(denoise_pred["flow_clean_norm"].shape[0])
            # v_init / delta_init: _compute_kinematic_init 헬퍼 사용 (closed-loop 와 동일 공식)
            _kp_ol = self.encoder.agent_encoder.kinematic_projector
            open_v_init, open_delta_init = (None, None)
            if _kp_ol is not None and denoise_pred["anchor_mask"].numel() > 0:
                open_v_init, open_delta_init = self._compute_kinematic_init(
                    tokenized_agent, denoise_pred["anchor_mask"], _kp_ol
                )
            # ── 검증용 print (ctx 경로 vs fallback 여부 확인) ──
            print(
                f"[open_loop_init] batch={batch_idx} "
                f"ctx_pos={'OK' if 'ctx_sampled_pos' in tokenized_agent else 'MISSING'} "
                f"ctx_head={'OK' if 'ctx_sampled_heading' in tokenized_agent else 'MISSING'} "
                f"kin_proj={'OK' if _kp_ol is not None else 'None'} "
                f"v_init={'ctx({:.3f})'.format(open_v_init.mean().item()) if open_v_init is not None else 'None→fallback'} "
                f"delta_init={'ctx({:.4f})'.format(open_delta_init.mean().item()) if open_delta_init is not None else 'None→fallback'}"
            )
            open_pred_clean_norm = self.encoder.sample_open_loop_future(
                anchor_hidden=denoise_pred["anchor_hidden"],
                anchor_mask=denoise_pred["anchor_mask"],
                sampling_noise=self.eval_sampling_noise,
                sampling_seed=self._get_validation_open_seed(batch_idx),
                agent_type=tokenized_agent.get("flow_eval_agent_type"),
                v_init=open_v_init,
                delta_init=open_delta_init,
                current_control=tokenized_agent.get("flow_eval_current_control"),
                current_control_valid=tokenized_agent.get("flow_eval_current_control_valid"),
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

            # kinematic projection open-loop metrics (separate tracker for before/after comparison)
            if hasattr(self, "val_kinematic_proj_epoch_metrics"):
                kin_metric_dict = {f"kin_{k}": v for k, v in open_metric_dict.items()}
                self._update_weighted_validation_metrics(
                    metric_store=self.val_kinematic_proj_epoch_metrics,
                    metric_dict=kin_metric_dict,
                    sample_count=open_sample_count,
                )

        # Projected Diffusion open-loop generation (feasibility projection at each ODE step)
        if self.projected_generator is not None:
            self._projected_generation_val_step(
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
                batch_idx=batch_idx,
            )

        # Final projection: standard ODE → post-hoc gradient descent to feasible region
        if self._final_proj_kin_projector is not None:
            self._final_projection_val_step(
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
            )

        if self.val_closed_loop:
            pred_traj, pred_z, pred_head = self._run_closed_loop_rollouts(
                data=data,
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
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
                max_scenarios = min(
                    int(self.n_vis_scenario),
                    len(data.get("tfrecord_path", [])),
                    len(scenario_rollouts),
                )
                for scen_idx in range(max_scenarios):
                    vis = VisWaymo(
                        scenario_path=data["tfrecord_path"][scen_idx],
                        save_dir=self.video_dir / f"batch_{batch_idx:02d}-scenario_{scen_idx:02d}",
                    )
                    vis.save_video_scenario_rollout(scenario_rollouts[scen_idx], self.n_vis_rollout)
                    for video_path in vis.video_paths:
                        if video_logger is not None:
                            video_logger.log_video("/".join(video_path.split("/")[-3:]), [video_path], format="gif")
                            if self.delete_local_videos_after_wandb_upload:
                                self._cleanup_local_video(video_path)

            # ── pretrained ref rollout (Δ RMM 기준선) ─────────────────────
            if (
                self._ref_val_enabled
                and self.ref_flow_decoder is not None
                and self.ref_sim_agents_metrics is not None
                and not self.sim_agents_submission.is_active
                and batch_idx < self.n_batch_sim_agents_metric
            ):
                # flow_decoder 를 pretrained ref 로 교체 후 동일 조건 rollout.
                # scenario_sampling_seeds 는 scenario_id+rollout_idx 해시로 결정되므로
                # 위 finetuned rollout 과 자동으로 같은 noise 를 사용합니다.
                _orig_fd = self.encoder.agent_encoder.flow_decoder
                self.encoder.agent_encoder.flow_decoder = self.ref_flow_decoder
                try:
                    ref_pred_traj, ref_pred_z, ref_pred_head = self._run_closed_loop_rollouts(
                        data=data,
                        tokenized_agent=tokenized_agent,
                        map_feature=map_feature,
                    )
                finally:
                    self.encoder.agent_encoder.flow_decoder = _orig_fd
                self.ref_sim_agents_metrics.update_from_prediction_tensors(
                    scenario_files=data["tfrecord_path"],
                    agent_id=data["agent"]["id"],
                    agent_batch=data["agent"]["batch"],
                    pred_traj=ref_pred_traj,
                    pred_z=ref_pred_z,
                    pred_head=ref_pred_head,
                )

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

        if self.projected_generator is not None:
            epoch_proj_metrics = self._compute_and_reset_validation_metrics(
                prefix="val_projected",
                metric_store=self.val_projected_epoch_metrics,
            )
            for metric_name, metric_value in epoch_proj_metrics.items():
                self.log(metric_name, metric_value, on_step=False, on_epoch=True, sync_dist=True)

        if self._final_proj_kin_projector is not None:
            epoch_fp_metrics = self._compute_and_reset_validation_metrics(
                prefix="val_final_proj",
                metric_store=self.val_final_proj_epoch_metrics,
            )
            for metric_name, metric_value in epoch_fp_metrics.items():
                self.log(metric_name, metric_value, on_step=False, on_epoch=True, sync_dist=True)

        if hasattr(self, "val_kinematic_proj_epoch_metrics"):
            epoch_kin_metrics = self._compute_and_reset_validation_metrics(
                prefix="val_kinematic",
                metric_store=self.val_kinematic_proj_epoch_metrics,
            )
            for metric_name, metric_value in epoch_kin_metrics.items():
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
                closed_loop_metric = epoch_sim_agents_metrics[self.closed_loop_metric_name]
                if self.global_rank == 0 and minade_value is not None:
                    epoch_sim_agents_metrics[self.val_closed_minade_name] = minade_value
                self.log(
                    self.closed_loop_metric_name,
                    closed_loop_metric,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                if self.global_rank == 0 and self.logger is not None:
                    epoch_sim_agents_metrics["epoch"] = (
                        self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                    )
                    self.logger.log_metrics(epoch_sim_agents_metrics)
                self.sim_agents_metrics.reset()
                self.minADE.reset()

                # ── ref model metrics + Δ RMM ─────────────────────────────
                if self._ref_val_enabled and self.ref_sim_agents_metrics is not None:
                    self.ref_sim_agents_metrics._drain_completed_futures(wait=True, drain_all=True)
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        ref_state = self.ref_sim_agents_metrics.get_state_tensor(device=self.device)
                        torch.distributed.all_reduce(ref_state)
                        ref_epoch_metrics = self.ref_sim_agents_metrics.compute_from_state_tensor(ref_state)
                    else:
                        ref_epoch_metrics = self.ref_sim_agents_metrics.compute()
                    ref_rmm_key = "val_ref/sim_agents_2025/realism_meta_metric"
                    delta_rmm_key = "val_delta/sim_agents_2025/realism_meta_metric"
                    ref_rmm = ref_epoch_metrics[ref_rmm_key]
                    delta_rmm = closed_loop_metric - ref_rmm
                    self.log(ref_rmm_key, ref_rmm, on_step=False, on_epoch=True, sync_dist=False)
                    self.log(delta_rmm_key, delta_rmm, on_step=False, on_epoch=True, sync_dist=False)
                    if self.global_rank == 0 and self.logger is not None:
                        wandb_payload = dict(ref_epoch_metrics)
                        wandb_payload[delta_rmm_key] = delta_rmm
                        wandb_payload["epoch"] = (
                            self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                        )
                        self.logger.log_metrics(wandb_payload)
                    self.ref_sim_agents_metrics.reset()

            if self.sim_agents_submission.is_active:
                self.sim_agents_submission.save_sub_file()

    def _resolve_lr_total_steps(self) -> int:
        """현재 스케줄 단위에 맞는 전체 step 수를 정합니다.

        Returns:
            int: cosine schedule 전체 길이입니다.
        """
        if self.lr_total_steps > 0:
            return self.lr_total_steps
        if self.lr_scheduler_unit == "step" and self.trainer is not None:
            # automatic_optimization=False에서는 estimated_stepping_batches가
            # 0을 반환할 수 있으므로 num_batches * max_epochs로 직접 추정
            try:
                n_batches = len(self.trainer.train_dataloader)
            except Exception:
                n_batches = 0
            if n_batches > 0:
                return max(1, n_batches * max(1, int(self.trainer.max_epochs)))
            estimated_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
            if estimated_steps > 0:
                return estimated_steps
        if self.trainer is not None:
            return max(int(self.trainer.max_epochs), 1)
        return 1

    def configure_optimizers(self):
        def lr_lambda(current_index: int) -> float:
            if not hasattr(self, "_cached_lr_total_steps"):
                self._cached_lr_total_steps = self._resolve_lr_total_steps()
            total_steps = self._cached_lr_total_steps
            current_step = current_index + 1
            if current_step < self.lr_warmup_steps:
                return self.lr_min_ratio + (1.0 - self.lr_min_ratio) * current_step / max(self.lr_warmup_steps, 1)
            return self.lr_min_ratio + 0.5 * (1.0 - self.lr_min_ratio) * (
                1.0
                + math.cos(
                    math.pi * min(
                        1.0,
                        (current_step - self.lr_warmup_steps) / max(total_steps - self.lr_warmup_steps, 1),
                    )
                )
            )

        if (
            self.finetune_config.enabled
            and self.finetune_config.mode == "dice_ft"
            and self.dice_critic is not None
        ):
            # DICE mode: two param groups in a single optimizer.
            # • Group 0 (actor): flow_decoder — updated by L_actor gradient
            # • Group 1 (critic): dice_critic — updated by L_critic gradient
            # Gradients are disjoint by construction (see _run_dice_ft_step).
            actor_params = [p for p in self.encoder.agent_encoder.flow_decoder.parameters() if p.requires_grad]
            critic_params = list(self.dice_critic.parameters())
            if not actor_params:
                raise RuntimeError("dice_ft: no trainable actor (flow_decoder) parameters found.")
            optimizer = torch.optim.AdamW(
                [
                    {"params": actor_params, "lr": self.lr},
                    {"params": critic_params, "lr": self.finetune_config.dice_critic_lr},
                ],
                weight_decay=self.weight_decay,
            )
        else:
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            if not trainable_params:
                raise RuntimeError("No trainable parameters were found.")
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.lr_scheduler_unit,
                "frequency": 1,
            },
        }

    def load_state_dict(
        self,
        state_dict: Dict[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """기존 checkpoint를 새 residual head 구조와 호환되게 읽습니다.

        Args:
            state_dict: 불러올 state dict 입니다.
            strict: True면 residual head를 뺀 나머지 키는 엄격히 검사합니다.
            assign: PyTorch 기본 ``load_state_dict`` 옵션을 그대로 전달합니다.

        Returns:
            _IncompatibleKeys: PyTorch가 돌려주는 키 검사 결과입니다.
        """
        incompatible_keys = super().load_state_dict(state_dict, strict=False, assign=assign)
        if not strict:
            return incompatible_keys

        _allowed_missing = ("residual_velocity_head", "_rmm_ema_mean")
        _allowed_unexpected = ("ref_flow_decoder", "_rmm_ema_mean")
        missing_keys = [
            key
            for key in incompatible_keys.missing_keys
            if not any(pat in key for pat in _allowed_missing)
        ]
        unexpected_keys = [
            key
            for key in incompatible_keys.unexpected_keys
            if not any(pat in key for pat in _allowed_unexpected)
        ]
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for SMARTFlow:\n"
                f"Missing key(s): {missing_keys}\n"
                f"Unexpected key(s): {unexpected_keys}"
            )
        return incompatible_keys

    def test_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        map_feature = self.encoder.encode_map(tokenized_map)
        pred_traj, pred_z, pred_head = self._run_closed_loop_rollouts(
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
