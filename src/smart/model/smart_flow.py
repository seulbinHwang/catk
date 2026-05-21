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
    ScenarioDiversityMetrics,
    SimAgentsMetrics,
    SimAgentsSubmission,
    WOSACDistributionMetrics,
    log_and_reset_scenario_diversity_metric,
    log_and_reset_wosac_distribution_metric,
    minADE,
    update_scenario_diversity_metric_from_model,
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
from src.smart.metrics.pwil_consistency_loss import (
    pwil_pairwise_distance,
    pwil_row_distance,
    pwil_hungarian_coupling,
    pwil_greedy_coupling,
    pwil_uniform_coupling,
    pwil_loss,
    pwil_loss_per_cl_row,
)
from src.smart.modules.flow_kinematic_projection import KinematicProjection
from src.smart.utils.geometry import wrap_angle
from src.smart.utils.rollout import transform_to_local
from src.smart.modules.smart_flow_decoder import SMARTFlowDecoder
from src.smart.tokens.flow_token_processor import FlowTokenProcessor
from src.smart.utils.finetune import FinetuneConfig, set_model_for_finetuning
from src.utils.pylogger import RankedLogger
from src.utils.vis_waymo import VisWaymo
from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts


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
        self.ref_flow_decoder: nn.Module | None = None
        # DMD (self_forcing_dmd) 의 fake_score (critic) 사본.
        # __init__ 시점에 deepcopy 해서 configure_optimizers 가 param 을 등록할 수 있게 함.
        # 가중치는 on_train_start 에서 main flow_decoder (pretrained ckpt 적용 후) 로 in-place sync.
        self.fake_score_decoder: nn.Module | None = None
        # DMD generator EMA (Self-Forcing 표준).  dmd_ema_weight > 0 일 때 활성.
        # 학습은 instantaneous weights 로, validation 은 EMA weights 로 (on_validation_start/end swap).
        self.gen_ema: nn.Module | None = None
        self._gen_ema_swap_backup: Dict[str, Tensor] | None = None
        if self.finetune_config.enabled and self.finetune_config.mode not in (
            "ocsc_ft", "road_ft", "self_forcing_dmd"
        ):
            raise ValueError(
                f"Unsupported finetune mode: {self.finetune_config.mode}. "
                "Supported: 'ocsc_ft', 'road_ft', 'self_forcing_dmd'."
            )
        if self._is_dmd_ft_enabled():
            from copy import deepcopy
            self.fake_score_decoder = deepcopy(self.encoder.agent_encoder.flow_decoder)
            # fake_score 의 학습 scope 결정 (dmd_fake_ft_scope):
            #   "full"   = 모든 param trainable (Self-Forcing 기본; main 이 velocity_head only
            #              여도 critic 은 full FT — paper convention).
            #   "mirror" = main flow_decoder 의 requires_grad mask 를 그대로 따라감
            #              (deepcopy 가 mask 도 복사하므로 별도 처리 불필요).  ablation 용.
            _fake_scope = str(getattr(self.finetune_config, "dmd_fake_ft_scope", "full")).lower()
            if _fake_scope == "full":
                for p in self.fake_score_decoder.parameters():
                    p.requires_grad_(True)
            elif _fake_scope != "mirror":
                log.warning(
                    f"[{self.finetune_config.mode}] unknown dmd_fake_ft_scope={_fake_scope!r}; "
                    "falling back to 'full' (all fake_score params trainable)."
                )
                for p in self.fake_score_decoder.parameters():
                    p.requires_grad_(True)
            n_params = sum(p.numel() for p in self.fake_score_decoder.parameters())
            n_train = sum(p.numel() for p in self.fake_score_decoder.parameters() if p.requires_grad)
            log.info(
                f"[{self.finetune_config.mode}] fake_score_decoder constructed "
                f"({n_params:,} total params, {n_train:,} trainable; scope={_fake_scope}). "
                "Weights will be synced from main flow_decoder in on_train_start (post-ckpt-load)."
            )
            # Generator EMA — dmd_ema_weight > 0 일 때만 생성.
            # validation 시 instantaneous ↔ EMA swap.
            _ema_w = float(getattr(self.finetune_config, "dmd_ema_weight", 0.0))
            if _ema_w > 0.0:
                self.gen_ema = deepcopy(self.encoder.agent_encoder.flow_decoder)
                for p in self.gen_ema.parameters():
                    p.requires_grad_(False)
                log.info(
                    f"[{self.finetune_config.mode}] gen_ema constructed "
                    f"(decay={_ema_w}, start_step={getattr(self.finetune_config, 'dmd_ema_start_step', 0)}). "
                    "Weights will be synced from main flow_decoder in on_train_start."
                )

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

        # Generated scenario의 행동(intent) 수준 다양성 metric (CPD 와 함께 로깅).
        self.scenario_diversity_metrics = ScenarioDiversityMetrics(
            prefix="val_closed",
            lat_threshold_m=float(getattr(model_config, "diversity_lat_threshold_m", 1.75)),
            stop_speed_mps=float(getattr(model_config, "diversity_stop_speed_mps", 0.5)),
        )

        # OCSC / RoaD: per-step HardRMM 모니터링용 인-프로세스 metric 객체 (current + ref)
        # (DMD 는 train-time RMM 모니터링 비활성 — 의도적으로 미지원.)
        _is_ocsc = self.finetune_config.enabled and self.finetune_config.mode == "ocsc_ft"
        _is_road = self.finetune_config.enabled and self.finetune_config.mode == "road_ft"
        _want_train_rmm = (
            _is_ocsc and bool(getattr(self.finetune_config, "ocsc_eval_hard_rmm", True))
        ) or (
            _is_road and bool(getattr(self.finetune_config, "road_eval_hard_rmm", False))
        )
        if _want_train_rmm:
            self._ocsc_train_hard_rmm: HardSimAgentsMetrics | None = HardSimAgentsMetrics("train_ocsc")
            self._ocsc_train_hard_rmm_ref: HardSimAgentsMetrics | None = HardSimAgentsMetrics("train_ocsc_ref")
        else:
            self._ocsc_train_hard_rmm = None
            self._ocsc_train_hard_rmm_ref = None

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

        self.video_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.video_dir = Path(self.video_dir) / "videos"

        self.eval_sampling_noise = model_config.eval_sampling_noise

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

    def _is_ocsc_ft_enabled(self) -> bool:
        return bool(
            self.finetune_config.enabled
            and self.finetune_config.mode == "ocsc_ft"
        )

    def _is_road_ft_enabled(self) -> bool:
        return bool(
            self.finetune_config.enabled
            and self.finetune_config.mode == "road_ft"
        )

    def _is_dmd_ft_enabled(self) -> bool:
        return bool(
            self.finetune_config.enabled
            and self.finetune_config.mode == "self_forcing_dmd"
        )

    def on_train_start(self) -> None:
        # OCSC / DMD: ref_flow_decoder 를 항상 생성 (open-loop target / real_score teacher / delta HardRMM 모니터링 공용)
        # OCSC: ocsc_use_pretrained_ref=False 여도 delta 계산을 위해 frozen ref 가 필요.
        # DMD: dmd_use_real_score=True (기본) 일 때 real_score teacher 로 사용.
        _needs_ref_ocsc = (
            self.finetune_config.enabled
            and self.ref_flow_decoder is None
            and self.finetune_config.mode == "ocsc_ft"
        )
        _needs_ref_dmd = (
            self._is_dmd_ft_enabled()
            and self.ref_flow_decoder is None
            and bool(getattr(self.finetune_config, "dmd_use_real_score", True))
        )
        if _needs_ref_ocsc or _needs_ref_dmd:
            from copy import deepcopy
            flow_decoder = self.encoder.agent_encoder.flow_decoder
            self.ref_flow_decoder = deepcopy(flow_decoder)
            for p in self.ref_flow_decoder.parameters():
                p.requires_grad_(False)
            print(f"[{self.finetune_config.mode}] frozen reference model created from pretrained checkpoint.")

        # DMD: fake_score_decoder 는 __init__ 에서 이미 생성됨 (configure_optimizers 가 참조해야 하므로).
        # 여기서는 pretrained ckpt 적용 후의 main flow_decoder 가중치로 in-place sync.
        # (deepcopy 가 아니라 load_state_dict 를 써서 optimizer 의 param 참조를 유지.)
        if self._is_dmd_ft_enabled() and self.fake_score_decoder is not None:
            cur_state = self.encoder.agent_encoder.flow_decoder.state_dict()
            self.fake_score_decoder.load_state_dict(cur_state, strict=True)
            print(
                f"[{self.finetune_config.mode}] fake_score_decoder synced from main "
                "flow_decoder (post-ckpt-load init)."
            )

        # DMD: gen_ema 도 동일 (post-ckpt) main flow_decoder 로 sync.
        if self._is_dmd_ft_enabled() and self.gen_ema is not None:
            cur_state = self.encoder.agent_encoder.flow_decoder.state_dict()
            self.gen_ema.load_state_dict(cur_state, strict=True)
            print(
                f"[{self.finetune_config.mode}] gen_ema synced from main flow_decoder "
                "(post-ckpt-load init)."
            )

        # ocsc_ft: BPTT backward through ODE steps can produce NaN/Inf
        # gradients (exploding Jacobian, numerical instability). Register nan_to_num
        # hooks on trainable parameters so any NaN/Inf gradient is zeroed out.
        if self._is_ocsc_ft_enabled() or self._is_road_ft_enabled() or self._is_dmd_ft_enabled():
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
        # ── PWIL mode 호환성 (loss_type=="pwil" 일 때만 작동) ──────────────────
        _is_pwil = (loss_type == "pwil")
        pwil_coupling = str(getattr(self.finetune_config, "ocsc_pwil_coupling", "hungarian")).lower()
        pwil_use_exp_reward = bool(getattr(self.finetune_config, "ocsc_pwil_use_exp_reward", True))
        pwil_alpha = float(getattr(self.finetune_config, "ocsc_pwil_alpha", 1.0))
        pwil_beta = float(getattr(self.finetune_config, "ocsc_pwil_beta", 5.0))
        if _is_pwil:
            if pwil_coupling not in ("hungarian", "greedy", "uniform"):
                raise ValueError(
                    f"[ocsc_ft] ocsc_pwil_coupling={pwil_coupling!r} invalid; "
                    "expected one of: hungarian, greedy, uniform."
                )
            if pwil_coupling == "hungarian" and M_ol != G:
                raise ValueError(
                    f"[ocsc_ft] PWIL hungarian requires M (ocsc_n_ol_rollouts) == G "
                    f"(ocsc_n_rollouts); got M={M_ol}, G={G}. Use 'greedy' for asymmetric."
                )
            if pwil_coupling == "greedy" and M_ol < G:
                log.warning(
                    f"[ocsc_ft] PWIL greedy: M={M_ol} < G={G} — coupling is feasible but "
                    "OL coverage is sparse; consider M >= G."
                )
            if use_mmd:
                log.warning("[ocsc_ft] loss_type=pwil: forcing use_mmd=False (incompatible).")
                use_mmd = False
            if _nearest_match:
                log.warning(
                    "[ocsc_ft] loss_type=pwil: forcing ocsc_ol_nearest_match=False "
                    "(PWIL coupling replaces nearest_match)."
                )
                _nearest_match = False
            if _shared_ol:
                # M < G broadcast 는 PWIL coupling 의미가 없음.
                raise ValueError(
                    f"[ocsc_ft] PWIL incompatible with _shared_ol (M={M_ol} < G={G} broadcast). "
                    "Set ocsc_n_ol_rollouts >= ocsc_n_rollouts."
                )
        # ocsc_gt_target=True: open-loop sample 대신 GT 궤적을 target으로 사용.
        # CL 예측을 2Hz로 다운샘플 후 GT(2Hz)와 비교.
        use_gt_target = bool(getattr(self.finetune_config, "ocsc_gt_target", False))
        if _is_pwil and use_gt_target:
            raise ValueError(
                "[ocsc_ft] loss_type=pwil + ocsc_gt_target=True is incompatible: "
                "PWIL requires an OL sample distribution (M>=1 OL samples) — "
                "GT is a single trajectory and yields a degenerate coupling. "
                "Use 'l2' / 'smooth_l1' with ocsc_gt_target=True for paired-to-GT mode."
            )
        # GT resolution: "2hz" (default, 기존) | "10hz" (raw fine 10Hz GT, no downsample).
        gt_resolution = str(getattr(self.finetune_config, "ocsc_gt_resolution", "2hz")).lower()
        if gt_resolution not in ("2hz", "10hz"):
            log.warning(
                f"[ocsc_ft] unknown ocsc_gt_resolution={gt_resolution!r}, falling back to '2hz'."
            )
            gt_resolution = "2hz"
        gt_is_10hz = (gt_resolution == "10hz")
        # OL target 분기 (use_gt_target=False) 의 시간 해상도.
        # "10hz" (default, 기존 동작): OL native fine 20 step ↔ CL native fine 10Hz.
        # "2hz": OL 출력을 4::5 로 2Hz coarse 다운샘플 ↔ CL 도 _cl_downsample_to_2hz 적용.
        # GT 분기는 ocsc_gt_resolution 사용. 두 토글 독립.
        ol_resolution = str(getattr(self.finetune_config, "ocsc_ol_resolution", "10hz")).lower()
        if ol_resolution not in ("2hz", "10hz"):
            log.warning(
                f"[ocsc_ft] unknown ocsc_ol_resolution={ol_resolution!r}, falling back to '10hz'."
            )
            ol_resolution = "10hz"
        ol_is_2hz = (ol_resolution == "2hz")
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
        # strict active_mask: future fine step 모두 valid 인 agent 만 OCSC anchor 로 사용.
        _strict_active_mask = bool(getattr(self.finetune_config, "ocsc_strict_active_mask", False))
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

        def _ol_downsample_to_2hz(ol_norm: Tensor, T_target: int) -> Tensor:
            """OL native 10Hz fine [n, 20, 4] → 2Hz coarse [n, T_target, 4].

            OL 출력은 이미 anchor-frame normalized 라 좌표 변환 없이 시간 축만 slice.
            CL 의 _cl_downsample_to_2hz 와 동일한 4::5 규칙으로 +0.5s, +1.0s, ... 추출.
            """
            return ol_norm[:, _shift - 1 :: _shift, :][:, :T_target]

        # OL-path suffix slicer 선택: ol_is_2hz=True 면 2Hz tail, 아니면 fine 10Hz tail.
        # (GT-path 는 별도로 gt_is_10hz 분기 그대로 사용.)
        _ol_slice_fn = _slice_consistency_suffix_2hz if ol_is_2hz else _slice_consistency_suffix

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
                if _strict_active_mask:
                    # main training 과 일관성: future fine step 모두 valid 인 agent 만 사용.
                    # 부분 invalid agent (anchor valid 인데 future 일부 invalid) 는 OCSC 학습에서 제외 →
                    # model 이 학습 안 한 영역의 hallucination self-consistency 방지.
                    _T_future_raw = pred_max_steps_raw * _shift
                    _anchor_now_10hz_str = (anchor_idx + 1) * _shift
                    _future_start_str = _anchor_now_10hz_str + 1
                    _future_end_str = _future_start_str + _T_future_raw
                    _raw_valid_full = data["agent"]["valid_mask"]
                    _seq_len = _raw_valid_full.shape[1]
                    if _future_end_str <= _seq_len and _T_future_raw > 0:
                        _future_valid_str = _raw_valid_full[:, _future_start_str:_future_end_str].all(dim=1)
                    else:
                        # sequence 끝 부근: future 부족하면 모두 invalid 처리 (안전)
                        _future_valid_str = torch.zeros(
                            _raw_valid_full.shape[0], dtype=torch.bool, device=_raw_valid_full.device,
                        )
                    active_mask = active_mask & _future_valid_str
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

                    # ol_is_2hz: OL native fine 20 step → 2Hz coarse 다운샘플.
                    # GT-2hz 분기와 동일하게 4::5 규칙. CL 도 아래에서 _cl_downsample_to_2hz 로 매칭.
                    if ol_is_2hz:
                        _T_ol_2hz = pred_max_steps_raw if pred_max_steps_raw > 0 else 4
                        ol_norms = [_ol_downsample_to_2hz(o, _T_ol_2hz) for o in ol_norms]

                    # nearest_include_gt: candidate pool 에 GT 1 개 추가.
                    # OL 이 2Hz coarse 면 GT 후보도 2Hz tokenized GT 로 일관 매칭.
                    # OL 이 10Hz fine 이면 GT 후보는 기존대로 raw 10Hz GT.
                    if _nearest_include_gt:
                        if ol_is_2hz:
                            _T_gt_inc = pred_max_steps_raw if pred_max_steps_raw > 0 else 4
                            _gt_start_inc = anchor_idx + 1
                            _gt_end_inc = _gt_start_inc + _T_gt_inc
                            _gt_pos_inc  = tokenized_agent["gt_pos"][active_mask, _gt_start_inc:_gt_end_inc, :]
                            _gt_head_inc = tokenized_agent["gt_heading"][active_mask, _gt_start_inc:_gt_end_inc]
                            _gt_valid_inc = tokenized_agent["valid_mask"][active_mask, _gt_start_inc:_gt_end_inc]
                        else:
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
                    # PWIL sequential: MMD 2-pass 트릭과 동형.
                    #   Pass 1 (no_grad): G CL rollouts → cl_norms_det → coupling γ 계산 (고정).
                    #   Pass 2 (with grad): rollout g 별 row 거리 ↔ γ[:,g,:] contribution → backward.
                    # γ 가 θ 와 독립이라 row 별 contribution 합 = full ∂L/∂θ (exact).
                    _do_seq_pwil = _is_pwil and G >= 2
                    cl_norms_det: list[Tensor] = []
                    sigma_sq_seq: Tensor | None = None
                    gamma_seq: Tensor | None = None
                    ol_sliced_det_pw: list[Tensor] = []

                    if _do_seq_mmd or _do_seq_pwil:
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
                                    if ol_is_2hz:
                                        _T_ol_2hz = pred_max_steps_raw if pred_max_steps_raw > 0 else 4
                                        _xy_d, _hd_d = _cl_downsample_to_2hz(
                                            _traj_d[active_mask, 0, :_T_d, :],
                                            _head_d[active_mask, 0, :_T_d],
                                            _T_ol_2hz,
                                        )
                                        _cl_norm_det = _cl_to_norm(_xy_d, _hd_d, current_pos_active, current_head_active)
                                    else:
                                        _cl_norm_det = _cl_to_norm(
                                            _traj_d[active_mask, 0, :_T_d, :],
                                            _head_d[active_mask, 0, :_T_d],
                                            current_pos_active, current_head_active,
                                        )
                                    _cl_norm_det = _ol_slice_fn(_cl_norm_det)
                                cl_norms_det.append(_cl_norm_det)
                                del _traj_d, _head_d

                        if _do_seq_pwil:
                            # ol_norms 이미 _ol_slice_fn 적용 전; detach + slice → coupling.
                            ol_sliced_det_pw = [_ol_slice_fn(o.detach()) for o in ol_norms]
                            with torch.no_grad():
                                _d_seq = pwil_pairwise_distance(
                                    cl_norms_det, ol_sliced_det_pw,
                                    pos_weight=pos_w, heading_weight=heading_w,
                                )  # [N_active, G, M_ol]
                                if pwil_coupling == "hungarian":
                                    gamma_seq = pwil_hungarian_coupling(_d_seq)
                                elif pwil_coupling == "greedy":
                                    gamma_seq = pwil_greedy_coupling(_d_seq)
                                else:
                                    gamma_seq = pwil_uniform_coupling(_d_seq)
                                _pw_transport_seq = (_d_seq * gamma_seq).sum(dim=(-2, -1)).mean().item()
                                # 표준 entropy (gamma 가 이미 anchor 별 합=1 분포):
                                #   hungarian = log(G), uniform = log(G·M), greedy = 중간.
                                _pw_entropy_seq = -(gamma_seq * torch.log(gamma_seq.clamp(min=1e-12))).sum(dim=(-2, -1)).mean().item()
                                # actual training loss (raw transport 또는 bounded reward 변환).
                                _pw_loss_seq = pwil_loss(
                                    d=_d_seq, gamma=gamma_seq,
                                    use_exp_reward=pwil_use_exp_reward,
                                    alpha=pwil_alpha, beta=pwil_beta,
                                ).item()
                            total_loss_accum += _pw_loss_seq
                            if not hasattr(self, "_pwil_log_accum"):
                                self._pwil_log_accum = {"transport": 0.0, "entropy": 0.0, "count": 0}
                            self._pwil_log_accum["transport"] += _pw_transport_seq
                            self._pwil_log_accum["entropy"] += _pw_entropy_seq
                            self._pwil_log_accum["count"] += 1
                            del _d_seq
                        elif use_gt_target:
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
                            _ol_det = [_ol_slice_fn(o.detach()) for o in ol_norms]
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
                            if ol_is_2hz:
                                _T_ol_2hz = pred_max_steps_raw if pred_max_steps_raw > 0 else 4
                                _xy_2hz, _hd_2hz = _cl_downsample_to_2hz(cl_xy_g, cl_head_g, _T_ol_2hz)
                                cl_norm_g = _cl_to_norm(_xy_2hz, _hd_2hz, current_pos_active, current_head_active)
                            else:
                                cl_norm_g = _cl_to_norm(cl_xy_g, cl_head_g, current_pos_active, current_head_active)
                            cl_norm_g = _ol_slice_fn(cl_norm_g)

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
                        elif _do_seq_pwil and gamma_seq is not None:
                            # row 거리 (grad through cl_norm_g) ↔ γ[:, g, :] (constant) contribution.
                            # contribution = (per-CL loss g)/(n_active · G); g 합 = full pwil_loss.
                            _d_row_g = pwil_row_distance(
                                cl_norm_g, ol_sliced_det_pw,
                                pos_weight=pos_w, heading_weight=heading_w,
                            )  # [N_active, M_ol]
                            _gamma_row_g = gamma_seq[:, g, :]  # [N_active, M_ol]
                            _n_active_g = int(_d_row_g.shape[0])
                            loss_g = pwil_loss_per_cl_row(
                                d_row=_d_row_g, gamma_row=_gamma_row_g,
                                use_exp_reward=pwil_use_exp_reward,
                                alpha=pwil_alpha, beta=pwil_beta,
                                G=G, n_active=_n_active_g,
                            )
                            (loss_g / n_anchors_total).backward()
                            del _d_row_g, _gamma_row_g, loss_g
                        elif use_gt_target:
                            loss_g = _consistency_loss_gt(cl_norm_g, _gt_slice_pass2, _gt_valid_slice)
                            total_loss_accum += loss_g.item()
                            (loss_g / (n_anchors_total * G)).backward()
                            del loss_g
                        else:
                            _ol_idx = 0 if _shared_ol else g
                            loss_g = _consistency_loss(
                                cl_norm_g,
                                _ol_slice_fn(ol_norms[_ol_idx]),
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
                    # CL 다운샘플 필요 조건:
                    #   - GT 분기 + 2Hz GT (use_gt_target=True, gt_is_10hz=False)
                    #   - OL 분기 + 2Hz OL  (use_gt_target=False, ol_is_2hz=True)
                    # 그 외 (GT-10Hz, OL-10Hz) 는 CL 도 native fine.
                    _need_2hz_cl = (use_gt_target and not gt_is_10hz) or (
                        (not use_gt_target) and ol_is_2hz
                    )
                    _T_cl_2hz = _T_gt if use_gt_target else (
                        pred_max_steps_raw if pred_max_steps_raw > 0 else 4
                    )
                    for g in range(G):
                        if _need_2hz_cl:
                            _xy_2hz, _hd_2hz = _cl_downsample_to_2hz(
                                pred_traj_all[active_mask, g, :T_cl, :],
                                pred_head_all[active_mask, g, :T_cl],
                                _T_cl_2hz,
                            )
                            cl_norms.append(_cl_to_norm(_xy_2hz, _hd_2hz, current_pos_active, current_head_active))
                        else:
                            # OL-10Hz 또는 GT-10Hz mode: CL 은 native fine 10Hz
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
                        T_min = min(cl_norms[0].shape[-2], ol_norms[0].shape[-2])
                        cl_stack = torch.stack(cl_norms, dim=0)[:, :, :T_min, :]
                        ol_stack = torch.stack(ol_norms, dim=0)[:, :, :T_min, :].detach()
                        cl_stack = _ol_slice_fn(cl_stack)
                        ol_stack = _ol_slice_fn(ol_stack)
                        anchor_loss = mmd_from_stacked(
                            cl_stack, ol_stack,
                            pos_weight=pos_w, heading_weight=heading_w,
                        )
                    elif _is_pwil:
                        # PWIL coupling: γ (no_grad) 로 valid transport plan 구성 후
                        #   L = Σ γ[i,j] · d(CL_i, OL_j) (mean over anchor).
                        # Hungarian (G=M): exact W_1. greedy: PWIL faithful, G≠M 허용.
                        # bounded reward 변환 (use_exp_reward): 원논문 r=α exp(-β c) 형태.
                        cl_sliced_pw = [_ol_slice_fn(cl_norms[g]) for g in range(G)]
                        ol_sliced_pw = [_ol_slice_fn(ol_norms[m].detach()) for m in range(M_ol)]
                        d_mat_pw = pwil_pairwise_distance(
                            cl_sliced_pw, ol_sliced_pw,
                            pos_weight=pos_w, heading_weight=heading_w,
                        )  # [N_active, G, M_ol]
                        with torch.no_grad():
                            if pwil_coupling == "hungarian":
                                gamma_pw = pwil_hungarian_coupling(d_mat_pw)
                            elif pwil_coupling == "greedy":
                                gamma_pw = pwil_greedy_coupling(d_mat_pw)
                            else:  # "uniform"
                                gamma_pw = pwil_uniform_coupling(d_mat_pw)
                        anchor_loss = pwil_loss(
                            d=d_mat_pw, gamma=gamma_pw,
                            use_exp_reward=pwil_use_exp_reward,
                            alpha=pwil_alpha, beta=pwil_beta,
                        )
                        # 진단 누적 (anchor 평균; final log 에서 epoch 평균).
                        # gamma 는 이미 anchor 별 합=1 인 distribution → 표준 entropy 직접 계산:
                        #   hungarian = log(G), uniform = log(G·M), greedy = 중간값.
                        with torch.no_grad():
                            _pw_transport = (d_mat_pw.detach() * gamma_pw).sum(dim=(-2, -1)).mean().item()
                            _pw_entropy = -(gamma_pw * torch.log(gamma_pw.clamp(min=1e-12))).sum(dim=(-2, -1)).mean().item()
                        if not hasattr(self, "_pwil_log_accum"):
                            self._pwil_log_accum = {"transport": 0.0, "entropy": 0.0, "count": 0}
                        self._pwil_log_accum["transport"] += _pw_transport
                        self._pwil_log_accum["entropy"] += _pw_entropy
                        self._pwil_log_accum["count"] += 1
                        del d_mat_pw, gamma_pw, cl_sliced_pw, ol_sliced_pw
                    else:
                        cl_sliced_pl = [_ol_slice_fn(cl_norms[g]) for g in range(G)]
                        if _nearest_match:
                            # 각 CL g 에 대해 M_ol 개 OL (+ optionally 1 GT) 중
                            # batch-flat weighted L2 거리 최소를 선택.
                            # 거리 = pos_w · Σ(Δx² + Δy²) + heading_w · Σ((Δcos)² + (Δsin)²)
                            #   — agent (N_active), timestep (T), 채널 합산.
                            # 선택 (argmin) 기준 = 학습 (loss) 기준 (pos_w, heading_w) 일치.
                            ol_sliced_pl = [_ol_slice_fn(ol_norms[m]) for m in range(M_ol)]
                            T_min_nm = min(cl_sliced_pl[0].shape[-2], ol_sliced_pl[0].shape[-2])
                            _use_gt_cand = (
                                _nearest_include_gt
                                and gt_norm_anchor_inc is not None
                                and gt_valid_anchor_inc is not None
                            )
                            if _use_gt_cand:
                                gt_sliced_nm = _ol_slice_fn(gt_norm_anchor_inc)
                                T_min_nm = min(T_min_nm, gt_sliced_nm.shape[-2])
                            with torch.no_grad():
                                cl_stk_nm = torch.stack(
                                    [c[:, :T_min_nm, :] for c in cl_sliced_pl], dim=0
                                ).detach()  # [G, N_active, T, F]
                                ol_stk_nm = torch.stack(
                                    [o[:, :T_min_nm, :] for o in ol_sliced_pl], dim=0
                                )  # [M, N_active, T, F]
                                # [G, M] weighted L2² distance: pos_w · Σpos² + heading_w · Σhead²
                                _diff_sq_ol = (cl_stk_nm.unsqueeze(1) - ol_stk_nm.unsqueeze(0)) ** 2  # [G, M, N, T, F]
                                _d2_ol_pos  = _diff_sq_ol[..., :2].sum(dim=(2, 3, 4))   # [G, M]
                                _d2_ol_head = _diff_sq_ol[..., 2:].sum(dim=(2, 3, 4))   # [G, M]
                                _d2_ol = pos_w * _d2_ol_pos + heading_w * _d2_ol_head
                                del _diff_sq_ol, _d2_ol_pos, _d2_ol_head
                                if _use_gt_cand:
                                    gt_stk = gt_sliced_nm[:, :T_min_nm, :]  # [N_active, T, F]
                                    gt_mask = gt_valid_anchor_inc[:, :T_min_nm].float().unsqueeze(-1)  # [N, T, 1]
                                    # GT 거리: invalid step 의 squared diff 는 0 (mask).
                                    # OL 과 동일하게 pos_w / heading_w 채널 분리 가중.
                                    _diff_sq_gt = (cl_stk_nm - gt_stk.unsqueeze(0)) ** 2 * gt_mask.unsqueeze(0)  # [G, N, T, F]
                                    _d2_gt_pos  = _diff_sq_gt[..., :2].sum(dim=(1, 2, 3))  # [G]
                                    _d2_gt_head = _diff_sq_gt[..., 2:].sum(dim=(1, 2, 3))  # [G]
                                    _d2_gt = (pos_w * _d2_gt_pos + heading_w * _d2_gt_head).unsqueeze(1)  # [G, 1]
                                    del _diff_sq_gt, _d2_gt_pos, _d2_gt_head
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
                                    _ol_slice_fn(ol_norms[0 if _shared_ol else g]),
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
        # PWIL 진단 (anchor 평균; coupling 의 quality 모니터링).
        # mean_transport: <d, γ> 평균 — W_1 (또는 그 upper bound) estimate.
        # coupling_entropy: log(G·M) (uniform) ~ 0 (deterministic) 사이; hungarian 은 정확히 log(G).
        if _is_pwil and hasattr(self, "_pwil_log_accum") and self._pwil_log_accum["count"] > 0:
            _cnt = max(1, int(self._pwil_log_accum["count"]))
            ret["train/pwil_mean_transport"] = torch.tensor(
                self._pwil_log_accum["transport"] / _cnt,
                dtype=torch.float32, device=agent_batch.device,
            )
            ret["train/pwil_coupling_entropy"] = torch.tensor(
                self._pwil_log_accum["entropy"] / _cnt,
                dtype=torch.float32, device=agent_batch.device,
            )
            self._pwil_log_accum = {"transport": 0.0, "entropy": 0.0, "count": 0}

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

    # ─────────────────────────────────────────────────────────────────────────
    # RoaD (Rollouts as Demonstrations) CL-SFT baseline
    # ─────────────────────────────────────────────────────────────────────────

    def _run_flow_road_ft_step(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        data: dict | None = None,
    ) -> dict:
        """RoaD closed-loop SFT fine-tuning step (OCSC 비교용 baseline).

        알고리즘 (RoaD, NVIDIA 2025, traffic-sim 세팅):
          1. Expert-guided closed-loop rollout (no_grad): 매 coarse step 마다 정책에서
             K 개 후보 trajectory 를 i.i.d. 샘플 → GT continuation 에 weighted
             step-wise L2 (Eq.6) 가 최소인 후보를 선택해 commit (Sample-K, Eq.4-5).
          2. BC loss: 선택된 후보를 clean target 으로 flow-matching loss.  RoaD loss
             는 -log π(a_t|o_<t) 이므로 conditioning (anchor_hidden) 과 target 은
             모두 detach — rollout 자체에는 gradient 가 흐르지 않습니다 (BPTT 없음).
          3. HardRMM 모니터링 (optional): free-running closed-loop 8초 rollout.

        OCSC 와의 차이: OCSC 는 CL/OL 분포 간 consistency loss, RoaD 는 GT-selected
        rollout 에 대한 단순 behavior cloning.  diversity 비교용으로 ``train/road_
        candidate_var`` (K 후보의 분산) 를 함께 로깅합니다.
        """
        cfg = self.finetune_config
        K = int(getattr(cfg, "road_sample_k", 64))
        G = max(1, int(getattr(cfg, "road_n_rollouts", 1)))
        pred_max_steps = int(getattr(cfg, "road_pred_max_steps", 16))
        temperature = float(getattr(cfg, "road_temperature", 0.8))
        pos_w = float(getattr(cfg, "road_position_weight", 1.0))
        heading_w = float(getattr(cfg, "road_heading_weight", 0.1))
        cmp_h = int(getattr(cfg, "road_comparison_horizon", 20))
        strict = bool(getattr(cfg, "road_strict_active_mask", True))
        eval_hard_rmm = bool(getattr(cfg, "road_eval_hard_rmm", False))
        eval_hard_rmm_interval = max(1, int(getattr(cfg, "road_eval_hard_rmm_interval", 10)))

        if data is None:
            raise ValueError("road_ft requires `data` dict with scenario metadata.")
        if "scenario_id" not in data:
            raise KeyError("road_ft requires data['scenario_id'].")

        agent_batch = tokenized_agent["batch"]
        device = agent_batch.device

        # ── 1. Encode map + rollout cache (no_grad; encoder/trunk frozen) ────
        with torch.no_grad():
            map_feature = self.encoder.encode_map(tokenized_map)
            rollout_cache = self.encoder.agent_encoder.prepare_inference_cache(
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
            )

        _agent_enc = self.encoder.agent_encoder
        flow_ode = _agent_enc.flow_ode
        flow_decoder = _agent_enc.flow_decoder

        gt_pos_10hz = data["agent"]["position"]
        gt_head_10hz = data["agent"]["heading"]
        gt_valid_10hz = data["agent"]["valid_mask"]

        # ── 2. Expert-guided rollouts → (anchor_hidden, chosen_x1) 수집 ──────
        bc_hidden: list[Tensor] = []
        bc_x1: list[Tensor] = []
        _cand_var_accum = 0.0
        _winner_dist_accum = 0.0
        _global_step = int(getattr(self, "global_step", 0))
        for g in range(G):
            eg = _agent_enc.rollout_from_cache_expert_guided(
                rollout_cache=rollout_cache,
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
                gt_pos_10hz=gt_pos_10hz,
                gt_head_10hz=gt_head_10hz,
                gt_valid_10hz=gt_valid_10hz,
                sample_k=K,
                temperature=temperature,
                pos_weight=pos_w,
                heading_weight=heading_w,
                comparison_horizon=cmp_h,
                max_steps=pred_max_steps,
                strict_active_mask=strict,
                sampling_seed=_global_step * 1000 + g,
            )
            _cand_var_accum += float(eg["mean_candidate_var"])
            _winner_dist_accum += float(eg["mean_winner_gt_dist"])
            for h, x1, bcm in zip(
                eg["per_step_anchor_hidden"],
                eg["per_step_chosen_x1"],
                eg["per_step_bc_mask"],
            ):
                if h is None or h.shape[0] == 0 or not bool(bcm.any()):
                    continue
                bc_hidden.append(h[bcm].detach())
                bc_x1.append(x1[bcm].detach())

        # ── 3. Behavior-cloning flow-matching loss ──────────────────────────
        # conditioning(anchor_hidden) / target(chosen_x1) 모두 detach 상태.
        # gradient 는 flow_decoder forward 에서만 발생 → flow_velocity_head_only
        # 적용 시 velocity_head 만 학습.
        bc_loss_accum = 0.0
        if len(bc_hidden) > 0:
            all_h = torch.cat(bc_hidden, dim=0).to(dtype=torch.float32)    # [N, D]
            all_x1 = torch.cat(bc_x1, dim=0).to(dtype=torch.float32)       # [N, 20, 4]
            N = int(all_h.shape[0])
            _chunk = 8192
            for start in range(0, N, _chunk):
                h_c = all_h[start:start + _chunk]
                x1_c = all_x1[start:start + _chunk]
                fm_sample = flow_ode.sample(x1_c, target_type="velocity")
                pred = flow_decoder(h_c, fm_sample.x_t, fm_sample.tau)
                fm = flow_matching_loss(pred, fm_sample.target)
                if not torch.isfinite(fm).all():
                    log.warning("[road_ft] non-finite BC FM loss chunk; skipping")
                    continue
                _w = float(h_c.shape[0]) / float(N)
                bc_loss_accum += fm.item() * _w
                (fm * _w).backward()

        log.info(
            f"[road] step={_global_step} bc_loss={bc_loss_accum:.4f} "
            f"n_terms={len(bc_hidden)} cand_var={_cand_var_accum / G:.4f} "
            f"winner_gt_dist={_winner_dist_accum / G:.4f}"
        )
        ret: dict = {
            "train/road_bc_loss": torch.tensor(bc_loss_accum, dtype=torch.float32, device=device),
            "train/road_candidate_var": torch.tensor(
                _cand_var_accum / G, dtype=torch.float32, device=device
            ),
            "train/road_winner_gt_dist": torch.tensor(
                _winner_dist_accum / G, dtype=torch.float32, device=device
            ),
            "train/loss": torch.tensor(bc_loss_accum, dtype=torch.float32, device=device),
        }

        # ── 4. Hard RMM monitoring (optional, free-running closed-loop 8초) ──
        if (
            eval_hard_rmm
            and self._ocsc_train_hard_rmm is not None
            and (_global_step % eval_hard_rmm_interval == 0)
            and "tfrecord_path" in data
        ):
            agent_ids = None
            try:
                agent_ids = data["agent"]["id"]
            except Exception:
                agent_ids = data.get("id") if isinstance(data, dict) else None
            if agent_ids is not None:
                _G_rmm = max(1, int(getattr(self, "n_rollout_closed_val", 4)))
                with torch.no_grad():
                    rmm_traj_all, rmm_z_all, rmm_head_all, _ = self._run_parallel_rollout_chunk(
                        data=data,
                        tokenized_agent=tokenized_agent,
                        map_feature=map_feature,
                        rollout_cache=rollout_cache,
                        rollout_indices=list(range(_G_rmm)),
                        return_anchor_hidden=True,
                        full_grad=False,
                        max_steps=None,
                    )
                hard_rmm_val = self._compute_ocsc_train_hard_rmm(
                    scenario_files=list(data["tfrecord_path"]),
                    agent_ids=agent_ids,
                    agent_batch=agent_batch,
                    traj_list=[rmm_traj_all[:, g] for g in range(_G_rmm)],
                    z_list=[rmm_z_all[:, g] for g in range(_G_rmm)],
                    head_list=[rmm_head_all[:, g] for g in range(_G_rmm)],
                    metric=self._ocsc_train_hard_rmm,
                )
                if hard_rmm_val is not None:
                    ret["train/hard_rmm"] = torch.tensor(
                        hard_rmm_val, dtype=torch.float32, device=device
                    )
                    log.info(f"[road] step={_global_step} hard_rmm={hard_rmm_val:.4f}")

        # DDP: 모든 trainable param을 dummy graph에 연결 (OCSC 와 동일 패턴).
        _ddp_dummy = sum(p.sum() * 0.0 for p in self.parameters() if p.requires_grad)
        ret["loss"] = _ddp_dummy
        return ret


    # ─────────────────────────────────────────────────────────────────────────
    # Self-Forcing DMD (Distribution Matching Distillation) fine-tuning
    # ─────────────────────────────────────────────────────────────────────────

    def _run_flow_dmd_ft_step(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        data: dict | None = None,
    ) -> dict:
        """Self-Forcing DMD step (anchor-sequential CL rollout + two manual_backward).

        Algorithm (per anchor):
          1. Generator CL rollout (with grad) → x_gen ∈ [n_active, T_10hz, 4]
             (anchor-local normalized [x/20, y/20, cos, sin]).
          2. Sample diffusion timestep τ ~ U(eps, 1), noise ε.
             x_τ = σ_t · ε + τ · x_gen.detach()  (score 입력은 stop_grad)
          3. DMD generator loss (real/fake 모두 stop_grad 로 score evaluation):
             pred_x0_{real,fake} = β_path · x_t + σ_t · v_{real,fake}
                                   (Self-Forcing DMD convention: predicted clean x_0)
             g = (1/β) · pred_x0_fake − pred_x0_real
             (∂L/∂x_gen = g/normalizer 가 되어 update 후 x_gen 이 real 쪽으로 이동)
             normalizer = mean|pred_x0_real.detach()| per anchor (dmd_normalize 시).
             target = (x_gen − g/normalizer).detach()
             L_gen = 0.5 · MSE(x_gen, target)  → manual_backward (opt_gen 만)
          4. Fake_score (critic) FM loss on generator's own rollout:
             sample = flow_ode.sample(x_gen.detach())
             v_fake = fake_score(active_hidden.detach(), sample.x_t, sample.tau)
             L_fake = flow_matching_loss(v_fake, sample.target)  → manual_backward (opt_fake 만)

        Hyperparams (FinetuneConfig):
          dmd_beta, dmd_n_rollouts (G), dmd_pred_max_steps, dmd_use_real_score,
          dmd_normalize, dmd_anchor_stride, dmd_strict_active_mask,
          dmd_warmup_fake_only_steps, dmd_gen_grad_clip, dmd_fake_lr_scale.
          (train-time HardRMM 모니터링 미지원 — validation 의 RMM 만 사용.)
          BPTT 토글 (bptt_use_adjoint, bptt_last_n_solver_steps,
          bptt_grad_clip_traj, bptt_warm_coarse_steps, bptt_last_coarse_only)
          은 OCSC 와 동일하게 적용.
        """
        # ── Hyperparams ──────────────────────────────────────────────────────
        G = max(1, int(getattr(self.finetune_config, "dmd_n_rollouts", 1)))
        pred_max_steps_raw = int(getattr(self.finetune_config, "dmd_pred_max_steps", 2))
        pred_max_steps: int | None = pred_max_steps_raw if pred_max_steps_raw > 0 else None
        beta = float(getattr(self.finetune_config, "dmd_beta", 1.0))
        if beta <= 0.0:
            raise ValueError(f"dmd_beta must be > 0, got {beta}")
        inv_beta = 1.0 / beta
        use_real = bool(getattr(self.finetune_config, "dmd_use_real_score", True))
        use_normalize = bool(getattr(self.finetune_config, "dmd_normalize", True))
        strict_active = bool(getattr(self.finetune_config, "dmd_strict_active_mask", True))
        anchor_stride = max(1, int(getattr(self.finetune_config, "dmd_anchor_stride", 1)))
        warmup_fake_only = int(getattr(self.finetune_config, "dmd_warmup_fake_only_steps", 0))
        use_adjoint = bool(getattr(self.finetune_config, "bptt_use_adjoint", False))
        warm_coarse = int(getattr(self.finetune_config, "bptt_warm_coarse_steps", 0))
        last_coarse_only = bool(getattr(self.finetune_config, "bptt_last_coarse_only", False))
        if last_coarse_only and pred_max_steps is not None and pred_max_steps > 1:
            warm_coarse = pred_max_steps - 1
        grad_clip = float(getattr(self.finetune_config, "bptt_grad_clip_traj", 1.0))
        last_n_solver = int(getattr(self.finetune_config, "bptt_last_n_solver_steps", 0))
        _shift = int(getattr(self.encoder.agent_encoder, "shift", 5))

        # score networks (flow_decoder) 의 noisy_future_encoder 가 T=num_steps hardcode.
        # pred_max_steps × shift 가 num_steps 와 일치해야 score forward 가능.
        _flow_decoder = self.encoder.agent_encoder.flow_decoder
        _expected_T = int(
            getattr(_flow_decoder, "num_future_steps", None)
            or getattr(getattr(_flow_decoder, "noisy_future_encoder", None), "num_steps", 20)
        )
        _actual_T = (pred_max_steps if pred_max_steps is not None else 0) * _shift
        if _actual_T != _expected_T:
            raise ValueError(
                f"self_forcing_dmd: dmd_pred_max_steps × shift ({_actual_T}) must equal "
                f"flow_decoder.noisy_future_encoder.num_steps ({_expected_T}).  "
                f"With shift={_shift}, set dmd_pred_max_steps={_expected_T // _shift}."
            )

        # ── Validation ──────────────────────────────────────────────────────
        if self.fake_score_decoder is None:
            raise RuntimeError("self_forcing_dmd: fake_score_decoder is None (expected from __init__).")
        if use_real and self.ref_flow_decoder is None:
            raise RuntimeError(
                "self_forcing_dmd: dmd_use_real_score=True but ref_flow_decoder is None."
            )

        agent_batch = tokenized_agent["batch"]
        device = agent_batch.device
        # warmup 은 "N training batches" 의미.  Lightning manual optimization 에서
        # self.global_step 은 LightningOptimizer.step() 호출 횟수의 합이라 두 optimizer
        # 사용 시 batch 보다 빠르게 (≈ 1.2×) 증가 → "warmup=N global_steps" 가 의도보다
        # 일찍 끝남.  _batches_that_stepped 는 실제 training batch 카운터.
        _batch_step = int(
            getattr(self.trainer.fit_loop.epoch_loop, "_batches_that_stepped", 0)
        )
        skip_gen = (_batch_step < warmup_fake_only)  # 초기엔 fake_score 만 학습

        # ── 1. Encode map + full rollout cache (no_grad; encoder/trunk frozen) ─
        with torch.no_grad():
            map_feature = self.encoder.encode_map(tokenized_map)
            rollout_cache = self.encoder.agent_encoder.prepare_inference_cache(
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
            )

        _agent_enc = self.encoder.agent_encoder
        flow_ode = _agent_enc.flow_ode
        flow_decoder = _agent_enc.flow_decoder
        fake_score = self.fake_score_decoder
        ref_score = self.ref_flow_decoder

        # Apply BPTT toggles to FlowODE.
        flow_ode.use_adjoint_for_bptt = use_adjoint
        flow_ode.last_n_grad_solver_steps = (
            min(last_n_solver, flow_ode.solver_steps) if last_n_solver > 0 else 0
        )

        # ── Anchor index selection (strided over GT 2Hz timeline) ────────────
        step_current_2hz = int(rollout_cache["valid_window"].shape[1])
        total_2hz_steps = int(tokenized_agent["gt_pos"].shape[1])
        pred_steps = pred_max_steps_raw if pred_max_steps_raw > 0 else 2
        valid_anchor_end = max(1, total_2hz_steps - pred_steps)
        all_anchor_indices = list(range(0, valid_anchor_end, anchor_stride))
        n_anchors_total = max(1, len(all_anchor_indices))

        # ── grad-clip hook helper (CL trajectory level) ──────────────────────
        def _make_norm_clip_hook(max_norm: float):
            def _hook(g: Tensor) -> Tensor:
                g = torch.nan_to_num(g, nan=0.0, posinf=max_norm, neginf=-max_norm)
                g_norm = g.norm()
                if g_norm > max_norm:
                    g = g * (max_norm / g_norm)
                return g
            return _hook

        # ── world → anchor-local normalized helper ───────────────────────────
        def _cl_to_norm(cl_xy, cl_head, current_pos_active, current_head_active):
            return self._world_traj_to_flow_norm(
                pred_traj=cl_xy,
                pred_head=cl_head,
                current_pos=current_pos_active,
                current_head=current_head_active,
            )

        # ── Logging accumulators ─────────────────────────────────────────────
        gen_loss_accum = 0.0
        fake_loss_accum = 0.0
        score_diff_norm_accum = 0.0
        v_real_norm_accum = 0.0
        v_fake_norm_accum = 0.0
        normalizer_mean_accum = 0.0
        n_valid_anchors = 0
        n_dmd_terms = 0  # generator term count (skip_gen 시 0)

        _seq_keys = {"gt_pos", "gt_heading", "valid_mask", "gt_idx"}

        # ── Anchor sequential loop ───────────────────────────────────────────
        for anchor_idx in all_anchor_indices:
            # 3a. Build anchor tokenized_agent (slice views).
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

            # 3b. Build anchor rollout_cache (no_grad).
            with torch.no_grad():
                rollout_cache_anchor = _agent_enc.prepare_inference_cache(
                    tokenized_agent=tokenized_agent_anchor,
                    map_feature=map_feature,
                )
            active_mask = rollout_cache_anchor["valid_window"][:, -1]
            if strict_active:
                _T_future_raw = pred_steps * _shift
                _anchor_now_10hz = (anchor_idx + 1) * _shift
                _future_start = _anchor_now_10hz + 1
                _future_end = _future_start + _T_future_raw
                _raw_valid_full = data["agent"]["valid_mask"]
                _seq_len = _raw_valid_full.shape[1]
                if _future_end <= _seq_len and _T_future_raw > 0:
                    _future_valid = _raw_valid_full[:, _future_start:_future_end].all(dim=1)
                else:
                    _future_valid = torch.zeros(
                        _raw_valid_full.shape[0], dtype=torch.bool, device=_raw_valid_full.device,
                    )
                active_mask = active_mask & _future_valid
            if not bool(active_mask.any()):
                del rollout_cache_anchor
                continue

            current_pos_active = rollout_cache_anchor["pos_window"][:, -1][active_mask]
            current_head_active = rollout_cache_anchor["head_window"][:, -1][active_mask]
            active_hidden = rollout_cache_anchor["feat_a_now"][active_mask]
            n_valid_anchors += 1

            # 3c. CL rollout — G parallel (OCSC paired L2 패턴 차용).
            # G 개의 rollout 을 batch dim 으로 묶어 ODE solver 1 회 호출.
            # output shape:  pred_traj_all [B, G, T_10hz, 2], pred_head_all [B, G, T_10hz]
            pred_traj_all, _pred_z_all, pred_head_all, _ = self._run_parallel_rollout_chunk(
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
            if pred_traj_all.requires_grad and grad_clip > 0:
                pred_traj_all.register_hook(_make_norm_clip_hook(grad_clip))
            if pred_head_all.requires_grad and grad_clip > 0:
                pred_head_all.register_hook(_make_norm_clip_hook(grad_clip))

            _T = pred_traj_all.shape[-2]
            # active_mask 선적용 후 G 를 batch dim 으로 flatten: [n_active*G, T, ...].
            _traj_act = pred_traj_all[active_mask]              # [n_active, G, T, 2]
            _head_act = pred_head_all[active_mask]              # [n_active, G, T]
            n_active = _traj_act.shape[0]
            _traj_flat = _traj_act.reshape(n_active * G, _T, 2)
            _head_flat = _head_act.reshape(n_active * G, _T)
            _pos_rep = current_pos_active.repeat_interleave(G, dim=0)        # [n_active*G, 2]
            _head_rep = current_head_active.repeat_interleave(G, dim=0)      # [n_active*G]
            x_gen = _cl_to_norm(
                _traj_flat, _head_flat, _pos_rep, _head_rep,
            ).to(dtype=torch.float32)                                        # [n_active*G, T, 4]

            # 3d. Sample diffusion timestep + noise (FlowODE 의 path 수식 그대로).
            # τ, noise 는 n_active*G 차원에서 독립 sample (G 개의 rollout 마다 다른 τ).
            with torch.no_grad():
                x_gen_d = x_gen.detach()
                tau = torch.rand(
                    x_gen_d.shape[0], device=x_gen_d.device, dtype=x_gen_d.dtype
                ) * (1.0 - flow_ode.eps) + flow_ode.eps                # [n_active*G]
                noise = torch.randn_like(x_gen_d)
                view_tau = tau.view(-1, 1, 1)
                view_sigma = flow_ode._sigma_t(tau).view(-1, 1, 1)
                x_t = view_sigma * noise + view_tau * x_gen_d            # [n_active*G, T, 4]
                # cond 도 G 만큼 expand: [n_active, hidden] → [n_active*G, hidden].
                cond_d = active_hidden.detach().to(dtype=torch.float32).repeat_interleave(G, dim=0)

                if use_real and ref_score is not None:
                    v_real = ref_score(cond_d, x_t, tau).to(dtype=torch.float32)
                else:
                    v_real = torch.zeros_like(x_t)
                v_fake_eval = fake_score(cond_d, x_t, tau).to(dtype=torch.float32)

                # Convert velocity → predicted clean x_0 (Self-Forcing DMD convention).
                #   pred_x0 = β_path · x_t + σ_t · v   (FlowODE.predict_clean_from_velocity)
                beta_path = flow_ode._beta()
                pred_x0_real = beta_path * x_t + view_sigma * v_real
                pred_x0_fake = beta_path * x_t + view_sigma * v_fake_eval

                # DMD synthetic gradient (entropy-weighted Self-Forcing form).
                #   g = (1/β) · pred_x0_fake − pred_x0_real
                g_dmd = inv_beta * pred_x0_fake - pred_x0_real          # [n_active*G, T, 4]
                if use_normalize:
                    # Self-Forcing 원본: abs(p_real).mean(spatial).  G 마다 독립 normalizer.
                    normalizer = pred_x0_real.abs().mean(
                        dim=(-2, -1), keepdim=True
                    ).clamp_min(1e-7)
                    g_n = g_dmd / normalizer
                    normalizer_mean_accum += float(normalizer.mean().item())
                else:
                    g_n = g_dmd
                    normalizer_mean_accum += 1.0

                # logging stats (no_grad ctx) — pred_x0 space 차이 (semantic 일관).
                _score_diff = (pred_x0_real - pred_x0_fake).abs().mean().item()
                score_diff_norm_accum += float(_score_diff)
                v_real_norm_accum += float(pred_x0_real.abs().mean().item())
                v_fake_norm_accum += float(pred_x0_fake.abs().mean().item())

            # 3e. DMD generator loss (skip during warmup).
            # MSE 는 n_active*G 전 elements 평균 → G 차원도 자동 평균.
            if not skip_gen:
                target = (x_gen - g_n).detach()
                L_gen = 0.5 * F.mse_loss(x_gen, target, reduction="mean")
                L_gen_scaled = L_gen / float(n_anchors_total)
                if torch.isfinite(L_gen_scaled).all():
                    self.manual_backward(L_gen_scaled)
                    gen_loss_accum += float(L_gen.item())
                    n_dmd_terms += 1
                else:
                    log.warning(f"[dmd] non-finite L_gen at anchor={anchor_idx}; skipping")

            # 3f. Fake_score (critic) FM loss on generator's own sample.
            # x_gen detach + cond detach → grad 는 fake_score params 로만.
            sample_fk = flow_ode.sample(x_gen.detach().to(dtype=torch.float32), target_type="velocity")
            v_fk = fake_score(
                cond_d,           # 이미 [n_active*G, hidden], detach 상태.
                sample_fk.x_t,
                sample_fk.tau,
            )
            L_fake = flow_matching_loss(v_fk, sample_fk.target)
            L_fake_scaled = L_fake / float(n_anchors_total)
            if torch.isfinite(L_fake_scaled).all():
                self.manual_backward(L_fake_scaled)
                fake_loss_accum += float(L_fake.item())
            else:
                log.warning(f"[dmd] non-finite L_fake at anchor={anchor_idx}; skipping")

            # cleanup per-anchor intermediates
            del pred_traj_all, pred_head_all, _traj_act, _head_act
            del _traj_flat, _head_flat, _pos_rep, _head_rep
            del x_gen, x_gen_d, x_t, v_fake_eval
            if use_real:
                del v_real
            del rollout_cache_anchor

        # ── Logging dict ─────────────────────────────────────────────────────
        # G parallel: anchor 당 누적이 한 번씩 (G 차원은 mean() 안에 흡수됨).
        _denom = max(1, n_valid_anchors)
        ret: dict = {
            "train/dmd/gen_loss": torch.tensor(
                gen_loss_accum / max(1, n_dmd_terms), dtype=torch.float32, device=device,
            ),
            "train/dmd/fake_loss": torch.tensor(
                fake_loss_accum / _denom, dtype=torch.float32, device=device,
            ),
            "train/dmd/score_diff_norm": torch.tensor(
                score_diff_norm_accum / _denom, dtype=torch.float32, device=device,
            ),
            "train/dmd/v_real_norm": torch.tensor(
                v_real_norm_accum / _denom, dtype=torch.float32, device=device,
            ),
            "train/dmd/v_fake_norm": torch.tensor(
                v_fake_norm_accum / _denom, dtype=torch.float32, device=device,
            ),
            "train/dmd/normalizer_mean": torch.tensor(
                normalizer_mean_accum / _denom, dtype=torch.float32, device=device,
            ),
            "train/dmd/beta": torch.tensor(beta, dtype=torch.float32, device=device),
            "train/dmd/n_valid_anchors": torch.tensor(
                float(n_valid_anchors), dtype=torch.float32, device=device,
            ),
            "train/dmd/skip_gen": torch.tensor(
                1.0 if skip_gen else 0.0, dtype=torch.float32, device=device,
            ),
            # alias for top-line dashboard
            "train/loss": torch.tensor(
                gen_loss_accum / max(1, n_dmd_terms) if not skip_gen
                else fake_loss_accum / _denom,
                dtype=torch.float32, device=device,
            ),
        }

        # (DMD: train-time HardRMM 모니터링 미지원 — validation 단계의 RMM 만 신뢰.)

        # ── DDP all-reduce dummy: 모든 trainable param (gen + fake) 을 dummy graph 에 연결.
        # OCSC 패턴 그대로 — anchor 수가 GPU 마다 달라도 deadlock 없이 1회 all-reduce.
        _ddp_dummy = sum(
            p.sum() * 0.0 for p in self.parameters() if p.requires_grad
        )
        if self.fake_score_decoder is not None:
            _ddp_dummy = _ddp_dummy + sum(
                p.sum() * 0.0 for p in self.fake_score_decoder.parameters() if p.requires_grad
            )
        ret["loss"] = _ddp_dummy
        return ret

    def training_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)

        # ── DMD: two-optimizer alternating (gen + fake_score) ────────────────
        if self._is_dmd_ft_enabled():
            opts = self.optimizers()
            # Lightning manual mode: returns list when configure_optimizers returns multiple.
            if not isinstance(opts, (list, tuple)):
                raise RuntimeError(
                    "self_forcing_dmd expects two optimizers; got single optimizer. "
                    "Check configure_optimizers."
                )
            opt_gen, opt_fake = opts[0], opts[1]
            opt_gen.zero_grad()
            opt_fake.zero_grad()

            _ddp_model = getattr(getattr(self, "trainer", None) and self.trainer.strategy, "model", None)
            _no_sync_ctx = (
                _ddp_model.no_sync()
                if _ddp_model is not None and hasattr(_ddp_model, "no_sync")
                else contextlib.nullcontext()
            )
            with _no_sync_ctx:
                diag = self._run_flow_dmd_ft_step(tokenized_map, tokenized_agent, data)

            # Final DDP all-reduce (grad 는 anchor loop 에서 이미 누적됨; dummy=0).
            if "loss" in diag:
                self.manual_backward(diag["loss"])

            for k, v in diag.items():
                if k == "loss":
                    continue
                if isinstance(v, (Tensor, float)):
                    self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)

            # Optional separate gen grad clip.  Lightning 의 self.clip_gradients 는 manual mode
            # + Trainer.gradient_clip_val 와 충돌하므로 torch.nn.utils.clip_grad_norm_ 직접 호출.
            _dmd_clip = float(getattr(self.finetune_config, "dmd_gen_grad_clip", 0.0) or 0.0)
            _bptt_clip = float(getattr(self.finetune_config, "bptt_grad_clip_traj", 0.0) or 0.0)
            _gen_clip = _dmd_clip if _dmd_clip > 0 else _bptt_clip
            if _gen_clip > 0:
                _gen_params = [p for g in opt_gen.param_groups for p in g["params"]]
                _fake_params = [p for g in opt_fake.param_groups for p in g["params"]]
                torch.nn.utils.clip_grad_norm_(_gen_params, max_norm=_gen_clip)
                torch.nn.utils.clip_grad_norm_(_fake_params, max_norm=_gen_clip)

            # ── Warmup / cadence counters (batch-based, NOT self.global_step) ─
            # self.global_step 은 manual mode 의 opt.step() 합산 → 두 opt 사용 시
            # batch 보다 빠르게 증가.  warmup 과 k:1 cadence 모두 "training batch"
            # 단위가 의미라 _batches_that_stepped 를 reference 로.
            _batch_step = int(
                getattr(self.trainer.fit_loop.epoch_loop, "_batches_that_stepped", 0)
            )
            _warmup_steps = int(getattr(self.finetune_config, "dmd_warmup_fake_only_steps", 0))
            _skip_gen = (_batch_step < _warmup_steps)

            # ── k:1 cadence (Self-Forcing dfake_gen_update_ratio).
            # critic 매 batch update, generator 매 k batch (default 1 — full alternating).
            gen_update_ratio = max(1, int(getattr(self.finetune_config, "dmd_gen_update_ratio", 1)))
            _do_gen_step = (_batch_step % gen_update_ratio == 0)

            # warmup 중에는 L_gen 자체가 _run_flow_dmd_ft_step 에서 backward 안 됨 →
            # opt_gen.step() 호출은 grad=None 으로 노옵이지만, 의도 명시 차원에서 가드.
            if (not _skip_gen) and _do_gen_step:
                opt_gen.step()
            opt_fake.step()

            # ── EMA on generator (instantaneous gen step 후 EMA update; validation 시 swap).
            # warmup 중엔 generator 안 변하므로 EMA 도 자가 = 자가 (무해)이지만 명시 가드.
            _ema_start = int(getattr(self.finetune_config, "dmd_ema_start_step", 0))
            if (
                self.gen_ema is not None
                and (not _skip_gen)
                and _do_gen_step
                and _batch_step >= _ema_start
            ):
                _ema_w = float(getattr(self.finetune_config, "dmd_ema_weight", 0.0))
                if _ema_w > 0.0:
                    with torch.no_grad():
                        cur_fd = self.encoder.agent_encoder.flow_decoder
                        for p_ema, p_cur in zip(self.gen_ema.parameters(), cur_fd.parameters()):
                            p_ema.mul_(_ema_w).add_(p_cur.detach(), alpha=(1.0 - _ema_w))
                        for b_ema, b_cur in zip(self.gen_ema.buffers(), cur_fd.buffers()):
                            b_ema.copy_(b_cur)
            return

        opt = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()

        if self._is_ocsc_ft_enabled():
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

        elif self._is_road_ft_enabled():
            # RoaD: OCSC 와 동일하게 step 내부에서 .backward() 누적 후 dummy 로 all-reduce 1회.
            _ddp_model = getattr(getattr(self, "trainer", None) and self.trainer.strategy, "model", None)
            _no_sync_ctx = (
                _ddp_model.no_sync()
                if _ddp_model is not None and hasattr(_ddp_model, "no_sync")
                else contextlib.nullcontext()
            )
            with _no_sync_ctx:
                diag = self._run_flow_road_ft_step(tokenized_map, tokenized_agent, data)

            if "loss" in diag:
                self.manual_backward(diag["loss"])

            for k, v in diag.items():
                if k == "loss":
                    continue
                if isinstance(v, (Tensor, float)):
                    self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)

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

        # OL target 용 ref_flow_decoder 갱신 (frozen / periodic / ema).
        self._maybe_refresh_ref_decoder()

    def _maybe_refresh_ref_decoder(self) -> None:
        """OL target 생성용 ``ref_flow_decoder`` 를 설정에 따라 갱신합니다.

        ``ocsc_ref_refresh_mode``:
          - ``"frozen"`` (기본): 갱신하지 않습니다 (학습 시작 시점 가중치 고정).
          - ``"periodic"``: ``ocsc_ref_refresh_interval`` step 마다 현재
            flow_decoder 의 state_dict 로 hard copy 합니다.
          - ``"ema"``: 매 step ``ref = decay·ref + (1-decay)·current`` 로
            지수이동평균 갱신합니다 (mean-teacher 방식).
        """
        if self.ref_flow_decoder is None or not self._is_ocsc_ft_enabled():
            return
        mode = str(getattr(self.finetune_config, "ocsc_ref_refresh_mode", "frozen")).lower()
        if mode == "frozen":
            return
        cur = self.encoder.agent_encoder.flow_decoder
        if mode == "periodic":
            interval = int(getattr(self.finetune_config, "ocsc_ref_refresh_interval", 0))
            if interval <= 0:
                return
            gstep = int(getattr(self, "global_step", 0))
            if gstep > 0 and gstep % interval == 0:
                self.ref_flow_decoder.load_state_dict(cur.state_dict())
                log.info(f"[ocsc] step={gstep} ref_flow_decoder hard-refreshed (periodic).")
        elif mode == "ema":
            decay = float(getattr(self.finetune_config, "ocsc_ref_ema_decay", 0.999))
            with torch.no_grad():
                for rp, cp in zip(self.ref_flow_decoder.parameters(), cur.parameters()):
                    rp.mul_(decay).add_(cp.detach(), alpha=1.0 - decay)
                for rb, cb in zip(self.ref_flow_decoder.buffers(), cur.buffers()):
                    rb.copy_(cb)
        else:
            log.warning(
                f"[ocsc] unknown ocsc_ref_refresh_mode={mode!r}; treating as 'frozen'."
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

    def on_validation_start(self) -> None:
        """DMD: validation 직전에 generator 의 instantaneous weight 를 EMA 로 swap.

        gen_ema.state_dict() 를 main flow_decoder 에 in-place load_state_dict 로 적용.
        instantaneous state 는 self._gen_ema_swap_backup 에 보관.  on_validation_end 에서 복원.
        """
        if (
            self._is_dmd_ft_enabled()
            and self.gen_ema is not None
            and float(getattr(self.finetune_config, "dmd_ema_weight", 0.0)) > 0.0
        ):
            cur_fd = self.encoder.agent_encoder.flow_decoder
            # Backup instantaneous state (clone to avoid aliasing).
            self._gen_ema_swap_backup = {
                k: v.detach().clone() for k, v in cur_fd.state_dict().items()
            }
            cur_fd.load_state_dict(self.gen_ema.state_dict(), strict=True)
            log.info(f"[dmd] on_validation_start: swapped flow_decoder → gen_ema weights.")

    def on_validation_end(self) -> None:
        """DMD: validation 후 instantaneous weight 복원."""
        if (
            self._is_dmd_ft_enabled()
            and self._gen_ema_swap_backup is not None
        ):
            cur_fd = self.encoder.agent_encoder.flow_decoder
            cur_fd.load_state_dict(self._gen_ema_swap_backup, strict=True)
            self._gen_ema_swap_backup = None
            log.info(f"[dmd] on_validation_end: restored flow_decoder ← instantaneous weights.")

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
            update_scenario_diversity_metric_from_model(
                model=self,
                data=data,
                pred_traj=pred_traj,
                pred_head=pred_head,
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

    def on_validation_epoch_end(self):
        log_and_reset_wosac_distribution_metric(
            model=self,
            metric=self.wosac_distribution_metrics,
        )
        log_and_reset_scenario_diversity_metric(
            model=self,
            metric=self.scenario_diversity_metrics,
        )
        if self.val_open_loop:
            epoch_open_metrics = self._compute_and_reset_validation_metrics(
                prefix="val_open",
                metric_store=self.val_open_epoch_metrics,
            )
            for metric_name, metric_value in epoch_open_metrics.items():
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
                # stdout 에도 sub-metric 출력 (wandb 외부에서 grep/monitor 용).
                if self.global_rank == 0:
                    _gstep = int(getattr(self, "global_step", 0))
                    def _fmt(_v: object) -> str:
                        try:
                            return f"{float(_v):.5f}"
                        except (TypeError, ValueError):
                            return str(_v)
                    _summary = " ".join(
                        f"{_k.rsplit('/', 1)[-1]}={_fmt(_v)}"
                        for _k, _v in sorted(epoch_sim_agents_metrics.items())
                    )
                    log.info(f"[val_closed] step={_gstep} {_summary}")
                self.sim_agents_metrics.reset()
                self.minADE.reset()

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

        # ── DMD (self_forcing_dmd): generator / fake_score 두 optimizer 분리 ──────
        if self._is_dmd_ft_enabled():
            if self.fake_score_decoder is None:
                raise RuntimeError(
                    "self_forcing_dmd: fake_score_decoder is None. "
                    "Expected __init__ to deepcopy from flow_decoder."
                )
            fake_param_ids = {id(p) for p in self.fake_score_decoder.parameters()}
            gen_params = [
                p for p in self.parameters()
                if p.requires_grad and id(p) not in fake_param_ids
            ]
            fake_params = [p for p in self.fake_score_decoder.parameters() if p.requires_grad]
            if not gen_params:
                raise RuntimeError("self_forcing_dmd: no trainable generator params.")
            if not fake_params:
                raise RuntimeError("self_forcing_dmd: no trainable fake_score params.")
            fake_lr_scale = float(getattr(self.finetune_config, "dmd_fake_lr_scale", 1.0))
            adam_beta1 = float(getattr(self.finetune_config, "dmd_adam_beta1", 0.9))
            adam_beta2 = float(getattr(self.finetune_config, "dmd_adam_beta2", 0.999))
            opt_gen = torch.optim.AdamW(
                gen_params, lr=self.lr, betas=(adam_beta1, adam_beta2),
                weight_decay=self.weight_decay,
            )
            opt_fake = torch.optim.AdamW(
                fake_params, lr=self.lr * fake_lr_scale, betas=(adam_beta1, adam_beta2),
                weight_decay=self.weight_decay,
            )
            sch_gen = LambdaLR(opt_gen, lr_lambda=lr_lambda)
            sch_fake = LambdaLR(opt_fake, lr_lambda=lr_lambda)
            log.info(
                f"[self_forcing_dmd] two-optimizer setup: "
                f"opt_gen={len(gen_params)} param tensors @ lr={self.lr:.2e}; "
                f"opt_fake={len(fake_params)} param tensors @ lr={self.lr*fake_lr_scale:.2e} "
                f"(scale={fake_lr_scale}); AdamW betas=({adam_beta1}, {adam_beta2})."
            )
            return (
                {"optimizer": opt_gen, "lr_scheduler": {"scheduler": sch_gen, "interval": self.lr_scheduler_unit, "frequency": 1}},
                {"optimizer": opt_fake, "lr_scheduler": {"scheduler": sch_fake, "interval": self.lr_scheduler_unit, "frequency": 1}},
            )

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

        # fake_score_decoder / gen_ema: DMD mode 에서만 __init__ 또는 on_train_start 가
        # deepcopy 로 생성하므로 (a) DMD-saved ckpt 를 non-DMD 모드에서 load 시 unexpected 로
        # 허용, (b) non-DMD ckpt 를 DMD 모드에서 load 시 missing 으로 허용.
        _allowed_missing = ("residual_velocity_head", "fake_score_decoder", "gen_ema")
        _allowed_unexpected = ("ref_flow_decoder", "fake_score_decoder", "gen_ema")
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
