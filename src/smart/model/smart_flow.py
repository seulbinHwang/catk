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
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

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
    ade_future,
    fde_future,
    flow_matching_loss,
    yaw_ade_future,
    yaw_fde_future,
)
from src.smart.metrics.mmd_consistency_loss import (
    mmd_from_stacked,
    mmd_per_rollout_proxy,
    mmd_precompute_sigma_sq,
)
from src.smart.modules.smart_flow_decoder import SMARTFlowDecoder
from src.smart.tokens.flow_token_processor import FlowTokenProcessor
from src.smart.utils.finetune import FinetuneConfig, set_model_for_finetuning
from src.smart.utils.flow_horizon import format_flow_horizon_tag
from src.smart.utils.rollout import transform_to_local
from src.utils.pylogger import RankedLogger
from src.utils.vis_waymo import VisWaymo
from src.utils.sim_agents_utils import get_scenario_id_int_tensor, get_scenario_rollouts


log = RankedLogger(__name__, rank_zero_only=True)


class SMARTFlow(LightningModule):

    def __init__(self, model_config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = model_config.lr
        self.lr_warmup_steps = model_config.lr_warmup_steps
        self.lr_total_steps = model_config.lr_total_steps
        self.lr_min_ratio = model_config.lr_min_ratio
        self.weight_decay = float(getattr(model_config, "weight_decay", 0.0))
        # OCSC 는 step 단위 cosine schedule 을 가정 (manual_optimization 모드).
        self.lr_scheduler_unit = str(getattr(model_config, "lr_scheduler_unit", "epoch"))
        if self.lr_scheduler_unit not in {"epoch", "step"}:
            raise ValueError(f"Unsupported lr_scheduler_unit: {self.lr_scheduler_unit}")
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
        self.finetune_config: FinetuneConfig = set_model_for_finetuning(
            self.encoder, model_config.finetune
        )

        self.minADE = minADE()
        self.minADE_predict = minADE()
        # validation_metric: "real" (공식 TF SimAgentsMetrics, 정확하지만 느림)
        # "hard" (PyTorch in-process HardSimAgentsMetrics, 빠름, parity 검증됨).
        # OCSC default 는 hard.
        _validation_metric = str(
            getattr(model_config, "validation_metric", "real")
        ).lower()
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

        # OCSC step 본체는 self.eval_sampling_noise 를 DictConfig 형태로 가정
        # (`getattr(sampling_noise, "noise_scale", 1.0)` 로 접근). project_3
        # model config 에는 eval_sampling_noise 키가 없어
        # validation_rollout_sampling 자체를 fallback 으로 둔다 — 그 안에 이미
        # noise_scale 항목이 있다.
        self.eval_sampling_noise = getattr(
            model_config,
            "eval_sampling_noise",
            self.validation_rollout_sampling,
        )

        # OCSC fine-tuning 시점에만 채워지는 frozen pretrained reference decoder.
        # on_train_start 에서 self.encoder.agent_encoder.flow_decoder 를 deepcopy.
        self.ref_flow_decoder: nn.Module | None = None

        # OCSC: per-step HardRMM 모니터링용 인-프로세스 metric 객체 (current + ref)
        _is_ocsc = (
            self.finetune_config.enabled
            and self.finetune_config.mode == "ocsc_ft"
        )
        if _is_ocsc:
            # OCSC step 본체가 manual_backward 를 사용하므로 manual optimization
            # 모드로 전환한다. 일반 pretraining/inference 는 그대로 automatic.
            self.automatic_optimization = False

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
        official scorer에 넣을 수 있도록 ``n_batch_sim_agents_metric`` 을 per-rank
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
                "[scorer_scene_num] 공식 sim_agents_2025 scorer batch 수를 "
                f"n_batch_sim_agents_metric={self.n_batch_sim_agents_metric} 으로 설정합니다 "
                f"(requested_scenes={scorer_scene_num}, world_size={world_size}, "
                f"val_batch_size={val_batch_size}).",
                flush=True,
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
        rollout_encoder: SMARTFlowDecoder | None = None,
        data=None,
        tokenized_agent: Dict[str, Tensor] | None = None,
        map_feature: Dict[str, Tensor] | None = None,
        rollout_cache: Dict[str, object] | None = None,
        rollout_indices: Sequence[int] = (),
        return_flow_2s_preview: bool = False,
        # OCSC step 이 OCSC_clean 시그니처로 호출하므로 호환을 위해 받는 인자들.
        # 현재 OCSC_clean_v2 에서는 wire 되지 않고 단순 받기만 한다 (받기 무시).
        return_anchor_hidden: bool = False,
        full_grad: bool = False,
        max_steps: int | None = None,
        warm_coarse_steps: int = 0,
        share_noise_across_time: bool = False,
        noise_tape_override: Tensor | None = None,
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
        # OCSC step 이 rollout_encoder 를 안 넘기는 경우 self.encoder 로 fallback.
        if rollout_encoder is None:
            rollout_encoder = self.encoder
        chunk_size = int(len(rollout_indices))
        scenario_device = tokenized_agent["batch"].device
        if chunk_size == 1:
            scenario_sampling_seeds = self._get_closed_loop_scenario_seeds(
                scenario_ids=data["scenario_id"],
                rollout_idx=int(rollout_indices[0]),
                device=scenario_device,
            )
            if full_grad:
                pred = rollout_encoder.training_rollout_from_cache(
                    rollout_cache=rollout_cache,
                    tokenized_agent=tokenized_agent,
                    map_feature=map_feature,
                    sampling_scheme=self.validation_rollout_sampling,
                    scenario_sampling_seeds=scenario_sampling_seeds,
                    rollout_steps_2hz=max_steps,
                )
            else:
                pred = rollout_encoder.rollout_from_cache(
                    rollout_cache=rollout_cache,
                    tokenized_agent=tokenized_agent,
                    map_feature=map_feature,
                    sampling_scheme=self.validation_rollout_sampling,
                    scenario_sampling_seeds=scenario_sampling_seeds,
                    return_flow_2s_preview=return_flow_2s_preview,
                )
            flow_preview = None
            if (not full_grad) and return_flow_2s_preview:
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
        if full_grad:
            pred = rollout_encoder.training_rollout_from_cache(
                rollout_cache=expanded_rollout_cache,
                tokenized_agent=expanded_tokenized_agent,
                map_feature=expanded_map_feature,
                sampling_scheme=self.validation_rollout_sampling,
                scenario_sampling_seeds=scenario_seed_table.reshape(-1).contiguous(),
                rollout_steps_2hz=max_steps,
            )
        else:
            pred = rollout_encoder.rollout_from_cache(
                rollout_cache=expanded_rollout_cache,
                tokenized_agent=expanded_tokenized_agent,
                map_feature=expanded_map_feature,
                sampling_scheme=self.validation_rollout_sampling,
                scenario_sampling_seeds=scenario_seed_table.reshape(-1).contiguous(),
                return_flow_2s_preview=return_flow_2s_preview,
            )
        flow_preview = None
        if (not full_grad) and return_flow_2s_preview:
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

    def on_fit_start(self) -> None:
        """학습 시작 전에 빠른 closed-loop validation 모드를 켭니다.

        Lightning은 ``on_fit_start`` 를 sanity check 전에 호출합니다.
        그래서 여기서 validation batch 개수를 줄이면 학습 전 sanity check와
        학습 중 validation 둘 다 같은 빠른 규칙을 사용하게 됩니다.

        Returns:
            None
        """
        self._apply_scorer_scene_num_overrides()
        self._apply_fit_time_validation_batch_limit()

    def on_validation_start(self) -> None:
        """validation 시작 직전에 scorer batch 수 자동 조정을 다시 시도합니다."""
        self._apply_scorer_scene_num_overrides()

    def on_fit_end(self) -> None:
        """학습이 끝나면 임시로 바꾼 validation 제한 값을 정리합니다.

        Returns:
            None
        """
        self._restore_fit_time_validation_batch_limit()

    def on_train_start(self) -> None:
        """OCSC 학습 시작 시 frozen reference flow decoder + NaN guard hook 을 준비합니다."""
        if self._is_ocsc_ft_enabled() and self.ref_flow_decoder is None:
            from copy import deepcopy

            flow_decoder = self.encoder.agent_encoder.flow_decoder
            self.ref_flow_decoder = deepcopy(flow_decoder)
            for p in self.ref_flow_decoder.parameters():
                p.requires_grad_(False)
            log.info(
                f"[{self.finetune_config.mode}] frozen reference flow decoder created "
                "from current weights for OCSC delta-RMM monitoring."
            )

        # ocsc_ft: BPTT backward through ODE steps can produce NaN/Inf gradients.
        # Register nan_to_num hooks on trainable parameters so any NaN/Inf
        # gradient is sanitised in place.
        if self._is_ocsc_ft_enabled():
            n_hooked = 0
            for p in self.parameters():
                if p.requires_grad:
                    p.register_hook(
                        lambda g: torch.nan_to_num(g, nan=0.0, posinf=1e4, neginf=-1e4)
                    )
                    n_hooked += 1
            log.info(
                f"[{self.finetune_config.mode}] registered nan_to_num grad hooks "
                f"on {n_hooked} trainable params"
            )


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


    # === OCSC (Open-Closed Self-Consistency) fine-tuning ===
    # Ported verbatim from origin/OCSC_clean — only ocsc_ft mode is wired
    # in __init__/training_step (per Q4(a)). Helpers _world_traj_to_flow_norm,
    # _compute_soft_rmm, _compute_rmm_group, _compute_rmm_bptt_gt_fm_loss are
    # also used by other finetune lines on OCSC_clean side; here they exist
    # only to support ocsc_ft.

    def _is_ocsc_ft_enabled(self) -> bool:
        return bool(
            self.finetune_config.enabled
            and self.finetune_config.mode == "ocsc_ft"
        )


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

        # train 시 HardRMM 모니터링 path 는 OCSC_clean_v2 에서 제거됨
        # (사용자 지시: tfrecord_path / scenario_id / eval_hard_rmm 모두 미사용).
        if data is None:
            raise ValueError("ocsc_ft requires `data` dict.")
        agent_batch = tokenized_agent["batch"]

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
                # pred_max_steps_raw 만큼의 2Hz GT 위치를 anchor frame으로 정규화한다.
                gt_norm_anchor: Tensor | None = None
                gt_valid_anchor: Tensor | None = None
                if use_gt_target:
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
                        if use_gt_target:
                            _xy_2hz, _hd_2hz = _cl_downsample_to_2hz(
                                pred_traj_all[active_mask, g, :T_cl, :],
                                pred_head_all[active_mask, g, :T_cl],
                                _T_gt,
                            )
                            cl_norms.append(_cl_to_norm(_xy_2hz, _hd_2hz, current_pos_active, current_head_active))
                        else:
                            cl_norms.append(_cl_to_norm(
                                pred_traj_all[active_mask, g, :T_cl, :],
                                pred_head_all[active_mask, g, :T_cl],
                                current_pos_active, current_head_active,
                            ))

                    if use_gt_target:
                        _gt_slice = _slice_consistency_suffix_2hz(gt_norm_anchor)
                        _gt_valid_slice = _slice_valid_suffix_2hz(gt_valid_anchor)
                        if use_mmd and G >= 2:
                            T_min = min(cl_norms[0].shape[-2], _gt_slice.shape[-2])
                            cl_stack = torch.stack(
                                [_slice_consistency_suffix_2hz(c) for c in cl_norms], dim=0
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
                                    _slice_consistency_suffix_2hz(cl_norms[g]),
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
                            # 각 CL g 에 대해 M_ol 개 OL 중 per-anchor flat L2 거리 최소를 선택.
                            ol_sliced_pl = [_slice_consistency_suffix(ol_norms[m]) for m in range(M_ol)]
                            T_min_nm = min(cl_sliced_pl[0].shape[-2], ol_sliced_pl[0].shape[-2])
                            with torch.no_grad():
                                cl_stk_nm = torch.stack(
                                    [c[:, :T_min_nm, :] for c in cl_sliced_pl], dim=0
                                ).detach()  # [G, N_active, T, F]
                                ol_stk_nm = torch.stack(
                                    [o[:, :T_min_nm, :] for o in ol_sliced_pl], dim=0
                                )  # [M, N_active, T, F]
                                # [G, M] flat L2² distance
                                _d2_nm = ((cl_stk_nm.unsqueeze(1) - ol_stk_nm.unsqueeze(0)) ** 2).flatten(2).sum(-1)
                                m_star_nm = _d2_nm.argmin(dim=1).tolist()
                            del cl_stk_nm, ol_stk_nm, _d2_nm
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
        # DDP: 모든 trainable param을 dummy graph에 연결해 bucket reducer가 정상 작동하도록 함.
        # no_sync 컨텍스트 내에서 .backward()로 누적한 grad를 training_step에서
        # manual_backward(_ddp_dummy)로 최종 all-reduce 한 번에 동기화.
        _ddp_dummy = sum(p.sum() * 0.0 for p in self.parameters() if p.requires_grad)
        ret["loss"] = _ddp_dummy
        return ret

    def training_step(self, data, batch_idx):
        """한 batch의 flow matching loss를 계산합니다.

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

        if self._is_ocsc_ft_enabled():
            import contextlib

            opt = self.optimizers()
            opt.zero_grad()

            # OCSC step 본체는 prepare_inference_cache 를 호출하므로
            # tokenized_agent 가 inference 형태 (valid_mask / gt_pos / gt_heading
            # / gt_idx / rollout_init_* / gt_pos_raw 등) 를 갖춰야 한다. project_3
            # 측 TokenProcessor 는 train mode 에서 그 키들을 일부 누락시키므로
            # OCSC step 직전에 강제로 eval mode 로 전환한다.
            _was_training = self.token_processor.training
            self.token_processor.eval()
            try:
                tokenized_map, tokenized_agent = self.token_processor(data)
            finally:
                self.token_processor.train(_was_training)

            # DDP multi-GPU: 모든 .backward() 호출을 no_sync 컨텍스트 안에서
            # 실행해 per-backward all-reduce 를 막고, 끝나서 manual_backward 로
            # all-reduce 1회만 트리거. anchor 수가 GPU 마다 달라도 deadlock 없음.
            _trainer = getattr(self, "trainer", None)
            _ddp_model = (
                getattr(_trainer.strategy, "model", None) if _trainer is not None else None
            )
            _no_sync_ctx = (
                _ddp_model.no_sync()
                if _ddp_model is not None and hasattr(_ddp_model, "no_sync")
                else contextlib.nullcontext()
            )
            with _no_sync_ctx:
                diag = self._run_flow_ocsc_ft_step(tokenized_map, tokenized_agent, data)

            if "loss" in diag:
                self.manual_backward(diag["loss"])

            opt.step()

            for k, v in diag.items():
                if k == "loss":
                    continue
                if isinstance(v, (Tensor, float)):
                    self.log(k, v, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
            if "train/consistency_loss" in diag:
                _fm_reg = diag.get("train/fm_reg_loss", 0.0)
                _fm_reg_lambda = float(self.finetune_config.ocsc_fm_reg_lambda)
                _consistency = diag["train/consistency_loss"]
                _fm_reg_t = (
                    _fm_reg
                    if isinstance(_fm_reg, Tensor)
                    else torch.tensor(_fm_reg, device=_consistency.device)
                )
                _total = _consistency + _fm_reg_lambda * _fm_reg_t
                self.log(
                    "train/loss",
                    _total,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=1,
                )
            return None

        tokenized_map, tokenized_agent = self.token_processor(data)
        pred = self.encoder(
            tokenized_map,
            tokenized_agent,
            anchor_mask_key="flow_train_mask",
        )
        fm_loss, open_metric_dict, _ = self._open_loop_denoise_metrics(pred)

        total_loss = fm_loss
        if not torch.isfinite(fm_loss):
            raise RuntimeError(
                f"Non-finite fm_loss detected: {self._summarize_nonfinite_tensor(fm_loss)}"
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
        return total_loss

    def on_after_backward(self) -> None:
        """역전파 직후 non-finite gradient를 fail-fast로 잡습니다.

        설명:
            ``precision='16-mixed'`` 에서는 Lightning이 ``GradScaler`` 로 loss를 스케일해
            backward를 수행하므로, 이 시점의 gradient는 정상적으로 scaled 상태이고
            fp16 overflow로 인한 inf/NaN도 흔하게 발생합니다. ``GradScaler.step`` 이
            optimizer step을 자동으로 건너뛰고 scale factor를 낮춰 회복하므로, scaler가
            활성인 경로에서는 여기서 ``raise`` 하지 않습니다. scaler가 없는 경로
            (bf16 / 32-true) 에서는 기존대로 fail-fast를 유지합니다.
        """
        bad_grad = self._find_first_nonfinite_gradient()
        if bad_grad is None:
            return
        bad_name, bad_tensor = bad_grad
        raise RuntimeError(
            "Detected non-finite gradient after backward: "
            f"{bad_name} ({self._summarize_nonfinite_tensor(bad_tensor)})"
        )

    def validation_step(self, data, batch_idx):
        eval_generator = self.encoder
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

    def _resolve_lr_total_steps(self) -> int:
        """현재 스케줄 단위에 맞는 전체 step 수를 정합니다."""
        if self.lr_total_steps > 0:
            return int(self.lr_total_steps)
        if self.lr_scheduler_unit == "step" and self.trainer is not None:
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

    def test_step(self, data, batch_idx):
        eval_generator = self.encoder
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
