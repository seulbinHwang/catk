from __future__ import annotations

import gc
import hashlib
import math
from pathlib import Path
from typing import Dict, Sequence

import hydra
import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

from src.smart.metrics import SimAgentsMetrics, SimAgentsSubmission, minADE
from src.smart.metrics.flow_metrics import (
    WeightedMeanMetric,
    ade_2s,
    fde_2s,
    flow_matching_loss,
    yaw_ade_2s,
    yaw_fde_2s,
)
from src.smart.modules.draft_physics import (
    DRAFT_PHYSICS_ACTUAL_UNIT_KEYS,
    DRAFT_PHYSICS_COMPONENT_KEYS,
    DraftPhysicsRegularizer,
)
from src.smart.modules.smart_flow_decoder import SMARTFlowDecoder
from src.smart.tokens.flow_token_processor import FlowTokenProcessor
from src.smart.utils.finetune import set_model_for_finetuning
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
        self.log_epoch = -1
        self.val_open_loop = model_config.val_open_loop
        self.val_closed_loop = model_config.val_closed_loop
        self.token_processor = FlowTokenProcessor(**model_config.token_processor)

        self.encoder = SMARTFlowDecoder(
            **model_config.decoder,
            n_token_agent=self.token_processor.n_token_agent,
        )
        set_model_for_finetuning(self.encoder, model_config.finetune)

        self.minADE = minADE()
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
        self.vis_flow_2s_preview = bool(getattr(model_config, "vis_flow_2s_preview", False))
        self.delete_local_videos_after_wandb_upload = model_config.delete_local_videos_after_wandb_upload
        self.n_batch_sim_agents_metric = model_config.n_batch_sim_agents_metric
        self._fit_time_original_limit_val_batches: int | float | None = None
        self._fit_time_checkpoint_only_validation_enabled = False

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
                deadzone_ratio=float(getattr(draft_physics, "deadzone_ratio", 0.02)),
                deadzone_softness=float(getattr(draft_physics, "deadzone_softness", 0.02)),
                gt_excess_only=bool(getattr(draft_config, "gt_excess_only", True)),
                speed_weight=float(getattr(draft_physics, "speed_weight", 1.0)),
                slip_weight=float(getattr(draft_physics, "slip_weight", 1.0)),
                accel_weight=float(getattr(draft_physics, "accel_weight", 1.0)),
                yaw_accel_weight=float(getattr(draft_physics, "yaw_accel_weight", 1.0)),
                turn_weight=float(getattr(draft_physics, "turn_weight", 1.0)),
                eps=float(getattr(draft_physics, "eps", 1e-6)),
            )
        else:
            self.draft_regularizer = None

        self.val_open_epoch_metrics = nn.ModuleDict(
            {
                "ADE2s": WeightedMeanMetric(),
                "FDE2s": WeightedMeanMetric(),
                "yaw_ADE2s": WeightedMeanMetric(),
                "yaw_FDE2s": WeightedMeanMetric(),
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
        return_flow_2s_preview: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Dict[str, Tensor] | None]:
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
            pred = self.encoder.rollout_from_cache(
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
                    "traj": pred["pred_flow_2s_traj"].unsqueeze(1),
                    "valid": pred["pred_flow_2s_valid"].unsqueeze(1),
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
        pred = self.encoder.rollout_from_cache(
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
                    pred["pred_flow_2s_traj"],
                    repeat_count=chunk_size,
                    num_agent=num_agent,
                ),
                "valid": self._reshape_parallel_rollout_prediction(
                    pred["pred_flow_2s_valid"],
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
            flow_preview_traj_chunks: list[Tensor] = []
            flow_preview_valid_chunks: list[Tensor] = []
            try:
                for chunk_start in range(0, len(rollout_indices), chunk_size):
                    chunk_rollout_indices = rollout_indices[chunk_start : chunk_start + chunk_size]
                    chunk_pred_traj, chunk_pred_z, chunk_pred_head, chunk_flow_preview = self._run_parallel_rollout_chunk(
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

    def _compute_draft_training_loss(
        self,
        pred_dict: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """실제 샘플러를 돌린 최종 미래에 physics loss를 계산합니다.

        Args:
            pred_dict: flow decoder 출력 사전입니다.
                ``anchor_hidden`` 은 ``[n_agent, 13, hidden_dim]`` 이고,
                ``flow_clean_norm`` 은 ``[n_valid_anchor, 20, 4]`` 입니다.
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
        pred_sample_norm = self.encoder.sample_open_loop_future(
            anchor_hidden=pred_dict["anchor_hidden"],
            anchor_mask=pred_dict["anchor_mask"],
            sampling_scheme=self.draft_sampling,
        )

        if pred_sample_norm.shape[0] != tokenized_agent["flow_train_agent_type"].shape[0]:
            raise ValueError(
                "DRaFT 샘플 개수와 packed anchor 메타데이터 개수가 다릅니다. "
                f"got {pred_sample_norm.shape[0]} and {tokenized_agent['flow_train_agent_type'].shape[0]}"
            )

        if not self.draft_physics_force_fp32:
            return self.draft_regularizer(
                pred_future_norm=pred_sample_norm,
                target_future_norm=pred_dict["flow_clean_norm"],
                packed_agent_type=tokenized_agent["flow_train_agent_type"],
                packed_prev_control=tokenized_agent["flow_train_prev_control"],
                packed_prev_control_valid=tokenized_agent["flow_train_prev_control_valid"],
            )

        # Keep the threshold-heavy physics penalty in fp32 even when the trainer
        # runs with bf16 autocast, while preserving gradients to pred_sample_norm.
        with torch.autocast(device_type=pred_sample_norm.device.type, enabled=False):
            return self.draft_regularizer(
                pred_future_norm=pred_sample_norm.float(),
                target_future_norm=pred_dict["flow_clean_norm"].float(),
                packed_agent_type=tokenized_agent["flow_train_agent_type"],
                packed_prev_control=tokenized_agent["flow_train_prev_control"].float(),
                packed_prev_control_valid=tokenized_agent["flow_train_prev_control_valid"],
            )

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
            "train/loss_phys_raw",
            physics_dict["raw_pred_loss"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        for metric_name in DRAFT_PHYSICS_COMPONENT_KEYS:
            self.log(
                f"raw_feaisble_gap/{metric_name}",
                physics_dict[metric_name],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
        for metric_name in DRAFT_PHYSICS_ACTUAL_UNIT_KEYS:
            self.log(
                f"raw_feaisble_gap/{metric_name}",
                physics_dict[metric_name],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                f"raw_feaisble_gap/pred_{metric_name}",
                physics_dict[f"pred_{metric_name}"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                f"gt_feasible_gap/{metric_name}",
                physics_dict[f"gt_{metric_name}"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )

    def training_step(self, data, batch_idx):
        """한 batch의 FM loss와 DRaFT physics loss를 함께 계산합니다.

        Args:
            data: 학습용 장면 배치입니다.
            batch_idx: 현재 batch 번호입니다.

        Returns:
            Tensor: 최종 학습 loss입니다.
        """
        """ tokenized_agent
flow_train_agent_type [n_valid_anchor]
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
        """ physics_dict : Dict[str, Tensor] #  모든 값은 scalar tensor 
        
        loss, raw_pred_loss
        
        speed, slip, accel, yaw_accel, turn
        
        speed_excess_mps, slip_beta_excess_deg,
        accel_excess_mps2, yaw_accel_excess_degps2, turn_yaw_rate_excess_degps, 
        turn_lat_accel_excess_mps2, turn_radius_shortfall_m
        
        """
        physics_dict = self._build_zero_draft_metrics(fm_loss)
        total_loss = fm_loss
        if draft_weight > 0.0:
            physics_dict = self._compute_draft_training_loss(
                pred_dict=pred,
                tokenized_agent=tokenized_agent,
            )
            total_loss = total_loss + draft_weight * 0.005 * physics_dict["loss"]

        self.log("train/loss", total_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/loss_fm", fm_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/ADE2s", open_metric_dict["ADE2s"], on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/FDE2s", open_metric_dict["FDE2s"], on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log(
            "train/ADEyaw2s",
            open_metric_dict["yaw_ADE2s"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        self.log(
            "train/FDEyaw2s",
            open_metric_dict["yaw_FDE2s"],
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
            open_pred_clean_norm = self.encoder.sample_open_loop_future(
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
                        vis_flow_2s_preview=self.vis_flow_2s_preview,
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
            if self.sim_agents_submission.is_active:
                self.sim_agents_submission.save_sub_file()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

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

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]

    def test_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        map_feature = self.encoder.encode_map(tokenized_map)
        pred_traj, pred_z, pred_head, _ = self._run_closed_loop_rollouts(
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
