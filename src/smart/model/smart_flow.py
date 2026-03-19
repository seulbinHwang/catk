from __future__ import annotations

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
        self.delete_local_videos_after_wandb_upload = model_config.delete_local_videos_after_wandb_upload
        self.n_batch_sim_agents_metric = model_config.n_batch_sim_agents_metric

        self.video_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.video_dir = Path(self.video_dir) / "videos"

        self.validation_rollout_sampling = model_config.validation_rollout_sampling
        self.val_open_epoch_metrics = nn.ModuleDict(
            {
                "ADE2s": WeightedMeanMetric(),
                "FDE2s": WeightedMeanMetric(),
                "yaw_ADE2s": WeightedMeanMetric(),
                "yaw_FDE2s": WeightedMeanMetric(),
            }
        )

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

    def _run_closed_loop_rollouts(
        self,
        data,
        tokenized_agent,
        map_feature: Dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """한 batch의 모든 closed-loop rollout을 같은 규칙으로 생성합니다.

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
        scenario_device = tokenized_agent["batch"].device
        pred_traj, pred_z, pred_head = [], [], []
        for rollout_idx in range(self.n_rollout_closed_val):
            scenario_sampling_seeds = self._get_closed_loop_scenario_seeds(
                scenario_ids=data["scenario_id"],
                rollout_idx=rollout_idx,
                device=scenario_device,
            )
            pred = self.encoder.rollout_from_cache(
                rollout_cache=rollout_cache,
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
                sampling_scheme=self.validation_rollout_sampling,
                scenario_sampling_seeds=scenario_sampling_seeds,
            )
            pred_traj.append(pred["pred_traj_10hz"])
            pred_z.append(pred["pred_z_10hz"])
            pred_head.append(pred["pred_head_10hz"])

        return (
            torch.stack(pred_traj, dim=1),
            torch.stack(pred_z, dim=1),
            torch.stack(pred_head, dim=1),
        )

    def _update_closed_loop_metric_states(
        self,
        data,
        batch_idx: int,
        pred_traj: Tensor,
        pred_z: Tensor,
        pred_head: Tensor,
    ) -> object:
        scenario_rollouts = None
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

    def training_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        pred = self.encoder(
            tokenized_map,
            tokenized_agent,
            anchor_mask_key="flow_train_mask",
        )
        loss, open_metric_dict, _ = self._open_loop_denoise_metrics(pred)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
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
        return loss

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
            pred_traj, pred_z, pred_head = self._run_closed_loop_rollouts(
                data=data,
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
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
                    global_rank=self.global_rank,
                )
                gpu_dict = self.sim_agents_submission.compute()
                if self.global_rank == 0:
                    for k in gpu_dict.keys():
                        if isinstance(gpu_dict[k], list):
                            gpu_dict[k] = gpu_dict[k][0]
                    scenario_rollouts = get_scenario_rollouts(**gpu_dict)
                    self.sim_agents_submission.aggregate_rollouts(scenario_rollouts)
                self.sim_agents_submission.reset()
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
                    )
                    vis.save_video_scenario_rollout(scenario_rollouts[scen_idx], self.n_vis_rollout)
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
                    minade_value = self.minADE.compute()
                closed_loop_metric = epoch_sim_agents_metrics[self.closed_loop_metric_name]
                if self.global_rank == 0:
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
            if self.global_rank == 0 and self.sim_agents_submission.is_active:
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
        pred_traj, pred_z, pred_head = self._run_closed_loop_rollouts(
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
            global_rank=self.global_rank,
        )
        gpu_dict = self.sim_agents_submission.compute()
        if self.global_rank == 0:
            for k in gpu_dict.keys():
                if isinstance(gpu_dict[k], list):
                    gpu_dict[k] = gpu_dict[k][0]
            scenario_rollouts = get_scenario_rollouts(**gpu_dict)
            self.sim_agents_submission.aggregate_rollouts(scenario_rollouts)
        self.sim_agents_submission.reset()

    def on_test_epoch_end(self):
        if self.global_rank == 0:
            self.sim_agents_submission.save_sub_file()
