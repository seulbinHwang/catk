# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import hydra
import torch
from lightning import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

from src.smart.metrics import FlowMatchingLoss, WOSACMetrics, WOSACSubmission, minADE
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils.finetune import set_model_for_finetuning
from src.utils.vis_waymo import VisWaymo
from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts


class SMART(LightningModule):
    """CAT-K/SMART 학습·검증·제출 파이프라인을 유지한 flow 버전 모델."""

    def __init__(self, model_config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = model_config.lr
        self.lr_warmup_steps = model_config.lr_warmup_steps
        self.lr_total_steps = model_config.lr_total_steps
        self.lr_min_ratio = model_config.lr_min_ratio
        self.num_historical_steps = model_config.decoder.num_historical_steps
        self.num_future_steps = model_config.decoder.num_future_steps
        self.future_window_steps = model_config.decoder.future_window_steps
        self.anchor_chunk_k = model_config.anchor_chunk_k
        self.closed_loop_unroll = model_config.closed_loop_unroll
        self.use_closed_loop_finetune = model_config.use_closed_loop_finetune
        self.log_epoch = -1
        self.val_open_loop = model_config.val_open_loop
        self.val_closed_loop = model_config.val_closed_loop
        self.token_processor = TokenProcessor(**model_config.token_processor)

        self.encoder = SMARTDecoder(**model_config.decoder, n_token_agent=self.token_processor.n_token_agent)
        set_model_for_finetuning(self.encoder, model_config.finetune)

        self.flow_loss = FlowMatchingLoss(**model_config.training_loss)
        self.minADE = minADE()
        self.wosac_metrics = WOSACMetrics("val_closed")
        self.wosac_submission = WOSACSubmission(**model_config.wosac_submission)

        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self.n_vis_batch = model_config.n_vis_batch
        self.n_vis_scenario = model_config.n_vis_scenario
        self.n_vis_rollout = model_config.n_vis_rollout
        self.n_batch_wosac_metric = model_config.n_batch_wosac_metric

        self.video_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.video_dir = Path(self.video_dir) / "videos"
        self.validation_rollout_sampling = model_config.validation_rollout_sampling
        self.automatic_optimization = False
        self.manual_gradient_clip_val = None
        self.manual_gradient_clip_algorithm = None
        self.manual_accumulate_grad_batches = None

    @property
    def anchor_steps(self) -> List[int]:
        """2초 supervisory window가 끝까지 존재하는 모든 anchor step 목록."""
        last_anchor = self.num_historical_steps - 1 + self.num_future_steps - self.future_window_steps
        return list(range(self.num_historical_steps - 1, last_anchor + 1, 5))

    def _select_train_anchor_steps(self) -> List[int]:
        """랜덤 1개 대신 deterministic anchor-chunk를 고른다."""
        anchors = self.anchor_steps
        if self.anchor_chunk_k >= len(anchors):
            return anchors
        start = (self.global_step * self.anchor_chunk_k) % len(anchors)
        return [anchors[(start + i) % len(anchors)] for i in range(self.anchor_chunk_k)]

    @staticmethod
    def _local_ade(pred_future_local: Tensor, gt_future_local: Tensor, mask: Tensor) -> Tensor:
        """2초 local ADE를 계산한다."""
        dist = torch.norm(pred_future_local[..., :2] - gt_future_local[..., :2], dim=-1)
        weight = mask[:, None].to(dist.dtype)
        denom = torch.clamp(weight.sum() * dist.shape[1], min=1.0)
        return (dist * weight).sum() / denom

    def _loss_mask_train(self, pred: Dict[str, Tensor], data) -> Tensor:
        """학습용 agent mask를 만든다."""
        return data["agent"]["train_mask"] & pred["future_valid"]

    def _loss_mask_eval(self, pred: Dict[str, Tensor], data, anchor_step: int) -> Tensor:
        """검증용 유효 agent mask를 만든다."""
        return data["agent"]["valid_mask"][:, anchor_step] & pred["future_valid"]

    def _compute_single_loss(self, pred: Dict[str, Tensor], loss_mask: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """한 anchor의 flow loss를 계산한다."""
        return self.flow_loss(
            flow_pred=pred["flow_pred"],
            flow_target=pred["flow_target"],
            pred_segments=pred["pred_segments"],
            gt_segments=pred["gt_segments"],
            pred_future_local=pred["pred_future_local"],
            gt_future_local=pred["gt_future_local"],
            loss_mask=loss_mask,
        )

    def _accumulate_grad_batches(self) -> int:
        if self.manual_accumulate_grad_batches is not None:
            return max(int(self.manual_accumulate_grad_batches), 1)
        return max(int(getattr(self.trainer, "accumulate_grad_batches", 1)), 1)

    def _is_last_train_batch(self, batch_idx: int) -> bool:
        trainer_last_batch = getattr(self.trainer, "is_last_batch", None)
        if isinstance(trainer_last_batch, bool):
            return trainer_last_batch
        num_batches = getattr(self.trainer, "num_training_batches", None)
        return isinstance(num_batches, int) and batch_idx + 1 >= num_batches

    def _should_step_optimizer(self, batch_idx: int, accumulate_grad_batches: int) -> bool:
        return (batch_idx + 1) % accumulate_grad_batches == 0 or self._is_last_train_batch(batch_idx)

    def set_manual_gradient_clipping(self, clip_val, clip_algorithm) -> None:
        self.manual_gradient_clip_val = clip_val
        self.manual_gradient_clip_algorithm = clip_algorithm

    def set_manual_accumulate_grad_batches(self, accumulate_grad_batches) -> None:
        self.manual_accumulate_grad_batches = accumulate_grad_batches

    def _clip_gradients_if_needed(self, optimizer) -> None:
        clip_val = self.manual_gradient_clip_val
        if clip_val is None:
            clip_val = getattr(self.trainer, "gradient_clip_val", None)
        if clip_val is None or clip_val <= 0:
            return
        clip_algorithm = self.manual_gradient_clip_algorithm
        if clip_algorithm is None:
            clip_algorithm = getattr(self.trainer, "gradient_clip_algorithm", None)
        self.clip_gradients(
            optimizer,
            gradient_clip_val=clip_val,
            gradient_clip_algorithm=clip_algorithm,
        )

    def training_step(self, data, batch_idx):
        optimizer = self.optimizers()
        accumulate_grad_batches = self._accumulate_grad_batches()
        if batch_idx % accumulate_grad_batches == 0:
            optimizer.zero_grad()
        should_step_optimizer = self._should_step_optimizer(batch_idx, accumulate_grad_batches)

        tokenized_map, tokenized_agent = self.token_processor(data)
        map_feature = self.encoder.encode_map(tokenized_map)

        total_loss = 0.0
        total_flow = 0.0
        total_overlap = 0.0
        total_recon = 0.0
        block_backward_sync = getattr(self.trainer.strategy, "block_backward_sync", None)
        backward_sync_ctx = (
            block_backward_sync(self, block=not should_step_optimizer)
            if callable(block_backward_sync)
            else nullcontext()
        )
        with backward_sync_ctx:
            if self.use_closed_loop_finetune:
                n_terms = self.closed_loop_unroll
                state = self.encoder.init_closed_loop_state(
                    tokenized_agent=tokenized_agent,
                    agent_raw=data["agent"],
                )
                for step_idx in range(self.closed_loop_unroll):
                    pred, state = self.encoder.closed_loop_train_step(
                        map_feature=map_feature,
                        tokenized_agent=tokenized_agent,
                        agent_raw=data["agent"],
                        state=state,
                        step_idx=step_idx,
                    )
                    loss_mask = self._loss_mask_train(pred, data)
                    loss, log_dict = self._compute_single_loss(pred, loss_mask)
                    total_loss = total_loss + loss.detach()
                    total_flow = total_flow + log_dict["flow"].detach()
                    total_overlap = total_overlap + log_dict["overlap"].detach()
                    total_recon = total_recon + log_dict["recon"].detach()
                    self.manual_backward(
                        loss / max(n_terms * accumulate_grad_batches, 1),
                        retain_graph=step_idx + 1 < n_terms,
                    )
                    del pred, loss, log_dict
            else:
                anchor_steps = self._select_train_anchor_steps()
                n_terms = len(anchor_steps)
                for anchor_idx, anchor_step in enumerate(anchor_steps):
                    pred = self.encoder.forward_from_map(
                        map_feature=map_feature,
                        tokenized_agent=tokenized_agent,
                        agent_raw=data["agent"],
                        anchor_step=anchor_step,
                    )
                    loss_mask = self._loss_mask_train(pred, data)
                    loss, log_dict = self._compute_single_loss(pred, loss_mask)
                    total_loss = total_loss + loss.detach()
                    total_flow = total_flow + log_dict["flow"].detach()
                    total_overlap = total_overlap + log_dict["overlap"].detach()
                    total_recon = total_recon + log_dict["recon"].detach()
                    self.manual_backward(
                        loss / max(n_terms * accumulate_grad_batches, 1),
                        retain_graph=anchor_idx + 1 < n_terms,
                    )
                    del pred, loss, log_dict

        total_loss = total_loss / max(n_terms, 1)
        self.log("train/loss", total_loss, on_step=True, batch_size=1)
        self.log("train/flow", total_flow / max(n_terms, 1), on_step=True, batch_size=1)
        self.log("train/overlap", total_overlap / max(n_terms, 1), on_step=True, batch_size=1)
        self.log("train/recon", total_recon / max(n_terms, 1), on_step=True, batch_size=1)
        if should_step_optimizer:
            self._clip_gradients_if_needed(optimizer)
            optimizer.step()
        return total_loss

    def validation_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        map_feature = self.encoder.encode_map(tokenized_map)

        if self.val_open_loop:
            total_loss = 0.0
            total_ade = 0.0
            n_terms = 0
            for anchor_step in self.anchor_steps:
                pred = self.encoder.forward_from_map(
                    map_feature=map_feature,
                    tokenized_agent=tokenized_agent,
                    agent_raw=data["agent"],
                    anchor_step=anchor_step,
                )
                loss_mask = self._loss_mask_eval(pred, data, anchor_step)
                loss, _ = self._compute_single_loss(pred, loss_mask)
                ade = self._local_ade(pred["pred_future_local"], pred["gt_future_local"], loss_mask)
                total_loss = total_loss + loss
                total_ade = total_ade + ade
                n_terms += 1
            self.log("val_open/loss", total_loss / max(n_terms, 1), on_epoch=True, sync_dist=True, batch_size=1)
            self.log("val_open/ade2s", total_ade / max(n_terms, 1), on_epoch=True, sync_dist=True, batch_size=1)

        if self.val_closed_loop:
            pred_traj, pred_z, pred_head = [], [], []
            for _ in range(self.n_rollout_closed_val):
                pred = self.encoder.agent_encoder.inference(
                    tokenized_agent=tokenized_agent,
                    map_feature=map_feature,
                    agent_raw=data["agent"],
                    sampling_scheme=self.validation_rollout_sampling,
                )
                pred_traj.append(pred["pred_traj_10hz"])
                pred_z.append(pred["pred_z_10hz"])
                pred_head.append(pred["pred_head_10hz"])

            pred_traj = torch.stack(pred_traj, dim=1)
            pred_z = torch.stack(pred_z, dim=1)
            pred_head = torch.stack(pred_head, dim=1)

            scenario_rollouts = None
            if self.wosac_submission.is_active:
                self.wosac_submission.update(
                    scenario_id=data["scenario_id"],
                    agent_id=data["agent"]["id"],
                    agent_batch=data["agent"]["batch"],
                    pred_traj=pred_traj,
                    pred_z=pred_z,
                    pred_head=pred_head,
                    global_rank=self.global_rank,
                )
                gpu_dict_sync = self.wosac_submission.compute()
                if self.global_rank == 0:
                    for k in gpu_dict_sync.keys():
                        if isinstance(gpu_dict_sync[k], list):
                            gpu_dict_sync[k] = gpu_dict_sync[k][0]
                    scenario_rollouts = get_scenario_rollouts(**gpu_dict_sync)
                    self.wosac_submission.aggregate_rollouts(scenario_rollouts)
                self.wosac_submission.reset()
            else:
                self.minADE.update(
                    pred=pred_traj,
                    target=data["agent"]["position"][:, self.num_historical_steps :, : pred_traj.shape[-1]],
                    target_valid=data["agent"]["valid_mask"][:, self.num_historical_steps :],
                )
                if batch_idx < self.n_batch_wosac_metric:
                    device = pred_traj.device
                    scenario_rollouts = get_scenario_rollouts(
                        scenario_id=get_scenario_id_int_tensor(data["scenario_id"], device),
                        agent_id=data["agent"]["id"],
                        agent_batch=data["agent"]["batch"],
                        pred_traj=pred_traj,
                        pred_z=pred_z,
                        pred_head=pred_head,
                    )
                    self.wosac_metrics.update(data["tfrecord_path"], scenario_rollouts)

            if self.global_rank == 0 and batch_idx < self.n_vis_batch and scenario_rollouts is not None:
                for i_sc in range(self.n_vis_scenario):
                    vis = VisWaymo(
                        scenario_path=data["tfrecord_path"][i_sc],
                        save_dir=self.video_dir / f"batch_{batch_idx:02d}-scenario_{i_sc:02d}",
                    )
                    vis.save_video_scenario_rollout(scenario_rollouts[i_sc], self.n_vis_rollout)
                    for path in vis.video_paths:
                        self.logger.log_video("/".join(path.split("/")[-3:]), [path])

    def on_validation_epoch_end(self):
        if self.val_closed_loop:
            if not self.wosac_submission.is_active:
                epoch_metrics = self.wosac_metrics.compute()
                epoch_metrics["val_closed/ADE"] = self.minADE.compute()
                if self.global_rank == 0:
                    epoch_metrics["epoch"] = self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                    self.logger.log_metrics(epoch_metrics)
                self.wosac_metrics.reset()
                self.minADE.reset()
            elif self.global_rank == 0:
                self.wosac_submission.save_sub_file()

    def on_train_epoch_end(self) -> None:
        scheduler = self.lr_schedulers()
        if scheduler is None:
            return
        if isinstance(scheduler, (list, tuple)):
            for lr_scheduler in scheduler:
                lr_scheduler.step()
            return
        scheduler.step()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        def lr_lambda(_current_step):
            current_step = self.current_epoch + 1
            if current_step < self.lr_warmup_steps:
                return self.lr_min_ratio + (1 - self.lr_min_ratio) * current_step / self.lr_warmup_steps
            return self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * (
                1.0 + math.cos(
                    math.pi
                    * min(
                        1.0,
                        (current_step - self.lr_warmup_steps)
                        / max(1, (self.lr_total_steps - self.lr_warmup_steps)),
                    )
                )
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]

    def test_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        pred_traj, pred_z, pred_head = [], [], []
        for _ in range(self.n_rollout_closed_val):
            pred = self.encoder.inference(
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
                agent_raw=data["agent"],
                sampling_scheme=self.validation_rollout_sampling,
            )
            pred_traj.append(pred["pred_traj_10hz"])
            pred_z.append(pred["pred_z_10hz"])
            pred_head.append(pred["pred_head_10hz"])

        pred_traj = torch.stack(pred_traj, dim=1)
        pred_z = torch.stack(pred_z, dim=1)
        pred_head = torch.stack(pred_head, dim=1)

        self.wosac_submission.update(
            scenario_id=data["scenario_id"],
            agent_id=data["agent"]["id"],
            agent_batch=data["agent"]["batch"],
            pred_traj=pred_traj,
            pred_z=pred_z,
            pred_head=pred_head,
            global_rank=self.global_rank,
        )
        gpu_dict_sync = self.wosac_submission.compute()
        if self.global_rank == 0:
            for k in gpu_dict_sync.keys():
                if isinstance(gpu_dict_sync[k], list):
                    gpu_dict_sync[k] = gpu_dict_sync[k][0]
            scenario_rollouts = get_scenario_rollouts(**gpu_dict_sync)
            self.wosac_submission.aggregate_rollouts(scenario_rollouts)
        self.wosac_submission.reset()

    def on_test_epoch_end(self):
        if self.global_rank == 0:
            self.wosac_submission.save_sub_file()
