# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import math
from pathlib import Path
from typing import Dict, List

import hydra
import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR

from src.smart.metrics import FlowMatchingLoss, WOSACMetrics, WOSACSubmission, minADE
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils.finetune import set_model_for_finetuning
from src.smart.utils.flow_traj import (
    build_anchor_10hz_indices,
    chunk_future_21_to_4x6,
    sample_anchor_10hz_indices,
)
from src.utils.vis_waymo import VisWaymo
from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts


class SMART(LightningModule):
    """CAT-K 학습/검증 파이프라인을 유지한 flow 버전 SMART입니다."""

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
        self.anchor_chunk_k = model_config.decoder.anchor_chunk_k
        self.closed_loop_steps = model_config.closed_loop_steps
        self.train_max_num = model_config.get("train_max_num")
        self.log_epoch = -1
        self.val_open_loop = model_config.val_open_loop
        self.val_closed_loop = model_config.val_closed_loop

        self.token_processor = TokenProcessor(**model_config.token_processor)
        self.encoder = SMARTDecoder(
            **model_config.decoder,
            n_token_agent=self.token_processor.n_token_agent,
        )
        set_model_for_finetuning(self.encoder, model_config.finetune)

        self.minADE = minADE()
        self.wosac_metrics = WOSACMetrics("val_closed")
        self.wosac_submission = WOSACSubmission(**model_config.wosac_submission)
        self.training_loss = FlowMatchingLoss(**model_config.training_loss)

        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self.n_vis_batch = model_config.n_vis_batch
        self.n_vis_scenario = model_config.n_vis_scenario
        self.n_vis_rollout = model_config.n_vis_rollout
        self.n_batch_wosac_metric = model_config.n_batch_wosac_metric

        self.flow_sampling = model_config.flow_sampling
        self.validation_flow_sampling = model_config.validation_flow_sampling

        self.video_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.video_dir = Path(self.video_dir) / "videos"

    def _get_train_mask(self, data) -> torch.Tensor | None:
        if "train_mask" in data["agent"]:
            return data["agent"]["train_mask"]
        return None

    def _limit_train_mask_per_graph(
        self,
        role_train_mask: torch.Tensor,
        extra_train_mask: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        train_mask = role_train_mask.clone()
        extra_only_mask = extra_train_mask & ~role_train_mask
        if self.train_max_num is None:
            train_mask |= extra_only_mask
            return train_mask

        for graph_idx in range(num_graphs):
            graph_mask = batch == graph_idx
            graph_role_count = int((role_train_mask & graph_mask).sum().item())
            remaining = self.train_max_num - graph_role_count
            if remaining <= 0:
                continue

            extra_indices = torch.where(extra_only_mask & graph_mask)[0]
            if extra_indices.numel() <= remaining:
                train_mask[extra_indices] = True
                continue

            selected = torch.randperm(extra_indices.numel(), device=batch.device)[:remaining]
            train_mask[extra_indices[selected]] = True
        return train_mask

    def _build_anchor_train_mask(
        self,
        data,
        anchor_10hz: int,
        target_valid: torch.Tensor,
    ) -> torch.Tensor:
        anchor_active = target_valid[:, 0]
        if self._get_train_mask(data) is None:
            return anchor_active

        agent_batch = data["agent"]["batch"]
        ego_mask = data["agent"]["role"][:, 0]
        if int(ego_mask.sum().item()) != data.num_graphs:
            raise ValueError("Expected exactly one ego agent per graph when building anchor train masks.")

        anchor_pos = data["agent"]["position"][:, anchor_10hz, :2]
        ego_anchor_pos = anchor_pos.new_zeros((data.num_graphs, 2))
        ego_anchor_pos[agent_batch[ego_mask]] = anchor_pos[ego_mask]
        anchor_distance = torch.norm(anchor_pos - ego_anchor_pos[agent_batch], dim=-1)

        role_train_mask = data["agent"]["role"].any(-1)
        future_valid_count = target_valid[:, 1:].sum(-1)
        extra_train_mask = (anchor_distance < 100.0) & (future_valid_count >= 5)
        train_mask = self._limit_train_mask_per_graph(
            role_train_mask=role_train_mask,
            extra_train_mask=extra_train_mask,
            batch=agent_batch,
            num_graphs=data.num_graphs,
        )
        return train_mask & anchor_active

    def _open_loop_anchor_loss(
        self,
        data,
        tokenized_map: Dict[str, torch.Tensor],
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        anchor_10hz: int,
    ):
        pred = self.encoder(
            tokenized_map=tokenized_map,
            tokenized_agent=tokenized_agent,
            data=data,
            anchor_10hz=anchor_10hz,
            sampling_cfg=self.flow_sampling,
            map_feature=map_feature,
        )
        train_mask = self._build_anchor_train_mask(
            data=data,
            anchor_10hz=anchor_10hz,
            target_valid=pred["target_valid"],
        )
        return self.training_loss(
            pred_segments=pred["pred_segments"],
            target_segments=pred["target_segments"],
            target_valid=pred["target_valid"],
            train_mask=train_mask,
            pred_velocity=pred["pred_velocity"],
            target_velocity=pred["target_velocity"],
        )

    def _closed_loop_train_loss(self, data, tokenized_map, tokenized_agent, map_feature):
        rollout = self.encoder.rollout(
            tokenized_map=tokenized_map,
            tokenized_agent=tokenized_agent,
            sampling_cfg=self.flow_sampling,
            data=data,
            rollout_steps=self.closed_loop_steps,
            return_targets=True,
            map_feature=map_feature,
        )
        pred_future = torch.cat(rollout["pred_local_futures"], dim=0)  # [n_agent * n_step, 21, 4]
        target_future = torch.cat(rollout["target_local_futures"], dim=0)  # [n_agent * n_step, 21, 4]
        target_valid = torch.cat(rollout["target_valids"], dim=0)  # [n_agent * n_step, 21]
        pred_segments = chunk_future_21_to_4x6(pred_future)
        target_segments = chunk_future_21_to_4x6(target_future)
        train_mask = None
        if self._get_train_mask(data) is not None:
            shift = self.encoder.agent_encoder.shift
            start_anchor = self.num_historical_steps - 1
            train_mask = torch.cat(
                [
                    self._build_anchor_train_mask(
                        data=data,
                        anchor_10hz=start_anchor + step * shift,
                        target_valid=step_target_valid,
                    )
                    for step, step_target_valid in enumerate(rollout["target_valids"])
                ],
                dim=0,
            )
        return self.training_loss(
            pred_segments=pred_segments,
            target_segments=target_segments,
            target_valid=target_valid,
            train_mask=train_mask,
            pred_velocity=None,
            target_velocity=None,
        )

    def training_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        # The static map does not depend on the sampled anchor, so reuse one graph per batch.
        map_feature = self.encoder.encode_map(tokenized_map)

        if self.closed_loop_steps > 0:
            loss_out = self._closed_loop_train_loss(data, tokenized_map, tokenized_agent, map_feature)
        else:
            candidate_anchors = build_anchor_10hz_indices(
                num_historical_steps=self.num_historical_steps,
                future_window_steps=self.future_window_steps,
                total_steps=data["agent"]["position"].shape[1],
                shift=5,
            )
            anchor_list = sample_anchor_10hz_indices(
                candidate_anchors=candidate_anchors,
                anchor_chunk_k=self.anchor_chunk_k,
                device=data["agent"]["position"].device,
            )
            loss_items = [
                self._open_loop_anchor_loss(
                    data,
                    tokenized_map,
                    tokenized_agent,
                    map_feature,
                    anchor_10hz=a,
                )
                for a in anchor_list
            ]
            total_loss = torch.stack([x.total_loss for x in loss_items]).mean()
            flow_loss = torch.stack([x.flow_loss for x in loss_items]).mean()
            overlap_loss = torch.stack([x.overlap_loss for x in loss_items]).mean()
            ade_2s = torch.stack([x.ade_2s for x in loss_items]).mean()
            loss_out = type(loss_items[0])(
                total_loss=total_loss,
                flow_loss=flow_loss,
                overlap_loss=overlap_loss,
                ade_2s=ade_2s,
            )

        self.log("train/loss", loss_out.total_loss, on_step=True, batch_size=1)
        self.log("train/flow_loss", loss_out.flow_loss, on_step=True, batch_size=1)
        self.log("train/overlap_loss", loss_out.overlap_loss, on_step=True, batch_size=1)
        self.log("train/ade_2s", loss_out.ade_2s, on_step=True, batch_size=1)
        return loss_out.total_loss

    def validation_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        # Validation reuses the same scene map across all anchors / rollouts as well.
        map_feature = self.encoder.encode_map(tokenized_map)

        if self.val_open_loop:
            candidate_anchors = build_anchor_10hz_indices(
                num_historical_steps=self.num_historical_steps,
                future_window_steps=self.future_window_steps,
                total_steps=data["agent"]["position"].shape[1],
                shift=5,
            )
            loss_items = [
                self._open_loop_anchor_loss(
                    data,
                    tokenized_map,
                    tokenized_agent,
                    map_feature,
                    anchor_10hz=a,
                )
                for a in candidate_anchors
            ]
            if len(loss_items) > 0:
                self.log(
                    "val_open/loss",
                    torch.stack([x.total_loss for x in loss_items]).mean(),
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=1,
                )
                self.log(
                    "val_open/flow_loss",
                    torch.stack([x.flow_loss for x in loss_items]).mean(),
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=1,
                )
                self.log(
                    "val_open/overlap_loss",
                    torch.stack([x.overlap_loss for x in loss_items]).mean(),
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=1,
                )
                self.log(
                    "val_open/ade_2s",
                    torch.stack([x.ade_2s for x in loss_items]).mean(),
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=1,
                )

        if self.val_closed_loop:
            pred_traj, pred_z, pred_head = [], [], []
            for _ in range(self.n_rollout_closed_val):
                pred = self.encoder.rollout(
                    tokenized_map=tokenized_map,
                    tokenized_agent=tokenized_agent,
                    sampling_cfg=self.validation_flow_sampling,
                    data=data,
                    rollout_steps=self.num_future_steps // 5,
                    return_targets=False,
                    map_feature=map_feature,
                )
                pred_traj.append(pred["pred_traj_10hz"])
                pred_z.append(pred["pred_z_10hz"])
                pred_head.append(pred["pred_head_10hz"])

            pred_traj = torch.stack(pred_traj, dim=1)  # [n_agent, n_rollout, 80, 2]
            pred_z = torch.stack(pred_z, dim=1)        # [n_agent, n_rollout, 80]
            pred_head = torch.stack(pred_head, dim=1)  # [n_agent, n_rollout, 80]

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
                _gpu_dict_sync = self.wosac_submission.compute()
                if self.global_rank == 0:
                    for k in _gpu_dict_sync.keys():
                        if isinstance(_gpu_dict_sync[k], list):
                            _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
                    scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
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
                for _i_sc in range(self.n_vis_scenario):
                    _vis = VisWaymo(
                        scenario_path=data["tfrecord_path"][_i_sc],
                        save_dir=self.video_dir / f"batch_{batch_idx:02d}-scenario_{_i_sc:02d}",
                    )
                    _vis.save_video_scenario_rollout(scenario_rollouts[_i_sc], self.n_vis_rollout)
                    for _path in _vis.video_paths:
                        self.logger.log_video("/".join(_path.split("/")[-3:]), [_path])

    def on_validation_epoch_end(self):
        if self.val_closed_loop:
            if not self.wosac_submission.is_active:
                epoch_wosac_metrics = self.wosac_metrics.compute()
                val_closed_ade = self.minADE.compute()
                self.log(
                    "val_closed/ADE",
                    val_closed_ade,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=1,
                )
                epoch_wosac_metrics["val_closed/ADE"] = val_closed_ade
                if self.global_rank == 0:
                    epoch_wosac_metrics["epoch"] = self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                    self.logger.log_metrics(epoch_wosac_metrics)
                self.wosac_metrics.reset()
                self.minADE.reset()

            if self.global_rank == 0 and self.wosac_submission.is_active:
                self.wosac_submission.save_sub_file()

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)

        def lr_lambda(current_step):
            current_step = max(int(current_step), 0)
            if current_step < self.lr_warmup_steps:
                return self.lr_min_ratio + (1 - self.lr_min_ratio) * current_step / self.lr_warmup_steps
            decay_steps = max(self.lr_total_steps - self.lr_warmup_steps, 1)
            return self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * (
                1.0
                + math.cos(
                    math.pi
                    * min(
                        1.0,
                        (current_step - self.lr_warmup_steps) / decay_steps,
                    )
                )
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def test_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        # Test rollout sees the same static map at every rollout step.
        map_feature = self.encoder.encode_map(tokenized_map)
        pred_traj, pred_z, pred_head = [], [], []
        for _ in range(self.n_rollout_closed_val):
            pred = self.encoder.rollout(
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
                sampling_cfg=self.validation_flow_sampling,
                data=data,
                rollout_steps=self.num_future_steps // 5,
                return_targets=False,
                map_feature=map_feature,
            )
            pred_traj.append(pred["pred_traj_10hz"])
            pred_z.append(pred["pred_z_10hz"])
            pred_head.append(pred["pred_head_10hz"])

        pred_traj = torch.stack(pred_traj, dim=1)
        pred_z = torch.stack(pred_z, dim=1)
        pred_head = torch.stack(pred_head, dim=1)

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
            _gpu_dict_sync = self.wosac_submission.compute()
            if self.global_rank == 0:
                for k in _gpu_dict_sync.keys():
                    if isinstance(_gpu_dict_sync[k], list):
                        _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
                scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
                self.wosac_submission.aggregate_rollouts(scenario_rollouts)
            self.wosac_submission.reset()

    def on_test_epoch_end(self):
        if self.global_rank == 0 and self.wosac_submission.is_active:
            self.wosac_submission.save_sub_file()
