from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import hydra
import torch
from lightning import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

from src.smart.metrics.min_ade import minADE
from src.smart.metrics.wosac_metrics import WOSACMetrics
from src.smart.metrics.wosac_submission import WOSACSubmission
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import wrap_angle
from src.utils.vis_waymo import VisWaymo
from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts


class SMART(LightningModule):
    """SMART-flow 7M pre-BC용 LightningModule이다.

    NTP token classification loss 대신, flow matching 단일 loss만 사용한다.
    validation/test의 WOSAC 출력 인터페이스는 기존 SMART와 맞춘다.
    """

    def __init__(self, model_config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = model_config.lr
        self.lr_warmup_steps = model_config.lr_warmup_steps
        self.lr_total_steps = model_config.lr_total_steps
        self.lr_min_ratio = model_config.lr_min_ratio
        self.num_historical_steps = model_config.decoder.num_historical_steps
        self.xy_loss_scale = model_config.xy_loss_scale
        self.val_open_loop = model_config.val_open_loop
        self.val_closed_loop = model_config.val_closed_loop

        self.token_processor = TokenProcessor(**model_config.token_processor)
        self.encoder = SMARTDecoder(
            **model_config.decoder,
            n_token_agent=self.token_processor.n_token_agent,
        )

        self.minADE = minADE()
        self.wosac_metrics = WOSACMetrics("val_closed")
        self.wosac_submission = WOSACSubmission(**model_config.wosac_submission)

        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self.n_vis_batch = model_config.n_vis_batch
        self.n_vis_scenario = model_config.n_vis_scenario
        self.n_vis_rollout = model_config.n_vis_rollout
        self.n_batch_wosac_metric = model_config.n_batch_wosac_metric
        self.validation_rollout_sampling = model_config.validation_rollout_sampling

        video_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.video_dir = Path(video_dir) / "videos"

    @staticmethod
    def _merge_train_mask(
        anchor_mask: Tensor,
        train_mask: Optional[Tensor],
    ) -> Tensor:
        """agent 수준 학습 마스크를 anchor 수준으로 늘린다.

        Args:
            anchor_mask: [n_agent, n_anchor] 모양의 anchor 마스크이다.
            train_mask: [n_agent] 또는 None이다.

        Returns:
            [n_agent, n_anchor] 모양의 최종 anchor 마스크이다.
        """
        if train_mask is None:
            return anchor_mask
        return anchor_mask & train_mask.unsqueeze(-1)

    def _flow_loss(
        self,
        pred_dict: Dict[str, Tensor],
        anchor_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """flow matching 단일 loss를 계산한다.

        Args:
            pred_dict: decoder 출력이다.
            anchor_mask: [n_agent, n_anchor] 모양의 실제 학습 anchor 마스크이다.

        Returns:
            스칼라 loss를 돌려준다.
        """
        if anchor_mask is None:
            anchor_mask = pred_dict.get("flow_train_mask", pred_dict["flow_anchor_valid"])
        step_mask = pred_dict["flow_future_valid"] & anchor_mask.unsqueeze(-1)
        step_mask = step_mask.unsqueeze(-1).to(pred_dict["pred_flow"].dtype)

        diff = pred_dict["pred_flow"] - pred_dict["target_flow"]
        if not pred_dict.get("flow_state_normalized", False):
            diff_xy = diff[..., :2] / self.xy_loss_scale
            diff_rest = diff[..., 2:]
            diff = torch.cat([diff_xy, diff_rest], dim=-1)
        loss = (diff.pow(2) * step_mask).sum()
        denom = step_mask.sum() * diff.shape[-1]
        return loss / denom.clamp_min(1.0)

    @staticmethod
    def _future_metrics(
        pred_dict: Dict[str, Tensor],
        anchor_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """open-loop 2초 위치/yaw 오차를 실제 단위로 계산한다.

        Args:
            pred_dict: decoder 출력이다.
            anchor_mask: [n_agent, n_anchor] 추가 마스크이다.

        Returns:
            meter 단위 위치 ADE와 degree 단위 yaw ADE를 돌려준다.
        """
        valid_mask = pred_dict["flow_future_valid"]
        if anchor_mask is not None:
            valid_mask = valid_mask & anchor_mask.unsqueeze(-1)
        valid_mask = valid_mask.to(pred_dict["pred_future_pos"].dtype)
        pos_error = torch.norm(
            pred_dict["pred_future_pos"] - pred_dict["gt_future_pos"],
            dim=-1,
        )
        yaw_error_deg = torch.rad2deg(
            wrap_angle(pred_dict["pred_future_head"] - pred_dict["gt_future_head"]).abs()
        )
        denom = valid_mask.sum().clamp_min(1.0)
        return {
            "ade_2s_m": (pos_error * valid_mask).sum() / denom,
            "ade_yaw_2s_deg": (yaw_error_deg * valid_mask).sum() / denom,
        }

    @staticmethod
    def _get_visualization_tfrecord_paths(tfrecord_path) -> list[str]:
        if isinstance(tfrecord_path, (list, tuple)):
            return list(tfrecord_path)
        return [tfrecord_path]

    def training_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        pred = self.encoder(tokenized_map, tokenized_agent)
        loss = self._flow_loss(pred, anchor_mask=pred["flow_train_mask"])
        future_metrics = self._future_metrics(
            pred,
            anchor_mask=pred["flow_train_mask"],
        )
        self.log("train/loss", loss, on_step=True, batch_size=1)
        self.log("train/ade_2s_m", future_metrics["ade_2s_m"], on_step=True, batch_size=1)
        self.log(
            "train/ade_yaw_2s_deg",
            future_metrics["ade_yaw_2s_deg"],
            on_step=True,
            batch_size=1,
        )
        return loss

    def validation_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)

        if self.val_open_loop:
            pred = self.encoder(tokenized_map, tokenized_agent)
            loss = self._flow_loss(pred, anchor_mask=pred["flow_eval_mask"])
            future_metrics = self._future_metrics(pred, anchor_mask=pred["flow_eval_mask"])
            self.log("val_open/loss", loss, on_epoch=True, sync_dist=True, batch_size=1)
            self.log(
                "val_open/ade_2s_m",
                future_metrics["ade_2s_m"],
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "val_open/ade_yaw_2s_deg",
                future_metrics["ade_yaw_2s_deg"],
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )

        if self.val_closed_loop:
            pred_traj, pred_z, pred_head = [], [], []
            for _ in range(self.n_rollout_closed_val):
                rollout = self.encoder.inference(
                    tokenized_map,
                    tokenized_agent,
                    self.validation_rollout_sampling,
                )
                pred_traj.append(rollout["pred_traj_10hz"])
                pred_z.append(rollout["pred_z_10hz"])
                pred_head.append(rollout["pred_head_10hz"])

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
                    for key in gpu_dict_sync.keys():
                        if isinstance(gpu_dict_sync[key], list):
                            gpu_dict_sync[key] = gpu_dict_sync[key][0]
                    scenario_rollouts = get_scenario_rollouts(**gpu_dict_sync)
                    self.wosac_submission.aggregate_rollouts(scenario_rollouts)
                self.wosac_submission.reset()
            else:
                self.minADE.update(
                    pred=pred_traj,
                    target=data["agent"]["position"][
                        :, self.num_historical_steps :, : pred_traj.shape[-1]
                    ],
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
                    tfrecord_paths = self._get_visualization_tfrecord_paths(
                        data["tfrecord_path"]
                    )
                    self.wosac_metrics.update(tfrecord_paths, scenario_rollouts)

            if self.global_rank == 0 and batch_idx < self.n_vis_batch:
                if scenario_rollouts is not None:
                    tfrecord_paths = self._get_visualization_tfrecord_paths(
                        data["tfrecord_path"]
                    )
                    n_vis_scenario = min(
                        self.n_vis_scenario,
                        len(tfrecord_paths),
                        len(scenario_rollouts),
                    )
                    for scenario_idx in range(n_vis_scenario):
                        vis = VisWaymo(
                            scenario_path=tfrecord_paths[scenario_idx],
                            save_dir=self.video_dir / f"batch_{batch_idx:02d}-scenario_{scenario_idx:02d}",
                        )
                        vis.save_video_scenario_rollout(
                            scenario_rollouts[scenario_idx],
                            self.n_vis_rollout,
                        )
                        for video_path in vis.video_paths:
                            self.logger.log_video("/".join(video_path.split("/")[-3:]), [video_path])

    def on_validation_epoch_end(self):
        if self.val_closed_loop:
            if not self.wosac_submission.is_active:
                epoch_wosac_metrics = self.wosac_metrics.compute()
                epoch_wosac_metrics["val_closed/ADE"] = self.minADE.compute()
                if self.global_rank == 0:
                    epoch_wosac_metrics["epoch"] = self.current_epoch
                    self.logger.log_metrics(epoch_wosac_metrics)
                self.wosac_metrics.reset()
                self.minADE.reset()

            if self.global_rank == 0 and self.wosac_submission.is_active:
                self.wosac_submission.save_sub_file()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        def lr_lambda(current_step):
            current_step = self.current_epoch + 1
            if current_step < self.lr_warmup_steps:
                return self.lr_min_ratio + (1 - self.lr_min_ratio) * current_step / self.lr_warmup_steps
            return self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * (
                1.0
                + math.cos(
                    math.pi
                    * min(
                        1.0,
                        (current_step - self.lr_warmup_steps)
                        / (self.lr_total_steps - self.lr_warmup_steps),
                    )
                )
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]

    def test_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        pred_traj, pred_z, pred_head = [], [], []
        for _ in range(self.n_rollout_closed_val):
            rollout = self.encoder.inference(
                tokenized_map,
                tokenized_agent,
                self.validation_rollout_sampling,
            )
            pred_traj.append(rollout["pred_traj_10hz"])
            pred_z.append(rollout["pred_z_10hz"])
            pred_head.append(rollout["pred_head_10hz"])

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
            for key in gpu_dict_sync.keys():
                if isinstance(gpu_dict_sync[key], list):
                    gpu_dict_sync[key] = gpu_dict_sync[key][0]
            scenario_rollouts = get_scenario_rollouts(**gpu_dict_sync)
            self.wosac_submission.aggregate_rollouts(scenario_rollouts)
        self.wosac_submission.reset()

    def on_test_epoch_end(self):
        if self.global_rank == 0:
            self.wosac_submission.save_sub_file()
