from __future__ import annotations

import math
from pathlib import Path

import hydra
import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR

from src.smart.metrics import WOSACMetrics, WOSACSubmission, minADE
from src.smart.metrics.flow_metrics import ade_2s, fde_2s, flow_matching_loss
from src.smart.modules.smart_flow_decoder import SMARTFlowDecoder
from src.smart.tokens.flow_token_processor import FlowTokenProcessor
from src.smart.utils.finetune import set_model_for_finetuning
from src.utils.vis_waymo import VisWaymo
from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts


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

    def _open_loop_loss_and_metrics(self, pred_dict):
        loss = flow_matching_loss(pred_dict["flow_pred_norm"], pred_dict["flow_target_norm"])
        ade = ade_2s(pred_dict["flow_pred_clean_norm"], pred_dict["flow_clean_norm"])
        fde = fde_2s(pred_dict["flow_pred_clean_norm"], pred_dict["flow_clean_norm"])
        return loss, ade, fde

    def training_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        pred = self.encoder(
            tokenized_map,
            tokenized_agent,
            anchor_mask_key="flow_train_mask",
        )
        loss, ade, fde = self._open_loop_loss_and_metrics(pred)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/ADE2s", ade, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log("train/FDE2s", fde, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        return loss

    def validation_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)

        if self.val_open_loop:
            pred = self.encoder(
                tokenized_map,
                tokenized_agent,
                anchor_mask_key="flow_eval_mask",
            )
            loss, ade, fde = self._open_loop_loss_and_metrics(pred)
            self.log("val_open/loss", loss, on_epoch=True, sync_dist=True, batch_size=1)
            self.log("val_open/ADE2s", ade, on_epoch=True, sync_dist=True, batch_size=1)
            self.log("val_open/FDE2s", fde, on_epoch=True, sync_dist=True, batch_size=1)

        if self.val_closed_loop:
            pred_traj, pred_z, pred_head = [], [], []
            for _ in range(self.n_rollout_closed_val):
                pred = self.encoder.inference(
                    tokenized_map,
                    tokenized_agent,
                    self.validation_rollout_sampling,
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
                gpu_dict = self.wosac_submission.compute()
                if self.global_rank == 0:
                    for k in gpu_dict.keys():
                        if isinstance(gpu_dict[k], list):
                            gpu_dict[k] = gpu_dict[k][0]
                    scenario_rollouts = get_scenario_rollouts(**gpu_dict)
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
                for scen_idx in range(self.n_vis_scenario):
                    vis = VisWaymo(
                        scenario_path=data["tfrecord_path"][scen_idx],
                        save_dir=self.video_dir / f"batch_{batch_idx:02d}-scenario_{scen_idx:02d}",
                    )
                    vis.save_video_scenario_rollout(scenario_rollouts[scen_idx], self.n_vis_rollout)
                    for video_path in vis.video_paths:
                        self.logger.log_video("/".join(video_path.split("/")[-3:]), [video_path])

    def on_validation_epoch_end(self):
        if self.val_closed_loop:
            if not self.wosac_submission.is_active:
                epoch_wosac_metrics = self.wosac_metrics.compute()
                epoch_wosac_metrics["val_closed/ADE"] = self.minADE.compute()
                self.log(
                    "val_closed/wosac/realism_meta_metric",
                    epoch_wosac_metrics["val_closed/wosac/realism_meta_metric"],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
                if self.global_rank == 0:
                    epoch_wosac_metrics["epoch"] = self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                    self.logger.log_metrics(epoch_wosac_metrics)
                self.wosac_metrics.reset()
                self.minADE.reset()
            if self.global_rank == 0 and self.wosac_submission.is_active:
                self.wosac_submission.save_sub_file()

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
        pred_traj, pred_z, pred_head = [], [], []
        for _ in range(self.n_rollout_closed_val):
            pred = self.encoder.inference(
                tokenized_map,
                tokenized_agent,
                self.validation_rollout_sampling,
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
        gpu_dict = self.wosac_submission.compute()
        if self.global_rank == 0:
            for k in gpu_dict.keys():
                if isinstance(gpu_dict[k], list):
                    gpu_dict[k] = gpu_dict[k][0]
            scenario_rollouts = get_scenario_rollouts(**gpu_dict)
            self.wosac_submission.aggregate_rollouts(scenario_rollouts)
        self.wosac_submission.reset()

    def on_test_epoch_end(self):
        if self.global_rank == 0:
            self.wosac_submission.save_sub_file()
