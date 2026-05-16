# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import gc
import hashlib
import math
from pathlib import Path
from typing import Dict, Sequence

import hydra
import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR

from src.smart.metrics import (
    CrossEntropy,
    SimAgentsSubmission,
    SimAgentsMetrics,
    TokenCls,
    WOSACDistributionMetrics,
    log_and_reset_wosac_distribution_metric,
    minADE,
    update_wosac_distribution_metric_from_model,
)
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils.finetune import set_model_for_finetuning
from src.utils.vis_waymo import VisWaymo
from src.utils.sim_agents_utils import get_scenario_id_int_tensor, get_scenario_rollouts


class SMART(LightningModule):

    def __init__(self, model_config) -> None:
        super(SMART, self).__init__()
        self.save_hyperparameters()
        self.lr = model_config.lr
        self.lr_warmup_steps = model_config.lr_warmup_steps
        self.lr_total_steps = model_config.lr_total_steps
        self.lr_min_ratio = model_config.lr_min_ratio
        self.num_historical_steps = model_config.decoder.num_historical_steps
        self.log_epoch = -1
        self.val_open_loop = model_config.val_open_loop
        self.val_closed_loop = model_config.val_closed_loop
        self.token_processor = TokenProcessor(**model_config.token_processor)

        self.encoder = SMARTDecoder(
            **model_config.decoder, n_token_agent=self.token_processor.n_token_agent
        )
        set_model_for_finetuning(self.encoder, model_config.finetune)

        self.minADE = minADE()
        self.TokenCls = TokenCls(max_guesses=5)
        self.sim_agents_metrics = SimAgentsMetrics("val_closed")
        self.sim_agents_submission = SimAgentsSubmission(
            **model_config.sim_agents_submission
        )
        wosac_cpd_reference = getattr(model_config, "wosac_cpd_reference", None)
        self.wosac_distribution_metrics = WOSACDistributionMetrics(
            "val_closed",
            cpd_reference=wosac_cpd_reference,
        )
        self.test_wosac_distribution_metrics = WOSACDistributionMetrics(
            "test",
            cpd_reference=wosac_cpd_reference,
        )
        self.training_loss = CrossEntropy(**model_config.training_loss)

        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self.n_vis_batch = model_config.n_vis_batch
        self.n_vis_scenario = model_config.n_vis_scenario
        self.n_vis_rollout = model_config.n_vis_rollout
        self.n_batch_sim_agents_metric = int(
            getattr(
                model_config,
                "n_batch_sim_agents_metric",
                getattr(model_config, "n_batch_wosac_metric", 10),
            )
        )
        self.scorer_scene_num = getattr(model_config, "scorer_scene_num", None)
        self._scorer_scene_num_last_key: tuple[int, int, int] | None = None
        self.closed_loop_metric_name = "val_closed/sim_agents_2025/realism_meta_metric"
        self.val_closed_minade_name = (
            "val_closed/sim_agents_2025/minADE_best_of_n_rollout_closed_val"
        )

        self.video_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.video_dir = Path(self.video_dir) / "videos"

        self.training_rollout_sampling = model_config.training_rollout_sampling
        self.validation_rollout_sampling = model_config.validation_rollout_sampling
        self.validation_closed_seed = int(
            getattr(model_config, "validation_closed_seed", 0)
        )

    @staticmethod
    def _repeat_tensor_on_first_dim(tensor: torch.Tensor, repeat_count: int) -> torch.Tensor:
        if repeat_count == 1:
            return tensor
        repeat_pattern = (repeat_count,) + (1,) * tensor.dim()
        return tensor.unsqueeze(0).repeat(repeat_pattern).flatten(0, 1).contiguous()

    @staticmethod
    def _expand_batch_index_for_rollouts(
        batch_index: torch.Tensor,
        repeat_count: int,
        num_graphs: int,
    ) -> torch.Tensor:
        if repeat_count == 1:
            return batch_index
        rollout_offsets = (
            torch.arange(repeat_count, device=batch_index.device, dtype=batch_index.dtype)
            * int(num_graphs)
        )
        expanded_batch = batch_index.unsqueeze(0).repeat(repeat_count, 1)
        expanded_batch = expanded_batch + rollout_offsets.unsqueeze(1)
        return expanded_batch.reshape(-1).contiguous()

    def _make_closed_loop_seed(self, scenario_id: str, rollout_idx: int) -> int:
        seed_payload = (
            f"{self.validation_closed_seed}:{scenario_id}:{int(rollout_idx)}".encode(
                "utf-8"
            )
        )
        digest = hashlib.blake2b(seed_payload, digest_size=8).digest()
        return (
            int.from_bytes(digest, byteorder="little", signed=False)
            & 0x7FFF_FFFF_FFFF_FFFF
        )

    def _build_closed_loop_seed_table(
        self,
        scenario_ids: Sequence[str],
        rollout_indices: Sequence[int],
        device: torch.device,
    ) -> torch.Tensor:
        seed_rows = [
            [
                self._make_closed_loop_seed(
                    scenario_id=str(scenario_id),
                    rollout_idx=rollout_idx,
                )
                for scenario_id in scenario_ids
            ]
            for rollout_idx in rollout_indices
        ]
        if len(seed_rows) == 0:
            return torch.zeros((0, len(scenario_ids)), dtype=torch.long, device=device)
        return torch.tensor(seed_rows, dtype=torch.long, device=device)

    def _resolve_val_batch_size(self) -> int | None:
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
        if self.sim_agents_submission.is_active:
            return

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
        self.n_batch_sim_agents_metric = max(
            1,
            math.ceil(per_rank_scenes / val_batch_size),
        )

        current_key = (int(scorer_scene_num), int(world_size), int(val_batch_size))
        if self._scorer_scene_num_last_key == current_key:
            return
        self._scorer_scene_num_last_key = current_key
        if getattr(trainer, "is_global_zero", True):
            print(
                "[scorer_scene_num] Fast WOSAC sim_agents_2025 scorer batch count set to "
                f"n_batch_sim_agents_metric={self.n_batch_sim_agents_metric} "
                f"(requested_scenes={scorer_scene_num}, world_size={world_size}, "
                f"val_batch_size={val_batch_size}).",
                flush=True,
            )

    def on_fit_start(self) -> None:
        self._apply_scorer_scene_num_overrides()

    def on_validation_start(self) -> None:
        self._apply_scorer_scene_num_overrides()

    def _build_parallel_rollout_map_feature(
        self,
        map_feature: Dict[str, torch.Tensor],
        repeat_count: int,
        num_graphs: int,
    ) -> Dict[str, torch.Tensor]:
        if repeat_count == 1:
            return map_feature

        expanded_map_feature = {
            "pt_token": self._repeat_tensor_on_first_dim(map_feature["pt_token"], repeat_count),
            "position": self._repeat_tensor_on_first_dim(map_feature["position"], repeat_count),
            "orientation": self._repeat_tensor_on_first_dim(
                map_feature["orientation"],
                repeat_count,
            ),
            "light_type": self._repeat_tensor_on_first_dim(map_feature["light_type"], repeat_count),
            "batch": self._expand_batch_index_for_rollouts(
                map_feature["batch"],
                repeat_count=repeat_count,
                num_graphs=num_graphs,
            ),
        }
        return expanded_map_feature

    def _build_parallel_rollout_tokenized_agent(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        repeat_count: int,
        num_graphs: int,
    ) -> Dict[str, torch.Tensor]:
        if repeat_count == 1:
            return tokenized_agent

        return {
            "num_graphs": int(num_graphs) * repeat_count,
            "type": self._repeat_tensor_on_first_dim(tokenized_agent["type"], repeat_count),
            "shape": self._repeat_tensor_on_first_dim(tokenized_agent["shape"], repeat_count),
            "token_agent_shape": self._repeat_tensor_on_first_dim(
                tokenized_agent["token_agent_shape"],
                repeat_count,
            ),
            "batch": self._expand_batch_index_for_rollouts(
                tokenized_agent["batch"],
                repeat_count=repeat_count,
                num_graphs=num_graphs,
            ),
            "token_traj_all": self._repeat_tensor_on_first_dim(
                tokenized_agent["token_traj_all"],
                repeat_count,
            ),
            "token_traj": self._repeat_tensor_on_first_dim(
                tokenized_agent["token_traj"],
                repeat_count,
            ),
            "trajectory_token_veh": tokenized_agent["trajectory_token_veh"],
            "trajectory_token_ped": tokenized_agent["trajectory_token_ped"],
            "trajectory_token_cyc": tokenized_agent["trajectory_token_cyc"],
            "gt_pos_raw": self._repeat_tensor_on_first_dim(
                tokenized_agent["gt_pos_raw"],
                repeat_count,
            ),
            "gt_head_raw": self._repeat_tensor_on_first_dim(
                tokenized_agent["gt_head_raw"],
                repeat_count,
            ),
            "gt_valid_raw": self._repeat_tensor_on_first_dim(
                tokenized_agent["gt_valid_raw"],
                repeat_count,
            ),
            "gt_z_raw": self._repeat_tensor_on_first_dim(
                tokenized_agent["gt_z_raw"],
                repeat_count,
            ),
            "valid_mask": self._repeat_tensor_on_first_dim(
                tokenized_agent["valid_mask"],
                repeat_count,
            ),
            "gt_idx": self._repeat_tensor_on_first_dim(tokenized_agent["gt_idx"], repeat_count),
            "gt_pos": self._repeat_tensor_on_first_dim(tokenized_agent["gt_pos"], repeat_count),
            "gt_heading": self._repeat_tensor_on_first_dim(
                tokenized_agent["gt_heading"],
                repeat_count,
            ),
        }

    @staticmethod
    def _reshape_parallel_rollout_prediction(
        pred_tensor: torch.Tensor,
        repeat_count: int,
        num_agent: int,
    ) -> torch.Tensor:
        pred_tensor = pred_tensor.reshape(repeat_count, num_agent, *pred_tensor.shape[1:])
        permute_order = (1, 0) + tuple(range(2, pred_tensor.dim()))
        return pred_tensor.permute(*permute_order).contiguous()

    def _run_parallel_rollout_chunk(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        scenario_ids: Sequence[str],
        rollout_indices: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk_size = int(len(rollout_indices))
        if chunk_size <= 0:
            raise ValueError("rollout_indices must contain at least one rollout index.")

        num_agent = int(tokenized_agent["batch"].shape[0])
        num_graphs = int(tokenized_agent["num_graphs"])
        rollout_map_feature = self._build_parallel_rollout_map_feature(
            map_feature=map_feature,
            repeat_count=chunk_size,
            num_graphs=num_graphs,
        )
        rollout_tokenized_agent = self._build_parallel_rollout_tokenized_agent(
            tokenized_agent=tokenized_agent,
            repeat_count=chunk_size,
            num_graphs=num_graphs,
        )
        scenario_seed_table = self._build_closed_loop_seed_table(
            scenario_ids=scenario_ids,
            rollout_indices=rollout_indices,
            device=tokenized_agent["batch"].device,
        )
        pred = self.encoder.agent_encoder.inference(
            rollout_tokenized_agent,
            rollout_map_feature,
            self.validation_rollout_sampling,
            scenario_sampling_seeds=scenario_seed_table.reshape(-1).contiguous(),
        )
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
        )

    def _build_rollout_chunk_size_candidates(self) -> list[int]:
        chunk_sizes: list[int] = []
        current = max(1, int(self.n_rollout_closed_val))
        while True:
            if current not in chunk_sizes:
                chunk_sizes.append(current)
            if current == 1:
                break
            current = max(1, math.ceil(current / 2))
        return chunk_sizes

    @staticmethod
    def _is_cuda_out_of_memory(error: RuntimeError) -> bool:
        error_message = str(error).lower()
        oom_patterns = (
            "out of memory",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
        )
        return any(pattern in error_message for pattern in oom_patterns)

    @staticmethod
    def _cleanup_after_rollout_oom() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_closed_loop_rollouts(
        self,
        tokenized_map: Dict[str, torch.Tensor],
        tokenized_agent: Dict[str, torch.Tensor],
        scenario_ids: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        repeat_count = int(self.n_rollout_closed_val)
        if repeat_count <= 0:
            raise ValueError(
                f"n_rollout_closed_val must be positive, got {self.n_rollout_closed_val}."
            )

        num_graphs = int(tokenized_agent["num_graphs"])
        if len(scenario_ids) != num_graphs:
            raise ValueError(
                "scenario_ids length must match tokenized_agent['num_graphs'], "
                f"got {len(scenario_ids)} and {num_graphs}."
            )
        map_feature = self.encoder.map_encoder(tokenized_map)
        rollout_indices = list(range(repeat_count))
        last_oom_error: RuntimeError | None = None

        for chunk_size in self._build_rollout_chunk_size_candidates():
            pred_traj_chunks: list[torch.Tensor] = []
            pred_z_chunks: list[torch.Tensor] = []
            pred_head_chunks: list[torch.Tensor] = []
            try:
                for chunk_start in range(0, repeat_count, chunk_size):
                    chunk_rollout_indices = rollout_indices[
                        chunk_start : chunk_start + chunk_size
                    ]
                    chunk_pred_traj, chunk_pred_z, chunk_pred_head = (
                        self._run_parallel_rollout_chunk(
                            tokenized_agent=tokenized_agent,
                            map_feature=map_feature,
                            scenario_ids=scenario_ids,
                            rollout_indices=chunk_rollout_indices,
                        )
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
        raise RuntimeError("closed-loop rollout failed before producing predictions.")

    def training_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        if self.training_rollout_sampling.num_k <= 0:
            pred = self.encoder(tokenized_map, tokenized_agent)
        else:
            pred = self.encoder.inference(
                tokenized_map,
                tokenized_agent,
                sampling_scheme=self.training_rollout_sampling,
            )

        train_mask = (
            data["agent"]["train_mask"] if "train_mask" in data["agent"] else None
        )
        loss = self.training_loss(
            **pred,
            token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
            token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
            train_mask=train_mask,  # [n_agent]
            current_epoch=self.current_epoch,
        )
        self.log("train/loss", loss, on_step=True, batch_size=1)

        return loss

    def validation_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)

        # ! open-loop vlidation
        if self.val_open_loop:
            pred = self.encoder(tokenized_map, tokenized_agent)
            loss = self.training_loss(
                **pred,
                token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
                token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
            )

            self.TokenCls.update(
                # action that goes from [(10->15), ..., (85->90)]
                pred=pred["next_token_logits"],  # [n_agent, 16, n_token]
                pred_valid=pred["next_token_valid"],  # [n_agent, 16]
                target=tokenized_agent["gt_idx"][:, 2:],
                target_valid=tokenized_agent["valid_mask"][:, 2:],
            )
            self.log(
                "val_open/acc",
                self.TokenCls,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log("val_open/loss", loss, on_epoch=True, sync_dist=True, batch_size=1)

        # ! closed-loop vlidation
        if self.val_closed_loop:
            pred_traj, pred_z, pred_head = self._run_closed_loop_rollouts(
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
                scenario_ids=data["scenario_id"],
            )

            update_wosac_distribution_metric_from_model(
                metric=self.wosac_distribution_metrics,
                model=self,
                data=data,
                pred_traj=pred_traj,
                include_gt=True,
            )

            # ! Sim Agents submission / metrics
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

            else:  # ! compute metrics, disable if saving Sim Agents submission
                self.minADE.update(
                    pred=pred_traj,
                    target=data["agent"]["position"][
                        :, self.num_historical_steps :, : pred_traj.shape[-1]
                    ],
                    target_valid=data["agent"]["valid_mask"][
                        :, self.num_historical_steps :
                    ],
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

            # ! visualization
            if self.global_rank == 0 and batch_idx < self.n_vis_batch:
                if scenario_rollouts is None:
                    device = pred_traj.device
                    scenario_rollouts = get_scenario_rollouts(
                        scenario_id=get_scenario_id_int_tensor(
                            data["scenario_id"], device
                        ),
                        agent_id=data["agent"]["id"],
                        agent_batch=data["agent"]["batch"],
                        pred_traj=pred_traj,
                        pred_z=pred_z,
                        pred_head=pred_head,
                    )
                if scenario_rollouts is not None:
                    for _i_sc in range(self.n_vis_scenario):
                        _vis = VisWaymo(
                            scenario_path=data["tfrecord_path"][_i_sc],
                            save_dir=self.video_dir
                            / f"batch_{batch_idx:02d}-scenario_{_i_sc:02d}",
                        )
                        _vis.save_video_scenario_rollout(
                            scenario_rollouts[_i_sc], self.n_vis_rollout
                        )
                        for _path in _vis.video_paths:
                            self.logger.log_video(
                                "/".join(_path.split("/")[-3:]), [_path]
                            )

    def on_validation_epoch_end(self):
        if self.val_closed_loop:
            epoch_distribution_metrics = log_and_reset_wosac_distribution_metric(
                self.wosac_distribution_metrics
            )
            if not self.sim_agents_submission.is_active:
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    reduced_metric_state = self.sim_agents_metrics.get_state_tensor(
                        device=self.device
                    )
                    torch.distributed.all_reduce(reduced_metric_state)
                    epoch_sim_agents_metrics = (
                        self.sim_agents_metrics.compute_from_state_tensor(
                            reduced_metric_state
                        )
                    )
                    reduced_minade_state = torch.stack(
                        [
                            self.minADE.sum.detach().to(device=self.device),
                            self.minADE.count.detach().to(device=self.device),
                        ]
                    )
                    torch.distributed.all_reduce(reduced_minade_state)
                    minade_value = reduced_minade_state[0] / reduced_minade_state[
                        1
                    ].clamp_min(1e-6)
                else:
                    epoch_sim_agents_metrics = self.sim_agents_metrics.compute()
                    minade_value = self.minADE.sum / self.minADE.count.clamp_min(1e-6)

                closed_loop_metric = epoch_sim_agents_metrics[
                    self.closed_loop_metric_name
                ]
                epoch_sim_agents_metrics[self.val_closed_minade_name] = minade_value
                epoch_sim_agents_metrics.update(epoch_distribution_metrics)
                self.log(
                    self.closed_loop_metric_name,
                    closed_loop_metric,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
                if self.global_rank == 0:
                    epoch_sim_agents_metrics["epoch"] = (
                        self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                    )
                    self.logger.log_metrics(epoch_sim_agents_metrics)

                self.sim_agents_metrics.reset()
                self.minADE.reset()

            if self.sim_agents_submission.is_active:
                if self.global_rank == 0 and epoch_distribution_metrics:
                    epoch_distribution_metrics["epoch"] = (
                        self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                    )
                    self.logger.log_metrics(epoch_distribution_metrics)
                self.sim_agents_submission.save_sub_file()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        def lr_lambda(current_step):
            current_step = self.current_epoch + 1
            if current_step < self.lr_warmup_steps:
                return (
                    self.lr_min_ratio
                    + (1 - self.lr_min_ratio) * current_step / self.lr_warmup_steps
                )
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

        # ! only closed-loop vlidation
        pred_traj, pred_z, pred_head = self._run_closed_loop_rollouts(
            tokenized_map=tokenized_map,
            tokenized_agent=tokenized_agent,
            scenario_ids=data["scenario_id"],
        )

        update_wosac_distribution_metric_from_model(
            metric=self.test_wosac_distribution_metrics,
            model=self,
            data=data,
            pred_traj=pred_traj,
            include_gt=False,
        )

        # ! Sim Agents submission save
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
        epoch_distribution_metrics = log_and_reset_wosac_distribution_metric(
            self.test_wosac_distribution_metrics
        )
        if self.global_rank == 0:
            if epoch_distribution_metrics:
                self.logger.log_metrics(epoch_distribution_metrics)
        self.sim_agents_submission.save_sub_file()
