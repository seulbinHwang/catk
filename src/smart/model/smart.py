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

import copy
import gc
import hashlib
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Sequence

import hydra
import torch
from lightning import LightningModule
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from waymo_open_dataset.utils.sim_agents import submission_specs

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
from src.smart.metrics.rlftsim_reward import RLFTSimMLOOReward
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils.finetune import set_model_for_finetuning
from src.utils.vis_waymo import VisWaymo
from src.utils.sim_agents_utils import get_scenario_id_int_tensor, get_scenario_rollouts


class SMART(LightningModule):
    _RLFTSIM_REFERENCE_STATE_PREFIX = "_rlftsim_ref_encoder."

    @classmethod
    def _drop_rlftsim_reference_state(
        cls,
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not any(key.startswith(cls._RLFTSIM_REFERENCE_STATE_PREFIX) for key in state_dict):
            return state_dict
        filtered_state_dict = state_dict.__class__(
            (key, value)
            for key, value in state_dict.items()
            if not key.startswith(cls._RLFTSIM_REFERENCE_STATE_PREFIX)
        )
        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            filtered_metadata = metadata.__class__(
                (key, value)
                for key, value in metadata.items()
                if not key.startswith(cls._RLFTSIM_REFERENCE_STATE_PREFIX)
            )
            filtered_state_dict._metadata = filtered_metadata
        return filtered_state_dict

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        state_dict = self._drop_rlftsim_reference_state(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def on_save_checkpoint(self, checkpoint) -> None:
        state_dict = checkpoint.get("state_dict")
        if state_dict is not None:
            checkpoint["state_dict"] = self._drop_rlftsim_reference_state(state_dict)

    def on_load_checkpoint(self, checkpoint) -> None:
        state_dict = checkpoint.get("state_dict")
        if state_dict is not None:
            checkpoint["state_dict"] = self._drop_rlftsim_reference_state(state_dict)

    @staticmethod
    def _required_sim_agents_rollout_count() -> int:
        submission_config = submission_specs.get_submission_config(
            submission_specs.ChallengeType.SIM_AGENTS
        )
        return int(submission_config.n_rollouts)

    @staticmethod
    def _check_sim_agents_submission_rollout_count(
        is_active: bool,
        n_rollout_closed_val: int,
    ) -> None:
        if not is_active:
            return
        expected_rollouts = SMART._required_sim_agents_rollout_count()
        if int(n_rollout_closed_val) != expected_rollouts:
            raise ValueError(
                "Sim Agents 2025 submission export requires "
                f"n_rollout_closed_val={expected_rollouts}, "
                f"got {n_rollout_closed_val}."
            )

    def __init__(self, model_config) -> None:
        super(SMART, self).__init__()
        self.save_hyperparameters()
        self.lr = model_config.lr
        self.lr_warmup_steps = model_config.lr_warmup_steps
        self.lr_total_steps = model_config.lr_total_steps
        self.lr_min_ratio = model_config.lr_min_ratio
        self.weight_decay = float(getattr(model_config, "weight_decay", 0.01))
        self.num_historical_steps = model_config.decoder.num_historical_steps
        self.log_epoch = -1
        self.val_open_loop = model_config.val_open_loop
        self.val_closed_loop = model_config.val_closed_loop
        self.token_processor = TokenProcessor(**model_config.token_processor)

        self.encoder = SMARTDecoder(
            **model_config.decoder, n_token_agent=self.token_processor.n_token_agent
        )
        set_model_for_finetuning(
            self.encoder,
            model_config.finetune,
            getattr(model_config, "finetune_freeze_mode", "legacy"),
        )

        self.minADE = minADE()
        self.TokenCls = TokenCls(max_guesses=5)
        self.sim_agents_metrics = SimAgentsMetrics("val_closed")
        self.sim_agents_submission = SimAgentsSubmission(
            **model_config.sim_agents_submission
        )
        wosac_cpd_reference = getattr(model_config, "wosac_cpd_reference", None)
        wosac_distribution_type_scale = getattr(
            model_config,
            "wosac_distribution_type_scale",
            None,
        )
        self.wosac_distribution_metrics = WOSACDistributionMetrics(
            "val_closed",
            cpd_reference=wosac_cpd_reference,
            type_scale=wosac_distribution_type_scale,
        )
        self.test_wosac_distribution_metrics = WOSACDistributionMetrics(
            "test",
            cpd_reference=wosac_cpd_reference,
            type_scale=wosac_distribution_type_scale,
        )
        self.training_loss = CrossEntropy(**model_config.training_loss)

        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self._check_sim_agents_submission_rollout_count(
            is_active=bool(self.sim_agents_submission.is_active),
            n_rollout_closed_val=int(self.n_rollout_closed_val),
        )
        self.n_vis_batch = model_config.n_vis_batch
        self.n_vis_scenario = model_config.n_vis_scenario
        self.n_vis_rollout = model_config.n_vis_rollout
        self.delete_local_videos_after_wandb_upload = bool(
            getattr(model_config, "delete_local_videos_after_wandb_upload", True)
        )
        self.n_batch_sim_agents_metric = int(
            getattr(
                model_config,
                "n_batch_sim_agents_metric",
                getattr(model_config, "n_batch_wosac_metric", 10),
            )
        )
        self.scorer_scene_num = getattr(model_config, "scorer_scene_num", None)
        self._scorer_scene_num_last_key: tuple[int, int, int] | None = None
        self.fit_time_fast_validation_only = bool(
            getattr(model_config, "fit_time_fast_validation_only", False)
        )
        self._fit_time_original_limit_val_batches: int | float | None = None
        self._fit_time_fast_validation_enabled = False
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
        self.rlftsim_config = getattr(model_config, "rlftsim", None)
        self.rlftsim_enabled = bool(
            getattr(self.rlftsim_config, "enabled", False)
            if self.rlftsim_config is not None
            else False
        )
        self.rlftsim_reward = (
            RLFTSimMLOOReward(
                ego_only=bool(getattr(self.rlftsim_config, "ego_only", False)),
                version=str(getattr(self.rlftsim_config, "wosac_version", "2025")),
            )
            if self.rlftsim_enabled
            else None
        )
        self.rlftsim_rollouts_per_scenario = int(
            getattr(self.rlftsim_config, "train_rollouts_per_scenario", 4)
            if self.rlftsim_config is not None
            else 4
        )
        self.rlftsim_sampling = (
            getattr(self.rlftsim_config, "sampling", self.validation_rollout_sampling)
            if self.rlftsim_config is not None
            else self.validation_rollout_sampling
        )
        self.rlftsim_reward_scale = float(
            getattr(self.rlftsim_config, "reward_scale", 1.0)
            if self.rlftsim_config is not None
            else 1.0
        )
        self.rlftsim_normalize_rewards = bool(
            getattr(self.rlftsim_config, "normalize_rewards", False)
            if self.rlftsim_config is not None
            else False
        )
        self.rlftsim_entropy_bonus = float(
            getattr(self.rlftsim_config, "entropy_bonus", 0.0)
            if self.rlftsim_config is not None
            else 0.0
        )
        self.rlftsim_kl_target = float(
            getattr(self.rlftsim_config, "kl_target", 0.01)
            if self.rlftsim_config is not None
            else 0.01
        )
        self.rlftsim_kl_horizon = float(
            getattr(self.rlftsim_config, "kl_horizon", 5.0)
            if self.rlftsim_config is not None
            else 5.0
        )
        self.rlftsim_kl_min = float(
            getattr(self.rlftsim_config, "kl_min", 1.0e-3)
            if self.rlftsim_config is not None
            else 1.0e-3
        )
        self.rlftsim_kl_max = float(
            getattr(self.rlftsim_config, "kl_max", 1.0e3)
            if self.rlftsim_config is not None
            else 1.0e3
        )
        self.rlftsim_kl_beta = float(
            getattr(self.rlftsim_config, "kl_initial_beta", 1.0e-2)
            if self.rlftsim_config is not None
            else 1.0e-2
        )
        self.rlftsim_ref_sync_steps = int(
            getattr(self.rlftsim_config, "reference_sync_steps", 500)
            if self.rlftsim_config is not None
            else 500
        )
        self.rlftsim_ref_sync_alpha = float(
            getattr(self.rlftsim_config, "reference_sync_alpha", 0.005)
            if self.rlftsim_config is not None
            else 0.005
        )
        self.rlftsim_accumulate_grad_batches = int(
            getattr(self.rlftsim_config, "accumulate_grad_batches", 1)
            if self.rlftsim_config is not None
            else 1
        )
        self.rlftsim_gradient_clip_val = float(
            getattr(self.rlftsim_config, "gradient_clip_val", 1.0)
            if self.rlftsim_config is not None
            else 1.0
        )
        self.rlftsim_gradient_clip_algorithm = str(
            getattr(self.rlftsim_config, "gradient_clip_algorithm", "norm")
            if self.rlftsim_config is not None
            else "norm"
        )
        self.rlftsim_replay_rollout_chunk_size = int(
            getattr(self.rlftsim_config, "replay_rollout_chunk_size", 1)
            if self.rlftsim_config is not None
            else 1
        )
        if self.rlftsim_enabled and self.rlftsim_rollouts_per_scenario < 2:
            raise ValueError("RLFTSim MLOO requires train_rollouts_per_scenario >= 2.")
        if self.rlftsim_enabled and self.rlftsim_accumulate_grad_batches < 1:
            raise ValueError("rlftsim.accumulate_grad_batches must be >= 1.")
        if self.rlftsim_enabled and self.rlftsim_replay_rollout_chunk_size < 1:
            raise ValueError("rlftsim.replay_rollout_chunk_size must be >= 1.")
        if self.rlftsim_enabled and self.rlftsim_sampling.criterium not in {
            "categorical",
            "full_prob",
            "topk_prob",
        }:
            raise ValueError(
                "RLFTSim goal-free policy sampling must not depend on GT distance. "
                "Use criterium=categorical, full_prob, or topk_prob."
            )
        if self.rlftsim_enabled:
            self.automatic_optimization = False
        self._rlftsim_ref_encoder = None
        self._last_rlftsim_kl: float | None = None
        self._rlftsim_kl_accum_sum = 0.0
        self._rlftsim_kl_accum_count = 0

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

    def _should_enable_fit_time_fast_validation(self) -> bool:
        return (
            self.fit_time_fast_validation_only
            and self.val_closed_loop
            and not self.val_open_loop
            and not self.sim_agents_submission.is_active
            and int(self.n_batch_sim_agents_metric) > 0
        )

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

    def _warn_if_fit_time_fast_validation_inactive(self) -> None:
        """``fit_time_fast_validation_only=True``로 켰지만 조건이 안 맞아 활성화되지
        못한 경우를 한 번만 알린다. README §"학습 중 checkpoint 확인" 예시는
        ``val_open_loop=false``를 같이 끄도록 안내하지만 둘 중 한쪽만 켜면 fast
        모드가 silently OFF가 된다. 사용자가 "왜 안 빨라지지?" 디버깅하지 않게 돕는다.
        """
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        if not getattr(trainer, "is_global_zero", True):
            return
        reasons: list[str] = []
        if not self.val_closed_loop:
            reasons.append("val_closed_loop=false")
        if self.val_open_loop:
            reasons.append("val_open_loop=true (false로 끄세요)")
        if self.sim_agents_submission.is_active:
            reasons.append("sim_agents_submission.is_active=true (submission 모드에선 의미 없음)")
        if int(self.n_batch_sim_agents_metric) <= 0:
            reasons.append("n_batch_sim_agents_metric<=0")
        print(
            "[fit_time_fast_validation_only] 옵션이 켜져 있지만 활성 조건이 충족되지 "
            f"않아 fast 모드가 적용되지 않았습니다. 원인: {', '.join(reasons)}.",
            flush=True,
        )

    def _apply_fit_time_validation_batch_limit(self) -> None:
        if not self._should_enable_fit_time_fast_validation():
            self._fit_time_fast_validation_enabled = False
            if self.fit_time_fast_validation_only:
                self._warn_if_fit_time_fast_validation_inactive()
            return

        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return

        if self._fit_time_original_limit_val_batches is None:
            self._fit_time_original_limit_val_batches = trainer.limit_val_batches

        target_batches = int(self.n_batch_sim_agents_metric)
        trainer.limit_val_batches = target_batches
        self._fit_time_fast_validation_enabled = True
        if getattr(trainer, "is_global_zero", True):
            print(
                "[fit_time_fast_validation_only] Fit-time validation is limited to "
                f"{target_batches} batch(es) per rank for fast checkpoint scoring. "
                "Run validate/test without this option for full validation.",
                flush=True,
            )

    def _restore_fit_time_validation_batch_limit(self) -> None:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            self._fit_time_fast_validation_enabled = False
            return

        if self._fit_time_original_limit_val_batches is not None:
            trainer.limit_val_batches = self._fit_time_original_limit_val_batches

        self._fit_time_original_limit_val_batches = None
        self._fit_time_fast_validation_enabled = False

    def _should_compute_closed_loop_minade(self) -> bool:
        return not self._fit_time_fast_validation_enabled

    def _get_video_logger(self):
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            for logger in getattr(trainer, "loggers", []) or []:
                if hasattr(logger, "log_video"):
                    return logger
        logger = getattr(self, "logger", None)
        if hasattr(logger, "log_video"):
            return logger
        return None

    def _log_metrics_to_logger(self, metrics: Dict[str, object]) -> None:
        logger = getattr(self, "logger", None)
        if logger is not None and hasattr(logger, "log_metrics"):
            logger.log_metrics(metrics)

    def _cleanup_local_video(self, video_path: str) -> None:
        video_file = Path(video_path)
        try:
            video_file.resolve().relative_to(self.video_dir.resolve())
        except ValueError:
            return

        video_file.unlink(missing_ok=True)
        current_dir = video_file.parent
        while current_dir != self.video_dir.parent:
            try:
                current_dir.rmdir()
            except OSError:
                break
            current_dir = current_dir.parent

    def _ensure_tokenized_agent_z_raw(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        data,
    ) -> None:
        if "gt_z_raw" not in tokenized_agent:
            tokenized_agent["gt_z_raw"] = data["agent"]["position"][
                :, self.num_historical_steps - 1, 2
            ]

    def _init_rlftsim_reference(self) -> None:
        if not self.rlftsim_enabled or self._rlftsim_ref_encoder is not None:
            return
        self._rlftsim_ref_encoder = copy.deepcopy(self.encoder)
        self._rlftsim_ref_encoder.eval()
        for parameter in self._rlftsim_ref_encoder.parameters():
            parameter.requires_grad_(False)

    def _sync_rlftsim_reference(self) -> None:
        if not self.rlftsim_enabled or self._rlftsim_ref_encoder is None:
            return
        alpha = float(self.rlftsim_ref_sync_alpha)
        if alpha <= 0.0:
            return
        alpha = min(1.0, alpha)
        with torch.no_grad():
            ref_state = self._rlftsim_ref_encoder.state_dict()
            cur_state = self.encoder.state_dict()
            for name, ref_value in ref_state.items():
                cur_value = cur_state[name].detach().to(device=ref_value.device)
                if torch.is_floating_point(ref_value):
                    ref_value.mul_(1.0 - alpha).add_(cur_value, alpha=alpha)
                else:
                    ref_value.copy_(cur_value)

    def _update_rlftsim_kl_controller(self) -> None:
        if not self.rlftsim_enabled or self._last_rlftsim_kl is None:
            return
        if self.rlftsim_kl_target <= 0.0 or self.rlftsim_kl_horizon <= 0.0:
            return
        proportional_error = self._last_rlftsim_kl / self.rlftsim_kl_target - 1.0
        self.rlftsim_kl_beta *= math.exp(proportional_error / self.rlftsim_kl_horizon)
        self.rlftsim_kl_beta = min(
            self.rlftsim_kl_max,
            max(self.rlftsim_kl_min, self.rlftsim_kl_beta),
        )

    def _record_rlftsim_kl_for_controller(self, kl: torch.Tensor) -> None:
        self._rlftsim_kl_accum_sum += float(kl.detach().item())
        self._rlftsim_kl_accum_count += 1

    def _consume_rlftsim_kl_for_controller(self) -> None:
        if self._rlftsim_kl_accum_count <= 0:
            return
        self._last_rlftsim_kl = (
            self._rlftsim_kl_accum_sum / float(self._rlftsim_kl_accum_count)
        )
        self._rlftsim_kl_accum_sum = 0.0
        self._rlftsim_kl_accum_count = 0

    def _is_rlftsim_optimizer_step_boundary(self, batch_idx: int) -> bool:
        accumulate_grad_batches = int(self.rlftsim_accumulate_grad_batches)
        if accumulate_grad_batches <= 1:
            return True
        if (int(batch_idx) + 1) % accumulate_grad_batches == 0:
            return True
        trainer = getattr(self, "trainer", None)
        num_training_batches = getattr(trainer, "num_training_batches", None)
        if isinstance(num_training_batches, int):
            return int(batch_idx) + 1 >= num_training_batches
        return False

    def _validate_rlftsim_batch(self, data) -> None:
        if "tfrecord_path" not in data:
            raise RuntimeError(
                "RLFTSim training requires data['tfrecord_path'] so fast RMM can "
                "load the matching Waymo scenario proto. Set "
                "data.train_tfrecords_splitted to a split TFRecord directory."
            )
        tfrecord_paths = data["tfrecord_path"]
        if isinstance(tfrecord_paths, str):
            tfrecord_paths = [tfrecord_paths]
        missing_paths = [
            str(path) for path in tfrecord_paths if not Path(str(path)).is_file()
        ]
        if missing_paths:
            preview = ", ".join(missing_paths[:3])
            raise FileNotFoundError(
                "RLFTSim training could not find split TFRecord file(s): "
                f"{preview}. Set data.train_tfrecords_splitted correctly."
            )

    def on_fit_start(self) -> None:
        self._init_rlftsim_reference()
        self._apply_scorer_scene_num_overrides()
        self._apply_fit_time_validation_batch_limit()

    def on_fit_end(self) -> None:
        self._restore_fit_time_validation_batch_limit()

    def on_train_epoch_start(self) -> None:
        # training_loss는 torchmetrics.Metric이라 forward 호출마다 loss_sum/count
        # 내부 상태가 누적된다. train과 val 양쪽에서 같은 인스턴스를 호출하므로,
        # phase 시작 시 한 번씩 reset해서 train/val 상태가 섞이거나 epoch 간에
        # 무한 누적되지 않도록 한다.
        self.training_loss.reset()

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if not self.rlftsim_enabled:
            return
        if not self._is_rlftsim_optimizer_step_boundary(batch_idx):
            return
        self._consume_rlftsim_kl_for_controller()
        self._update_rlftsim_kl_controller()
        if self.rlftsim_ref_sync_steps <= 0:
            return
        if self.global_step <= 0:
            return
        if self.global_step % self.rlftsim_ref_sync_steps == 0:
            self._sync_rlftsim_reference()

    def on_validation_start(self) -> None:
        self._apply_scorer_scene_num_overrides()
        self.training_loss.reset()

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
            "batch": self._expand_batch_index_for_rollouts(
                map_feature["batch"],
                repeat_count=repeat_count,
                num_graphs=num_graphs,
            ),
        }
        if "light_type" in map_feature:
            expanded_map_feature["light_type"] = self._repeat_tensor_on_first_dim(
                map_feature["light_type"],
                repeat_count,
            )
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

    def _rlftsim_policy_terms_for_logits(
        self,
        *,
        logits: torch.Tensor,
        ref_logits: torch.Tensor,
        selected_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temperature = float(self.rlftsim_sampling.temp)
        if temperature <= 0.0:
            raise ValueError(
                f"RLFTSim sampling temperature must be positive, got {temperature}."
            )

        criterium = str(self.rlftsim_sampling.criterium)
        if criterium in {"categorical", "full_prob"}:
            log_probs = F.log_softmax(logits.float() / temperature, dim=-1)
            action_log_prob = log_probs.gather(
                dim=-1,
                index=selected_idx.long().unsqueeze(-1),
            ).squeeze(-1)
            probs = log_probs.exp()
            ref_log_probs = F.log_softmax(
                ref_logits.detach().float() / temperature,
                dim=-1,
            )
        elif criterium == "topk_prob":
            num_k = int(self.rlftsim_sampling.num_k)
            if num_k <= 0:
                raise ValueError(f"topk_prob requires num_k > 0, got {num_k}.")
            num_k = min(num_k, int(logits.shape[-1]))
            topk_logits, topk_idx = torch.topk(logits, num_k, dim=-1, sorted=False)
            log_probs = F.log_softmax(topk_logits.float() / temperature, dim=-1)
            selected_match = topk_idx == selected_idx.long().unsqueeze(-1)
            action_log_prob = torch.where(
                selected_match,
                log_probs,
                torch.zeros_like(log_probs),
            ).sum(dim=-1)
            probs = log_probs.exp()
            ref_topk_logits = ref_logits.detach().gather(dim=-1, index=topk_idx)
            ref_log_probs = F.log_softmax(ref_topk_logits.float() / temperature, dim=-1)
        else:
            raise ValueError(f"Unsupported RLFTSim sampling criterium: {criterium}")

        kl = (probs * (log_probs - ref_log_probs)).sum(dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return action_log_prob, kl, entropy

    def _compute_rlftsim_policy_stats(
        self,
        *,
        pred: Dict[str, torch.Tensor],
        ref_pred: Dict[str, torch.Tensor],
        rollout_tokenized_agent: Dict[str, torch.Tensor],
        repeat_count: int,
        num_graphs: int,
    ) -> dict[str, torch.Tensor]:
        graph_count = int(repeat_count) * int(num_graphs)
        device = rollout_tokenized_agent["batch"].device
        log_prob_sum = torch.zeros(graph_count, dtype=torch.float32, device=device)
        kl_sum = torch.zeros_like(log_prob_sum)
        entropy_sum = torch.zeros_like(log_prob_sum)
        action_count = torch.zeros_like(log_prob_sum)

        selected_idx = pred["pred_idx"][:, 2 : 2 + pred["next_token_valid"].shape[1]]
        valid = pred["next_token_valid"].bool()
        batch_index = rollout_tokenized_agent["batch"].long()

        def accumulate(
            *,
            logits: torch.Tensor,
            ref_logits: torch.Tensor,
            action_idx: torch.Tensor,
            action_valid: torch.Tensor,
            action_batch: torch.Tensor,
        ) -> None:
            action_log_prob, kl, entropy = self._rlftsim_policy_terms_for_logits(
                logits=logits,
                ref_logits=ref_logits,
                selected_idx=action_idx,
            )
            flat_batch = action_batch.unsqueeze(1).expand_as(action_valid).reshape(-1)
            flat_valid = action_valid.reshape(-1).to(dtype=torch.float32)
            log_prob_sum.index_add_(
                0,
                flat_batch,
                action_log_prob.reshape(-1) * flat_valid,
            )
            kl_sum.index_add_(0, flat_batch, kl.reshape(-1) * flat_valid)
            entropy_sum.index_add_(0, flat_batch, entropy.reshape(-1) * flat_valid)
            action_count.index_add_(0, flat_batch, flat_valid)

        if isinstance(pred["next_token_logits"], dict):
            for agent_type, logits in pred["next_token_logits"].items():
                mask = pred["type_mask"][agent_type]
                if not bool(mask.any()):
                    continue
                accumulate(
                    logits=logits,
                    ref_logits=ref_pred["next_token_logits"][agent_type],
                    action_idx=selected_idx[mask],
                    action_valid=valid[mask],
                    action_batch=batch_index[mask],
                )
        else:
            accumulate(
                logits=pred["next_token_logits"],
                ref_logits=ref_pred["next_token_logits"],
                action_idx=selected_idx,
                action_valid=valid,
                action_batch=batch_index,
            )

        log_prob = log_prob_sum.reshape(repeat_count, num_graphs).transpose(0, 1)
        count = action_count.reshape(repeat_count, num_graphs).transpose(0, 1)
        kl = (kl_sum / action_count.clamp_min(1.0)).reshape(
            repeat_count, num_graphs
        ).transpose(0, 1)
        entropy = (entropy_sum / action_count.clamp_min(1.0)).reshape(
            repeat_count, num_graphs
        ).transpose(0, 1)
        return {
            "log_prob": log_prob.contiguous(),
            "kl": kl.contiguous(),
            "entropy": entropy.contiguous(),
            "action_count": count.contiguous(),
        }

    def _build_rlftsim_rollout_chunk_size_candidates(self) -> list[int]:
        chunk_sizes: list[int] = []
        current = max(1, int(self.rlftsim_rollouts_per_scenario))
        while True:
            if current not in chunk_sizes:
                chunk_sizes.append(current)
            if current == 1:
                break
            current = max(1, math.ceil(current / 2))
        return chunk_sizes

    def _build_rlftsim_replay_chunk_size_candidates(
        self,
        repeat_count: int,
    ) -> list[int]:
        chunk_sizes: list[int] = []
        current = min(
            max(1, int(self.rlftsim_replay_rollout_chunk_size)),
            max(1, int(repeat_count)),
        )
        while True:
            if current not in chunk_sizes:
                chunk_sizes.append(current)
            if current == 1:
                break
            current = max(1, math.ceil(current / 2))
        return chunk_sizes

    def _run_rlftsim_sample_rollouts(
        self,
        tokenized_map: Dict[str, torch.Tensor],
        tokenized_agent: Dict[str, torch.Tensor],
        scenario_ids: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        self._init_rlftsim_reference()
        if self._rlftsim_ref_encoder is None:
            raise RuntimeError("RLFTSim reference encoder was not initialized.")

        repeat_count = int(self.rlftsim_rollouts_per_scenario)
        num_agent = int(tokenized_agent["batch"].shape[0])
        num_graphs = int(tokenized_agent["num_graphs"])
        rollout_indices = list(range(repeat_count))

        was_encoder_training = self.encoder.training
        self.encoder.eval()
        try:
            with torch.no_grad():
                map_feature = self.encoder.map_encoder(tokenized_map)

            last_oom_error: RuntimeError | None = None
            for chunk_size in self._build_rlftsim_rollout_chunk_size_candidates():
                pred_traj_chunks: list[torch.Tensor] = []
                pred_z_chunks: list[torch.Tensor] = []
                pred_head_chunks: list[torch.Tensor] = []
                forced_idx_chunks: list[torch.Tensor] = []
                try:
                    for chunk_start in range(0, repeat_count, chunk_size):
                        chunk_rollout_indices = rollout_indices[
                            chunk_start : chunk_start + chunk_size
                        ]
                        chunk_repeat = int(len(chunk_rollout_indices))
                        rollout_map_feature = self._build_parallel_rollout_map_feature(
                            map_feature=map_feature,
                            repeat_count=chunk_repeat,
                            num_graphs=num_graphs,
                        )
                        rollout_tokenized_agent = (
                            self._build_parallel_rollout_tokenized_agent(
                                tokenized_agent=tokenized_agent,
                                repeat_count=chunk_repeat,
                                num_graphs=num_graphs,
                            )
                        )
                        scenario_seed_table = self._build_closed_loop_seed_table(
                            scenario_ids=scenario_ids,
                            rollout_indices=chunk_rollout_indices,
                            device=tokenized_agent["batch"].device,
                        )
                        with torch.no_grad():
                            pred = self.encoder.agent_encoder.inference(
                                rollout_tokenized_agent,
                                rollout_map_feature,
                                self.rlftsim_sampling,
                                scenario_sampling_seeds=scenario_seed_table.reshape(
                                    -1
                                ).contiguous(),
                            )
                        forced_idx = pred["pred_idx"][
                            :, 2 : 2 + pred["next_token_valid"].shape[1]
                        ].detach()
                        pred_traj_chunks.append(
                            self._reshape_parallel_rollout_prediction(
                                pred["pred_traj_10hz"].detach(),
                                repeat_count=chunk_repeat,
                                num_agent=num_agent,
                            )
                        )
                        pred_z_chunks.append(
                            self._reshape_parallel_rollout_prediction(
                                pred["pred_z_10hz"].detach(),
                                repeat_count=chunk_repeat,
                                num_agent=num_agent,
                            )
                        )
                        pred_head_chunks.append(
                            self._reshape_parallel_rollout_prediction(
                                pred["pred_head_10hz"].detach(),
                                repeat_count=chunk_repeat,
                                num_agent=num_agent,
                            )
                        )
                        forced_idx_chunks.append(
                            self._reshape_parallel_rollout_prediction(
                                forced_idx,
                                repeat_count=chunk_repeat,
                                num_agent=num_agent,
                            )
                        )
                    return {
                        "pred_traj": torch.cat(pred_traj_chunks, dim=1),
                        "pred_z": torch.cat(pred_z_chunks, dim=1),
                        "pred_head": torch.cat(pred_head_chunks, dim=1),
                        "forced_next_token_idx": torch.cat(forced_idx_chunks, dim=1),
                    }
                except RuntimeError as error:
                    if (not self._is_cuda_out_of_memory(error)) or chunk_size == 1:
                        raise
                    last_oom_error = error
                    del (
                        pred_traj_chunks,
                        pred_z_chunks,
                        pred_head_chunks,
                        forced_idx_chunks,
                    )
                    self._cleanup_after_rollout_oom()
                    continue
            if last_oom_error is not None:
                raise last_oom_error
            raise RuntimeError("RLFTSim rollout failed before producing predictions.")
        finally:
            if was_encoder_training:
                self.encoder.train()

    def _compute_rlftsim_forced_replay_stats(
        self,
        *,
        tokenized_map: Dict[str, torch.Tensor],
        tokenized_agent: Dict[str, torch.Tensor],
        forced_next_token_idx: torch.Tensor,
        rollout_indices: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        self._init_rlftsim_reference()
        if self._rlftsim_ref_encoder is None:
            raise RuntimeError("RLFTSim reference encoder was not initialized.")

        if forced_next_token_idx.ndim != 3:
            raise ValueError(
                "forced_next_token_idx must have shape [n_agent, n_rollout, n_step], "
                f"got {tuple(forced_next_token_idx.shape)}."
            )

        num_agent = int(tokenized_agent["batch"].shape[0])
        num_graphs = int(tokenized_agent["num_graphs"])
        chunk_repeat = int(len(rollout_indices))
        if chunk_repeat <= 0:
            raise ValueError("rollout_indices must not be empty.")

        was_encoder_training = self.encoder.training
        self.encoder.eval()
        try:
            with self._rlftsim_replay_precision_context(tokenized_agent["batch"].device):
                map_feature = self.encoder.map_encoder(tokenized_map)
                with torch.no_grad():
                    ref_map_feature = self._rlftsim_ref_encoder.map_encoder(tokenized_map)
                rollout_map_feature = self._build_parallel_rollout_map_feature(
                    map_feature=map_feature,
                    repeat_count=chunk_repeat,
                    num_graphs=num_graphs,
                )
                ref_rollout_map_feature = self._build_parallel_rollout_map_feature(
                    map_feature=ref_map_feature,
                    repeat_count=chunk_repeat,
                    num_graphs=num_graphs,
                )
                rollout_tokenized_agent = self._build_parallel_rollout_tokenized_agent(
                    tokenized_agent=tokenized_agent,
                    repeat_count=chunk_repeat,
                    num_graphs=num_graphs,
                )
                forced_chunk = forced_next_token_idx[:, list(rollout_indices)]
                forced_chunk = (
                    forced_chunk.permute(1, 0, 2)
                    .reshape(chunk_repeat * num_agent, forced_chunk.shape[-1])
                    .contiguous()
                )
                pred = self.encoder.agent_encoder.inference(
                    rollout_tokenized_agent,
                    rollout_map_feature,
                    self.rlftsim_sampling,
                    scenario_sampling_seeds=None,
                    forced_next_token_idx=forced_chunk,
                )
                with torch.no_grad():
                    ref_pred = self._rlftsim_ref_encoder.agent_encoder.inference(
                        rollout_tokenized_agent,
                        ref_rollout_map_feature,
                        self.rlftsim_sampling,
                        scenario_sampling_seeds=None,
                        forced_next_token_idx=forced_chunk,
                    )
                return self._compute_rlftsim_policy_stats(
                    pred=pred,
                    ref_pred=ref_pred,
                    rollout_tokenized_agent=rollout_tokenized_agent,
                    repeat_count=chunk_repeat,
                    num_graphs=num_graphs,
                )
        finally:
            if was_encoder_training:
                self.encoder.train()

    def _rlftsim_backward_sync_context(self, *, enabled: bool):
        if enabled:
            return nullcontext()
        trainer = getattr(self, "trainer", None)
        strategy = getattr(trainer, "strategy", None)
        if strategy is not None and hasattr(strategy, "block_backward_sync"):
            return strategy.block_backward_sync()
        return nullcontext()

    def _rlftsim_replay_precision_context(self, device: torch.device):
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", enabled=False)
        return nullcontext()

    def _rlftsim_optimizer_step(self, optimizer) -> None:
        if self.rlftsim_gradient_clip_val > 0.0:
            parameters = [
                parameter for parameter in self.parameters() if parameter.grad is not None
            ]
            if self.rlftsim_gradient_clip_algorithm == "norm":
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    max_norm=self.rlftsim_gradient_clip_val,
                )
            elif self.rlftsim_gradient_clip_algorithm == "value":
                torch.nn.utils.clip_grad_value_(
                    parameters,
                    clip_value=self.rlftsim_gradient_clip_val,
                )
            else:
                raise ValueError(
                    "rlftsim.gradient_clip_algorithm must be 'norm' or 'value', "
                    f"got {self.rlftsim_gradient_clip_algorithm!r}."
                )
        optimizer.step()
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    def _rlftsim_forced_replay_backward(
        self,
        *,
        tokenized_map: Dict[str, torch.Tensor],
        tokenized_agent: Dict[str, torch.Tensor],
        forced_next_token_idx: torch.Tensor,
        rewards: torch.Tensor,
        replay_chunk_size: int,
        should_step_optimizer: bool,
        accumulate_grad_batches: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        repeat_count = int(rewards.shape[1])
        loss_value = rewards.new_tensor(0.0)
        policy_loss_value = rewards.new_tensor(0.0)
        kl_value = rewards.new_tensor(0.0)
        entropy_value = rewards.new_tensor(0.0)
        for chunk_start in range(0, repeat_count, replay_chunk_size):
            chunk_end = min(repeat_count, chunk_start + replay_chunk_size)
            rollout_indices = list(range(chunk_start, chunk_end))
            chunk_weight = float(chunk_end - chunk_start) / float(repeat_count)
            sync_grad = should_step_optimizer and chunk_end >= repeat_count
            with self._rlftsim_backward_sync_context(enabled=sync_grad):
                stats = self._compute_rlftsim_forced_replay_stats(
                    tokenized_map=tokenized_map,
                    tokenized_agent=tokenized_agent,
                    forced_next_token_idx=forced_next_token_idx,
                    rollout_indices=rollout_indices,
                )
                chunk_rewards = rewards[:, chunk_start:chunk_end]
                chunk_policy_loss = -(stats["log_prob"] * chunk_rewards).mean()
                chunk_kl = stats["kl"].mean()
                chunk_entropy = stats["entropy"].mean()
                chunk_loss = chunk_policy_loss + float(self.rlftsim_kl_beta) * chunk_kl
                if self.rlftsim_entropy_bonus != 0.0:
                    chunk_loss = (
                        chunk_loss - float(self.rlftsim_entropy_bonus) * chunk_entropy
                    )
                self.manual_backward(
                    chunk_loss
                    * chunk_weight
                    / float(max(1, accumulate_grad_batches))
                )
            loss_value = loss_value + chunk_loss.detach() * chunk_weight
            policy_loss_value = (
                policy_loss_value + chunk_policy_loss.detach() * chunk_weight
            )
            kl_value = kl_value + chunk_kl.detach() * chunk_weight
            entropy_value = entropy_value + chunk_entropy.detach() * chunk_weight
            del stats, chunk_loss, chunk_policy_loss, chunk_kl, chunk_entropy
        return loss_value, policy_loss_value, kl_value, entropy_value

    def _rlftsim_training_step(self, data, batch_idx):
        self._validate_rlftsim_batch(data)
        optimizer = self.optimizers()
        accumulate_grad_batches = int(self.rlftsim_accumulate_grad_batches)
        is_accumulation_start = int(batch_idx) % max(1, accumulate_grad_batches) == 0
        should_step_optimizer = self._is_rlftsim_optimizer_step_boundary(batch_idx)
        if is_accumulation_start:
            optimizer.zero_grad()

        tokenized_map, tokenized_agent = self.token_processor(data)
        self._ensure_tokenized_agent_z_raw(tokenized_agent, data)
        with torch.no_grad():
            rollout = self._run_rlftsim_sample_rollouts(
                tokenized_map=tokenized_map,
                tokenized_agent=tokenized_agent,
                scenario_ids=data["scenario_id"],
            )
        if self.rlftsim_reward is None:
            raise RuntimeError("RLFTSim reward calculator is not initialized.")
        scenario_files = data["tfrecord_path"]
        if isinstance(scenario_files, str):
            scenario_files = [scenario_files]
        reward_batch = self.rlftsim_reward.compute_from_prediction_tensors(
            scenario_files=scenario_files,
            agent_id=data["agent"]["id"],
            agent_batch=data["agent"]["batch"],
            pred_traj=rollout["pred_traj"],
            pred_z=rollout["pred_z"],
            pred_head=rollout["pred_head"],
        )
        rewards = reward_batch.rewards.detach() * float(self.rlftsim_reward_scale)
        if self.rlftsim_normalize_rewards:
            reward_std = rewards.std(dim=1, keepdim=True).clamp_min(1.0e-6)
            rewards = rewards / reward_std

        repeat_count = int(rewards.shape[1])
        replay_chunk_size = max(1, int(self.rlftsim_replay_rollout_chunk_size))
        last_oom_error: RuntimeError | None = None
        for candidate_chunk_size in self._build_rlftsim_replay_chunk_size_candidates(
            repeat_count
        ):
            try:
                (
                    loss_value,
                    policy_loss_value,
                    kl_value,
                    entropy_value,
                ) = self._rlftsim_forced_replay_backward(
                    tokenized_map=tokenized_map,
                    tokenized_agent=tokenized_agent,
                    forced_next_token_idx=rollout["forced_next_token_idx"],
                    rewards=rewards,
                    replay_chunk_size=candidate_chunk_size,
                    should_step_optimizer=should_step_optimizer,
                    accumulate_grad_batches=accumulate_grad_batches,
                )
                replay_chunk_size = candidate_chunk_size
                self.rlftsim_replay_rollout_chunk_size = candidate_chunk_size
                break
            except RuntimeError as error:
                if (
                    (not self._is_cuda_out_of_memory(error))
                    or candidate_chunk_size == 1
                ):
                    raise
                if not is_accumulation_start:
                    raise RuntimeError(
                        "RLFTSim replay chunk fallback cannot safely retry in the "
                        "middle of gradient accumulation; use accumulate_grad_batches=1."
                    ) from error
                last_oom_error = error
                next_chunk_size = max(1, math.ceil(candidate_chunk_size / 2))
                self.rlftsim_replay_rollout_chunk_size = next_chunk_size
                optimizer.zero_grad()
                self._cleanup_after_rollout_oom()
                continue
        else:
            if last_oom_error is not None:
                raise last_oom_error
            raise RuntimeError("RLFTSim forced replay failed before producing a loss.")

        if should_step_optimizer:
            self._rlftsim_optimizer_step(optimizer)

        self._record_rlftsim_kl_for_controller(kl_value)
        self.log("train/rlftsim_loss", loss_value, on_step=True, batch_size=1)
        self.log(
            "train/rlftsim_policy_loss",
            policy_loss_value,
            on_step=True,
            batch_size=1,
        )
        self.log("train/rlftsim_kl", kl_value, on_step=True, batch_size=1)
        self.log(
            "train/rlftsim_kl_beta",
            torch.tensor(self.rlftsim_kl_beta, device=self.device),
            on_step=True,
            batch_size=1,
        )
        self.log(
            "train/rlftsim_entropy",
            entropy_value,
            on_step=True,
            batch_size=1,
        )
        self.log(
            "train/rlftsim_replay_chunk_size",
            torch.tensor(float(replay_chunk_size), device=self.device),
            on_step=True,
            batch_size=1,
        )
        self.log(
            "train/rlftsim_reward_std",
            rewards.detach().std(),
            on_step=True,
            batch_size=1,
        )
        self.log(
            "train/rlftsim_reward_abs_mean",
            rewards.detach().abs().mean(),
            on_step=True,
            batch_size=1,
        )
        self.log(
            "train/rlftsim_full_rmm",
            reward_batch.full_rmm.mean(),
            on_step=True,
            batch_size=1,
        )
        self.log(
            "train/rlftsim_leave_one_out_rmm",
            reward_batch.leave_one_out_rmm.mean(),
            on_step=True,
            batch_size=1,
        )
        return loss_value

    def training_step(self, data, batch_idx):
        if self.rlftsim_enabled:
            return self._rlftsim_training_step(data, batch_idx)

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
                if self._should_compute_closed_loop_minade():
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
                video_logger = self._get_video_logger()
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
                            if video_logger is not None:
                                video_logger.log_video(
                                    "/".join(_path.split("/")[-3:]), [_path]
                                )
                                if self.delete_local_videos_after_wandb_upload:
                                    self._cleanup_local_video(_path)

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
                    minade_value = None
                    if self._should_compute_closed_loop_minade():
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
                    minade_value = None
                    if self._should_compute_closed_loop_minade():
                        minade_value = (
                            self.minADE.sum / self.minADE.count.clamp_min(1e-6)
                        )

                closed_loop_metric = epoch_sim_agents_metrics[
                    self.closed_loop_metric_name
                ]
                if minade_value is not None:
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
                    self._log_metrics_to_logger(epoch_sim_agents_metrics)

                self.sim_agents_metrics.reset()
                self.minADE.reset()

            if self.sim_agents_submission.is_active:
                if self.global_rank == 0 and epoch_distribution_metrics:
                    epoch_distribution_metrics["epoch"] = (
                        self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                    )
                    self._log_metrics_to_logger(epoch_distribution_metrics)
                self.sim_agents_submission.save_sub_file()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.rlftsim_enabled:
            total_steps = int(self.lr_total_steps)
            trainer = getattr(self, "trainer", None)
            estimated_steps = int(getattr(trainer, "estimated_stepping_batches", 0) or 0)
            if total_steps <= 0 and estimated_steps > 0:
                total_steps = estimated_steps
            total_steps = max(total_steps, int(self.lr_warmup_steps) + 1, 1)

            def rlftsim_lr_lambda(current_step):
                current_step = int(current_step)
                warmup_steps = int(self.lr_warmup_steps)
                if warmup_steps > 0 and current_step < warmup_steps:
                    return self.lr_min_ratio + (
                        1.0 - self.lr_min_ratio
                    ) * current_step / float(warmup_steps)
                decay_denominator = max(1, total_steps - warmup_steps)
                progress = min(
                    1.0,
                    max(0.0, (current_step - warmup_steps) / float(decay_denominator)),
                )
                return self.lr_min_ratio + 0.5 * (1.0 - self.lr_min_ratio) * (
                    1.0 + math.cos(math.pi * progress)
                )

            lr_scheduler = LambdaLR(optimizer, lr_lambda=rlftsim_lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

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
        if self.sim_agents_submission.is_active:
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
                self._log_metrics_to_logger(epoch_distribution_metrics)
        if self.sim_agents_submission.is_active:
            self.sim_agents_submission.save_sub_file()
