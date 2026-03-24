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

import tarfile
from pathlib import Path
from typing import Dict, List

import hydra
import torch.distributed as dist
from omegaconf import ListConfig
from torch import Tensor
from torchmetrics.metric import Metric
from waymo_open_dataset.protos import sim_agents_submission_pb2

from src.utils import RankedLogger
from src.utils.sim_agents_utils import get_scenario_id_int_tensor, get_scenario_rollouts

log = RankedLogger(__name__, rank_zero_only=False)
_SIM_AGENTS_2025_SUBMISSION_DIRNAME = "sim_agents_2025_submission"


class SimAgentsSubmission(Metric):
    """Waymo 2025 Sim Agents 제출 파일을 shard와 tar.gz로 저장합니다."""

    def __init__(
        self,
        is_active: bool,
        method_name: str,
        authors: ListConfig[str],
        affiliation: str,
        description: str,
        method_link: str,
        account_name: str,
    ) -> None:
        # Evaluation data is already sharded exactly once per rank by ExactDistributedSampler.
        # Submission export must therefore stay rank-local and only pack shards together at epoch end.
        super().__init__(sync_on_compute=False)
        self.is_active = is_active
        if self.is_active:
            self.method_name = method_name
            self.authors = authors
            self.affiliation = affiliation
            self.description = description
            self.method_link = method_link
            self.account_name = account_name
            self.buffer_scenario_rollouts = []
            self.i_file = 0
            self.submission_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            self.submission_dir = Path(self.submission_dir) / _SIM_AGENTS_2025_SUBMISSION_DIRNAME
            self.submission_dir.mkdir(parents=True, exist_ok=True)
            self.submission_scenario_id = []

            self.data_keys = [
                "scenario_id",
                "agent_id",
                "agent_batch",
                "pred_traj",
                "pred_z",
                "pred_head",
            ]
            for k in self.data_keys:
                self.add_state(k, default=[], dist_reduce_fx="cat")

    def update(
        self,
        scenario_id: List[str],
        agent_id: List[List[float]],
        agent_batch: Tensor,
        pred_traj: Tensor,
        pred_z: Tensor,
        pred_head: Tensor,
    ) -> None:
        _device = pred_traj.device
        self.agent_id.append(agent_id)
        self.scenario_id.append(get_scenario_id_int_tensor(scenario_id, _device))
        self.pred_traj.append(pred_traj)
        self.pred_z.append(pred_z)
        self.pred_head.append(pred_head)
        self.agent_batch.append(agent_batch)

    def compute(self) -> Dict[str, Tensor]:
        return {k: getattr(self, k) for k in self.data_keys}

    def aggregate_current_batch(self) -> List[sim_agents_submission_pb2.ScenarioRollouts]:
        local_batch = self.compute()
        for key, value in local_batch.items():
            if isinstance(value, list):
                if len(value) != 1:
                    raise RuntimeError(
                        f"Expected a single local submission state for {key}, got {len(value)}."
                    )
                local_batch[key] = value[0]
        scenario_rollouts = get_scenario_rollouts(**local_batch)
        self.aggregate_rollouts(scenario_rollouts)
        self.reset()
        return scenario_rollouts

    def aggregate_rollouts(
        self, scenario_rollouts: List[sim_agents_submission_pb2.ScenarioRollouts]
    ) -> None:
        for rollout in scenario_rollouts:
            if rollout.scenario_id not in self.submission_scenario_id:
                self.submission_scenario_id.append(rollout.scenario_id)
                self.buffer_scenario_rollouts.append(rollout)
                if len(self.buffer_scenario_rollouts) > 300:
                    self._save_shard()

    def save_sub_file(self) -> None:
        """모든 rank가 만든 shard를 모아 Waymo 제출용 tar.gz를 만듭니다.

        Returns:
            None: 파일 저장만 하고 값을 돌려주지 않습니다.
        """
        self._save_shard()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        tar_file_name = self.submission_dir.as_posix() + ".tar.gz"
        if self._get_global_rank() != 0:
            return

        shard_files = sorted(self.submission_dir.glob("*.binproto"))
        if len(shard_files) == 0:
            log.info("No Sim Agents 2025 submission shards were produced. Skip tar.gz export.")
            return

        num_shards = len(shard_files)
        log.info(f"Saving Sim Agents 2025 submission files to {tar_file_name}")
        with tarfile.open(tar_file_name, "w:gz") as tar:
            for shard_index, shard_path in enumerate(shard_files):
                tar.add(
                    shard_path.as_posix(),
                    arcname=self._build_archive_member_name(
                        shard_index=shard_index,
                        num_shards=num_shards,
                    ),
                )
        log.info(f"DONE: Saved Sim Agents 2025 submission files to {tar_file_name}")
        self.i_file = 0

    def _save_shard(self) -> None:
        if len(self.buffer_scenario_rollouts) == 0:
            return

        shard_submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission(
            scenario_rollouts=self.buffer_scenario_rollouts,
            submission_type=sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
            account_name=self.account_name,
            unique_method_name=self.method_name,
            authors=self.authors,
            affiliation=self.affiliation,
            description=self.description,
            method_link=self.method_link,
            uses_lidar_data=False,
            uses_camera_data=False,
            uses_public_model_pretraining=False,
            num_model_parameters="7M",
            acknowledge_complies_with_closed_loop_requirement=True,
        )
        output_filename = self.submission_dir / (
            f"submission-rank{self._get_global_rank():02d}-{self.i_file:05d}.binproto"
        )
        log.info(f"Saving Sim Agents 2025 submission shard to {output_filename}")
        with open(output_filename, "wb") as f:
            f.write(shard_submission.SerializeToString())
        self.i_file += 1
        self.buffer_scenario_rollouts = []

    @staticmethod
    def _build_archive_member_name(shard_index: int, num_shards: int) -> str:
        """tar 안에 들어갈 binproto 이름을 Waymo 규격에 맞게 만듭니다.

        Args:
            shard_index: 현재 shard 순번입니다.
            num_shards: 전체 shard 개수입니다.

        Returns:
            str: ``submission.binproto-00000-of-00006`` 형태의 이름입니다.
        """
        return f"submission.binproto-{shard_index:05d}-of-{num_shards:05d}"

    @staticmethod
    def _get_global_rank() -> int:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
        return 0