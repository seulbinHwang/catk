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

import os
import shutil
import socket
import struct
import tarfile
import time
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

# Waymo 리더보드 거절을 피하기 위해 placeholder가 그대로 들어간 submission archive를
# 생성하기 전에 미리 차단한다. configs/model/smart.yaml 등의 default placeholder와
# 정확히 일치하는 문자열만 거른다.
_SUBMISSION_FIELD_PLACEHOLDERS: Dict[str, frozenset] = {
    "affiliation": frozenset({"YOUR_AFFILIATION"}),
    "description": frozenset({"YOUR_DESCRIPTION"}),
    "method_link": frozenset({"YOUR_METHOD_LINK"}),
    "account_name": frozenset({"YOUR_ACCOUNT_NAME"}),
}
_SUBMISSION_AUTHORS_PLACEHOLDERS: frozenset = frozenset({"Anonymous"})


def _is_authors_placeholder(authors: ListConfig[str]) -> bool:
    if authors is None:
        return True
    author_list = [str(name).strip() for name in authors if str(name).strip()]
    if not author_list:
        return True
    return all(name in _SUBMISSION_AUTHORS_PLACEHOLDERS for name in author_list)


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
        num_model_parameters: str = "7M",
    ) -> None:
        # Evaluation data is already sharded exactly once per rank by ExactDistributedSampler.
        # Submission export must therefore stay rank-local and only pack shards together at epoch end.
        super().__init__(sync_on_compute=False)
        self.is_active = is_active
        if self.is_active:
            self._raise_if_metadata_is_placeholder(
                authors=authors,
                affiliation=affiliation,
                description=description,
                method_link=method_link,
                account_name=account_name,
            )
            self.method_name = method_name
            self.authors = authors
            self.affiliation = affiliation
            self.description = description
            self.method_link = method_link
            self.account_name = account_name
            self.num_model_parameters = str(num_model_parameters)
            self.buffer_scenario_rollouts = []
            self.i_file = 0
            self.submission_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            self.submission_dir = Path(self.submission_dir) / _SIM_AGENTS_2025_SUBMISSION_DIRNAME
            self.submission_dir.mkdir(parents=True, exist_ok=True)
            # scenario_id 중복 검사는 list보다 set이 빠르다. 전체 WOMD test split처럼
            # rank당 수천 scenario가 누적되면 list 기반 ``in`` 검사가 O(n^2)이 되므로
            # set으로 dedup해서 aggregate_rollouts가 O(n)에 끝나도록 만든다.
            self.submission_scenario_id = set()

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
        device = pred_traj.device
        self.agent_id.append(agent_id)
        self.scenario_id.append(get_scenario_id_int_tensor(scenario_id, device))
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
        self,
        scenario_rollouts: List[sim_agents_submission_pb2.ScenarioRollouts],
    ) -> None:
        for rollout in scenario_rollouts:
            if rollout.scenario_id not in self.submission_scenario_id:
                self.submission_scenario_id.add(rollout.scenario_id)
                self.buffer_scenario_rollouts.append(rollout)
                if len(self.buffer_scenario_rollouts) > 300:
                    self._save_shard()

    def save_sub_file(self) -> None:
        """모든 rank가 만든 shard를 모아 Waymo 제출용 tar.gz를 만듭니다."""
        self._save_shard()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        archive_source_dir = self._collect_multinode_shards_if_requested()
        tar_file_name = self.submission_dir.as_posix() + ".tar.gz"
        if self._get_global_rank() != 0:
            return

        shard_files = sorted(archive_source_dir.glob("*.binproto"))
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

    def _collect_multinode_shards_if_requested(self) -> Path:
        """Collect per-node submission shards onto global rank 0 before tar export.

        The default path keeps the original shared-filesystem behavior. Some static
        pod pairs use the same path string on each pod but do not share the actual
        filesystem. In that case, set ``CATK_SUBMISSION_STREAM_SHARDS=1`` so
        non-master nodes stream their local shard files to rank 0 over TCP.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return self.submission_dir
        if os.environ.get("CATK_SUBMISSION_STREAM_SHARDS", "0") not in {"1", "true", "TRUE"}:
            return self.submission_dir

        rank = self._get_global_rank()
        world_size = int(dist.get_world_size())
        node_rank = int(os.environ.get("NODE_RANK", "0"))
        gathered_node_ranks: list[int | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_node_ranks, node_rank)
        remote_ranks = [
            i for i, gathered_node_rank in enumerate(gathered_node_ranks)
            if gathered_node_rank not in (None, 0)
        ]

        collect_dir = self.submission_dir.parent / f"{self.submission_dir.name}_rank0_collect"
        if rank == 0:
            if collect_dir.exists():
                shutil.rmtree(collect_dir)
            collect_dir.mkdir(parents=True, exist_ok=True)
            for shard_path in sorted(self.submission_dir.glob("*.binproto")):
                shutil.copy2(shard_path, collect_dir / shard_path.name)

        dist.barrier()
        if rank == 0:
            self._receive_remote_shards(
                collect_dir=collect_dir,
                expected_connections=len(remote_ranks),
            )
        elif rank in remote_ranks:
            self._send_local_shards_to_rank0()

        dist.barrier()
        if rank == 0:
            shard_count = len(list(collect_dir.glob("*.binproto")))
            log.info(
                "Collected Sim Agents 2025 submission shards from all nodes "
                f"into {collect_dir} ({shard_count} files)."
            )
            return collect_dir
        return self.submission_dir

    def _receive_remote_shards(self, collect_dir: Path, expected_connections: int) -> None:
        if expected_connections <= 0:
            return

        port = int(os.environ.get("CATK_SUBMISSION_SHARD_STREAM_PORT", "29631"))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("0.0.0.0", port))
            server.listen(expected_connections)
            for _ in range(expected_connections):
                conn, _ = server.accept()
                with conn:
                    self._receive_one_rank_shards(conn, collect_dir)

    def _send_local_shards_to_rank0(self) -> None:
        host = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = int(os.environ.get("CATK_SUBMISSION_SHARD_STREAM_PORT", "29631"))
        shard_files = sorted(self.submission_dir.glob("*.binproto"))
        deadline = time.time() + 120.0
        last_error: OSError | None = None
        while time.time() < deadline:
            try:
                with socket.create_connection((host, port), timeout=10.0) as conn:
                    self._send_one_rank_shards(conn, shard_files)
                return
            except OSError as exc:
                last_error = exc
                time.sleep(1.0)
        raise RuntimeError(
            f"Failed to connect to rank-0 shard collection server at {host}:{port}."
        ) from last_error

    @classmethod
    def _receive_one_rank_shards(cls, conn: socket.socket, collect_dir: Path) -> None:
        num_files = cls._recv_uint64(conn)
        for _ in range(num_files):
            name_len = cls._recv_uint64(conn)
            name = cls._recv_exact(conn, name_len).decode("utf-8")
            safe_name = Path(name).name
            file_size = cls._recv_uint64(conn)
            output_path = collect_dir / safe_name
            with output_path.open("wb") as output_file:
                remaining = file_size
                while remaining > 0:
                    chunk = conn.recv(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise RuntimeError(f"Connection closed while receiving {safe_name}.")
                    output_file.write(chunk)
                    remaining -= len(chunk)

    @classmethod
    def _send_one_rank_shards(cls, conn: socket.socket, shard_files: List[Path]) -> None:
        cls._send_uint64(conn, len(shard_files))
        for shard_path in shard_files:
            name_bytes = shard_path.name.encode("utf-8")
            cls._send_uint64(conn, len(name_bytes))
            conn.sendall(name_bytes)
            file_size = shard_path.stat().st_size
            cls._send_uint64(conn, file_size)
            with shard_path.open("rb") as shard_file:
                while True:
                    chunk = shard_file.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    conn.sendall(chunk)

    @staticmethod
    def _send_uint64(conn: socket.socket, value: int) -> None:
        conn.sendall(struct.pack("!Q", int(value)))

    @classmethod
    def _recv_uint64(cls, conn: socket.socket) -> int:
        return struct.unpack("!Q", cls._recv_exact(conn, 8))[0]

    @staticmethod
    def _recv_exact(conn: socket.socket, n_bytes: int) -> bytes:
        chunks: list[bytes] = []
        remaining = n_bytes
        while remaining > 0:
            chunk = conn.recv(remaining)
            if not chunk:
                raise RuntimeError("Connection closed while receiving submission shard data.")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

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
            num_model_parameters=self.num_model_parameters,
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
    def _raise_if_metadata_is_placeholder(
        authors: ListConfig[str],
        affiliation: str,
        description: str,
        method_link: str,
        account_name: str,
    ) -> None:
        """``is_active=True``인 submission 모드에서 placeholder가 들어가지 않았는지 검사한다.

        ``configs/model/smart.yaml``의 default 값이 그대로 남아 있으면 Waymo
        리더보드가 archive를 거절한다. 그래서 모델 초기화 단계에서 미리 차단해
        몇 시간짜리 rollout 끝에서야 잘못된 tar.gz가 만들어지는 일을 막는다.
        """
        offending_fields: List[str] = []
        for field_name, value in (
            ("affiliation", affiliation),
            ("description", description),
            ("method_link", method_link),
            ("account_name", account_name),
        ):
            placeholders = _SUBMISSION_FIELD_PLACEHOLDERS.get(field_name, frozenset())
            if str(value).strip() in placeholders:
                offending_fields.append(field_name)
        if _is_authors_placeholder(authors):
            offending_fields.append("authors")
        if offending_fields:
            field_list = ", ".join(sorted(offending_fields))
            raise ValueError(
                "Sim Agents 2025 submission 메타데이터가 default placeholder인 채로 "
                f"is_active=True로 켜졌습니다: {field_list}. "
                "Waymo 리더보드가 placeholder가 들어간 archive를 거절하므로 실행 인자로 "
                "model.model_config.sim_agents_submission.* 필드를 실제 값으로 "
                "override해 주세요. 예: "
                "model.model_config.sim_agents_submission.account_name=\"<your_account>\"."
            )

    @staticmethod
    def _build_archive_member_name(shard_index: int, num_shards: int) -> str:
        return f"submission.binproto-{shard_index:05d}-of-{num_shards:05d}"

    @staticmethod
    def _get_global_rank() -> int:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
        return 0
