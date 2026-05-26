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
import subprocess
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
                self.submission_scenario_id.add(rollout.scenario_id)
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
        compresslevel = int(os.environ.get("CATK_SUBMISSION_TAR_GZ_COMPRESSLEVEL", "1"))
        self._save_submission_archive(
            shard_files=shard_files,
            tar_file_name=tar_file_name,
            compresslevel=compresslevel,
        )
        log.info(f"DONE: Saved Sim Agents 2025 submission files to {tar_file_name}")
        self.i_file = 0

    def _save_submission_archive(
        self,
        shard_files: List[Path],
        tar_file_name: str,
        compresslevel: int,
    ) -> None:
        pigz_path = shutil.which("pigz")
        if pigz_path is None:
            with tarfile.open(tar_file_name, "w:gz", compresslevel=compresslevel) as tar:
                num_shards = len(shard_files)
                for shard_index, shard_path in enumerate(shard_files):
                    tar.add(
                        shard_path.as_posix(),
                        arcname=self._build_archive_member_name(
                            shard_index=shard_index,
                            num_shards=num_shards,
                        ),
                    )
            return

        archive_path = Path(tar_file_name)
        link_dir = archive_path.parent / f".{archive_path.name}.links.tmp"
        if link_dir.exists():
            shutil.rmtree(link_dir)
        link_dir.mkdir(parents=True, exist_ok=True)
        try:
            num_shards = len(shard_files)
            member_names: list[str] = []
            for shard_index, shard_path in enumerate(shard_files):
                member_name = self._build_archive_member_name(
                    shard_index=shard_index,
                    num_shards=num_shards,
                )
                os.link(shard_path, link_dir / member_name)
                member_names.append(member_name)

            with archive_path.open("wb") as output_file:
                tar_proc = subprocess.Popen(
                    ["tar", "-C", link_dir.as_posix(), "-cf", "-", *member_names],
                    stdout=subprocess.PIPE,
                )
                assert tar_proc.stdout is not None
                pigz_proc = subprocess.Popen(
                    [pigz_path, f"-{compresslevel}"],
                    stdin=tar_proc.stdout,
                    stdout=output_file,
                )
                tar_proc.stdout.close()
                pigz_rc = pigz_proc.wait()
                tar_rc = tar_proc.wait()
            if tar_rc != 0 or pigz_rc != 0:
                raise RuntimeError(
                    f"Failed to create Sim Agents 2025 submission archive with pigz "
                    f"(tar_rc={tar_rc}, pigz_rc={pigz_rc})."
                )
        finally:
            shutil.rmtree(link_dir, ignore_errors=True)

    def _collect_multinode_shards_if_requested(self) -> Path:
        """멀티 pod의 rank-local 제출 shard를 rank 0 파일시스템으로 모읍니다."""
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
        max_attempts = max(
            expected_connections * 3,
            int(os.environ.get("CATK_SUBMISSION_SHARD_STREAM_MAX_ATTEMPTS", "16")),
        )
        successful_connections = 0
        attempts = 0

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("0.0.0.0", port))
            server.listen(expected_connections)
            while successful_connections < expected_connections and attempts < max_attempts:
                attempts += 1
                conn, _ = server.accept()
                with conn:
                    try:
                        self._receive_one_rank_shards(conn, collect_dir)
                    except Exception as exc:
                        log.warning(
                            "Failed while receiving Sim Agents 2025 submission shards "
                            f"on attempt {attempts}/{max_attempts}; waiting for sender retry. "
                            f"Error: {exc}"
                        )
                        continue
                    successful_connections += 1

        if successful_connections != expected_connections:
            raise RuntimeError(
                "Failed to collect all remote Sim Agents 2025 submission shards: "
                f"{successful_connections}/{expected_connections} connections succeeded "
                f"after {attempts} attempts."
            )

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
            tmp_output_path = output_path.with_suffix(output_path.suffix + ".part")
            try:
                with tmp_output_path.open("wb") as output_file:
                    remaining = file_size
                    while remaining > 0:
                        chunk = conn.recv(min(8 * 1024 * 1024, remaining))
                        if not chunk:
                            raise RuntimeError(f"Connection closed while receiving {safe_name}.")
                        output_file.write(chunk)
                        remaining -= len(chunk)
                tmp_size = tmp_output_path.stat().st_size
                if tmp_size != file_size:
                    raise RuntimeError(
                        f"Received size mismatch for {safe_name}: expected {file_size}, got {tmp_size}."
                    )
                tmp_output_path.replace(output_path)
            except Exception:
                tmp_output_path.unlink(missing_ok=True)
                raise

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
