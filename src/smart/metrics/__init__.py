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

from src.smart.metrics.cross_entropy import CrossEntropy
from src.smart.metrics.ego_nll import EgoNLL
from src.smart.metrics.gmm_ade import GMMADE
from src.smart.metrics.min_ade import minADE
from src.smart.metrics.next_token_cls import TokenCls
from src.smart.metrics.wosac_metrics import WOSACMetrics
from src.smart.metrics.wosac_submission import WOSACSubmission

import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch_geometric.utils import degree as _tg_degree


def _sim_agents_worker(
    config_bytes: bytes,
    scenario_file: str,
    agent_ids: np.ndarray,
    pred_traj_np: np.ndarray,
    pred_z_np: np.ndarray,
    pred_head_np: np.ndarray,
) -> float:
    """Subprocess worker: 모든 TF/proto 연산을 격리된 프로세스에서 실행합니다."""
    import tensorflow as tf
    import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as wm
    from waymo_open_dataset.protos import (
        scenario_pb2,
        sim_agents_metrics_pb2,
        sim_agents_submission_pb2,
    )

    tf.config.set_visible_devices([], "GPU")

    config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
    config.ParseFromString(config_bytes)

    scenario = scenario_pb2.Scenario()
    for tfdata in tf.data.TFRecordDataset([scenario_file], compression_type=""):
        scenario.ParseFromString(bytes(tfdata.numpy()))
        break

    n_agents, n_rollout = pred_traj_np.shape[:2]
    joint_scenes = []
    for i_rollout in range(n_rollout):
        simulated_trajectories = []
        for i_agent in range(n_agents):
            simulated_trajectories.append(
                sim_agents_submission_pb2.SimulatedTrajectory(
                    center_x=pred_traj_np[i_agent, i_rollout, :, 0],
                    center_y=pred_traj_np[i_agent, i_rollout, :, 1],
                    center_z=pred_z_np[i_agent, i_rollout],
                    heading=pred_head_np[i_agent, i_rollout],
                    object_id=int(agent_ids[i_agent]),
                )
            )
        joint_scenes.append(
            sim_agents_submission_pb2.JointScene(simulated_trajectories=simulated_trajectories)
        )

    scenario_rollout = sim_agents_submission_pb2.ScenarioRollouts(
        joint_scenes=joint_scenes,
        scenario_id=scenario.scenario_id,
    )
    result = wm.compute_scenario_metrics_for_bundle(config, scenario, scenario_rollout)
    return float(result.metametric)


class SimAgentsMetrics:
    """Waymo Sim Agents 2025 Challenge 기준 realism_meta_metric 계산기.

    TF 연산을 subprocess(forkserver)에서 실행해 PyTorch DDP CUDA context와
    충돌하지 않도록 격리합니다.
    """

    def __init__(self, prefix: str, max_workers: int = 0) -> None:
        self.prefix = prefix
        self._metric_key = f"{prefix}/sim_agents_2025/realism_meta_metric"
        self._config_bytes = self._load_config_bytes()
        self._metametric_sum: float = 0.0
        self._count: int = 0
        self._is_mp_init: bool = False

    @staticmethod
    def _load_config_bytes() -> bytes:
        from google.protobuf import text_format
        from waymo_open_dataset.protos import sim_agents_metrics_pb2
        import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as wm

        config_path = Path(wm.__file__).parent / "challenge_2025_sim_agents_config.textproto"
        with open(config_path, "r") as f:
            cfg = sim_agents_metrics_pb2.SimAgentMetricsConfig()
            text_format.Parse(f.read(), cfg)
        return cfg.SerializeToString()

    def _ensure_mp_init(self) -> None:
        if not self._is_mp_init:
            self._is_mp_init = True
            try:
                mp.set_start_method("forkserver", force=True)
            except RuntimeError:
                pass  # 이미 설정됨

    def update_from_prediction_tensors(
        self,
        *,
        scenario_files: List[str],
        agent_id: torch.Tensor,
        agent_batch: torch.Tensor,
        pred_traj: torch.Tensor,
        pred_z: torch.Tensor,
        pred_head: torch.Tensor,
    ) -> None:
        sizes = [int(s) for s in _tg_degree(agent_batch, dtype=torch.long).tolist()]
        agent_id_list = [t.cpu().numpy() for t in agent_id.cpu().split(sizes)]
        pred_traj_list = [t.cpu().numpy() for t in pred_traj.cpu().split(sizes)]
        pred_z_list = [t.cpu().numpy() for t in pred_z.cpu().split(sizes)]
        pred_head_list = [t.cpu().numpy() for t in pred_head.cpu().split(sizes)]

        n_scenarios = len(scenario_files)
        args = [
            (
                self._config_bytes,
                scenario_files[i],
                agent_id_list[i],
                pred_traj_list[i],
                pred_z_list[i],
                pred_head_list[i],
            )
            for i in range(n_scenarios)
        ]

        self._ensure_mp_init()
        with mp.Pool(processes=max(1, n_scenarios)) as pool:
            results = pool.starmap(_sim_agents_worker, args)
            pool.close()
            pool.join()

        for metametric in results:
            self._metametric_sum += metametric
            self._count += 1

    def _drain_completed_futures(self, wait: bool = True, drain_all: bool = True) -> None:
        return None

    def get_state_tensor(self, device: torch.device) -> torch.Tensor:
        """DDP all_reduce 용으로 [metametric_sum, count] 2-element tensor 반환."""
        return torch.tensor(
            [self._metametric_sum, float(self._count)],
            device=device,
            dtype=torch.float64,
        )

    def compute_from_state_tensor(self, reduced_metric_state: torch.Tensor) -> Dict[str, Any]:
        """all_reduce 이후 [total_sum, total_count]로부터 평균 메트릭 계산."""
        total_count = reduced_metric_state[1].clamp_min(1.0)
        value = (reduced_metric_state[0] / total_count).to(torch.float32)
        return {self._metric_key: value}

    def compute(self) -> Dict[str, Any]:
        if self._count == 0:
            return {self._metric_key: torch.tensor(0.0)}
        return {
            self._metric_key: torch.tensor(
                self._metametric_sum / self._count, dtype=torch.float32
            )
        }

    def reset(self) -> None:
        self._metametric_sum = 0.0
        self._count = 0


class HardSimAgentsMetrics:
    """Pure-PyTorch hard RMM 계산기 — subprocess·TF metric computation 없음.

    SimAgentsMetrics 와 동일한 인터페이스를 제공하므로 smart_flow.py 에서
    드롭인 교체가 가능합니다.

    속도 개선 포인트 (SimAgentsMetrics 대비):
    - subprocess(forkserver) 생성 오버헤드 없음 — 배치당 proc spawn 비용 제거.
    - TF metric/feature computation 없음 — PyTorch 로 인-프로세스 계산.
    - Scenario proto LRU 캐시(기본 256) — 같은 epoch 내 동일 시나리오
      TFRecord 재파싱 방지.
    - WOSAC_TORCH_COMPILE=1 env var 로 dno/ttc/d_road kernel torch.compile 활성화.
    """

    _SCENARIO_CACHE_MAX: int = 256

    _LIKELIHOOD_NAMES = [
        "linear_speed_likelihood",
        "linear_acceleration_likelihood",
        "angular_speed_likelihood",
        "angular_acceleration_likelihood",
        "distance_to_nearest_object_likelihood",
        "collision_indication_likelihood",
        "time_to_collision_likelihood",
        "distance_to_road_edge_likelihood",
        "offroad_indication_likelihood",
        "traffic_light_violation_likelihood",
    ]

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self._metric_key = f"{prefix}/sim_agents_2025/realism_meta_metric"
        self._config = self._load_config()
        self._metametric_sum: float = 0.0
        self._count: int = 0
        self._scenario_cache: Dict[str, Any] = {}
        self._per_metric_sums: Dict[str, float] = {n: 0.0 for n in self._LIKELIHOOD_NAMES}

    @staticmethod
    def _load_config():
        from google.protobuf import text_format
        from waymo_open_dataset.protos import sim_agents_metrics_pb2
        import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as wm

        config_path = Path(wm.__file__).parent / "challenge_2025_sim_agents_config.textproto"
        with open(config_path) as f:
            cfg = sim_agents_metrics_pb2.SimAgentMetricsConfig()
            text_format.Parse(f.read(), cfg)
        return cfg

    def _load_scenario(self, scenario_file: str):
        cached = self._scenario_cache.get(scenario_file)
        if cached is not None:
            return cached
        import tensorflow as tf
        from waymo_open_dataset.protos import scenario_pb2

        tf.config.set_visible_devices([], "GPU")
        scenario = scenario_pb2.Scenario()
        for tfdata in tf.data.TFRecordDataset([scenario_file], compression_type=""):
            scenario.ParseFromString(bytes(tfdata.numpy()))
            break
        if len(self._scenario_cache) >= self._SCENARIO_CACHE_MAX:
            self._scenario_cache.pop(next(iter(self._scenario_cache)))
        self._scenario_cache[scenario_file] = scenario
        return scenario

    def _compute_one(
        self,
        scenario_file: str,
        agent_ids: np.ndarray,
        pred_traj_np: np.ndarray,
        pred_z_np: np.ndarray,
        pred_head_np: np.ndarray,
    ) -> float:
        from waymo_open_dataset.protos import sim_agents_submission_pb2
        from src.smart.metrics.wosac_metric_features_torch.metric_features_torch import (
            compute_scenario_rollouts_features,
        )
        from src.smart.metrics.wosac_metametric_pytorch import (
            compute_wosac_metametric_from_features_torch,
        )

        scenario = self._load_scenario(scenario_file)

        n_agents, n_rollout = pred_traj_np.shape[:2]
        joint_scenes = []
        for i_rollout in range(n_rollout):
            simulated_trajectories = []
            for i_agent in range(n_agents):
                simulated_trajectories.append(
                    sim_agents_submission_pb2.SimulatedTrajectory(
                        center_x=pred_traj_np[i_agent, i_rollout, :, 0],
                        center_y=pred_traj_np[i_agent, i_rollout, :, 1],
                        center_z=pred_z_np[i_agent, i_rollout],
                        heading=pred_head_np[i_agent, i_rollout],
                        object_id=int(agent_ids[i_agent]),
                    )
                )
            joint_scenes.append(
                sim_agents_submission_pb2.JointScene(simulated_trajectories=simulated_trajectories)
            )

        scenario_rollouts = sim_agents_submission_pb2.ScenarioRollouts(
            joint_scenes=joint_scenes,
            scenario_id=scenario.scenario_id,
        )

        log_feat, sim_feat = compute_scenario_rollouts_features(scenario, scenario_rollouts)
        result = compute_wosac_metametric_from_features_torch(
            self._config, log_feat.as_dict(), sim_feat.as_dict()
        )
        return float(result.metametric)

    def update_from_prediction_tensors(
        self,
        *,
        scenario_files: List[str],
        agent_id: torch.Tensor,
        agent_batch: torch.Tensor,
        pred_traj: torch.Tensor,
        pred_z: torch.Tensor,
        pred_head: torch.Tensor,
    ) -> None:
        from src.smart.metrics.wosac_metric_features_torch.metric_features_torch_differentiable import (
            PredictedSimTrajectories,
            compute_metric_features_batched_scenes,
        )
        from src.smart.metrics.wosac_metric_features_torch.metric_features_torch import (
            compute_metric_features,
            scenario_to_joint_scene,
        )
        from waymo_open_dataset.utils.sim_agents import submission_specs

        n_scenarios = len(scenario_files)
        if n_scenarios == 0:
            return

        sizes = [int(s) for s in _tg_degree(agent_batch, dtype=torch.long).tolist()]
        device = pred_traj.device
        G = pred_traj.shape[1]  # (n_agents, G, T, 2)
        T_pred = pred_traj.shape[2]
        _challenge = submission_specs.ChallengeType.SIM_AGENTS

        id_splits = agent_id.split(sizes)
        traj_splits = pred_traj.split(sizes)
        z_splits = pred_z.split(sizes)
        head_splits = pred_head.split(sizes)

        scenarios = [self._load_scenario(sf) for sf in scenario_files]

        # Log features: one per scenario (cached by scenario ID)
        log_feat_dicts: List[dict] = []
        for sc in scenarios:
            log_joint = scenario_to_joint_scene(sc, _challenge)
            lf = compute_metric_features(sc, log_joint, challenge_type=_challenge, use_log_validity=True)
            log_feat_dicts.append(lf.as_dict())

        # Sim features: for each rollout g, batch across all scenarios
        sim_feat_per_g: List[list] = []  # G lists, each of length n_scenarios
        with torch.no_grad():
            for g in range(G):
                preds_g = [
                    PredictedSimTrajectories(
                        object_id=id_splits[i].cpu(),
                        center_x=traj_splits[i][:, g, :, 0],
                        center_y=traj_splits[i][:, g, :, 1],
                        center_z=z_splits[i][:, g, :],
                        heading=head_splits[i][:, g, :],
                        valid=torch.ones(sizes[i], T_pred, dtype=torch.bool, device=device),
                    )
                    for i in range(n_scenarios)
                ]
                feat_list_g = compute_metric_features_batched_scenes(
                    scenarios=scenarios, preds=preds_g, surrogate=None,
                )
                sim_feat_per_g.append(feat_list_g)

        # Stack G rollouts per scenario → (G, E_i, T) then batched hard RMM
        sim_feat_dicts: List[dict] = []
        for i in range(n_scenarios):
            feats_i = [sim_feat_per_g[g][i] for g in range(G)]

            def _cat(field: str) -> torch.Tensor:
                return torch.cat([getattr(f, field) for f in feats_i], dim=0)

            sim_feat_dicts.append({
                "object_id": feats_i[0].object_id,
                "object_type": _cat("object_type"),
                "valid": _cat("valid"),
                "average_displacement_error": _cat("average_displacement_error"),
                "linear_speed": _cat("linear_speed"),
                "linear_acceleration": _cat("linear_acceleration"),
                "angular_speed": _cat("angular_speed"),
                "angular_acceleration": _cat("angular_acceleration"),
                "distance_to_nearest_object": _cat("distance_to_nearest_object"),
                "collision_per_step": _cat("collision_per_step"),
                "time_to_collision": _cat("time_to_collision"),
                "distance_to_road_edge": _cat("distance_to_road_edge"),
                "offroad_per_step": _cat("offroad_per_step"),
                "traffic_light_violation_per_step": _cat("traffic_light_violation_per_step"),
            })

        from src.smart.metrics.wosac_metametric_pytorch import (
            compute_wosac_metametric_from_features_torch,
        )

        for i in range(n_scenarios):
            result = compute_wosac_metametric_from_features_torch(
                self._config, log_feat_dicts[i], sim_feat_dicts[i]
            )
            self._metametric_sum += float(result.metametric)
            for name in self._LIKELIHOOD_NAMES:
                self._per_metric_sums[name] += float(getattr(result, name))
            self._count += 1

    def _drain_completed_futures(self, wait: bool = True, drain_all: bool = True) -> None:
        return None

    def get_state_tensor(self, device: torch.device) -> torch.Tensor:
        """DDP all_reduce 용으로 [metametric_sum, count, *per_metric_sums] tensor 반환."""
        vals = [self._metametric_sum, float(self._count)]
        for name in self._LIKELIHOOD_NAMES:
            vals.append(self._per_metric_sums[name])
        return torch.tensor(vals, device=device, dtype=torch.float64)

    def compute_from_state_tensor(self, reduced_metric_state: torch.Tensor) -> Dict[str, Any]:
        """all_reduce 이후 tensor로부터 metametric + per-metric 평균 계산."""
        total_count = reduced_metric_state[1].clamp_min(1.0)
        result = {
            self._metric_key: (reduced_metric_state[0] / total_count).to(torch.float32)
        }
        for j, name in enumerate(self._LIKELIHOOD_NAMES):
            key = f"{self.prefix}/sim_agents_2025/{name}"
            result[key] = (reduced_metric_state[2 + j] / total_count).to(torch.float32)
        return result

    def compute(self) -> Dict[str, Any]:
        count = max(self._count, 1)
        result = {
            self._metric_key: torch.tensor(self._metametric_sum / count, dtype=torch.float32)
        }
        for name in self._LIKELIHOOD_NAMES:
            key = f"{self.prefix}/sim_agents_2025/{name}"
            result[key] = torch.tensor(self._per_metric_sums[name] / count, dtype=torch.float32)
        return result

    def reset(self) -> None:
        self._metametric_sum = 0.0
        self._count = 0
        self._per_metric_sums = {n: 0.0 for n in self._LIKELIHOOD_NAMES}


class SimAgentsSubmission:
    """Waymo Sim Agents Challenge 제출 파일 생성기.

    validation/test epoch 중 각 배치에서 ScenarioRollouts를 누적하고,
    epoch 종료 시 binproto 샤드 파일들을 tar.gz로 패킹합니다.

    DDP 동작:
    - 각 rank가 자신에게 할당된 시나리오를 독립적으로 binproto 샤드로 저장
    - save_sub_file() 에서 dist.barrier() 후 rank 0만 전체 샤드를 tar.gz로 묶음
    """

    _SHARD_SIZE = 300

    def __init__(
        self,
        is_active: bool,
        method_name: str,
        authors: Any,
        affiliation: str,
        description: str,
        method_link: str,
        account_name: str,
    ) -> None:
        self.is_active = bool(is_active)
        if not self.is_active:
            return

        import hydra as _hydra

        self.method_name = method_name
        self.authors = list(authors)
        self.affiliation = affiliation
        self.description = description
        self.method_link = method_link
        self.account_name = account_name

        output_dir = _hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.submission_dir = Path(output_dir) / "sim_agents_submission"
        self.submission_dir.mkdir(parents=True, exist_ok=True)

        self._buffer: List = []
        self._seen_scenario_ids: set = set()
        self._shard_idx: int = 0
        self._pending_rollouts: List = []

    @staticmethod
    def _rank() -> int:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass
        return 0

    def update(
        self,
        *,
        scenario_id: List[str],
        agent_id: Any,
        agent_batch: Any,
        pred_traj: Any,
        pred_z: Any,
        pred_head: Any,
    ) -> None:
        if not self.is_active:
            return
        import logging as _log
        from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts

        device = pred_traj.device
        scenario_id_tensor = get_scenario_id_int_tensor(scenario_id, device)
        self._pending_rollouts = get_scenario_rollouts(
            scenario_id=scenario_id_tensor,
            agent_id=agent_id,
            agent_batch=agent_batch,
            pred_traj=pred_traj,
            pred_z=pred_z,
            pred_head=pred_head,
        )

    def aggregate_current_batch(self) -> List:
        if not self.is_active:
            return []

        rollouts = self._pending_rollouts
        self._pending_rollouts = []

        rank = self._rank()
        for rollout in rollouts:
            if rollout.scenario_id not in self._seen_scenario_ids:
                self._seen_scenario_ids.add(rollout.scenario_id)
                self._buffer.append(rollout)
                if len(self._buffer) >= self._SHARD_SIZE:
                    self._write_shard(rank)

        return rollouts

    def _write_shard(self, rank: int) -> None:
        import logging as _logging
        from waymo_open_dataset.protos import sim_agents_submission_pb2 as _pb2

        _log = _logging.getLogger(__name__)

        shard_submission = _pb2.SimAgentsChallengeSubmission(
            scenario_rollouts=self._buffer,
            submission_type=_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
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
        filename = self.submission_dir / f"submission_r{rank:02d}-{self._shard_idx:05d}.binproto"
        _log.info(f"[SimAgentsSubmission] Saving shard → {filename}")
        with open(filename, "wb") as f:
            f.write(shard_submission.SerializeToString())
        self._shard_idx += 1
        self._buffer = []

    def save_sub_file(self) -> None:
        if not self.is_active:
            return

        import logging as _logging
        import tarfile as _tarfile

        _log = _logging.getLogger(__name__)
        rank = self._rank()

        if self._buffer:
            self._write_shard(rank)

        # 모든 rank의 샤드 파일 쓰기가 끝날 때까지 대기
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception:
            pass

        # rank 0만 tar.gz 패킹
        if rank != 0:
            return

        tar_path = self.submission_dir.parent / (self.submission_dir.name + ".tar.gz")
        shard_files = sorted(self.submission_dir.glob("*.binproto"))
        n_shards = len(shard_files)

        _log.info(f"[SimAgentsSubmission] Packing {n_shards} shards → {tar_path}")
        with _tarfile.open(str(tar_path), "w:gz") as tar:
            for shard_file in shard_files:
                tar.add(
                    str(shard_file),
                    arcname=shard_file.name + f"-of-{n_shards:05d}",
                )
        _log.info(f"[SimAgentsSubmission] DONE: {tar_path}")
