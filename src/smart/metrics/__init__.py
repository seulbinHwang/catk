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

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self._metric_key = f"{prefix}/sim_agents_2025/realism_meta_metric"
        self._config = self._load_config()
        self._metametric_sum: float = 0.0
        self._count: int = 0
        self._scenario_cache: Dict[str, Any] = {}

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
        sizes = [int(s) for s in _tg_degree(agent_batch, dtype=torch.long).tolist()]
        agent_id_list = [t.cpu().numpy() for t in agent_id.cpu().split(sizes)]
        pred_traj_list = [t.cpu().numpy() for t in pred_traj.cpu().split(sizes)]
        pred_z_list = [t.cpu().numpy() for t in pred_z.cpu().split(sizes)]
        pred_head_list = [t.cpu().numpy() for t in pred_head.cpu().split(sizes)]

        for i, scenario_file in enumerate(scenario_files):
            metametric = self._compute_one(
                scenario_file,
                agent_id_list[i],
                pred_traj_list[i],
                pred_z_list[i],
                pred_head_list[i],
            )
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


class SimAgentsSubmission:
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

    def update(self, **kwargs: Any) -> None:
        return None

    def aggregate_current_batch(self) -> List[Any]:
        return []

    def save_sub_file(self) -> None:
        return None
