"""Pure-torch WOSAC 2025 Sim Agents metrics.

Replaces TF feature extraction + TF RMM with:
  - Torch feature extraction (wosac_metric_features_torch)
  - Torch hard RMM (wosac_metametric_pytorch)

Interface is compatible with SimAgentsMetrics so smart_flow.py works unchanged.

State tensor layout (for DDP all_reduce):
  [0]    scenario_counter
  [1]    metametric_sum
  [2:12] 10 individual likelihood sums (see _LIKELIHOOD_FIELDS)
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import torch
from torch import Tensor, tensor
from torchmetrics import Metric
from waymo_open_dataset.protos import (
    scenario_pb2,
    sim_agents_submission_pb2,
)
from waymo_open_dataset.utils.sim_agents import submission_specs

from src.smart.metrics.sim_agents_metrics import (
    _load_waymo_sim_agents_2025_config,
    _read_single_record_tfrecord,
)
from src.smart.metrics.wosac_metametric_pytorch import (
    WosacMetametricTorchResult,
    compute_wosac_metametric_from_features_torch,
)
from src.smart.metrics.wosac_metric_features_torch.metric_features_torch import (
    compute_metric_features,
    scenario_to_joint_scene,
)
from src.smart.metrics.wosac_metric_features_torch.types import MetricFeaturesTorch

_ChallengeType = submission_specs.ChallengeType
_SIM_AGENTS_2025_NAMESPACE = "sim_agents_2025"

# All 10 component likelihoods in proto-field order
_LIKELIHOOD_FIELDS: Tuple[str, ...] = (
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
)

# State tensor indices
_IDX_COUNTER = 0
_IDX_METAMETRIC = 1
_IDX_LIKELIHOODS_START = 2  # [2, 12)

_SCENARIO_PROTO_CACHE: Dict[Tuple[str, bool], scenario_pb2.Scenario] = {}
_SCENARIO_PROTO_CACHE_MAX = 512


def _get_or_parse_scenario(scenario_file: str, ego_only: bool) -> scenario_pb2.Scenario:
    key = (scenario_file, ego_only)
    cached = _SCENARIO_PROTO_CACHE.get(key)
    if cached is not None:
        return cached

    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(_read_single_record_tfrecord(scenario_file))

    if ego_only:
        for i in range(len(scenario.tracks)):
            if i != scenario.sdc_track_index:
                for t in range(91):
                    scenario.tracks[i].states[t].valid = False
        while len(scenario.tracks_to_predict) > 1:
            scenario.tracks_to_predict.pop()
        scenario.tracks_to_predict[0].track_index = scenario.sdc_track_index

    _SCENARIO_PROTO_CACHE[key] = scenario
    if len(_SCENARIO_PROTO_CACHE) > _SCENARIO_PROTO_CACHE_MAX:
        _SCENARIO_PROTO_CACHE.pop(next(iter(_SCENARIO_PROTO_CACHE)))
    return scenario


def _build_joint_scene(
    agent_ids,
    traj_np,   # (A, T, 2): x, y
    z_np,      # (A, T)
    head_np,   # (A, T)
) -> sim_agents_submission_pb2.JointScene:
    trajs = []
    for a in range(len(agent_ids)):
        trajs.append(
            sim_agents_submission_pb2.SimulatedTrajectory(
                object_id=int(agent_ids[a]),
                center_x=traj_np[a, :, 0].tolist(),
                center_y=traj_np[a, :, 1].tolist(),
                center_z=z_np[a, :].tolist(),
                heading=head_np[a, :].tolist(),
            )
        )
    return sim_agents_submission_pb2.JointScene(simulated_trajectories=trajs)


def _cat_metric_features(feats: List[MetricFeaturesTorch]) -> MetricFeaturesTorch:
    def cat(field: str) -> Tensor:
        return torch.cat([getattr(f, field) for f in feats], dim=0)

    return MetricFeaturesTorch(
        object_id=feats[0].object_id,
        object_type=cat("object_type"),
        valid=cat("valid"),
        average_displacement_error=cat("average_displacement_error"),
        linear_speed=cat("linear_speed"),
        linear_acceleration=cat("linear_acceleration"),
        angular_speed=cat("angular_speed"),
        angular_acceleration=cat("angular_acceleration"),
        distance_to_nearest_object=cat("distance_to_nearest_object"),
        collision_per_step=cat("collision_per_step"),
        time_to_collision=cat("time_to_collision"),
        distance_to_road_edge=cat("distance_to_road_edge"),
        offroad_per_step=cat("offroad_per_step"),
        traffic_light_violation_per_step=cat("traffic_light_violation_per_step"),
    )


class SimAgentsTorchMetrics(Metric):
    """Torch-native 2025 Sim Agents metametric — no TF ops during validation.

    Same external interface as SimAgentsMetrics so smart_flow.py works unchanged.

    Output dict (compute / compute_from_state_tensor):
      {prefix}/sim_agents_2025/scenario_counter
      {prefix}/sim_agents_2025/realism_meta_metric      ← checkpoint monitor
      {prefix}/sim_agents_2025_mean/linear_speed_likelihood
      {prefix}/sim_agents_2025_mean/linear_acceleration_likelihood
      {prefix}/sim_agents_2025_mean/angular_speed_likelihood
      {prefix}/sim_agents_2025_mean/angular_acceleration_likelihood
      {prefix}/sim_agents_2025_mean/distance_to_nearest_object_likelihood
      {prefix}/sim_agents_2025_mean/collision_indication_likelihood
      {prefix}/sim_agents_2025_mean/time_to_collision_likelihood
      {prefix}/sim_agents_2025_mean/distance_to_road_edge_likelihood
      {prefix}/sim_agents_2025_mean/offroad_indication_likelihood
      {prefix}/sim_agents_2025_mean/traffic_light_violation_likelihood
    """

    full_state_update = False

    def __init__(self, prefix: str, ego_only: bool = False, **kwargs) -> None:
        super().__init__()
        self.prefix = prefix
        self.ego_only = ego_only
        self.metric_namespace = f"{self.prefix}/{_SIM_AGENTS_2025_NAMESPACE}"
        self.metric_mean_namespace = f"{self.prefix}/{_SIM_AGENTS_2025_NAMESPACE}_mean"

        self.sim_agents_config = _load_waymo_sim_agents_2025_config()

        # State: scenario_counter + metametric_sum + 10 likelihood sums
        self.add_state("scenario_counter", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("metametric_sum", default=tensor(0.0), dist_reduce_fx="sum")
        for field in _LIKELIHOOD_FIELDS:
            self.add_state(field + "_sum", default=tensor(0.0), dist_reduce_fx="sum")

    # ------------------------------------------------------------------
    # Compatibility shim: no async workers in torch backend
    # ------------------------------------------------------------------

    def _drain_completed_futures(self, wait: bool, drain_all: bool = False) -> None:
        pass

    # ------------------------------------------------------------------
    # DDP-compatible state serialization (mirrors SimAgentsMetrics API)
    # ------------------------------------------------------------------

    def get_state_tensor(self, device: torch.device) -> Tensor:
        parts = [
            self.scenario_counter.detach().to(device=device),
            self.metametric_sum.detach().to(device=device),
        ]
        for field in _LIKELIHOOD_FIELDS:
            parts.append(getattr(self, field + "_sum").detach().to(device=device))
        return torch.stack(parts)  # shape (12,)

    def compute_from_state_tensor(self, state_tensor: Tensor) -> Dict[str, Tensor]:
        n = state_tensor[_IDX_COUNTER]
        safe_n = n.clamp_min(1.0)

        out: Dict[str, Tensor] = {
            f"{self.metric_namespace}/scenario_counter": n,
            f"{self.metric_namespace}/realism_meta_metric": state_tensor[_IDX_METAMETRIC] / safe_n,
        }
        for i, field in enumerate(_LIKELIHOOD_FIELDS):
            out[f"{self.metric_mean_namespace}/{field}"] = (
                state_tensor[_IDX_LIKELIHOODS_START + i] / safe_n
            )
        return out

    # ------------------------------------------------------------------
    # torchmetrics API
    # ------------------------------------------------------------------

    def update(self, *args, **kwargs) -> None:
        pass

    def compute(self) -> Dict[str, Tensor]:
        return self.compute_from_state_tensor(
            self.get_state_tensor(self.metametric_sum.device)
        )

    def reset(self) -> None:
        super().reset()
        _SCENARIO_PROTO_CACHE.clear()

    # ------------------------------------------------------------------
    # Main update entry points (same signature as SimAgentsMetrics)
    # ------------------------------------------------------------------

    @staticmethod
    def build_prediction_payloads(
        scenario_files: List[str],
        agent_id: Tensor,
        agent_batch: Tensor,
        pred_traj: Tensor,
        pred_z: Tensor,
        pred_head: Tensor,
    ) -> List[Tuple]:
        agent_batch_cpu = agent_batch.detach().to(device="cpu", dtype=torch.long)
        sizes = torch.bincount(agent_batch_cpu).tolist()
        agent_id_cpu = agent_id.detach().cpu()
        pred_traj_cpu = pred_traj.detach().cpu()
        pred_z_cpu = pred_z.detach().cpu()
        pred_head_cpu = pred_head.detach().cpu()

        start = 0
        payloads = []
        for scenario_file, size in zip(scenario_files, sizes):
            end = start + int(size)
            payloads.append((
                scenario_file,
                agent_id_cpu[start:end].numpy(),
                pred_traj_cpu[start:end].numpy(),
                pred_z_cpu[start:end].numpy(),
                pred_head_cpu[start:end].numpy(),
            ))
            start = end
        return payloads

    def update_from_prediction_tensors(
        self,
        scenario_files: List[str],
        agent_id: Tensor,
        agent_batch: Tensor,
        pred_traj: Tensor,
        pred_z: Tensor,
        pred_head: Tensor,
    ) -> None:
        if len(scenario_files) == 0:
            return
        self.update_from_prediction_payloads(
            self.build_prediction_payloads(
                scenario_files, agent_id, agent_batch, pred_traj, pred_z, pred_head
            )
        )

    def update_from_prediction_payloads(
        self,
        scenario_payloads: List[Tuple],
    ) -> None:
        if not scenario_payloads:
            return

        self._computed = None
        self._update_count += 1

        for scenario_file, agent_ids_np, pred_traj_np, pred_z_np, pred_head_np in scenario_payloads:
            scenario = _get_or_parse_scenario(scenario_file, self.ego_only)

            # Log features (once per scenario, uses log validity)
            log_joint = scenario_to_joint_scene(scenario)
            log_feat = compute_metric_features(
                scenario, log_joint, use_log_validity=True
            )

            # Sim features for all G rollouts, concatenated along sample dim
            G = pred_traj_np.shape[1]
            sim_feats: List[MetricFeaturesTorch] = []
            for g in range(G):
                joint_scene = _build_joint_scene(
                    agent_ids_np,
                    pred_traj_np[:, g],   # (A, T, 2)
                    pred_z_np[:, g],      # (A, T)
                    pred_head_np[:, g],   # (A, T)
                )
                sim_feats.append(
                    compute_metric_features(scenario, joint_scene, use_log_validity=False)
                )

            sim_combined = _cat_metric_features(sim_feats)

            # Single-scenario hard RMM — returns all 10 individual likelihoods
            result: WosacMetametricTorchResult = compute_wosac_metametric_from_features_torch(
                self.sim_agents_config,
                log_feat.as_dict(),
                sim_combined.as_dict(),
            )

            self.metametric_sum.add_(tensor(result.metametric))
            self.scenario_counter.add_(1.0)
            for field in _LIKELIHOOD_FIELDS:
                getattr(self, field + "_sum").add_(tensor(getattr(result, field)))
