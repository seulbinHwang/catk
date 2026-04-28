"""Pure-torch WOSAC 2025 Sim Agents metrics.

Replaces TF feature extraction + TF RMM with:
  - Torch feature extraction (wosac_metric_features_torch)
  - Torch hard RMM (wosac_metametric_pytorch)

Interface is compatible with SimAgentsMetrics so smart_flow.py works unchanged.

State tensor layout (for DDP all_reduce):
  [0]    scenario_counter
  [1]    metametric_sum
  [2:12] 10 individual likelihood sums (see _LIKELIHOOD_FIELDS)

Key design decisions vs naive G-rollout batching:
  - DNO and TTC are O(N²) pairwise — batching G rollouts would make them O(G²N²),
    causing OOM (G=32, N=50 → 1600² intermediate tensors). Process per rollout.
  - Road-edge is O(N*R*S) with large intermediate (N*T*4_corners × R × S_segments);
    batching G would exceed VRAM. Process per rollout.
  - Kinematics are O(N*T) — safe to batch, but savings are minor; kept per-rollout
    for simplicity.

What IS faster than official TF:
  1. logged_full_cpu cached once per scenario (was called 2*(G+1) times)
  2. road_edge_polylines_tensor cached once per scenario (not rebuilt per rollout)
  3. Trajectory tensors built directly from numpy arrays — no JointScene proto round-trip
  4. All heavy ops run on GPU (device="cuda")
  5. No subprocess IPC, no TF eager overhead, no main-thread blocking on futures
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import torch
from torch import Tensor, tensor
from torchmetrics import Metric
from waymo_open_dataset.protos import scenario_pb2
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
    _cache_get_or_build,
    object_trajectories_from_scenario,
    ObjectTrajectoriesTorch,
    scenario_to_joint_scene,
    compute_metric_features,
)
from src.smart.metrics.wosac_metric_features_torch import (
    interaction_features_torch as inter,
    map_metric_features_torch as map_feat,
    trajectory_features_torch as traj_feat,
    traffic_light_features_torch as tl_feat,
)
from src.smart.metrics.wosac_metric_features_torch.types import MetricFeaturesTorch

_ChallengeType = submission_specs.ChallengeType
_SIM_AGENTS_2025_NAMESPACE = "sim_agents_2025"

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

_IDX_COUNTER = 0
_IDX_METAMETRIC = 1
_IDX_LIKELIHOODS_START = 2

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


def _cat_metric_features(feats: List[MetricFeaturesTorch]) -> MetricFeaturesTorch:
    def cat(f: str) -> Tensor:
        return torch.cat([getattr(x, f) for x in feats], dim=0)

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


def _build_sim_traj(
    logged_hist: ObjectTrajectoriesTorch,
    sim_ids: Tensor,
    traj_np,    # (A, T_sim, 2)  x, y
    z_np,       # (A, T_sim)
    head_np,    # (A, T_sim)
    device: torch.device,
) -> ObjectTrajectoriesTorch:
    """Build a full (history + future) ObjectTrajectoriesTorch directly from arrays.

    Avoids JointScene proto construction + re-parsing, which was the main CPU
    overhead in the per-rollout loop.
    """
    T_sim = traj_np.shape[1]

    hist_x   = logged_hist.x.to(device)          # (A, T_hist)
    hist_y   = logged_hist.y.to(device)
    hist_z   = logged_hist.z.to(device)
    hist_h   = logged_hist.heading.to(device)
    hist_len = logged_hist.length.to(device)
    hist_wid = logged_hist.width.to(device)
    hist_hei = logged_hist.height.to(device)
    hist_val = logged_hist.valid.to(device)

    sim_x   = torch.tensor(traj_np[:, :, 0], dtype=torch.float32, device=device)
    sim_y   = torch.tensor(traj_np[:, :, 1], dtype=torch.float32, device=device)
    sim_z   = torch.tensor(z_np,   dtype=torch.float32, device=device)
    sim_h   = torch.tensor(head_np, dtype=torch.float32, device=device)
    sim_len = hist_len[:, -1:].expand(-1, T_sim)
    sim_wid = hist_wid[:, -1:].expand(-1, T_sim)
    sim_hei = hist_hei[:, -1:].expand(-1, T_sim)
    sim_val = torch.ones(len(sim_ids), T_sim, dtype=torch.bool, device=device)

    return ObjectTrajectoriesTorch(
        x=torch.cat([hist_x, sim_x], dim=1),
        y=torch.cat([hist_y, sim_y], dim=1),
        z=torch.cat([hist_z, sim_z], dim=1),
        heading=torch.cat([hist_h, sim_h], dim=1),
        length=torch.cat([hist_len, sim_len], dim=1),
        width=torch.cat([hist_wid, sim_wid], dim=1),
        height=torch.cat([hist_hei, sim_hei], dim=1),
        valid=torch.cat([hist_val, sim_val], dim=1),
        object_id=sim_ids.to(device),
        object_type=logged_hist.object_type.to(device),
    )


def _compute_one_rollout_features(
    sim_traj: ObjectTrajectoriesTorch,
    logged_full: ObjectTrajectoriesTorch,
    eval_ids: Tensor,
    reordered_ids: Tensor,
    cached: dict,
    cfg,
    device: torch.device,
) -> MetricFeaturesTorch:
    """Compute MetricFeaturesTorch for a single rollout.

    Uses cached map tensors (road_edge_polylines_tensor, lane_tensor, etc.)
    so per-rollout cost is just the actual feature computation, not map parsing.
    """
    ct_idx = cfg.current_time_index
    step_dur = float(cfg.step_duration_seconds)

    # Reorder: evaluated agents first, then rest
    sim_reordered = sim_traj.gather_objects_by_id(reordered_ids.to(device))
    eval_logged   = logged_full.gather_objects_by_id(eval_ids).slice_time(0, None)

    N_sim  = len(reordered_ids)
    N_eval = len(eval_ids)
    eval_mask = torch.zeros(N_sim, dtype=torch.bool, device=device)
    eval_mask[:N_eval] = True

    # Kinematics (evaluated agents only)
    evaluated = sim_reordered.slice_time(0, None)   # full timeline
    eval_x = evaluated.x[:N_eval]
    eval_y = evaluated.y[:N_eval]
    eval_z = evaluated.z[:N_eval]
    eval_h = evaluated.heading[:N_eval]
    eval_v = evaluated.valid[:N_eval]

    lin_spd, lin_acc, ang_spd, ang_acc = traj_feat.compute_kinematic_features(
        eval_x, eval_y, eval_z, eval_h, seconds_per_step=step_dur
    )

    # ADE
    elog_x = eval_logged.x.to(device)
    elog_y = eval_logged.y.to(device)
    elog_z = eval_logged.z.to(device)
    elog_val = eval_logged.valid.to(device)
    disp_err = traj_feat.compute_displacement_error(eval_x, eval_y, eval_z, elog_x, elog_y, elog_z)
    valid_steps = elog_val.float().sum(dim=1).clamp_min(1.0)
    ade = (torch.where(elog_val, disp_err, torch.zeros_like(disp_err)).sum(dim=1) / valid_steps)

    # Pairwise distances — kept per-rollout to avoid O(G²N²) memory
    dno = inter.compute_distance_to_nearest_object(
        center_x=sim_reordered.x,
        center_y=sim_reordered.y,
        center_z=sim_reordered.z,
        length=sim_reordered.length,
        width=sim_reordered.width,
        height=sim_reordered.height,
        heading=sim_reordered.heading,
        valid=sim_reordered.valid,
        evaluated_object_mask=eval_mask,
    )  # (N_eval, T_full)

    ttc = inter.compute_time_to_collision_with_object_in_front(
        center_x=sim_reordered.x,
        center_y=sim_reordered.y,
        length=sim_reordered.length,
        width=sim_reordered.width,
        heading=sim_reordered.heading,
        valid=sim_reordered.valid,
        evaluated_object_mask=eval_mask,
        seconds_per_step=step_dur,
    )  # (N_eval, T_full)

    # Road edge — uses cached tensor (not rebuilt per rollout)
    d_road = map_feat.compute_distance_to_road_edge(
        center_x=sim_reordered.x,
        center_y=sim_reordered.y,
        center_z=sim_reordered.z,
        length=sim_reordered.length,
        width=sim_reordered.width,
        height=sim_reordered.height,
        heading=sim_reordered.heading,
        valid=sim_reordered.valid,
        evaluated_object_mask=eval_mask,
        road_edge_polylines=cached.get("road_edges") or [],
        road_edge_polylines_tensor=cached.get("road_edge_polylines_tensor"),
        is_polyline_cyclic=cached.get("road_edge_is_cyclic"),
    )  # (N_eval, T_full)

    # Traffic light
    lane_polys = cached.get("lane_polys") or []
    traffic_signals = cached.get("traffic_signals") or []
    if lane_polys and traffic_signals:
        red_light = tl_feat.compute_red_light_violation(
            center_x=sim_reordered.x,
            center_y=sim_reordered.y,
            valid=sim_reordered.valid,
            evaluated_object_mask=eval_mask,
            lane_polylines=lane_polys,
            lane_ids=cached.get("lane_ids") or [],
            traffic_signals=traffic_signals,
            lane_tensor=cached.get("lane_tensor"),
            lane_ids_tensor=cached.get("lane_ids_tensor"),
            ts_lane_id=cached.get("ts_lane_id"),
            ts_state=cached.get("ts_state"),
            ts_stop_point=cached.get("ts_stop_point"),
        )  # (N_eval, T_full)
    else:
        T_full = sim_reordered.x.shape[1]
        red_light = torch.zeros(N_eval, T_full, dtype=torch.bool, device=device)

    # Slice off history
    s = ct_idx + 1
    eval_val_fut = eval_v[:, s:]

    return MetricFeaturesTorch(
        object_id=eval_ids.to(device),
        object_type=sim_reordered.object_type[:N_eval].unsqueeze(0),
        valid=eval_val_fut.unsqueeze(0),
        average_displacement_error=ade.unsqueeze(0),
        linear_speed=lin_spd[:, s:].unsqueeze(0),
        linear_acceleration=lin_acc[:, s:].unsqueeze(0),
        angular_speed=ang_spd[:, s:].unsqueeze(0),
        angular_acceleration=ang_acc[:, s:].unsqueeze(0),
        distance_to_nearest_object=dno[:, s:].unsqueeze(0),
        collision_per_step=(dno[:, s:] < inter.COLLISION_DISTANCE_THRESHOLD).unsqueeze(0),
        time_to_collision=ttc[:, s:].unsqueeze(0),
        distance_to_road_edge=d_road[:, s:].unsqueeze(0),
        offroad_per_step=(d_road[:, s:] > map_feat.OFFROAD_DISTANCE_THRESHOLD).unsqueeze(0),
        traffic_light_violation_per_step=red_light[:, s:].unsqueeze(0),
    )


def _compute_sim_features_all_rollouts(
    scenario: scenario_pb2.Scenario,
    agent_ids_np,    # (A,)
    pred_traj_np,    # (A, G, T_sim, 2)
    pred_z_np,       # (A, G, T_sim)
    pred_head_np,    # (A, G, T_sim)
    device: torch.device,
) -> MetricFeaturesTorch:
    """Compute sim MetricFeaturesTorch for all G rollouts.

    Processes rollouts sequentially to avoid O(G²N²) memory in DNO/TTC.
    Builds trajectory tensors directly from arrays (no JointScene proto).
    Uses cached logged_full and map tensors.
    """
    cfg = submission_specs.get_submission_config(_ChallengeType.SIM_AGENTS)
    ct_idx = cfg.current_time_index

    cached = _cache_get_or_build(scenario)
    logged_full = cached.get("logged_full_cpu") or object_trajectories_from_scenario(scenario)

    eval_ids_list = cached.get("eval_ids_list")
    if eval_ids_list is None:
        eval_ids_list = list(
            submission_specs.get_evaluation_sim_agent_ids(scenario, _ChallengeType.SIM_AGENTS)
        )

    sim_ids  = torch.tensor(agent_ids_np, dtype=torch.int64)
    eval_ids = torch.tensor(eval_ids_list, dtype=torch.int64)

    # Reorder once: evaluated first, then non-evaluated
    eval_set = set(eval_ids_list)
    non_eval = [oid for oid in agent_ids_np.tolist() if oid not in eval_set]
    reordered_ids = torch.tensor(list(eval_ids_list) + non_eval, dtype=torch.int64)

    # History slice from cache — gathered once, shared across all G rollouts
    logged_hist = logged_full.slice_time(0, ct_idx + 1).gather_objects_by_id(sim_ids)

    sim_feats: List[MetricFeaturesTorch] = []
    G = pred_traj_np.shape[1]
    for g in range(G):
        sim_traj = _build_sim_traj(
            logged_hist, sim_ids,
            pred_traj_np[:, g], pred_z_np[:, g], pred_head_np[:, g],
            device,
        )
        feat = _compute_one_rollout_features(
            sim_traj, logged_full, eval_ids, reordered_ids, cached, cfg, device
        )
        sim_feats.append(feat)

    return _cat_metric_features(sim_feats)


class SimAgentsTorchMetrics(Metric):
    """Torch-native 2025 Sim Agents metametric — no TF ops during validation.

    Same external interface as SimAgentsMetrics so smart_flow.py works unchanged.

    Output dict (compute / compute_from_state_tensor):
      {prefix}/sim_agents_2025/scenario_counter
      {prefix}/sim_agents_2025/realism_meta_metric      ← checkpoint monitor
      {prefix}/sim_agents_2025_mean/<likelihood_name>   ← 10 individual likelihoods

    Args:
        prefix:    metric key prefix (e.g. "val_closed")
        ego_only:  if True, mask out non-SDC agents in the scenario
        device:    torch device for feature computation. Defaults to CUDA if available.
                   Pass "cpu" to force CPU (useful for debugging parity).
    """

    full_state_update = False

    def __init__(
        self,
        prefix: str,
        ego_only: bool = False,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.ego_only = ego_only
        self._compute_device = torch.device(device) if device else None
        self.metric_namespace = f"{self.prefix}/{_SIM_AGENTS_2025_NAMESPACE}"
        self.metric_mean_namespace = f"{self.prefix}/{_SIM_AGENTS_2025_NAMESPACE}_mean"

        self.sim_agents_config = _load_waymo_sim_agents_2025_config()

        self.add_state("scenario_counter", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("metametric_sum",   default=tensor(0.0), dist_reduce_fx="sum")
        for field in _LIKELIHOOD_FIELDS:
            self.add_state(field + "_sum", default=tensor(0.0), dist_reduce_fx="sum")

    def _feature_device(self) -> torch.device:
        if self._compute_device is not None:
            return self._compute_device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Compatibility shim: no async worker pool
    # ------------------------------------------------------------------

    def _drain_completed_futures(self, wait: bool, drain_all: bool = False) -> None:
        pass

    # ------------------------------------------------------------------
    # DDP state tensor (mirrors SimAgentsMetrics API for all_reduce)
    # ------------------------------------------------------------------

    def get_state_tensor(self, device: torch.device) -> Tensor:
        parts = [
            self.scenario_counter.detach().to(device=device),
            self.metametric_sum.detach().to(device=device),
        ]
        for field in _LIKELIHOOD_FIELDS:
            parts.append(getattr(self, field + "_sum").detach().to(device=device))
        return torch.stack(parts)  # (12,)

    def compute_from_state_tensor(self, state_tensor: Tensor) -> Dict[str, Tensor]:
        n = state_tensor[_IDX_COUNTER]
        safe_n = n.clamp_min(1.0)
        out: Dict[str, Tensor] = {
            f"{self.metric_namespace}/scenario_counter":     n,
            f"{self.metric_namespace}/realism_meta_metric":  state_tensor[_IDX_METAMETRIC] / safe_n,
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
    # Main entry points (same signature as SimAgentsMetrics)
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
        agent_id_cpu  = agent_id.detach().cpu()
        pred_traj_cpu = pred_traj.detach().cpu()
        pred_z_cpu    = pred_z.detach().cpu()
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
        if not scenario_files:
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

        dev = self._feature_device()

        for scenario_file, agent_ids_np, pred_traj_np, pred_z_np, pred_head_np in scenario_payloads:
            scenario = _get_or_parse_scenario(scenario_file, self.ego_only)

            # Log features: once per scenario, uses cached logged_full
            log_joint = scenario_to_joint_scene(scenario)
            log_feat  = compute_metric_features(scenario, log_joint, use_log_validity=True)
            log_dict  = {k: v.to(dev) for k, v in log_feat.as_dict().items()}

            # Sim features: per-rollout loop (avoids O(G²N²) memory)
            sim_feat = _compute_sim_features_all_rollouts(
                scenario,
                agent_ids_np,
                pred_traj_np,   # (A, G, T_sim, 2)
                pred_z_np,
                pred_head_np,
                device=dev,
            )
            sim_dict = {k: v for k, v in sim_feat.as_dict().items()}

            result: WosacMetametricTorchResult = compute_wosac_metametric_from_features_torch(
                self.sim_agents_config, log_dict, sim_dict
            )

            self.metametric_sum.add_(tensor(result.metametric))
            self.scenario_counter.add_(1.0)
            for field in _LIKELIHOOD_FIELDS:
                getattr(self, field + "_sum").add_(tensor(getattr(result, field)))
