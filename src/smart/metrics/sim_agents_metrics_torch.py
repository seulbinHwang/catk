"""Pure-torch WOSAC 2025 Sim Agents metrics.

Replaces TF feature extraction + TF RMM with:
  - Torch feature extraction (wosac_metric_features_torch)
  - Torch hard RMM (wosac_metametric_pytorch)

Interface is compatible with SimAgentsMetrics so smart_flow.py works unchanged.

State tensor layout (for DDP all_reduce):
  [0]    scenario_counter
  [1]    metametric_sum
  [2:12] 10 individual likelihood sums (see _LIKELIHOOD_FIELDS)

Optimizations vs official TF backend:
  1. road_edge_polylines_tensor cached once per scenario (not per-rollout)
  2. logged_full_cpu (ObjectTrajectoriesTorch) cached once per scenario
  3. All G rollouts processed in a single batched call using scene_batch masking
  4. device="cuda" routes all feature computation to GPU
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import torch
import torch.nn.functional as F
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
    compute_metric_features,
    scenario_to_joint_scene,
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


def _compute_sim_features_all_rollouts(
    scenario: scenario_pb2.Scenario,
    agent_ids_np,       # (A,)  int
    pred_traj_np,       # (A, G, T_sim, 2)  x, y
    pred_z_np,          # (A, G, T_sim)
    pred_head_np,       # (A, G, T_sim)
    device: torch.device,
) -> MetricFeaturesTorch:
    """Process all G rollouts in one batched pass — no per-rollout Python loop.

    Instead of calling compute_metric_features G times (each building tensors,
    computing distances independently), this function:
      - Builds (G*N_sim, T_full) trajectory tensors once
      - Runs compute_distance_to_nearest_object once with scene_batch masking
      - Runs compute_distance_to_road_edge once with the cached road edge tensor
      - Reshapes output to (G, N_eval, T_sim)

    All heavy ops execute on `device` (pass device="cuda" for GPU acceleration).
    """
    cfg = submission_specs.get_submission_config(_ChallengeType.SIM_AGENTS)
    ct_idx = cfg.current_time_index
    step_dur = float(cfg.step_duration_seconds)

    cached = _cache_get_or_build(scenario)
    logged_full = cached.get("logged_full_cpu") or object_trajectories_from_scenario(scenario)
    eval_ids_list = cached.get("eval_ids_list")
    if eval_ids_list is None:
        eval_ids_list = list(submission_specs.get_evaluation_sim_agent_ids(
            scenario, _ChallengeType.SIM_AGENTS
        ))

    sim_ids = torch.tensor(agent_ids_np, dtype=torch.int64)
    eval_ids = torch.tensor(eval_ids_list, dtype=torch.int64)

    A = len(agent_ids_np)
    G = pred_traj_np.shape[1]
    T_sim = pred_traj_np.shape[2]
    T_hist = ct_idx + 1
    T_full = T_hist + T_sim

    # ── static (history) tensors from log — shape (A, T_hist) ──────────────
    logged_hist = logged_full.slice_time(0, T_hist).gather_objects_by_id(sim_ids)

    hist_x   = logged_hist.x.to(device)        # (A, T_hist)
    hist_y   = logged_hist.y.to(device)
    hist_z   = logged_hist.z.to(device)
    hist_h   = logged_hist.heading.to(device)
    hist_len = logged_hist.length.to(device)
    hist_wid = logged_hist.width.to(device)
    hist_hei = logged_hist.height.to(device)
    hist_val = logged_hist.valid.to(device)

    # ── predicted future tensors — broadcast to (A, G, T_sim) ──────────────
    pred_x   = torch.tensor(pred_traj_np[:, :, :, 0], dtype=torch.float32, device=device)
    pred_y   = torch.tensor(pred_traj_np[:, :, :, 1], dtype=torch.float32, device=device)
    pred_z   = torch.tensor(pred_z_np,   dtype=torch.float32, device=device)
    pred_h   = torch.tensor(pred_head_np, dtype=torch.float32, device=device)

    last_len = hist_len[:, -1:].unsqueeze(1).expand(A, G, T_sim)
    last_wid = hist_wid[:, -1:].unsqueeze(1).expand(A, G, T_sim)
    last_hei = hist_hei[:, -1:].unsqueeze(1).expand(A, G, T_sim)

    # ── full trajectory: (A, G, T_full) ─────────────────────────────────────
    hist_exp = lambda h: h.unsqueeze(1).expand(A, G, T_hist)  # broadcast over G
    full_x   = torch.cat([hist_exp(hist_x),   pred_x],  dim=2)  # (A, G, T_full)
    full_y   = torch.cat([hist_exp(hist_y),   pred_y],  dim=2)
    full_z   = torch.cat([hist_exp(hist_z),   pred_z],  dim=2)
    full_h   = torch.cat([hist_exp(hist_h),   pred_h],  dim=2)
    full_len = torch.cat([hist_exp(hist_len), last_len], dim=2)
    full_wid = torch.cat([hist_exp(hist_wid), last_wid], dim=2)
    full_hei = torch.cat([hist_exp(hist_hei), last_hei], dim=2)
    hist_val_bool = hist_val.bool()
    sim_val_future = torch.ones(A, G, T_sim, dtype=torch.bool, device=device)
    full_val = torch.cat([hist_exp(hist_val_bool), sim_val_future], dim=2)

    # ── flatten: (G*A, T_full) for batched ops ───────────────────────────────
    def flat(t):  # (A, G, T) → (G*A, T)
        return t.permute(1, 0, 2).reshape(G * A, -1)

    f_x   = flat(full_x)
    f_y   = flat(full_y)
    f_z   = flat(full_z)
    f_h   = flat(full_h)
    f_len = flat(full_len)
    f_wid = flat(full_wid)
    f_hei = flat(full_hei)
    f_val = flat(full_val)

    # scene_batch: (G*A,) — agents from different rollouts never interact
    scene_batch = torch.arange(G, device=device).repeat_interleave(A)

    # ── evaluated object mask (same for all rollouts) ────────────────────────
    eval_set = set(eval_ids.tolist())
    sim_id_list = sim_ids.tolist()
    # reorder: evaluated first, then rest (matches official reordering)
    non_eval_ids = [oid for oid in sim_id_list if oid not in eval_set]
    reordered_ids = torch.tensor(list(eval_ids.tolist()) + non_eval_ids, dtype=torch.int64)
    N_eval = len(eval_ids)
    N_sim = A

    id_to_flat_idx = {oid: i for i, oid in enumerate(sim_id_list)}
    reorder_idx = torch.tensor([id_to_flat_idx[oid] for oid in reordered_ids.tolist()],
                                dtype=torch.long, device=device)

    def reorder_flat(t):  # (G*A, ...) reorder agents within each rollout
        # reorder_idx applies per rollout — expand by G with offset
        offset = torch.arange(G, device=device).repeat_interleave(A) * A
        full_idx = (reorder_idx.repeat(G) + offset.repeat_interleave(1))
        # Cleaner: do it per-rollout
        t_reshaped = t.reshape(G, A, -1)
        t_reordered = t_reshaped.index_select(1, reorder_idx)
        return t_reordered.reshape(G * N_sim, -1)

    # Simpler: reorder once at the (G, A) level
    def reorder_ga(t_ga):  # (A, G, T) or (G, A, T) after permute
        return t_ga.index_select(1, reorder_idx)  # already (G, A, T)

    # Permute to (G, A, T) first for clean reordering
    def pga(t):  # (A, G, T) → (G, A, T)
        return t.permute(1, 0, 2)

    gf_x   = reorder_ga(pga(full_x)).reshape(G * N_sim, T_full)
    gf_y   = reorder_ga(pga(full_y)).reshape(G * N_sim, T_full)
    gf_z   = reorder_ga(pga(full_z)).reshape(G * N_sim, T_full)
    gf_h   = reorder_ga(pga(full_h)).reshape(G * N_sim, T_full)
    gf_len = reorder_ga(pga(full_len)).reshape(G * N_sim, T_full)
    gf_wid = reorder_ga(pga(full_wid)).reshape(G * N_sim, T_full)
    gf_hei = reorder_ga(pga(full_hei)).reshape(G * N_sim, T_full)
    gf_val = reorder_ga(pga(full_val)).reshape(G * N_sim, T_full)

    eval_object_mask = torch.zeros(N_sim, dtype=torch.bool, device=device)
    eval_object_mask[:N_eval] = True
    # For (G*N_sim,): each block of N_sim agents has the same mask pattern
    flat_eval_mask = eval_object_mask.repeat(G)

    # scene_batch for reordered agents
    flat_scene_batch = torch.arange(G, device=device).repeat_interleave(N_sim)

    # ── kinematics — (G*N_eval, T_full) then slice history ──────────────────
    eval_idx_t = torch.arange(N_eval, device=device)
    # Build evaluated-only tensors: (G, N_eval, T_full) → (G*N_eval, T_full)
    ga_x = gf_x.reshape(G, N_sim, T_full)
    ga_y = gf_y.reshape(G, N_sim, T_full)
    ga_z = gf_z.reshape(G, N_sim, T_full)
    ga_h = gf_h.reshape(G, N_sim, T_full)
    ga_val = gf_val.reshape(G, N_sim, T_full)

    eval_x = ga_x[:, :N_eval, :].reshape(G * N_eval, T_full)
    eval_y = ga_y[:, :N_eval, :].reshape(G * N_eval, T_full)
    eval_z = ga_z[:, :N_eval, :].reshape(G * N_eval, T_full)
    eval_h = ga_h[:, :N_eval, :].reshape(G * N_eval, T_full)
    eval_val = ga_val[:, :N_eval, :].reshape(G * N_eval, T_full)

    lin_spd, lin_acc, ang_spd, ang_acc = traj_feat.compute_kinematic_features(
        eval_x, eval_y, eval_z, eval_h, seconds_per_step=step_dur
    )  # each (G*N_eval, T_full)

    # ── ADE — log vs sim ────────────────────────────────────────────────────
    eval_logged_full = logged_full.slice_time(0, None).gather_objects_by_id(eval_ids)
    elog_x = eval_logged_full.x.to(device)   # (N_eval, T_full)
    elog_y = eval_logged_full.y.to(device)
    elog_z = eval_logged_full.z.to(device)
    elog_val = eval_logged_full.valid.to(device)

    # expand logged to G
    elog_x_g = elog_x.unsqueeze(0).expand(G, -1, -1).reshape(G * N_eval, T_full)
    elog_y_g = elog_y.unsqueeze(0).expand(G, -1, -1).reshape(G * N_eval, T_full)
    elog_z_g = elog_z.unsqueeze(0).expand(G, -1, -1).reshape(G * N_eval, T_full)
    elog_val_g = elog_val.unsqueeze(0).expand(G, -1, -1).reshape(G * N_eval, T_full)

    disp_err = traj_feat.compute_displacement_error(
        eval_x, eval_y, eval_z, elog_x_g, elog_y_g, elog_z_g
    )  # (G*N_eval, T_full)
    valid_steps = elog_val_g.float().sum(dim=-1).clamp_min(1.0)
    ade = (torch.where(elog_val_g, disp_err, torch.zeros_like(disp_err)).sum(dim=-1) / valid_steps)
    # (G*N_eval,) → reshape later

    # ── distance to nearest object (all sims, scene_batch) ──────────────────
    dno = inter.compute_distance_to_nearest_object(
        center_x=gf_x, center_y=gf_y, center_z=gf_z,
        length=gf_len, width=gf_wid, height=gf_hei,
        heading=gf_h, valid=gf_val,
        evaluated_object_mask=flat_eval_mask,
        scene_batch=flat_scene_batch,
    )  # (G*N_eval, T_full)

    # ── time to collision (all sims, scene_batch) ────────────────────────────
    ttc = inter.compute_time_to_collision_with_object_in_front(
        center_x=gf_x, center_y=gf_y,
        length=gf_len, width=gf_wid, heading=gf_h,
        valid=gf_val,
        evaluated_object_mask=flat_eval_mask,
        seconds_per_step=step_dur,
        scene_batch=flat_scene_batch,
    )  # (G*N_eval, T_full)

    # ── distance to road edge (cached tensor, all sims at once) ─────────────
    d_road = map_feat.compute_distance_to_road_edge(
        center_x=gf_x, center_y=gf_y, center_z=gf_z,
        length=gf_len, width=gf_wid, height=gf_hei,
        heading=gf_h, valid=gf_val,
        evaluated_object_mask=flat_eval_mask,
        road_edge_polylines=cached.get("road_edges") or [],
        road_edge_polylines_tensor=cached.get("road_edge_polylines_tensor"),
        is_polyline_cyclic=cached.get("road_edge_is_cyclic"),
    )  # (G*N_eval, T_full)

    # ── traffic light violation (static map, all sims) ───────────────────────
    lane_polys = cached.get("lane_polys") or []
    traffic_signals = cached.get("traffic_signals") or [
        list(dms.lane_states) for dms in scenario.dynamic_map_states
    ]
    if lane_polys and traffic_signals:
        red_light = tl_feat.compute_red_light_violation(
            center_x=gf_x, center_y=gf_y, valid=gf_val,
            evaluated_object_mask=flat_eval_mask,
            lane_polylines=lane_polys,
            lane_ids=cached.get("lane_ids") or [],
            traffic_signals=traffic_signals,
            lane_tensor=cached.get("lane_tensor"),
            lane_ids_tensor=cached.get("lane_ids_tensor"),
            ts_lane_id=cached.get("ts_lane_id"),
            ts_state=cached.get("ts_state"),
            ts_stop_point=cached.get("ts_stop_point"),
        )  # (G*N_eval, T_full)
    else:
        red_light = torch.zeros(G * N_eval, T_full, dtype=torch.bool, device=device)

    # ── slice history off, keep only sim future steps ────────────────────────
    s = ct_idx + 1  # start of future
    eval_val_fut = eval_val[:, s:]          # (G*N_eval, T_sim)
    lin_spd  = lin_spd[:, s:]
    lin_acc  = lin_acc[:, s:]
    ang_spd  = ang_spd[:, s:]
    ang_acc  = ang_acc[:, s:]
    dno      = dno[:, s:]
    ttc      = ttc[:, s:]
    d_road   = d_road[:, s:]
    red_light = red_light[:, s:]

    is_collision  = (dno < inter.COLLISION_DISTANCE_THRESHOLD)
    is_offroad    = (d_road > map_feat.OFFROAD_DISTANCE_THRESHOLD)

    # ── reshape: (G*N_eval, T_sim) → (G, N_eval, T_sim) → unsqueeze G as
    # sample dim to get (G, N_eval, T_sim) for MetricFeaturesTorch ───────────
    def to_g(t, dtype=None):  # (G*N_eval, T_sim) → (G, N_eval, T_sim)
        r = t.reshape(G, N_eval, -1)
        return r.to(dtype) if dtype else r

    # object_type — static: (N_eval,) → (G, N_eval)
    eval_otypes = eval_logged_full.object_type.to(device).unsqueeze(0).expand(G, -1)
    ade_g = ade.reshape(G, N_eval)  # (G, N_eval)

    return MetricFeaturesTorch(
        object_id=eval_ids.to(device),
        object_type=eval_otypes,                      # (G, N_eval)
        valid=to_g(eval_val_fut, torch.bool),         # (G, N_eval, T_sim)
        average_displacement_error=ade_g,             # (G, N_eval)
        linear_speed=to_g(lin_spd),
        linear_acceleration=to_g(lin_acc),
        angular_speed=to_g(ang_spd),
        angular_acceleration=to_g(ang_acc),
        distance_to_nearest_object=to_g(dno),
        collision_per_step=to_g(is_collision, torch.bool),
        time_to_collision=to_g(ttc),
        distance_to_road_edge=to_g(d_road),
        offroad_per_step=to_g(is_offroad, torch.bool),
        traffic_light_violation_per_step=to_g(red_light, torch.bool),
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

    Args:
        prefix: metric key prefix (e.g. "val_closed")
        ego_only: if True, mask out non-SDC agents in the scenario
        device: torch device for feature computation ("cpu" or "cuda").
                Pass "cuda" to route all distance ops to GPU for ~3-5× speedup
                over CPU on typical scenarios with G=32 rollouts.
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
        self.add_state("metametric_sum", default=tensor(0.0), dist_reduce_fx="sum")
        for field in _LIKELIHOOD_FIELDS:
            self.add_state(field + "_sum", default=tensor(0.0), dist_reduce_fx="sum")

    def _feature_device(self) -> torch.device:
        if self._compute_device is not None:
            return self._compute_device
        # Default: use GPU if available, otherwise CPU
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

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

        dev = self._feature_device()

        for scenario_file, agent_ids_np, pred_traj_np, pred_z_np, pred_head_np in scenario_payloads:
            scenario = _get_or_parse_scenario(scenario_file, self.ego_only)

            # Log features: computed once, on device
            log_joint = scenario_to_joint_scene(scenario)
            log_feat = compute_metric_features(
                scenario, log_joint, use_log_validity=True
            )
            log_feat_dict = {k: v.to(dev) for k, v in log_feat.as_dict().items()}

            # Sim features: all G rollouts in one batched call
            sim_feat = _compute_sim_features_all_rollouts(
                scenario,
                agent_ids_np,
                pred_traj_np,  # (A, G, T_sim, 2)
                pred_z_np,
                pred_head_np,
                device=dev,
            )
            sim_feat_dict = {k: v for k, v in sim_feat.as_dict().items()}

            result: WosacMetametricTorchResult = compute_wosac_metametric_from_features_torch(
                self.sim_agents_config,
                log_feat_dict,
                sim_feat_dict,
            )

            self.metametric_sum.add_(tensor(result.metametric))
            self.scenario_counter.add_(1.0)
            for field in _LIKELIHOOD_FIELDS:
                getattr(self, field + "_sum").add_(tensor(getattr(result, field)))
