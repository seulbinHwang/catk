from __future__ import annotations

"""Differentiable metric feature extraction entrypoints.

기존 `metric_features_torch.compute_metric_features`는 `JointScene` proto를 입력으로 받아
python list -> torch.tensor 변환 과정에서 그래프가 끊긴다.

학습(Flow/OE fine-tuning)에서 predicted trajectory 텐서로부터 gradient를 유지하려면,
proto를 거치지 않고 torch 텐서를 직접 받아 feature를 계산하는 경로가 필요하다.
"""

from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

from waymo_open_dataset.protos import map_pb2, scenario_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs

from .types import MetricFeaturesTorch
from .surrogate import SurrogateConfig
from . import trajectory_features_torch as traj
from . import interaction_features_torch as inter
from . import map_metric_features_torch as map_feat
from . import traffic_light_features_torch as tl_feat
from .metric_features_torch import (
    ObjectTrajectoriesTorch,
    object_trajectories_from_scenario,
    _cache_get_or_build,
)

_ChallengeType = submission_specs.ChallengeType
_LaneType = map_pb2.LaneCenter.LaneType


def _trajectories_to_device(t: ObjectTrajectoriesTorch, device: torch.device) -> ObjectTrajectoriesTorch:
    return ObjectTrajectoriesTorch(
        x=t.x.to(device),
        y=t.y.to(device),
        z=t.z.to(device),
        heading=t.heading.to(device),
        length=t.length.to(device),
        width=t.width.to(device),
        height=t.height.to(device),
        valid=t.valid.to(device),
        object_id=t.object_id.to(device),
        object_type=t.object_type.to(device),
    )


@dataclass(frozen=True)
class PredictedSimTrajectories:
    """Predicted future trajectories for sim agents.

    Shapes: (n_sim_agents, n_sim_steps)
    """

    object_id: Tensor  # (A,) int64
    center_x: Tensor  # (A, Tf) float
    center_y: Tensor  # (A, Tf) float
    center_z: Tensor  # (A, Tf) float
    heading: Tensor  # (A, Tf) float
    valid: Tensor  # (A, Tf) bool


def _build_simulated_trajectories_from_prediction(
    *,
    scenario: scenario_pb2.Scenario,
    pred: PredictedSimTrajectories,
    challenge_type: _ChallengeType = _ChallengeType.SIM_AGENTS,
    logged_full_cpu: ObjectTrajectoriesTorch | None = None,
) -> ObjectTrajectoriesTorch:
    device = pred.center_x.device
    cfg = submission_specs.get_submission_config(challenge_type)
    _lf_cpu = logged_full_cpu if logged_full_cpu is not None else object_trajectories_from_scenario(scenario)
    logged_full = _trajectories_to_device(_lf_cpu.gather_objects_by_id(pred.object_id.cpu()), device)
    logged_hist = logged_full.slice_time(0, cfg.current_time_index + 1)

    # sizes: repeat from last history step
    last_len = logged_hist.length[:, -1:]
    last_wid = logged_hist.width[:, -1:]
    last_hei = logged_hist.height[:, -1:]
    sim_length = last_len.repeat(1, pred.center_x.shape[-1])
    sim_width = last_wid.repeat(1, pred.center_x.shape[-1])
    sim_height = last_hei.repeat(1, pred.center_x.shape[-1])

    return ObjectTrajectoriesTorch(
        x=torch.cat([logged_hist.x, pred.center_x], dim=-1),
        y=torch.cat([logged_hist.y, pred.center_y], dim=-1),
        z=torch.cat([logged_hist.z, pred.center_z], dim=-1),
        heading=torch.cat([logged_hist.heading, pred.heading], dim=-1),
        length=torch.cat([logged_hist.length, sim_length], dim=-1),
        width=torch.cat([logged_hist.width, sim_width], dim=-1),
        height=torch.cat([logged_hist.height, sim_height], dim=-1),
        valid=torch.cat([logged_hist.valid, pred.valid], dim=-1),
        object_id=pred.object_id,
        object_type=logged_hist.object_type,
    )


def compute_metric_features_from_predicted_sim_trajectories(
    *,
    scenario: scenario_pb2.Scenario,
    pred: PredictedSimTrajectories,
    surrogate: SurrogateConfig | None = None,
    challenge_type: _ChallengeType = _ChallengeType.SIM_AGENTS,
) -> MetricFeaturesTorch:
    """Differentiable feature extraction for predicted sim trajectories."""
    if challenge_type != _ChallengeType.SIM_AGENTS:
        raise NotImplementedError("Only SIM_AGENTS supported")

    device = pred.center_x.device

    # Use per-process scenario cache to avoid G-fold redundant proto→tensor conversion.
    _sc = _cache_get_or_build(scenario)
    logged_full_cpu: ObjectTrajectoriesTorch = (
        _sc["logged_full_cpu"]
        if "logged_full_cpu" in _sc
        else object_trajectories_from_scenario(scenario)
    )

    simulated = _build_simulated_trajectories_from_prediction(
        scenario=scenario, pred=pred, challenge_type=challenge_type,
        logged_full_cpu=logged_full_cpu,
    )

    # Reorder simulated so evaluated agents come first.
    if "eval_ids_list" in _sc:
        _eval_ids_list = _sc["eval_ids_list"]
    else:
        _eval_ids_list = list(submission_specs.get_evaluation_sim_agent_ids(scenario, challenge_type))
    evaluated_ids = torch.tensor(_eval_ids_list, dtype=torch.int64, device=device)

    evaluated = simulated.gather_objects_by_id(evaluated_ids)
    non_eval = [oid for oid in simulated.object_id.tolist() if oid not in set(_eval_ids_list)]
    reordered_ids = torch.tensor(
        _eval_ids_list + non_eval, dtype=torch.int64, device=device
    )
    simulated = simulated.gather_objects_by_id(reordered_ids)

    # Gather eval-agent logged trajectories using cached logged_full (no re-parse).
    eval_logged = _trajectories_to_device(
        logged_full_cpu.gather_objects_by_id(evaluated_ids.cpu()), device
    )
    # 시나리오 GT는 전체 시간(예: 91 step), closed-loop pred는 짧을 수 있음 → 같은 글로벌 인덱스 0..T-1만 비교.
    _t_sim = int(simulated.x.shape[1])
    _t_log = int(eval_logged.x.shape[1])
    if _t_log != _t_sim:
        if _t_sim > _t_log:
            raise ValueError(
                f"Simulated horizon ({_t_sim}) exceeds scenario log length ({_t_log})."
            )
        eval_logged = eval_logged.slice_time(0, _t_sim)

    cfg = submission_specs.get_submission_config(challenge_type)
    ct_idx = cfg.current_time_index

    # Kinematics on history+future then slice (match TF metric_features behavior)
    linear_speed, linear_accel, angular_speed, angular_accel = traj.compute_kinematic_features(
        evaluated.x, evaluated.y, evaluated.z, evaluated.heading, seconds_per_step=cfg.step_duration_seconds
    )

    eval_object_mask = torch.any(evaluated_ids[:, None].eq(simulated.object_id[None, :]), dim=0)

    dno = inter.compute_distance_to_nearest_object(
        center_x=simulated.x,
        center_y=simulated.y,
        center_z=simulated.z,
        length=simulated.length,
        width=simulated.width,
        height=simulated.height,
        heading=simulated.heading,
        valid=simulated.valid,
        evaluated_object_mask=eval_object_mask,
    )
    ttc = inter.compute_time_to_collision_with_object_in_front(
        center_x=simulated.x,
        center_y=simulated.y,
        length=simulated.length,
        width=simulated.width,
        heading=simulated.heading,
        valid=simulated.valid,
        evaluated_object_mask=eval_object_mask,
        seconds_per_step=cfg.step_duration_seconds,
    )

    road_edges = _sc.get("road_edges") or [list(mf.road_edge.polyline) for mf in scenario.map_features if mf.HasField("road_edge")]
    d_road = map_feat.compute_distance_to_road_edge(
        center_x=simulated.x,
        center_y=simulated.y,
        center_z=simulated.z,
        length=simulated.length,
        width=simulated.width,
        height=simulated.height,
        heading=simulated.heading,
        valid=simulated.valid,
        evaluated_object_mask=eval_object_mask,
        road_edge_polylines=road_edges,
        road_edge_polylines_tensor=_sc.get("road_edge_polylines_tensor"),
        is_polyline_cyclic=_sc.get("road_edge_is_cyclic"),
    )

    lane_ids: List[int] = _sc.get("lane_ids") or []
    lane_polys: List[List[map_pb2.MapPoint]] = _sc.get("lane_polys") or []
    traffic_signals = _sc.get("traffic_signals") or []
    if not lane_ids:
        for mf in scenario.map_features:
            if mf.HasField("lane") and mf.lane.type == _LaneType.TYPE_SURFACE_STREET:
                lane_ids.append(int(mf.id))
                lane_polys.append(list(mf.lane.polyline))
    if not traffic_signals:
        traffic_signals = [list(dms.lane_states) for dms in scenario.dynamic_map_states]

    if lane_polys and traffic_signals:
        if surrogate is None:
            red_light = tl_feat.compute_red_light_violation(
                center_x=simulated.x,
                center_y=simulated.y,
                valid=simulated.valid,
                evaluated_object_mask=eval_object_mask,
                lane_polylines=lane_polys,
                lane_ids=lane_ids,
                traffic_signals=traffic_signals,
            )
        else:
            red_light = tl_feat.compute_red_light_violation_soft(
                center_x=simulated.x,
                center_y=simulated.y,
                valid=simulated.valid,
                evaluated_object_mask=eval_object_mask,
                lane_polylines=lane_polys,
                lane_ids=lane_ids,
                traffic_signals=traffic_signals,
                crossing_temperature=surrogate.red_light_crossing_temperature,
                lane_tensor=_sc.get("lane_tensor"),
                lane_ids_tensor=_sc.get("lane_ids_tensor"),
                ts_lane_id=_sc.get("ts_lane_id"),
                ts_state=_sc.get("ts_state"),
                ts_stop_point=_sc.get("ts_stop_point"),
            )
    else:
        red_light = torch.zeros(
            (len(evaluated_ids), simulated.valid.shape[1]),
            dtype=torch.float32 if surrogate is not None else torch.bool,
            device=device,
        )

    # Slice time for SIM_AGENTS (remove history)
    validity_mask = evaluated.valid[:, ct_idx + 1 :]

    displacement_error = traj.compute_displacement_error(
        evaluated.x, evaluated.y, evaluated.z, eval_logged.x, eval_logged.y, eval_logged.z
    )
    object_valid_steps = torch.clamp(
        eval_logged.valid.to(torch.float32).sum(dim=1), min=1.0
    )
    ade = (
        torch.where(eval_logged.valid, displacement_error, torch.zeros_like(displacement_error)).sum(dim=1)
        / object_valid_steps
    )

    linear_speed = linear_speed[:, ct_idx + 1 :]
    linear_accel = linear_accel[:, ct_idx + 1 :]
    angular_speed = angular_speed[:, ct_idx + 1 :]
    angular_accel = angular_accel[:, ct_idx + 1 :]
    dno = dno[:, ct_idx + 1 :]
    ttc = ttc[:, ct_idx + 1 :]
    d_road = d_road[:, ct_idx + 1 :]
    red_light = red_light[:, ct_idx + 1 :]

    if surrogate is None:
        is_colliding = dno < inter.COLLISION_DISTANCE_THRESHOLD
        is_offroad = d_road > map_feat.OFFROAD_DISTANCE_THRESHOLD
        tl_out = red_light
    else:
        k_col = 1.0 / float(surrogate.collision_temperature)
        k_off = 1.0 / float(surrogate.offroad_temperature)
        is_colliding = torch.sigmoid(-k_col * dno)
        is_offroad = torch.sigmoid(k_off * d_road)
        tl_out = red_light

    return MetricFeaturesTorch(
        object_id=evaluated.object_id,
        object_type=evaluated.object_type.unsqueeze(0),
        valid=validity_mask.unsqueeze(0),
        average_displacement_error=ade.unsqueeze(0),
        linear_speed=linear_speed.unsqueeze(0),
        linear_acceleration=linear_accel.unsqueeze(0),
        angular_speed=angular_speed.unsqueeze(0),
        angular_acceleration=angular_accel.unsqueeze(0),
        distance_to_nearest_object=dno.unsqueeze(0),
        collision_per_step=is_colliding.unsqueeze(0),
        time_to_collision=ttc.unsqueeze(0),
        distance_to_road_edge=d_road.unsqueeze(0),
        offroad_per_step=is_offroad.unsqueeze(0),
        traffic_light_violation_per_step=tl_out.unsqueeze(0),
    )


def compute_metric_features_batched_scenes(
    *,
    scenarios: list,
    preds: list,
    surrogate: SurrogateConfig | None = None,
    challenge_type: _ChallengeType = _ChallengeType.SIM_AGENTS,
) -> list:
    """Differentiable feature extraction for multiple scenes in one batched DNO/TTC call.

    DNO and TTC are computed with all agents from all scenes concatenated into a single flat
    tensor.  Cross-scene pairs are masked via ``scene_batch`` so agents from different scenes
    never interact.  Road edge, traffic lights, and kinematics are still computed per scene
    (different polylines per scenario).

    Args:
        scenarios: list of ``Scenario`` protos, length n_scenes
        preds:     list of ``PredictedSimTrajectories``, length n_scenes
        surrogate: optional surrogate sigmoid config; if None produces hard bool features
        challenge_type: must be SIM_AGENTS

    Returns:
        list of ``MetricFeaturesTorch``, length n_scenes, in the same order as inputs
    """
    if challenge_type != _ChallengeType.SIM_AGENTS:
        raise NotImplementedError("Only SIM_AGENTS supported")

    n_scenes = len(scenarios)
    assert n_scenes == len(preds), "scenarios and preds must have the same length"
    if n_scenes == 0:
        return []

    device = preds[0].center_x.device
    cfg = submission_specs.get_submission_config(challenge_type)
    ct_idx = cfg.current_time_index

    # ── Per-scene setup ────────────────────────────────────────────────────
    simulated_list: list = []
    evaluated_list: list = []
    eval_logged_list: list = []
    eval_mask_list: list = []  # bool (A_i,) in simulated order

    for i, (scenario, pred) in enumerate(zip(scenarios, preds)):
        _sc = _cache_get_or_build(scenario)
        logged_full_cpu: ObjectTrajectoriesTorch = (
            _sc["logged_full_cpu"] if "logged_full_cpu" in _sc
            else object_trajectories_from_scenario(scenario)
        )

        simulated = _build_simulated_trajectories_from_prediction(
            scenario=scenario, pred=pred, challenge_type=challenge_type,
            logged_full_cpu=logged_full_cpu,
        )

        if "eval_ids_list" in _sc:
            _eval_ids_list = _sc["eval_ids_list"]
        else:
            _eval_ids_list = list(submission_specs.get_evaluation_sim_agent_ids(scenario, challenge_type))

        evaluated_ids = torch.tensor(_eval_ids_list, dtype=torch.int64, device=device)

        # simulated.object_id comes from pred.object_id which is CPU — move to device for comparison
        sim_oid_cpu = simulated.object_id  # (A_i,) CPU
        sim_oid_dev = sim_oid_cpu.to(device)

        # eval mask in simulated order (used for DNO/TTC and road/TL features) — keep on device
        eval_mask = torch.any(evaluated_ids[:, None].eq(sim_oid_dev[None, :]), dim=0)  # (A_i,) device

        # local positions on CPU for indexing the CPU object_id tensor
        local_eval_pos_cpu = eval_mask.cpu().nonzero(as_tuple=True)[0]  # (E_i,) CPU
        eval_ids_in_order_cpu = sim_oid_cpu[local_eval_pos_cpu]  # (E_i,) CPU

        # Build evaluated in simulated order (must match DNO/TTC output order)
        evaluated = simulated.gather_objects_by_id(eval_ids_in_order_cpu)

        eval_logged = _trajectories_to_device(
            logged_full_cpu.gather_objects_by_id(eval_ids_in_order_cpu), device
        )

        # Time alignment (scenario GT may be longer than the prediction horizon)
        _t_sim = int(simulated.x.shape[1])
        _t_log = int(eval_logged.x.shape[1])
        if _t_log != _t_sim:
            if _t_sim > _t_log:
                raise ValueError(
                    f"Scene {i}: simulated horizon ({_t_sim}) exceeds scenario log length ({_t_log})."
                )
            eval_logged = eval_logged.slice_time(0, _t_sim)

        simulated_list.append(simulated)
        evaluated_list.append(evaluated)
        eval_logged_list.append(eval_logged)
        eval_mask_list.append(eval_mask)

    # ── Build flat tensors for batched DNO + TTC ───────────────────────────
    flat_x     = torch.cat([s.x      for s in simulated_list], dim=0)
    flat_y     = torch.cat([s.y      for s in simulated_list], dim=0)
    flat_z     = torch.cat([s.z      for s in simulated_list], dim=0)
    flat_len   = torch.cat([s.length for s in simulated_list], dim=0)
    flat_wid   = torch.cat([s.width  for s in simulated_list], dim=0)
    flat_hei   = torch.cat([s.height for s in simulated_list], dim=0)
    flat_head  = torch.cat([s.heading for s in simulated_list], dim=0)
    flat_valid = torch.cat([s.valid  for s in simulated_list], dim=0)
    eval_mask_flat = torch.cat(eval_mask_list, dim=0)  # (N_total,)

    scene_batch_flat = torch.cat(
        [torch.full((simulated_list[i].x.shape[0],), i, dtype=torch.long, device=device)
         for i in range(n_scenes)],
        dim=0,
    )  # (N_total,)

    # ── Batched DNO ────────────────────────────────────────────────────────
    dno_flat = inter.compute_distance_to_nearest_object(
        center_x=flat_x, center_y=flat_y, center_z=flat_z,
        length=flat_len, width=flat_wid, height=flat_hei,
        heading=flat_head, valid=flat_valid,
        evaluated_object_mask=eval_mask_flat,
        scene_batch=scene_batch_flat,
    )  # (E_total, T_full)

    # ── Batched TTC ────────────────────────────────────────────────────────
    ttc_flat = inter.compute_time_to_collision_with_object_in_front(
        center_x=flat_x, center_y=flat_y,
        length=flat_len, width=flat_wid,
        heading=flat_head, valid=flat_valid,
        evaluated_object_mask=eval_mask_flat,
        seconds_per_step=cfg.step_duration_seconds,
        scene_batch=scene_batch_flat,
    )  # (E_total, T_full)

    # Cumulative eval-agent offsets for slicing dno_flat / ttc_flat per scene
    eval_counts = [int(m.sum().item()) for m in eval_mask_list]
    eval_offsets = [0]
    for ec in eval_counts:
        eval_offsets.append(eval_offsets[-1] + ec)

    # ── Per-scene assembly ─────────────────────────────────────────────────
    results = []
    for i in range(n_scenes):
        scenario = scenarios[i]
        simulated = simulated_list[i]
        evaluated = evaluated_list[i]
        eval_logged = eval_logged_list[i]
        eval_mask = eval_mask_list[i]
        _sc = _cache_get_or_build(scenario)

        dno_i = dno_flat[eval_offsets[i]:eval_offsets[i + 1]]  # (E_i, T_full)
        ttc_i = ttc_flat[eval_offsets[i]:eval_offsets[i + 1]]  # (E_i, T_full)

        # eval_object_mask in simulated order (for road edge and traffic lights)
        # evaluated.object_id comes from gather_objects_by_id(cpu tensor) — ensure device match
        eval_ids_i_dev = evaluated.object_id.to(device)
        eval_object_mask_i = torch.any(eval_ids_i_dev[:, None].eq(simulated.object_id.to(device)[None, :]), dim=0)

        # Kinematics on eval agents
        linear_speed, linear_accel, angular_speed, angular_accel = traj.compute_kinematic_features(
            evaluated.x, evaluated.y, evaluated.z, evaluated.heading,
            seconds_per_step=cfg.step_duration_seconds,
        )

        # Road edge (per-scene polylines)
        road_edges = _sc.get("road_edges") or [
            list(mf.road_edge.polyline)
            for mf in scenario.map_features
            if mf.HasField("road_edge")
        ]
        d_road = map_feat.compute_distance_to_road_edge(
            center_x=simulated.x, center_y=simulated.y, center_z=simulated.z,
            length=simulated.length, width=simulated.width, height=simulated.height,
            heading=simulated.heading, valid=simulated.valid,
            evaluated_object_mask=eval_object_mask_i,
            road_edge_polylines=road_edges,
            road_edge_polylines_tensor=_sc.get("road_edge_polylines_tensor"),
            is_polyline_cyclic=_sc.get("road_edge_is_cyclic"),
        )  # (E_i, T_full)

        # Traffic lights (per-scene lane/signal data)
        lane_ids: List[int] = _sc.get("lane_ids") or []
        lane_polys: List[List[map_pb2.MapPoint]] = _sc.get("lane_polys") or []
        traffic_signals = _sc.get("traffic_signals") or []
        if not lane_ids:
            for mf in scenario.map_features:
                if mf.HasField("lane") and mf.lane.type == _LaneType.TYPE_SURFACE_STREET:
                    lane_ids.append(int(mf.id))
                    lane_polys.append(list(mf.lane.polyline))
        if not traffic_signals:
            traffic_signals = [list(dms.lane_states) for dms in scenario.dynamic_map_states]

        if lane_polys and traffic_signals:
            if surrogate is None:
                red_light = tl_feat.compute_red_light_violation(
                    center_x=simulated.x, center_y=simulated.y,
                    valid=simulated.valid,
                    evaluated_object_mask=eval_object_mask_i,
                    lane_polylines=lane_polys, lane_ids=lane_ids,
                    traffic_signals=traffic_signals,
                )
            else:
                red_light = tl_feat.compute_red_light_violation_soft(
                    center_x=simulated.x, center_y=simulated.y,
                    valid=simulated.valid,
                    evaluated_object_mask=eval_object_mask_i,
                    lane_polylines=lane_polys, lane_ids=lane_ids,
                    traffic_signals=traffic_signals,
                    crossing_temperature=surrogate.red_light_crossing_temperature,
                    lane_tensor=_sc.get("lane_tensor"),
                    lane_ids_tensor=_sc.get("lane_ids_tensor"),
                    ts_lane_id=_sc.get("ts_lane_id"),
                    ts_state=_sc.get("ts_state"),
                    ts_stop_point=_sc.get("ts_stop_point"),
                )
        else:
            red_light = torch.zeros(
                (int(eval_mask.sum().item()), simulated.valid.shape[1]),
                dtype=torch.float32 if surrogate is not None else torch.bool,
                device=device,
            )

        # ADE against logged trajectories
        displacement_error = traj.compute_displacement_error(
            evaluated.x, evaluated.y, evaluated.z,
            eval_logged.x, eval_logged.y, eval_logged.z,
        )
        object_valid_steps = torch.clamp(eval_logged.valid.to(torch.float32).sum(dim=1), min=1.0)
        ade = (
            torch.where(eval_logged.valid, displacement_error, torch.zeros_like(displacement_error)).sum(dim=1)
            / object_valid_steps
        )

        # Slice to future horizon (remove history)
        validity_mask = evaluated.valid[:, ct_idx + 1:]
        linear_speed   = linear_speed[:, ct_idx + 1:]
        linear_accel   = linear_accel[:, ct_idx + 1:]
        angular_speed  = angular_speed[:, ct_idx + 1:]
        angular_accel  = angular_accel[:, ct_idx + 1:]
        dno_i   = dno_i[:, ct_idx + 1:]
        ttc_i   = ttc_i[:, ct_idx + 1:]
        d_road  = d_road[:, ct_idx + 1:]
        red_light = red_light[:, ct_idx + 1:]

        if surrogate is None:
            is_colliding = dno_i < inter.COLLISION_DISTANCE_THRESHOLD
            is_offroad   = d_road > map_feat.OFFROAD_DISTANCE_THRESHOLD
            tl_out = red_light
        else:
            k_col = 1.0 / float(surrogate.collision_temperature)
            k_off = 1.0 / float(surrogate.offroad_temperature)
            is_colliding = torch.sigmoid(-k_col * dno_i)
            is_offroad   = torch.sigmoid(k_off * d_road)
            tl_out = red_light

        results.append(MetricFeaturesTorch(
            object_id=evaluated.object_id,
            object_type=evaluated.object_type.unsqueeze(0),
            valid=validity_mask.unsqueeze(0),
            average_displacement_error=ade.unsqueeze(0),
            linear_speed=linear_speed.unsqueeze(0),
            linear_acceleration=linear_accel.unsqueeze(0),
            angular_speed=angular_speed.unsqueeze(0),
            angular_acceleration=angular_accel.unsqueeze(0),
            distance_to_nearest_object=dno_i.unsqueeze(0),
            collision_per_step=is_colliding.unsqueeze(0),
            time_to_collision=ttc_i.unsqueeze(0),
            distance_to_road_edge=d_road.unsqueeze(0),
            offroad_per_step=is_offroad.unsqueeze(0),
            traffic_light_violation_per_step=tl_out.unsqueeze(0),
        ))

    return results


__all__ = [
    "PredictedSimTrajectories",
    "compute_metric_features_from_predicted_sim_trajectories",
    "compute_metric_features_batched_scenes",
]

