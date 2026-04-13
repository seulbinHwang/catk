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
from .metric_features_torch import ObjectTrajectoriesTorch, object_trajectories_from_scenario

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
) -> ObjectTrajectoriesTorch:
    device = pred.center_x.device
    cfg = submission_specs.get_submission_config(challenge_type)
    logged_full = object_trajectories_from_scenario(scenario).gather_objects_by_id(pred.object_id.cpu())
    logged_full = _trajectories_to_device(logged_full, device)
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
    simulated = _build_simulated_trajectories_from_prediction(
        scenario=scenario, pred=pred, challenge_type=challenge_type
    )

    logged_full = object_trajectories_from_scenario(scenario).gather_objects_by_id(simulated.object_id.cpu())
    logged_full = _trajectories_to_device(logged_full, device)

    evaluated_ids = torch.tensor(
        submission_specs.get_evaluation_sim_agent_ids(scenario, challenge_type),
        dtype=torch.int64,
        device=simulated.x.device,
    )
    evaluated = simulated.gather_objects_by_id(evaluated_ids)

    # reorder simulated so evaluated first (then non-evaluated)
    non_eval = [oid for oid in simulated.object_id.tolist() if oid not in set(evaluated_ids.tolist())]
    reordered_ids = torch.tensor(
        list(evaluated_ids.tolist()) + non_eval, dtype=torch.int64, device=simulated.x.device
    )
    simulated = simulated.gather_objects_by_id(reordered_ids)

    eval_logged = logged_full.gather_objects_by_id(evaluated_ids)

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

    road_edges = [list(mf.road_edge.polyline) for mf in scenario.map_features if mf.HasField("road_edge")]
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
    )

    lane_ids: List[int] = []
    lane_polys: List[List[map_pb2.MapPoint]] = []
    for mf in scenario.map_features:
        if mf.HasField("lane") and mf.lane.type == _LaneType.TYPE_SURFACE_STREET:
            lane_ids.append(int(mf.id))
            lane_polys.append(list(mf.lane.polyline))
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


__all__ = [
    "PredictedSimTrajectories",
    "compute_metric_features_from_predicted_sim_trajectories",
]

