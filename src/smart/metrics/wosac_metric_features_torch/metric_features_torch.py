from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import Tensor

from waymo_open_dataset.protos import map_pb2, scenario_pb2, sim_agents_submission_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs

from .types import MetricFeaturesTorch
from . import trajectory_features_torch as traj
from . import interaction_features_torch as inter
from . import map_metric_features_torch as map_feat
from . import traffic_light_features_torch as tl_feat
from .surrogate import SurrogateConfig

_ChallengeType = submission_specs.ChallengeType
_LaneType = map_pb2.LaneCenter.LaneType

_STATIC_SCENARIO_CACHE: dict[str, dict] = {}
_STATIC_SCENARIO_CACHE_MAX = 2048

_COMPILED_FNS: dict[str, object] = {}


def _maybe_compile(name: str, fn):
    if name in _COMPILED_FNS:
        return _COMPILED_FNS[name]
    # torch.compile is optional and can regress depending on shapes/calls.
    # Keep it best-effort and cached (never compile per-call).
    try:
        compiled = torch.compile(fn, dynamic=False)  # type: ignore[attr-defined]
    except Exception:
        compiled = fn
    _COMPILED_FNS[name] = compiled
    return compiled


def _get_scenario_id(scenario: scenario_pb2.Scenario) -> str:
    sid = getattr(scenario, "scenario_id", "")
    return str(sid) if sid is not None else ""


def _cache_get_or_build(scenario: scenario_pb2.Scenario) -> dict:
    sid = _get_scenario_id(scenario)
    if not sid:
        return {}
    hit = _STATIC_SCENARIO_CACHE.get(sid)
    if hit is not None:
        return hit

    road_edges = [list(mf.road_edge.polyline) for mf in scenario.map_features if mf.HasField("road_edge")]
    road_edge_polylines_tensor, _ = map_feat.tensorize_polylines(road_edges) if road_edges else (None, None)
    road_edge_is_cyclic = map_feat.check_polyline_cycles(road_edges) if road_edges else None

    lane_ids: List[int] = []
    lane_polys: List[List[map_pb2.MapPoint]] = []
    for mf in scenario.map_features:
        if mf.HasField("lane") and mf.lane.type == _LaneType.TYPE_SURFACE_STREET:
            lane_ids.append(int(mf.id))
            lane_polys.append(list(mf.lane.polyline))
    lane_tensor, lane_ids_tensor = map_feat.tensorize_polylines(lane_polys, lane_ids) if lane_polys else (None, None)

    traffic_signals = [list(dms.lane_states) for dms in scenario.dynamic_map_states]
    if traffic_signals:
        ts_lane_id, ts_state, ts_stop_point = tl_feat._tensorize_traffic_signals(traffic_signals)
    else:
        ts_lane_id = ts_state = ts_stop_point = None

    # Pre-build logged trajectory tensor (proto → torch) and evaluated agent IDs.
    # These are reused across all G rollouts in BPTT training, eliminating G-fold
    # redundant proto parsing.
    logged_full_cpu = object_trajectories_from_scenario(scenario)
    eval_ids_list = list(submission_specs.get_evaluation_sim_agent_ids(
        scenario, _ChallengeType.SIM_AGENTS
    ))

    built = {
        "road_edges": road_edges,
        "road_edge_polylines_tensor": road_edge_polylines_tensor,
        "road_edge_is_cyclic": road_edge_is_cyclic,
        "lane_ids": lane_ids,
        "lane_polys": lane_polys,
        "lane_tensor": lane_tensor,
        "lane_ids_tensor": lane_ids_tensor,
        "traffic_signals": traffic_signals,
        "ts_lane_id": ts_lane_id,
        "ts_state": ts_state,
        "ts_stop_point": ts_stop_point,
        "logged_full_cpu": logged_full_cpu,
        "eval_ids_list": eval_ids_list,
    }
    _STATIC_SCENARIO_CACHE[sid] = built
    if len(_STATIC_SCENARIO_CACHE) > _STATIC_SCENARIO_CACHE_MAX:
        # drop oldest inserted item (py3.7+ dict keeps insertion order)
        _STATIC_SCENARIO_CACHE.pop(next(iter(_STATIC_SCENARIO_CACHE)))
    return built


@dataclass(frozen=True)
class ObjectTrajectoriesTorch:
    x: Tensor
    y: Tensor
    z: Tensor
    heading: Tensor
    length: Tensor
    width: Tensor
    height: Tensor
    valid: Tensor
    object_id: Tensor
    object_type: Tensor  # int

    def gather_objects_by_id(self, ids: Tensor) -> "ObjectTrajectoriesTorch":
        # ids: (n_objects,) int
        id_to_idx = {int(oid): i for i, oid in enumerate(self.object_id.tolist())}
        idx = torch.tensor(
            [id_to_idx[int(oid)] for oid in ids.tolist()],
            dtype=torch.long,
            device=self.x.device,
        )
        return ObjectTrajectoriesTorch(
            x=self.x.index_select(0, idx),
            y=self.y.index_select(0, idx),
            z=self.z.index_select(0, idx),
            heading=self.heading.index_select(0, idx),
            length=self.length.index_select(0, idx),
            width=self.width.index_select(0, idx),
            height=self.height.index_select(0, idx),
            valid=self.valid.index_select(0, idx),
            object_id=ids,
            object_type=self.object_type.index_select(0, idx),
        )

    def slice_time(self, start_index: int = 0, end_index: int | None = None) -> "ObjectTrajectoriesTorch":
        sl = slice(start_index, end_index)
        return ObjectTrajectoriesTorch(
            x=self.x[:, sl],
            y=self.y[:, sl],
            z=self.z[:, sl],
            heading=self.heading[:, sl],
            length=self.length[:, sl],
            width=self.width[:, sl],
            height=self.height[:, sl],
            valid=self.valid[:, sl],
            object_id=self.object_id,
            object_type=self.object_type,
        )


def object_trajectories_from_scenario(scenario: scenario_pb2.Scenario) -> ObjectTrajectoriesTorch:
    """Torch equivalent of `trajectory_utils.ObjectTrajectories.from_scenario` (SIM_AGENTS fields only)."""
    object_ids: List[int] = []
    object_types: List[int] = []
    xs: List[List[float]] = []
    ys: List[List[float]] = []
    zs: List[List[float]] = []
    headings: List[List[float]] = []
    lengths: List[List[float]] = []
    widths: List[List[float]] = []
    heights: List[List[float]] = []
    valids: List[List[bool]] = []

    for tr in scenario.tracks:
        object_ids.append(int(tr.id))
        object_types.append(int(tr.object_type))
        xs.append([st.center_x for st in tr.states])
        ys.append([st.center_y for st in tr.states])
        zs.append([st.center_z for st in tr.states])
        headings.append([st.heading for st in tr.states])
        lengths.append([st.length for st in tr.states])
        widths.append([st.width for st in tr.states])
        heights.append([st.height for st in tr.states])
        # WOMD: state.valid indicates availability
        valids.append([bool(st.valid) for st in tr.states])

    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    z = torch.tensor(zs, dtype=torch.float32)
    heading = torch.tensor(headings, dtype=torch.float32)
    length = torch.tensor(lengths, dtype=torch.float32)
    width = torch.tensor(widths, dtype=torch.float32)
    height = torch.tensor(heights, dtype=torch.float32)
    valid = torch.tensor(valids, dtype=torch.bool)
    object_id = torch.tensor(object_ids, dtype=torch.int64)
    object_type = torch.tensor(object_types, dtype=torch.int64)
    return ObjectTrajectoriesTorch(
        x=x,
        y=y,
        z=z,
        heading=heading,
        length=length,
        width=width,
        height=height,
        valid=valid,
        object_id=object_id,
        object_type=object_type,
    )


def scenario_to_joint_scene(
    scenario: scenario_pb2.Scenario,
    challenge_type: _ChallengeType = _ChallengeType.SIM_AGENTS,
) -> sim_agents_submission_pb2.JointScene:
    """Lightweight torch-side equivalent of `converters.scenario_to_joint_scene` (SIM_AGENTS only)."""
    if challenge_type != _ChallengeType.SIM_AGENTS:
        raise NotImplementedError("Only SIM_AGENTS supported in torch port")
    cfg = submission_specs.get_submission_config(challenge_type)
    sim_ids = submission_specs.get_sim_agent_ids(scenario, challenge_type)
    tracks = {t.id: t for t in scenario.tracks}
    t0 = cfg.current_time_index + 1
    trajs = []
    for oid in sim_ids:
        tr = tracks[int(oid)]
        trajs.append(
            sim_agents_submission_pb2.SimulatedTrajectory(
                object_id=int(oid),
                center_x=[tr.states[ti].center_x for ti in range(t0, t0 + cfg.n_simulation_steps)],
                center_y=[tr.states[ti].center_y for ti in range(t0, t0 + cfg.n_simulation_steps)],
                center_z=[tr.states[ti].center_z for ti in range(t0, t0 + cfg.n_simulation_steps)],
                heading=[tr.states[ti].heading for ti in range(t0, t0 + cfg.n_simulation_steps)],
            )
        )
    return sim_agents_submission_pb2.JointScene(simulated_trajectories=trajs)


def joint_scene_to_trajectories(
    joint_scene: sim_agents_submission_pb2.JointScene,
    scenario: scenario_pb2.Scenario,
    *,
    use_log_validity: bool = False,
) -> ObjectTrajectoriesTorch:
    """Torch equivalent of `converters.joint_scene_to_trajectories` (SIM_AGENTS only)."""
    cached = _cache_get_or_build(scenario)
    logged_full = cached.get("logged_full_cpu") or object_trajectories_from_scenario(scenario)
    cfg = submission_specs.get_submission_config(_ChallengeType.SIM_AGENTS)
    logged_hist = logged_full.slice_time(0, cfg.current_time_index + 1)

    sim_ids: List[int] = []
    sim_x: List[List[float]] = []
    sim_y: List[List[float]] = []
    sim_z: List[List[float]] = []
    sim_heading: List[List[float]] = []
    for st in joint_scene.simulated_trajectories:
        sim_ids.append(int(st.object_id))
        sim_x.append(list(st.center_x))
        sim_y.append(list(st.center_y))
        sim_z.append(list(st.center_z))
        sim_heading.append(list(st.heading))

    sim_ids_t = torch.tensor(sim_ids, dtype=torch.int64)
    sim_x_t = torch.tensor(sim_x, dtype=torch.float32)
    sim_y_t = torch.tensor(sim_y, dtype=torch.float32)
    sim_z_t = torch.tensor(sim_z, dtype=torch.float32)
    sim_heading_t = torch.tensor(sim_heading, dtype=torch.float32)

    logged_hist = logged_hist.gather_objects_by_id(sim_ids_t)

    if use_log_validity:
        logged_full_aligned = logged_full.gather_objects_by_id(sim_ids_t)
        logged_future = logged_full_aligned.slice_time(cfg.current_time_index + 1, None)
        sim_valid = logged_future.valid
    else:
        sim_valid = torch.ones_like(sim_x_t, dtype=torch.bool)

    # Repeat static dims from last history step
    last_len = logged_hist.length[:, -1:]
    last_wid = logged_hist.width[:, -1:]
    last_hei = logged_hist.height[:, -1:]
    sim_length = last_len.repeat(1, sim_x_t.shape[-1])
    sim_width = last_wid.repeat(1, sim_x_t.shape[-1])
    sim_height = last_hei.repeat(1, sim_x_t.shape[-1])

    return ObjectTrajectoriesTorch(
        x=torch.cat([logged_hist.x, sim_x_t], dim=-1),
        y=torch.cat([logged_hist.y, sim_y_t], dim=-1),
        z=torch.cat([logged_hist.z, sim_z_t], dim=-1),
        heading=torch.cat([logged_hist.heading, sim_heading_t], dim=-1),
        length=torch.cat([logged_hist.length, sim_length], dim=-1),
        width=torch.cat([logged_hist.width, sim_width], dim=-1),
        height=torch.cat([logged_hist.height, sim_height], dim=-1),
        valid=torch.cat([logged_hist.valid, sim_valid], dim=-1),
        object_id=sim_ids_t,
        object_type=logged_hist.object_type,
    )


def compute_metric_features(
    scenario: scenario_pb2.Scenario,
    joint_scene: sim_agents_submission_pb2.JointScene,
    *,
    challenge_type: _ChallengeType = _ChallengeType.SIM_AGENTS,
    use_log_validity: bool = False,
    surrogate: SurrogateConfig | None = None,
) -> MetricFeaturesTorch:
    if challenge_type != _ChallengeType.SIM_AGENTS:
        raise NotImplementedError("Only SIM_AGENTS supported in torch port")

    simulated = joint_scene_to_trajectories(joint_scene, scenario, use_log_validity=use_log_validity)

    cached = _cache_get_or_build(scenario)
    logged_full = (cached.get("logged_full_cpu") or object_trajectories_from_scenario(scenario))
    logged_full = logged_full.gather_objects_by_id(simulated.object_id)

    evaluated_ids = torch.tensor(
        submission_specs.get_evaluation_sim_agent_ids(scenario, challenge_type),
        dtype=torch.int64,
    )
    evaluated = simulated.gather_objects_by_id(evaluated_ids)

    # reorder simulated so evaluated first (then non-evaluated)
    non_eval = [oid for oid in simulated.object_id.tolist() if oid not in set(evaluated_ids.tolist())]
    reordered_ids = torch.tensor(list(evaluated_ids.tolist()) + non_eval, dtype=torch.int64)
    simulated = simulated.gather_objects_by_id(reordered_ids)

    eval_logged = logged_full.gather_objects_by_id(evaluated_ids)

    validity_mask = eval_logged.valid if use_log_validity else evaluated.valid
    cfg = submission_specs.get_submission_config(challenge_type)
    ct_idx = cfg.current_time_index

    # Match TF: compute kinematics on (history+future) so first simulated step
    # has a valid central difference, then slice out history.
    linear_speed, linear_accel, angular_speed, angular_accel = traj.compute_kinematic_features(
        evaluated.x, evaluated.y, evaluated.z, evaluated.heading, seconds_per_step=cfg.step_duration_seconds
    )

    eval_object_mask = torch.any(evaluated_ids[:, None].eq(simulated.object_id[None, :]), dim=0)

    dno_fn = inter.compute_distance_to_nearest_object
    ttc_fn = inter.compute_time_to_collision_with_object_in_front
    d_road_fn = map_feat.compute_distance_to_road_edge

    # Keep compile behind an env flag so we can A/B easily without changing call sites.
    if bool(int(__import__("os").environ.get("WOSAC_TORCH_COMPILE", "0"))):
        dno_fn = _maybe_compile("dno", dno_fn)
        ttc_fn = _maybe_compile("ttc", ttc_fn)
        d_road_fn = _maybe_compile("d_road", d_road_fn)

    dno = dno_fn(
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
    ttc = ttc_fn(
        center_x=simulated.x,
        center_y=simulated.y,
        length=simulated.length,
        width=simulated.width,
        heading=simulated.heading,
        valid=simulated.valid,
        evaluated_object_mask=eval_object_mask,
        seconds_per_step=cfg.step_duration_seconds,
    )

    cached = _cache_get_or_build(scenario)
    road_edges = cached.get("road_edges") or [list(mf.road_edge.polyline) for mf in scenario.map_features if mf.HasField("road_edge")]
    d_road = d_road_fn(
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
        road_edge_polylines_tensor=cached.get("road_edge_polylines_tensor"),
        is_polyline_cyclic=cached.get("road_edge_is_cyclic"),
    )

    lane_ids = cached.get("lane_ids") or []
    lane_polys = cached.get("lane_polys") or []
    traffic_signals = cached.get("traffic_signals") or [list(dms.lane_states) for dms in scenario.dynamic_map_states]

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
                lane_tensor=cached.get("lane_tensor"),
                lane_ids_tensor=cached.get("lane_ids_tensor"),
                ts_lane_id=cached.get("ts_lane_id"),
                ts_state=cached.get("ts_state"),
                ts_stop_point=cached.get("ts_stop_point"),
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
                lane_tensor=cached.get("lane_tensor"),
                lane_ids_tensor=cached.get("lane_ids_tensor"),
                ts_lane_id=cached.get("ts_lane_id"),
                ts_state=cached.get("ts_state"),
                ts_stop_point=cached.get("ts_stop_point"),
            )
    else:
        red_light = torch.zeros(
            (
                len(evaluated_ids),
                simulated.valid.shape[1],
            ),
            dtype=torch.float32 if surrogate is not None else torch.bool,
        )

    # Slice time for SIM_AGENTS (remove history)
    validity_mask = validity_mask[:, ct_idx + 1 :]
    displacement_error = traj.compute_displacement_error(
        evaluated.x, evaluated.y, evaluated.z, eval_logged.x, eval_logged.y, eval_logged.z
    )
    object_valid_steps = eval_logged.valid.to(torch.float32).sum(dim=1)
    ade = (torch.where(eval_logged.valid, displacement_error, torch.zeros_like(displacement_error)).sum(dim=1) / object_valid_steps)

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
        # collision distance threshold is 0.0: negative => penetration => high prob
        is_colliding = torch.sigmoid(-k_col * dno)
        is_offroad = torch.sigmoid(k_off * d_road)
        tl_out = red_light  # already float

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


def compute_scenario_rollouts_features(
    scenario: scenario_pb2.Scenario,
    scenario_rollouts: sim_agents_submission_pb2.ScenarioRollouts,
    *,
    challenge_type: _ChallengeType = _ChallengeType.SIM_AGENTS,
) -> Tuple[MetricFeaturesTorch, MetricFeaturesTorch]:
    """Torch port of `metric_features.compute_scenario_rollouts_features` (SIM_AGENTS only)."""
    log_joint = scenario_to_joint_scene(scenario, challenge_type)
    log_feat = compute_metric_features(
        scenario,
        log_joint,
        challenge_type=challenge_type,
        use_log_validity=True,
    )

    sims: List[MetricFeaturesTorch] = []
    for js in scenario_rollouts.joint_scenes:
        sims.append(
            compute_metric_features(
                scenario,
                js,
                challenge_type=challenge_type,
                use_log_validity=False,
            )
        )

    # Concatenate along sample axis (0)
    def cat(field: str) -> Tensor:
        return torch.cat([getattr(m, field) for m in sims], dim=0)

    sim_feat = MetricFeaturesTorch(
        object_id=log_feat.object_id,
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
    return log_feat, sim_feat


__all__ = ["compute_metric_features", "compute_scenario_rollouts_features"]

