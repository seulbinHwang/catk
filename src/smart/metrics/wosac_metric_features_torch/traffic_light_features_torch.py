from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor

from waymo_open_dataset.protos import map_pb2

from . import geometry_utils_torch as geom
from . import map_metric_features_torch as map_feat

EXTREMELY_LARGE_DISTANCE = 1e10

_Polyline = List[map_pb2.MapPoint]


def _argmin_2d(t: Tensor) -> Tensor:
    """Torch port of TF `_argmin_2d` for tensors [B,R,C] -> [B,2] (rows, cols)."""
    if t.dim() != 3:
        raise ValueError(f"_argmin_2d expects [B,R,C], got {t.shape}")
    b, r, c = t.shape
    flat = t.reshape(b, r * c)
    flat_idx = torch.argmin(flat, dim=1)
    cols = flat_idx % c
    rows = flat_idx // c
    return torch.stack([rows, cols], dim=1)


def _get_nearest_lane_segment_index(*, xy: Tensor, lane_xyz_valid: Tensor) -> Tensor:
    """Torch port of TF `_get_nearest_lane_segment_index`.

    Args:
      xy: (num_points, 2)
      lane_xyz_valid: (num_points, num_lanes, num_segments+1, 4)
    Returns:
      (num_points, 2): (lane_index, segment_index)
    """
    lane_xy = lane_xyz_valid[..., :2]  # (P,L,S+1,2)
    lane_valid = lane_xyz_valid[..., 3].to(torch.bool)  # (P,L,S+1)

    segment_start_xy = lane_xy[..., :-1, :]  # (P,L,S,2)
    segment_end_xy = lane_xy[..., 1:, :]  # (P,L,S,2)

    start_to_point = xy[:, None, None, :] - segment_start_xy
    start_to_end = segment_end_xy - segment_start_xy

    rel_t = geom.divide_no_nan(
        geom.dot_product_2d(start_to_point, start_to_end),
        geom.dot_product_2d(start_to_end, start_to_end),
    )
    clipped_rel_t = rel_t.clamp(0.0, 1.0)
    distance_to_segment = torch.linalg.norm(
        start_to_point + start_to_end * clipped_rel_t[..., None], dim=-1
    )

    distance_to_segment = torch.where(
        lane_valid[..., :-1], distance_to_segment, torch.full_like(distance_to_segment, EXTREMELY_LARGE_DISTANCE)
    )
    return _argmin_2d(distance_to_segment)


def _tensorize_traffic_signals(
    traffic_signals: List[List[map_pb2.TrafficSignalLaneState]],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Torch port of TF `_tensorize_traffic_signals`."""
    num_steps = len(traffic_signals)
    all_lane_ids: List[int] = []
    for signals_at_t in traffic_signals:
        for state in signals_at_t:
            all_lane_ids.append(int(state.lane))
    all_lane_ids = list(set(all_lane_ids))
    num_tl_lanes = len(all_lane_ids)
    if num_tl_lanes == 0:
        all_lane_ids = [-1]
        num_tl_lanes = 1

    lane_id_to_idx = {lid: i for i, lid in enumerate(all_lane_ids)}
    lane_ids = torch.tensor(all_lane_ids, dtype=torch.int32).unsqueeze(0).repeat(num_steps, 1)
    states = torch.zeros((num_steps, num_tl_lanes), dtype=torch.int64)
    stop_points = torch.zeros((num_steps, num_tl_lanes, 2), dtype=torch.float32)
    for t, signals_at_t in enumerate(traffic_signals):
        for st in signals_at_t:
            idx = lane_id_to_idx[int(st.lane)]
            states[t, idx] = int(st.state)
            stop_points[t, idx, 0] = float(st.stop_point.x)
            stop_points[t, idx, 1] = float(st.stop_point.y)
    return lane_ids, states, stop_points


def compute_red_light_violation(
    *,
    center_x: Tensor,
    center_y: Tensor,
    valid: Tensor,
    evaluated_object_mask: Tensor,
    lane_polylines: List[_Polyline],
    lane_ids: List[int],
    traffic_signals: List[List[map_pb2.TrafficSignalLaneState]],
    lane_tensor: Tensor | None = None,
    lane_ids_tensor: Tensor | None = None,
    ts_lane_id: Tensor | None = None,
    ts_state: Tensor | None = None,
    ts_stop_point: Tensor | None = None,
) -> Tensor:
    """Torch port of TF `compute_red_light_violation`.

    Returns:
      (num_eval_objects, num_steps) bool
    """
    if not lane_polylines:
        raise ValueError("Missing lanes.")
    if not traffic_signals:
        raise ValueError("Missing traffic signals.")
    if len(lane_polylines) != len(lane_ids):
        raise ValueError("Inconsistent number of lane polylines and lane ids.")

    # evaluated_object_indices: (num_eval,)
    evaluated_object_indices = torch.where(evaluated_object_mask)[0]
    xy = torch.stack([center_x, center_y], dim=-1).index_select(0, evaluated_object_indices)
    valid = valid.index_select(0, evaluated_object_indices)
    num_objects, num_steps = valid.shape

    if lane_tensor is None or lane_ids_tensor is None:
        lane_tensor, lane_ids_tensor = map_feat.tensorize_polylines(lane_polylines, lane_ids)
    lane_tensor = lane_tensor.to(center_x.device)
    lane_ids_tensor = lane_ids_tensor.to(center_x.device)
    # xy_flat: (num_objects*num_steps, 2)
    xy_flat = xy.reshape(-1, 2)

    # nearest segment: lane_xyz_valid needs (P,L,S+1,4)
    lane_xyz_valid = lane_tensor.unsqueeze(0).expand(xy_flat.shape[0], -1, -1, -1)
    nearest_lane_segment_index = _get_nearest_lane_segment_index(xy=xy_flat, lane_xyz_valid=lane_xyz_valid)
    nearest_lane_index = nearest_lane_segment_index[:, 0]

    current_lane_id_flat = lane_ids_tensor.to(nearest_lane_index.device).index_select(0, nearest_lane_index)
    current_lane_id = current_lane_id_flat.reshape(num_objects, num_steps)

    if ts_lane_id is None or ts_state is None or ts_stop_point is None:
        ts_lane_id, ts_state, ts_stop_point = _tensorize_traffic_signals(traffic_signals)
    ts_lane_id = ts_lane_id.to(center_x.device)
    ts_state = ts_state.to(center_x.device)
    ts_stop_point = ts_stop_point.to(center_x.device)
    # Trajectory may be shorter than full scenario (e.g. truncated closed-loop rollout): align TL time axis.
    _t_tl = ts_lane_id.shape[0]
    if num_steps != _t_tl:
        if num_steps > _t_tl:
            raise ValueError(
                f"trajectory num_steps={num_steps} exceeds traffic_signals length={_t_tl}"
            )
        ts_lane_id = ts_lane_id[:num_steps]
        ts_state = ts_state[:num_steps]
        ts_stop_point = ts_stop_point[:num_steps]
    num_traffic_signals = ts_lane_id.shape[1]

    ts_match = current_lane_id[:, :, None].eq(ts_lane_id[None, :, :])  # (O,T,TL)
    ts_is_stop = (ts_state == int(map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_STOP)) | (
        ts_state == int(map_pb2.TrafficSignalLaneState.State.LANE_STATE_STOP)
    )  # (T,TL)

    ts_match_lane = lane_ids_tensor[None, None, :].eq(ts_lane_id[:, :, None])  # (T,TL,L)
    ts_lane_index = torch.argmax(ts_match_lane.to(torch.int64), dim=-1)  # (T,TL)
    ts_lane_valid = torch.any(ts_match_lane, dim=-1)  # (T,TL)

    ts_segments = lane_tensor.index_select(0, ts_lane_index.reshape(-1)).reshape(
        num_steps, num_traffic_signals, lane_tensor.shape[1], 4
    )  # (T,TL,S+1,4)

    # Fence index for stop point: run nearest on each (t,tl) within its lane.
    ts_stop_flat = ts_stop_point.reshape(num_steps * num_traffic_signals, 2)
    ts_segments_flat = ts_segments.reshape(num_steps * num_traffic_signals, -1, 4)
    nearest = _get_nearest_lane_segment_index(
        xy=ts_stop_flat,
        lane_xyz_valid=ts_segments_flat[:, None, :, :],
    )
    ts_stop_point_segment_index = nearest[:, 1].reshape(num_steps, num_traffic_signals)

    idx_pair = torch.stack(
        [ts_stop_point_segment_index, ts_stop_point_segment_index + 1], dim=-1
    )  # (T,TL,2)
    # Gather segment endpoints along axis=2 from ts_segments (T,TL,S+1,4).
    gather_idx = idx_pair[..., None].expand(-1, -1, -1, 4)
    ts_stop_point_segment = torch.gather(ts_segments, dim=2, index=gather_idx)  # (T,TL,2,4)

    start_to_end = ts_stop_point_segment[..., 1, :2] - ts_stop_point_segment[..., 0, :2]  # (T,TL,2)
    seg_len2 = geom.dot_product_2d(start_to_end, start_to_end)  # (T,TL)
    start_to_stop = ts_stop_point - ts_stop_point_segment[..., 0, :2]  # (T,TL,2)
    start_to_xy = xy[:, :, None, :] - ts_stop_point_segment[None, :, :, 0, :2]  # (O,T,TL,2)

    stop_rel_t = geom.divide_no_nan(
        geom.dot_product_2d(start_to_stop, start_to_end),
        seg_len2,
    )  # (T,TL)
    obj_rel_t = geom.divide_no_nan(
        geom.dot_product_2d(start_to_xy, start_to_end[None, :, :, :]),
        seg_len2[None, :, :],
    )  # (O,T,TL)

    behind = obj_rel_t < stop_rel_t[None, :, :]
    ahead = obj_rel_t > stop_rel_t[None, :, :]
    crossed = behind[:, :-1, :] & ahead[:, 1:, :]
    crossed = torch.cat([torch.zeros_like(crossed[:, 0:1, :]), crossed], dim=1)  # (O,T,TL)

    return torch.any(
        valid[:, :, None]
        & ts_match
        & ts_is_stop[None, :, :]
        & ts_lane_valid[None, :, :]
        & crossed,
        dim=-1,
    )


__all__ = ["compute_red_light_violation"]


def compute_red_light_violation_soft(
    *,
    center_x: Tensor,
    center_y: Tensor,
    valid: Tensor,
    evaluated_object_mask: Tensor,
    lane_polylines: List[_Polyline],
    lane_ids: List[int],
    traffic_signals: List[List[map_pb2.TrafficSignalLaneState]],
    crossing_temperature: float = 0.05,
    lane_tensor: Tensor | None = None,
    lane_ids_tensor: Tensor | None = None,
    ts_lane_id: Tensor | None = None,
    ts_state: Tensor | None = None,
    ts_stop_point: Tensor | None = None,
) -> Tensor:
    """Differentiable surrogate of `compute_red_light_violation`.

    Returns:
      (num_eval_objects, num_steps) float in [0,1] (violation probability-like).
    """
    if crossing_temperature <= 0:
        raise ValueError("crossing_temperature must be positive")

    # Reuse hard pipeline up to rel_t computations by duplicating minimal code.
    evaluated_object_indices = torch.where(evaluated_object_mask)[0]
    xy = torch.stack([center_x, center_y], dim=-1).index_select(0, evaluated_object_indices)
    valid = valid.index_select(0, evaluated_object_indices)
    num_objects, num_steps = valid.shape

    if lane_tensor is None or lane_ids_tensor is None:
        lane_tensor, lane_ids_tensor = map_feat.tensorize_polylines(lane_polylines, lane_ids)
    lane_tensor = lane_tensor.to(center_x.device)
    lane_ids_tensor = lane_ids_tensor.to(center_x.device)
    xy_flat = xy.reshape(-1, 2)
    lane_xyz_valid = lane_tensor.unsqueeze(0).expand(xy_flat.shape[0], -1, -1, -1)
    nearest_lane_segment_index = _get_nearest_lane_segment_index(xy=xy_flat, lane_xyz_valid=lane_xyz_valid)
    nearest_lane_index = nearest_lane_segment_index[:, 0]
    current_lane_id_flat = lane_ids_tensor.to(nearest_lane_index.device).index_select(0, nearest_lane_index)
    current_lane_id = current_lane_id_flat.reshape(num_objects, num_steps)

    if ts_lane_id is None or ts_state is None or ts_stop_point is None:
        ts_lane_id, ts_state, ts_stop_point = _tensorize_traffic_signals(traffic_signals)
    ts_lane_id = ts_lane_id.to(center_x.device)
    ts_state = ts_state.to(center_x.device)
    ts_stop_point = ts_stop_point.to(center_x.device)
    _t_tl = ts_lane_id.shape[0]
    if num_steps != _t_tl:
        if num_steps > _t_tl:
            raise ValueError(
                f"trajectory num_steps={num_steps} exceeds traffic_signals length={_t_tl}"
            )
        ts_lane_id = ts_lane_id[:num_steps]
        ts_state = ts_state[:num_steps]
        ts_stop_point = ts_stop_point[:num_steps]
    num_traffic_signals = ts_lane_id.shape[1]
    ts_match = current_lane_id[:, :, None].eq(ts_lane_id[None, :, :]).to(torch.float32)

    ts_is_stop = (
        (ts_state == int(map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_STOP))
        | (ts_state == int(map_pb2.TrafficSignalLaneState.State.LANE_STATE_STOP))
    ).to(torch.float32)  # (T,TL)

    ts_match_lane = lane_ids_tensor[None, None, :].eq(ts_lane_id[:, :, None])  # (T,TL,L)
    ts_lane_index = torch.argmax(ts_match_lane.to(torch.int64), dim=-1)
    ts_lane_valid = torch.any(ts_match_lane, dim=-1).to(torch.float32)

    ts_segments = lane_tensor.index_select(0, ts_lane_index.reshape(-1)).reshape(
        num_steps, num_traffic_signals, lane_tensor.shape[1], 4
    )

    ts_stop_flat = ts_stop_point.reshape(num_steps * num_traffic_signals, 2)
    ts_segments_flat = ts_segments.reshape(num_steps * num_traffic_signals, -1, 4)
    nearest = _get_nearest_lane_segment_index(
        xy=ts_stop_flat, lane_xyz_valid=ts_segments_flat[:, None, :, :]
    )
    seg_idx = nearest[:, 1].reshape(num_steps, num_traffic_signals)
    idx_pair = torch.stack([seg_idx, seg_idx + 1], dim=-1)
    ts_stop_point_segment = torch.gather(
        ts_segments, dim=2, index=idx_pair[..., None].expand(-1, -1, -1, 4)
    )  # (T,TL,2,4)

    start_to_end = ts_stop_point_segment[..., 1, :2] - ts_stop_point_segment[..., 0, :2]  # (T,TL,2)
    seg_len2 = geom.dot_product_2d(start_to_end, start_to_end)  # (T,TL)
    start_to_stop = ts_stop_point - ts_stop_point_segment[..., 0, :2]  # (T,TL,2)
    start_to_xy = xy[:, :, None, :] - ts_stop_point_segment[None, :, :, 0, :2]  # (O,T,TL,2)

    stop_rel_t = geom.divide_no_nan(geom.dot_product_2d(start_to_stop, start_to_end), seg_len2)  # (T,TL)
    obj_rel_t = geom.divide_no_nan(
        geom.dot_product_2d(start_to_xy, start_to_end[None, :, :, :]),
        seg_len2[None, :, :],
    )  # (O,T,TL)

    # Smooth crossing probability: behind(t-1) * ahead(t)
    k = 1.0 / float(crossing_temperature)
    delta = obj_rel_t - stop_rel_t[None, :, :]
    behind_prob = torch.sigmoid(-k * delta)
    ahead_prob = torch.sigmoid(k * delta)
    crossed_prob = behind_prob[:, :-1, :] * ahead_prob[:, 1:, :]
    crossed_prob = torch.cat([torch.zeros_like(crossed_prob[:, 0:1, :]), crossed_prob], dim=1)

    # Combine conditions (soft): valid * lane match * stop state * lane_valid * crossed
    p = (
        valid.to(torch.float32)[:, :, None]
        * ts_match
        * ts_is_stop[None, :, :]
        * ts_lane_valid[None, :, :]
        * crossed_prob
    )  # (O,T,TL)
    # Soft any over TL dimension: 1 - Π(1-p)
    return 1.0 - torch.prod(1.0 - p.clamp(0.0, 1.0), dim=-1)


__all__.append("compute_red_light_violation_soft")

