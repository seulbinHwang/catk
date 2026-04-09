from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from . import geometry_utils_torch as geom

NUM_VERTICES_IN_BOX = 4


def rotate_2d_points(xys: Tensor, rotation_yaws: Tensor) -> Tensor:
    """Torch port of Waymo `geometry_utils.rotate_2d_points`."""
    c = torch.cos(rotation_yaws)
    s = torch.sin(rotation_yaws)
    xs_out = c * xys[..., 0] - s * xys[..., 1]
    ys_out = s * xys[..., 0] + c * xys[..., 1]
    return torch.stack([xs_out, ys_out], dim=-1)


def _get_downmost_edge_in_box(box: Tensor) -> Tuple[Tensor, Tensor]:
    """Torch port of `_get_downmost_edge_in_box`.

    Args:
      box: (B, 4, 2) CCW corners.
    Returns:
      downmost_vertex_idx: (B,1) int64
      downmost_edge_direction: (B,1,2) float
    """
    downmost_vertex_idx = torch.argmin(box[..., 1], dim=-1, keepdim=True)  # (B,1)
    edge_start = torch.gather(box, 1, downmost_vertex_idx[..., None].expand(-1, -1, 2))
    edge_end_idx = torch.remainder(downmost_vertex_idx + 1, NUM_VERTICES_IN_BOX)
    edge_end = torch.gather(box, 1, edge_end_idx[..., None].expand(-1, -1, 2))
    edge = edge_end - edge_start
    edge_len = torch.linalg.norm(edge, dim=-1, keepdim=True)
    edge_dir = edge / edge_len
    return downmost_vertex_idx, edge_dir


def minkowski_sum_of_box_and_box_points(box1_points: Tensor, box2_points: Tensor) -> Tensor:
    """Torch port of Waymo `minkowski_sum_of_box_and_box_points`.

    Inputs: (B,4,2) each, CCW corners.
    Output: (B,8,2) CCW corners.
    """
    point_order_1 = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long, device=box1_points.device)
    point_order_2 = torch.tensor([0, 1, 1, 2, 2, 3, 3, 0], dtype=torch.long, device=box1_points.device)

    box1_start_idx, box1_dir = _get_downmost_edge_in_box(box1_points)
    box2_start_idx, box2_dir = _get_downmost_edge_in_box(box2_points)

    condition = (geom.cross_product_2d(box1_dir, box2_dir) >= 0.0)  # (B,1)
    condition = condition.expand(-1, 8)

    box1_order = torch.where(condition, point_order_2, point_order_1)
    box1_order = torch.remainder(box1_order + box1_start_idx, NUM_VERTICES_IN_BOX)
    ordered_box1 = torch.gather(box1_points, 1, box1_order[..., None].expand(-1, -1, 2))

    box2_order = torch.where(condition, point_order_1, point_order_2)
    box2_order = torch.remainder(box2_order + box2_start_idx, NUM_VERTICES_IN_BOX)
    ordered_box2 = torch.gather(box2_points, 1, box2_order[..., None].expand(-1, -1, 2))

    return ordered_box1 + ordered_box2


def _get_edge_info(polygon_points: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Torch port of Waymo `_get_edge_info`."""
    first = polygon_points[:, 0:1, :]
    shifted = torch.cat([polygon_points[:, 1:, :], first], dim=-2)
    edge_vec = shifted - polygon_points
    edge_len = torch.linalg.norm(edge_vec, dim=-1)
    tangent = edge_vec / edge_len[..., None]
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
    return tangent, normal, edge_len


def signed_distance_from_point_to_convex_polygon(query_points: Tensor, polygon_points: Tensor) -> Tensor:
    """Torch port of Waymo `signed_distance_from_point_to_convex_polygon`.

    query_points: (B,2)
    polygon_points: (B,K,2) CCW vertices
    returns: (B,)
    """
    tangent, normal, edge_len = _get_edge_info(polygon_points)
    qp = query_points[:, None, :]
    v_to_q = qp - polygon_points  # (B,K,2)
    v_dist = torch.linalg.norm(v_to_q, dim=-1)

    edge_signed_perp = torch.sum(-normal * v_to_q, dim=-1)  # (B,K)
    is_inside = torch.all(edge_signed_perp <= 0, dim=-1)

    proj = torch.sum(tangent * v_to_q, dim=-1)
    proj_prop = proj / edge_len
    is_on_edge = (proj_prop >= 0.0) & (proj_prop <= 1.0)

    edge_perp = torch.abs(edge_signed_perp)
    inf = torch.tensor(float("inf"), device=polygon_points.device, dtype=polygon_points.dtype)
    edge_dist = torch.where(is_on_edge, edge_perp, inf)

    edge_and_vertex = torch.cat([edge_dist, v_dist], dim=-1)
    min_dist = torch.min(edge_and_vertex, dim=-1).values
    return torch.where(is_inside, -min_dist, min_dist)


__all__ = [
    "rotate_2d_points",
    "minkowski_sum_of_box_and_box_points",
    "signed_distance_from_point_to_convex_polygon",
]

