from __future__ import annotations

import torch
from torch import Tensor


def wrap_angle(angle: Tensor) -> Tensor:
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


def rotate_points(points: Tensor, angle: Tensor) -> Tensor:
    """Rotate 2D points by ``angle`` radians.

    ``points`` may have extra dimensions before the final xy dimension. ``angle``
    must broadcast to ``points[..., 0]``.
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    x = points[..., 0]
    y = points[..., 1]
    return torch.stack((cos * x - sin * y, sin * x + cos * y), dim=-1)


def global_to_local_xy(points: Tensor, origin: Tensor, heading: Tensor) -> Tensor:
    while origin.dim() < points.dim():
        origin = origin.unsqueeze(-2)
    while heading.dim() < points.dim() - 1:
        heading = heading.unsqueeze(-1)
    return rotate_points(points - origin, -heading)


def local_to_global_xy(points: Tensor, origin: Tensor, heading: Tensor) -> Tensor:
    while origin.dim() < points.dim():
        origin = origin.unsqueeze(-2)
    while heading.dim() < points.dim() - 1:
        heading = heading.unsqueeze(-1)
    return rotate_points(points, heading) + origin


def heading_vector(heading: Tensor) -> Tensor:
    return torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)


def relation_features(
    query_pos: Tensor,
    query_heading: Tensor,
    key_pos: Tensor,
    key_heading: Tensor,
    distance_scale: float = 100.0,
    self_relation_value: float | None = 1.0e-4,
) -> Tensor:
    """Build relative distance, bearing and heading features.

    Args:
        query_pos: ``[B, Q, 2]`` query anchor positions.
        query_heading: ``[B, Q]`` query anchor headings.
        key_pos: ``[B, K, 2]`` key anchor positions.
        key_heading: ``[B, K]`` key anchor headings.
        distance_scale: Constant used to keep distances numerically modest.

    Returns:
        ``[B, Q, K, 3]`` relation features.
    """
    rel = key_pos.unsqueeze(1) - query_pos.unsqueeze(2)
    dist = torch.linalg.norm(rel, dim=-1) / distance_scale
    bearing_global = torch.atan2(rel[..., 1], rel[..., 0])
    bearing = wrap_angle(bearing_global - query_heading.unsqueeze(-1)) / torch.pi
    dheading = wrap_angle(key_heading.unsqueeze(1) - query_heading.unsqueeze(-1)) / torch.pi
    rel_features = torch.stack((dist, bearing, dheading), dim=-1)
    if self_relation_value is not None and query_pos.shape[1] == key_pos.shape[1]:
        diag = torch.arange(query_pos.shape[1], device=query_pos.device)
        rel_features[:, diag, diag] = self_relation_value
    return rel_features
