from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from waymo_open_dataset.protos import map_pb2

from . import box_utils_torch
from . import geometry_utils_torch as geom

# Mirror constants in TF implementation for parity.
EXTREMELY_LARGE_DISTANCE = 1e10
OFFROAD_DISTANCE_THRESHOLD = 0.0

_CYCLIC_MAP_FEATURE_TOLERANCE_M2 = 1.0

_Polyline = List[map_pb2.MapPoint]


def tensorize_polylines(
    polylines: List[_Polyline], ids: Optional[List[int]] = None
) -> Tuple[Tensor, Tensor]:
    """Torch port of `map_metric_features.tensorize_polylines`.

    Returns:
      stacked_polylines: (num_polylines, max_length, 4) float32 [x,y,z,valid]
      ids_tensor: (num_polylines,) int32
    """
    if ids is None:
        ids = [0] * len(polylines)
    elif len(polylines) != len(ids):
        raise ValueError("Inconsistent number of polylines and ids.")

    poly_tensors: List[Tensor] = []
    id_list: List[int] = []
    max_length = 0
    for polyline, fid in zip(polylines, ids):
        if len(polyline) < 2:
            continue
        max_length = max(max_length, len(polyline))
        pts = torch.tensor(
            [[p.x, p.y, p.z, 1.0] for p in polyline], dtype=torch.float32
        )
        poly_tensors.append(pts)
        id_list.append(int(fid))

    if not poly_tensors:
        # Match TF: stack would fail; raise earlier in callers for missing polylines.
        raise ValueError("No non-degenerate polylines.")

    padded = []
    for p in poly_tensors:
        if p.shape[0] == max_length:
            padded.append(p)
        else:
            pad = torch.zeros((max_length - p.shape[0], 4), dtype=p.dtype)
            padded.append(torch.cat([p, pad], dim=0))
    stacked_polylines = torch.stack(padded, dim=0)
    ids_tensor = torch.tensor(id_list, dtype=torch.int32)
    return stacked_polylines, ids_tensor


def check_polyline_cycles(polylines: List[_Polyline]) -> Tensor:
    """Torch port of `_check_polyline_cycles`."""
    cycles: List[Tensor] = []
    for polyline in polylines:
        if len(polyline) < 2:
            continue
        first = torch.tensor([polyline[0].x, polyline[0].y, polyline[0].z], dtype=torch.float32)
        last = torch.tensor([polyline[-1].x, polyline[-1].y, polyline[-1].z], dtype=torch.float32)
        cycles.append(((first - last) ** 2).sum(dim=-1) < _CYCLIC_MAP_FEATURE_TOLERANCE_M2)
    if not cycles:
        raise ValueError("No non-degenerate polylines.")
    return torch.stack(cycles, dim=0)


def compute_distance_to_road_edge(
    *,
    center_x: Tensor,
    center_y: Tensor,
    center_z: Tensor,
    length: Tensor,
    width: Tensor,
    height: Tensor,
    heading: Tensor,
    valid: Tensor,
    evaluated_object_mask: Tensor,
    road_edge_polylines: List[_Polyline],
    z_stretch: float = 3.0,
    road_edge_polylines_tensor: Tensor | None = None,
    is_polyline_cyclic: Tensor | None = None,
) -> Tensor:
    """Torch port of `map_metric_features.compute_distance_to_road_edge`.

    Returns:
      (num_eval_objects, num_steps) signed distances (off-road positive).
    """
    if not road_edge_polylines:
        raise ValueError("Missing road edges.")
    boxes = torch.stack([center_x, center_y, center_z, length, width, height, heading], dim=-1)
    num_objects, num_steps, num_features = boxes.shape
    boxes_flat = boxes.reshape(num_objects * num_steps, num_features)
    box_corners = box_utils_torch.get_upright_3d_box_corners(boxes_flat)[:, :4]  # bottom 4
    box_corners = box_corners.reshape(num_objects, num_steps, 4, 3)

    eval_idx = torch.where(evaluated_object_mask)[0]
    eval_corners = box_corners.index_select(0, eval_idx)
    num_eval_objects = eval_corners.shape[0]
    flat_eval_corners = eval_corners.reshape(-1, 3)

    if road_edge_polylines_tensor is None:
        polylines_tensor, _ = tensorize_polylines(road_edge_polylines)
    else:
        polylines_tensor = road_edge_polylines_tensor
    if is_polyline_cyclic is None:
        is_polyline_cyclic = check_polyline_cycles(road_edge_polylines)

    corner_dist = _compute_signed_distance_to_polylines(
        xyzs=flat_eval_corners,
        polylines=polylines_tensor,
        is_polyline_cyclic=is_polyline_cyclic,
        z_stretch=z_stretch,
    )
    corner_dist = corner_dist.reshape(num_eval_objects, num_steps, 4)
    signed_distances = corner_dist.max(dim=-1).values

    eval_validity = valid.index_select(0, eval_idx)
    return torch.where(eval_validity, signed_distances, torch.full_like(signed_distances, -EXTREMELY_LARGE_DISTANCE))


def _compute_signed_distance_to_polylines(
    *,
    xyzs: Tensor,
    polylines: Tensor,
    is_polyline_cyclic: Tensor | None = None,
    z_stretch: float = 1.0,
) -> Tensor:
    """Torch port of `_compute_signed_distance_to_polylines`."""
    num_points = xyzs.shape[0]
    if xyzs.shape != (num_points, 3):
        raise ValueError(f"xyzs must be (P,3), got {xyzs.shape}")
    num_polylines = polylines.shape[0]
    num_segments = polylines.shape[1] - 1
    if polylines.shape[2] != 4:
        raise ValueError(f"polylines must be (L,S+1,4), got {polylines.shape}")

    # Chunking over polylines: process L in blocks to reduce peak memory.
    # Result is identical because we only do min/argmin-style reduction over segments.
    polyline_chunk = 64
    if num_polylines <= polyline_chunk:
        polyline_chunk = num_polylines

    best_dist3 = torch.full((num_points,), EXTREMELY_LARGE_DISTANCE, dtype=xyzs.dtype, device=xyzs.device)
    best_signed2 = torch.full((num_points,), EXTREMELY_LARGE_DISTANCE, dtype=xyzs.dtype, device=xyzs.device)
    stretch = torch.tensor([1.0, 1.0, float(z_stretch)], dtype=xyzs.dtype, device=xyzs.device)

    for start in range(0, num_polylines, polyline_chunk):
        end = min(num_polylines, start + polyline_chunk)
        pol = polylines[start:end]  # (Lc,S+1,4)
        Lc = pol.shape[0]

        is_point_valid = pol[:, :, 3].to(torch.bool)
        is_segment_valid = is_point_valid[:, :-1] & is_point_valid[:, 1:]  # (Lc,S)

        if is_polyline_cyclic is None:
            cyc = torch.zeros((Lc,), dtype=torch.bool, device=pol.device)
        else:
            cyc = is_polyline_cyclic[start:end].to(torch.bool).to(pol.device)

        xyz_starts = pol[None, :, :-1, :3]  # (1,Lc,S,3)
        xyz_ends = pol[None, :, 1:, :3]
        start_to_point = xyzs[:, None, None, :3] - xyz_starts  # (P,Lc,S,3)
        start_to_end = xyz_ends - xyz_starts  # (1,Lc,S,3)

        rel_t = geom.divide_no_nan(
            geom.dot_product_2d(start_to_point[..., :2], start_to_end[..., :2]),
            geom.dot_product_2d(start_to_end[..., :2], start_to_end[..., :2]),
        )  # (P,Lc,S)

        n = torch.sign(geom.cross_product_2d(start_to_point[..., :2], start_to_end[..., :2]))  # (P,Lc,S)

        segment_to_point = start_to_point - (
            start_to_end * rel_t.clamp(0.0, 1.0)[..., None]
        )  # (P,Lc,S,3)

        dist_3d = torch.linalg.norm(segment_to_point * stretch[None, None, None, :], dim=-1)  # (P,Lc,S)
        dist_2d = torch.linalg.norm(segment_to_point[..., :2], dim=-1)

        start_to_end_padded = torch.cat(
            [
                start_to_end[:, :, -1:, :2],
                start_to_end[..., :2],
                start_to_end[:, :, :1, :2],
            ],
            dim=-2,
        )  # (1,Lc,S+2,2)

        is_locally_convex = geom.cross_product_2d(
            start_to_end_padded[:, :, :-1, :], start_to_end_padded[:, :, 1:, :]
        ) > 0.0  # (1,Lc,S+1)

        # Shifted n (P,Lc,S) and validity (Lc,S)
        n_prior = torch.cat(
            [
                torch.where(cyc[None, :, None], n[:, :, -1:], n[:, :, :1]),
                n[:, :, :-1],
            ],
            dim=-1,
        )
        n_next = torch.cat(
            [
                n[:, :, 1:],
                torch.where(cyc[None, :, None], n[:, :, :1], n[:, :, -1:]),
            ],
            dim=-1,
        )

        is_prior_valid = torch.cat(
            [
                torch.where(cyc[:, None], is_segment_valid[:, -1:], is_segment_valid[:, :1]),
                is_segment_valid[:, :-1],
            ],
            dim=-1,
        )
        is_next_valid = torch.cat(
            [
                is_segment_valid[:, 1:],
                torch.where(cyc[:, None], is_segment_valid[:, :1], is_segment_valid[:, -1:]),
            ],
            dim=-1,
        )

        sign_if_before = torch.where(
            is_locally_convex[:, :, :-1].expand_as(n),
            torch.maximum(n, n_prior),
            torch.minimum(n, n_prior),
        )
        sign_if_after = torch.where(
            is_locally_convex[:, :, 1:].expand_as(n),
            torch.maximum(n, n_next),
            torch.minimum(n, n_next),
        )

        sign_to_segment = torch.where(
            (rel_t < 0.0) & is_prior_valid[None, :, :],
            sign_if_before,
            torch.where((rel_t > 1.0) & is_next_valid[None, :, :], sign_if_after, n),
        )  # (P,Lc,S)

        # Flatten segments within chunk
        dist_3d_f = dist_3d.reshape(num_points, Lc * num_segments)
        dist_2d_f = dist_2d.reshape(num_points, Lc * num_segments)
        sign_f = sign_to_segment.reshape(num_points, Lc * num_segments)
        seg_valid_f = is_segment_valid.reshape(Lc * num_segments).to(dist_3d_f.device)

        dist_3d_f = torch.where(
            seg_valid_f[None, :], dist_3d_f, torch.full_like(dist_3d_f, EXTREMELY_LARGE_DISTANCE)
        )
        dist_2d_f = torch.where(
            seg_valid_f[None, :], dist_2d_f, torch.full_like(dist_2d_f, EXTREMELY_LARGE_DISTANCE)
        )

        closest = torch.argmin(dist_3d_f, dim=-1)  # (P,)
        idx = closest[:, None]
        chunk_best3 = torch.gather(dist_3d_f, 1, idx).squeeze(1)
        chunk_sign = torch.gather(sign_f, 1, idx).squeeze(1)
        chunk_dist2 = torch.gather(dist_2d_f, 1, idx).squeeze(1)
        chunk_signed2 = chunk_sign * chunk_dist2

        take = chunk_best3 < best_dist3
        best_signed2 = torch.where(take, chunk_signed2, best_signed2)
        best_dist3 = torch.where(take, chunk_best3, best_dist3)

    return best_signed2


__all__ = [
    "EXTREMELY_LARGE_DISTANCE",
    "OFFROAD_DISTANCE_THRESHOLD",
    "tensorize_polylines",
    "check_polyline_cycles",
    "compute_distance_to_road_edge",
]

