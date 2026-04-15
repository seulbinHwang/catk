from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor

from . import box_utils_torch
from . import geometry_box_distance_torch as geom_box
from . import geometry_utils_torch as geom
from . import trajectory_features_torch as traj

EXTREMELY_LARGE_DISTANCE = 1e10
COLLISION_DISTANCE_THRESHOLD = 0.0
CORNER_ROUNDING_FACTOR = 0.7

MAX_HEADING_DIFF = math.radians(75.0)
MAX_HEADING_DIFF_FOR_SMALL_OVERLAP = math.radians(10.0)
SMALL_OVERLAP_THRESHOLD = 0.5

MAXIMUM_TIME_TO_COLLISION = 5.0


def compute_distance_to_nearest_object(
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
    corner_rounding_factor: float = CORNER_ROUNDING_FACTOR,
    scene_batch: "Tensor | None" = None,
) -> Tensor:
    """Torch port of TF `interaction_features.compute_distance_to_nearest_object`."""
    boxes = torch.stack([center_x, center_y, center_z, length, width, height, heading], dim=-1)
    num_objects, num_steps, num_features = boxes.shape

    shrinking_distance = torch.minimum(boxes[:, :, 3], boxes[:, :, 4]) * float(corner_rounding_factor) / 2.0
    boxes = torch.cat(
        [
            boxes[:, :, :3],
            boxes[:, :, 3:4] - 2.0 * shrinking_distance[..., None],
            boxes[:, :, 4:5] - 2.0 * shrinking_distance[..., None],
            boxes[:, :, 5:],
        ],
        dim=2,
    )

    boxes_flat = boxes.reshape(num_objects * num_steps, num_features)
    box_corners = box_utils_torch.get_upright_3d_box_corners(boxes_flat)[:, :4, :2]
    box_corners = box_corners.reshape(num_objects, num_steps, 4, 2)

    eval_idx = torch.where(evaluated_object_mask)[0]
    other_idx = torch.where(~evaluated_object_mask)[0]
    eval_corners = box_corners.index_select(0, eval_idx)
    other_corners = box_corners.index_select(0, other_idx)
    num_eval = eval_corners.shape[0]
    all_corners = torch.cat([eval_corners, other_corners], dim=0)

    # Chunk over 'other objects' dimension to reduce peak memory.
    # Result is identical because we take min over objects (with masking).
    block = 32
    best = torch.full((num_eval, num_steps), EXTREMELY_LARGE_DISTANCE, dtype=box_corners.dtype, device=box_corners.device)

    # subtract shrinking distances to recover rounded-rect distance
    eval_shrink = shrinking_distance.index_select(0, eval_idx)  # (E,T)
    other_shrink = shrinking_distance.index_select(0, other_idx)
    all_shrink = torch.cat([eval_shrink, other_shrink], dim=0)  # (O,T)

    eval_valid = valid.index_select(0, eval_idx)  # (E,T)
    other_valid = valid.index_select(0, other_idx)
    all_valid = torch.cat([eval_valid, other_valid], dim=0)  # (O,T)

    # scene-batch cross-scene masking: reorder scene labels to match [eval, other]
    eval_scene: "Tensor | None" = None
    all_scene: "Tensor | None" = None
    if scene_batch is not None:
        eval_scene = scene_batch.index_select(0, eval_idx)  # (E,)
        all_scene = torch.cat([eval_scene, scene_batch.index_select(0, other_idx)], dim=0)  # (O,)

    for start in range(0, num_objects, block):
        end = min(num_objects, start + block)
        corners_blk = all_corners[start:end]  # (B,T,4,2)
        B = corners_blk.shape[0]

        eval_bc = eval_corners[:, None, ...].expand(num_eval, B, num_steps, 4, 2)
        blk_bc = corners_blk[None, ...].expand(num_eval, B, num_steps, 4, 2)
        eval_flat = eval_bc.reshape(num_eval * B * num_steps, 4, 2)
        blk_flat = blk_bc.reshape(num_eval * B * num_steps, 4, 2)

        mink = geom_box.minkowski_sum_of_box_and_box_points(eval_flat, -1.0 * blk_flat)
        signed_flat = geom_box.signed_distance_from_point_to_convex_polygon(
            query_points=torch.zeros_like(mink[:, 0, :]), polygon_points=mink
        )
        signed_blk = signed_flat.reshape(num_eval, B, num_steps)

        # recover rounded-rect distances for this block
        shrink_blk = all_shrink[start:end]  # (B,T)
        signed_blk = signed_blk - eval_shrink[:, None, :] - shrink_blk[None, :, :]

        # mask self distances when block overlaps evaluated objects portion
        if start < num_eval:
            diag_end = min(num_eval, end)
            ii = torch.arange(start, diag_end, device=signed_blk.device)
            jj = ii - start
            signed_blk[ii, jj, :] = signed_blk[ii, jj, :] + EXTREMELY_LARGE_DISTANCE

        # cross-scene masking: agents from different scenes never interact
        if scene_batch is not None:
            blk_scene = all_scene[start:end]  # (B,)
            diff_scene = eval_scene[:, None] != blk_scene[None, :]  # (E, B)
            signed_blk = torch.where(
                diff_scene[:, :, None].expand(-1, -1, num_steps),
                torch.full_like(signed_blk, EXTREMELY_LARGE_DISTANCE),
                signed_blk,
            )

        # validity masking: both ego and other must be valid
        valid_blk = all_valid[start:end]  # (B,T)
        valid_mask = eval_valid[:, None, :] & valid_blk[None, :, :]
        signed_blk = torch.where(valid_mask, signed_blk, torch.full_like(signed_blk, EXTREMELY_LARGE_DISTANCE))

        best = torch.minimum(best, signed_blk.min(dim=1).values)

    return best


def _get_object_following_mask(long_distance: Tensor, lat_overlap: Tensor, yaw_diff: Tensor) -> Tensor:
    valid_mask = long_distance > 0.0
    valid_mask = valid_mask & (yaw_diff <= MAX_HEADING_DIFF)
    valid_mask = valid_mask & (lat_overlap < 0.0)
    return valid_mask & (
        (lat_overlap < -SMALL_OVERLAP_THRESHOLD) | (yaw_diff <= MAX_HEADING_DIFF_FOR_SMALL_OVERLAP)
    )


def compute_time_to_collision_with_object_in_front(
    *,
    center_x: Tensor,
    center_y: Tensor,
    length: Tensor,
    width: Tensor,
    heading: Tensor,
    valid: Tensor,
    evaluated_object_mask: Tensor,
    seconds_per_step: float,
    scene_batch: "Tensor | None" = None,
) -> Tensor:
    """Torch port of TF `interaction_features.compute_time_to_collision_with_object_in_front`."""
    speed = traj.compute_kinematic_features(
        x=center_x,
        y=center_y,
        z=torch.zeros_like(center_x),
        heading=heading,
        seconds_per_step=seconds_per_step,
    )[0]  # (O,T)

    boxes = torch.stack([center_x, center_y, length, width, heading, speed], dim=-1)  # (O,T,6)
    boxes = boxes.permute(1, 0, 2)  # (T,O,6)
    valid_t = valid.permute(1, 0)  # (T,O)

    eval_idx = torch.where(evaluated_object_mask)[0]
    eval_boxes = boxes.index_select(1, eval_idx)  # (T,E,6)

    ego_xy, ego_sizes, ego_yaw, ego_speed = torch.split(eval_boxes, [2, 2, 1, 1], dim=-1)
    other_xy, other_sizes, other_yaw, _ = torch.split(boxes, [2, 2, 1, 1], dim=-1)

    yaw_diff = torch.abs(other_yaw[:, None, :, :] - ego_yaw[:, :, None, :])  # (T,E,O,1)
    yaw_diff_cos = torch.cos(yaw_diff)
    yaw_diff_sin = torch.sin(yaw_diff)

    other_long_offset = geom.dot_product_2d(
        (other_sizes[:, None, :, :] / 2.0),
        torch.abs(torch.cat([yaw_diff_cos, yaw_diff_sin], dim=-1)),
    )  # (T,E,O)
    other_lat_offset = geom.dot_product_2d(
        (other_sizes[:, None, :, :] / 2.0),
        torch.abs(torch.cat([yaw_diff_sin, yaw_diff_cos], dim=-1)),
    )  # (T,E,O)

    other_rel_xy = geom_box.rotate_2d_points(
        (other_xy[:, None, :, :] - ego_xy[:, :, None, :]), -ego_yaw
    )  # (T,E,O,2)

    long_distance = other_rel_xy[..., 0] - ego_sizes[:, :, None, 0] / 2.0 - other_long_offset
    lat_overlap = torch.abs(other_rel_xy[..., 1]) - ego_sizes[:, :, None, 1] / 2.0 - other_lat_offset

    following_mask = _get_object_following_mask(long_distance, lat_overlap, yaw_diff[..., 0])

    # cross-scene masking: an eval agent cannot follow an agent from a different scene
    if scene_batch is not None:
        eval_scene = scene_batch.index_select(0, eval_idx)  # (E,)
        same_scene = eval_scene[None, :, None] == scene_batch[None, None, :]  # (1, E, O)
        following_mask = following_mask & same_scene.expand(valid_t.shape[0], -1, -1)

    valid_mask = valid_t[:, None, :] & following_mask
    masked_long = long_distance + (~valid_mask).to(long_distance.dtype) * EXTREMELY_LARGE_DISTANCE

    box_ahead_index = torch.argmin(masked_long, dim=-1)  # (T,E)
    idx = box_ahead_index.unsqueeze(-1)
    distance_to_ahead = torch.gather(masked_long, dim=-1, index=idx).squeeze(-1)  # (T,E)

    # box_ahead_speed: need broadcast speed into (T,E,O)
    speed_b = speed.permute(1, 0).unsqueeze(1).expand_as(masked_long)  # (T,E,O)
    box_ahead_speed = torch.gather(speed_b, dim=-1, index=idx).squeeze(-1)  # (T,E)

    rel_speed = ego_speed[..., 0] - box_ahead_speed
    # Avoid NaN/Inf gradients: don't form distance/rel_speed when rel_speed <= 0.
    rel_speed_safe = torch.where(rel_speed > 0.0, rel_speed, torch.ones_like(rel_speed))
    ttc_raw = distance_to_ahead / rel_speed_safe
    ttc = torch.where(
        rel_speed > 0.0,
        torch.minimum(ttc_raw, torch.full_like(distance_to_ahead, MAXIMUM_TIME_TO_COLLISION)),
        torch.full_like(distance_to_ahead, MAXIMUM_TIME_TO_COLLISION),
    )
    return ttc.permute(1, 0)


__all__ = [
    "EXTREMELY_LARGE_DISTANCE",
    "COLLISION_DISTANCE_THRESHOLD",
    "compute_distance_to_nearest_object",
    "compute_time_to_collision_with_object_in_front",
]

