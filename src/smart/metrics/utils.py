# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Optional, Tuple

import torch
from torch import Tensor

from src.smart.utils import cal_polygon_contour, transform_to_local, wrap_angle

SPATIAL_SMOOTHING_CONTOUR_DISTANCE_CHUNK_SIZE = 128
CURRENT_STATE_TARGET_CHUNK_SIZE = 256


def _assign_spatial_aware_prob_target(
    flat_prob_target: Tensor,
    start: int,
    end: int,
    chunk_gt_idx: Tensor,
    dists: Tensor,
    label_smoothing: float,
) -> None:
    inv_sq_dist = 1.0 / (dists.square() + 1.0e-4)
    inv_sq_dist.scatter_(1, chunk_gt_idx.unsqueeze(1), 0.0)
    normalizer = inv_sq_dist.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    neighbor_target = inv_sq_dist / normalizer

    flat_prob_target[start:end] = neighbor_target * label_smoothing
    flat_prob_target[
        torch.arange(start, end, device=chunk_gt_idx.device),
        chunk_gt_idx,
    ] = 1.0 - label_smoothing


def _get_prob_targets_from_contour_index(
    gt_idx: Tensor,
    token_contour_trajectory: Tensor,  # [n_token, 5, 4, 2] or [n_token, 6, 4, 2]
    label_smoothing: float,
) -> Tensor:
    if token_contour_trajectory.shape[1] == 6:
        token_contour_trajectory = token_contour_trajectory[:, 1:]
    n_token = token_contour_trajectory.shape[0]
    prob_target = token_contour_trajectory.new_zeros(*gt_idx.shape, n_token)
    flat_gt_idx = gt_idx.reshape(-1)
    if flat_gt_idx.numel() == 0:
        return prob_target

    unique_gt_idx, inverse = torch.unique(
        flat_gt_idx,
        sorted=False,
        return_inverse=True,
    )
    unique_prob_target = token_contour_trajectory.new_zeros(
        unique_gt_idx.shape[0],
        n_token,
    )
    for start in range(
        0,
        unique_gt_idx.shape[0],
        SPATIAL_SMOOTHING_CONTOUR_DISTANCE_CHUNK_SIZE,
    ):
        end = min(
            start + SPATIAL_SMOOTHING_CONTOUR_DISTANCE_CHUNK_SIZE,
            unique_gt_idx.shape[0],
        )
        chunk_gt_idx = unique_gt_idx[start:end]
        gt_chunk = token_contour_trajectory[chunk_gt_idx]
        dists = torch.norm(
            gt_chunk[:, None, :, :, :] - token_contour_trajectory[None, :, :, :, :],
            dim=-1,
        ).mean(dim=(-1, -2))
        _assign_spatial_aware_prob_target(
            flat_prob_target=unique_prob_target,
            start=start,
            end=end,
            chunk_gt_idx=chunk_gt_idx,
            dists=dists,
            label_smoothing=label_smoothing,
        )
    prob_target.view(-1, n_token).copy_(unique_prob_target[inverse])
    return prob_target


@torch.no_grad()
def get_prob_targets_from_index(
    gt_idx: Tensor,  # [n_agent, n_step]
    token_traj: Tensor,  # [n_agent, n_token, 4, 2]
    token_contour_trajectory: Optional[Tensor] = None,  # [n_token, 5, 4, 2]
    label_smoothing: float = 0.0,
    spatial_aware_smoothing: bool = False,
) -> Tensor:  # [n_agent, n_step, n_token] prob
    n_token = (
        token_contour_trajectory.shape[0]
        if token_contour_trajectory is not None
        else token_traj.shape[1]
    )
    prob_target = torch.zeros(
        gt_idx.shape[0],
        gt_idx.shape[1],
        n_token,
        device=gt_idx.device,
        dtype=(
            token_contour_trajectory.dtype
            if token_contour_trajectory is not None
            else token_traj.dtype
        ),
    )

    if label_smoothing <= 0:
        prob_target.scatter_(-1, gt_idx.unsqueeze(-1), 1.0)
        return prob_target

    if not spatial_aware_smoothing:
        prob_target.scatter_(-1, gt_idx.unsqueeze(-1), 1.0)
        return prob_target

    if token_contour_trajectory is None:
        raise ValueError(
            "spatial-aware smoothing requires token_contour_trajectory "
            "with shape [n_token, 5, 4, 2] or [n_token, 6, 4, 2]."
        )

    return _get_prob_targets_from_contour_index(
        gt_idx=gt_idx,
        token_contour_trajectory=token_contour_trajectory,
        label_smoothing=label_smoothing,
    )


@torch.no_grad()
def get_prob_targets(
    target: Tensor,  # [n_agent, n_step, 3] x,y,yaw in local coord
    token_agent_shape: Tensor,  # [n_agent, 2]
    token_traj: Tensor,  # [n_agent, n_token, 4, 2]
    label_smoothing: float = 0.0,
    spatial_aware_smoothing: bool = False,
) -> Tensor:  # [n_agent, n_step, n_token] prob
    # ! tokenize to index, then compute prob
    contour = cal_polygon_contour(
        target[..., :2],  # [n_agent, n_step, 2]
        target[..., 2],  # [n_agent, n_step]
        token_agent_shape[:, None, :],  # [n_agent, 1, 1, 2]
    )  # [n_agent, n_step, 4, 2] in local coord

    # [n_agent, n_step, 1, 4, 2] - [n_agent, 1, n_token, 4, 2]
    target_token_index = (
        torch.norm(contour.unsqueeze(2) - token_traj[:, None, :, :, :], dim=-1)
        .sum(-1)
        .argmin(-1)
    )  # [n_agent, n_step]

    return get_prob_targets_from_index(
        gt_idx=target_token_index,
        token_traj=token_traj,
        label_smoothing=label_smoothing,
        spatial_aware_smoothing=spatial_aware_smoothing,
    )


@torch.no_grad()
def get_euclidean_targets(
    pred_pos: Tensor,  # [n_agent, 18, 2]
    pred_head: Tensor,  # [n_agent, 18]
    pred_valid: Tensor,  # [n_agent, 18]
    gt_pos: Tensor,  # [n_agent, 18, 2]
    gt_head: Tensor,  # [n_agent, 18]
    gt_valid: Tensor,  # [n_agent, 18]
) -> Tuple[Tensor, Tensor]:
    """
    Return: action that goes from [(10->15), ..., (85->90)]
        target: [n_agent, 16, 3], x,y,yaw
        target_valid: [n_agent, 16]
    """
    gt_last_pos = gt_pos.roll(shifts=-1, dims=1).flatten(0, 1)
    gt_last_head = gt_head.roll(shifts=-1, dims=1).flatten(0, 1)
    gt_last_valid = gt_valid.roll(shifts=-1, dims=1)  # [n_agent, 18]
    gt_last_valid[:, -1:] = False  # [n_agent, 18]

    target_pos, target_head = transform_to_local(
        pos_global=gt_last_pos.unsqueeze(1),  # [n_agent*18, 1, 2]
        head_global=gt_last_head.unsqueeze(1),  # [n_agent*18, 1]
        pos_now=pred_pos.flatten(0, 1),  # [n_agent*18, 2]
        head_now=pred_head.flatten(0, 1),  # [n_agent*18]
    )
    target_valid = pred_valid & gt_last_valid  # [n_agent, 18]

    target_pos = target_pos.squeeze(1).view(gt_pos.shape)  # n_agent, 18, 2]
    target_head = wrap_angle(target_head)  # [n_agent, 18]
    target_head = target_head.squeeze(1).view(gt_head.shape)
    target = torch.cat((target_pos, target_head.unsqueeze(-1)), dim=-1)

    # truncate [(5->10), ..., (90->5)] to [(10->15), ..., (85->90)]
    target = target[:, 1:-1]  # [n_agent, 16, 3], x,y,yaw
    target_valid = target_valid[:, 1:-1]  # [n_agent, 16]
    return target, target_valid


@torch.no_grad()
def match_current_state_trajectory_token_rows(
    pred_pos: Tensor,  # [n_row, 2]
    pred_head: Tensor,  # [n_row]
    gt_pos_segment: Tensor,  # [n_row, 5, 2]
    gt_head_segment: Tensor,  # [n_row, 5]
    token_trajectory: Tensor,  # [n_token, 5, 3]
    chunk_size: int = CURRENT_STATE_TARGET_CHUNK_SIZE,
) -> Tensor:  # [n_row]
    """Match full TrajTok trajectories from the current rollout state.

    The fixed TrajTok ``gt_idx`` is built during tokenization from the
    teacher-forced tokenized state. During rollout-style training, however, the
    model state can drift. This matcher transforms the raw 0.5 s GT segment into
    that current state frame and chooses the token whose full ``(x, y, yaw)``
    trajectory is closest.
    """
    n_row = int(pred_pos.shape[0])
    if n_row == 0:
        return torch.empty(0, dtype=torch.long, device=pred_pos.device)

    chunk_size = max(1, int(chunk_size))
    target_chunks = []
    token_pos = token_trajectory[:, :, :2]
    token_head = token_trajectory[:, :, 2]
    for start in range(0, n_row, chunk_size):
        end = min(start + chunk_size, n_row)
        gt_pos_local, gt_head_local = transform_to_local(
            pos_global=gt_pos_segment[start:end],
            head_global=gt_head_segment[start:end],
            pos_now=pred_pos[start:end],
            head_now=pred_head[start:end],
        )
        gt_head_local = wrap_angle(gt_head_local)

        pos_delta = token_pos.unsqueeze(0) - gt_pos_local.unsqueeze(1)
        head_delta = wrap_angle(token_head.unsqueeze(0) - gt_head_local.unsqueeze(1))
        dist = torch.sqrt(pos_delta.square().sum(-1) + head_delta.square()).mean(-1)
        target_chunks.append(dist.argmin(dim=-1))
    return torch.cat(target_chunks, dim=0)
