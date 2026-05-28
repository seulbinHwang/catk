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

from typing import Tuple

import torch
from torch import Tensor
from torch.nn.functional import one_hot

from src.smart.utils import cal_polygon_contour, transform_to_local, wrap_angle


@torch.no_grad()
def get_prob_targets_from_index(
    gt_idx: Tensor,  # [n_agent, n_step]
    token_traj: Tensor,  # [n_agent, n_token, 4, 2]
    label_smoothing: float = 0.0,
    spatial_aware_smoothing: bool = False,
) -> Tensor:  # [n_agent, n_step, n_token] prob, last dim sum up to 1
    closest_token_mask = one_hot(gt_idx, num_classes=token_traj.shape[1]).to(bool)
    prob_target = torch.zeros(
        gt_idx.shape[0],
        gt_idx.shape[1],
        token_traj.shape[1],
        device=gt_idx.device,
        dtype=token_traj.dtype,
    )

    if label_smoothing <= 0:
        prob_target[closest_token_mask] = 1.0
        return prob_target

    if not spatial_aware_smoothing:
        prob_target[closest_token_mask] = 1.0
        return prob_target

    gt_token_traj = torch.gather(
        token_traj,
        dim=1,
        index=gt_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 2),
    )
    dists = torch.norm(
        gt_token_traj[:, :, None, :, :] - token_traj[:, None, :, :, :],
        dim=-1,
    ).mean(-1)
    prob_target[closest_token_mask] = 1.0 - label_smoothing
    inv_sq_dist = 1.0 / ((1.0e-4 + dists) ** 2)
    inv_sq_dist = inv_sq_dist.masked_fill(closest_token_mask, 0.0)
    normalizer = inv_sq_dist.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    prob_target += inv_sq_dist / normalizer * label_smoothing
    return prob_target


@torch.no_grad()
def get_prob_targets(
    target: Tensor,  # [n_agent, n_step, 3] x,y,yaw in local coord
    token_agent_shape: Tensor,  # [n_agent, 2]
    token_traj: Tensor,  # [n_agent, n_token, 4, 2]
    label_smoothing: float = 0.0,
    spatial_aware_smoothing: bool = False,
) -> Tensor:  # [n_agent, n_step, n_token] prob, last dim sum up to 1
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
