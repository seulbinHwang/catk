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


@torch.no_grad()
def cal_polygon_contour(
    pos: Tensor,  # [n_agent, n_step, n_target, 2]
    head: Tensor,  # [n_agent, n_step, n_target]
    width_length: Tensor,  # [n_agent, 1, 1, 2]
) -> Tensor:
    x, y = pos[..., 0], pos[..., 1]
    width, length = width_length[..., 0], width_length[..., 1]

    half_cos = 0.5 * head.cos()
    half_sin = 0.5 * head.sin()
    length_cos = length * half_cos
    length_sin = length * half_sin
    width_cos = width * half_cos
    width_sin = width * half_sin

    left_front = torch.stack((x + length_cos - width_sin, y + length_sin + width_cos), dim=-1)
    right_front = torch.stack((x + length_cos + width_sin, y + length_sin - width_cos), dim=-1)
    right_back = torch.stack((x - length_cos + width_sin, y - length_sin - width_cos), dim=-1)
    left_back = torch.stack((x - length_cos - width_sin, y - length_sin + width_cos), dim=-1)
    return torch.stack((left_front, right_front, right_back, left_back), dim=-2)


def transform_to_global(
    pos_local: Tensor,
    head_local: Optional[Tensor],
    pos_now: Tensor,
    head_now: Tensor,
) -> Tuple[Tensor, Optional[Tensor]]:
    cos, sin = head_now.cos(), head_now.sin()
    rot_mat = torch.zeros((head_now.shape[0], 2, 2), device=head_now.device)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = sin
    rot_mat[:, 1, 0] = -sin
    rot_mat[:, 1, 1] = cos

    pos_global = torch.bmm(pos_local, rot_mat)
    pos_global = pos_global + pos_now.unsqueeze(1)
    if head_local is None:
        return pos_global, None
    return pos_global, head_local + head_now.unsqueeze(1)


def transform_to_local(
    pos_global: Tensor,
    head_global: Optional[Tensor],
    pos_now: Tensor,
    head_now: Tensor,
) -> Tuple[Tensor, Optional[Tensor]]:
    cos, sin = head_now.cos(), head_now.sin()
    rot_mat = torch.zeros((head_now.shape[0], 2, 2), device=head_now.device)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = -sin
    rot_mat[:, 1, 0] = sin
    rot_mat[:, 1, 1] = cos

    pos_local = pos_global - pos_now.unsqueeze(1)
    pos_local = torch.bmm(pos_local, rot_mat)
    if head_global is None:
        return pos_local, None
    return pos_local, head_global - head_now.unsqueeze(1)
