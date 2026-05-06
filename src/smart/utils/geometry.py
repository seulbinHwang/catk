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

import math

import torch


def angle_between_2d_vectors(
    ctr_vector: torch.Tensor, nbr_vector: torch.Tensor
) -> torch.Tensor:
    # If the neighbor vector is exactly zero, its relative angle is undefined
    # and atan2(0, 0) has an undefined backward. Replacing that vector with the
    # center heading preserves the intended "no relative direction" feature as
    # angle 0 while keeping the backward denominator non-zero.
    ctr_xy = ctr_vector[..., :2]
    nbr_xy = nbr_vector[..., :2]
    fallback_ctr = torch.zeros_like(ctr_xy)
    fallback_ctr[..., 0] = 1.0
    ctr_norm_sq = ctr_xy.pow(2).sum(dim=-1, keepdim=True)
    ctr_safe = torch.where(ctr_norm_sq > 1.0e-12, ctr_xy, fallback_ctr)
    nbr_norm_sq = nbr_xy.pow(2).sum(dim=-1, keepdim=True)
    nbr_safe = torch.where(nbr_norm_sq > 1.0e-12, nbr_xy, ctr_safe)
    return torch.atan2(
        ctr_safe[..., 0] * nbr_safe[..., 1]
        - ctr_safe[..., 1] * nbr_safe[..., 0],
        (ctr_safe * nbr_safe).sum(dim=-1),
    )


def safe_angle_from_2d_vector(vec: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    """Angle of a 2D vector with zero-vector fallback to 0 radians."""
    xy = vec[..., :2]
    fallback = torch.zeros_like(xy)
    fallback[..., 0] = 1.0
    norm_sq = xy.pow(2).sum(dim=-1, keepdim=True)
    safe_xy = torch.where(norm_sq > eps, xy, fallback)
    return torch.atan2(safe_xy[..., 1], safe_xy[..., 0])


def safe_norm_2d(vec: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    """L2 norm of a 2D vector with a NaN-safe backward at zero input.

    `torch.norm(x, p=2)` has backward `x / ||x||` which becomes ``0 / 0`` at
    a zero input vector and produces NaN gradients. ``(sum(x^2) + eps).sqrt()``
    keeps the denominator strictly positive in the backward pass, so the
    gradient at the origin is finite (zero in the limit).
    """
    return (vec.pow(2).sum(dim=-1) + eps).sqrt()


def wrap_angle(
    angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi
) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)
