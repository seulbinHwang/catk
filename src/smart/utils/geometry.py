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
    # If `nbr_vector` is exactly the zero vector (e.g. a stationary agent
    # produces zero displacement, two agents are at the same position, or
    # an agent overlaps a map node), both `cross` and `dot` are zero and
    # `atan2(0, 0)` has an undefined backward (`1/(y^2+x^2) -> 1/0 -> NaN`).
    # That NaN silently flows back into encoder weight gradients during
    # the self-forced rollout. Add a tiny x-axis bias to the neighbor
    # vector so the atan2 inputs are never simultaneously zero. The bias
    # (~1e-6) shifts the computed angle by at most ~1e-6 rad relative to a
    # well-separated input and is negligible for training.
    nbr_safe = nbr_vector + nbr_vector.new_tensor([1.0e-6, 0.0])
    return torch.atan2(
        ctr_vector[..., 0] * nbr_safe[..., 1]
        - ctr_vector[..., 1] * nbr_safe[..., 0],
        (ctr_vector[..., :2] * nbr_safe[..., :2]).sum(dim=-1),
    )


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
