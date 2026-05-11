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
    return torch.atan2(
        ctr_vector[..., 0] * nbr_vector[..., 1]
        - ctr_vector[..., 1] * nbr_vector[..., 0],
        (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1),
    )


def wrap_angle(
    angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi
) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)


def safe_angle_from_2d_vector(vec: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    """Angle of a 2D vector with zero-vector fallback to 0 radians.

    V2 backbone 모듈 (LQR commit / NaN-safe encoder) 에서 사용되는 헬퍼.
    V1 코드 경로에서는 사용하지 않음.
    """
    xy = vec[..., :2]
    fallback = torch.zeros_like(xy)
    fallback[..., 0] = 1.0
    norm_sq = xy.pow(2).sum(dim=-1, keepdim=True)
    safe_xy = torch.where(norm_sq > eps, xy, fallback)
    return torch.atan2(safe_xy[..., 1], safe_xy[..., 0])


def safe_norm_2d(vec: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    """L2 norm of a 2D vector with a NaN-safe backward at zero input.

    `torch.norm(x, p=2)` 의 backward 가 ``x / ||x||`` 라서 zero input 에서 NaN
    gradient 가 나옴. ``(sum(x^2) + eps).sqrt()`` 형태는 분모가 항상 양수라
    backward 가 finite (origin 에서 0) 임.

    V2 backbone 모듈 전용 헬퍼.
    """
    return (vec.pow(2).sum(dim=-1) + eps).sqrt()
