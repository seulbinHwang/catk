from __future__ import annotations

from typing import Tuple

import math
import torch
from torch import Tensor


def central_diff(t: Tensor, pad_value: float) -> Tensor:
    """TF `trajectory_features.central_diff` port (difference along last axis)."""
    pad_shape = (*t.shape[:-1], 1)
    pad_tensor = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
    diff_t = (t[..., 2:] - t[..., :-2]) / 2.0
    return torch.cat([pad_tensor, diff_t, pad_tensor], dim=-1)


def central_logical_and(t: Tensor, pad_value: bool) -> Tensor:
    """TF `trajectory_features.central_logical_and` port."""
    if t.dtype != torch.bool:
        raise ValueError("central_logical_and expects bool tensor")
    pad_shape = (*t.shape[:-1], 1)
    pad_tensor = torch.full(pad_shape, pad_value, dtype=torch.bool, device=t.device)
    inner = t[..., 2:] & t[..., :-2]
    return torch.cat([pad_tensor, inner, pad_tensor], dim=-1)


def compute_displacement_error(
    x: Tensor, y: Tensor, z: Tensor, ref_x: Tensor, ref_y: Tensor, ref_z: Tensor
) -> Tensor:
    """TF `trajectory_features.compute_displacement_error` port."""
    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z
    # Match tf.linalg.norm euclidean: sqrt(sum(x^2)).
    return torch.sqrt(dx * dx + dy * dy + dz * dz)


def _wrap_angle(angle: Tensor) -> Tensor:
    """Wraps angles in range [-pi, pi], matching TF implementation."""
    two_pi = 2.0 * math.pi
    return torch.remainder(angle + math.pi, two_pi) - math.pi


def compute_kinematic_features(
    x: Tensor, y: Tensor, z: Tensor, heading: Tensor, seconds_per_step: float
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """TF `trajectory_features.compute_kinematic_features` port."""
    if seconds_per_step <= 0:
        raise ValueError("seconds_per_step must be positive")
    nan = float("nan")

    dpos = central_diff(torch.stack([x, y, z], dim=0), pad_value=nan)
    linear_speed = torch.sqrt((dpos * dpos).sum(dim=0)) / float(seconds_per_step)
    linear_accel = central_diff(linear_speed, pad_value=nan) / float(seconds_per_step)

    dh_step = _wrap_angle(central_diff(heading, pad_value=nan) * 2.0) / 2.0
    dh = dh_step / float(seconds_per_step)
    d2h_step = _wrap_angle(central_diff(dh_step, pad_value=nan) * 2.0) / 2.0
    d2h = d2h_step / (float(seconds_per_step) ** 2)
    return linear_speed, linear_accel, dh, d2h


def compute_kinematic_validity(valid: Tensor) -> Tuple[Tensor, Tensor]:
    """TF `trajectory_features.compute_kinematic_validity` port."""
    speed_validity = central_logical_and(valid, pad_value=False)
    acceleration_validity = central_logical_and(speed_validity, pad_value=False)
    return speed_validity, acceleration_validity


__all__ = [
    "central_diff",
    "central_logical_and",
    "compute_displacement_error",
    "compute_kinematic_features",
    "compute_kinematic_validity",
]

