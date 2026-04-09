from __future__ import annotations

import torch
from torch import Tensor


def dot_product_2d(a: Tensor, b: Tensor) -> Tensor:
    """Waymo `geometry_utils.dot_product_2d` equivalent.

    Inputs: (..., 2)
    Output: (...,)
    """
    return (a[..., 0] * b[..., 0]) + (a[..., 1] * b[..., 1])


def cross_product_2d(a: Tensor, b: Tensor) -> Tensor:
    """Waymo `geometry_utils.cross_product_2d` equivalent (z-component of 2D cross)."""
    return (a[..., 0] * b[..., 1]) - (a[..., 1] * b[..., 0])


def divide_no_nan(num: Tensor, den: Tensor) -> Tensor:
    """TF `tf.math.divide_no_nan` equivalent."""
    return torch.where(den != 0, num / den, torch.zeros_like(num))


__all__ = ["dot_product_2d", "cross_product_2d", "divide_no_nan"]

