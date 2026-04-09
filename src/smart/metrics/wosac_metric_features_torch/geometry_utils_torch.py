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
    # Important for autograd: `torch.where(mask, a, b)` still evaluates `a` and `b`
    # before selecting, so `num/den` can produce NaNs/Infs and poison gradients even
    # where mask is False. Use masked division to avoid evaluating invalid divisions.
    # Make division safe by replacing 0 denominators with 1, then zeroing outputs.
    # This avoids producing NaNs/Infs in forward/backward while still matching
    # divide_no_nan semantics.
    zero = torch.zeros_like(num)
    den_safe = torch.where(den != 0, den, torch.ones_like(den))
    out = num / den_safe
    return torch.where(den != 0, out, zero)


__all__ = ["dot_product_2d", "cross_product_2d", "divide_no_nan"]

