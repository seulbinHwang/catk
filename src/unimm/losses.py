from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Laplace, VonMises


def unimm_nll_loss(
    pred: Dict[str, Tensor],
    target_local: Tensor,
    target_valid: Tensor,
) -> Tensor:
    """Continuous UniMM regression NLL.

    Position uses independent Laplace distributions. Heading uses independent
    von Mises distributions. The final scalar is averaged over valid timesteps.
    """

    pos_dist = Laplace(pred["mean_pos"], pred["pos_scale"])
    pos_nll = -pos_dist.log_prob(target_local[..., :2]).sum(dim=-1)

    head_dist = VonMises(pred["mean_head"], pred["head_concentration"])
    head_nll = -head_dist.log_prob(target_local[..., 2])

    per_step = pos_nll + head_nll
    weights = target_valid.to(dtype=per_step.dtype)
    return (per_step * weights).sum() / weights.sum().clamp_min(1.0)


def unimm_classification_loss(
    logits: Tensor,
    z_star: Tensor,
    valid: Tensor,
    match_steps: int | None = None,
) -> Tensor:
    """Hard-assignment anchor classification loss."""

    if match_steps is not None:
        valid = valid[..., : int(match_steps)]
    valid_row = valid.any(dim=-1)
    if not bool(valid_row.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid_row], z_star[valid_row])
