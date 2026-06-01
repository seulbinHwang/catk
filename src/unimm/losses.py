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
    von Mises distributions. The trajectory likelihood is reduced by summing
    valid timesteps within each agent/context row, then averaging valid rows.
    """

    per_step, weights = _unimm_per_step_nll(pred, target_local, target_valid)
    per_row = (per_step * weights).sum(dim=-1)
    valid_row = weights.sum(dim=-1) > 0
    if not bool(valid_row.any()):
        return per_step.sum() * 0.0
    return per_row[valid_row].mean()


def unimm_per_step_nll_loss(
    pred: Dict[str, Tensor],
    target_local: Tensor,
    target_valid: Tensor,
) -> Tensor:
    """Mean per-valid-timestep NLL for logging the old loss scale."""

    per_step, weights = _unimm_per_step_nll(pred, target_local, target_valid)
    return (per_step * weights).sum() / weights.sum().clamp_min(1.0)


def _unimm_per_step_nll(
    pred: Dict[str, Tensor],
    target_local: Tensor,
    target_valid: Tensor,
) -> tuple[Tensor, Tensor]:
    pos_dist = Laplace(pred["mean_pos"], pred["pos_scale"])
    pos_nll = -pos_dist.log_prob(target_local[..., :2]).sum(dim=-1)

    head_dist = VonMises(pred["mean_head"], pred["head_concentration"])
    head_nll = -head_dist.log_prob(target_local[..., 2])

    per_step = pos_nll + head_nll
    weights = target_valid.to(dtype=per_step.dtype)
    return per_step, weights


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
