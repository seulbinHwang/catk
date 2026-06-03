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
    von Mises distributions. The loss is averaged over valid timesteps so the
    regression term stays on the same optimization scale as anchor
    classification, including late rollout contexts with partial future labels.
    """

    per_step, weights = _unimm_per_step_nll(pred, target_local, target_valid)
    denom = weights.sum()
    if not bool(denom > 0):
        return per_step.sum() * 0.0
    return (per_step * weights).sum() / denom


def _unimm_per_step_nll(
    pred: Dict[str, Tensor],
    target_local: Tensor,
    target_valid: Tensor,
) -> tuple[Tensor, Tensor]:
    target_local = target_local.float()
    mean_pos = pred["mean_pos"].float()
    pos_scale = pred["pos_scale"].float().clamp_min(1.0e-6)
    mean_head = pred["mean_head"].float()
    head_concentration = pred["head_concentration"].float().clamp(1.0e-6, 100.0)

    pos_dist = Laplace(mean_pos, pos_scale)
    pos_nll = -pos_dist.log_prob(target_local[..., :2]).sum(dim=-1)

    head_dist = VonMises(mean_head, head_concentration)
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
