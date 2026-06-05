from __future__ import annotations

import math
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

    while target_local.dim() < mean_pos.dim():
        target_local = target_local.unsqueeze(-3)

    pos_dist = Laplace(mean_pos, pos_scale)
    pos_nll = -pos_dist.log_prob(target_local[..., :2]).sum(dim=-1)

    head_dist = VonMises(mean_head, head_concentration)
    head_nll = -head_dist.log_prob(target_local[..., 2])

    per_step = pos_nll + head_nll
    weights = target_valid.to(dtype=per_step.dtype)
    while weights.dim() < per_step.dim():
        weights = weights.unsqueeze(-2)
    weights = weights.expand_as(per_step)
    return per_step, weights


def unimm_top_m_mixture_nll_loss(
    pred: Dict[str, Tensor],
    logits: Tensor,
    z_candidates: Tensor,
    target_local: Tensor,
    target_valid: Tensor,
    front_weight_steps: int = 5,
    front_weight: float = 4.0,
) -> tuple[Tensor, Tensor]:
    """Top-M anchor mixture NLL.

    Only candidate anchors are decoded. The scorer probability for each
    candidate is combined with that candidate's continuous trajectory
    likelihood through log-sum-exp. The trajectory NLL is a normalized
    time-weighted mean, emphasizing the committed 0.5s prefix without changing
    the overall optimization scale.
    """

    per_step, weights = _unimm_per_step_nll(pred, target_local, target_valid)
    weights = _apply_front_horizon_weights(
        weights,
        front_weight_steps=front_weight_steps,
        front_weight=front_weight,
    )
    denom = weights.sum(dim=-1).clamp_min(1.0)
    candidate_nll = (per_step * weights).sum(dim=-1) / denom

    row_valid = target_valid.any(dim=-1)
    if not bool(row_valid.any()):
        return candidate_nll.sum() * 0.0, candidate_nll.detach()

    log_probs = F.log_softmax(logits.float(), dim=-1)
    candidate_log_probs = log_probs.gather(-1, z_candidates.long())
    mixture_log_prob = torch.logsumexp(candidate_log_probs - candidate_nll.float(), dim=-1)
    return -mixture_log_prob[row_valid].mean(), candidate_nll.detach()


def _apply_front_horizon_weights(
    weights: Tensor,
    front_weight_steps: int,
    front_weight: float,
) -> Tensor:
    front_weight_steps = int(front_weight_steps)
    front_weight = float(front_weight)
    if front_weight_steps <= 0 or front_weight == 1.0:
        return weights
    if front_weight <= 0.0:
        raise ValueError(f"front_weight must be positive, got {front_weight}.")
    time_weight = torch.ones(weights.shape[-1], dtype=weights.dtype, device=weights.device)
    time_weight[: min(front_weight_steps, weights.shape[-1])] = front_weight
    view_shape = (1,) * (weights.dim() - 1) + (weights.shape[-1],)
    return weights * time_weight.view(view_shape)


def unimm_soft_anchor_ce_loss(
    logits: Tensor,
    z_candidates: Tensor,
    candidate_error: Tensor,
    valid: Tensor,
    agent_type: Tensor | None = None,
    match_steps: int | None = None,
    min_temperature: float = 1.0e-4,
) -> tuple[Tensor, Dict[str, Tensor]]:
    """Auxiliary soft CE over the top-M positive-matching candidates.

    Soft targets are built only from the 0.5s positive-matching distances
    already computed by the processor. A category-specific temperature is
    estimated from the valid rows in the current batch as
    median(second_best - best) / log(2), so an average near-tie gives the
    second candidate about half the unnormalized target mass of the best one.
    """

    if match_steps is not None:
        valid = valid[..., : int(match_steps)]
    valid_row = valid.any(dim=-1)
    if not bool(valid_row.any()):
        zero = logits.sum() * 0.0
        return zero, {
            "soft_anchor_entropy": zero.detach(),
            "soft_anchor_top1_prob": zero.detach(),
            "soft_anchor_temperature": zero.detach(),
        }

    candidate_error = candidate_error.float()
    if candidate_error.shape != z_candidates.shape:
        raise ValueError(
            "candidate_error must match z_candidates shape, "
            f"got {tuple(candidate_error.shape)} and {tuple(z_candidates.shape)}"
        )
    if candidate_error.shape[:-1] != valid_row.shape:
        raise ValueError(
            "candidate_error rows must match valid rows, "
            f"got {tuple(candidate_error.shape)} and {tuple(valid_row.shape)}"
        )
    finite_fill = torch.finfo(candidate_error.dtype).max / 4.0
    candidate_error = candidate_error.masked_fill(~torch.isfinite(candidate_error), finite_fill)

    log_probs = F.log_softmax(logits.float(), dim=-1)
    candidate_log_probs = log_probs.gather(-1, z_candidates.long())
    tau = _soft_anchor_temperature(
        candidate_error=candidate_error,
        valid_row=valid_row,
        agent_type=agent_type,
        min_temperature=min_temperature,
    )
    shifted_error = candidate_error - candidate_error.min(dim=-1, keepdim=True).values
    target = torch.softmax(-shifted_error / tau.unsqueeze(-1), dim=-1)
    loss_per_row = -(target.detach() * candidate_log_probs).sum(dim=-1)
    loss = loss_per_row[valid_row].mean()

    target_valid = target[valid_row].detach()
    tau_valid = tau[valid_row].detach()
    entropy = -(target_valid * target_valid.clamp_min(1.0e-12).log()).sum(dim=-1).mean()
    stats = {
        "soft_anchor_entropy": entropy,
        "soft_anchor_top1_prob": target_valid.max(dim=-1).values.mean(),
        "soft_anchor_temperature": tau_valid.mean(),
    }
    return loss, stats


def _soft_anchor_temperature(
    candidate_error: Tensor,
    valid_row: Tensor,
    agent_type: Tensor | None,
    min_temperature: float,
) -> Tensor:
    min_temperature = float(min_temperature)
    if min_temperature <= 0.0:
        raise ValueError(f"min_temperature must be positive, got {min_temperature}.")

    num_candidates = int(candidate_error.shape[-1])
    if num_candidates < 2:
        return torch.full_like(candidate_error[..., 0], min_temperature)

    top2 = torch.topk(candidate_error, k=2, dim=-1, largest=False, sorted=True).values
    gap = (top2[..., 1] - top2[..., 0]).clamp_min(0.0)
    finite_positive = torch.isfinite(gap) & (gap > 0.0) & valid_row
    if bool(finite_positive.any()):
        fallback_tau = gap[finite_positive].median() / math.log(2.0)
    else:
        fallback_tau = gap.new_tensor(min_temperature)
    fallback_tau = fallback_tau.clamp_min(min_temperature)

    tau = torch.full_like(gap, fallback_tau)
    if agent_type is None:
        return tau

    row_type = agent_type.long()
    if row_type.shape != valid_row.shape:
        if row_type.dim() == 1 and valid_row.dim() == 2 and row_type.shape[0] == valid_row.shape[0]:
            row_type = row_type[:, None].expand_as(valid_row)
        else:
            raise ValueError(
                "agent_type must be shaped like valid rows or [N_agent], "
                f"got {tuple(row_type.shape)} and {tuple(valid_row.shape)}"
            )
    for type_idx in range(3):
        type_mask = finite_positive & (row_type == type_idx)
        if bool(type_mask.any()):
            tau_type = (gap[type_mask].median() / math.log(2.0)).clamp_min(min_temperature)
        else:
            tau_type = fallback_tau
        tau = torch.where(row_type == type_idx, tau_type.to(dtype=tau.dtype, device=tau.device), tau)
    return tau.clamp_min(min_temperature)
