"""Flow-RWR: Reward-Weighted Regression for Flow Matching trajectory models.

Objective (variational KL interpretation):
    Minimise KL( π*(y) || π_θ(y) )
    where  π*(y) ∝ π_old(y) · exp(R(y) / β)

In practice with G Monte-Carlo rollouts sampled from π_old:

    L_RWR = -Σ_g  w_g · log π_θ(y^g | s)
    where  w_g = softmax_g( R^g / β )   (normalised per scenario-group)

Using the FM ELBO proxy  log π_θ(y|s) ≈ -E_{t,x₀}[||v_θ(x_t,t,s)-(y-x₀)||²]:

    L_RWR = Σ_g  w_g · E_{t,x₀}[ ||v_θ(x_t,t,s^g) - (y^g-x₀)||² ]

This is a weighted flow-matching regression — no IS weighting, no reference
model, no clipping.  Simpler and often more stable than EPG.

Multi-anchor extension (AR factorisation):
    Each anchor step k ∈ {0, …, K-1} contributes an independent RWR loss
    with an optional temporal discount γ^k (default γ=1.0 = uniform).

    The reward is attached at the trajectory level (full RMM score) so the
    same weight w_g applies to every anchor.  Temporal discounting down-weights
    later anchors to account for compounding closed-loop error.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.smart.modules.flow_dpo import compute_fm_log_prob


def flow_rwr_loss(
    flow_decoder: nn.Module,
    anchor_hidden: Tensor,    # [n_valid, G, d_model]  or  [n_valid, d_model]
    trajectories: Tensor,     # [n_valid, G, 20, 4]  flow-norm space
    weights: Tensor,          # [n_valid, G]  reward weights (sum-to-1 per agent)
    flow_ode: nn.Module,
    n_samples: int = 8,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute Flow-RWR loss for one anchor step.

    A single batched forward pass handles all G rollouts at once by flattening
    [n_valid, G] → [n_valid*G].

    Args:
        flow_decoder: trainable flow decoder (π_θ).
        anchor_hidden: context encoding.
            - ``[n_valid, d_model]`` → broadcast across G.
            - ``[n_valid, G, d_model]`` → per-rollout context (closed-loop hidden).
        trajectories: G rollout trajectories in flow-normalised local frame.
            ``[n_valid, G, 20, 4]``.
        weights: per-agent, per-rollout importance weights (sum to 1 along G).
            ``[n_valid, G]``.  Typically ``softmax(R/β, dim=-1)``.
        flow_ode: provides ``flow_ode.eps``.
        n_samples: MC samples for FM ELBO estimation.

    Returns:
        Tuple of (scalar loss, metrics dict).
    """
    n_valid, G = trajectories.shape[:2]

    # Flatten [n_valid, G, ...] → [n_valid*G, ...]
    traj_flat = trajectories.reshape(n_valid * G, 20, 4)
    if anchor_hidden.dim() == 2:
        anchor_rep = anchor_hidden.unsqueeze(1).expand(-1, G, -1).reshape(n_valid * G, -1)
    elif anchor_hidden.dim() == 3:
        anchor_rep = anchor_hidden.reshape(n_valid * G, -1)
    else:
        raise ValueError(
            f"anchor_hidden must be 2D or 3D, got shape={tuple(anchor_hidden.shape)}"
        )
    weights_flat = weights.reshape(n_valid * G)  # [n_valid*G], detach is caller's responsibility

    # log π_θ(y^g | s): gradient flows through here
    # shape [n_valid*G]
    log_p = compute_fm_log_prob(
        flow_decoder, anchor_rep, traj_flat, flow_ode, n_samples=n_samples
    )  # [n_valid*G]

    # L = -Σ_g w_g · log π_θ(y^g)  (minimise → maximise weighted log-prob)
    loss = -(weights_flat * log_p).sum() / max(n_valid, 1)

    # Preference accuracy: fraction where argmax(log_p) == argmax(weight)
    with torch.no_grad():
        log_p_g = log_p.reshape(n_valid, G)
        w_g = weights.reshape(n_valid, G)
        pref_acc = (log_p_g.argmax(dim=-1) == w_g.argmax(dim=-1)).float().mean()

    metrics: Dict[str, Tensor] = {
        "train/rwr_loss":      loss.detach(),
        "train/rwr_log_p":     log_p.mean().detach(),
        "train/rwr_weight_entropy": -(weights * (weights + 1e-12).log()).sum(dim=-1).mean().detach(),
        "train/rwr_pref_acc":  pref_acc,
    }
    return loss, metrics
