"""Flow-EPG: Exact Policy Gradient for Flow Matching trajectory models.

Based on: "Rethinking the Design Space of RL for Diffusion Models" (Choi et al., 2026)
  arXiv: 2602.04663

ELBO likelihood surrogate (w(t) = 1, simple weighting):
    log π_θ(y | s) ≈ -E_{t~U[ε,1], x₀~N(0,I)}[ ||v_θ(x_t, t, s) - (y - x₀)||² ]
    where x_t = (1-t)·x₀ + t·y

EPG objective (Eq. 3):
    L_EPG = -E_{y^i ~ π_{θ_old}}[ sg(ρ_θ)(y^i) · A^i · log π_θ(y^i) ]
            + β · E[ KL(π_θ || π_ref) ]

where:
    A^i     = R^i - mean(R¹,...,Rᴳ)               no std normalisation (EPG design choice)
    sg(ρ_θ) = sg(π_θ(y)) / π_{θ_old}(y)           stop-gradient on numerator
             = exp( sg(log π_θ(y)) - log π_{θ_old}(y) )
    KL      = log π_θ(y^i) - log π_ref(y^i)       per-sample, both via ELBO

Three distinct models:
    π_{θ_old} : policy that generated the rollouts  → log_p_old (no_grad, same weights)
    π_θ       : current trainable policy             → log_p_theta (grad flows here)
    π_ref     : frozen pretrained reference          → log_p_ref (no_grad, separate weights)

Variance reduction for ρ_θ:
    log_p_old and log_p_theta share the same MC samples (t, x₀) so that
    the ratio ρ_θ = exp(log_p_theta - log_p_old) cancels most Monte-Carlo noise.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.smart.modules.flow_dpo import compute_fm_log_prob


def flow_epg_loss(
    flow_decoder: nn.Module,
    ref_flow_decoder: Optional[nn.Module],  # π_ref — frozen pretrained, None → KL-free
    anchor_hidden: Tensor,   # [n_valid, d_model] or [n_valid, G, d_model]
    trajectories: Tensor,    # [n_valid, G, 20, 4]  flow-norm space
    advantages: Tensor,      # [n_valid, G]          per-agent, per-rollout
    log_p_old: Tensor,       # [n_valid, G]          log π_{θ_old}, pre-computed with same shared MC samples
    shared_t: Tensor,        # [K, n_valid*G]        same MC time steps used for log_p_old
    shared_x0: Tensor,       # [n_valid*G, 20, 4]    same MC noise used for log_p_old
    flow_ode: nn.Module,
    beta: float = 0.1,
    n_samples: int = 8,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute Flow-EPG loss.

    Args:
        flow_decoder: trainable flow decoder (π_θ).
        ref_flow_decoder: frozen reference model (π_ref). None → skip KL.
        anchor_hidden: anchor context.
            ``[n_valid, d_model]``이면 rollout 축 G로 broadcast하고,
            ``[n_valid, G, d_model]``이면 rollout별 condition을 직접 사용합니다.
        trajectories: G rollout trajectories in flow-normalised local frame.
            ``[n_valid, G, 20, 4]``.
        advantages: group-normalised advantages A^i = R^i - mean(R).
            ``[n_valid, G]``. No std normalisation (EPG paper Eq. 3).
        log_p_old: log π_{θ_old}(y^i) computed with no_grad at rollout time
            using the SAME shared_t and shared_x0. ``[n_valid, G]``.
        shared_t: MC time samples ``[K, n_valid*G]``, shared with log_p_old
            so that ρ_θ = exp(log_p_theta - log_p_old) has low MC variance.
        shared_x0: MC noise samples ``[n_valid*G, 20, 4]``, shared with log_p_old.
        flow_ode: provides ``flow_ode.eps``.
        beta: KL regularisation weight β.
        n_samples: MC samples (must match K = shared_t.shape[0]).

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
    log_p_old_flat = log_p_old.reshape(n_valid * G)   # [n_valid*G], constant (no grad)

    # ── π_θ: current policy, gradient flows through ───────────────────────────
    # Use the SAME shared_t and shared_x0 as log_p_old → ρ_θ noise cancels
    log_p_theta = compute_fm_log_prob(
        flow_decoder, anchor_rep, traj_flat, flow_ode,
        n_samples=n_samples, shared_t=shared_t, shared_x0=shared_x0,
    )  # [n_valid*G]

    # ── sg(ρ_θ) = exp( sg(log π_θ) - log π_{θ_old} ) ─────────────────────────
    # stop-gradient on numerator: gradient only passes through log π_θ in PG term.
    # Clamp to [0, 10] for numerical stability (similar to PPO clip but on ratio itself).
    rho_sg = (log_p_theta.detach() - log_p_old_flat).exp().clamp(max=10.0)  # [n_valid*G]

    # ── π_ref: frozen reference, for KL term ──────────────────────────────────
    # KL(π_θ || π_ref) ≈ E_{y~π_old}[ρ_θ(y) · (log π_θ(y) - log π_ref(y))]
    # IS weight ρ_θ (stop-grad) corrects for distribution shift π_old→π_θ.
    if ref_flow_decoder is not None:
        with torch.no_grad():
            log_p_ref = compute_fm_log_prob(
                ref_flow_decoder,
                anchor_rep.detach(), traj_flat.detach(), flow_ode,
                n_samples=n_samples, shared_t=shared_t, shared_x0=shared_x0,
            )  # [n_valid*G], no grad
        # IS-weighted KL: rho_sg reweights samples from π_old to π_θ
        kl = rho_sg * (log_p_theta - log_p_ref)   # [n_valid*G], grad via log_p_theta
    else:
        kl = torch.zeros_like(log_p_theta)
        log_p_ref = None

    adv_flat = advantages.reshape(n_valid * G).detach()  # [n_valid*G], no grad

    # ── EPG loss: L = -mean(sg(ρ_θ) · A · log π_θ) + β · mean(KL) ───────────
    pg_loss = -(rho_sg * adv_flat * log_p_theta).mean()
    loss = pg_loss + beta * kl.mean()

    # Preference accuracy: fraction where argmax(log_p_theta) == argmax(advantage)
    with torch.no_grad():
        log_p_g = log_p_theta.reshape(n_valid, G)
        pref_acc = (log_p_g.argmax(dim=-1) == advantages.argmax(dim=-1)).float().mean()
        rho_mean = rho_sg.mean()
        rho_max = rho_sg.max()

    metrics: Dict[str, Tensor] = {
        "train/epg_loss": loss.detach(),
        "train/epg_pg_loss": pg_loss.detach(),
        "train/epg_kl": kl.mean().detach(),
        "train/epg_log_p_theta": log_p_theta.mean().detach(),
        "train/epg_log_p_old": log_p_old_flat.mean().detach(),
        "train/epg_rho_mean": rho_mean,
        "train/epg_rho_max": rho_max,
        "train/epg_advantage_mean": adv_flat.mean().detach(),
        "train/epg_advantage_std": adv_flat.std().detach() if adv_flat.numel() > 1 else adv_flat.new_zeros(1),
        "train/epg_pref_acc": pref_acc,
    }
    if ref_flow_decoder is not None:
        metrics["train/epg_log_p_ref"] = log_p_ref.mean().detach()

    return loss, metrics
