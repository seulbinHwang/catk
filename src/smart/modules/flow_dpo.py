"""Flow-DPO: Direct Preference Optimization for Flow Matching trajectory models.

FM log-probability proxy (OT-Flow Matching):
    log π_θ(y | s) ≈ -E_{t,x₀}[ ||v_θ(x_t, t, s) - (y - x₀)||² ]

where x_t = (1-t)·x₀ + t·y  (OT linear interpolation).

DPO loss (with reference model):
    L_DPO = -log σ( β · [ (log π_θ(y_w|s) - log π_ref(y_w|s))
                          - (log π_θ(y_l|s) - log π_ref(y_l|s)) ] )

y_w = GT (winner), y_l = policy rollout (loser).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def compute_fm_log_prob(
    flow_decoder: nn.Module,
    anchor_hidden: Tensor,         # [n, d_model]
    trajectory: Tensor,            # [n, 20, 4]  — the trajectory to evaluate
    flow_ode: nn.Module,           # for flow_ode.eps
    n_samples: int = 8,
    shared_x0: Optional[Tensor] = None,  # [n, 20, 4] shared noise (reduces variance)
    shared_t: Optional[Tensor] = None,   # [K, n] shared time steps (reduces variance for ρ_θ)
    max_forward_batch: int = 256,  # max [K*n] per forward to avoid CUDA SDPA kernel limits
) -> Tensor:                       # [n]
    """Compute the FM log-probability proxy for a batch of trajectories.

    Processes MC samples in chunks of at most ``max_forward_batch`` to avoid
    CUDA SDPA kernel failures caused by very large batch dimensions (the
    ChunkStepRefiner reshapes to [batch*4, 5, dim] before attention, which
    exceeds CUDA block-config limits beyond ~1024).

    Higher return value = model assigns higher probability to ``trajectory``.

    Args:
        flow_decoder: the flow decoder whose parameters we are training/evaluating.
        anchor_hidden: context encoding ``[n, d_model]``.
        trajectory: the trajectory to evaluate (x₁). ``[n, 20, 4]``.
        flow_ode: provides ``flow_ode.eps`` (minimum time step).
        n_samples: number of (t, x₀) Monte-Carlo samples.
        shared_x0: if supplied, use this as the noise x₀ (same across calls
            reduces variance when computing ratios like ρ_θ).
        shared_t: if supplied, use this as the time samples ``[K, n]`` (same across
            calls reduces variance for ρ_θ = π_θ / π_{θ_old} computation).
        max_forward_batch: maximum K*n size per forward pass. Samples are chunked
            so that no single call exceeds this limit.

    Returns:
        Tensor: per-sample log-prob proxy ``[n]``.
    """
    n = trajectory.shape[0]
    device = trajectory.device
    dtype = trajectory.dtype
    t0 = float(flow_ode.eps)

    K = n_samples

    # t ∈ [eps, 1], shape [K, n]
    if shared_t is not None:
        t_k = shared_t.to(device=device, dtype=dtype)
    else:
        t_k = t0 + (1.0 - t0) * torch.rand(K, n, device=device, dtype=dtype)

    if shared_x0 is not None:
        x0_k = shared_x0.unsqueeze(0).expand(K, -1, -1, -1)
    else:
        x0_k = torch.randn(K, n, 20, 4, device=device, dtype=dtype)

    y_k = trajectory.unsqueeze(0).expand(K, -1, -1, -1)

    # OT interpolation: x_t = (1-t)*x0 + t*y
    t_exp = t_k.view(K, n, 1, 1)
    x_t_k = (1.0 - t_exp) * x0_k + t_exp * y_k  # [K, n, 20, 4]

    # Target vector field (detached — does not carry gradient)
    v_target_k = (y_k - x0_k).detach()  # [K, n, 20, 4]

    # Flatten to [K*n, ...]
    anchor_rep = anchor_hidden.unsqueeze(0).expand(K, -1, -1).reshape(K * n, -1)
    x_t_flat = x_t_k.reshape(K * n, 20, 4)
    tau_flat = t_k.reshape(K * n)
    v_target_flat = v_target_k.reshape(K * n, 20, 4)

    total = K * n
    if total <= max_forward_batch:
        # Single forward pass (common case for small batches)
        v_pred_flat = flow_decoder.forward_components(
            anchor_hidden=anchor_rep,
            x_t_norm=x_t_flat,
            tau=tau_flat,
        )["velocity"]
    else:
        # Chunked forward to keep SDPA batch within CUDA kernel limits
        chunks = []
        for start in range(0, total, max_forward_batch):
            end = min(start + max_forward_batch, total)
            v_chunk = flow_decoder.forward_components(
                anchor_hidden=anchor_rep[start:end],
                x_t_norm=x_t_flat[start:end],
                tau=tau_flat[start:end],
            )["velocity"]
            chunks.append(v_chunk)
        v_pred_flat = torch.cat(chunks, dim=0)

    # Negative MSE = log-prob proxy, averaged over (T=20, D=4) dims
    neg_mse_flat = -F.mse_loss(v_pred_flat, v_target_flat, reduction="none").mean(
        dim=(-2, -1)
    )  # [K*n]

    # Average over K samples → [n]
    return neg_mse_flat.reshape(K, n).mean(dim=0)


def flow_dpo_loss(
    flow_decoder: nn.Module,
    ref_flow_decoder: Optional[nn.Module],  # None → KL-free DPO
    anchor_hidden: Tensor,    # [n, d_model]
    y_w: Tensor,              # [n, 20, 4] winner (GT)
    y_l: Tensor,              # [n, 20, 4] loser  (policy rollout, detached)
    flow_ode: nn.Module,
    beta: float = 0.1,
    n_samples: int = 8,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute Flow-DPO loss.

    Evaluates both y_w and y_l in a single batched forward pass by concatenating
    them along the batch dimension.

    Args:
        flow_decoder: trainable flow decoder.
        ref_flow_decoder: frozen reference model (pretrained checkpoint).
            If ``None``, uses the KL-free (simplified) DPO variant.
        anchor_hidden: context encoding ``[n, d_model]``.
        y_w: winner trajectory (GT). ``[n, 20, 4]``.
        y_l: loser trajectory (policy rollout, already detached). ``[n, 20, 4]``.
        flow_ode: for ``flow_ode.eps``.
        beta: DPO temperature β.
        n_samples: MC samples for log-prob estimation.

    Returns:
        Tuple of (scalar loss, metrics dict).
    """
    n = anchor_hidden.shape[0]

    # Share noise between y_w and y_l to reduce MC variance in the comparison.
    shared_x0 = torch.randn(n, 20, 4, device=anchor_hidden.device, dtype=anchor_hidden.dtype)

    # Concatenate y_w and y_l for a single batched evaluation.
    y_both = torch.cat([y_w, y_l], dim=0)           # [2n, 20, 4]
    anchor_both = anchor_hidden.repeat(2, 1)          # [2n, d_model]
    x0_both = shared_x0.repeat(2, 1, 1)              # [2n, 20, 4]

    # θ-model log probs (gradient flows through flow_decoder)
    log_p_both = compute_fm_log_prob(
        flow_decoder, anchor_both, y_both, flow_ode,
        n_samples=n_samples, shared_x0=x0_both,
    )  # [2n]
    log_p_w_theta = log_p_both[:n]   # [n]
    log_p_l_theta = log_p_both[n:]   # [n]

    # Reference log probs (no gradient)
    if ref_flow_decoder is not None:
        with torch.no_grad():
            log_p_ref_both = compute_fm_log_prob(
                ref_flow_decoder, anchor_both.detach(), y_both.detach(), flow_ode,
                n_samples=n_samples, shared_x0=x0_both.detach(),
            )
        log_p_w_ref = log_p_ref_both[:n]
        log_p_l_ref = log_p_ref_both[n:]

        log_ratio_w = log_p_w_theta - log_p_w_ref
        log_ratio_l = log_p_l_theta - log_p_l_ref
    else:
        # KL-free: treat raw log probs as ratios (implicit uniform reference)
        log_ratio_w = log_p_w_theta
        log_ratio_l = log_p_l_theta

    # DPO loss: -log σ(β · (log_ratio_w - log_ratio_l))
    margin = beta * (log_ratio_w - log_ratio_l)  # [n]
    loss = -F.logsigmoid(margin).mean()

    # Preference accuracy: fraction of agents where model prefers y_w over y_l
    with torch.no_grad():
        pref_acc = (log_p_w_theta > log_p_l_theta).float().mean()

    metrics: Dict[str, Tensor] = {
        "train/dpo_loss": loss.detach(),
        "train/dpo_log_p_w": log_p_w_theta.mean().detach(),
        "train/dpo_log_p_l": log_p_l_theta.mean().detach(),
        "train/dpo_margin": margin.mean().detach(),
        "train/dpo_pref_acc": pref_acc,
    }
    if ref_flow_decoder is not None:
        metrics["train/dpo_log_ratio_w"] = log_ratio_w.mean().detach()
        metrics["train/dpo_log_ratio_l"] = log_ratio_l.mean().detach()

    return loss, metrics
