"""
Per-channel-group MMD-based OL/CL consistency loss.

문제: 기존 mean_g(L2(CL_g, OL_g)) = cross-term only
  E[||CL-OL||²] = Var(CL) + Var(OL) + ||μ_CL - μ_OL||²
  → Var 항도 최소화 → mode collapse

해결: 올바른 MMD²(P_CL, P_OL)
  = E[k(CL,CL')] + E[k(OL,OL')] - 2E[k(CL,OL)]
  ≈ (1/σ²)||μ_CL - μ_OL||²  (Taylor 전개 시 Var 항 상쇄)
  → 분포 평균만 정렬, variance는 건드리지 않음

채널 분리: input tensor의 마지막 차원은 [x/20, y/20, cos_h, sin_h] (C=4).
  position group (ch[0:2]) 과 heading group (ch[2:4]) 의 RBF kernel 을 별도로
  계산하고 가중합. group 별로 sigma 가 따로 잡혀서, scale 큰 position 항이
  median-heuristic bandwidth 를 dominate 해 heading channel supervision 이
  saturate 되는 문제를 방지.
"""

import math
import torch
from torch import Tensor


_POS_SLICE = slice(0, 2)        # [x/20, y/20]
_HEADING_SLICE = slice(2, 4)    # [cos_h, sin_h]


def _global_sigma_sq(ol_flat: Tensor, cl_flat: Tensor, max_pts: int = 512) -> Tensor:
    """Median heuristic bandwidth. 두 텐서 모두 detach 후 사용."""
    with torch.no_grad():
        combined = torch.cat([
            ol_flat.reshape(-1, ol_flat.shape[-1]),
            cl_flat.reshape(-1, cl_flat.shape[-1]),
        ], dim=0)
        n_total = combined.shape[0]
        if n_total > max_pts:
            idx = torch.randperm(n_total, device=combined.device)[:max_pts]
            combined = combined[idx]
        n = combined.shape[0]
        dists_sq = torch.cdist(combined, combined).pow(2)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=combined.device), diagonal=1)
        sigma_sq = (dists_sq[mask].median() / math.log(n + 1)).clamp(min=1e-6)
    return sigma_sq


def _rbf_batch(a: Tensor, b: Tensor, sigma_sq: Tensor) -> Tensor:
    """[A, N, d] × [A, M, d] → [A, N, M] Gaussian kernel."""
    a_sq = a.pow(2).sum(-1, keepdim=True)
    b_sq = b.pow(2).sum(-1, keepdim=True)
    ab = torch.bmm(a, b.transpose(-2, -1))
    dist_sq = (a_sq + b_sq.transpose(-2, -1) - 2 * ab).clamp(min=0)
    return torch.exp(-dist_sq / (2 * sigma_sq))


def _group_mmd(cl_flat: Tensor, ol_flat: Tensor) -> Tensor:
    """Single-group biased MMD². cl_flat/ol_flat: [A, G, d]."""
    sigma_sq = _global_sigma_sq(ol_flat.detach(), cl_flat.detach())
    kcc = _rbf_batch(cl_flat, cl_flat, sigma_sq)
    koo = _rbf_batch(ol_flat, ol_flat, sigma_sq)
    kco = _rbf_batch(cl_flat, ol_flat, sigma_sq)
    per_agent_mmd = (
        kcc.mean((-2, -1)) + koo.mean((-2, -1)) - 2 * kco.mean((-2, -1))
    ).clamp(min=0)
    return per_agent_mmd.mean()


def _flat_for_group(stack: Tensor, ch_slice: slice) -> Tensor:
    """[G, n, T, C] → [n, G, T*|ch|] for the given channel slice."""
    G, n, T, _ = stack.shape
    sub = stack[..., ch_slice]                                         # [G, n, T, |ch|]
    return sub.permute(1, 0, 2, 3).reshape(n, G, -1)                   # [n, G, T*|ch|]


def mmd_from_stacked(
    cl_stack: Tensor,                          # [G, n_active, T, C]
    ol_stack: Tensor,                          # [G, n_active, T, C], detached
    pos_weight: float = 1.0,
    heading_weight: float = 0.0,
) -> Tensor:
    """Weighted sum of per-group (pos / heading) biased MMD².

    Args:
        cl_stack: closed-loop samples with gradient.
        ol_stack: open-loop / GT samples, detached.
        pos_weight: weight for position-channel MMD (ch[0:2]).
        heading_weight: weight for heading-channel MMD (ch[2:4]).

    Channel layout assumes [x/20, y/20, cos_h, sin_h]. Each group has its
    own median-heuristic bandwidth so position scale does not dominate the
    heading channel supervision.
    """
    G, n_active, T, C = cl_stack.shape
    assert C == 4, f"expected 4 channels [x/20, y/20, cos_h, sin_h], got C={C}"

    total = cl_stack.new_zeros(())
    if pos_weight > 0.0:
        total = total + pos_weight * _group_mmd(
            _flat_for_group(cl_stack, _POS_SLICE),
            _flat_for_group(ol_stack, _POS_SLICE),
        )
    if heading_weight > 0.0:
        total = total + heading_weight * _group_mmd(
            _flat_for_group(cl_stack, _HEADING_SLICE),
            _flat_for_group(ol_stack, _HEADING_SLICE),
        )
    return total


def mmd_precompute_sigma_sq(
    ol_norms: list,                            # G × Tensor[n, T, C], all detached
    cl_norms: list,                            # G × Tensor[n, T, C], all detached
    pos_weight: float = 1.0,
    heading_weight: float = 0.0,
) -> dict:
    """Median-heuristic bandwidth per active channel group.

    Returns dict with keys "pos" / "heading" (only for groups with weight > 0).
    Used by sequential MMD pass-2 to share bandwidth between log/proxy.
    """
    G = len(ol_norms)
    n, T, C = ol_norms[0].shape
    assert C == 4, f"expected 4 channels [x/20, y/20, cos_h, sin_h], got C={C}"

    out: dict = {}
    for name, ch_slice, w in (
        ("pos", _POS_SLICE, pos_weight),
        ("heading", _HEADING_SLICE, heading_weight),
    ):
        if w <= 0.0:
            continue
        with torch.no_grad():
            ol_flat = torch.stack([o[..., ch_slice] for o in ol_norms], dim=0).reshape(G * n, -1)
            cl_flat = torch.stack([c[..., ch_slice] for c in cl_norms], dim=0).reshape(G * n, -1)
        out[name] = _global_sigma_sq(ol_flat, cl_flat)
    return out


def mmd_per_rollout_proxy(
    cl_norm_g: Tensor,                         # [n, T, C] with gradient
    cl_norms_ref: list,                        # G × Tensor[n, T, C], all detached
    ol_norms_ref: list,                        # G × Tensor[n, T, C], all detached
    sigma_sqs: dict,                           # output of mmd_precompute_sigma_sq
    pos_weight: float = 1.0,
    heading_weight: float = 0.0,
) -> Tensor:
    """Per-rollout proxy loss for sequential MMD backward (per-group).

    Gradient identity (per group):
        ∂proxy_g/∂cl_g == ∂mmd_from_stacked(cl_stack, ol_stack)/∂cl_g
    because detaching cl_j (j≠g) does not affect the gradient w.r.t. cl_g.

    Usage: call (proxy_g / n_anchors).backward() for each rollout g.
    Summing over all g gives ∂(mean_anchor weighted-MMD²)/∂θ exactly.
    """
    G = len(cl_norms_ref)
    n, T, C = cl_norm_g.shape
    assert C == 4, f"expected 4 channels [x/20, y/20, cos_h, sin_h], got C={C}"

    def _proxy_one_group(ch_slice: slice, sigma_sq: Tensor) -> Tensor:
        d = T * (ch_slice.stop - ch_slice.start)
        cl_g_flat = cl_norm_g[..., ch_slice].reshape(n, 1, d)
        cl_ref_flat = (
            torch.stack([r[..., ch_slice] for r in cl_norms_ref], dim=0)
            .reshape(G, n, d)
            .permute(1, 0, 2)
        )
        ol_ref_flat = (
            torch.stack([r[..., ch_slice] for r in ol_norms_ref], dim=0)
            .reshape(G, n, d)
            .permute(1, 0, 2)
        )
        kcc = _rbf_batch(cl_g_flat, cl_ref_flat, sigma_sq)
        kco = _rbf_batch(cl_g_flat, ol_ref_flat, sigma_sq)
        per_agent = kcc.mean(-1).squeeze(1) - kco.mean(-1).squeeze(1)
        return (2.0 / G) * per_agent.mean()

    total = cl_norm_g.new_zeros(())
    if pos_weight > 0.0 and "pos" in sigma_sqs:
        total = total + pos_weight * _proxy_one_group(_POS_SLICE, sigma_sqs["pos"])
    if heading_weight > 0.0 and "heading" in sigma_sqs:
        total = total + heading_weight * _proxy_one_group(_HEADING_SLICE, sigma_sqs["heading"])
    return total
