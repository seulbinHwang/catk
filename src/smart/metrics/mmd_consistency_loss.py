"""
Proper MMD-based OL/CL consistency loss.

문제: 기존 mean_g(L2(CL_g, OL_g)) = cross-term only
  E[||CL-OL||²] = Var(CL) + Var(OL) + ||μ_CL - μ_OL||²
  → Var 항도 최소화 → mode collapse

해결: 올바른 MMD²(P_CL, P_OL)
  = E[k(CL,CL')] + E[k(OL,OL')] - 2E[k(CL,OL)]
  ≈ (1/σ²)||μ_CL - μ_OL||²  (Taylor 전개 시 Var 항 상쇄)
  → 분포 평균만 정렬, variance는 건드리지 않음
"""

import math
import torch
from torch import Tensor


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


def mmd_from_stacked(
    cl_stack: Tensor,   # [G, n_active, T, C] — CL 샘플, gradient 있음
    ol_stack: Tensor,   # [G, n_active, T, C] — OL 샘플, 반드시 detach 상태로 전달
) -> Tensor:
    """Per-agent biased MMD² (Gaussian kernel, median bandwidth).

    Biased estimator: 항상 ≥ 0 보장.
    OL gradient 차단: ol_stack.detach() 를 호출해서 전달해야 합니다.
    """
    G, n_active, T, C = cl_stack.shape
    d = T * C

    cl_flat = cl_stack.permute(1, 0, 2, 3).reshape(n_active, G, d)   # [A, G, d]
    ol_flat = ol_stack.permute(1, 0, 2, 3).reshape(n_active, G, d)   # [A, G, d]

    sigma_sq = _global_sigma_sq(ol_flat.detach(), cl_flat.detach())

    kcc = _rbf_batch(cl_flat, cl_flat, sigma_sq)   # [A, G, G] — CL self-term (prevents mode collapse)
    koo = _rbf_batch(ol_flat, ol_flat, sigma_sq)   # [A, G, G] — OL self-term (no gradient)
    kco = _rbf_batch(cl_flat, ol_flat, sigma_sq)   # [A, G, G] — cross-term

    per_agent_mmd = (
        kcc.mean((-2, -1)) + koo.mean((-2, -1)) - 2 * kco.mean((-2, -1))
    ).clamp(min=0)  # biased estimator → ≥ 0, clamp for numerical safety

    return per_agent_mmd.mean()
