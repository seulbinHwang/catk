"""PWIL (Primal Wasserstein Imitation Learning) coupling-based OL/CL consistency loss.

문제: 기존 paired L2 / nearest_match 는 단순 매칭 — 분포 정렬을 직접 보장 못함.
MMD² 는 분포 정렬 보장은 있으나 kernel bandwidth 에 민감, "분포 거리" 자체는 측정 안 함.

해결: Wasserstein-1 거리의 명시적 upper bound 를 valid transport coupling 으로 구성.
  W_1(\\hat\\rho_CL^{(k)}, \\hat\\rho_OL^{(k)}) ≤ Σ_{i,j} γ^{(k)}[i,j] · d(CL_i^{(k)}, OL_j^{(k)})
  s.t.  Σ_j γ[i,j] = 1/G,  Σ_i γ[i,j] = 1/M,  γ ≥ 0
  Hungarian (G=M) 시 등호 (exact W_1) 성립.

채널: [x/20, y/20, cos_h, sin_h] (C=4) — pos_w / heading_w 별도 가중.
거리 d: sqrt(pos_w · Σpos² + heading_w · Σhead²) — proper metric (triangle inequality 보존).

옵션 bounded reward 변환: r = α(1 - exp(-β c)), c = per-CL transport cost.
PWIL 원논문의 bounded reward 성질을 보존; small c 영역에서 단조 동등 (Taylor: αβc).
"""

from __future__ import annotations

import torch
from torch import Tensor

_POS_SLICE = slice(0, 2)
_HEADING_SLICE = slice(2, 4)


def pwil_pairwise_distance(
    cl_norms: list[Tensor],
    ol_norms: list[Tensor],
    pos_weight: float,
    heading_weight: float,
) -> Tensor:
    """CL/OL 모든 쌍의 가중 L2 거리를 anchor 별로 계산합니다.

    Args:
        cl_norms: 길이 G 의 CL 궤적 리스트. 각 원소 shape ``[N_active, T, 4]``.
        ol_norms: 길이 M 의 OL 궤적 리스트. 각 원소 shape ``[N_active, T, 4]``.
            target 이므로 호출자가 detach 해서 넘기는 것을 권장합니다.
        pos_weight: position 채널 (ch[0:2]) 가중치.
        heading_weight: heading 채널 (ch[2:4]) 가중치.

    Returns:
        Tensor: ``[N_active, G, M]`` 가중 L2 거리.
            ``d[n, i, j] = sqrt(pos_w · Σ(Δxy)² + heading_w · Σ(Δhead)²)`` —
            T 와 채널 내 합산. clamp(min=1e-12) 로 sqrt 0-근처 NaN gradient 회피.
    """
    if not cl_norms or not ol_norms:
        raise ValueError("pwil_pairwise_distance: empty cl_norms / ol_norms")
    T = min(int(cl_norms[0].shape[-2]), int(ol_norms[0].shape[-2]))
    if T <= 0:
        raise ValueError(f"pwil_pairwise_distance: degenerate time dim T={T}")
    cl_stack = torch.stack([c[:, :T, :] for c in cl_norms], dim=1)   # [N, G, T, 4]
    ol_stack = torch.stack([o[:, :T, :] for o in ol_norms], dim=1)   # [N, M, T, 4]
    ol_stack = ol_stack.detach()

    diff = cl_stack.unsqueeze(2) - ol_stack.unsqueeze(1)             # [N, G, M, T, 4]
    diff_sq = diff * diff
    d2_pos = diff_sq[..., _POS_SLICE].sum(dim=(-1, -2))              # [N, G, M]
    d2_head = diff_sq[..., _HEADING_SLICE].sum(dim=(-1, -2))         # [N, G, M]
    d2 = pos_weight * d2_pos + heading_weight * d2_head
    return torch.sqrt(d2.clamp(min=1e-12))


def pwil_row_distance(
    cl_norm: Tensor,
    ol_norms: list[Tensor],
    pos_weight: float,
    heading_weight: float,
) -> Tensor:
    """단일 CL g 와 M 개 OL 의 가중 L2 거리 — sequential PWIL 의 Pass-2 용.

    Args:
        cl_norm: ``[N_active, T, 4]`` (gradient 살아있는 1 개 CL g).
        ol_norms: 길이 M, 각 ``[N_active, T, 4]`` (detached target).
        pos_weight: position 가중치.
        heading_weight: heading 가중치.

    Returns:
        Tensor: ``[N_active, M]`` 가중 L2 거리.
    """
    if not ol_norms:
        raise ValueError("pwil_row_distance: empty ol_norms")
    T = min(int(cl_norm.shape[-2]), int(ol_norms[0].shape[-2]))
    if T <= 0:
        raise ValueError(f"pwil_row_distance: degenerate time dim T={T}")
    ol_stack = torch.stack([o[:, :T, :] for o in ol_norms], dim=1).detach()  # [N, M, T, 4]
    cl_b = cl_norm[:, :T, :].unsqueeze(1)                                    # [N, 1, T, 4]
    diff = cl_b - ol_stack                                                   # [N, M, T, 4]
    diff_sq = diff * diff
    d2_pos = diff_sq[..., _POS_SLICE].sum(dim=(-1, -2))                      # [N, M]
    d2_head = diff_sq[..., _HEADING_SLICE].sum(dim=(-1, -2))                 # [N, M]
    d2 = pos_weight * d2_pos + heading_weight * d2_head
    return torch.sqrt(d2.clamp(min=1e-12))


def pwil_hungarian_coupling(d: Tensor) -> Tensor:
    """anchor 별 1-to-1 optimal assignment (scipy Hungarian, G=M 필수).

    coupling 은 θ 와 독립 (no_grad 에서 계산). 반환 텐서는 grad 추적 없음.

    Args:
        d: ``[N_active, G, G]`` 거리 행렬.

    Returns:
        Tensor: ``[N_active, G, G]``. ``γ[n, i, j] = 1/G`` iff i↔j assignment, 그 외 0.
            row sum = col sum = ``1/G``, 전체 mass = 1 (per anchor).
    """
    from scipy.optimize import linear_sum_assignment

    if d.dim() != 3 or d.shape[-1] != d.shape[-2]:
        raise ValueError(f"pwil_hungarian_coupling: expected [N, G, G], got {tuple(d.shape)}")
    N, G, _ = d.shape
    gamma = torch.zeros_like(d)
    inv_G = 1.0 / float(G)
    with torch.no_grad():
        d_cpu = d.detach().cpu().numpy()
        for n in range(N):
            row_idx, col_idx = linear_sum_assignment(d_cpu[n])
            for i, j in zip(row_idx, col_idx):
                gamma[n, int(i), int(j)] = inv_G
    return gamma


def pwil_greedy_coupling(d: Tensor) -> Tensor:
    """anchor 별 greedy mass transport — G ≠ M 허용 (원논문 PWIL faithful).

    각 CL i 에 mass 1/G 를, capacity 1/M 의 OL j 들에 nearest-first 로 transport.
    iteration order 는 anchor 별 random permutation (index bias 회피).

    Args:
        d: ``[N_active, G, M]`` 거리 행렬.

    Returns:
        Tensor: ``[N_active, G, M]`` valid coupling.
            row sum = ``1/G``, col sum = ``1/M``, 전체 mass = 1.
    """
    if d.dim() != 3:
        raise ValueError(f"pwil_greedy_coupling: expected [N, G, M], got {tuple(d.shape)}")
    N, G, M = d.shape
    gamma = torch.zeros_like(d)
    cap_unit = 1.0 / float(M)
    mass_per_cl = 1.0 / float(G)
    eps = 1e-9
    with torch.no_grad():
        d_cpu = d.detach().cpu()
        for n in range(N):
            remaining_cap = torch.full((M,), cap_unit, dtype=torch.float64)
            cl_order = torch.randperm(G).tolist()
            for i in cl_order:
                remaining_mass = mass_per_cl
                row_dist = d_cpu[n, i].to(torch.float64).clone()
                while remaining_mass > eps:
                    available = remaining_cap > eps
                    if not bool(available.any()):
                        break
                    masked = row_dist.clone()
                    masked[~available] = float("inf")
                    j_star = int(torch.argmin(masked).item())
                    transfer = min(remaining_mass, float(remaining_cap[j_star].item()))
                    gamma[n, i, j_star] += transfer
                    remaining_cap[j_star] -= transfer
                    remaining_mass -= transfer
    return gamma


def pwil_uniform_coupling(d: Tensor) -> Tensor:
    """uniform coupling ``γ[n,i,j] = 1/(G·M)`` — ablation baseline (가장 느슨한 bound).

    Args:
        d: ``[N_active, G, M]``.

    Returns:
        Tensor: ``[N_active, G, M]`` — row sum 1/G, col sum 1/M.
    """
    if d.dim() != 3:
        raise ValueError(f"pwil_uniform_coupling: expected [N, G, M], got {tuple(d.shape)}")
    _, G, M = d.shape
    return torch.full_like(d, 1.0 / (float(G) * float(M)))


def pwil_loss(
    d: Tensor,
    gamma: Tensor,
    use_exp_reward: bool,
    alpha: float,
    beta: float,
) -> Tensor:
    """PWIL transport cost / bounded-reward 손실을 계산합니다.

    Args:
        d: ``[N_active, G, M]`` — CL g 에서 backward 가능.
        gamma: ``[N_active, G, M]`` — no_grad 로 계산된 상수 coupling.
        use_exp_reward: True 면 per-CL transport cost 에 ``α(1 - exp(-β c))`` 변환,
            False 면 raw transport cost ``<d, γ>``.
        alpha: bounded reward scale (use_exp_reward 전용).
        beta: bounded reward decay 계수.

    Returns:
        Tensor: scalar loss.
    """
    if gamma.requires_grad:
        gamma = gamma.detach()
    if not use_exp_reward:
        loss_per_anchor = (d * gamma).sum(dim=(-2, -1))        # [N_active]
        return loss_per_anchor.mean()
    G = int(gamma.shape[-2])
    c_per_cl = (d * gamma).sum(dim=-1) * float(G)              # [N_active, G]
    loss_per_cl = float(alpha) * (1.0 - torch.exp(-float(beta) * c_per_cl))
    return loss_per_cl.mean()


def pwil_loss_per_cl_row(
    d_row: Tensor,
    gamma_row: Tensor,
    use_exp_reward: bool,
    alpha: float,
    beta: float,
    G: int,
    n_active: int,
) -> Tensor:
    """단일 CL g 의 row 기반 PWIL contribution — sequential pass-2 backward 전용.

    full loss ``L = mean_n mean_g loss_per_cl(n, g)`` 를 g 별로 쪼개 backward 할 때,
    각 row contribution = ``loss_per_cl(:, g).sum() / (n_active · G)``. 모든 g 합 = L.

    Args:
        d_row: ``[N_active, M]`` — CL g 와 M 개 OL 의 거리.
        gamma_row: ``[N_active, M]`` — ``gamma[:, g, :]`` row.
        use_exp_reward: pwil_loss 와 동일.
        alpha: bounded reward scale.
        beta: bounded reward decay.
        G: row 가 속한 G (mean over G 정규화에 사용).
        n_active: anchor 의 N_active (mean over n 정규화에 사용).

    Returns:
        Tensor: scalar — g 별 contribution. ΣG = full pwil_loss.
    """
    if gamma_row.requires_grad:
        gamma_row = gamma_row.detach()
    if not use_exp_reward:
        # full pwil_loss(raw) = mean_n (Σ_{g,j} γ d) = (1/n) Σ_n Σ_{g,j} γ d
        # row g contribution = (1/n) Σ_n Σ_j γ[n,g,j] d[n,g,j] → Σ_g = full.
        return (d_row * gamma_row).sum() / max(1.0, float(n_active))
    # full pwil_loss(exp) = mean over (n, g) of α(1 - exp(-β c_{n,g}))
    #                    = (1/(n·G)) Σ_n Σ_g loss_per_cl[n, g]
    # row g contribution  = (1/(n·G)) Σ_n loss_per_cl[n, g] → Σ_g = full.
    c_g = (d_row * gamma_row).sum(dim=-1) * float(G)           # [N_active]
    loss_g = float(alpha) * (1.0 - torch.exp(-float(beta) * c_g))
    return loss_g.sum() / max(1.0, float(n_active) * float(G))
