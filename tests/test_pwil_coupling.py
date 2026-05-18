"""PWIL coupling 의 수학적 보장 (valid coupling, metric distance, gradient flow) 단위 테스트.

spec 의 7가지 조건 중 모델-비의존 6 가지를 검증합니다:
  1. coupling validity (hungarian / greedy / uniform): row/col sum + non-negativity.
  2. optimality 순서: hungarian ≤ greedy ≤ uniform (W_1 upper bound 강도 순).
  3. distance metric property: symmetric, non-negative, identity (d(x,x)=0).
  4. gradient flow: CL trajectory 에 .grad 가 흐른다.
  5. coupling detachment: gamma 는 requires_grad=False (constant).
  6. bounded reward range: exp 변환 시 loss ∈ [0, α].

7번째 (sequential vs parallel gradient equivalence) 는 SMARTFlow 전체 모델이 필요하므로
end-to-end smoke test 로 별도 검증.
"""

from __future__ import annotations

import torch

from src.smart.metrics.pwil_consistency_loss import (
    pwil_pairwise_distance,
    pwil_row_distance,
    pwil_hungarian_coupling,
    pwil_greedy_coupling,
    pwil_uniform_coupling,
    pwil_loss,
    pwil_loss_per_cl_row,
)


# ─── Test 1: coupling validity ────────────────────────────────────────────────


def test_hungarian_coupling_validity() -> None:
    """hungarian: row sum 1/G, col sum 1/G, ≥0, 총 mass 1 per anchor."""
    torch.manual_seed(0)
    d = torch.rand(10, 4, 4)
    gamma = pwil_hungarian_coupling(d)
    assert gamma.shape == (10, 4, 4)
    assert torch.allclose(gamma.sum(dim=-1), torch.full((10, 4), 0.25), atol=1e-6)
    assert torch.allclose(gamma.sum(dim=-2), torch.full((10, 4), 0.25), atol=1e-6)
    assert (gamma >= 0).all()
    assert torch.allclose(gamma.sum(dim=(-2, -1)), torch.ones(10), atol=1e-6)
    assert not gamma.requires_grad


def test_greedy_coupling_validity_asymmetric() -> None:
    """greedy: G ≠ M 에서도 row sum 1/G, col sum 1/M, ≥0 보존."""
    torch.manual_seed(1)
    d = torch.rand(8, 4, 6)
    gamma = pwil_greedy_coupling(d)
    assert gamma.shape == (8, 4, 6)
    assert torch.allclose(gamma.sum(dim=-1), torch.full((8, 4), 1.0 / 4), atol=1e-5)
    assert torch.allclose(gamma.sum(dim=-2), torch.full((8, 6), 1.0 / 6), atol=1e-5)
    assert (gamma >= 0).all()
    assert torch.allclose(gamma.sum(dim=(-2, -1)), torch.ones(8), atol=1e-5)


def test_uniform_coupling_validity() -> None:
    """uniform: row sum 1/G, col sum 1/M (정의상 자동)."""
    d = torch.rand(5, 3, 7)
    gamma = pwil_uniform_coupling(d)
    assert gamma.shape == (5, 3, 7)
    assert torch.allclose(gamma.sum(dim=-1), torch.full((5, 3), 1.0 / 3), atol=1e-6)
    assert torch.allclose(gamma.sum(dim=-2), torch.full((5, 7), 1.0 / 7), atol=1e-6)


# ─── Test 2: optimality ordering ──────────────────────────────────────────────


def test_coupling_optimality_ordering() -> None:
    """G=M 일 때 hungarian ≤ greedy ≤ uniform (transport cost 기준)."""
    torch.manual_seed(2)
    d = torch.rand(20, 5, 5)
    gamma_h = pwil_hungarian_coupling(d)
    gamma_g = pwil_greedy_coupling(d)
    gamma_u = pwil_uniform_coupling(d)
    cost_h = (d * gamma_h).sum(dim=(-2, -1)).mean()
    cost_g = (d * gamma_g).sum(dim=(-2, -1)).mean()
    cost_u = (d * gamma_u).sum(dim=(-2, -1)).mean()
    assert cost_h.item() <= cost_g.item() + 1e-6
    assert cost_g.item() <= cost_u.item() + 1e-6


# ─── Test 3: distance metric property ─────────────────────────────────────────


def test_distance_is_metric() -> None:
    """d 는 symmetric, non-negative, identity (d(x,x)=0)."""
    torch.manual_seed(3)
    cl = [torch.randn(5, 10, 4) for _ in range(3)]
    ol = [torch.randn(5, 10, 4) for _ in range(3)]
    d_co = pwil_pairwise_distance(cl, ol, pos_weight=1.0, heading_weight=1.0)
    d_oc = pwil_pairwise_distance(ol, cl, pos_weight=1.0, heading_weight=1.0)
    assert torch.allclose(d_co, d_oc.transpose(-2, -1), atol=1e-5)
    assert (d_co >= 0).all()
    d_self = pwil_pairwise_distance(cl, cl, pos_weight=1.0, heading_weight=1.0)
    diag = d_self.diagonal(dim1=-2, dim2=-1)
    # sqrt(clamp 1e-12) ≈ 1e-6 (실수 underflow, 0 직접 검증은 못함)
    assert (diag.abs() < 1e-3).all()


def test_row_distance_matches_pairwise() -> None:
    """pwil_row_distance(cl_g, ol) == pwil_pairwise_distance([cl_g], ol)[:, 0, :]."""
    torch.manual_seed(4)
    cl_g = torch.randn(6, 8, 4)
    ol = [torch.randn(6, 8, 4) for _ in range(5)]
    d_row = pwil_row_distance(cl_g, ol, pos_weight=1.0, heading_weight=0.1)
    d_full = pwil_pairwise_distance([cl_g], ol, pos_weight=1.0, heading_weight=0.1)
    assert torch.allclose(d_row, d_full[:, 0, :], atol=1e-5)


# ─── Test 4: gradient flow ────────────────────────────────────────────────────


def test_pwil_loss_gradient_flow_raw() -> None:
    """raw transport cost: CL 에 .grad 가 흐른다."""
    torch.manual_seed(5)
    cl = [torch.randn(5, 10, 4, requires_grad=True) for _ in range(4)]
    ol = [torch.randn(5, 10, 4) for _ in range(4)]
    d = pwil_pairwise_distance(cl, ol, pos_weight=1.0, heading_weight=0.1)
    gamma = pwil_hungarian_coupling(d)
    loss = pwil_loss(d, gamma, use_exp_reward=False, alpha=1.0, beta=5.0)
    loss.backward()
    for c in cl:
        assert c.grad is not None
        assert c.grad.abs().sum().item() > 0


def test_pwil_loss_gradient_flow_exp_reward() -> None:
    """bounded reward 변환: CL 에 .grad 가 흐른다 (small c 영역)."""
    torch.manual_seed(6)
    cl = [(torch.randn(5, 10, 4) * 0.1).requires_grad_(True) for _ in range(4)]
    ol = [torch.randn(5, 10, 4) * 0.1 for _ in range(4)]
    d = pwil_pairwise_distance(cl, ol, pos_weight=1.0, heading_weight=0.1)
    gamma = pwil_hungarian_coupling(d)
    loss = pwil_loss(d, gamma, use_exp_reward=True, alpha=1.0, beta=5.0)
    loss.backward()
    for c in cl:
        assert c.grad is not None
        assert c.grad.abs().sum().item() > 0


# ─── Test 5: coupling detachment ──────────────────────────────────────────────


def test_coupling_is_detached() -> None:
    """gamma 텐서는 requires_grad=False (autograd graph 와 분리)."""
    cl = [torch.randn(5, 10, 4, requires_grad=True) for _ in range(4)]
    ol = [torch.randn(5, 10, 4) for _ in range(4)]
    d = pwil_pairwise_distance(cl, ol, pos_weight=1.0, heading_weight=0.1)
    assert pwil_hungarian_coupling(d).requires_grad is False
    assert pwil_greedy_coupling(d).requires_grad is False
    assert pwil_uniform_coupling(d).requires_grad is False


# ─── Test 6: bounded reward range ─────────────────────────────────────────────


def test_exp_reward_bounded() -> None:
    """exp 변환: 큰 거리에서도 loss ∈ [0, α + eps]."""
    torch.manual_seed(7)
    alpha = 2.5
    cl = [torch.randn(5, 10, 4) * 100 for _ in range(4)]
    ol = [torch.randn(5, 10, 4) for _ in range(4)]
    d = pwil_pairwise_distance(cl, ol, pos_weight=1.0, heading_weight=0.1)
    gamma = pwil_hungarian_coupling(d)
    loss = pwil_loss(d, gamma, use_exp_reward=True, alpha=alpha, beta=5.0)
    assert 0.0 <= loss.item() <= alpha + 1e-6


# ─── Bonus: row contribution sums to full pwil_loss ───────────────────────────


def test_row_contribution_sums_to_full_loss_raw() -> None:
    """row 별 contribution 합 == full pwil_loss (raw transport)."""
    torch.manual_seed(8)
    cl = [torch.randn(5, 10, 4) for _ in range(4)]
    ol = [torch.randn(5, 10, 4) for _ in range(4)]
    d = pwil_pairwise_distance(cl, ol, pos_weight=1.0, heading_weight=0.1)
    gamma = pwil_hungarian_coupling(d)
    full = pwil_loss(d, gamma, use_exp_reward=False, alpha=1.0, beta=5.0)
    G = len(cl)
    n_active = cl[0].shape[0]
    row_sum = sum(
        pwil_loss_per_cl_row(
            d_row=pwil_row_distance(cl[g], ol, pos_weight=1.0, heading_weight=0.1),
            gamma_row=gamma[:, g, :],
            use_exp_reward=False,
            alpha=1.0, beta=5.0,
            G=G, n_active=n_active,
        )
        for g in range(G)
    )
    # raw mode 에서 row 분해: mean_n Σ_g Σ_j γ d == Σ_g (1/(n·G)) Σ_n Σ_j γ d
    # = (1/n) Σ_n Σ_{g,j} γ d == full 의 정의.
    # 단, pwil_loss raw 는 .mean() over N_active 만 (G factor 없음); per-row 는 /(n·G) 로 나눠 G 합.
    # → row_sum == full 가 되도록 정규화 맞춤 (helper 의 정의).
    assert torch.allclose(row_sum, full, atol=1e-5)


def test_row_contribution_sums_to_full_loss_exp() -> None:
    """row 별 contribution 합 == full pwil_loss (bounded reward)."""
    torch.manual_seed(9)
    cl = [torch.randn(5, 10, 4) * 0.1 for _ in range(4)]
    ol = [torch.randn(5, 10, 4) * 0.1 for _ in range(4)]
    d = pwil_pairwise_distance(cl, ol, pos_weight=1.0, heading_weight=0.1)
    gamma = pwil_hungarian_coupling(d)
    full = pwil_loss(d, gamma, use_exp_reward=True, alpha=1.0, beta=5.0)
    G = len(cl)
    n_active = cl[0].shape[0]
    row_sum = sum(
        pwil_loss_per_cl_row(
            d_row=pwil_row_distance(cl[g], ol, pos_weight=1.0, heading_weight=0.1),
            gamma_row=gamma[:, g, :],
            use_exp_reward=True,
            alpha=1.0, beta=5.0,
            G=G, n_active=n_active,
        )
        for g in range(G)
    )
    assert torch.allclose(row_sum, full, atol=1e-5)


if __name__ == "__main__":
    # 빠른 sanity-check 실행: pytest 없이도 돌 수 있게.
    fns = [
        test_hungarian_coupling_validity,
        test_greedy_coupling_validity_asymmetric,
        test_uniform_coupling_validity,
        test_coupling_optimality_ordering,
        test_distance_is_metric,
        test_row_distance_matches_pairwise,
        test_pwil_loss_gradient_flow_raw,
        test_pwil_loss_gradient_flow_exp_reward,
        test_coupling_is_detached,
        test_exp_reward_bounded,
        test_row_contribution_sums_to_full_loss_raw,
        test_row_contribution_sums_to_full_loss_exp,
    ]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print("all PWIL tests passed")
