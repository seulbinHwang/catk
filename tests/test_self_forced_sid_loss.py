from __future__ import annotations

import torch

from src.smart.modules.self_forced_sid_loss import compute_clean_sid_loss


def test_compute_clean_sid_loss_keeps_gradient_on_committed_path() -> None:
    committed = torch.zeros(2, 5, 4, requires_grad=True)
    target = torch.ones(2, 5, 4)
    generated = torch.zeros(2, 5, 4)

    loss = compute_clean_sid_loss(
        committed_path_norm=committed,
        target_clean_norm=target,
        generated_clean_norm=generated,
        sid_alpha=1.0,
    )
    loss.backward()

    assert committed.grad is not None
    assert torch.isfinite(committed.grad).all()
    assert target.grad is None
    assert generated.grad is None


def test_compute_clean_sid_loss_matches_expanded_formula() -> None:
    committed = torch.tensor([[[0.2, -0.1, 0.3, 0.0]]], requires_grad=True)
    target = torch.tensor([[[1.0, 0.5, -0.2, 0.3]]])
    generated = torch.tensor([[[0.4, 0.1, 0.0, -0.1]]])
    alpha = 0.7
    eps = 1.0e-3

    actual = compute_clean_sid_loss(
        committed_path_norm=committed,
        target_clean_norm=target,
        generated_clean_norm=generated,
        sid_alpha=alpha,
        normalizer_eps=eps,
    )

    score_gap = target - generated
    expected = score_gap * ((target - committed) - alpha * score_gap)
    normalizer = (committed.detach() - target).abs().mean(dim=(1, 2), keepdim=True).clamp_min(eps)
    expected = (expected / normalizer).mean()

    assert torch.allclose(actual, expected)
