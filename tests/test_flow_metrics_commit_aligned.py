from __future__ import annotations

import torch
import torch.nn.functional as F

from src.smart.metrics.flow_metrics import flow_matching_loss


def _build_commit_aligned_weights() -> torch.Tensor:
    return torch.tensor([1.25] * 5 + [1.01] * 5 + [0.87] * 10, dtype=torch.float32)


def test_flow_matching_loss_without_future_weights_matches_plain_mse() -> None:
    """가중치를 넘기지 않으면 기존 Flow Matching MSE와 같아야 합니다."""
    torch.manual_seed(0)
    pred = torch.randn(3, 20, 4)
    target = torch.randn(3, 20, 4)

    actual = flow_matching_loss(pred, target)
    expected = F.mse_loss(pred, target)

    torch.testing.assert_close(actual, expected)


def test_flow_matching_loss_uses_explicit_commit_aligned_weights() -> None:
    """open-loop 호출이 명시적으로 넘긴 1.25 / 1.01 / 0.87 가중치를 사용합니다."""
    pred = torch.zeros(2, 20, 4)
    target = torch.zeros(2, 20, 4)
    target[:, :, 0] = torch.arange(20, dtype=torch.float32).view(1, 20)
    weights = _build_commit_aligned_weights()

    actual = flow_matching_loss(pred, target, future_step_weights=weights)
    expected = ((pred - target).square() * weights.view(1, 20, 1)).mean()

    torch.testing.assert_close(actual, expected)


def test_flow_matching_loss_normalizes_weights_inside_each_valid_prefix() -> None:
    """partial-valid anchor는 자기 유효 구간 안에서 평균 가중치가 1이어야 합니다."""
    pred = torch.zeros(3, 20, 4)
    target = torch.zeros(3, 20, 4)
    target[:, :, 0] = torch.arange(1, 21, dtype=torch.float32).view(1, 20)
    valid_mask = torch.zeros(3, 20, dtype=torch.bool)
    valid_mask[0, :20] = True
    valid_mask[1, :15] = True
    valid_mask[2, :5] = True
    weights = _build_commit_aligned_weights()

    actual = flow_matching_loss(
        pred,
        target,
        valid_mask=valid_mask,
        future_step_weights=weights,
    )

    mask_float = valid_mask.float()
    per_anchor_mean = (mask_float * weights.view(1, 20)).sum(dim=1, keepdim=True) / mask_float.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1.0)
    normalized_weights = mask_float * (weights.view(1, 20) / per_anchor_mean)
    expected = ((pred - target).square() * normalized_weights.unsqueeze(-1)).sum() / (
        mask_float.sum() * pred.shape[-1]
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        normalized_weights.sum(dim=1) / valid_mask.float().sum(dim=1),
        torch.ones(3),
    )


def test_flow_matching_loss_keeps_generated_estimator_style_call_unweighted() -> None:
    """self-forced generated estimator처럼 weight를 넘기지 않는 호출은 후반 step을 깎지 않습니다."""
    pred = torch.zeros(1, 20, 3)
    first_error_target = torch.zeros_like(pred)
    preview_error_target = torch.zeros_like(pred)
    first_error_target[0, 0, 0] = 1.0
    preview_error_target[0, 19, 0] = 1.0

    first_loss = flow_matching_loss(pred, first_error_target)
    preview_loss = flow_matching_loss(pred, preview_error_target)

    torch.testing.assert_close(first_loss, preview_loss)


def test_flow_matching_loss_backpropagates_with_future_weights() -> None:
    """가중 손실도 예측 tensor로 정상 gradient를 보내야 합니다."""
    torch.manual_seed(1)
    pred = torch.randn(2, 20, 3, requires_grad=True)
    target = torch.randn(2, 20, 3)
    loss = flow_matching_loss(pred, target, future_step_weights=_build_commit_aligned_weights())

    loss.backward()

    assert pred.grad is not None
    assert tuple(pred.grad.shape) == tuple(pred.shape)
    assert torch.isfinite(pred.grad).all()
