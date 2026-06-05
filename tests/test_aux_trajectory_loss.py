from __future__ import annotations

import math

import torch

from src.smart.metrics.flow_metrics import auxiliary_best_mode_trajectory_loss


def test_auxiliary_best_mode_selects_by_xy_only_and_wraps_heading() -> None:
    pred = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    target = torch.zeros((1, 2, 3), dtype=torch.float32)
    mask = torch.ones((1, 2), dtype=torch.bool)

    # Mode 0 has perfect xy and a wrapped heading error near zero.
    pred[0, 0, :, 2] = torch.tensor([math.pi - 0.01, -math.pi + 0.01])
    target[0, :, 2] = torch.tensor([-math.pi + 0.01, math.pi - 0.01])
    # Mode 1 has better raw heading but worse xy, so it must not be selected.
    pred[0, 1, :, :2] = 1.0
    pred[0, 1, :, 2] = target[0, :, 2]

    loss = auxiliary_best_mode_trajectory_loss(
        pred_local=pred,
        target_local=target,
        valid_mask=mask,
    )

    expected_heading_error = torch.tensor([-0.02, 0.02], dtype=torch.float32)
    expected = torch.nn.functional.smooth_l1_loss(
        expected_heading_error,
        torch.zeros_like(expected_heading_error),
        reduction="sum",
    ) / 2.0
    torch.testing.assert_close(loss, expected)


def test_auxiliary_best_mode_ignores_invalid_future_steps() -> None:
    pred = torch.zeros((1, 1, 2, 3), dtype=torch.float32)
    target = torch.zeros((1, 2, 3), dtype=torch.float32)
    mask = torch.tensor([[True, False]])
    pred[0, 0, 1] = torch.tensor([1000.0, 1000.0, 1000.0])
    target[0, 1] = torch.tensor([-1000.0, -1000.0, -1000.0])

    loss = auxiliary_best_mode_trajectory_loss(
        pred_local=pred,
        target_local=target,
        valid_mask=mask,
    )

    torch.testing.assert_close(loss, torch.zeros(()))


def test_auxiliary_best_mode_all_invalid_keeps_trainable_graph() -> None:
    pred = torch.zeros((1, 1, 2, 3), dtype=torch.float32, requires_grad=True)
    target = torch.zeros((1, 2, 3), dtype=torch.float32)
    mask = torch.zeros((1, 2), dtype=torch.bool)

    loss = auxiliary_best_mode_trajectory_loss(
        pred_local=pred,
        target_local=target,
        valid_mask=mask,
    )

    assert loss.requires_grad
    loss.backward()
    assert pred.grad is not None
    torch.testing.assert_close(pred.grad, torch.zeros_like(pred.grad))
