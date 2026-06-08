from __future__ import annotations

import torch

from src.smart.metrics.flow_metrics import mdg_state_loss


def test_mdg_state_loss_sums_state_dims_before_step_average() -> None:
    pred = torch.ones((2, 2, 5), dtype=torch.float32)
    target = torch.zeros_like(pred)
    mask = torch.tensor([[True, False], [True, True]])

    loss = mdg_state_loss(pred, target, valid_mask=mask)

    torch.testing.assert_close(loss, torch.tensor(5.0))


def test_mdg_state_loss_returns_trainable_zero_for_empty_mask() -> None:
    pred = torch.ones((1, 2, 5), dtype=torch.float32, requires_grad=True)
    target = torch.zeros_like(pred)
    mask = torch.zeros((1, 2), dtype=torch.bool)

    loss = mdg_state_loss(pred, target, valid_mask=mask)
    loss.backward()

    torch.testing.assert_close(loss.detach(), torch.tensor(0.0))
    assert pred.grad is not None
