from __future__ import annotations

import torch
import torch.nn as nn

from src.smart.model.smart_flow import SMARTFlow


def _make_minimal_model() -> SMARTFlow:
    model = SMARTFlow.__new__(SMARTFlow)
    nn.Module.__init__(model)
    model.encoder = nn.Linear(2, 1, bias=False)
    model.automatic_optimization = True
    model.open_metric_names = {
        "ade": "ADE2s",
        "fde": "FDE2s",
        "yaw_ade": "yaw_ADE2s",
        "yaw_fde": "yaw_FDE2s",
    }
    model._automatic_open_loop_has_target_pending = []
    model._build_open_loop_metric_dict = lambda **_: {
        "ADE2s": torch.zeros(()),
        "FDE2s": torch.zeros(()),
        "yaw_ADE2s": torch.zeros(()),
        "yaw_FDE2s": torch.zeros(()),
    }
    return model


def _empty_pred_dict() -> dict[str, torch.Tensor]:
    empty = torch.zeros((0, 20, 4), dtype=torch.float32)
    return {
        "flow_pred_norm": empty,
        "flow_target_norm": empty,
        "flow_pred_clean_norm": empty,
        "flow_clean_norm": empty,
        "flow_loss_mask": torch.zeros((0, 20), dtype=torch.bool),
    }


def test_open_loop_empty_target_loss_keeps_trainable_graph() -> None:
    model = _make_minimal_model()

    loss, _metrics, sample_count, has_targets = model._open_loop_denoise_metrics(
        _empty_pred_dict(),
        zero_loss_module=model.encoder,
    )

    assert sample_count == 0
    assert has_targets is False
    assert loss.requires_grad
    loss.backward()
    assert model.encoder.weight.grad is not None
    torch.testing.assert_close(model.encoder.weight.grad, torch.zeros_like(model.encoder.weight.grad))


def test_open_loop_all_false_future_mask_is_treated_as_empty_target() -> None:
    model = _make_minimal_model()
    pred = {
        "flow_pred_norm": torch.zeros((2, 20, 4), dtype=torch.float32),
        "flow_target_norm": torch.zeros((2, 20, 4), dtype=torch.float32),
        "flow_pred_clean_norm": torch.zeros((2, 20, 4), dtype=torch.float32),
        "flow_clean_norm": torch.zeros((2, 20, 4), dtype=torch.float32),
        "flow_loss_mask": torch.zeros((2, 20), dtype=torch.bool),
    }

    loss, _metrics, sample_count, has_targets = model._open_loop_denoise_metrics(
        pred,
        zero_loss_module=model.encoder,
    )

    assert sample_count == 2
    assert has_targets is False
    assert loss.requires_grad


def test_empty_target_automatic_step_clears_zero_grads_before_adamw_decay() -> None:
    model = _make_minimal_model()
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=1.0, weight_decay=0.1)
    before = model.encoder.weight.detach().clone()
    model._automatic_open_loop_has_target_since_step = False
    model._skip_next_automatic_optimizer_step = False

    loss = model._build_trainable_connected_zero_loss(model.encoder)
    loss.backward()
    assert model.encoder.weight.grad is not None

    model.on_before_optimizer_step(optimizer)
    optimizer.step()

    torch.testing.assert_close(model.encoder.weight, before)
    assert model._skip_next_automatic_optimizer_step is False


def test_empty_local_target_keeps_grad_when_another_rank_has_target() -> None:
    model = _make_minimal_model()
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=1.0, weight_decay=0.1)
    model._automatic_open_loop_has_target_since_step = False
    model._skip_next_automatic_optimizer_step = False
    model._sync_distributed_bool_any = lambda value, *, device=None: True  # type: ignore[method-assign]
    model._automatic_open_loop_has_target_pending = [(torch.tensor(1, dtype=torch.long), None)]

    loss = model._build_trainable_connected_zero_loss(model.encoder)
    loss.backward()
    assert model.encoder.weight.grad is not None

    model.on_before_optimizer_step(optimizer)

    assert model.encoder.weight.grad is not None
    assert model._automatic_open_loop_has_target_since_step is False
    assert model._skip_next_automatic_optimizer_step is False


def test_manual_optimizer_hook_does_not_clear_existing_gradients() -> None:
    model = _make_minimal_model()
    model.automatic_optimization = False
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=1.0, weight_decay=0.1)

    loss = model.encoder(torch.ones((1, 2), dtype=torch.float32)).sum()
    loss.backward()
    assert model.encoder.weight.grad is not None
    grad_before = model.encoder.weight.grad.detach().clone()

    model.on_before_optimizer_step(optimizer)

    assert model.encoder.weight.grad is not None
    torch.testing.assert_close(model.encoder.weight.grad, grad_before)
