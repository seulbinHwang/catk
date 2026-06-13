from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.smart.metrics.flow_metrics import clean_path_consistency_loss, flow_matching_loss
from src.smart.model.smart_flow import SMARTFlow, _resolve_clean_path_consistency_config


def test_flow_matching_loss_stays_plain_masked_mse() -> None:
    pred = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            [[4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0]],
        ]
    )
    target = torch.zeros_like(pred)
    valid_mask = torch.tensor([[True, False], [False, True]])

    loss = flow_matching_loss(pred, target, valid_mask=valid_mask)
    expected = F.mse_loss(
        torch.stack([pred[0, 0], pred[1, 1]], dim=0),
        torch.zeros((2, 4)),
    )

    torch.testing.assert_close(loss, expected)


def test_clean_path_consistency_ignores_early_tau() -> None:
    pred_clean_norm = torch.ones(2, 4, 4, requires_grad=True)
    target_clean_norm = torch.zeros(2, 4, 4)
    tau = torch.full((2,), 0.50)

    loss = clean_path_consistency_loss(
        pred_clean_norm=pred_clean_norm,
        target_clean_norm=target_clean_norm,
        tau=tau,
        tau_min=0.75,
        commit_steps=2,
        commit_weight=2.0,
    )

    assert loss.item() == 0.0
    loss.backward()
    assert pred_clean_norm.grad is not None
    assert torch.count_nonzero(pred_clean_norm.grad).item() == 0


def test_clean_path_consistency_uses_tau_mask_valid_mask_and_commit_weight() -> None:
    pred_clean_norm = torch.zeros(2, 4, 4)
    target_clean_norm = torch.zeros(2, 4, 4)
    pred_clean_norm[0] = 1.0
    pred_clean_norm[1] = 3.0
    tau = torch.tensor([0.80, 0.50])
    valid_mask = torch.tensor([[True, True, True, False], [True, True, True, True]])

    loss = clean_path_consistency_loss(
        pred_clean_norm=pred_clean_norm,
        target_clean_norm=target_clean_norm,
        tau=tau,
        valid_mask=valid_mask,
        tau_min=0.75,
        commit_steps=2,
        commit_weight=2.0,
    )

    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_clean_path_consistency_keeps_batch_vectorized_gradients() -> None:
    pred_clean_norm = torch.zeros(3, 5, 4, requires_grad=True)
    target_clean_norm = torch.ones(3, 5, 4)
    tau = torch.tensor([0.90, 0.90, 0.10])
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, False, False, False, False],
            [True, True, True, True, True],
        ]
    )

    loss = clean_path_consistency_loss(
        pred_clean_norm=pred_clean_norm,
        target_clean_norm=target_clean_norm,
        tau=tau,
        valid_mask=valid_mask,
        tau_min=0.75,
        commit_steps=2,
        commit_weight=2.0,
    )
    loss.backward()

    assert pred_clean_norm.grad is not None
    assert torch.count_nonzero(pred_clean_norm.grad[0]).item() > 0
    assert torch.count_nonzero(pred_clean_norm.grad[1]).item() > 0
    assert torch.count_nonzero(pred_clean_norm.grad[2]).item() == 0


def test_clean_path_consistency_config_accepts_recommended_values() -> None:
    config = SimpleNamespace(
        clean_path_consistency=SimpleNamespace(
            enabled=True,
            weight=0.25,
            unit_impact_scale=300.0,
            tau_min=0.75,
            commit_steps=5,
            commit_weight=2.0,
        )
    )

    assert _resolve_clean_path_consistency_config(config) == (True, 0.25, 300.0, 0.75, 5, 2.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weight", 0.75),
        ("weight", 0.0),
        ("unit_impact_scale", 0.0),
        ("unit_impact_scale", float("inf")),
        ("tau_min", 0.50),
        ("tau_min", 1.0),
        ("commit_steps", 0),
        ("commit_weight", 0.5),
    ],
)
def test_clean_path_consistency_config_rejects_unsafe_values(field: str, value: float) -> None:
    values = {
        "enabled": True,
        "weight": 0.25,
        "unit_impact_scale": 300.0,
        "tau_min": 0.75,
        "commit_steps": 5,
        "commit_weight": 2.0,
    }
    values[field] = value
    config = SimpleNamespace(clean_path_consistency=SimpleNamespace(**values))

    with pytest.raises(ValueError):
        _resolve_clean_path_consistency_config(config)


def test_model_clean_path_loss_uses_pose_metric_tensors() -> None:
    model = SMARTFlow.__new__(SMARTFlow)
    nn.Module.__init__(model)
    model.clean_path_consistency_enabled = True
    model.clean_path_consistency_unit_impact_scale = 300.0
    model.clean_path_consistency_tau_min = 0.75
    model.clean_path_consistency_commit_steps = 2
    model.clean_path_consistency_commit_weight = 2.0

    pred_dict = {
        "flow_pred_norm": torch.zeros((2, 4, 3)),
        "flow_target_norm": torch.zeros((2, 4, 3)),
        "flow_pred_clean_norm": torch.zeros((2, 4, 3)),
        "flow_clean_norm": torch.zeros((2, 4, 3)),
        "flow_pred_clean_metric_norm": torch.ones((2, 4, 4)),
        "flow_clean_metric_norm": torch.zeros((2, 4, 4)),
        "flow_tau": torch.tensor([0.90, 0.50]),
        "flow_loss_mask": torch.tensor([[True, True, True, False], [True, True, True, True]]),
    }

    loss = model._compute_clean_path_consistency_loss(pred_dict, has_loss_targets=True)

    torch.testing.assert_close(loss, torch.tensor(300.0))


def test_open_loop_epoch_accumulator_keeps_aux_loss_when_train_metrics_are_disabled() -> None:
    model = SMARTFlow.__new__(SMARTFlow)
    nn.Module.__init__(model)
    model._train_open_epoch_log_names = (
        "train/loss",
        "train/loss_fm",
        "train/loss_clean_path_consistency",
        "train/ADE2s",
        "train/FDE2s",
        "train/ADEyaw2s",
        "train/FDEyaw2s",
    )
    model.register_buffer(
        "_train_open_epoch_metric_sums",
        torch.zeros(len(model._train_open_epoch_log_names), dtype=torch.float32),
        persistent=False,
    )
    model.register_buffer(
        "_train_open_epoch_metric_counts",
        torch.zeros(len(model._train_open_epoch_log_names), dtype=torch.float32),
        persistent=False,
    )

    model._accumulate_open_loop_train_epoch_metrics(
        total_loss=torch.tensor(3.0),
        fm_loss=torch.tensor(2.0),
        clean_path_consistency_loss_value=torch.tensor(4.0),
        open_metric_dict={},
        sample_count=5,
    )

    torch.testing.assert_close(
        model._train_open_epoch_metric_counts,
        torch.tensor([5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        model._train_open_epoch_metric_sums,
        torch.tensor([15.0, 10.0, 20.0, 0.0, 0.0, 0.0, 0.0]),
    )
