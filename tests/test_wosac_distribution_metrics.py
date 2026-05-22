from __future__ import annotations

from types import SimpleNamespace

import torch

from src.smart.metrics.wosac_distribution_metrics import (
    WOSACDistributionMetrics,
    log_and_reset_wosac_distribution_metric,
    update_wosac_distribution_metric_from_model,
)


def test_wosac_distribution_metrics_compute_cpd_ces() -> None:
    """단일 scenario 예제로 CPD와 CES 계산식을 검증합니다."""
    metric = WOSACDistributionMetrics(prefix="val_closed")

    pred_traj = torch.tensor(
        [
            [
                [[1.0, 0.0], [2.0, 0.0]],
                [[1.0, 0.0], [4.0, 0.0]],
            ]
        ]
    )  # [n_agent=1, n_rollout=2, n_step=2, 2]
    gt_traj = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])  # [n_agent=1, n_step=2, 2]
    gt_valid_mask = torch.tensor([[True, True]])  # [n_agent=1, n_step=2]
    current_pos = torch.tensor([[0.0, 0.0]])  # [n_agent=1, 2]

    metric.update(
        pred_traj=pred_traj,
        agent_type=torch.tensor([0]),
        agent_batch=torch.tensor([0]),
        current_pos=current_pos,
        gt_traj=gt_traj,
        gt_valid_mask=gt_valid_mask,
        agent_valid_mask=torch.tensor([True]),
    )
    result = metric.compute()

    expected_cpd = torch.tensor(0.8944272)
    expected_ces = torch.tensor(0.2236068)
    assert torch.isclose(result["val_closed/WOSAC-CPD/value"], expected_cpd, atol=1.0e-5)
    assert torch.isclose(result["val_closed/WOSAC-CES/value"], expected_ces, atol=1.0e-5)


def test_wosac_distribution_metrics_dpr() -> None:
    """기준 CPD가 있을 때 DPR이 함께 기록되는지 검증합니다."""
    metric = WOSACDistributionMetrics(prefix="val_closed", cpd_reference=2.0)

    pred_traj = torch.tensor(
        [
            [
                [[0.0, 0.0]],
                [[2.0, 0.0]],
            ]
        ]
    )  # [n_agent=1, n_rollout=2, n_step=1, 2]
    metric.update(
        pred_traj=pred_traj,
        agent_type=torch.tensor([0]),
        agent_batch=torch.tensor([0]),
        agent_valid_mask=torch.tensor([True]),
    )
    result = metric.compute()

    assert torch.isclose(result["val_closed/WOSAC-CPD/value"], torch.tensor(2.0))
    assert torch.isclose(result["val_closed/WOSAC-CPD/DPR"], torch.tensor(1.0))


def test_wosac_distribution_metrics_clamps_rms_scale_only_at_eps() -> None:
    """정지 GT의 type scale은 eps로만 clamp되어야 합니다."""
    metric = WOSACDistributionMetrics(prefix="val_closed")

    pred_traj = torch.tensor(
        [
            [
                [[0.0, 0.0]],
                [[2.0, 0.0]],
            ]
        ]
    )
    gt_traj = torch.zeros((1, 1, 2), dtype=torch.float32)
    gt_valid_mask = torch.tensor([[True]])
    current_pos = torch.zeros((1, 2), dtype=torch.float32)

    metric.update(
        pred_traj=pred_traj,
        agent_type=torch.tensor([0]),
        agent_batch=torch.tensor([0]),
        current_pos=current_pos,
        gt_traj=gt_traj,
        gt_valid_mask=gt_valid_mask,
        agent_valid_mask=torch.tensor([True]),
    )
    result = metric.compute()

    expected_cpd = torch.tensor(2.0 / metric.eps)
    assert torch.isclose(
        result["val_closed/WOSAC-CPD/value"],
        expected_cpd,
        rtol=1.0e-6,
    )


def test_wosac_distribution_metric_skips_duplicate_model_update() -> None:
    """같은 batch와 같은 rollout tensor가 helper를 통해 두 번 누적되지 않아야 합니다."""
    metric = WOSACDistributionMetrics(prefix="val_closed")
    model = SimpleNamespace(num_historical_steps=2)
    data = {
        "agent": {
            "position": torch.tensor([[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]]),
            "valid_mask": torch.tensor([[True, True, True]]),
            "type": torch.tensor([0]),
            "batch": torch.tensor([0]),
        }
    }
    pred_traj = torch.tensor(
        [
            [
                [[1.0, 0.0]],
                [[2.0, 0.0]],
            ]
        ]
    )

    update_wosac_distribution_metric_from_model(
        metric=metric,
        model=model,
        data=data,
        pred_traj=pred_traj,
        include_gt=True,
    )
    assert len(metric.pair_count) == 1

    update_wosac_distribution_metric_from_model(
        metric=metric,
        model=model,
        data=data,
        pred_traj=pred_traj,
        include_gt=True,
    )
    assert len(metric.pair_count) == 1

    update_wosac_distribution_metric_from_model(
        metric=metric,
        model=model,
        data=data,
        pred_traj=pred_traj.clone(),
        include_gt=True,
    )
    assert len(metric.pair_count) == 2

    log_and_reset_wosac_distribution_metric(metric)
    assert not hasattr(metric, "_last_wosac_distribution_update_key")

    update_wosac_distribution_metric_from_model(
        metric=metric,
        model=model,
        data=data,
        pred_traj=pred_traj,
        include_gt=True,
    )
    assert len(metric.pair_count) == 1
