from __future__ import annotations

import torch

from src.smart.metrics.wosac_distribution_metrics import WOSACDistributionMetrics


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
