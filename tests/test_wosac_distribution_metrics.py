import torch

from src.smart.metrics.wosac_distribution_metrics import WOSACDistributionMetrics


def test_wosac_distribution_metric_logs_cpd_and_ces_for_validation() -> None:
    """GT가 있는 validation batch에서 CPD와 CES가 함께 계산되는지 확인합니다."""
    metric = WOSACDistributionMetrics(prefix="val_closed")
    current_pos = torch.zeros((2, 2), dtype=torch.float32)
    gt_traj = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0]],
            [[0.0, 1.0], [0.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    pred_traj = torch.stack(
        [
            gt_traj,
            gt_traj + torch.tensor([1.0, 0.0], dtype=torch.float32),
        ],
        dim=1,
    )

    metric.update(
        pred_traj=pred_traj,
        agent_type=torch.tensor([0, 1], dtype=torch.long),
        agent_batch=torch.zeros(2, dtype=torch.long),
        current_pos=current_pos,
        gt_traj=gt_traj,
        gt_valid_mask=torch.ones((2, 2), dtype=torch.bool),
        agent_valid_mask=torch.ones(2, dtype=torch.bool),
    )

    result = metric.compute()
    assert "val_closed/WOSAC-CPD/value" in result
    assert "val_closed/WOSAC-CES/value" in result
    assert torch.isfinite(result["val_closed/WOSAC-CPD/value"])
    assert torch.isfinite(result["val_closed/WOSAC-CES/value"])


def test_wosac_distribution_metric_logs_only_cpd_without_gt() -> None:
    """GT가 없는 test batch에서는 CES 없이 CPD만 계산되는지 확인합니다."""
    metric = WOSACDistributionMetrics(prefix="test")
    pred_traj = torch.tensor(
        [
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [2.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )

    metric.update(
        pred_traj=pred_traj,
        agent_type=torch.zeros(1, dtype=torch.long),
        agent_batch=torch.zeros(1, dtype=torch.long),
    )

    result = metric.compute()
    assert "test/WOSAC-CPD/value" in result
    assert "test/WOSAC-CES/value" not in result
    assert torch.isfinite(result["test/WOSAC-CPD/value"])
