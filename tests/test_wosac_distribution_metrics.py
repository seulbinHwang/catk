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


def test_wosac_distribution_metrics_uses_fixed_type_scale_without_gt() -> None:
    """고정 type scale이 있으면 test CPD도 같은 scale로 정규화해야 합니다."""
    metric = WOSACDistributionMetrics(prefix="test", type_scale=[2.0, 1.0, 1.0])

    pred_traj = torch.tensor(
        [
            [
                [[0.0, 0.0]],
                [[2.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    metric.update(
        pred_traj=pred_traj,
        agent_type=torch.tensor([0]),
        agent_batch=torch.tensor([0]),
        agent_valid_mask=torch.tensor([True]),
    )
    result = metric.compute()

    assert torch.isclose(result["test/WOSAC-CPD/value"], torch.tensor(1.0))


def test_wosac_distribution_metrics_fixed_type_scale_overrides_gt_scale() -> None:
    """고정 type scale이 있으면 validation GT scale 누적값보다 우선해야 합니다."""
    metric = WOSACDistributionMetrics(prefix="val_closed", type_scale=[2.0, 1.0, 1.0])

    pred_traj = torch.tensor(
        [
            [
                [[0.0, 0.0]],
                [[2.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    gt_traj = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
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

    assert torch.isclose(result["val_closed/WOSAC-CPD/value"], torch.tensor(1.0))
    assert torch.all(metric.scale_count == 0)


def test_wosac_distribution_metrics_falls_back_to_gt_scale_without_fixed_scale() -> None:
    """고정 scale이 없으면 기존처럼 validation GT로 type scale을 누적합니다."""
    metric = WOSACDistributionMetrics(prefix="val_closed")

    pred_traj = torch.tensor(
        [
            [
                [[0.0, 0.0]],
                [[2.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    gt_traj = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
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

    assert metric.scale_count[0] == 1
    assert torch.isclose(metric._compute_type_scale()[0], torch.tensor(1.0, dtype=torch.float64))


def test_wosac_distribution_metrics_rejects_invalid_fixed_type_scale() -> None:
    """고정 scale config가 잘못되면 조용히 fallback하지 않고 즉시 실패해야 합니다."""
    try:
        WOSACDistributionMetrics(prefix="val_closed", type_scale=[1.0, 2.0])
    except ValueError as error:
        assert "type_scale length" in str(error)
    else:
        raise AssertionError("Expected invalid type_scale length to raise ValueError.")

    try:
        WOSACDistributionMetrics(prefix="val_closed", type_scale=[1.0, 0.0, 2.0])
    except ValueError as error:
        assert "finite positive" in str(error)
    else:
        raise AssertionError("Expected non-positive type_scale to raise ValueError.")
