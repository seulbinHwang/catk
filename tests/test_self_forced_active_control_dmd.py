from __future__ import annotations

import torch

from src.smart.modules.kinematic_control import (
    CYCLIST_TYPE_ID,
    PEDESTRIAN_TYPE_ID,
    VEHICLE_TYPE_ID,
)
from src.smart.modules.self_forced_dmd_guidance import (
    active_control_dmd_surrogate_loss,
    build_active_control_mask,
    build_clean_dmd_direction,
)


def test_active_control_mask_keeps_lateral_only_for_pedestrians() -> None:
    agent_type = torch.tensor([VEHICLE_TYPE_ID, PEDESTRIAN_TYPE_ID, CYCLIST_TYPE_ID])

    mask = build_active_control_mask(
        agent_type=agent_type,
        flow_dim=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        use_kinematic_control_flow=True,
        use_holonomic_model_only=False,
    )

    expected = torch.tensor(
        [
            [[1.0, 0.0, 1.0]],
            [[1.0, 1.0, 1.0]],
            [[1.0, 0.0, 1.0]],
        ]
    )
    torch.testing.assert_close(mask, expected)


def test_active_control_mask_keeps_all_axes_for_holonomic_only_mode() -> None:
    agent_type = torch.tensor([VEHICLE_TYPE_ID, PEDESTRIAN_TYPE_ID, CYCLIST_TYPE_ID])

    mask = build_active_control_mask(
        agent_type=agent_type,
        flow_dim=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        use_kinematic_control_flow=True,
        use_holonomic_model_only=True,
    )

    torch.testing.assert_close(mask, torch.ones((3, 1, 3)))


def test_active_control_dmd_normalizer_ignores_nonholonomic_lateral_axis() -> None:
    committed = torch.tensor(
        [
            [
                [1.0, 100.0, 3.0],
                [1.0, 100.0, 3.0],
            ]
        ]
    )
    target_clean = torch.tensor(
        [
            [
                [2.0, 5.0, 4.0],
                [2.0, 5.0, 4.0],
            ]
        ]
    )
    generated_clean = torch.zeros_like(target_clean)
    active_mask = torch.tensor([[[1.0, 0.0, 1.0]]])

    direction = build_clean_dmd_direction(
        committed_path_norm=committed,
        target_clean_norm=target_clean,
        generated_clean_norm=generated_clean,
        active_mask=active_mask,
        normalizer_eps=1.0e-3,
    )

    expected = torch.tensor(
        [
            [
                [2.0, 0.0, 4.0],
                [2.0, 0.0, 4.0],
            ]
        ]
    )
    torch.testing.assert_close(direction, expected)


def test_active_control_dmd_loss_has_no_vehicle_lateral_gradient() -> None:
    committed = torch.zeros((2, 1, 3), requires_grad=True)
    dmd_direction = torch.ones_like(committed)
    active_mask = torch.tensor(
        [
            [[1.0, 0.0, 1.0]],
            [[1.0, 1.0, 1.0]],
        ]
    )

    loss, target = active_control_dmd_surrogate_loss(
        committed_path_norm=committed,
        dmd_direction=dmd_direction,
        active_mask=active_mask,
    )
    loss.backward()

    assert target.requires_grad is False
    torch.testing.assert_close(committed.grad[0, 0, 1], torch.tensor(0.0))
    torch.testing.assert_close(
        committed.grad,
        torch.tensor(
            [
                [[-0.25, 0.0, -0.25]],
                [[-1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0]],
            ]
        ),
    )
