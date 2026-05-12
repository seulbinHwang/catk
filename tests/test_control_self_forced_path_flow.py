from __future__ import annotations

import torch

from src.smart.modules.kinematic_control import (
    VEHICLE_TYPE_ID,
    denormalize_control,
)
from src.smart.modules.self_forced_path_flow import build_anchor0_normalized_committed_control


CONTROL_YAW_SCALE_KWARGS = {
    "vehicle_yaw_scale_rad": 0.025,
    "pedestrian_yaw_scale_rad": 0.20,
    "cyclist_yaw_scale_rad": 0.06,
}


def test_control_self_forced_projection_returns_control_state() -> None:
    committed_path_norm = torch.tensor(
        [
            [
                [1.0 / 20.0, 0.2 / 20.0, 1.0, 0.0],
                [2.0 / 20.0, 0.5 / 20.0, 0.9800666, 0.1986693],
            ]
        ],
        dtype=torch.float32,
    )
    tokenized_agent = {"type": torch.tensor([VEHICLE_TYPE_ID])}
    anchor_mask = torch.tensor([True])

    control_norm = build_anchor0_normalized_committed_control(
        committed_path_norm=committed_path_norm,
        tokenized_agent=tokenized_agent,
        anchor_mask=anchor_mask,
        pos_scale_m=1.0,
        **CONTROL_YAW_SCALE_KWARGS,
    )
    control = denormalize_control(
        control_norm=control_norm,
        agent_type=tokenized_agent["type"][anchor_mask],
        pos_scale_m=1.0,
        **CONTROL_YAW_SCALE_KWARGS,
    )

    assert tuple(control_norm.shape) == (1, 2, 3)
    torch.testing.assert_close(control[..., 1], torch.zeros_like(control[..., 1]))


def test_control_self_forced_projection_keeps_generator_gradient_path() -> None:
    future_x = torch.tensor([[1.0, 2.0]], dtype=torch.float32, requires_grad=True)
    future_y = torch.tensor([[0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    future_head = torch.tensor([[0.0, 0.2]], dtype=torch.float32, requires_grad=True)
    committed_path_norm = torch.stack(
        [
            future_x / 20.0,
            future_y / 20.0,
            future_head.cos(),
            future_head.sin(),
        ],
        dim=-1,
    )
    tokenized_agent = {"type": torch.tensor([VEHICLE_TYPE_ID])}
    anchor_mask = torch.tensor([True])

    control_norm = build_anchor0_normalized_committed_control(
        committed_path_norm=committed_path_norm,
        tokenized_agent=tokenized_agent,
        anchor_mask=anchor_mask,
        pos_scale_m=1.0,
        **CONTROL_YAW_SCALE_KWARGS,
    )
    loss = control_norm.square().sum()
    loss.backward()

    assert future_x.grad is not None
    assert future_head.grad is not None
    assert torch.isfinite(future_x.grad).all()
    assert torch.isfinite(future_head.grad).all()
