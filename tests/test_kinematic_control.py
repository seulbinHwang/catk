from __future__ import annotations

import torch

from src.smart.modules.kinematic_control import (
    build_rolling_control_target,
    control_norm_to_pose_norm,
    decode_control_sequence,
    denormalize_control,
)


def test_pedestrian_rolling_control_reconstructs_target_position() -> None:
    current_pos = torch.tensor([[0.0, 0.0]])
    current_head = torch.tensor([0.0])
    future_pos = torch.tensor([[[1.0, 1.0], [2.0, 1.0]]])
    future_head = torch.tensor([[0.2, 0.2]])
    agent_type = torch.tensor([1])

    control_norm = build_rolling_control_target(
        future_pos=future_pos,
        future_head=future_head,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=agent_type,
    )
    decoded_pos, decoded_head = decode_control_sequence(
        control=denormalize_control(control_norm),
        agent_type=agent_type,
        current_pos=current_pos,
        current_head=current_head,
    )

    torch.testing.assert_close(decoded_pos, future_pos, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(decoded_head, future_head, atol=1.0e-5, rtol=1.0e-5)


def test_vehicle_rolling_control_uses_no_lateral_channel() -> None:
    current_pos = torch.tensor([[0.0, 0.0]])
    current_head = torch.tensor([0.0])
    future_pos = torch.tensor([[[1.0, 0.2], [2.0, 0.5]]])
    future_head = torch.tensor([[0.1, 0.2]])
    agent_type = torch.tensor([0])

    control_norm = build_rolling_control_target(
        future_pos=future_pos,
        future_head=future_head,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=agent_type,
    )
    control = denormalize_control(control_norm)

    assert torch.allclose(control[..., 1], torch.zeros_like(control[..., 1]))


def test_control_norm_to_pose_norm_returns_pose_space_shape() -> None:
    control_norm = torch.zeros((2, 5, 3))
    control_norm[..., 0] = 1.0
    agent_type = torch.tensor([0, 1])

    pose_norm = control_norm_to_pose_norm(control_norm=control_norm, agent_type=agent_type)

    assert tuple(pose_norm.shape) == (2, 5, 4)
    torch.testing.assert_close(
        pose_norm[0, :, 0],
        torch.arange(1, 6, dtype=pose_norm.dtype) / 20.0,
    )
    torch.testing.assert_close(pose_norm[..., 2], torch.ones_like(pose_norm[..., 2]))
    torch.testing.assert_close(pose_norm[..., 3], torch.zeros_like(pose_norm[..., 3]))
