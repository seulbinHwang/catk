from __future__ import annotations

import math

import torch

from src.smart.modules.kinematic_control import (
    CONTROL_FLOW_DIM,
    CYCLIST_TYPE_ID,
    MDG_STATE_DIM,
    PEDESTRIAN_TYPE_ID,
    VEHICLE_TYPE_ID,
    build_rolling_control_target,
    build_rolling_control_target_with_round_trip_error,
    control_norm_to_mdg_state_norm,
    control_norm_to_pose_norm,
    decode_control_sequence,
    denormalize_control,
    normalize_control,
    safe_sinc,
)


CONTROL_YAW_SCALE_KWARGS = {
    "vehicle_yaw_scale_rad": 0.5,
    "pedestrian_yaw_scale_rad": 0.5,
    "cyclist_yaw_scale_rad": 0.5,
}


def test_control_dim_is_mdg_style_2d() -> None:
    assert CONTROL_FLOW_DIM == 2
    assert MDG_STATE_DIM == 5


def test_normalize_denormalize_acceleration_yaw_rate_round_trip() -> None:
    agent_type = torch.tensor([VEHICLE_TYPE_ID, PEDESTRIAN_TYPE_ID, CYCLIST_TYPE_ID])
    control = torch.tensor(
        [
            [[1.0, 0.5]],
            [[-2.0, -0.25]],
            [[0.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    control_norm = normalize_control(control, agent_type=agent_type, **CONTROL_YAW_SCALE_KWARGS)
    recovered = denormalize_control(control_norm, agent_type=agent_type, **CONTROL_YAW_SCALE_KWARGS)

    torch.testing.assert_close(control_norm[..., 0], control[..., 0])
    torch.testing.assert_close(control_norm[..., 1], control[..., 1] / 0.5)
    torch.testing.assert_close(recovered, control)


def test_build_target_uses_per_10hz_acceleration_and_yaw_rate_without_chunks() -> None:
    current_pos = torch.zeros((1, 2), dtype=torch.float32)
    current_head = torch.zeros(1, dtype=torch.float32)
    current_speed = torch.tensor([1.0], dtype=torch.float32)
    future_velocity = torch.tensor([[[1.2, 0.0], [1.6, 0.0], [1.6, 0.0]]], dtype=torch.float32)
    future_head = torch.tensor([[0.1, 0.3, 0.3]], dtype=torch.float32)
    future_pos = torch.cumsum(
        torch.stack([future_head.cos(), future_head.sin()], dim=-1)
        * torch.linalg.vector_norm(future_velocity, dim=-1).unsqueeze(-1)
        * 0.1,
        dim=1,
    )

    control_norm = build_rolling_control_target(
        future_pos=future_pos,
        future_head=future_head,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=torch.tensor([VEHICLE_TYPE_ID]),
        current_speed=current_speed,
        future_velocity=future_velocity,
        **CONTROL_YAW_SCALE_KWARGS,
    )
    control = denormalize_control(
        control_norm,
        agent_type=torch.tensor([VEHICLE_TYPE_ID]),
        **CONTROL_YAW_SCALE_KWARGS,
    )

    expected_acc = torch.tensor([[2.0, 4.0, 0.0]], dtype=torch.float32)
    expected_yaw_rate = torch.tensor([[1.0, 2.0, 0.0]], dtype=torch.float32)
    assert tuple(control_norm.shape) == (1, 3, 2)
    torch.testing.assert_close(control[..., 0], expected_acc, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(control[..., 1], expected_yaw_rate, atol=1.0e-6, rtol=1.0e-6)


def test_decode_control_sequence_integrates_speed_heading_and_position() -> None:
    control = torch.tensor([[[1.0, 0.5], [1.0, 0.5]]], dtype=torch.float32)
    current_speed = torch.tensor([1.0], dtype=torch.float32)
    current_pos = torch.zeros((1, 2), dtype=torch.float32)
    current_head = torch.zeros(1, dtype=torch.float32)

    pos, head = decode_control_sequence(
        control=control,
        agent_type=torch.tensor([VEHICLE_TYPE_ID]),
        current_pos=current_pos,
        current_head=current_head,
        current_speed=current_speed,
    )

    expected_speed = torch.tensor([[1.1, 1.2]], dtype=torch.float32)
    expected_head = torch.tensor([[0.05, 0.10]], dtype=torch.float32)
    expected_step = torch.stack([expected_head.cos(), expected_head.sin()], dim=-1)
    expected_pos = torch.cumsum(expected_step * expected_speed.unsqueeze(-1) * 0.1, dim=1)

    torch.testing.assert_close(head, expected_head)
    torch.testing.assert_close(pos, expected_pos)


def test_yaw_rate_is_gated_when_speed_is_too_low() -> None:
    control = torch.tensor([[[0.0, 5.0], [0.0, 5.0]]], dtype=torch.float32)

    _, head = decode_control_sequence(
        control=control,
        agent_type=torch.tensor([VEHICLE_TYPE_ID]),
        current_speed=torch.tensor([0.0]),
    )

    torch.testing.assert_close(head, torch.zeros_like(head))


def test_mdg_state_is_unnormalized_local_xy_heading_and_speed() -> None:
    control = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
    control_norm = normalize_control(
        control,
        agent_type=torch.tensor([VEHICLE_TYPE_ID]),
        **CONTROL_YAW_SCALE_KWARGS,
    )

    state = control_norm_to_mdg_state_norm(
        control_norm=control_norm,
        agent_type=torch.tensor([VEHICLE_TYPE_ID]),
        current_speed=torch.tensor([1.0]),
        **CONTROL_YAW_SCALE_KWARGS,
    )

    expected_speed = torch.tensor([[1.1, 1.2]], dtype=torch.float32)
    expected_x = torch.tensor([[0.11, 0.23]], dtype=torch.float32)
    assert tuple(state.shape) == (1, 2, 5)
    torch.testing.assert_close(state[..., 0], expected_x)
    torch.testing.assert_close(state[..., 1], torch.zeros_like(expected_x))
    torch.testing.assert_close(state[..., 2], torch.ones_like(expected_x))
    torch.testing.assert_close(state[..., 3], torch.zeros_like(expected_x))
    torch.testing.assert_close(state[..., 4], expected_speed)


def test_pose_metric_view_still_uses_pose_position_scale() -> None:
    control = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
    control_norm = normalize_control(
        control,
        agent_type=torch.tensor([VEHICLE_TYPE_ID]),
        **CONTROL_YAW_SCALE_KWARGS,
    )

    pose_norm = control_norm_to_pose_norm(
        control_norm=control_norm,
        agent_type=torch.tensor([VEHICLE_TYPE_ID]),
        current_speed=torch.tensor([2.0]),
        **CONTROL_YAW_SCALE_KWARGS,
    )

    torch.testing.assert_close(pose_norm[0, 0, 0], torch.tensor(0.2 / 20.0))
    torch.testing.assert_close(pose_norm[0, 0, 1], torch.tensor(0.0))
    torch.testing.assert_close(pose_norm[0, 0, 2], torch.tensor(1.0))
    torch.testing.assert_close(pose_norm[0, 0, 3], torch.tensor(0.0))


def test_round_trip_error_is_small_for_dynamics_consistent_target() -> None:
    current_pos = torch.zeros((1, 2), dtype=torch.float32)
    current_head = torch.zeros(1, dtype=torch.float32)
    current_speed = torch.tensor([1.0], dtype=torch.float32)
    control = torch.tensor([[[1.0, 0.2], [0.0, 0.2], [-1.0, 0.0]]], dtype=torch.float32)
    future_pos, future_head = decode_control_sequence(
        control=control,
        agent_type=torch.tensor([VEHICLE_TYPE_ID]),
        current_pos=current_pos,
        current_head=current_head,
        current_speed=current_speed,
    )
    speed = current_speed.unsqueeze(1) + torch.cumsum(control[..., 0] * 0.1, dim=1)
    future_velocity = torch.stack([future_head.cos(), future_head.sin()], dim=-1) * speed.unsqueeze(-1)

    _, round_trip_error_m = build_rolling_control_target_with_round_trip_error(
        future_pos=future_pos,
        future_head=future_head,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=torch.tensor([VEHICLE_TYPE_ID]),
        current_speed=current_speed,
        future_velocity=future_velocity,
        **CONTROL_YAW_SCALE_KWARGS,
    )

    torch.testing.assert_close(round_trip_error_m, torch.zeros_like(round_trip_error_m), atol=1.0e-6, rtol=1.0e-6)


def test_vehicle_pedestrian_and_cyclist_use_same_dynamics() -> None:
    agent_type = torch.tensor([VEHICLE_TYPE_ID, PEDESTRIAN_TYPE_ID, CYCLIST_TYPE_ID])
    control = torch.tensor([[[1.0, 0.2]], [[1.0, 0.2]], [[1.0, 0.2]]], dtype=torch.float32)

    pos, head = decode_control_sequence(
        control=control,
        agent_type=agent_type,
        current_speed=torch.ones(3),
    )

    torch.testing.assert_close(pos[0], pos[1])
    torch.testing.assert_close(pos[1], pos[2])
    torch.testing.assert_close(head[0], head[1])
    torch.testing.assert_close(head[1], head[2])


def test_safe_sinc_is_smooth_around_zero() -> None:
    x = torch.tensor([-1.0e-7, 0.0, 1.0e-7], dtype=torch.float32)
    out = safe_sinc(x)
    torch.testing.assert_close(out, torch.ones_like(out), atol=1.0e-6, rtol=1.0e-6)


def test_agent_type_constants_match_repo_convention() -> None:
    assert VEHICLE_TYPE_ID == 0
    assert PEDESTRIAN_TYPE_ID == 1
    assert CYCLIST_TYPE_ID == 2
