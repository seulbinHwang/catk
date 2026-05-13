from __future__ import annotations

import math

import pytest
import torch

from src.smart.modules.kinematic_control import (
    CYCLIST_TYPE_ID,
    PEDESTRIAN_TYPE_ID,
    VEHICLE_TYPE_ID,
    build_rolling_control_target,
    build_rolling_control_target_with_round_trip_error,
    control_norm_to_pose_norm,
    decode_control_sequence,
    denormalize_control,
    safe_sinc,
)

CONTROL_YAW_SCALE_KWARGS = {
    "vehicle_yaw_scale_rad": 0.025,
    "pedestrian_yaw_scale_rad": 0.20,
    "cyclist_yaw_scale_rad": 0.06,
}


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
        **CONTROL_YAW_SCALE_KWARGS,
    )
    decoded_pos, decoded_head = decode_control_sequence(
        control=denormalize_control(
            control_norm,
            agent_type=agent_type,
            **CONTROL_YAW_SCALE_KWARGS,
        ),
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
        **CONTROL_YAW_SCALE_KWARGS,
    )
    control = denormalize_control(control_norm, agent_type=agent_type, **CONTROL_YAW_SCALE_KWARGS)

    assert torch.allclose(control[..., 1], torch.zeros_like(control[..., 1]))


def test_holonomic_model_only_lets_vehicle_use_lateral_channel_and_round_trip() -> None:
    current_pos = torch.tensor([[0.0, 0.0]])
    current_head = torch.tensor([0.0])
    future_pos = torch.tensor([[[1.0, 0.2], [2.0, 0.5]]])
    future_head = torch.tensor([[0.1, 0.2]])
    agent_type = torch.tensor([VEHICLE_TYPE_ID])

    control_norm, round_trip_error_m = build_rolling_control_target_with_round_trip_error(
        future_pos=future_pos,
        future_head=future_head,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=agent_type,
        use_holonomic_model_only=True,
        **CONTROL_YAW_SCALE_KWARGS,
    )
    control = denormalize_control(control_norm, agent_type=agent_type, **CONTROL_YAW_SCALE_KWARGS)
    decoded_pos, decoded_head = decode_control_sequence(
        control=control,
        agent_type=agent_type,
        current_pos=current_pos,
        current_head=current_head,
        use_holonomic_model_only=True,
    )

    assert torch.any(control[..., 1].abs() > 1.0e-6)
    torch.testing.assert_close(decoded_pos, future_pos, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(decoded_head, future_head, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(round_trip_error_m, torch.zeros_like(round_trip_error_m), atol=1.0e-5, rtol=1.0e-5)


def test_cyclist_rolling_control_uses_no_lateral_channel_and_round_trips() -> None:
    current_pos = torch.tensor([[0.0, 0.0]])
    current_head = torch.tensor([0.0])
    future_pos = torch.tensor([[[1.0, 0.05], [2.0, 0.2]]])
    future_head = torch.tensor([[0.05, 0.15]])
    agent_type = torch.tensor([CYCLIST_TYPE_ID])

    control_norm = build_rolling_control_target(
        future_pos=future_pos,
        future_head=future_head,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=agent_type,
        **CONTROL_YAW_SCALE_KWARGS,
    )
    control = denormalize_control(control_norm, agent_type=agent_type, **CONTROL_YAW_SCALE_KWARGS)
    decoded_pos, decoded_head = decode_control_sequence(
        control=control,
        agent_type=agent_type,
        current_pos=current_pos,
        current_head=current_head,
    )

    # cyclist는 non-holonomic 분기를 따라야 하므로 lateral 채널이 0이어야 한다.
    assert torch.allclose(control[..., 1], torch.zeros_like(control[..., 1]))
    # 같은 분기로 decoder를 통과시키면 head는 GT를 따라가지만 위치는 lateral 성분이 빠진다.
    torch.testing.assert_close(decoded_head, future_head, atol=1.0e-5, rtol=1.0e-5)
    assert not torch.allclose(decoded_pos, future_pos, atol=1.0e-3, rtol=1.0e-3)


def test_invalid_agent_type_id_is_rejected() -> None:
    current_pos = torch.tensor([[0.0, 0.0]])
    current_head = torch.tensor([0.0])
    future_pos = torch.tensor([[[1.0, 0.0]]])
    future_head = torch.tensor([[0.0]])
    bad_agent_type = torch.tensor([3])

    with pytest.raises(ValueError, match="repo convention"):
        build_rolling_control_target(
            future_pos=future_pos,
            future_head=future_head,
            current_pos=current_pos,
            current_head=current_head,
            agent_type=bad_agent_type,
            **CONTROL_YAW_SCALE_KWARGS,
        )


def test_safe_sinc_is_smooth_around_zero() -> None:
    # 양수, 음수, 정확히 0, 큰 값을 한 번에 검증한다.
    x = torch.tensor([1.0e-9, -1.0e-9, 0.0, 0.5, -0.5, 2.0])
    out = safe_sinc(x)
    # 0 근처는 1, 큰 값은 sin(x)/x 와 일치해야 한다.
    expected_large = torch.tensor([math.sin(0.5) / 0.5, math.sin(-0.5) / -0.5, math.sin(2.0) / 2.0])
    torch.testing.assert_close(out[:3], torch.ones_like(out[:3]), atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(out[3:], expected_large, atol=1.0e-6, rtol=1.0e-6)


def test_safe_sinc_gradient_finite_at_zero() -> None:
    x = torch.zeros(1, requires_grad=True)
    out = safe_sinc(x).sum()
    out.backward()
    assert torch.isfinite(x.grad).all()


def test_rolling_projection_round_trip_for_vehicle_uses_decoder_consistent_pose() -> None:
    # decoder-consistent rolling projection에서 vehicle는 head는 GT, pose는 h_mid 투영분으로 전진한다.
    current_pos = torch.tensor([[0.0, 0.0]])
    current_head = torch.tensor([0.0])
    future_pos = torch.tensor([[[1.0, 0.3]]])
    future_head = torch.tensor([[0.4]])
    agent_type = torch.tensor([VEHICLE_TYPE_ID])

    control_norm = build_rolling_control_target(
        future_pos=future_pos,
        future_head=future_head,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=agent_type,
        **CONTROL_YAW_SCALE_KWARGS,
    )
    control = denormalize_control(control_norm, agent_type=agent_type, **CONTROL_YAW_SCALE_KWARGS)
    decoded_pos, decoded_head = decode_control_sequence(
        control=control,
        agent_type=agent_type,
        current_pos=current_pos,
        current_head=current_head,
    )

    # head는 GT와 같아야 한다.
    torch.testing.assert_close(decoded_head[:, 0], future_head[:, 0], atol=1.0e-5, rtol=1.0e-5)
    # pose는 GT 변위의 h_mid 투영분만큼만 진행한다.
    delta_head = future_head[:, 0] - current_head
    mid = current_head + 0.5 * delta_head
    h_mid = torch.stack([mid.cos(), mid.sin()], dim=-1)
    delta_vec = future_pos[:, 0] - current_pos
    expected_pos = current_pos + (delta_vec * h_mid).sum(dim=-1, keepdim=True) * h_mid
    torch.testing.assert_close(decoded_pos[:, 0], expected_pos, atol=1.0e-5, rtol=1.0e-5)


def test_round_trip_error_reports_vehicle_lateral_teleport() -> None:
    current_pos = torch.tensor([[0.0, 0.0]])
    current_head = torch.tensor([0.0])
    future_pos = torch.tensor([[[0.0, 6.0]]])
    future_head = torch.tensor([[0.0]])
    agent_type = torch.tensor([VEHICLE_TYPE_ID])

    control_norm, round_trip_error_m = build_rolling_control_target_with_round_trip_error(
        future_pos=future_pos,
        future_head=future_head,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=agent_type,
        **CONTROL_YAW_SCALE_KWARGS,
    )

    assert tuple(control_norm.shape) == (1, 1, 3)
    torch.testing.assert_close(round_trip_error_m, torch.tensor([[6.0]]), atol=1.0e-5, rtol=1.0e-5)


def test_round_trip_error_is_zero_for_pedestrian() -> None:
    current_pos = torch.tensor([[0.0, 0.0]])
    current_head = torch.tensor([0.0])
    future_pos = torch.tensor([[[0.0, 6.0]]])
    future_head = torch.tensor([[0.0]])
    agent_type = torch.tensor([PEDESTRIAN_TYPE_ID])

    _, round_trip_error_m = build_rolling_control_target_with_round_trip_error(
        future_pos=future_pos,
        future_head=future_head,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=agent_type,
        **CONTROL_YAW_SCALE_KWARGS,
    )

    torch.testing.assert_close(round_trip_error_m, torch.zeros_like(round_trip_error_m))


def test_pedestrian_uses_pedestrian_type_id_constant() -> None:
    assert PEDESTRIAN_TYPE_ID == 1
    assert VEHICLE_TYPE_ID == 0
    assert CYCLIST_TYPE_ID == 2


def test_control_norm_to_pose_norm_returns_pose_space_shape() -> None:
    control_norm = torch.zeros((2, 5, 3))
    control_norm[..., 0] = 1.0
    agent_type = torch.tensor([0, 1])

    pose_norm = control_norm_to_pose_norm(
        control_norm=control_norm,
        agent_type=agent_type,
        **CONTROL_YAW_SCALE_KWARGS,
    )

    assert tuple(pose_norm.shape) == (2, 5, 4)
    torch.testing.assert_close(
        pose_norm[0, :, 0],
        torch.arange(1, 6, dtype=pose_norm.dtype) / 20.0,
    )
    torch.testing.assert_close(pose_norm[..., 2], torch.ones_like(pose_norm[..., 2]))
    torch.testing.assert_close(pose_norm[..., 3], torch.zeros_like(pose_norm[..., 3]))
