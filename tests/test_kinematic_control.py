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


def _reference_round_trip_error(
    control_norm: torch.Tensor,
    future_pos: torch.Tensor,
    current_pos: torch.Tensor,
    current_head: torch.Tensor,
    agent_type: torch.Tensor,
    agent_length: torch.Tensor | None = None,
    *,
    use_holonomic_model_only: bool = False,
    vehicle_no_slip_point_ratio: float = 0.0,
    cyclist_no_slip_point_ratio: float = 0.0,
) -> torch.Tensor:
    control = denormalize_control(
        control_norm,
        agent_type=agent_type,
        **CONTROL_YAW_SCALE_KWARGS,
    )
    decoded_pos, _ = decode_control_sequence(
        control=control,
        agent_type=agent_type,
        agent_length=agent_length,
        current_pos=current_pos,
        current_head=current_head,
        use_holonomic_model_only=use_holonomic_model_only,
        vehicle_no_slip_point_ratio=vehicle_no_slip_point_ratio,
        cyclist_no_slip_point_ratio=cyclist_no_slip_point_ratio,
    )
    return torch.linalg.vector_norm(decoded_pos - future_pos, dim=-1)


def test_pedestrian_rolling_control_uses_nonholonomic_projection() -> None:
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
    control = denormalize_control(control_norm, agent_type=agent_type, **CONTROL_YAW_SCALE_KWARGS)

    torch.testing.assert_close(control[..., 1], torch.zeros_like(control[..., 1]))
    assert not torch.allclose(decoded_pos, future_pos, atol=1.0e-3, rtol=1.0e-3)
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


def test_vehicle_no_slip_point_ratio_zero_preserves_box_center_rule() -> None:
    current_pos = torch.tensor([[0.0, 0.0]])
    current_head = torch.tensor([0.2])
    agent_type = torch.tensor([VEHICLE_TYPE_ID])
    control = torch.tensor([[[1.0, 0.0, 0.3], [0.5, 0.0, -0.2]]])

    old_pos, old_head = decode_control_sequence(
        control=control,
        agent_type=agent_type,
        current_pos=current_pos,
        current_head=current_head,
    )
    ratio_zero_pos, ratio_zero_head = decode_control_sequence(
        control=control,
        agent_type=agent_type,
        agent_length=torch.tensor([4.5]),
        current_pos=current_pos,
        current_head=current_head,
        vehicle_no_slip_point_ratio=0.0,
    )

    torch.testing.assert_close(ratio_zero_pos, old_pos)
    torch.testing.assert_close(ratio_zero_head, old_head)


def test_vehicle_no_slip_point_ratio_adds_box_center_rotation_offset() -> None:
    current_pos = torch.tensor([[0.0, 0.0]])
    current_head = torch.tensor([0.0])
    agent_type = torch.tensor([VEHICLE_TYPE_ID])
    agent_length = torch.tensor([4.0])
    control = torch.tensor([[[1.0, 0.0, math.pi / 2.0]]])
    offset = 0.5 * agent_length

    decoded_pos, decoded_head = decode_control_sequence(
        control=control,
        agent_type=agent_type,
        agent_length=agent_length,
        current_pos=current_pos,
        current_head=current_head,
        vehicle_no_slip_point_ratio=0.5,
    )

    delta_head = control[:, 0, 2]
    mid_head = current_head + 0.5 * delta_head
    arc = control[:, 0, 0].unsqueeze(-1) * safe_sinc(0.5 * delta_head).unsqueeze(-1) * torch.stack(
        [mid_head.cos(), mid_head.sin()],
        dim=-1,
    )
    heading_offset = offset.unsqueeze(-1) * (
        torch.stack([delta_head.cos(), delta_head.sin()], dim=-1)
        - torch.stack([current_head.cos(), current_head.sin()], dim=-1)
    )
    expected_pos = current_pos + arc + heading_offset

    torch.testing.assert_close(decoded_pos[:, 0], expected_pos, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(decoded_head[:, 0], torch.tensor([math.pi / 2.0]), atol=1.0e-5, rtol=1.0e-5)


def test_vehicle_and_cyclist_no_slip_point_ratios_are_type_specific() -> None:
    current_pos = torch.zeros((2, 2))
    current_head = torch.zeros(2)
    agent_type = torch.tensor([VEHICLE_TYPE_ID, CYCLIST_TYPE_ID])
    agent_length = torch.tensor([4.0, 2.0])
    control = torch.zeros((2, 1, 3))
    control[:, 0, 2] = math.pi / 2.0

    decoded_pos, _ = decode_control_sequence(
        control=control,
        agent_type=agent_type,
        agent_length=agent_length,
        current_pos=current_pos,
        current_head=current_head,
        vehicle_no_slip_point_ratio=0.25,
        cyclist_no_slip_point_ratio=0.10,
    )

    expected_offset = torch.tensor([1.0, 0.2])
    expected_pos = torch.stack([-expected_offset, expected_offset], dim=-1)
    torch.testing.assert_close(decoded_pos[:, 0], expected_pos, atol=1.0e-5, rtol=1.0e-5)


def test_vehicle_no_slip_point_rolling_label_round_trips_with_same_transition() -> None:
    current_pos = torch.tensor([[2.0, -1.0]])
    current_head = torch.tensor([0.3])
    agent_type = torch.tensor([VEHICLE_TYPE_ID])
    agent_length = torch.tensor([4.0])
    original_control = torch.tensor([[[1.2, 0.0, 0.35], [0.8, 0.0, -0.15]]])

    future_pos, future_head = decode_control_sequence(
        control=original_control,
        agent_type=agent_type,
        agent_length=agent_length,
        current_pos=current_pos,
        current_head=current_head,
        vehicle_no_slip_point_ratio=0.5,
    )
    control_norm = build_rolling_control_target(
        future_pos=future_pos,
        future_head=future_head,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=agent_type,
        agent_length=agent_length,
        vehicle_no_slip_point_ratio=0.5,
        **CONTROL_YAW_SCALE_KWARGS,
    )
    rebuilt_control = denormalize_control(control_norm, agent_type=agent_type, **CONTROL_YAW_SCALE_KWARGS)
    rebuilt_pos, rebuilt_head = decode_control_sequence(
        control=rebuilt_control,
        agent_type=agent_type,
        agent_length=agent_length,
        current_pos=current_pos,
        current_head=current_head,
        vehicle_no_slip_point_ratio=0.5,
    )

    torch.testing.assert_close(rebuilt_control, original_control, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(rebuilt_pos, future_pos, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(rebuilt_head, future_head, atol=1.0e-5, rtol=1.0e-5)


def test_vehicle_no_slip_point_ratio_requires_agent_length() -> None:
    with pytest.raises(ValueError, match="agent_length is required"):
        decode_control_sequence(
            control=torch.zeros((1, 1, 3)),
            agent_type=torch.tensor([VEHICLE_TYPE_ID]),
            vehicle_no_slip_point_ratio=0.5,
        )


def test_holonomic_model_only_is_rejected() -> None:
    current_pos = torch.tensor([[0.0, 0.0]])
    current_head = torch.tensor([0.0])
    future_pos = torch.tensor([[[1.0, 0.2], [2.0, 0.5]]])
    future_head = torch.tensor([[0.1, 0.2]])
    agent_type = torch.tensor([VEHICLE_TYPE_ID])

    with pytest.raises(ValueError, match="all-agent non-holonomic"):
        build_rolling_control_target_with_round_trip_error(
            future_pos=future_pos,
            future_head=future_head,
            current_pos=current_pos,
            current_head=current_head,
            agent_type=agent_type,
            use_holonomic_model_only=True,
            **CONTROL_YAW_SCALE_KWARGS,
        )
    with pytest.raises(ValueError, match="all-agent non-holonomic"):
        decode_control_sequence(
            control=torch.zeros((1, 2, 3)),
            agent_type=agent_type,
            current_pos=current_pos,
            current_head=current_head,
            use_holonomic_model_only=True,
        )


def test_raw_pose_pair_supervision_differs_from_rolling_for_nonholonomic_vehicle() -> None:
    current_pos = torch.tensor([[0.0, 0.0]])
    current_head = torch.tensor([0.0])
    future_pos = torch.tensor([[[1.0, 1.0], [2.0, 1.5]]])
    future_head = torch.tensor([[0.4, 0.4]])
    agent_type = torch.tensor([VEHICLE_TYPE_ID])

    rolling_norm = build_rolling_control_target(
        future_pos=future_pos,
        future_head=future_head,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=agent_type,
        use_holonomic_model_only=False,
        use_rolling_supervision=True,
        **CONTROL_YAW_SCALE_KWARGS,
    )
    raw_pair_norm = build_rolling_control_target(
        future_pos=future_pos,
        future_head=future_head,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=agent_type,
        use_holonomic_model_only=False,
        use_rolling_supervision=False,
        **CONTROL_YAW_SCALE_KWARGS,
    )

    rolling = denormalize_control(rolling_norm, agent_type=agent_type, **CONTROL_YAW_SCALE_KWARGS)
    raw_pair = denormalize_control(raw_pair_norm, agent_type=agent_type, **CONTROL_YAW_SCALE_KWARGS)

    torch.testing.assert_close(raw_pair[:, 0], rolling[:, 0], atol=1.0e-5, rtol=1.0e-5)
    assert not torch.allclose(raw_pair[:, 1], rolling[:, 1], atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(raw_pair[..., 1], torch.zeros_like(raw_pair[..., 1]))


def test_rolling_supervision_still_changes_nonholonomic_projection() -> None:
    current_pos = torch.tensor([[0.0, 0.0]])
    current_head = torch.tensor([0.0])
    future_pos = torch.tensor([[[1.0, 1.0], [2.0, 1.5]]])
    future_head = torch.tensor([[0.4, 0.4]])
    agent_type = torch.tensor([VEHICLE_TYPE_ID])

    rolling_norm = build_rolling_control_target(
        future_pos=future_pos,
        future_head=future_head,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=agent_type,
        use_rolling_supervision=True,
        **CONTROL_YAW_SCALE_KWARGS,
    )
    raw_pair_norm = build_rolling_control_target(
        future_pos=future_pos,
        future_head=future_head,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=agent_type,
        use_rolling_supervision=False,
        **CONTROL_YAW_SCALE_KWARGS,
    )

    assert not torch.allclose(raw_pair_norm, rolling_norm, atol=1.0e-5, rtol=1.0e-5)


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


def test_round_trip_error_reports_pedestrian_lateral_motion() -> None:
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

    torch.testing.assert_close(round_trip_error_m, torch.tensor([[6.0]]), atol=1.0e-5, rtol=1.0e-5)


def test_fused_round_trip_error_matches_separate_decode_reference() -> None:
    generator = torch.Generator().manual_seed(20260521)
    current_pos = torch.randn((9, 2), generator=generator)
    current_head = torch.randn((9,), generator=generator) * 0.5
    increments = torch.randn((9, 7, 2), generator=generator) * 0.4
    future_pos = current_pos.unsqueeze(1) + increments.cumsum(dim=1)
    future_head = current_head.unsqueeze(1) + torch.randn((9, 7), generator=generator) * 0.15
    agent_type = torch.tensor(
        [
            VEHICLE_TYPE_ID,
            PEDESTRIAN_TYPE_ID,
            CYCLIST_TYPE_ID,
            VEHICLE_TYPE_ID,
            PEDESTRIAN_TYPE_ID,
            CYCLIST_TYPE_ID,
            VEHICLE_TYPE_ID,
            PEDESTRIAN_TYPE_ID,
            CYCLIST_TYPE_ID,
        ],
        dtype=torch.long,
    )
    agent_length = torch.tensor([4.8, 0.8, 1.8, 4.2, 0.7, 1.6, 5.0, 0.9, 1.7])

    expected_control_norm = build_rolling_control_target(
        future_pos=future_pos,
        future_head=future_head,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=agent_type,
        agent_length=agent_length,
        vehicle_no_slip_point_ratio=0.2289518863,
        cyclist_no_slip_point_ratio=0.0495847873,
        **CONTROL_YAW_SCALE_KWARGS,
    )
    expected_round_trip_error = _reference_round_trip_error(
        control_norm=expected_control_norm,
        future_pos=future_pos,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=agent_type,
        agent_length=agent_length,
        vehicle_no_slip_point_ratio=0.2289518863,
        cyclist_no_slip_point_ratio=0.0495847873,
    )

    control_norm, round_trip_error_m = build_rolling_control_target_with_round_trip_error(
        future_pos=future_pos,
        future_head=future_head,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=agent_type,
        agent_length=agent_length,
        vehicle_no_slip_point_ratio=0.2289518863,
        cyclist_no_slip_point_ratio=0.0495847873,
        **CONTROL_YAW_SCALE_KWARGS,
    )

    assert torch.equal(control_norm, expected_control_norm)
    torch.testing.assert_close(round_trip_error_m, expected_round_trip_error, atol=1.0e-5, rtol=1.0e-5)


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
