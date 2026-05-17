from __future__ import annotations

import math

import pytest
import torch

from src.smart.modules.kinematic_control import (
    CYCLIST_TYPE_ID,
    PEDESTRIAN_TYPE_ID,
    VEHICLE_TYPE_ID,
    build_rolling_control_target,
    build_transition_aligned_control_trajectory,
    compute_aligned_substep_validity,
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


def test_holonomic_model_only_lets_vehicle_use_lateral_channel_and_reconstruct() -> None:
    current_pos = torch.tensor([[0.0, 0.0]])
    current_head = torch.tensor([0.0])
    future_pos = torch.tensor([[[1.0, 0.2], [2.0, 0.5]]])
    future_head = torch.tensor([[0.1, 0.2]])
    agent_type = torch.tensor([VEHICLE_TYPE_ID])

    control_norm = build_rolling_control_target(
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


def test_transition_aligned_trajectory_keeps_history_raw_and_projects_future() -> None:
    pos = torch.zeros((1, 4, 2), dtype=torch.float32)
    pos[0, 0] = torch.tensor([-1.0, 0.5])
    pos[0, 1] = torch.tensor([0.0, 0.0])
    pos[0, 2] = torch.tensor([0.0, 6.0])
    pos[0, 3] = torch.tensor([1.0, 6.0])
    heading = torch.zeros((1, 4), dtype=torch.float32)
    agent_type = torch.tensor([VEHICLE_TYPE_ID])

    aligned_pos, aligned_head, control_norm_by_step = build_transition_aligned_control_trajectory(
        pos=pos,
        heading=heading,
        agent_type=agent_type,
        current_step=1,
        **CONTROL_YAW_SCALE_KWARGS,
    )

    control = denormalize_control(
        control_norm_by_step[:, 2:],
        agent_type=agent_type,
        **CONTROL_YAW_SCALE_KWARGS,
    )

    torch.testing.assert_close(aligned_pos[:, :2], pos[:, :2])
    torch.testing.assert_close(aligned_head[:, :2], heading[:, :2])
    torch.testing.assert_close(control[..., 1], torch.zeros_like(control[..., 1]))
    torch.testing.assert_close(aligned_pos[:, 2, 1], torch.zeros(1), atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(aligned_head[:, 2:], heading[:, 2:], atol=1.0e-5, rtol=1.0e-5)


def test_transition_aligned_vehicle_uses_block_endpoint_substeps() -> None:
    pos = torch.zeros((1, 6, 2), dtype=torch.float32)
    pos[0, 1:5, 1] = 10.0
    pos[0, 5] = torch.tensor([5.0, 0.0])
    heading = torch.zeros((1, 6), dtype=torch.float32)
    agent_type = torch.tensor([VEHICLE_TYPE_ID])

    aligned_pos, aligned_head, control_norm_by_step = build_transition_aligned_control_trajectory(
        pos=pos,
        heading=heading,
        agent_type=agent_type,
        current_step=0,
        commit_steps=5,
        **CONTROL_YAW_SCALE_KWARGS,
    )

    expected_x = torch.arange(1, 6, dtype=torch.float32)
    torch.testing.assert_close(aligned_pos[0, 1:6, 0], expected_x)
    torch.testing.assert_close(aligned_pos[0, 1:6, 1], torch.zeros(5))
    torch.testing.assert_close(aligned_head[0, 1:6], torch.zeros(5))
    torch.testing.assert_close(control_norm_by_step[0, 1:6, 0], torch.ones(5))
    torch.testing.assert_close(control_norm_by_step[0, 1:6, 1], torch.zeros(5))


def test_transition_aligned_pedestrian_interpolates_block_endpoint() -> None:
    pos = torch.zeros((1, 6, 2), dtype=torch.float32)
    pos[0, 1:5, 0] = -3.0
    pos[0, 1:5, 1] = 7.0
    pos[0, 5] = torch.tensor([5.0, 5.0])
    heading = torch.zeros((1, 6), dtype=torch.float32)
    heading[0, 5] = 0.5
    agent_type = torch.tensor([PEDESTRIAN_TYPE_ID])

    aligned_pos, aligned_head, _ = build_transition_aligned_control_trajectory(
        pos=pos,
        heading=heading,
        agent_type=agent_type,
        current_step=0,
        commit_steps=5,
        **CONTROL_YAW_SCALE_KWARGS,
    )

    expected_xy = torch.arange(1, 6, dtype=torch.float32).unsqueeze(-1).repeat(1, 2)
    expected_head = torch.linspace(0.1, 0.5, 5)
    torch.testing.assert_close(aligned_pos[0, 1:6], expected_xy, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(aligned_head[0, 1:6], expected_head, atol=1.0e-5, rtol=1.0e-5)


def test_pedestrian_uses_pedestrian_type_id_constant() -> None:
    assert PEDESTRIAN_TYPE_ID == 1
    assert VEHICLE_TYPE_ID == 0
    assert CYCLIST_TYPE_ID == 2


def test_aligned_substep_validity_handles_all_valid_trajectory() -> None:
    valid = torch.ones((2, 11), dtype=torch.bool)
    aligned_valid = compute_aligned_substep_validity(valid, current_step=0, commit_steps=5)
    assert aligned_valid.shape == valid.shape
    assert torch.equal(aligned_valid, torch.ones_like(valid))


def test_aligned_substep_validity_flags_block_endpoint_invalid() -> None:
    # 2 agents, T=11, current_step=0, commit_steps=5.
    valid = torch.ones((2, 11), dtype=torch.bool)
    valid[1, 5] = False  # invalid endpoint of block (0, 5]

    aligned_valid = compute_aligned_substep_validity(valid, current_step=0, commit_steps=5)
    # Agent 0: all-valid → all True.
    assert torch.equal(aligned_valid[0], torch.ones(11, dtype=torch.bool))
    # Agent 1:
    #   step 0   : valid[0]=True → True
    #   steps 1-4: clean[0] AND valid[5] = True AND False = False
    #   step 5   : endpoint, valid[5] = False
    #   steps 6-9: clean[5] AND valid[10] = False AND True = False
    #   step 10  : endpoint, valid[10] = True
    expected = torch.tensor(
        [True, False, False, False, False, False, False, False, False, False, True]
    )
    assert torch.equal(aligned_valid[1], expected)


def test_aligned_substep_validity_recovers_after_re_anchor() -> None:
    # invalid endpoint then valid endpoint should recover at the next valid endpoint.
    # current_step=0, commit_steps=5, T=16: blocks (0,5], (5,10], (10,15].
    valid = torch.ones((1, 16), dtype=torch.bool)
    valid[0, 5] = False  # invalid endpoint of first block
    aligned_valid = compute_aligned_substep_validity(valid, current_step=0, commit_steps=5)
    expected = torch.tensor(
        [True,  # step 0 current
         False, False, False, False,  # block (0,5] mid: clean[0]=True AND valid[5]=False
         False,  # block (0,5] endpoint: valid[5]=False
         False, False, False, False,  # block (5,10] mid: clean[5]=False AND valid[10]=True = False
         True,  # block (5,10] endpoint: valid[10]=True, re-anchors
         True, True, True, True,  # block (10,15] mid: clean[10]=True AND valid[15]=True = True
         True]  # block (10,15] endpoint
    )
    assert torch.equal(aligned_valid[0], expected)


def test_aligned_substep_validity_propagates_from_invalid_current_step() -> None:
    # If current_step itself is invalid, the whole rolling is unreliable until
    # a re-anchor at the next valid block endpoint.
    valid = torch.ones((1, 11), dtype=torch.bool)
    valid[0, 0] = False  # current_step invalid
    aligned_valid = compute_aligned_substep_validity(valid, current_step=0, commit_steps=5)
    # step 0: valid[0]=False → False
    # mid 1-4: clean[0]=False AND valid[5]=True = False
    # endpoint 5: valid[5]=True → True (re-anchors)
    # mid 6-9: clean[5]=True AND valid[10]=True = True
    # endpoint 10: valid[10]=True → True
    expected = torch.tensor(
        [False, False, False, False, False, True, True, True, True, True, True]
    )
    assert torch.equal(aligned_valid[0], expected)


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
