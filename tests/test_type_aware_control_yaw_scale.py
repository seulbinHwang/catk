from __future__ import annotations

import torch

from src.smart.modules.kinematic_control import (
    CYCLIST_TYPE_ID,
    DEFAULT_CONTROL_CYCLIST_YAW_SCALE_RAD,
    DEFAULT_CONTROL_PEDESTRIAN_YAW_SCALE_RAD,
    DEFAULT_CONTROL_VEHICLE_YAW_SCALE_RAD,
    PEDESTRIAN_TYPE_ID,
    VEHICLE_TYPE_ID,
    control_norm_to_pose_norm,
    denormalize_control,
    normalize_control,
    resolve_control_yaw_scale,
)


def test_type_aware_yaw_scale_values_are_used() -> None:
    """agent 종류별 yaw scale이 의도한 값으로 선택되는지 확인합니다."""
    agent_type = torch.tensor([VEHICLE_TYPE_ID, PEDESTRIAN_TYPE_ID, CYCLIST_TYPE_ID])

    yaw_scale = resolve_control_yaw_scale(agent_type=agent_type, dtype=torch.float32)

    expected = torch.tensor(
        [
            DEFAULT_CONTROL_VEHICLE_YAW_SCALE_RAD,
            DEFAULT_CONTROL_PEDESTRIAN_YAW_SCALE_RAD,
            DEFAULT_CONTROL_CYCLIST_YAW_SCALE_RAD,
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(yaw_scale, expected)


def test_type_aware_normalize_and_denormalize_round_trip() -> None:
    """type-aware 정규화와 역정규화가 같은 실제 제어값으로 되돌아오는지 확인합니다."""
    agent_type = torch.tensor([VEHICLE_TYPE_ID, PEDESTRIAN_TYPE_ID, CYCLIST_TYPE_ID])
    control = torch.zeros((3, 1, 3), dtype=torch.float32)
    control[:, 0, 0] = torch.tensor([1.0, 1.0, 1.0])
    control[:, 0, 2] = torch.tensor(
        [
            DEFAULT_CONTROL_VEHICLE_YAW_SCALE_RAD,
            DEFAULT_CONTROL_PEDESTRIAN_YAW_SCALE_RAD,
            DEFAULT_CONTROL_CYCLIST_YAW_SCALE_RAD,
        ]
    )

    control_norm = normalize_control(control=control, agent_type=agent_type)
    recovered = denormalize_control(control_norm=control_norm, agent_type=agent_type)

    torch.testing.assert_close(control_norm[:, 0, 2], torch.ones(3))
    torch.testing.assert_close(recovered, control)


def test_scalar_fallback_still_works_without_agent_type() -> None:
    """agent_type을 주지 않는 기존 호출은 기존 scalar yaw scale을 그대로 쓰는지 확인합니다."""
    control = torch.zeros((1, 1, 3), dtype=torch.float32)
    control[..., 2] = DEFAULT_CONTROL_PEDESTRIAN_YAW_SCALE_RAD

    control_norm = normalize_control(control=control)
    recovered = denormalize_control(control_norm=control_norm)

    torch.testing.assert_close(control_norm[..., 2], torch.ones_like(control_norm[..., 2]))
    torch.testing.assert_close(recovered, control)


def test_control_norm_to_pose_norm_uses_type_aware_yaw_scale() -> None:
    """metric/rollout용 pose 복원에서도 type별 yaw scale이 적용되는지 확인합니다."""
    agent_type = torch.tensor([VEHICLE_TYPE_ID, PEDESTRIAN_TYPE_ID, CYCLIST_TYPE_ID])
    control_norm = torch.zeros((3, 1, 3), dtype=torch.float32)
    control_norm[:, 0, 2] = 1.0

    pose_norm = control_norm_to_pose_norm(control_norm=control_norm, agent_type=agent_type)
    decoded_yaw = torch.atan2(pose_norm[:, 0, 3], pose_norm[:, 0, 2])

    expected_yaw = torch.tensor(
        [
            DEFAULT_CONTROL_VEHICLE_YAW_SCALE_RAD,
            DEFAULT_CONTROL_PEDESTRIAN_YAW_SCALE_RAD,
            DEFAULT_CONTROL_CYCLIST_YAW_SCALE_RAD,
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(decoded_yaw, expected_yaw, atol=1.0e-6, rtol=1.0e-6)
