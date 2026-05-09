from __future__ import annotations

import torch

from src.smart.modules.kinematic_control import (
    CYCLIST_TYPE_ID,
    PEDESTRIAN_TYPE_ID,
    VEHICLE_TYPE_ID,
    control_norm_to_pose_norm,
    denormalize_control,
    normalize_control,
    resolve_control_yaw_scale,
)

CONTROL_YAW_SCALE_KWARGS = {
    "vehicle_yaw_scale_rad": 0.031,
    "pedestrian_yaw_scale_rad": 0.23,
    "cyclist_yaw_scale_rad": 0.071,
}


def test_type_aware_yaw_scale_values_are_used() -> None:
    """agent 종류별 yaw scale이 config 값으로 선택되는지 확인합니다."""
    agent_type = torch.tensor([VEHICLE_TYPE_ID, PEDESTRIAN_TYPE_ID, CYCLIST_TYPE_ID])

    yaw_scale = resolve_control_yaw_scale(
        agent_type=agent_type,
        dtype=torch.float32,
        **CONTROL_YAW_SCALE_KWARGS,
    )

    expected = torch.tensor(
        [
            CONTROL_YAW_SCALE_KWARGS["vehicle_yaw_scale_rad"],
            CONTROL_YAW_SCALE_KWARGS["pedestrian_yaw_scale_rad"],
            CONTROL_YAW_SCALE_KWARGS["cyclist_yaw_scale_rad"],
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
            CONTROL_YAW_SCALE_KWARGS["vehicle_yaw_scale_rad"],
            CONTROL_YAW_SCALE_KWARGS["pedestrian_yaw_scale_rad"],
            CONTROL_YAW_SCALE_KWARGS["cyclist_yaw_scale_rad"],
        ]
    )

    control_norm = normalize_control(
        control=control,
        agent_type=agent_type,
        **CONTROL_YAW_SCALE_KWARGS,
    )
    recovered = denormalize_control(
        control_norm=control_norm,
        agent_type=agent_type,
        **CONTROL_YAW_SCALE_KWARGS,
    )

    torch.testing.assert_close(control_norm[:, 0, 2], torch.ones(3))
    torch.testing.assert_close(recovered, control)


def test_agent_type_is_required_for_control_normalization() -> None:
    """control 정규화/역정규화는 agent type 없이는 실행되지 않아야 합니다."""
    control = torch.zeros((1, 1, 3), dtype=torch.float32)
    control[..., 2] = CONTROL_YAW_SCALE_KWARGS["pedestrian_yaw_scale_rad"]

    try:
        normalize_control(control=control, **CONTROL_YAW_SCALE_KWARGS)
    except TypeError:
        pass
    else:
        raise AssertionError("normalize_control must require agent_type.")

    try:
        denormalize_control(control_norm=control, **CONTROL_YAW_SCALE_KWARGS)
    except TypeError:
        pass
    else:
        raise AssertionError("denormalize_control must require agent_type.")


def test_control_yaw_scale_config_is_required_for_control_normalization() -> None:
    """control yaw scale config 없이 숨은 fallback으로 실행되면 안 됩니다."""
    agent_type = torch.tensor([PEDESTRIAN_TYPE_ID])
    control = torch.zeros((1, 1, 3), dtype=torch.float32)

    try:
        normalize_control(control=control, agent_type=agent_type)
    except TypeError:
        pass
    else:
        raise AssertionError("normalize_control must require explicit yaw scale config.")

    try:
        denormalize_control(control_norm=control, agent_type=agent_type)
    except TypeError:
        pass
    else:
        raise AssertionError("denormalize_control must require explicit yaw scale config.")


def test_control_norm_to_pose_norm_uses_type_aware_yaw_scale() -> None:
    """metric/rollout용 pose 복원에서도 type별 yaw scale이 적용되는지 확인합니다."""
    agent_type = torch.tensor([VEHICLE_TYPE_ID, PEDESTRIAN_TYPE_ID, CYCLIST_TYPE_ID])
    control_norm = torch.zeros((3, 1, 3), dtype=torch.float32)
    control_norm[:, 0, 2] = 1.0

    pose_norm = control_norm_to_pose_norm(
        control_norm=control_norm,
        agent_type=agent_type,
        **CONTROL_YAW_SCALE_KWARGS,
    )
    decoded_yaw = torch.atan2(pose_norm[:, 0, 3], pose_norm[:, 0, 2])

    expected_yaw = torch.tensor(
        [
            CONTROL_YAW_SCALE_KWARGS["vehicle_yaw_scale_rad"],
            CONTROL_YAW_SCALE_KWARGS["pedestrian_yaw_scale_rad"],
            CONTROL_YAW_SCALE_KWARGS["cyclist_yaw_scale_rad"],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(decoded_yaw, expected_yaw, atol=1.0e-6, rtol=1.0e-6)
