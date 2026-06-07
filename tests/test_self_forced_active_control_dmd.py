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
    compute_self_forced_dmd_injection_scale,
    normalize_pose_heading_vector,
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


def _active_rms(value: torch.Tensor, active_mask: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    active = active_mask.to(dtype=value.dtype).expand_as(value)
    denom = active.flatten(1).sum(dim=1, keepdim=True).clamp_min(1.0) + eps
    return ((value.square() * active).flatten(1).sum(dim=1, keepdim=True) / denom).sqrt()


def test_active_control_dmd_ignores_nonholonomic_lateral_axis_and_caps_step() -> None:
    committed = torch.tensor(
        [
            [
                [0.0, 100.0, 0.0],
                [0.0, 100.0, 0.0],
            ]
        ]
    )
    target_clean = torch.tensor(
        [
            [
                [0.01, 5.0, 0.01],
                [0.01, 5.0, 0.01],
            ]
        ]
    )
    generated_clean = torch.tensor(
        [
            [
                [-0.99, -5.0, -0.99],
                [-0.99, -5.0, -0.99],
            ]
        ]
    )
    active_mask = torch.tensor([[[1.0, 0.0, 1.0]]])

    direction = build_clean_dmd_direction(
        committed_path_norm=committed,
        target_clean_norm=target_clean,
        generated_clean_norm=generated_clean,
        active_mask=active_mask,
        normalizer_eps=0.05,
        use_stable_scale_filter=True,
        use_teacher_alignment_filter=True,
        use_trust_region_filter=True,
    )

    generator_teacher_rms = _active_rms(committed - target_clean, active_mask)
    direction_rms = _active_rms(direction, active_mask)
    torch.testing.assert_close(direction[:, :, 1], torch.zeros_like(direction[:, :, 1]))
    assert torch.all(direction_rms <= generator_teacher_rms + 1.0e-6)
    torch.testing.assert_close(
        direction,
        torch.tensor(
            [
                [
                    [0.01, 0.0, 0.01],
                    [0.01, 0.0, 0.01],
                ]
            ]
        ),
        atol=2.0e-6,
        rtol=2.0e-6,
    )


def test_active_control_dmd_drops_agent_when_direction_is_against_teacher() -> None:
    committed = torch.ones((1, 2, 3))
    target_clean = torch.zeros_like(committed)
    generated_clean = -torch.ones_like(committed)
    active_mask = torch.ones((1, 1, 3))

    direction = build_clean_dmd_direction(
        committed_path_norm=committed,
        target_clean_norm=target_clean,
        generated_clean_norm=generated_clean,
        active_mask=active_mask,
        normalizer_eps=0.05,
        use_stable_scale_filter=True,
        use_teacher_alignment_filter=True,
        use_trust_region_filter=False,
    )

    torch.testing.assert_close(direction, torch.zeros_like(direction))


def test_active_control_dmd_filters_are_independently_configurable() -> None:
    committed = torch.ones((1, 2, 3))
    target_clean = torch.zeros_like(committed)
    generated_clean = -torch.ones_like(committed)
    active_mask = torch.ones((1, 1, 3))

    raw_direction = build_clean_dmd_direction(
        committed_path_norm=committed,
        target_clean_norm=target_clean,
        generated_clean_norm=generated_clean,
        active_mask=active_mask,
        normalizer_eps=0.05,
        use_stable_scale_filter=False,
        use_teacher_alignment_filter=False,
        use_trust_region_filter=False,
    )
    torch.testing.assert_close(raw_direction, torch.ones_like(raw_direction))

    normalized_direction = build_clean_dmd_direction(
        committed_path_norm=committed,
        target_clean_norm=target_clean,
        generated_clean_norm=generated_clean,
        active_mask=active_mask,
        normalizer_eps=0.05,
        use_stable_scale_filter=True,
        use_teacher_alignment_filter=False,
        use_trust_region_filter=False,
    )
    torch.testing.assert_close(normalized_direction, torch.ones_like(normalized_direction))

    aligned_direction = build_clean_dmd_direction(
        committed_path_norm=committed,
        target_clean_norm=target_clean,
        generated_clean_norm=generated_clean,
        active_mask=active_mask,
        normalizer_eps=0.05,
        use_stable_scale_filter=True,
        use_teacher_alignment_filter=True,
        use_trust_region_filter=False,
    )
    torch.testing.assert_close(aligned_direction, torch.zeros_like(aligned_direction))


def test_active_control_dmd_stable_scale_ignores_teacher_estimator_rms() -> None:
    committed = torch.zeros((1, 2, 3))
    target_clean = torch.zeros_like(committed)
    generated_clean = -torch.ones_like(committed)
    active_mask = torch.ones((1, 1, 3))

    direction = build_clean_dmd_direction(
        committed_path_norm=committed,
        target_clean_norm=target_clean,
        generated_clean_norm=generated_clean,
        active_mask=active_mask,
        normalizer_eps=0.05,
        use_stable_scale_filter=True,
        use_teacher_alignment_filter=False,
        use_trust_region_filter=False,
    )

    torch.testing.assert_close(direction, torch.full_like(direction, 20.0))


def test_active_control_dmd_stable_scale_uses_generator_teacher_abs_mean() -> None:
    committed = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]])
    target_clean = torch.zeros_like(committed)
    generated_clean = -torch.ones_like(committed)
    active_mask = torch.ones((1, 1, 4))

    direction = build_clean_dmd_direction(
        committed_path_norm=committed,
        target_clean_norm=target_clean,
        generated_clean_norm=generated_clean,
        active_mask=active_mask,
        normalizer_eps=0.05,
        use_stable_scale_filter=True,
        use_teacher_alignment_filter=False,
        use_trust_region_filter=False,
    )

    torch.testing.assert_close(direction, torch.full_like(direction, 2.0))


def test_self_forced_dmd_injection_scale_ramps_for_two_epochs() -> None:
    assert compute_self_forced_dmd_injection_scale(current_epoch=0, dmd_start_epoch=0) == 0.25
    assert compute_self_forced_dmd_injection_scale(current_epoch=1, dmd_start_epoch=0) == 0.625
    assert compute_self_forced_dmd_injection_scale(current_epoch=2, dmd_start_epoch=0) == 1.0
    assert compute_self_forced_dmd_injection_scale(current_epoch=5, dmd_start_epoch=5) == 0.25


def test_pose_projected_dmd_heading_vector_is_renormalized() -> None:
    pose = torch.tensor(
        [
            [
                [0.1, 0.2, 3.0, 4.0],
                [0.2, 0.3, 0.0, 2.0],
            ]
        ],
        dtype=torch.float32,
    )

    normalized = normalize_pose_heading_vector(pose)

    torch.testing.assert_close(normalized[..., :2], pose[..., :2])
    torch.testing.assert_close(
        normalized[..., 2:].norm(dim=-1),
        torch.ones((1, 2)),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


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
        dmd_injection_scale=0.25,
    )
    loss.backward()

    assert target.requires_grad is False
    torch.testing.assert_close(target, torch.full_like(target, 0.25))
    torch.testing.assert_close(committed.grad[0, 0, 1], torch.tensor(0.0))
    torch.testing.assert_close(
        committed.grad,
        torch.tensor(
            [
                [[-0.0625, 0.0, -0.0625]],
                [[-1.0 / 24.0, -1.0 / 24.0, -1.0 / 24.0]],
            ]
        ),
    )
