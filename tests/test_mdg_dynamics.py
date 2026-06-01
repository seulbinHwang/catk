import math

import torch

from src.mdg.modules import KinematicDynamics
from src.mdg.geometry import global_to_local_xy, heading_vector, wrap_angle


def loop_reference_dynamics(dynamics, action, current_pos, current_heading, current_speed, current_velocity=None):
    action = dynamics.denormalize(action)
    acc = action[..., 0]
    yaw_rate = action[..., 1]
    pos = current_pos
    heading = current_heading
    speed = current_speed
    velocity = current_velocity if current_velocity is not None else heading_vector(heading) * speed.unsqueeze(-1)
    full_pos = []
    full_heading = []
    full_speed = []
    chunk_state = []
    for idx in range(int(action.shape[2])):
        for _ in range(dynamics.action_chunk):
            pos = pos + velocity * dynamics.dt
            heading = wrap_angle(heading + yaw_rate[:, :, idx] * dynamics.dt)
            speed = speed + acc[:, :, idx] * dynamics.dt
            velocity = heading_vector(heading) * speed.unsqueeze(-1)
            full_pos.append(pos)
            full_heading.append(heading)
            full_speed.append(speed)
        local_pos = global_to_local_xy(pos, current_pos, current_heading)
        local_heading = wrap_angle(heading - current_heading)
        chunk_state.append(
            torch.cat(
                (
                    local_pos,
                    torch.cos(local_heading).unsqueeze(-1),
                    torch.sin(local_heading).unsqueeze(-1),
                    speed.unsqueeze(-1),
                ),
                dim=-1,
            )
        )
    return (
        torch.stack(full_pos, dim=2),
        torch.stack(full_heading, dim=2),
        torch.stack(full_speed, dim=2),
        torch.stack(chunk_state, dim=2),
        action,
    )


def test_kinematic_dynamics_uses_vbd_update_order():
    dynamics = KinematicDynamics(
        action_chunk=1,
        dt=1.0,
        action_mean=(0.0, 0.0),
        action_std=(1.0, 1.0),
    )
    action = torch.tensor([[[[1.0, math.pi / 2.0]]]])
    pos, heading, speed, _, _ = dynamics(
        action=action,
        current_pos=torch.tensor([[[0.0, 0.0]]]),
        current_heading=torch.tensor([[0.0]]),
        current_speed=torch.tensor([[1.0]]),
    )

    torch.testing.assert_close(pos[0, 0, 0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(heading[0, 0, 0], torch.tensor(math.pi / 2.0))
    torch.testing.assert_close(speed[0, 0, 0], torch.tensor(2.0))


def test_kinematic_dynamics_advances_position_with_current_velocity_vector():
    dynamics = KinematicDynamics(
        action_chunk=1,
        dt=1.0,
        action_mean=(0.0, 0.0),
        action_std=(1.0, 1.0),
    )
    pos, _, _, _, _ = dynamics(
        action=torch.zeros(1, 1, 1, 2),
        current_pos=torch.tensor([[[0.0, 0.0]]]),
        current_heading=torch.tensor([[math.pi / 2.0]]),
        current_speed=torch.tensor([[1.0]]),
        current_velocity=torch.tensor([[[1.0, 0.0]]]),
    )

    torch.testing.assert_close(pos[0, 0, 0], torch.tensor([1.0, 0.0]))


def test_trajectory_to_actions_uses_position_consistent_speed():
    dynamics = KinematicDynamics(
        action_chunk=2,
        dt=1.0,
        action_mean=(0.0, 0.0),
        action_std=(1.0, 1.0),
    )
    future_pos = torch.tensor([[[[2.0, 0.0], [5.0, 0.0], [9.0, 0.0], [14.0, 0.0]]]])
    future_velocity = torch.tensor([[[[99.0, 0.0], [99.0, 0.0], [99.0, 0.0], [5.0, 0.0]]]])
    future_heading = torch.zeros(1, 1, 4)
    action = dynamics.trajectory_to_actions(
        current_pos=torch.zeros(1, 1, 2),
        current_heading=torch.tensor([[0.0]]),
        current_speed=torch.tensor([[1.0]]),
        future_pos=future_pos,
        future_heading=future_heading,
        future_velocity=future_velocity,
    )

    expected = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    torch.testing.assert_close(action, expected)


def test_vectorized_dynamics_matches_loop_reference():
    torch.manual_seed(3)
    dynamics = KinematicDynamics(
        action_chunk=2,
        dt=0.1,
        action_mean=(0.0, 0.0),
        action_std=(1.0, 0.5),
    )
    action = torch.randn(2, 4, 40, 2) * torch.tensor([0.5, 0.2])
    current_pos = torch.randn(2, 4, 2)
    current_heading = torch.randn(2, 4)
    current_speed = torch.rand(2, 4) * 8.0
    current_velocity = torch.randn(2, 4, 2)

    expected = loop_reference_dynamics(
        dynamics,
        action,
        current_pos,
        current_heading,
        current_speed,
        current_velocity=current_velocity,
    )
    actual = dynamics(
        action,
        current_pos,
        current_heading,
        current_speed,
        current_velocity=current_velocity,
    )

    for expected_tensor, actual_tensor in zip(expected, actual):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-4, atol=1e-5)
