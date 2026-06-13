from __future__ import annotations

import torch

from src.smart.modules.flow_local_decoder import PlanFirstResidualVelocityHead


def test_plan_first_head_starts_as_frame_velocity_head() -> None:
    torch.manual_seed(7)
    head = PlanFirstResidualVelocityHead(flow_dim=16, flow_state_dim=3)
    step_tokens = torch.randn(2, 20, 16)
    chunk_tokens = torch.randn(2, 4, 16)
    plan_token = torch.randn(2, 16)
    tau = torch.tensor([0.02, 0.95])

    output = head(
        step_tokens=step_tokens,
        chunk_tokens=chunk_tokens,
        plan_token=plan_token,
        tau=tau,
    )
    expected = head.frame_velocity_head(step_tokens)

    assert torch.allclose(output, expected, atol=1.0e-7, rtol=1.0e-7)


def test_tau_does_not_suppress_frame_velocity() -> None:
    torch.manual_seed(11)
    head = PlanFirstResidualVelocityHead(flow_dim=16, flow_state_dim=3)
    step_tokens = torch.randn(2, 20, 16)
    chunk_tokens = torch.randn(2, 4, 16)
    plan_token = torch.randn(2, 16)
    low_tau = torch.full((2,), 0.01)
    high_tau = torch.full((2,), 0.99)

    low_output = head(
        step_tokens=step_tokens,
        chunk_tokens=chunk_tokens,
        plan_token=plan_token,
        tau=low_tau,
    )
    high_output = head(
        step_tokens=step_tokens,
        chunk_tokens=chunk_tokens,
        plan_token=plan_token,
        tau=high_tau,
    )

    assert torch.allclose(low_output, high_output, atol=1.0e-7, rtol=1.0e-7)


def test_invalid_chunk_fill_uses_nearest_valid_chunk() -> None:
    head = PlanFirstResidualVelocityHead(flow_dim=4, flow_state_dim=3)
    chunk_value = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [0.0, 0.0, 0.0],
                [10.0, 11.0, 12.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [7.0, 8.0, 9.0],
                [0.0, 0.0, 0.0],
                [13.0, 14.0, 15.0],
            ],
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
            ],
        ]
    )
    chunk_valid_mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, True],
            [False, False, False, False],
        ]
    )

    filled = head._fill_invalid_chunk_velocity(chunk_value, chunk_valid_mask)

    assert torch.equal(filled[0, 1], chunk_value[0, 0])
    assert torch.equal(filled[0, 3], chunk_value[0, 2])
    assert torch.equal(filled[1, 0], chunk_value[1, 1])
    assert torch.equal(filled[1, 2], chunk_value[1, 1])
    assert torch.equal(filled[2], torch.zeros_like(filled[2]))


def test_chunk_bias_expands_by_chunk_repeat() -> None:
    head = PlanFirstResidualVelocityHead(flow_dim=4, flow_state_dim=1)
    chunk_bias = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])

    expanded = head._expand_chunk_bias(chunk_bias, num_steps=20)

    expected = torch.tensor([[[1.0]] * 5 + [[2.0]] * 5 + [[3.0]] * 5 + [[4.0]] * 5])
    assert torch.equal(expanded, expected)
