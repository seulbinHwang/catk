from __future__ import annotations

from collections import OrderedDict

import torch

from src.smart.metrics.rlftsim_reward import compute_mloo_rewards_from_leave_one_out
from src.smart.model.smart import SMART


def test_mloo_rewards_are_mean_minus_leave_one_out_and_zero_sum() -> None:
    leave_one_out = torch.tensor(
        [
            [0.7, 0.9, 0.8, 1.0],
            [0.2, 0.1, 0.4, 0.3],
        ],
        dtype=torch.float32,
    )

    rewards = compute_mloo_rewards_from_leave_one_out(leave_one_out)

    expected = leave_one_out.mean(dim=1, keepdim=True) - leave_one_out
    torch.testing.assert_close(rewards, expected)
    torch.testing.assert_close(rewards.sum(dim=1), torch.zeros(2))


def test_mloo_rewards_reject_single_rollout() -> None:
    try:
        compute_mloo_rewards_from_leave_one_out(torch.zeros(2, 1))
    except ValueError as exc:
        assert "at least two rollouts" in str(exc)
    else:
        raise AssertionError("single-rollout MLOO should fail")


def test_rlftsim_reference_encoder_state_is_not_persistent() -> None:
    state_dict = OrderedDict(
        [
            ("encoder.agent_encoder.weight", torch.tensor([1.0])),
            ("_rlftsim_ref_encoder.agent_encoder.weight", torch.tensor([2.0])),
        ]
    )

    filtered = SMART._drop_rlftsim_reference_state(state_dict)

    assert list(filtered.keys()) == ["encoder.agent_encoder.weight"]
    torch.testing.assert_close(
        filtered["encoder.agent_encoder.weight"],
        torch.tensor([1.0]),
    )
