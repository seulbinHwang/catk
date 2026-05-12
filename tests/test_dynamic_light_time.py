from __future__ import annotations

import pytest
import torch

from src.smart.modules.dynamic_light_time import (
    build_constant_light_time_delta_norm,
    build_context_light_time_delta_norm,
    normalize_light_time_delta_seconds,
    validate_observed_current_raw_step,
)


def test_normalize_light_time_delta_seconds_clips_and_scales() -> None:
    value = torch.tensor([-2.0, -1.0, 0.0, 3.0, 6.0, 8.0])
    normalized = normalize_light_time_delta_seconds(value)
    expected = torch.tensor([-1.0 / 6.0, -1.0 / 6.0, 0.0, 0.5, 1.0, 1.0])
    assert torch.allclose(normalized, expected)


def test_observed_current_raw_step_guard_accepts_waymo_standard_current() -> None:
    assert validate_observed_current_raw_step(10, scenario_id="ok") == 10


def test_observed_current_raw_step_guard_rejects_nonstandard_current() -> None:
    with pytest.raises(ValueError, match="current_time_index=12"):
        validate_observed_current_raw_step(12, scenario_id="bad")


def test_context_light_time_delta_matches_waymo_context_slots() -> None:
    delta = build_context_light_time_delta_norm(
        num_agents=2,
        num_steps=14,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert delta.shape == (2, 14)
    assert torch.allclose(delta[0, 0], torch.tensor(-0.5 / 6.0))
    assert torch.allclose(delta[0, 1], torch.tensor(0.0))
    assert torch.allclose(delta[0, -1], torch.tensor(1.0))
    assert torch.allclose(delta[0], delta[1])


def test_constant_light_time_delta_handles_rollout_tail() -> None:
    delta = build_constant_light_time_delta_norm(
        num_agents=3,
        num_steps=1,
        delta_seconds=7.5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert delta.shape == (3, 1)
    assert torch.allclose(delta, torch.ones_like(delta))
