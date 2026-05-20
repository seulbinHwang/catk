import torch
import pytest

from src.smart.modules.dynamic_light_time import (
    build_constant_light_time_delta_norm,
    build_context_light_time_delta_norm,
    mask_light_time_delta_norm_by_light_type,
    normalize_light_time_delta_seconds,
    resolve_light_time_delta_norm,
    resolve_step_light_time_delta_norm,
    validate_observed_current_raw_step,
)


def test_normalize_light_time_delta_seconds_clips_to_expected_range():
    delta = torch.tensor([-2.0, 0.0, 3.0, 9.0])
    actual = normalize_light_time_delta_seconds(delta)
    expected = torch.tensor([-1.0 / 6.0, 0.0, 0.5, 1.0])
    torch.testing.assert_close(actual, expected)


def test_context_light_time_matches_womd_current_step_convention():
    actual = build_context_light_time_delta_norm(
        num_agents=2,
        num_steps=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected_row = torch.tensor([-0.5 / 6.0, 0.0, 0.5 / 6.0])
    torch.testing.assert_close(actual, expected_row.view(1, 3).expand(2, 3))


def test_constant_light_time_uses_rollout_elapsed_time():
    actual = build_constant_light_time_delta_norm(
        num_agents=2,
        num_steps=1,
        delta_seconds=1.5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, torch.full((2, 1), 1.5 / 6.0))


def test_no_signal_map_elements_do_not_receive_stale_time():
    light_time = torch.tensor([0.25, 0.25, 0.25])
    light_type = torch.tensor([0, 1, 3])
    actual = mask_light_time_delta_norm_by_light_type(light_time, light_type)
    expected = torch.tensor([0.0, 0.25, 0.25])
    torch.testing.assert_close(actual, expected)


def test_resolve_light_time_accepts_scalar_and_checks_shape():
    actual = resolve_light_time_delta_norm(
        light_time_delta_norm=torch.tensor(0.5),
        num_agents=2,
        num_steps=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, torch.full((2, 3), 0.5))

    with pytest.raises(ValueError):
        resolve_light_time_delta_norm(
            light_time_delta_norm=torch.ones(4),
            num_agents=2,
            num_steps=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_resolve_step_light_time_reduces_to_time_axis():
    actual = resolve_step_light_time_delta_norm(
        light_time_delta_norm=torch.tensor([[0.1, 0.2], [0.1, 0.2]]),
        num_agents=2,
        num_steps=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, torch.tensor([0.1, 0.2]))

    with pytest.raises(ValueError):
        resolve_step_light_time_delta_norm(
            light_time_delta_norm=torch.ones(3, 2),
            num_agents=2,
            num_steps=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_current_raw_step_validation_is_strict():
    assert validate_observed_current_raw_step(10) == 10
    with pytest.raises(ValueError):
        validate_observed_current_raw_step(11, scenario_id="toy")
