from __future__ import annotations

from typing import Optional

import torch


DEFAULT_LIGHT_TIME_MIN_SECONDS = -1.0
DEFAULT_LIGHT_TIME_MAX_SECONDS = 6.0
DEFAULT_LIGHT_TIME_NORMALIZER_SECONDS = 6.0
DEFAULT_WAYMO_CURRENT_RAW_STEP = 10
DEFAULT_SECONDS_PER_RAW_STEP = 0.1
NO_LANE_STATE_LIGHT_TYPE = 0


def validate_observed_current_raw_step(
    current_time_index: int,
    *,
    expected_raw_step: int = DEFAULT_WAYMO_CURRENT_RAW_STEP,
    scenario_id: str | None = None,
) -> int:
    current = int(current_time_index)
    expected = int(expected_raw_step)
    if current != expected:
        scenario_suffix = f" for scenario {scenario_id}" if scenario_id else ""
        raise ValueError(
            "Dynamic traffic-light staleness assumes the observed current raw step "
            f"is {expected}, but got current_time_index={current}{scenario_suffix}. "
            "Regenerate the cache with the standard WOMD current step or make the "
            "observed traffic-light raw step explicit in the model input."
        )
    return current


def normalize_light_time_delta_seconds(
    delta_seconds: torch.Tensor,
    *,
    min_seconds: float = DEFAULT_LIGHT_TIME_MIN_SECONDS,
    max_seconds: float = DEFAULT_LIGHT_TIME_MAX_SECONDS,
    normalizer_seconds: float = DEFAULT_LIGHT_TIME_NORMALIZER_SECONDS,
) -> torch.Tensor:
    if normalizer_seconds <= 0:
        raise ValueError(f"normalizer_seconds must be positive, got {normalizer_seconds}.")
    return delta_seconds.clamp(min=float(min_seconds), max=float(max_seconds)) / float(
        normalizer_seconds
    )


def mask_light_time_delta_norm_by_light_type(
    light_time_delta_norm: torch.Tensor,
    light_type: torch.Tensor,
) -> torch.Tensor:
    observed_signal = light_type.to(
        device=light_time_delta_norm.device,
        dtype=torch.long,
    ) != NO_LANE_STATE_LIGHT_TYPE
    return light_time_delta_norm.masked_fill(~observed_signal, 0.0)


def build_context_light_time_delta_norm(
    *,
    num_agents: int,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    shift_steps: int = 5,
    observed_raw_step: int = DEFAULT_WAYMO_CURRENT_RAW_STEP,
    seconds_per_raw_step: float = DEFAULT_SECONDS_PER_RAW_STEP,
) -> torch.Tensor:
    if num_agents < 0 or num_steps < 0:
        raise ValueError(
            f"num_agents and num_steps must be non-negative, got {num_agents}, {num_steps}."
        )
    if num_steps == 0:
        return torch.zeros((num_agents, 0), device=device, dtype=dtype)

    raw_steps = torch.arange(1, num_steps + 1, device=device, dtype=dtype) * float(
        shift_steps
    )
    delta_seconds = (raw_steps - float(observed_raw_step)) * float(seconds_per_raw_step)
    delta_norm = normalize_light_time_delta_seconds(delta_seconds)
    return delta_norm.view(1, num_steps).expand(num_agents, num_steps)


def build_constant_light_time_delta_norm(
    *,
    num_agents: int,
    num_steps: int,
    delta_seconds: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    base = torch.full((num_agents, num_steps), float(delta_seconds), device=device, dtype=dtype)
    return normalize_light_time_delta_seconds(base)


def resolve_light_time_delta_norm(
    *,
    light_time_delta_norm: Optional[torch.Tensor],
    num_agents: int,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    shift_steps: int = 5,
) -> torch.Tensor:
    if light_time_delta_norm is None:
        return build_context_light_time_delta_norm(
            num_agents=num_agents,
            num_steps=num_steps,
            device=device,
            dtype=dtype,
            shift_steps=shift_steps,
        )

    value = light_time_delta_norm.to(device=device, dtype=dtype)
    if value.ndim == 0:
        return value.view(1, 1).expand(num_agents, num_steps)
    if value.ndim == 1:
        if value.shape[0] != num_steps:
            raise ValueError(
                "1D light_time_delta_norm must have length num_steps, "
                f"got {value.shape[0]} and {num_steps}."
            )
        return value.view(1, num_steps).expand(num_agents, num_steps)
    if value.ndim == 2:
        expected = (num_agents, num_steps)
        if tuple(value.shape) != expected:
            raise ValueError(
                "2D light_time_delta_norm must have shape [num_agents, num_steps], "
                f"got {tuple(value.shape)} and {expected}."
            )
        return value
    raise ValueError(
        "light_time_delta_norm must be None, scalar, 1D, or 2D tensor, "
        f"got {value.ndim}D."
    )
