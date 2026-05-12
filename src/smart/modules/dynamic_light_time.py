from __future__ import annotations

from typing import Optional

import torch


DEFAULT_LIGHT_TIME_MIN_SECONDS = -1.0
DEFAULT_LIGHT_TIME_MAX_SECONDS = 6.0
DEFAULT_LIGHT_TIME_NORMALIZER_SECONDS = 6.0
DEFAULT_WAYMO_CURRENT_RAW_STEP = 10
DEFAULT_SECONDS_PER_RAW_STEP = 0.1


def validate_observed_current_raw_step(
    current_time_index: int,
    *,
    expected_raw_step: int = DEFAULT_WAYMO_CURRENT_RAW_STEP,
    scenario_id: str | None = None,
) -> int:
    """traffic-light 관측 시점이 모델의 stale-time 기준과 일치하는지 확인합니다."""
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
    """신호 관측 시점과 예측 기준 시점의 차이를 안정적인 입력값으로 바꿉니다.

    Args:
        delta_seconds: 예측 기준 시점에서 봤을 때, 현재 관측 신호가 몇 초 전 정보인지
            나타내는 값입니다. shape은 임의입니다.
        min_seconds: 너무 과거 문맥을 보지 않도록 허용할 최소 초 단위 값입니다.
        max_seconds: rollout 후반처럼 너무 오래된 신호를 하나의 오래된 정보로 묶기 위한
            최대 초 단위 값입니다.
        normalizer_seconds: 정규화에 사용할 초 단위 값입니다.

    Returns:
        torch.Tensor: clip 후 정규화한 시간 차입니다. shape은 ``delta_seconds`` 와 같습니다.
            예를 들어 0초는 0, 3초는 0.5, 6초 이상은 1이 됩니다.
    """
    if normalizer_seconds <= 0:
        raise ValueError(f"normalizer_seconds must be positive, got {normalizer_seconds}.")
    return delta_seconds.clamp(min=float(min_seconds), max=float(max_seconds)) / float(
        normalizer_seconds
    )


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
    """0.5초 context 칸마다 현재 신호가 얼마나 오래된 정보인지 만듭니다.

    Args:
        num_agents: 장면 batch 안 agent 수입니다.
        num_steps: context 칸 개수입니다. 보통 pretrain에서는 14입니다.
        device: 반환 tensor를 둘 장치입니다.
        dtype: 반환 tensor 자료형입니다.
        shift_steps: 한 context 칸이 몇 개의 10Hz step을 건너뛰는지 나타냅니다.
        observed_raw_step: 신호가 관측된 원본 10Hz 시점입니다. WOMD 현재 시점은 보통 10입니다.
        seconds_per_raw_step: 원본 한 step의 초 단위 길이입니다.

    Returns:
        torch.Tensor: 각 agent와 context 칸의 정규화된 신호 시간 차입니다.
            shape은 ``[num_agents, num_steps]`` 입니다.
            14칸 기준 값은 대략 ``[-0.083, 0, 0.083, ..., 1]`` 입니다.
    """
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
    """모든 agent와 step에 같은 신호 시간 차를 채웁니다.

    Args:
        num_agents: 장면 batch 안 agent 수입니다.
        num_steps: 같은 값을 넣을 시간 칸 개수입니다.
        delta_seconds: 신호 관측 후 지난 시간입니다. 초 단위 scalar입니다.
        device: 반환 tensor를 둘 장치입니다.
        dtype: 반환 tensor 자료형입니다.

    Returns:
        torch.Tensor: 정규화된 신호 시간 차입니다. shape은 ``[num_agents, num_steps]`` 입니다.
    """
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
    """외부 입력 또는 기본 context 규칙으로 신호 시간 차를 준비합니다.

    Args:
        light_time_delta_norm: 이미 정규화된 시간 차입니다. ``None`` 이면 0.5초 context 칸
            규칙으로 자동 생성합니다. shape은 ``[num_agents, num_steps]``, ``[num_steps]``
            또는 scalar를 허용합니다.
        num_agents: 장면 batch 안 agent 수입니다.
        num_steps: agent 시간 칸 개수입니다.
        device: 반환 tensor를 둘 장치입니다.
        dtype: 반환 tensor 자료형입니다.
        shift_steps: 기본 context 규칙에서 한 칸이 몇 raw step인지 나타냅니다.

    Returns:
        torch.Tensor: agent와 시간 칸별 정규화된 신호 시간 차입니다.
            shape은 ``[num_agents, num_steps]`` 입니다.
    """
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
