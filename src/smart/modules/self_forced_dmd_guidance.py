from __future__ import annotations

import torch
from torch import Tensor


def build_clean_dmd_direction(
    committed_path_norm: Tensor,
    target_clean_norm: Tensor,
    generated_clean_norm: Tensor,
    normalizer_eps: float = 1.0e-3,
    channel_mask: Tensor | None = None,
    per_channel_normalizer: bool = True,
    normalize_direction: bool = True,
) -> Tensor:
    """teacher와 generated estimator의 clean path 차이로 DMD 방향을 만듭니다.

    Args:
        committed_path_norm: Generator가 closed-loop로 실제 실행한 path입니다.
            shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
        target_clean_norm: frozen teacher가 같은 noisy path에서 추정한 clean path입니다.
            shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
        generated_clean_norm: generated estimator가 같은 noisy path에서 추정한 clean path입니다.
            shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
        normalizer_eps: agent별 정규화 분모의 최소값입니다.

    Returns:
        Tensor: 현재 committed path에 더할 정규화된 DMD 방향입니다.
        shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.

    설명:
        이 함수는 raw velocity 차이나 시간/노이즈 계수가 섞인 값을 그대로 쓰지 않습니다.
        먼저 ``target_clean_norm - generated_clean_norm`` 방향을 만들고, 각 agent의 전체
        미래 path 기준으로 ``committed_path_norm``과 ``target_clean_norm`` 사이의 평균
        거리로 나눕니다. 이렇게 하면 teacher가 보는 방향은 유지하면서도 특정 tau 구간에서
        target path가 과하게 커지는 문제를 줄일 수 있습니다.
    """
    expected_shape = tuple(committed_path_norm.shape)
    if tuple(target_clean_norm.shape) != expected_shape:
        raise ValueError(
            "target_clean_norm shape must match committed_path_norm shape: "
            f"expected={expected_shape}, actual={tuple(target_clean_norm.shape)}."
        )
    if tuple(generated_clean_norm.shape) != expected_shape:
        raise ValueError(
            "generated_clean_norm shape must match committed_path_norm shape: "
            f"expected={expected_shape}, actual={tuple(generated_clean_norm.shape)}."
        )
    if committed_path_norm.dim() < 2:
        raise ValueError(
            "committed_path_norm must have at least agent and path dimensions, "
            f"got shape={expected_shape}."
        )

    committed = committed_path_norm.float()
    target_clean = target_clean_norm.float()
    generated_clean = generated_clean_norm.float()

    clean_dmd_direction = target_clean - generated_clean
    # normalize_direction=False (권장): 거리-나눗셈 제거. dir = raw (teacher - fake).
    # 표준 DMD/VSD 형태로, generator가 teacher 분포에 가까워지면 (teacher-fake)→0 이라
    # push 가 자연히 사라져 수렴한다.  거리(committed-teacher)로 나누면 가까워질수록
    # 분모↓ → push↑ (clamp 에서 폭발)로 수렴이 깨져 발산하던 문제를 없앤다.
    # path_step_size 가 고정 계수 역할(raw gap 이 작으므로 더 큰 값 필요).
    if normalize_direction:
        # per_channel_normalizer=True: 시간축만 평균하고 채널축은 남겨 채널 스케일 균형.
        if per_channel_normalizer:
            reduce_dims = tuple(range(1, committed.dim() - 1))  # 시간축만 (채널 유지)
        else:
            reduce_dims = tuple(range(1, committed.dim()))
        agent_distance = (committed - target_clean).abs().mean(
            dim=reduce_dims,
            keepdim=True,
        )
        normalizer = agent_distance.clamp_min(float(normalizer_eps))
        normalized_direction = clean_dmd_direction / normalizer
    else:
        normalized_direction = clean_dmd_direction
    if channel_mask is not None:
        # 죽은 채널(예: non-holonomic delta_n) 제외 — direction을 0으로.
        normalized_direction = normalized_direction * channel_mask.to(
            device=normalized_direction.device,
            dtype=normalized_direction.dtype,
        )
    normalized_direction = torch.nan_to_num(
        normalized_direction,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return normalized_direction.to(dtype=committed_path_norm.dtype)
