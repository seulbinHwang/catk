from __future__ import annotations

import torch
from torch import Tensor


def build_clean_dmd_direction(
    committed_path_norm: Tensor,
    target_clean_norm: Tensor,
    generated_clean_norm: Tensor,
    normalizer_eps: float = 1.0e-3,
    channel_mask: Tensor | None = None,
    per_channel_normalizer: bool = False,
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

    # 죽은 채널(예: non-holonomic delta_n) mask 를 미리 준비한다.  이 mask 는 direction
    # 과 normalizer 양쪽에 모두 적용해, 죽은 채널을 "아예 없는 tensor"처럼 다룬다.
    mask = (
        channel_mask.to(device=committed.device, dtype=committed.dtype)
        if channel_mask is not None
        else None
    )

    clean_dmd_direction = target_clean - generated_clean
    # 죽은 채널을 direction 단계에서 먼저 0으로 만들어, 아래 정규화 분모(평균)에도
    # 끼지 않게 한다.
    if mask is not None:
        clean_dmd_direction = clean_dmd_direction * mask

    # 원본 Self-Forcing(DMD) 정합: normalizer = |x0 - real|.mean(dim=[1..]) 처럼
    # 시간+채널 전체를 평균해 agent 당 단일 스칼라를 쓴다(per_channel_normalizer=False).
    # 시간축만 평균(per_channel=True)하던 기존 방식은 분모가 채널별로 쪼개져 작아지기
    # 쉬워 push 가 폭발(발산)했다.  full 평균은 분모를 안정화한다.
    if normalize_direction:
        abs_gap = (committed - target_clean).abs()
        if mask is not None:
            # 죽은 채널의 gap 은 분자/분모 모두에서 제외(masked mean).
            abs_gap = abs_gap * mask
        if per_channel_normalizer:
            reduce_dims = tuple(range(1, committed.dim() - 1))  # 시간축만 (채널 유지)
        else:
            reduce_dims = tuple(range(1, committed.dim()))  # 시간+채널 (agent 스칼라)
        if mask is not None:
            valid_count = mask.expand_as(abs_gap).sum(dim=reduce_dims, keepdim=True)
            agent_distance = abs_gap.sum(dim=reduce_dims, keepdim=True) / valid_count.clamp_min(1.0)
        else:
            agent_distance = abs_gap.mean(dim=reduce_dims, keepdim=True)
        normalizer = agent_distance.clamp_min(float(normalizer_eps))
        normalized_direction = clean_dmd_direction / normalizer
    else:
        normalized_direction = clean_dmd_direction
    if mask is not None:
        # 나눗셈 후 부동소수 잔차까지 죽은 채널을 확실히 0으로 고정.
        normalized_direction = normalized_direction * mask
    normalized_direction = torch.nan_to_num(
        normalized_direction,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return normalized_direction.to(dtype=committed_path_norm.dtype)
