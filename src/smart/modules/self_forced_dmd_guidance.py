from __future__ import annotations

import torch
from torch import Tensor


def resolve_self_forced_entropy_beta(config: object | None) -> float:
    """Resolve the ERD entropy temperature beta from self-forced config."""
    if config is None:
        return 1.0
    raw_beta = getattr(config, "entropy_beta", getattr(config, "dmd_beta", 1.0))
    beta = float(raw_beta)
    if not 0.0 < beta <= 1.0:
        raise ValueError(
            "self_forced.entropy_beta must satisfy 0 < beta <= 1, "
            f"got {beta}."
        )
    return beta


def _expand_tau_for_clean_prior(tau: Tensor, reference: Tensor) -> Tensor:
    tau_tensor = tau.to(device=reference.device, dtype=reference.dtype)
    batch_size = int(reference.shape[0])
    if tau_tensor.ndim == 0:
        tau_tensor = tau_tensor.expand(batch_size)
    elif tuple(tau_tensor.shape) != (batch_size,):
        raise ValueError(
            "tau must have shape [] or [n_valid_agent] when entropy_beta < 1, "
            f"got {tuple(tau_tensor.shape)} for n_valid_agent={batch_size}."
        )
    return tau_tensor.view(batch_size, *([1] * (reference.dim() - 1)))


def build_clean_dmd_direction(
    committed_path_norm: Tensor,
    target_clean_norm: Tensor,
    generated_clean_norm: Tensor,
    noisy_path_norm: Tensor | None = None,
    tau: Tensor | None = None,
    normalizer_eps: float = 1.0e-3,
    channel_mask: Tensor | None = None,
    per_channel_normalizer: bool = False,
    normalize_direction: bool = True,
    entropy_beta: float = 1.0,
) -> Tensor:
    """teacher와 generated estimator의 clean path 차이로 DMD 방향을 만듭니다.

    Args:
        committed_path_norm: Generator가 closed-loop로 실제 실행한 path입니다.
            shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
        target_clean_norm: frozen teacher가 같은 noisy path에서 추정한 clean path입니다.
            shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
        generated_clean_norm: generated estimator가 같은 noisy path에서 추정한 clean path입니다.
            shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
        noisy_path_norm: 같은 flow time에서 noised 된 path입니다. ``entropy_beta < 1``
            에서만 필요합니다.
        tau: ``noisy_path_norm`` 의 flow interpolation time입니다. shape은
            ``[n_valid_agent]`` 입니다. 이 코드의 flow path에서 clean coefficient
            ``alpha_tau`` 는 ``tau`` 입니다.
        normalizer_eps: agent별 정규화 분모의 최소값입니다.

    Returns:
        Tensor: 현재 committed path에 더할 정규화된 DMD 방향입니다.
        shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.

    설명:
        이 함수는 raw velocity 차이나 시간/노이즈 계수가 섞인 값을 그대로 쓰지 않습니다.
        ERD beta가 1이면 기존 DMD처럼 ``target_clean_norm - generated_clean_norm``
        방향을 씁니다. beta가 1보다 작으면 논문 식의 음수 방향인
        ``beta * target_clean + (1 - beta) * noisy / tau - generated_clean`` 을
        사용합니다. 이후 각 agent의 전체 미래 path 기준으로 ``committed_path_norm``과
        ``target_clean_norm`` 사이의 평균 거리로 나눕니다.
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
    beta = float(entropy_beta)
    if not 0.0 < beta <= 1.0:
        raise ValueError(f"entropy_beta must satisfy 0 < beta <= 1, got {beta}.")
    if beta < 1.0:
        if noisy_path_norm is None or tau is None:
            raise ValueError(
                "noisy_path_norm and tau are required when entropy_beta < 1."
            )
        if tuple(noisy_path_norm.shape) != expected_shape:
            raise ValueError(
                "noisy_path_norm shape must match committed_path_norm shape: "
                f"expected={expected_shape}, actual={tuple(noisy_path_norm.shape)}."
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

    if beta < 1.0:
        assert noisy_path_norm is not None and tau is not None
        noisy = noisy_path_norm.to(device=committed.device, dtype=committed.dtype)
        tau_view = _expand_tau_for_clean_prior(tau, committed)
        clean_prior = noisy / tau_view.clamp_min(torch.finfo(committed.dtype).eps)
        tempered_target_clean = beta * target_clean + (1.0 - beta) * clean_prior
    else:
        tempered_target_clean = target_clean

    clean_dmd_direction = tempered_target_clean - generated_clean
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
