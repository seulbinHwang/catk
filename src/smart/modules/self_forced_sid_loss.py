from __future__ import annotations

import torch
from torch import Tensor


def _require_same_shape(reference: Tensor, candidate: Tensor, candidate_name: str) -> None:
    """두 텐서의 모양이 같은지 확인합니다.

    Args:
        reference: 기준 텐서입니다.
            shape은 보통 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
        candidate: 기준과 비교할 텐서입니다.
            shape은 ``reference`` 와 같아야 합니다.
        candidate_name: 오류 메시지에 넣을 텐서 이름입니다.

    Returns:
        None
    """
    expected_shape = tuple(reference.shape)
    actual_shape = tuple(candidate.shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"{candidate_name} shape must match committed_path_norm shape: "
            f"expected={expected_shape}, actual={actual_shape}."
        )


def _build_agentwise_normalizer(
    committed_path_norm: Tensor,
    target_clean_norm: Tensor,
    normalizer_eps: float,
) -> Tensor:
    """agent별 path 거리 기준의 안정화 분모를 만듭니다.

    Args:
        committed_path_norm: Generator가 closed-loop로 실제 실행한 path입니다.
            shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
        target_clean_norm: frozen teacher가 같은 noisy path에서 추정한 clean path입니다.
            shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
        normalizer_eps: 분모가 너무 작아지는 것을 막는 최소값입니다.

    Returns:
        Tensor: agent별 분모입니다.
            shape은 ``[n_valid_agent, 1, 1]`` 입니다.
    """
    if committed_path_norm.dim() < 2:
        raise ValueError(
            "committed_path_norm must have at least agent and path dimensions, "
            f"got shape={tuple(committed_path_norm.shape)}."
        )

    # committed_path_norm / target_clean_norm: [n_valid_agent, flow_window_steps, 4]
    reduce_dims = tuple(range(1, committed_path_norm.dim()))
    agent_distance = (committed_path_norm - target_clean_norm).abs().mean(
        dim=reduce_dims,
        keepdim=True,
    )
    # agent_distance: [n_valid_agent, 1, 1]
    return agent_distance.clamp_min(float(normalizer_eps))


def compute_clean_sid_loss(
    committed_path_norm: Tensor,
    target_clean_norm: Tensor,
    generated_clean_norm: Tensor,
    *,
    sid_alpha: float = 1.0,
    normalizer_eps: float = 1.0e-3,
) -> Tensor:
    """clean path 공간에서 SiD-lite generator loss를 계산합니다.

    Args:
        committed_path_norm: Generator가 closed-loop로 실제 실행한 path ``X`` 입니다.
            shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
        target_clean_norm: frozen teacher가 같은 noisy path에서 추정한 clean path ``R`` 입니다.
            shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
        generated_clean_norm: generated estimator가 같은 noisy path에서 추정한 clean path ``F`` 입니다.
            shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
        sid_alpha: ``R - F`` 보정항의 강도입니다. Self-Forcing SiD 기본값은 ``1.0`` 입니다.
        normalizer_eps: agent별 정규화 분모의 최소값입니다.

    Returns:
        Tensor: scalar SiD-lite loss입니다. shape은 ``[]`` 입니다.

    설명:
        DMD는 ``X`` 를 움직일 detached target을 만든 뒤 MSE를 겁니다.
        SiD-lite는 target을 만들지 않고 ``X``, ``R``, ``F`` 의 관계식을 바로 줄입니다.
        여기서는 공식 SiD 식 ``(R - F) * ((R - X) - alpha * (R - F))`` 을
        같은 뜻의 ``(R - F) * (F - X) + (1 - alpha) * (R - F)^2`` 로 계산합니다.
        ``R`` 과 ``F`` 는 teacher/estimator 신호이므로 gradient를 막고,
        gradient는 ``committed_path_norm`` 으로만 흐르게 둡니다.
    """
    _require_same_shape(committed_path_norm, target_clean_norm, "target_clean_norm")
    _require_same_shape(committed_path_norm, generated_clean_norm, "generated_clean_norm")

    if committed_path_norm.numel() == 0:
        return committed_path_norm.sum() * 0.0

    # X, R, F: [n_valid_agent, flow_window_steps, 4]
    current_path = committed_path_norm.float()
    target_clean = target_clean_norm.detach().float()
    generated_clean = generated_clean_norm.detach().float()

    # score_gap: [n_valid_agent, flow_window_steps, 4]
    score_gap = target_clean - generated_clean
    # estimator_to_current_gap: [n_valid_agent, flow_window_steps, 4]
    estimator_to_current_gap = generated_clean - current_path

    sid_element = (
        score_gap * estimator_to_current_gap
        + (1.0 - float(sid_alpha)) * score_gap.square()
    )
    normalizer = _build_agentwise_normalizer(
        committed_path_norm=current_path.detach(),
        target_clean_norm=target_clean,
        normalizer_eps=normalizer_eps,
    )
    sid_loss = sid_element / normalizer
    sid_loss = torch.nan_to_num(
        sid_loss,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return sid_loss.mean()
