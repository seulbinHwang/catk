from __future__ import annotations

import torch
from torch import Tensor

from src.smart.modules.kinematic_control import CONTROL_FLOW_DIM, PEDESTRIAN_TYPE_ID


def build_active_control_mask(
    *,
    agent_type: Tensor | None,
    flow_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    use_kinematic_control_flow: bool,
    use_holonomic_model_only: bool,
) -> Tensor:
    """self-forced DMD가 실제 실행에 쓰는 control 축만 고릅니다.

    Args:
        agent_type: packed agent type입니다. shape은 ``[n_valid_agent]`` 입니다.
            control-space non-holonomic DMD일 때만 필요합니다.
        flow_dim: flow state 마지막 차원입니다.
        device: 반환 tensor device입니다.
        dtype: 반환 tensor dtype입니다.
        use_kinematic_control_flow: control-space flow 여부입니다.
        use_holonomic_model_only: 모든 agent를 holonomic control로 복원하는지 여부입니다.

    Returns:
        Tensor: ``[n_valid_agent, 1, flow_dim]`` active mask입니다.
    """
    num_agent = int(agent_type.shape[0]) if agent_type is not None else 0
    if (
        not use_kinematic_control_flow
        or use_holonomic_model_only
        or int(flow_dim) != CONTROL_FLOW_DIM
    ):
        return torch.ones((num_agent, 1, int(flow_dim)), device=device, dtype=dtype)
    if agent_type is None:
        raise ValueError("agent_type is required for non-holonomic control-space DMD.")
    if agent_type.ndim != 1:
        raise ValueError(f"agent_type must have shape [n_valid_agent], got {tuple(agent_type.shape)}.")

    active_mask = torch.ones((agent_type.shape[0], 1, CONTROL_FLOW_DIM), device=device, dtype=dtype)
    non_holonomic = agent_type.to(device=device) != PEDESTRIAN_TYPE_ID
    active_mask[non_holonomic, :, 1] = 0.0
    return active_mask


def active_control_dmd_surrogate_loss(
    committed_path_norm: Tensor,
    dmd_direction: Tensor,
    active_mask: Tensor | None = None,
    eps: float = 1.0e-6,
) -> tuple[Tensor, Tensor]:
    """DMD 방향을 detached target으로 주입하는 active-axis surrogate loss입니다.

    Args:
        committed_path_norm: Generator가 self-rollout으로 실행한 path/control ``X`` 입니다.
        dmd_direction: teacher와 generated estimator 차이로 만든 방향 ``D`` 입니다.
        active_mask: 실행에 영향을 주는 축 mask입니다. ``None`` 이면 모든 축을 씁니다.
        eps: 분모 안정화 값입니다.

    Returns:
        tuple[Tensor, Tensor]: ``(loss, detached_target)`` 입니다.
    """
    expected_shape = tuple(committed_path_norm.shape)
    if tuple(dmd_direction.shape) != expected_shape:
        raise ValueError(
            "dmd_direction shape must match committed_path_norm shape: "
            f"expected={expected_shape}, actual={tuple(dmd_direction.shape)}."
        )
    if committed_path_norm.ndim != 3:
        raise ValueError(
            "committed_path_norm must have shape [n_valid_agent, path_steps, flow_dim], "
            f"got {expected_shape}."
        )

    target_path_norm = (committed_path_norm + dmd_direction).detach()
    if committed_path_norm.shape[0] == 0:
        return committed_path_norm.sum() * 0.0, target_path_norm

    diff_square = (committed_path_norm.float() - target_path_norm.float()).square()
    if active_mask is None:
        loss = 0.5 * diff_square.flatten(1).mean(dim=1).mean()
        return loss, target_path_norm

    active = active_mask.to(device=committed_path_norm.device, dtype=diff_square.dtype)
    try:
        active = active.expand_as(diff_square)
    except RuntimeError as exc:
        raise ValueError(
            "active_mask must be broadcastable to committed_path_norm shape: "
            f"mask={tuple(active_mask.shape)}, path={expected_shape}."
        ) from exc

    denom = active.flatten(1).sum(dim=1).clamp_min(float(eps))
    per_agent_loss = (diff_square * active).flatten(1).sum(dim=1) / denom
    return 0.5 * per_agent_loss.mean(), target_path_norm


def build_clean_dmd_direction(
    committed_path_norm: Tensor,
    target_clean_norm: Tensor,
    generated_clean_norm: Tensor,
    active_mask: Tensor | None = None,
    normalizer_eps: float = 1.0e-3,
) -> Tensor:
    """teacher와 generated estimator의 clean path 차이로 DMD 방향을 만듭니다.

    Args:
        committed_path_norm: Generator가 closed-loop로 실제 실행한 path입니다.
            shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
        target_clean_norm: frozen teacher가 같은 noisy path에서 추정한 clean path입니다.
            shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
        generated_clean_norm: generated estimator가 같은 noisy path에서 추정한 clean path입니다.
            shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
        active_mask: 선택적 active 축 mask입니다. shape은 ``[n_valid_agent, 1, C]``
            처럼 ``committed_path_norm`` 에 broadcast 가능해야 합니다.
        normalizer_eps: agent별 정규화 분모의 최소값입니다.

    Returns:
        Tensor: 현재 committed path에 더할 정규화된 DMD 방향입니다.
        shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.

    설명:
        이 함수는 raw velocity 차이나 시간/노이즈 계수가 섞인 값을 그대로 쓰지 않습니다.
        먼저 ``target_clean_norm - generated_clean_norm`` 방향을 만들고, 각 agent의
        active 미래 path 기준으로 ``committed_path_norm``과 ``target_clean_norm`` 사이의
        평균 거리로 나눕니다. control-space non-holonomic DMD에서는 vehicle/cyclist의
        lateral 축을 active mask로 제거합니다.
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

    reduce_dims = tuple(range(1, committed.dim()))
    clean_dmd_direction = target_clean - generated_clean
    if active_mask is None:
        agent_distance = (committed - target_clean).abs().mean(
            dim=reduce_dims,
            keepdim=True,
        )
        active = None
    else:
        active = active_mask.to(device=committed.device, dtype=committed.dtype)
        try:
            active = active.expand_as(committed)
        except RuntimeError as exc:
            raise ValueError(
                "active_mask must be broadcastable to committed_path_norm shape: "
                f"mask={tuple(active_mask.shape)}, path={expected_shape}."
            ) from exc
        active_count = active.sum(dim=reduce_dims, keepdim=True).clamp_min(1.0)
        agent_distance = ((committed - target_clean).abs() * active).sum(
            dim=reduce_dims,
            keepdim=True,
        ) / active_count
        clean_dmd_direction = clean_dmd_direction * active
    normalizer = agent_distance.clamp_min(float(normalizer_eps))

    normalized_direction = clean_dmd_direction / normalizer
    if active is not None:
        normalized_direction = normalized_direction * active
    normalized_direction = torch.nan_to_num(
        normalized_direction,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return normalized_direction.to(dtype=committed_path_norm.dtype)
