from __future__ import annotations

import torch
from torch import Tensor


POSE_FLOW_DIM = 4
CONTROL_FLOW_DIM = 3
DEFAULT_CONTROL_POS_SCALE_M = 20.0
DEFAULT_CONTROL_YAW_SCALE_RAD = 1.0


def wrap_angle(angle: Tensor) -> Tensor:
    """각도를 안정적인 범위로 접습니다.

    Args:
        angle: 접을 각도입니다. shape은 임의입니다.

    Returns:
        Tensor: 입력과 같은 shape의 각도입니다. 값은 ``[-pi, pi]`` 범위에 있습니다.
    """
    return torch.atan2(angle.sin(), angle.cos())


def safe_sinc(x: Tensor, eps: float = 1.0e-6) -> Tensor:
    """작은 각도에서도 안전하게 ``sin(x) / x`` 를 계산합니다.

    Args:
        x: sinc 값을 계산할 입력입니다. shape은 임의입니다.
        eps: 0에 가까운지 판단할 기준값입니다.

    Returns:
        Tensor: 입력과 같은 shape의 sinc 값입니다.
    """
    x2 = x * x
    approx = 1.0 - x2 / 6.0 + x2 * x2 / 120.0
    exact = x.sin() / x.clamp_min(eps)
    exact = torch.where(x < 0.0, x.sin() / x.clamp_max(-eps), exact)
    return torch.where(x.abs() < eps, approx, exact)


def normalize_control(
    control: Tensor,
    pos_scale_m: float = DEFAULT_CONTROL_POS_SCALE_M,
    yaw_scale_rad: float = DEFAULT_CONTROL_YAW_SCALE_RAD,
) -> Tensor:
    """제어값을 Flow Matching 학습 스케일로 바꿉니다.

    Args:
        control: 실제 단위 제어값입니다. shape은 ``[..., 3]`` 입니다.
            마지막 차원은 ``[앞뒤 이동량, 좌우 이동량, 방향 변화량]`` 입니다.
        pos_scale_m: 이동량을 나눌 meter 단위 값입니다.
        yaw_scale_rad: 방향 변화량을 나눌 radian 단위 값입니다.

    Returns:
        Tensor: 정규화된 제어값입니다. shape은 ``[..., 3]`` 입니다.
    """
    if control.shape[-1] != CONTROL_FLOW_DIM:
        raise ValueError(f"control last dim must be 3, got {control.shape[-1]}.")
    scale = control.new_tensor([float(pos_scale_m), float(pos_scale_m), float(yaw_scale_rad)])
    return control / scale


def denormalize_control(
    control_norm: Tensor,
    pos_scale_m: float = DEFAULT_CONTROL_POS_SCALE_M,
    yaw_scale_rad: float = DEFAULT_CONTROL_YAW_SCALE_RAD,
) -> Tensor:
    """정규화된 제어값을 실제 단위로 되돌립니다.

    Args:
        control_norm: 정규화된 제어값입니다. shape은 ``[..., 3]`` 입니다.
        pos_scale_m: 이동량 정규화에 쓴 meter 단위 값입니다.
        yaw_scale_rad: 방향 변화량 정규화에 쓴 radian 단위 값입니다.

    Returns:
        Tensor: 실제 단위 제어값입니다. shape은 ``[..., 3]`` 입니다.
    """
    if control_norm.shape[-1] != CONTROL_FLOW_DIM:
        raise ValueError(f"control_norm last dim must be 3, got {control_norm.shape[-1]}.")
    scale = control_norm.new_tensor([float(pos_scale_m), float(pos_scale_m), float(yaw_scale_rad)])
    return control_norm * scale


def decode_control_sequence(
    control: Tensor,
    agent_type: Tensor,
    current_pos: Tensor | None = None,
    current_head: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """제어 시퀀스를 pose 시퀀스로 바꿉니다.

    Args:
        control: 실제 단위 제어값입니다. shape은 ``[N, T, 3]`` 입니다.
            마지막 차원은 ``[앞뒤 이동량, 좌우 이동량, 방향 변화량]`` 입니다.
        agent_type: agent 종류입니다. shape은 ``[N]`` 입니다.
            repo의 기존 규칙대로 ``0=vehicle, 1=pedestrian, 2=cyclist`` 를 사용합니다.
        current_pos: 시작 위치입니다. shape은 ``[N, 2]`` 입니다.
            값이 없으면 원점에서 시작합니다.
        current_head: 시작 방향입니다. shape은 ``[N]`` 입니다.
            값이 없으면 0 rad에서 시작합니다.

    Returns:
        tuple[Tensor, Tensor]:
            복원된 위치와 방향입니다. 위치 shape은 ``[N, T, 2]`` 이고,
            방향 shape은 ``[N, T]`` 입니다.
    """
    if control.ndim != 3 or control.shape[-1] != CONTROL_FLOW_DIM:
        raise ValueError(f"control must have shape [N, T, 3], got {tuple(control.shape)}.")
    if agent_type.ndim != 1 or agent_type.shape[0] != control.shape[0]:
        raise ValueError(
            "agent_type must have shape [N] and match control batch, "
            f"got {tuple(agent_type.shape)} and {tuple(control.shape)}."
        )

    num_agent = control.shape[0]
    device = control.device
    dtype = control.dtype
    if current_pos is None:
        roll_pos = torch.zeros((num_agent, 2), device=device, dtype=dtype)
    else:
        roll_pos = current_pos.to(device=device, dtype=dtype)
    if current_head is None:
        roll_head = torch.zeros((num_agent,), device=device, dtype=dtype)
    else:
        roll_head = current_head.to(device=device, dtype=dtype)

    ped_mask = agent_type.to(device=device) == 1
    pos_steps: list[Tensor] = []
    head_steps: list[Tensor] = []

    for step_idx in range(control.shape[1]):
        step_control = control[:, step_idx]
        delta_s = step_control[:, 0]
        delta_n = step_control[:, 1]
        delta_head = step_control[:, 2]

        cos_head = roll_head.cos()
        sin_head = roll_head.sin()
        delta_pos_ped = torch.stack(
            [
                delta_s * cos_head - delta_n * sin_head,
                delta_s * sin_head + delta_n * cos_head,
            ],
            dim=-1,
        )

        mid_head = roll_head + 0.5 * delta_head
        arc_scale = delta_s * safe_sinc(0.5 * delta_head)
        delta_pos_nonhol = torch.stack(
            [arc_scale * mid_head.cos(), arc_scale * mid_head.sin()],
            dim=-1,
        )

        delta_pos = torch.where(ped_mask.unsqueeze(-1), delta_pos_ped, delta_pos_nonhol)
        roll_pos = roll_pos + delta_pos
        roll_head = wrap_angle(roll_head + delta_head)
        pos_steps.append(roll_pos)
        head_steps.append(roll_head)

    if len(pos_steps) == 0:
        return (
            control.new_zeros((num_agent, 0, 2)),
            control.new_zeros((num_agent, 0)),
        )
    return torch.stack(pos_steps, dim=1), torch.stack(head_steps, dim=1)


def control_norm_to_pose_norm(
    control_norm: Tensor,
    agent_type: Tensor,
    pos_scale_m: float = DEFAULT_CONTROL_POS_SCALE_M,
    yaw_scale_rad: float = DEFAULT_CONTROL_YAW_SCALE_RAD,
) -> Tensor:
    """정규화된 제어 시퀀스를 기존 pose-space 표현으로 바꿉니다.

    Args:
        control_norm: 정규화된 제어 시퀀스입니다. shape은 ``[N, T, 3]`` 입니다.
        agent_type: agent 종류입니다. shape은 ``[N]`` 입니다.
        pos_scale_m: 이동량 정규화에 쓴 meter 단위 값입니다.
        yaw_scale_rad: 방향 변화량 정규화에 쓴 radian 단위 값입니다.

    Returns:
        Tensor: 기존 Flow Matching 평가/추론 경로가 쓰는 pose 표현입니다.
            shape은 ``[N, T, 4]`` 이고, 마지막 차원은
            ``[x / 20, y / 20, cos(yaw), sin(yaw)]`` 입니다.
    """
    control = denormalize_control(
        control_norm=control_norm,
        pos_scale_m=pos_scale_m,
        yaw_scale_rad=yaw_scale_rad,
    )
    pos, head = decode_control_sequence(control=control, agent_type=agent_type)
    return torch.stack(
        [
            pos[..., 0] / float(pos_scale_m),
            pos[..., 1] / float(pos_scale_m),
            head.cos(),
            head.sin(),
        ],
        dim=-1,
    )


def build_rolling_control_target(
    future_pos: Tensor,
    future_head: Tensor,
    current_pos: Tensor,
    current_head: Tensor,
    agent_type: Tensor,
    pos_scale_m: float = DEFAULT_CONTROL_POS_SCALE_M,
    yaw_scale_rad: float = DEFAULT_CONTROL_YAW_SCALE_RAD,
) -> Tensor:
    """GT pose를 decoder-consistent rolling control label로 바꿉니다.

    Args:
        future_pos: GT 미래 위치입니다. shape은 ``[N, T, 2]`` 입니다.
        future_head: GT 미래 방향입니다. shape은 ``[N, T]`` 입니다.
        current_pos: anchor 현재 위치입니다. shape은 ``[N, 2]`` 입니다.
        current_head: anchor 현재 방향입니다. shape은 ``[N]`` 입니다.
        agent_type: agent 종류입니다. shape은 ``[N]`` 입니다.
        pos_scale_m: 이동량 정규화에 쓸 meter 단위 값입니다.
        yaw_scale_rad: 방향 변화량 정규화에 쓸 radian 단위 값입니다.

    Returns:
        Tensor: 정규화된 rolling control label입니다. shape은 ``[N, T, 3]`` 입니다.
            마지막 차원은 ``[앞뒤 이동량, 좌우 이동량, 방향 변화량]`` 입니다.
    """
    if future_pos.ndim != 3 or future_pos.shape[-1] != 2:
        raise ValueError(f"future_pos must have shape [N, T, 2], got {tuple(future_pos.shape)}.")
    if tuple(future_head.shape) != tuple(future_pos.shape[:2]):
        raise ValueError(
            "future_head must have shape [N, T], "
            f"got {tuple(future_head.shape)} for future_pos {tuple(future_pos.shape)}."
        )
    if tuple(current_pos.shape) != (future_pos.shape[0], 2):
        raise ValueError(f"current_pos must have shape [N, 2], got {tuple(current_pos.shape)}.")
    if tuple(current_head.shape) != (future_pos.shape[0],):
        raise ValueError(f"current_head must have shape [N], got {tuple(current_head.shape)}.")
    if tuple(agent_type.shape) != (future_pos.shape[0],):
        raise ValueError(f"agent_type must have shape [N], got {tuple(agent_type.shape)}.")

    roll_pos = current_pos.clone()
    roll_head = current_head.clone()
    ped_mask = agent_type.to(device=future_pos.device) == 1
    control_steps: list[Tensor] = []

    for step_idx in range(future_pos.shape[1]):
        target_pos = future_pos[:, step_idx]
        target_head = future_head[:, step_idx]
        delta_head = wrap_angle(target_head - roll_head)
        delta_vec = target_pos - roll_pos

        cos_head = roll_head.cos()
        sin_head = roll_head.sin()
        ped_delta_s = delta_vec[:, 0] * cos_head + delta_vec[:, 1] * sin_head
        ped_delta_n = -delta_vec[:, 0] * sin_head + delta_vec[:, 1] * cos_head

        mid_head = roll_head + 0.5 * delta_head
        h_mid = torch.stack([mid_head.cos(), mid_head.sin()], dim=-1)
        nonhol_delta_s = (delta_vec * h_mid).sum(dim=-1) / safe_sinc(0.5 * delta_head)
        nonhol_delta_n = torch.zeros_like(nonhol_delta_s)

        delta_s = torch.where(ped_mask, ped_delta_s, nonhol_delta_s)
        delta_n = torch.where(ped_mask, ped_delta_n, nonhol_delta_n)
        step_control = torch.stack([delta_s, delta_n, delta_head], dim=-1)
        control_steps.append(step_control)

        decoded_pos, decoded_head = decode_control_sequence(
            control=step_control.unsqueeze(1),
            agent_type=agent_type,
            current_pos=roll_pos,
            current_head=roll_head,
        )
        roll_pos = decoded_pos[:, -1]
        roll_head = decoded_head[:, -1]

    if len(control_steps) == 0:
        return future_pos.new_zeros((future_pos.shape[0], 0, CONTROL_FLOW_DIM))
    control = torch.stack(control_steps, dim=1)
    return normalize_control(
        control=control,
        pos_scale_m=pos_scale_m,
        yaw_scale_rad=yaw_scale_rad,
    )
