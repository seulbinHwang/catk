from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
from torch import Tensor

from src.smart.utils.geometry import wrap_angle
from src.smart.utils.rollout import transform_to_global, transform_to_local


def renorm_sin_cos(x: Tensor, start_dim: int = -1) -> Tensor:
    """sin, cos 쌍을 다시 길이 1로 맞춘다.

    Args:
        x: 마지막 차원에 ``[..., sin, cos]`` 를 포함하는 텐서.
        start_dim: sin 값이 시작되는 차원 인덱스. 기본값은 마지막 차원 기준
            ``x[..., 2:4]`` 를 정규화하는 용도에 맞춘 ``-1`` 이다.

    Returns:
        정규화된 텐서. 입력 shape은 그대로 유지된다.
    """
    if start_dim == -1:
        sin_cos = x[..., -2:]
        denom = torch.clamp(torch.norm(sin_cos, dim=-1, keepdim=True), min=1e-6)
        x = x.clone()
        x[..., -2:] = sin_cos / denom
        return x

    sin_cos = x[..., start_dim : start_dim + 2]
    denom = torch.clamp(torch.norm(sin_cos, dim=-1, keepdim=True), min=1e-6)
    x = x.clone()
    x[..., start_dim : start_dim + 2] = sin_cos / denom
    return x


def chunk_future_21_to_4x6(future_21: Tensor) -> Tensor:
    """21개 future 점을 4개의 겹치는 0.5초 segment로 바꾼다.

    Args:
        future_21: ``[N, 21, 4]``. 마지막 차원은
            ``(x_local, y_local, sin(dyaw), cos(dyaw))`` 이다.

    Returns:
        ``[N, 4, 6, 4]`` segment 텐서.
    """
    return torch.stack(
        [
            future_21[:, 0:6],
            future_21[:, 5:11],
            future_21[:, 10:16],
            future_21[:, 15:21],
        ],
        dim=1,
    )


def assemble_4x6_to_21(segments: Tensor) -> Tensor:
    """4개의 겹치는 segment를 다시 21개 future 점으로 합친다.

    Args:
        segments: ``[N, 4, 6, 4]``.

    Returns:
        ``[N, 21, 4]``.
    """
    segments = renorm_sin_cos(segments)
    future = torch.zeros(
        segments.shape[0], 21, segments.shape[-1], device=segments.device, dtype=segments.dtype
    )
    count = torch.zeros(
        segments.shape[0], 21, 1, device=segments.device, dtype=segments.dtype
    )
    spans = [(0, 6), (5, 11), (10, 16), (15, 21)]
    for seg_idx, (start, end) in enumerate(spans):
        future[:, start:end] += segments[:, seg_idx]
        count[:, start:end] += 1.0
    future = future / torch.clamp(count, min=1.0)
    future = renorm_sin_cos(future)
    return future


def build_local_future_target(
    pos_global: Tensor,
    head_global: Tensor,
    anchor_step: int,
    future_window_steps: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """GT 2초 future를 현재 agent-local 좌표계로 바꾼다.

    Args:
        pos_global: ``[N, 91, 2]`` global 중심점.
        head_global: ``[N, 91]`` global heading.
        anchor_step: 현재 기준 raw step. SMART 기본값에서는 10, 15, ... 이다.
        future_window_steps: future 길이. 여기서는 20을 권장한다.

    Returns:
        tuple:
            - future_local: ``[N, 21, 4]``
            - pos_now: ``[N, 2]``
            - head_now: ``[N]``
            - head_delta: ``[N, 21]``
    """
    pos_now = pos_global[:, anchor_step]
    head_now = head_global[:, anchor_step]
    future_pos = pos_global[:, anchor_step : anchor_step + future_window_steps + 1]
    future_head = head_global[:, anchor_step : anchor_step + future_window_steps + 1]

    future_pos_local, future_head_local = transform_to_local(
        pos_global=future_pos,
        head_global=future_head,
        pos_now=pos_now,
        head_now=head_now,
    )
    future_local = torch.cat(
        [
            future_pos_local,
            future_head_local.sin().unsqueeze(-1),
            future_head_local.cos().unsqueeze(-1),
        ],
        dim=-1,
    )
    future_local = renorm_sin_cos(future_local)
    return future_local, pos_now, head_now, future_head_local


def local_future_to_global(
    future_local: Tensor,
    pos_now: Tensor,
    head_now: Tensor,
) -> Tuple[Tensor, Tensor]:
    """agent-local future를 world 좌표로 바꾼다.

    Args:
        future_local: ``[N, T, 4]``.
        pos_now: ``[N, 2]``.
        head_now: ``[N]``.

    Returns:
        tuple:
            - global_pos: ``[N, T, 2]``
            - global_head: ``[N, T]``
    """
    future_local = renorm_sin_cos(future_local)
    local_head = torch.atan2(future_local[..., 2], future_local[..., 3])
    global_pos, global_head = transform_to_global(
        pos_local=future_local[..., :2],
        head_local=local_head,
        pos_now=pos_now,
        head_now=head_now,
    )
    global_head = wrap_angle(global_head)
    return global_pos, global_head


def segment_end_pose_global(
    segments_local: Tensor,
    pos_now: Tensor,
    head_now: Tensor,
) -> Tuple[Tensor, Tensor]:
    """각 0.5초 segment의 마지막 점을 world pose로 바꾼다.

    Args:
        segments_local: ``[N, 4, 6, 4]``.
        pos_now: ``[N, 2]``.
        head_now: ``[N]``.

    Returns:
        tuple:
            - seg_end_pos_global: ``[N, 4, 2]``
            - seg_end_head_global: ``[N, 4]``
    """
    end_local = renorm_sin_cos(segments_local[:, :, -1])
    end_global_pos, end_global_head = local_future_to_global(end_local, pos_now, head_now)
    return end_global_pos, end_global_head


def build_flow_path(x0: Tensor, x1: Tensor, tau: Tensor) -> Tuple[Tensor, Tensor]:
    """선형 conditional flow matching path를 만든다.

    이 함수는 flow path와 target field가 같은 선형 상태공간에서 정의되도록
    상태 자체에는 sin/cos 재정규화를 적용하지 않는다. sin/cos 재정규화는
    world 좌표 변환이나 최종 trajectory 조립처럼 기하 해석이 필요한 곳에서만
    별도로 수행한다.

    Args:
        x0: ``[N, 4, 6, 4]`` source noise.
        x1: ``[N, 4, 6, 4]`` clean target.
        tau: ``[N, 1, 1, 1]`` 또는 broadcast 가능한 shape.

    Returns:
        tuple:
            - x_tau: ``[N, 4, 6, 4]``
            - u_t: ``[N, 4, 6, 4]`` target velocity field
    """
    x_tau = (1.0 - tau) * x0 + tau * x1
    u_t = x1 - x0
    return x_tau, u_t


def midpoint_ode_integrate(
    x0: Tensor,
    ode_steps: int,
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """4-step midpoint ODE 적분을 수행한다.

    이 적분도 학습 때와 같은 선형 상태공간에서 진행한다. 따라서 적분 중간
    상태에는 sin/cos 재정규화를 넣지 않고, world 좌표 변환이나 최종 결과
    조립 단계에서만 재정규화를 적용한다.

    Args:
        x0: ``[N, 4, 6, 4]`` 초기 noise.
        ode_steps: 적분 step 수. 이번 구현에서는 4를 기본값으로 둔다.
        velocity_fn: ``f(x, t) -> dx/dt`` 형태의 함수.

    Returns:
        ``[N, 4, 6, 4]`` 최종 적분 결과.
    """
    x = x0.clone()
    dt = 1.0 / float(ode_steps)
    for step in range(ode_steps):
        t = x.new_full((x.shape[0], 1), float(step) * dt)
        k1 = velocity_fn(x, t)
        x_mid = x + 0.5 * dt * k1
        t_mid = x.new_full((x.shape[0], 1), (float(step) + 0.5) * dt)
        k2 = velocity_fn(x_mid, t_mid)
        x = x + dt * k2
    return x


def _center_to_contour(pos: Tensor, head: Tensor, agent_shape: Tensor) -> Tensor:
    """중심점과 heading을 4개 꼭짓점 contour로 바꾼다.

    Args:
        pos: ``[N, 2]``.
        head: ``[N]``.
        agent_shape: ``[N, 2]``. ``(width, length)``.

    Returns:
        ``[N, 4, 2]`` contour.
    """
    width = agent_shape[:, 0]
    length = agent_shape[:, 1]
    half_cos = 0.5 * head.cos()
    half_sin = 0.5 * head.sin()
    length_cos = length * half_cos
    length_sin = length * half_sin
    width_cos = width * half_cos
    width_sin = width * half_sin
    left_front = torch.stack([pos[:, 0] + length_cos - width_sin, pos[:, 1] + length_sin + width_cos], dim=-1)
    right_front = torch.stack([pos[:, 0] + length_cos + width_sin, pos[:, 1] + length_sin - width_cos], dim=-1)
    right_back = torch.stack([pos[:, 0] - length_cos + width_sin, pos[:, 1] - length_sin - width_cos], dim=-1)
    left_back = torch.stack([pos[:, 0] - length_cos - width_sin, pos[:, 1] - length_sin + width_cos], dim=-1)
    return torch.stack([left_front, right_front, right_back, left_back], dim=-2)


def local_traj_to_local_contour(
    future_local: Tensor,
    agent_shape: Tensor,
) -> Tensor:
    """center trajectory + heading을 contour trajectory로 바꾼다.

    Args:
        future_local: ``[N, 6, 4]``. 첫 점 포함 0.5초 trajectory.
        agent_shape: ``[N, 2]``. ``(width, length)``.

    Returns:
        ``[N, 6, 4, 2]`` local contour trajectory.
    """
    future_local = renorm_sin_cos(future_local)
    local_head = torch.atan2(future_local[..., 2], future_local[..., 3])
    contour_list = []
    for t in range(future_local.shape[1]):
        contour_list.append(_center_to_contour(future_local[:, t, :2], local_head[:, t], agent_shape))
    return torch.stack(contour_list, dim=1)


def nearest_agent_token_idx(
    local_chunk_6: Tensor,
    agent_shape: Tensor,
    token_traj_all: Tensor,
) -> Tensor:
    """첫 0.5초 continuous chunk를 가장 가까운 SMART token id로 바꾼다.

    Args:
        local_chunk_6: ``[N, 6, 4]``.
        agent_shape: ``[N, 2]``.
        token_traj_all: ``[N, V, 6, 4, 2]``.

    Returns:
        ``[N]`` nearest token index.
    """
    contour_local = local_traj_to_local_contour(local_chunk_6, agent_shape).unsqueeze(1)
    dist = torch.norm(token_traj_all - contour_local, dim=-1).mean(dim=(-1, -2))
    return dist.argmin(dim=-1)


def executed_chunk_to_rollout_update(
    future_local_21: Tensor,
    pos_now: Tensor,
    head_now: Tensor,
) -> Dict[str, Tensor]:
    """2초 예측에서 실제로 실행할 첫 0.5초와 다음 현재 상태를 뽑는다.

    Args:
        future_local_21: ``[N, 21, 4]``.
        pos_now: ``[N, 2]``.
        head_now: ``[N]``.

    Returns:
        dict:
            - exec_local_6: ``[N, 6, 4]``
            - exec_global_pos_6: ``[N, 6, 2]``
            - exec_global_head_6: ``[N, 6]``
            - next_pos: ``[N, 2]``
            - next_head: ``[N]``
            - next_vel: ``[N, 2]``
            - next_yaw_rate: ``[N]``
    """
    exec_local_6 = renorm_sin_cos(future_local_21[:, :6])
    exec_global_pos_6, exec_global_head_6 = local_future_to_global(exec_local_6, pos_now, head_now)
    next_pos = exec_global_pos_6[:, -1]
    next_head = exec_global_head_6[:, -1]
    next_vel = (exec_global_pos_6[:, -1] - exec_global_pos_6[:, -2]) / 0.1
    next_yaw_rate = wrap_angle(exec_global_head_6[:, -1] - exec_global_head_6[:, -2]) / 0.1
    return {
        "exec_local_6": exec_local_6,
        "exec_global_pos_6": exec_global_pos_6,
        "exec_global_head_6": exec_global_head_6,
        "next_pos": next_pos,
        "next_head": next_head,
        "next_vel": next_vel,
        "next_yaw_rate": next_yaw_rate,
    }
