from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch
from torch import Tensor

from src.smart.utils.geometry import wrap_angle
from src.smart.utils.rollout import transform_to_global, transform_to_local


def renorm_sin_cos(x: Tensor, start_dim: int = -1) -> Tensor:
    """sin, cos 쌍을 다시 길이 1로 맞춘다.

    Args:
        x: 마지막 차원에 ``[..., sin, cos]`` 를 포함하는 텐서.
        start_dim: sin 값이 시작되는 차원 인덱스.
            기본값 ``-1`` 은 마지막 두 값을 정규화할 때 쓴다.

    Returns:
        입력 shape을 그대로 유지한 정규화 결과.
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


def _validate_segment_config(
    num_future_points: int,
    future_num_segments: int,
    future_segment_points: int,
) -> int:
    """future segment 설정이 서로 맞는지 검사한다.

    Args:
        num_future_points: 전체 future 점 개수. 시작점 1개를 포함한다.
        future_num_segments: 미래를 몇 개 토큰으로 나눌지 나타내는 값.
        future_segment_points: 각 미래 토큰이 들고 있는 점 개수.
            현재 구현은 인접 토큰이 마지막 점 하나를 공유한다고 본다.

    Returns:
        인접 segment 시작점 사이의 간격인 ``segment_stride_steps``.

    Raises:
        ValueError: 설정끼리 모순이 있을 때 발생한다.
    """
    if future_num_segments <= 0:
        raise ValueError(f"future_num_segments must be positive, got {future_num_segments}.")
    if future_segment_points < 2:
        raise ValueError(
            "future_segment_points must be at least 2 because each segment needs a start and an end point. "
            f"Got {future_segment_points}."
        )

    segment_stride_steps = future_segment_points - 1
    expected_num_future_points = segment_stride_steps * future_num_segments + 1
    if num_future_points != expected_num_future_points:
        raise ValueError(
            "Inconsistent future segment setup. "
            f"num_future_points={num_future_points}, "
            f"future_num_segments={future_num_segments}, "
            f"future_segment_points={future_segment_points}, "
            f"expected_num_future_points={expected_num_future_points}."
        )
    return segment_stride_steps


def build_segment_slices(
    num_future_points: int,
    future_num_segments: int,
    future_segment_points: int,
) -> List[Tuple[int, int]]:
    """전체 future trajectory를 segment 구간 목록으로 바꾼다.

    Args:
        num_future_points: 전체 future 점 개수. 시작점 1개를 포함한다.
        future_num_segments: 미래를 몇 개 token으로 나눌지 나타내는 값.
        future_segment_points: 각 segment가 들고 있는 점 개수.

    Returns:
        ``[(start, end), ...]`` 형식의 구간 목록.
        각 구간은 Python slice 규칙과 같은 ``[start:end]`` 를 뜻한다.
    """
    segment_stride_steps = _validate_segment_config(
        num_future_points=num_future_points,
        future_num_segments=future_num_segments,
        future_segment_points=future_segment_points,
    )
    return [
        (
            segment_idx * segment_stride_steps,
            segment_idx * segment_stride_steps + future_segment_points,
        )
        for segment_idx in range(future_num_segments)
    ]


def chunk_future_to_segments(
    future_local: Tensor,
    future_num_segments: int,
    future_segment_points: int,
) -> Tensor:
    """전체 future trajectory를 겹치는 segment 묶음으로 바꾼다.

    Args:
        future_local: ``[N, T, 4]``. 마지막 차원은
            ``(x_local, y_local, sin(dyaw), cos(dyaw))`` 이다.
        future_num_segments: 미래를 몇 개 token으로 나눌지 나타내는 값.
        future_segment_points: 각 token이 들고 있는 점 개수.

    Returns:
        ``[N, S, P, 4]`` segment 텐서.
        여기서 ``S=future_num_segments``, ``P=future_segment_points`` 이다.
    """
    if future_local.dim() != 3:
        raise ValueError(f"Expected future_local with shape [N, T, 4], got {tuple(future_local.shape)}.")

    spans = build_segment_slices(
        num_future_points=int(future_local.shape[1]),
        future_num_segments=future_num_segments,
        future_segment_points=future_segment_points,
    )
    return torch.stack([future_local[:, start:end] for start, end in spans], dim=1)


def chunk_future_21_to_4x6(future_21: Tensor) -> Tensor:
    """기존 4x6 설정을 위한 하위 호환 wrapper다.

    Args:
        future_21: ``[N, T, 4]`` 전체 future trajectory.

    Returns:
        ``[N, 4, 6, 4]`` segment 텐서.
    """
    return chunk_future_to_segments(
        future_local=future_21,
        future_num_segments=4,
        future_segment_points=6,
    )


def assemble_segments_to_future(segments: Tensor) -> Tensor:
    """겹치는 future segment를 다시 전체 future trajectory로 합친다.

    Args:
        segments: ``[N, S, P, 4]``. 미래 token 묶음.
            인접 token은 마지막 점 하나를 공유한다고 본다.

    Returns:
        ``[N, T, 4]`` 전체 future trajectory.
        여기서 ``T = S * (P - 1) + 1`` 이다.
    """
    if segments.dim() != 4:
        raise ValueError(f"Expected segments with shape [N, S, P, 4], got {tuple(segments.shape)}.")

    segments = renorm_sin_cos(segments)
    future_num_segments = int(segments.shape[1])
    future_segment_points = int(segments.shape[2])
    num_future_points = (future_segment_points - 1) * future_num_segments + 1
    spans = build_segment_slices(
        num_future_points=num_future_points,
        future_num_segments=future_num_segments,
        future_segment_points=future_segment_points,
    )

    future = torch.zeros(
        segments.shape[0],
        num_future_points,
        segments.shape[-1],
        device=segments.device,
        dtype=segments.dtype,
    )
    count = torch.zeros(
        segments.shape[0],
        num_future_points,
        1,
        device=segments.device,
        dtype=segments.dtype,
    )
    for seg_idx, (start, end) in enumerate(spans):
        future[:, start:end] += segments[:, seg_idx]
        count[:, start:end] += 1.0

    future = future / torch.clamp(count, min=1.0)
    future = renorm_sin_cos(future)
    return future


def assemble_4x6_to_21(segments: Tensor) -> Tensor:
    """기존 4x6 설정을 위한 하위 호환 wrapper다.

    Args:
        segments: ``[N, S, P, 4]`` segment 텐서.

    Returns:
        ``[N, T, 4]`` 전체 future trajectory.
    """
    return assemble_segments_to_future(segments)


def build_local_future_target(
    pos_global: Tensor,
    head_global: Tensor,
    anchor_step: int,
    future_window_steps: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """GT future를 현재 agent-local 좌표계로 바꾼다.

    Args:
        pos_global: ``[N, T_all, 2]`` global 중심점.
        head_global: ``[N, T_all]`` global heading.
        anchor_step: 현재 기준 raw step.
        future_window_steps: 앞으로 몇 개 raw step을 볼지 나타내는 값.

    Returns:
        tuple:
            - future_local: ``[N, future_window_steps + 1, 4]``
            - pos_now: ``[N, 2]``
            - head_now: ``[N]``
            - head_delta: ``[N, future_window_steps + 1]``
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
        future_local: ``[N, T, 4]`` local future trajectory.
        pos_now: ``[N, 2]`` 현재 위치.
        head_now: ``[N]`` 현재 heading.

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
    """각 future segment의 마지막 점을 world pose로 바꾼다.

    Args:
        segments_local: ``[N, S, P, 4]`` future segment 텐서.
        pos_now: ``[N, 2]`` 현재 위치.
        head_now: ``[N]`` 현재 heading.

    Returns:
        tuple:
            - seg_end_pos_global: ``[N, S, 2]``
            - seg_end_head_global: ``[N, S]``
    """
    end_local = renorm_sin_cos(segments_local[:, :, -1])
    end_global_pos, end_global_head = local_future_to_global(end_local, pos_now, head_now)
    return end_global_pos, end_global_head


def build_flow_path(x0: Tensor, x1: Tensor, tau: Tensor) -> Tuple[Tensor, Tensor]:
    """선형 conditional flow matching path를 만든다.

    이 함수는 flow path와 target field가 같은 선형 상태공간에서 정의되도록
    상태 자체에는 sin/cos 재정규화를 적용하지 않는다.

    Args:
        x0: ``[N, S, P, 4]`` source noise.
        x1: ``[N, S, P, 4]`` clean target.
        tau: ``[N, 1, 1, 1]`` 또는 broadcast 가능한 shape.

    Returns:
        tuple:
            - x_tau: ``[N, S, P, 4]``
            - u_t: ``[N, S, P, 4]`` target velocity field
    """
    x_tau = (1.0 - tau) * x0 + tau * x1
    u_t = x1 - x0
    return x_tau, u_t


def midpoint_ode_integrate(
    x0: Tensor,
    ode_steps: int,
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """midpoint ODE 적분을 수행한다.

    학습 때와 같은 선형 상태공간에서 적분하므로,
    적분 중간 상태에는 sin/cos 재정규화를 넣지 않는다.

    Args:
        x0: ``[N, S, P, 4]`` 초기 noise.
        ode_steps: 적분 step 수.
        velocity_fn: ``f(x, t) -> dx/dt`` 형태의 함수.

    Returns:
        ``[N, S, P, 4]`` 최종 적분 결과.
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
        pos: ``[N, 2]`` 중심점.
        head: ``[N]`` heading.
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

    left_front = torch.stack(
        [pos[:, 0] + length_cos - width_sin, pos[:, 1] + length_sin + width_cos],
        dim=-1,
    )
    right_front = torch.stack(
        [pos[:, 0] + length_cos + width_sin, pos[:, 1] + length_sin - width_cos],
        dim=-1,
    )
    right_back = torch.stack(
        [pos[:, 0] - length_cos + width_sin, pos[:, 1] - length_sin - width_cos],
        dim=-1,
    )
    left_back = torch.stack(
        [pos[:, 0] - length_cos - width_sin, pos[:, 1] - length_sin + width_cos],
        dim=-1,
    )
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
    """전체 future 예측에서 실제로 실행할 첫 0.5초를 뽑아 다음 상태를 만든다.

    Args:
        future_local_21: ``[N, T, 4]`` 전체 local future trajectory.
            현재 구현은 첫 0.5초를 실행하기 위해 최소 ``T >= 6`` 을 요구한다.
        pos_now: ``[N, 2]`` 현재 위치.
        head_now: ``[N]`` 현재 heading.

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
    if future_local_21.shape[1] < 6:
        raise ValueError(
            "future_local_21 must contain at least 6 points to execute the first 0.5-second chunk. "
            f"Got shape {tuple(future_local_21.shape)}."
        )

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
