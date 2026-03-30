from __future__ import annotations

import torch
from torch import Tensor


def rotate_points_to_local(
    points_global: Tensor,
    origin_pos: Tensor,
    origin_heading: Tensor,
) -> Tensor:
    """전역 좌표 점들을 기준 시점의 local 좌표로 바꿉니다.

    Args:
        points_global: 전역 좌표 점들입니다. shape은 ``[..., n_point, 2]`` 입니다.
        origin_pos: local 좌표의 원점이 되는 위치입니다. shape은 ``[..., 2]`` 입니다.
        origin_heading: local 좌표의 x축이 바라보는 각도입니다.
            shape은 ``[...]`` 입니다.

    Returns:
        Tensor: local 좌표로 바뀐 점들입니다.
        shape은 ``[..., n_point, 2]`` 입니다.
    """
    delta = points_global - origin_pos.unsqueeze(-2)
    cos_h = origin_heading.cos().unsqueeze(-1)
    sin_h = origin_heading.sin().unsqueeze(-1)
    local_x = delta[..., 0] * cos_h + delta[..., 1] * sin_h
    local_y = -delta[..., 0] * sin_h + delta[..., 1] * cos_h
    return torch.stack([local_x, local_y], dim=-1)


def compute_segment_heading_from_points(
    start_pos: Tensor,
    segment_pos: Tensor,
    prev_heading: Tensor,
    min_total_disp: float = 0.20,
    min_tail_disp: float = 0.05,
) -> Tensor:
    """0.5초 구간의 끝 방향을 안정적으로 계산합니다.

    WOSAC closed-loop에서는 순간 0.1초 꼬리 방향만 쓰면 작은 흔들림이 다음 문맥으로
    바로 누적되기 쉽습니다. 그래서 기본은 0.5초 전체 변위를 보고 방향을 정하고,
    전체 변위가 너무 작을 때만 마지막 0.1초 꼬리를 보고, 그것도 작으면 이전 방향을
    그대로 유지합니다.

    Args:
        start_pos: 구간 시작 위치입니다. shape은 ``[n_agent, 2]`` 입니다.
        segment_pos: 구간 안 5개 중심점입니다. shape은 ``[n_agent, 5, 2]`` 입니다.
        prev_heading: 구간 시작 시점 방향입니다. shape은 ``[n_agent]`` 입니다.
        min_total_disp: 0.5초 전체 변위를 신뢰할 최소 거리입니다.
        min_tail_disp: 마지막 0.1초 꼬리를 신뢰할 최소 거리입니다.

    Returns:
        Tensor: 구간 끝 방향입니다. shape은 ``[n_agent]`` 입니다.
    """
    total_vec = segment_pos[:, -1] - start_pos
    total_norm = torch.norm(total_vec, p=2, dim=-1)
    total_heading = torch.atan2(total_vec[:, 1], total_vec[:, 0])

    tail_vec = segment_pos[:, -1] - segment_pos[:, -2]
    tail_norm = torch.norm(tail_vec, p=2, dim=-1)
    tail_heading = torch.atan2(tail_vec[:, 1], tail_vec[:, 0])

    heading = prev_heading.clone()
    use_tail = tail_norm >= min_tail_disp
    heading = torch.where(use_tail, tail_heading, heading)
    use_total = total_norm >= min_total_disp
    heading = torch.where(use_total, total_heading, heading)
    return heading


def build_context_from_raw(
    pos_raw: Tensor,
    head_raw: Tensor,
    valid_raw: Tensor,
    shift: int,
    num_context_steps: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """10Hz 실제 좌표에서 0.5초 구간 문맥을 만듭니다.

    각 coarse slot은 0.5초 구간 하나를 뜻합니다. slot 내부 표현은 구간 시작 시점 기준
    local 좌표의 5개 점입니다. slot 끝 위치와 끝 방향도 같은 5개 점에서 다시 계산합니다.

    Args:
        pos_raw: 10Hz 실제 중심점입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
        head_raw: 10Hz 실제 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.
        valid_raw: 10Hz 유효 여부입니다. shape은 ``[n_agent, n_step]`` 입니다.
        shift: coarse 한 칸이 포함하는 10Hz step 수입니다. 여기서는 5입니다.
        num_context_steps: 만들 coarse slot 개수입니다. 보통 14입니다.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]:
            - motion_local: 각 slot의 5개 local 점입니다.
              shape은 ``[n_agent, num_context_steps, 5, 2]`` 입니다.
            - ctx_pos: 각 slot의 끝 위치입니다.
              shape은 ``[n_agent, num_context_steps, 2]`` 입니다.
            - ctx_heading: 각 slot의 끝 방향입니다.
              shape은 ``[n_agent, num_context_steps]`` 입니다.
            - ctx_valid: 각 slot의 유효 여부입니다.
              shape은 ``[n_agent, num_context_steps]`` 입니다.
    """
    motion_local_list: list[Tensor] = []
    ctx_pos_list: list[Tensor] = []
    ctx_heading_list: list[Tensor] = []
    ctx_valid_list: list[Tensor] = []

    for slot_idx in range(num_context_steps):
        end_idx = (slot_idx + 1) * shift
        start_idx = end_idx - shift

        start_pos = pos_raw[:, start_idx]
        start_heading = head_raw[:, start_idx]
        segment_pos = pos_raw[:, start_idx + 1 : end_idx + 1]
        segment_valid = valid_raw[:, start_idx : end_idx + 1].all(dim=1)

        motion_local = rotate_points_to_local(
            points_global=segment_pos,
            origin_pos=start_pos,
            origin_heading=start_heading,
        )
        end_pos = segment_pos[:, -1]
        end_heading = compute_segment_heading_from_points(
            start_pos=start_pos,
            segment_pos=segment_pos,
            prev_heading=start_heading,
        )

        motion_local = motion_local.clone()
        end_pos = end_pos.clone()
        end_heading = end_heading.clone()

        motion_local[~segment_valid] = 0.0
        end_pos[~segment_valid] = 0.0
        end_heading[~segment_valid] = 0.0

        motion_local_list.append(motion_local)
        ctx_pos_list.append(end_pos)
        ctx_heading_list.append(end_heading)
        ctx_valid_list.append(segment_valid)

    motion_local = torch.stack(motion_local_list, dim=1)
    ctx_pos = torch.stack(ctx_pos_list, dim=1)
    ctx_heading = torch.stack(ctx_heading_list, dim=1)
    ctx_valid = torch.stack(ctx_valid_list, dim=1)
    return motion_local, ctx_pos, ctx_heading, ctx_valid


def build_next_segment_from_commit(
    commit_pos: Tensor,
    current_pos: Tensor,
    current_heading: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """새로 commit된 0.5초 경로에서 다음 coarse 상태를 만듭니다.

    Args:
        commit_pos: 이번 step에서 commit된 5개 중심점입니다.
            shape은 ``[n_agent, 5, 2]`` 입니다.
        current_pos: commit 직전 coarse 위치입니다. shape은 ``[n_agent, 2]`` 입니다.
        current_heading: commit 직전 coarse 방향입니다. shape은 ``[n_agent]`` 입니다.

    Returns:
        tuple[Tensor, Tensor, Tensor]:
            - motion_local: 현재 상태 기준 local 5개 점입니다.
              shape은 ``[n_agent, 5, 2]`` 입니다.
            - next_pos: 다음 coarse 위치입니다. shape은 ``[n_agent, 2]`` 입니다.
            - next_heading: 다음 coarse 방향입니다. shape은 ``[n_agent]`` 입니다.
    """
    motion_local = rotate_points_to_local(
        points_global=commit_pos,
        origin_pos=current_pos,
        origin_heading=current_heading,
    )
    next_pos = commit_pos[:, -1]
    next_heading = compute_segment_heading_from_points(
        start_pos=current_pos,
        segment_pos=commit_pos,
        prev_heading=current_heading,
    )
    return motion_local, next_pos, next_heading


def build_motion_curve_token_features(motion_points_local: Tensor) -> Tensor:
    """5개 local 점을 기존 8차원 토큰 임베딩 입력으로 바꿉니다.

    기존 checkpoint와 최대한 잘 이어지도록, 새 learnable 분기는 추가하지 않고
    기존 type별 MLP 입력 크기 8을 유지합니다. 대신 5개 점에서 x/y 궤적 모양을 뽑아
    8차원 곡선 요약으로 넣습니다.

    Args:
        motion_points_local: local 5개 점입니다. shape은 ``[n_item, 5, 2]`` 입니다.

    Returns:
        Tensor: type별 MLP에 넣을 8차원 곡선 요약입니다.
        shape은 ``[n_item, 8]`` 입니다.
    """
    p1 = motion_points_local[:, 0]
    p2 = motion_points_local[:, 1]
    p3 = motion_points_local[:, 2]
    p4 = motion_points_local[:, 3]
    p5 = motion_points_local[:, 4]
    return torch.stack(
        [
            p1[:, 0],
            p1[:, 1],
            p2[:, 1],
            p3[:, 0],
            p3[:, 1],
            p4[:, 1],
            p5[:, 0],
            p5[:, 1],
        ],
        dim=-1,
    )


def build_motion_scalar_features(motion_points_local: Tensor) -> Tensor:
    """5개 local 점에서 2차원 스칼라 요약을 만듭니다.

    첫 번째 값은 0.5초 구간 전체 이동 길이입니다. 두 번째 값은 마지막 0.1초 이동 길이로,
    현재 속도 변화를 더 직접 반영합니다.

    Args:
        motion_points_local: local 5개 점입니다. shape은 ``[n_item, 5, 2]`` 입니다.

    Returns:
        Tensor: ``[전체 길이, 마지막 0.1초 길이]`` 입니다.
        shape은 ``[n_item, 2]`` 입니다.
    """
    start_zero = motion_points_local.new_zeros((motion_points_local.shape[0], 1, 2))
    path_points = torch.cat([start_zero, motion_points_local], dim=1)
    delta = path_points[:, 1:] - path_points[:, :-1]
    step_length = torch.norm(delta, p=2, dim=-1)
    total_length = step_length.sum(dim=-1)
    tail_length = step_length[:, -1]
    return torch.stack([total_length, tail_length], dim=-1)
