from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from torch import Tensor

from src.smart.utils.geometry import wrap_angle
from src.smart.utils.rollout import cal_polygon_contour, transform_to_global, transform_to_local


SEGMENT_STARTS: Tuple[int, int, int, int] = (0, 5, 10, 15)
SEGMENT_LEN: int = 6


def normalize_sincos(traj: Tensor, eps: float = 1e-6) -> Tensor:
    """`(sin, cos)` 두 값을 다시 길이 1로 맞춥니다.

    Args:
        traj: 마지막 축의 뒤 2개가 `(sin, cos)`인 텐서입니다.
            예시 shape:
                - `[n_agent, 21, 4]`
                - `[n_agent, 4, 6, 4]`
        eps: 0으로 나누는 문제를 막기 위한 작은 값입니다.

    Returns:
        같은 shape의 텐서입니다.
    """
    out = traj.clone()
    vec = out[..., 2:4]
    denom = torch.clamp(torch.norm(vec, dim=-1, keepdim=True), min=eps)
    out[..., 2:4] = vec / denom
    return out



def chunk_future_21_to_4x6(future: Tensor) -> Tensor:
    """21개 점 미래를 4개의 겹치는 조각으로 바꿉니다.

    Args:
        future: 로컬 좌표 미래입니다.
            shape: `[n_agent, 21, 4]`

    Returns:
        조각 미래입니다.
        shape: `[n_agent, 4, 6, 4]`
    """
    chunks = [future[:, start : start + SEGMENT_LEN] for start in SEGMENT_STARTS]
    return torch.stack(chunks, dim=1)



def chunk_valid_21_to_4x6(valid: Tensor) -> Tensor:
    """21개 점 유효 마스크를 조각 단위 마스크로 바꿉니다.

    Args:
        valid: 점 단위 유효 여부입니다.
            shape: `[n_agent, 21]`

    Returns:
        조각 단위 유효 마스크입니다.
        shape: `[n_agent, 4, 6]`
    """
    chunks = [valid[:, start : start + SEGMENT_LEN] for start in SEGMENT_STARTS]
    return torch.stack(chunks, dim=1)



def assemble_4x6_to_21(segments: Tensor) -> Tensor:
    """4개의 겹치는 조각을 다시 21개 점 미래로 합칩니다.

    겹치는 경계점은 평균으로 합치고, 마지막에 `(sin, cos)`를 다시 정규화합니다.

    Args:
        segments: 조각 미래입니다.
            shape: `[n_agent, 4, 6, 4]`

    Returns:
        다시 합친 미래입니다.
        shape: `[n_agent, 21, 4]`
    """
    n_agent = segments.shape[0]
    out = segments.new_zeros((n_agent, 21, 4))
    cnt = segments.new_zeros((n_agent, 21, 1))
    for seg_idx, start in enumerate(SEGMENT_STARTS):
        out[:, start : start + SEGMENT_LEN] += segments[:, seg_idx]
        cnt[:, start : start + SEGMENT_LEN] += 1.0
    out = out / cnt.clamp_min(1.0)
    out[:, 0, 0] = 0.0
    out[:, 0, 1] = 0.0
    out[:, 0, 2] = 0.0
    out[:, 0, 3] = 1.0
    return normalize_sincos(out)



def overlap_consistency_residual(segments: Tensor) -> Tensor:
    """이웃 조각 경계가 얼마나 안 맞는지 계산합니다.

    Args:
        segments: 조각 미래입니다.
            shape: `[n_agent, 4, 6, 4]`

    Returns:
        경계 차이입니다.
        shape: `[n_agent, 3, 4]`
    """
    return segments[:, :-1, -1] - segments[:, 1:, 0]



def build_ot_flow_path(target: Tensor, noise_scale: float, eps: float = 1e-3) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """선형 OT 경로용 noised 입력과 정답 속도를 만듭니다.

    Args:
        target: 정답 조각 미래입니다.
            shape: `[n_agent, 4, 6, 4]`
        noise_scale: 시작 잡음 크기입니다.
        eps: 시간 값이 0이나 1에 너무 붙지 않게 막는 작은 값입니다.

    Returns:
        noise: 시작 잡음.
            shape: `[n_agent, 4, 6, 4]`
        z_tau: 시간 `tau`에서의 noised 입력.
            shape: `[n_agent, 4, 6, 4]`
        tau: agent별 시간 값.
            shape: `[n_agent]`
        target_velocity: 맞춰야 하는 속도장.
            shape: `[n_agent, 4, 6, 4]`
    """
    tau = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
    tau = tau.clamp(min=eps, max=1.0 - eps)
    tau_view = tau.view(-1, 1, 1, 1)
    noise = torch.randn_like(target) * noise_scale
    noise[:, 0, 0, 0] = 0.0
    noise[:, 0, 0, 1] = 0.0
    noise[:, 0, 0, 2] = 0.0
    noise[:, 0, 0, 3] = 1.0
    z_tau = (1.0 - tau_view) * noise + tau_view * target
    z_tau[:, 0, 0, 0] = 0.0
    z_tau[:, 0, 0, 1] = 0.0
    z_tau[:, 0, 0, 2] = 0.0
    z_tau[:, 0, 0, 3] = 1.0
    z_tau = normalize_sincos(z_tau)
    target_velocity = target - noise
    return noise, z_tau, tau, target_velocity



def build_anchor_10hz_indices(
    num_historical_steps: int,
    future_window_steps: int,
    total_steps: int = 91,
    shift: int = 5,
) -> List[int]:
    """학습에 사용할 10Hz anchor 시각 후보를 만듭니다."""
    start = num_historical_steps - 1
    last = total_steps - future_window_steps - 1
    return list(range(start, last + 1, shift))



def sample_anchor_10hz_indices(
    candidate_anchors: Iterable[int],
    anchor_chunk_k: int,
    device: torch.device,
) -> List[int]:
    """anchor 후보 중 일부만 무작위로 고릅니다."""
    candidates = list(candidate_anchors)
    if len(candidates) <= anchor_chunk_k:
        return candidates
    perm = torch.randperm(len(candidates), device=device)[:anchor_chunk_k].cpu().tolist()
    return [candidates[i] for i in perm]



def build_local_future_target(
    pos_global: Tensor,
    head_global: Tensor,
    valid_mask: Tensor,
    anchor_10hz: int,
    anchor_pos: Tensor,
    anchor_head: Tensor,
    future_window_steps: int,
) -> Tuple[Tensor, Tensor]:
    """anchor 기준 로컬 미래 정답을 만듭니다."""
    end = anchor_10hz + future_window_steps + 1
    pos_slice = pos_global[:, anchor_10hz:end]
    head_slice = head_global[:, anchor_10hz:end]
    valid_slice = valid_mask[:, anchor_10hz:end]

    pos_local, _ = transform_to_local(
        pos_global=pos_slice,
        head_global=None,
        pos_now=anchor_pos,
        head_now=anchor_head,
    )
    delta_head = wrap_angle(head_slice - anchor_head.unsqueeze(1))
    future_local = torch.cat(
        [
            pos_local,
            delta_head.sin().unsqueeze(-1),
            delta_head.cos().unsqueeze(-1),
        ],
        dim=-1,
    )
    future_local[:, 0, 0] = 0.0
    future_local[:, 0, 1] = 0.0
    future_local[:, 0, 2] = 0.0
    future_local[:, 0, 3] = 1.0
    future_local = normalize_sincos(future_local)
    return future_local, valid_slice



def build_current_anchor_feature(
    current_head: Tensor,
    current_vel_global: Tensor,
    current_yaw_rate: Tensor,
    agent_shape: Tensor,
    agent_type: Tensor,
) -> Tensor:
    """현재 정확한 상태를 8차원 anchor 입력으로 바꿉니다."""
    cos_h = current_head.cos()
    sin_h = current_head.sin()
    vx_local = current_vel_global[:, 0] * cos_h + current_vel_global[:, 1] * sin_h
    vy_local = -current_vel_global[:, 0] * sin_h + current_vel_global[:, 1] * cos_h
    return torch.stack(
        [
            vx_local,
            vy_local,
            current_head.sin(),
            current_head.cos(),
            current_yaw_rate,
            agent_shape[:, 0],
            agent_shape[:, 1],
            agent_type.float(),
        ],
        dim=-1,
    )



def segment_local_to_global(
    segment_local: Tensor,
    current_pos: Tensor,
    current_head: Tensor,
) -> Tuple[Tensor, Tensor]:
    """첫 0.5초 조각을 글로벌 위치/heading으로 바꿉니다."""
    local_pos = segment_local[..., :2]
    local_head = torch.atan2(segment_local[..., 2], segment_local[..., 3])
    pos_global, head_global = transform_to_global(
        pos_local=local_pos,
        head_local=local_head,
        pos_now=current_pos,
        head_now=current_head,
    )
    return pos_global, wrap_angle(head_global)



def segment_endpoint_pose_global(
    segments_local: Tensor,
    current_pos: Tensor,
    current_head: Tensor,
) -> Tuple[Tensor, Tensor]:
    """각 조각의 마지막 점을 글로벌 pose로 바꿉니다."""
    flat_pos_local = segments_local[:, :, -1, :2].reshape(-1, 1, 2)
    flat_head_local = torch.atan2(
        segments_local[:, :, -1, 2].reshape(-1, 1),
        segments_local[:, :, -1, 3].reshape(-1, 1),
    )
    flat_anchor_pos = current_pos.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 2)
    flat_anchor_head = current_head.unsqueeze(1).expand(-1, 4).reshape(-1)
    pos_global, head_global = transform_to_global(
        pos_local=flat_pos_local,
        head_local=flat_head_local,
        pos_now=flat_anchor_pos,
        head_now=flat_anchor_head,
    )
    return pos_global[:, 0].reshape(-1, 4, 2), wrap_angle(head_global[:, 0].reshape(-1, 4))



def match_first_segment_token(
    first_segment_local: Tensor,
    token_traj_all: Tensor,
    token_agent_shape: Tensor,
) -> Tensor:
    """예측한 첫 0.5초 조각을 가장 가까운 SMART token으로 바꿉니다."""
    local_pos = first_segment_local[..., :2]
    local_head = torch.atan2(first_segment_local[..., 2], first_segment_local[..., 3])
    contour = cal_polygon_contour(
        pos=local_pos,
        head=local_head,
        width_length=token_agent_shape.unsqueeze(1),
    )
    dist = torch.norm(token_traj_all - contour.unsqueeze(1), dim=-1).mean(dim=(-1, -2))
    return torch.argmin(dist, dim=-1)
