from __future__ import annotations

import torch
from torch import Tensor


def wrap_angle(angle: Tensor) -> Tensor:
    """각도를 안정적인 범위로 정리합니다.

    Args:
        angle: 임의 shape의 각도 tensor입니다.

    Returns:
        Tensor: 입력과 같은 shape이며, 값은 ``[-pi, pi]`` 범위로 정리됩니다.
    """
    return torch.atan2(angle.sin(), angle.cos())


def _reshape_agent_shape(shape_lwh: Tensor, target_dim: int) -> tuple[Tensor, Tensor]:
    """agent 크기를 후보 궤적 shape에 맞게 넓힙니다.

    Args:
        shape_lwh: agent별 ``[length, width, height]`` tensor입니다.
            shape은 ``[n_agent, 3]`` 입니다.
        target_dim: 중심점 tensor의 차원 수입니다.

    Returns:
        tuple[Tensor, Tensor]: 길이와 폭 tensor입니다. 반환 shape은 중심점 tensor에
        바로 broadcast 될 수 있도록 ``[1, n_agent, 1]`` 또는 ``[n_agent, 1]`` 입니다.
    """
    if shape_lwh.dim() != 2 or shape_lwh.shape[-1] < 2:
        raise ValueError(
            "shape_lwh must have shape [n_agent, >=2], "
            f"got {tuple(shape_lwh.shape)}."
        )
    length = shape_lwh[:, 0]
    width = shape_lwh[:, 1]
    if target_dim == 4:
        return length.view(1, -1, 1), width.view(1, -1, 1)
    if target_dim == 3:
        return length.view(-1, 1), width.view(-1, 1)
    raise ValueError(f"center_xy must be [K,N,T,2] or [N,T,2], got dim={target_dim}.")


def build_box_corners(center_xy: Tensor, heading: Tensor, shape_lwh: Tensor) -> Tensor:
    """중심점, 방향, 크기에서 사각형 꼭지점 네 개를 만듭니다.

    Args:
        center_xy: agent 중심점입니다. shape은 ``[n_candidate, n_agent, n_step, 2]``
            또는 ``[n_agent, n_step, 2]`` 입니다.
        heading: agent 방향입니다. shape은 ``center_xy``에서 마지막 좌표축을 뺀
            shape과 같습니다.
        shape_lwh: agent 크기입니다. shape은 ``[n_agent, 3]`` 이며 앞 두 값은
            길이와 폭입니다.

    Returns:
        Tensor: 사각형 꼭지점입니다. shape은 ``center_xy.shape[:-1] + [4, 2]`` 입니다.
    """
    if center_xy.shape[:-1] != heading.shape:
        raise ValueError(
            "center_xy and heading shapes do not match: "
            f"center_xy={tuple(center_xy.shape)}, heading={tuple(heading.shape)}."
        )
    if center_xy.shape[-1] != 2:
        raise ValueError(f"center_xy last dim must be 2, got {center_xy.shape[-1]}.")

    length, width = _reshape_agent_shape(shape_lwh.to(center_xy.device), center_xy.dim())
    half_l = length.to(center_xy.dtype) * 0.5
    half_w = width.to(center_xy.dtype) * 0.5

    local_x = torch.stack([half_l, half_l, -half_l, -half_l], dim=-1)
    local_y = torch.stack([half_w, -half_w, -half_w, half_w], dim=-1)

    cos_h = heading.cos().unsqueeze(-1)
    sin_h = heading.sin().unsqueeze(-1)
    corner_x = local_x * cos_h - local_y * sin_h
    corner_y = local_x * sin_h + local_y * cos_h
    corners = torch.stack([corner_x, corner_y], dim=-1)
    return center_xy.unsqueeze(-2) + corners


def corner_distance_score(
    pred_xy: Tensor,
    pred_heading: Tensor,
    gt_xy: Tensor,
    gt_heading: Tensor,
    shape_lwh: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """RoaD 후보 선택에 쓰는 사각형 꼭지점 평균 거리를 계산합니다.

    Args:
        pred_xy: 후보 중심점입니다. shape은 ``[n_candidate, n_agent, n_step, 2]`` 입니다.
        pred_heading: 후보 방향입니다. shape은 ``[n_candidate, n_agent, n_step]`` 입니다.
        gt_xy: 정답 중심점입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
        gt_heading: 정답 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.
        shape_lwh: agent 크기입니다. shape은 ``[n_agent, 3]`` 입니다.
        valid_mask: 점수에 포함할 정답 유효 여부입니다. shape은 ``[n_agent, n_step]`` 입니다.

    Returns:
        Tensor: candidate와 agent별 평균 거리입니다. shape은 ``[n_candidate, n_agent]`` 입니다.
    """
    if pred_xy.dim() != 4:
        raise ValueError(f"pred_xy must have shape [K,N,T,2], got {tuple(pred_xy.shape)}.")
    if gt_xy.dim() != 3:
        raise ValueError(f"gt_xy must have shape [N,T,2], got {tuple(gt_xy.shape)}.")
    if pred_xy.shape[1:3] != gt_xy.shape[:2]:
        raise ValueError(
            "Candidate and GT horizon shapes do not match: "
            f"pred={tuple(pred_xy.shape)}, gt={tuple(gt_xy.shape)}."
        )

    pred_corners = build_box_corners(pred_xy, pred_heading, shape_lwh)
    gt_corners = build_box_corners(gt_xy, gt_heading, shape_lwh).unsqueeze(0)
    corner_dist = torch.linalg.norm(pred_corners - gt_corners, dim=-1).mean(dim=-1)

    mask = valid_mask.to(device=pred_xy.device, dtype=torch.bool).unsqueeze(0)
    masked_dist = corner_dist.masked_fill(~mask, 0.0)
    denom = mask.sum(dim=-1).clamp_min(1).to(dtype=pred_xy.dtype)
    score = masked_dist.sum(dim=-1) / denom
    no_valid = mask.sum(dim=-1) == 0
    return score.masked_fill(no_valid, torch.inf)
