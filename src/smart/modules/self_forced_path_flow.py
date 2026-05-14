from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from src.smart.modules.kinematic_control import (
    CONTROL_FLOW_DIM,
    POSE_FLOW_DIM,
    POSE_NORM_POS_SCALE_M,
    build_rolling_control_target,
)
from src.smart.utils import transform_to_local


def masked_mean_square_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    eps: float = 1.0e-6,
) -> Tensor:
    """마스크를 고려한 평균 제곱 오차를 계산합니다.

    Args:
        pred: 비교할 예측 텐서입니다. shape은 보통 ``[n, T, C]`` 입니다.
        target: 예측과 같은 shape의 목표 텐서입니다.
        mask: 선택적으로 적용할 유효 마스크입니다. ``None`` 이면 모든 값을 씁니다.
            shape은 ``pred`` 에 broadcast 가능한 형태여야 합니다.
        eps: 분모가 0이 되는 상황을 막기 위한 작은 값입니다.

    Returns:
        Tensor: 평균 제곱 오차 스칼라입니다. shape은 ``[]`` 입니다.
    """
    diff_square = (pred - target).square()
    if mask is None:
        return diff_square.mean()

    weight = mask.to(device=pred.device, dtype=pred.dtype)
    while weight.dim() < diff_square.dim():
        weight = weight.unsqueeze(-1)
    return (diff_square * weight).sum() / weight.expand_as(diff_square).sum().clamp_min(eps)


def get_anchor0_valid_mask(tokenized_agent: Dict[str, Tensor]) -> Tensor:
    """초기 1초 context에서 시작하는 첫 flow anchor의 유효 agent를 고릅니다.

    Args:
        tokenized_agent: ``FlowTokenProcessor`` 가 만든 에이전트 사전입니다.
            평가 모드에서는 ``flow_eval_mask`` 와 closed-loop rollout용 상태가 들어 있습니다.

    Returns:
        Tensor: 첫 anchor를 self-forced 학습에 쓸지 나타내는 마스크입니다.
            shape은 ``[n_agent]`` 입니다.
    """
    if "flow_eval_mask" in tokenized_agent:
        flow_eval_mask = tokenized_agent["flow_eval_mask"]
        if flow_eval_mask.dim() != 2 or flow_eval_mask.shape[1] == 0:
            raise ValueError(
                "flow_eval_mask must have shape [n_agent, n_anchor] with at least one anchor."
            )
        return flow_eval_mask[:, 0].bool()

    if "valid_mask" not in tokenized_agent:
        raise KeyError("tokenized_agent must contain either flow_eval_mask or valid_mask.")
    valid_mask = tokenized_agent["valid_mask"]
    if valid_mask.dim() != 2 or valid_mask.shape[1] == 0:
        raise ValueError("valid_mask must have shape [n_agent, n_step].")
    return valid_mask[:, -1].bool()


def build_anchor0_normalized_committed_path(
    pred_traj_10hz: Tensor,
    pred_head_10hz: Tensor,
    tokenized_agent: Dict[str, Tensor],
    flow_window_steps: int,
    pos_scale_m: float = 20.0,
) -> Tensor:
    """실제로 commit된 N초 rollout을 첫 anchor 기준 정규화 path로 바꿉니다.

    Args:
        pred_traj_10hz: closed-loop에서 실제 실행된 중심점입니다.
            shape은 ``[n_agent, T_rollout, 2]`` 입니다.
        pred_head_10hz: 같은 rollout의 heading입니다. shape은 ``[n_agent, T_rollout]`` 입니다.
        tokenized_agent: 평가 모드 토큰 사전입니다. ``ctx_sampled_pos`` 와
            ``ctx_sampled_heading`` 이 들어 있어야 합니다.
        flow_window_steps: pretrain flow window 길이입니다. 10Hz step 수입니다.
        pos_scale_m: flow 학습에서 위치를 정규화할 때 쓴 meter scale입니다.

    Returns:
        Tensor: 첫 anchor 기준의 정규화된 committed rollout입니다.
            shape은 ``[n_agent, flow_window_steps, 4]`` 이고 마지막 차원은
            ``[x/scale, y/scale, cos(local_heading), sin(local_heading)]`` 입니다.
    """
    if pred_traj_10hz.dim() != 3 or pred_traj_10hz.shape[-1] != 2:
        raise ValueError("pred_traj_10hz must have shape [n_agent, T, 2].")
    if pred_head_10hz.shape[:2] != pred_traj_10hz.shape[:2]:
        raise ValueError("pred_head_10hz must have shape [n_agent, T] matching pred_traj_10hz.")
    if pred_traj_10hz.shape[1] < flow_window_steps:
        raise ValueError(
            "Committed rollout is shorter than flow_window_steps: "
            f"got {pred_traj_10hz.shape[1]} and {flow_window_steps}."
        )
    if "ctx_sampled_pos" not in tokenized_agent or "ctx_sampled_heading" not in tokenized_agent:
        raise KeyError("tokenized_agent must contain ctx_sampled_pos and ctx_sampled_heading.")

    # path_pos/head shape: [n_agent, flow_window_steps, 2] / [n_agent, flow_window_steps]
    path_pos = pred_traj_10hz[:, :flow_window_steps]
    path_head = pred_head_10hz[:, :flow_window_steps]

    # anchor 0은 history 마지막 1초 시점(raw step 10)을 뜻하므로 ctx slot 1을 원점으로 씁니다.
    current_pos = tokenized_agent["ctx_sampled_pos"][:, 1]
    current_head = tokenized_agent["ctx_sampled_heading"][:, 1]
    path_pos_local, path_head_local = transform_to_local(
        pos_global=path_pos,
        head_global=path_head,
        pos_now=current_pos,
        head_now=current_head,
    )
    return torch.stack(
        [
            path_pos_local[..., 0] / float(pos_scale_m),
            path_pos_local[..., 1] / float(pos_scale_m),
            path_head_local.cos(),
            path_head_local.sin(),
        ],
        dim=-1,
    )


def build_anchor0_normalized_committed_control(
    committed_path_norm: Tensor,
    tokenized_agent: Dict[str, Tensor],
    anchor_mask: Tensor,
    *,
    pos_scale_m: float,
    vehicle_yaw_scale_rad: float,
    pedestrian_yaw_scale_rad: float,
    cyclist_yaw_scale_rad: float,
    use_holonomic_model_only: bool = False,
    use_rolling_supervision: bool = True,
    no_slip_point_ratio: float = 0.0,
    pose_pos_scale_m: float = POSE_NORM_POS_SCALE_M,
) -> Tensor:
    """첫 anchor 기준 pose rollout을 control-space self-forced flow state로 바꿉니다.

    Args:
        committed_path_norm: ``build_anchor0_normalized_committed_path`` 가 만든
            anchor-local pose path입니다. shape은 ``[n_valid_agent, T, 4]`` 입니다.
        tokenized_agent: 평가 모드 토큰 사전입니다. ``type`` 이 들어 있어야 합니다.
        anchor_mask: 첫 anchor에서 사용할 agent mask입니다. shape은 ``[n_agent]`` 입니다.
        pos_scale_m: control 이동량 정규화 scale입니다.
        vehicle_yaw_scale_rad: vehicle yaw 정규화 scale입니다.
        pedestrian_yaw_scale_rad: pedestrian yaw 정규화 scale입니다.
        cyclist_yaw_scale_rad: cyclist yaw 정규화 scale입니다.
        use_holonomic_model_only: ``True`` 이면 모든 agent type에 holonomic control
            projection을 씁니다.
        use_rolling_supervision: ``True`` 이면 decoder-consistent rolling projection을
            사용하고, ``False`` 이면 raw pose pair inverse를 사용합니다.
        no_slip_point_ratio: vehicle/cyclist box length에 곱하는 no-slip point offset 비율입니다.
        pose_pos_scale_m: pose-space 위치 정규화 scale입니다.

    Returns:
        Tensor: 정규화된 rolling control path입니다.
            shape은 ``[n_valid_agent, T, 3]`` 입니다.
    """
    if committed_path_norm.ndim != 3 or committed_path_norm.shape[-1] != POSE_FLOW_DIM:
        raise ValueError(
            "committed_path_norm must have shape [n_valid_agent, T, 4], "
            f"got {tuple(committed_path_norm.shape)}."
        )
    if "type" not in tokenized_agent:
        raise KeyError("tokenized_agent must contain type for control-space self-forced loss.")
    if no_slip_point_ratio > 0.0 and "shape" not in tokenized_agent:
        raise KeyError(
            "tokenized_agent must contain shape when no_slip_point_ratio > 0 "
            "for control-space self-forced loss."
        )
    if anchor_mask.ndim != 1:
        raise ValueError(f"anchor_mask must have shape [n_agent], got {tuple(anchor_mask.shape)}.")

    agent_type = tokenized_agent["type"][anchor_mask].to(device=committed_path_norm.device)
    agent_length = (
        tokenized_agent["shape"][anchor_mask, 0].to(device=committed_path_norm.device)
        if "shape" in tokenized_agent
        else None
    )
    if agent_type.shape[0] != committed_path_norm.shape[0]:
        raise ValueError(
            "anchor_mask selected agent count must match committed_path_norm batch: "
            f"got {agent_type.shape[0]} and {committed_path_norm.shape[0]}."
        )
    if committed_path_norm.shape[0] == 0:
        return committed_path_norm.new_zeros((0, committed_path_norm.shape[1], CONTROL_FLOW_DIM))

    future_pos = committed_path_norm[..., :2] * float(pose_pos_scale_m)
    future_head = torch.atan2(committed_path_norm[..., 3], committed_path_norm[..., 2])
    current_pos = future_pos.new_zeros((future_pos.shape[0], 2))
    current_head = future_head.new_zeros((future_head.shape[0],))
    return build_rolling_control_target(
        future_pos=future_pos,
        future_head=future_head,
        current_pos=current_pos,
        current_head=current_head,
        agent_type=agent_type,
        agent_length=agent_length,
        pos_scale_m=pos_scale_m,
        vehicle_yaw_scale_rad=vehicle_yaw_scale_rad,
        pedestrian_yaw_scale_rad=pedestrian_yaw_scale_rad,
        cyclist_yaw_scale_rad=cyclist_yaw_scale_rad,
        use_holonomic_model_only=use_holonomic_model_only,
        use_rolling_supervision=use_rolling_supervision,
        no_slip_point_ratio=no_slip_point_ratio,
    )
