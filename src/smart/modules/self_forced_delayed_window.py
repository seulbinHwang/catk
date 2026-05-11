from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor

from src.smart.utils import transform_to_local


@dataclass(frozen=True)
class SelfForcedDelayedWindow:
    """self-forcing에서 이번 epoch가 학습할 2초 구간을 담습니다.

    Attributes:
        stage_index: 4 epoch 단위 단계입니다. 0, 1, 2, 3 중 하나입니다.
        skipped_blocks_2hz: 앞에서 학습하지 않고 굴리기만 할 0.5초 block 개수입니다.
        target_blocks_2hz: 실제 학습할 0.5초 block 개수입니다. 2초 horizon이면 4입니다.
        rollout_steps_2hz: 전체로 굴릴 0.5초 block 개수입니다.
        start_step_10hz: 실제 학습 구간이 시작되는 10Hz 미래 index입니다.
        target_steps_10hz: 실제 학습할 미래 step 수입니다. 2초 horizon이면 20입니다.
        anchor_offset: 기존 13개 anchor 중 현재 학습 시작 시점에 해당하는 offset입니다.
        start_seconds: 실제 학습 구간 시작 시간입니다.
        end_seconds: 실제 학습 구간 종료 시간입니다.
    """

    stage_index: int
    skipped_blocks_2hz: int
    target_blocks_2hz: int
    rollout_steps_2hz: int
    start_step_10hz: int
    target_steps_10hz: int
    anchor_offset: int
    start_seconds: float
    end_seconds: float


def resolve_self_forced_delayed_window(
    current_epoch: int,
    start_epoch: int,
    flow_window_steps: int,
    commit_steps: int,
    stage_epochs: int = 4,
    max_stage_index: int = 3,
    enabled: bool = True,
) -> SelfForcedDelayedWindow:
    """epoch에 맞는 지연 시작 self-forcing 학습 구간을 정합니다.

    Args:
        current_epoch: 현재 학습 epoch입니다.
        start_epoch: self-forcing이 시작되는 epoch입니다.
        flow_window_steps: 한 번에 학습할 미래 길이입니다. shape 기준으로는
            ``[N, flow_window_steps, 4]`` 의 가운데 차원입니다.
        commit_steps: 한 번 commit하는 10Hz step 수입니다. 0.5초 commit이면 5입니다.
        stage_epochs: 같은 시작 시점을 유지할 epoch 수입니다. 기본값 4는
            0~3, 4~7, 8~11, 12~15 schedule을 만듭니다.
        max_stage_index: 마지막 stage 번호입니다. 기본값 3은 6~8초 stage에서 멈춥니다.
        enabled: 지연 시작 기능을 쓸지 여부입니다. 꺼져 있으면 항상 0~2초만 학습합니다.

    Returns:
        SelfForcedDelayedWindow: 이번 epoch에서 전체로 굴릴 길이와 실제 학습할
        2초 구간 정보를 담은 값입니다.

    설명:
        기능이 켜져 있을 때는 flow horizon을 2초로 고정합니다. 즉
        ``flow_window_steps=20`` 과 ``commit_steps=5`` 를 기대합니다. 이렇게 해야
        stage가 정확히 0~2초, 2~4초, 4~6초, 6~8초가 됩니다.
    """
    if commit_steps <= 0:
        raise ValueError(f"commit_steps must be positive, got {commit_steps}.")
    if flow_window_steps % commit_steps != 0:
        raise ValueError(
            "flow_window_steps must be divisible by commit_steps, "
            f"got flow_window_steps={flow_window_steps}, commit_steps={commit_steps}."
        )
    if stage_epochs <= 0:
        raise ValueError(f"stage_epochs must be positive, got {stage_epochs}.")
    if max_stage_index < 0:
        raise ValueError(f"max_stage_index must be non-negative, got {max_stage_index}.")
    if enabled and flow_window_steps != 20:
        raise ValueError(
            "Delayed-window self-forcing is intentionally fixed to a 2-second target. "
            f"Set decoder.flow_window_steps=20, got {flow_window_steps}."
        )

    target_blocks_2hz = int(flow_window_steps // commit_steps)
    if not enabled:
        stage_index = 0
    else:
        relative_epoch = max(0, int(current_epoch) - int(start_epoch))
        stage_index = min(int(relative_epoch // stage_epochs), int(max_stage_index))

    skipped_blocks_2hz = int(stage_index * target_blocks_2hz)
    rollout_steps_2hz = int(skipped_blocks_2hz + target_blocks_2hz)
    start_step_10hz = int(skipped_blocks_2hz * commit_steps)
    start_seconds = float(start_step_10hz) * 0.1
    end_seconds = start_seconds + float(flow_window_steps) * 0.1

    return SelfForcedDelayedWindow(
        stage_index=int(stage_index),
        skipped_blocks_2hz=skipped_blocks_2hz,
        target_blocks_2hz=target_blocks_2hz,
        rollout_steps_2hz=rollout_steps_2hz,
        start_step_10hz=start_step_10hz,
        target_steps_10hz=int(flow_window_steps),
        anchor_offset=skipped_blocks_2hz,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


def _clone_tensor_dict_value(value: object) -> object:
    """dict 값을 안전하게 복사합니다.

    Args:
        value: tensor 또는 일반 값입니다. tensor shape은 key마다 다릅니다.

    Returns:
        object: tensor이면 clone한 tensor이고, 일반 값이면 그대로 반환합니다.
    """
    if torch.is_tensor(value):
        return value.clone()
    return value


def build_delayed_anchor0_tokenized_agent(
    tokenized_agent: Dict[str, Tensor],
    pred_traj_10hz: Tensor,
    pred_head_10hz: Tensor,
    window: SelfForcedDelayedWindow,
    commit_steps: int = 5,
) -> Dict[str, Tensor]:
    """지연 시작 시점을 기존 anchor 0처럼 보이도록 agent 입력을 정리합니다.

    Args:
        tokenized_agent: 기존 token processor가 만든 agent 사전입니다.
            ``ctx_sampled_pos`` shape은 ``[N, 14, 2]`` 입니다.
            ``ctx_sampled_heading`` shape은 ``[N, 14]`` 입니다.
            ``flow_eval_mask`` shape은 ``[N, 13]`` 입니다.
        pred_traj_10hz: 현재 모델이 앞에서 스스로 만든 closed-loop 위치입니다.
            shape은 ``[N, T, 2]`` 입니다.
        pred_head_10hz: 현재 모델이 앞에서 스스로 만든 closed-loop heading입니다.
            shape은 ``[N, T]`` 입니다.
        window: 이번 epoch가 사용할 지연 시작 구간 정보입니다.
        commit_steps: 한 번 commit하는 10Hz step 수입니다. 기본값 5는 0.5초입니다.

    Returns:
        Dict[str, Tensor]: anchor 0이 지연 시작 시점을 가리키도록 정리된 agent 사전입니다.

    설명:
        앞구간에서 만든 현재 위치와 방향은 context로만 씁니다. 그래서 이 함수는
        그 값을 detach해서 뒤 2초 loss가 앞구간 실행 결과를 직접 바꾸지 않게 합니다.
        학습 구간 내부 2초는 별도로 detach하지 않습니다.
    """
    if pred_traj_10hz.dim() != 3 or pred_traj_10hz.shape[-1] != 2:
        raise ValueError("pred_traj_10hz must have shape [N, T, 2].")
    if pred_head_10hz.shape[:2] != pred_traj_10hz.shape[:2]:
        raise ValueError("pred_head_10hz must have shape [N, T] matching pred_traj_10hz.")
    if "flow_eval_mask" not in tokenized_agent:
        raise KeyError("tokenized_agent must contain flow_eval_mask.")
    flow_eval_mask = tokenized_agent["flow_eval_mask"]
    if flow_eval_mask.dim() != 2:
        raise ValueError("flow_eval_mask must have shape [N, 13].")
    if window.anchor_offset >= flow_eval_mask.shape[1]:
        raise ValueError(
            "Delayed-window anchor offset exceeds available anchors: "
            f"anchor_offset={window.anchor_offset}, n_anchor={flow_eval_mask.shape[1]}."
        )

    delayed_agent = {
        key: _clone_tensor_dict_value(value)
        for key, value in tokenized_agent.items()
    }
    delayed_agent["_self_forced_delayed_start_step_10hz"] = int(window.start_step_10hz)
    delayed_agent["_self_forced_delayed_anchor_offset"] = int(window.anchor_offset)

    # flow_eval_mask: [N, 13]. anchor 0이 이번 delayed window의 시작 시점을 보게 합니다.
    delayed_agent["flow_eval_mask"][:, 0] = flow_eval_mask[:, window.anchor_offset].bool()

    source_ctx_slot = int(window.anchor_offset + 1)
    if "ctx_valid" in delayed_agent and source_ctx_slot < delayed_agent["ctx_valid"].shape[1]:
        # ctx_valid: [N, 14]
        delayed_agent["ctx_valid"][:, 1] = delayed_agent["ctx_valid"][:, source_ctx_slot]
    if "ctx_sampled_idx" in delayed_agent and source_ctx_slot < delayed_agent["ctx_sampled_idx"].shape[1]:
        # ctx_sampled_idx: [N, 14]
        delayed_agent["ctx_sampled_idx"][:, 1] = delayed_agent["ctx_sampled_idx"][:, source_ctx_slot]

    if window.start_step_10hz <= 0:
        return delayed_agent

    current_index = int(window.start_step_10hz - 1)
    if pred_traj_10hz.shape[1] <= current_index:
        raise ValueError(
            "pred_traj_10hz is shorter than the delayed current point: "
            f"T={pred_traj_10hz.shape[1]}, required_index={current_index}."
        )

    # 현재 slot 1: D초 시점의 자기 생성 현재 상태입니다.
    # ctx_sampled_pos[:, 1]: [N, 2], ctx_sampled_heading[:, 1]: [N]
    delayed_agent["ctx_sampled_pos"][:, 1] = pred_traj_10hz[:, current_index].detach()
    delayed_agent["ctx_sampled_heading"][:, 1] = pred_head_10hz[:, current_index].detach()

    prev_index = current_index - int(commit_steps)
    if prev_index >= 0:
        # 이전 slot 0: D-0.5초 자기 생성 상태입니다.
        delayed_agent["ctx_sampled_pos"][:, 0] = pred_traj_10hz[:, prev_index].detach()
        delayed_agent["ctx_sampled_heading"][:, 0] = pred_head_10hz[:, prev_index].detach()
        if "ctx_valid" in delayed_agent:
            delayed_agent["ctx_valid"][:, 0] = delayed_agent["ctx_valid"][:, 1]

    return delayed_agent


def build_delayed_normalized_committed_path(
    pred_traj_10hz: Tensor,
    pred_head_10hz: Tensor,
    tokenized_agent: Dict[str, Tensor],
    flow_window_steps: int,
    pos_scale_m: float = 20.0,
) -> Tensor:
    """이번 epoch의 실제 학습 2초 구간만 local path로 만듭니다.

    Args:
        pred_traj_10hz: closed-loop에서 실제 실행된 중심점입니다.
            shape은 ``[N, T, 2]`` 입니다.
        pred_head_10hz: 같은 rollout의 heading입니다. shape은 ``[N, T]`` 입니다.
        tokenized_agent: agent 사전입니다. ``_self_forced_delayed_start_step_10hz`` 가 있으면
            그 지점부터 2초를 자릅니다. 없으면 기존처럼 0초부터 씁니다.
        flow_window_steps: 실제 학습할 미래 step 수입니다. 2초면 20입니다.
        pos_scale_m: 위치 정규화에 쓰는 meter scale입니다.

    Returns:
        Tensor: 시작 시점을 새 원점으로 본 정규화 path입니다.
        shape은 ``[N, flow_window_steps, 4]`` 입니다.
    """
    if pred_traj_10hz.dim() != 3 or pred_traj_10hz.shape[-1] != 2:
        raise ValueError("pred_traj_10hz must have shape [N, T, 2].")
    if pred_head_10hz.shape[:2] != pred_traj_10hz.shape[:2]:
        raise ValueError("pred_head_10hz must have shape [N, T] matching pred_traj_10hz.")

    start_step_10hz = int(tokenized_agent.get("_self_forced_delayed_start_step_10hz", 0))
    end_step_10hz = start_step_10hz + int(flow_window_steps)
    if pred_traj_10hz.shape[1] < end_step_10hz:
        raise ValueError(
            "Committed rollout is shorter than the selected delayed window: "
            f"T={pred_traj_10hz.shape[1]}, required={end_step_10hz}."
        )

    # path_pos/head: [N, flow_window_steps, 2] / [N, flow_window_steps]
    path_pos = pred_traj_10hz[:, start_step_10hz:end_step_10hz]
    path_head = pred_head_10hz[:, start_step_10hz:end_step_10hz]

    if start_step_10hz <= 0:
        if "ctx_sampled_pos" not in tokenized_agent or "ctx_sampled_heading" not in tokenized_agent:
            raise KeyError("tokenized_agent must contain ctx_sampled_pos and ctx_sampled_heading.")
        current_pos = tokenized_agent["ctx_sampled_pos"][:, 1]
        current_head = tokenized_agent["ctx_sampled_heading"][:, 1]
    else:
        current_index = start_step_10hz - 1
        current_pos = pred_traj_10hz[:, current_index].detach()
        current_head = pred_head_10hz[:, current_index].detach()

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
