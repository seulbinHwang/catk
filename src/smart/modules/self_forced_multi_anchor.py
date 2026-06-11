"""Self-forced 학습을 여러 GT anchor 시작점으로 확장하는 유틸입니다.

pretraining open-loop 학습은 한 장면에서 16개 anchor(0.5초 간격 GT 시작점)를
한 번의 forward로 병렬 학습합니다. 이 모듈은 self-forced rollout에도 같은
구조를 적용하기 위해 (scene × anchor) 복제 입력을 만들어 rollout 한 번으로
모든 시작점을 병렬 실행하고, 그 결과를 pretraining과 같은 anchor-major
packing으로 DMD/critic 경로에 넘길 수 있게 합니다.

설계 규칙:
    - anchor k는 10Hz raw step ``shift*(k+2)`` 을 현재 시점으로 쓰는 시작점입니다.
      ctx token slot으로는 ``1+k`` 입니다(anchor0=slot1 규칙의 일반화).
    - 복제 텐서는 항상 anchor-major(anchor 0 전체 agent, anchor s 전체 agent, ...)
      순서로 배치합니다. 이는 decoder ``_pack_anchor_hidden`` 의 packing 순서와
      같아서 committed rollout 행과 anchor hidden 행이 1:1로 맞습니다.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor

# rollout cache와 closed-loop rollout이 읽는 agent 축 불변 필드입니다.
_ANCHOR_INVARIANT_AGENT_KEYS = (
    "type",
    "shape",
    "ego_mask",
    "token_agent_shape",
    "gt_pos_raw",
    "gt_head_raw",
    "gt_valid_raw",
)

# agent 축이 없어 복제 없이 그대로 공유하는 필드입니다.
_SHARED_KEYS = (
    "trajectory_token_veh",
    "trajectory_token_ped",
    "trajectory_token_cyc",
    "token_bank_all_veh",
    "token_bank_all_ped",
    "token_bank_all_cyc",
)

# anchor offset만큼 시간 축을 shift해야 하는 2Hz 토큰 window 필드입니다.
_TOKEN_WINDOW_KEYS = ("gt_idx", "gt_pos", "gt_heading")


def select_self_forced_anchor_offsets(
    anchor_stride: int,
    num_anchor: int,
    num_raw_steps: int,
    flow_window_steps: int,
    shift: int,
) -> List[int]:
    """stride 규칙으로 self-forced 시작 anchor offset 목록을 만듭니다.

    Args:
        anchor_stride: anchor 간격입니다. 0 이하이면 기존 동작(anchor 0 단독)입니다.
            예를 들어 4면 0.5초 단위 4 step(2초)마다 시작점을 둡니다.
        num_anchor: token processor가 만드는 전체 anchor 수입니다.
        num_raw_steps: GT 10Hz step 수입니다.
        flow_window_steps: self-forced rollout이 채울 flow window 길이(10Hz)입니다.
        shift: coarse token 간 10Hz step 수입니다.

    Returns:
        List[int]: 사용할 anchor offset 목록입니다. 항상 0(기존 시작점)을 포함하고,
        GT 미래 window가 전부 들어가는 anchor까지만 선택합니다.
    """
    if anchor_stride <= 0:
        return [0]
    # anchor k의 미래 window는 raw step ``shift*(k+2)+1 .. shift*(k+2)+flow_window``
    # 까지 GT가 있어야 합니다(flow_eval_mask의 full-window 규칙과 동일).
    max_offset_by_future = (num_raw_steps - 1 - flow_window_steps) // shift - 2
    max_offset = min(num_anchor - 1, max_offset_by_future)
    if max_offset < 0:
        return [0]
    return list(range(0, max_offset + 1, int(anchor_stride)))


def build_multi_anchor_mask(
    flow_eval_mask: Tensor,
    anchor_offsets: List[int],
) -> Tensor:
    """선택한 anchor offset들의 유효 agent mask를 모읍니다.

    Args:
        flow_eval_mask: 평가 모드 anchor 유효 마스크입니다.
            shape은 ``[n_agent, n_anchor]`` 입니다.
        anchor_offsets: 사용할 anchor offset 목록입니다.

    Returns:
        Tensor: shape ``[n_agent, n_selected]`` 의 bool mask입니다.
    """
    if flow_eval_mask.dim() != 2:
        raise ValueError(
            f"flow_eval_mask must have shape [n_agent, n_anchor], got {tuple(flow_eval_mask.shape)}."
        )
    if max(anchor_offsets) >= flow_eval_mask.shape[1]:
        raise ValueError(
            "anchor_offsets exceed flow_eval_mask anchor count: "
            f"max offset {max(anchor_offsets)}, n_anchor {flow_eval_mask.shape[1]}."
        )
    return flow_eval_mask[:, list(anchor_offsets)].bool()


def build_anchor_current_pose(
    tokenized_agent: Dict[str, Tensor],
    anchor_offsets: List[int],
) -> tuple[Tensor, Tensor]:
    """anchor별 frame 원점(현재 pose)을 ctx token에서 읽습니다.

    Args:
        tokenized_agent: 평가 모드 토큰 사전입니다. ``ctx_sampled_pos`` 와
            ``ctx_sampled_heading`` 이 있어야 합니다.
        anchor_offsets: 사용할 anchor offset 목록입니다.

    Returns:
        tuple[Tensor, Tensor]: anchor별 현재 중심점과 방향입니다.
            shape은 ``[n_agent, n_selected, 2]`` 와 ``[n_agent, n_selected]`` 입니다.
    """
    slots = [1 + int(offset) for offset in anchor_offsets]
    if max(slots) >= tokenized_agent["ctx_sampled_pos"].shape[1]:
        raise ValueError(
            "anchor offset ctx slot exceeds ctx token count: "
            f"slot {max(slots)}, ctx tokens {tokenized_agent['ctx_sampled_pos'].shape[1]}."
        )
    return (
        tokenized_agent["ctx_sampled_pos"][:, slots],
        tokenized_agent["ctx_sampled_heading"][:, slots],
    )


def pack_anchor_invariant(values: Tensor, anchor_mask: Tensor) -> Tensor:
    """anchor 축이 없는 agent 값을 anchor-major로 packing합니다.

    Args:
        values: agent 값입니다. shape은 ``[n_agent, ...]`` 입니다.
        anchor_mask: 유효 (agent, anchor) 마스크입니다.
            shape은 ``[n_agent, n_selected]`` 입니다.

    Returns:
        Tensor: shape ``[n_valid, ...]`` 의 packed 값입니다.
            ``_pack_anchor_hidden`` 과 같은 anchor-major 순서입니다.
    """
    num_selected = anchor_mask.shape[1]
    expanded = values.unsqueeze(0).expand(num_selected, *values.shape)
    return expanded[anchor_mask.transpose(0, 1)]


def pack_anchor_variant(values: Tensor, anchor_mask: Tensor) -> Tensor:
    """anchor 축이 있는 agent 값을 anchor-major로 packing합니다.

    Args:
        values: anchor별 agent 값입니다. shape은 ``[n_agent, n_selected, ...]`` 입니다.
        anchor_mask: 유효 (agent, anchor) 마스크입니다.
            shape은 ``[n_agent, n_selected]`` 입니다.

    Returns:
        Tensor: shape ``[n_valid, ...]`` 의 packed 값입니다.
    """
    return values.transpose(0, 1)[anchor_mask.transpose(0, 1)]


def pack_replicated_rows(values: Tensor, anchor_mask: Tensor) -> Tensor:
    """anchor-major 복제 행 텐서에서 유효 (agent, anchor) 행만 packing합니다.

    Args:
        values: 복제 rollout 결과처럼 anchor-major로 펼쳐진 텐서입니다.
            shape은 ``[n_selected * n_agent, ...]`` 입니다.
        anchor_mask: 유효 (agent, anchor) 마스크입니다.
            shape은 ``[n_agent, n_selected]`` 입니다.

    Returns:
        Tensor: shape ``[n_valid, ...]`` 의 packed 값입니다.
    """
    num_agent, num_selected = anchor_mask.shape
    if values.shape[0] != num_agent * num_selected:
        raise ValueError(
            "values first dim must be n_selected * n_agent: "
            f"got {values.shape[0]}, expected {num_agent * num_selected}."
        )
    return values[anchor_mask.transpose(0, 1).reshape(-1)]


def _flatten_anchor_major(values: Tensor) -> Tensor:
    """``[n_agent, n_selected, ...]`` 값을 ``[n_selected * n_agent, ...]`` 로 폅니다."""
    return values.transpose(0, 1).reshape(
        values.shape[0] * values.shape[1],
        *values.shape[2:],
    )


def replicate_tokenized_agent_for_anchors(
    tokenized_agent: Dict[str, Tensor],
    anchor_offsets: List[int],
) -> Dict[str, Tensor]:
    """선택한 anchor마다 장면을 복제해 단일 rollout 입력을 만듭니다.

    복제 dict는 rollout cache(``_prepare_rollout_cache_impl``)와 closed-loop
    rollout(``_rollout_from_cache_impl``)이 읽는 필드만 담습니다. 복제 a의
    토큰 window는 anchor offset만큼 shift되어, 기존 anchor0 코드가 그대로
    "anchor k에서 시작하는 장면"으로 동작합니다.

    Args:
        tokenized_agent: 평가 모드 토큰 사전입니다. ``sf_anchor_*`` 필드가
            있어야 합니다(``FlowTokenProcessor`` eval 분기가 만듭니다).
        anchor_offsets: 사용할 anchor offset 목록입니다.

    Returns:
        Dict[str, Tensor]: anchor-major로 복제된 토큰 사전입니다.
            ``anchor_offsets`` 가 ``[0]`` 이면 입력을 그대로 돌려줍니다.
    """
    num_selected = len(anchor_offsets)
    if num_selected == 1 and int(anchor_offsets[0]) == 0:
        return tokenized_agent

    batch = tokenized_agent["batch"]
    device = batch.device
    num_agent = int(batch.shape[0])
    num_graphs = int(tokenized_agent["num_graphs"])
    offsets = torch.tensor(list(anchor_offsets), device=device, dtype=torch.long)

    replicated: Dict[str, Tensor] = {
        "num_graphs": num_graphs * num_selected,
        "batch": (
            batch.unsqueeze(0)
            + (torch.arange(num_selected, device=device, dtype=batch.dtype) * num_graphs).view(-1, 1)
        ).reshape(-1),
    }
    for key in _SHARED_KEYS:
        if key in tokenized_agent:
            replicated[key] = tokenized_agent[key]
    for key in _ANCHOR_INVARIANT_AGENT_KEYS:
        if key in tokenized_agent:
            value = tokenized_agent[key]
            replicated[key] = value.repeat(num_selected, *([1] * (value.dim() - 1)))

    # 2Hz 토큰 window shift: 복제 a의 토큰 t는 원본 토큰 t + offset_a 입니다.
    # GT 범위를 벗어난 자리는 마지막 토큰을 반복하되 valid를 False로 둡니다.
    num_token = tokenized_agent["gt_pos"].shape[1]
    token_arange = torch.arange(num_token, device=device, dtype=torch.long)
    shifted_token_idx = offsets.view(-1, 1) + token_arange.view(1, -1)
    in_range = shifted_token_idx <= (num_token - 1)
    shifted_token_idx = shifted_token_idx.clamp(max=num_token - 1)
    for key in _TOKEN_WINDOW_KEYS:
        replicated[key] = _flatten_anchor_major(tokenized_agent[key][:, shifted_token_idx])
    shifted_valid = tokenized_agent["valid_mask"][:, shifted_token_idx] & in_range.view(
        1, num_selected, num_token
    )
    replicated["valid_mask"] = _flatten_anchor_major(shifted_valid)

    # anchor별 10Hz fine 초기 상태와 z를 anchor0 전용 필드 이름으로 복제합니다.
    fine_pos = tokenized_agent["sf_anchor_fine_pos_history"][:, offsets]
    fine_head = tokenized_agent["sf_anchor_fine_head_history"][:, offsets]
    fine_valid = tokenized_agent["sf_anchor_fine_valid_history"][:, offsets]
    replicated["rollout_init_fine_pos_history"] = _flatten_anchor_major(fine_pos)
    replicated["rollout_init_fine_head_history"] = _flatten_anchor_major(fine_head)
    replicated["rollout_init_fine_valid_history"] = _flatten_anchor_major(fine_valid)
    replicated["rollout_init_fine_pos_pair"] = _flatten_anchor_major(fine_pos[:, :, -2:])
    replicated["rollout_init_fine_head_pair"] = _flatten_anchor_major(fine_head[:, :, -2:])
    replicated["rollout_init_fine_valid_pair"] = _flatten_anchor_major(fine_valid[:, :, -2:])
    replicated["gt_z_raw"] = _flatten_anchor_major(
        tokenized_agent["sf_anchor_z"][:, offsets].unsqueeze(-1)
    ).squeeze(-1)
    return replicated


def replicate_map_feature_for_anchors(
    map_feature: Dict[str, Tensor],
    num_graphs: int,
    num_selected_anchors: int,
) -> Dict[str, Tensor]:
    """인코딩된 map feature를 anchor 복제 수만큼 반복합니다.

    map 인코딩은 한 번만 수행하고, 텐서만 반복해 (scene × anchor) 복제
    rollout이 같은 지도를 공유하게 합니다. ``batch`` 는 anchor 복제마다
    ``num_graphs`` 씩 offset되어 agent ``batch`` 와 graph id가 맞습니다.

    Args:
        map_feature: ``encode_map`` 출력입니다.
        num_graphs: 원본 장면 수입니다.
        num_selected_anchors: anchor 복제 수입니다.

    Returns:
        Dict[str, Tensor]: 복제된 map feature입니다. 복제 수가 1이면 입력을
        그대로 돌려줍니다.
    """
    if num_selected_anchors == 1:
        return map_feature
    batch = map_feature["batch"]
    replicated = {
        key: value.repeat(num_selected_anchors, *([1] * (value.dim() - 1)))
        for key, value in map_feature.items()
        if key != "batch"
    }
    replicated["batch"] = (
        batch.unsqueeze(0)
        + (
            torch.arange(num_selected_anchors, device=batch.device, dtype=batch.dtype)
            * int(num_graphs)
        ).view(-1, 1)
    ).reshape(-1)
    return replicated
