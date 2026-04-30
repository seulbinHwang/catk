from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def detach_tensor_tree(value: object) -> object:
    """값 안에 들어 있는 tensor의 gradient 연결을 모두 끊습니다.

    Args:
        value: tensor, dict, list, tuple 또는 일반 값입니다.
            tensor의 shape은 어떤 값이든 가능합니다.

    Returns:
        object: tensor이면 gradient 연결만 끊은 tensor를 돌려주고,
        dict, list, tuple이면 같은 구조 안의 tensor만 모두 끊어서 돌려줍니다.

    설명:
        이 함수는 값을 복사하지 않고, tensor의 계산 연결만 끊습니다.
        그래서 shape과 실제 값은 그대로 유지됩니다.
    """
    if torch.is_tensor(value):
        tensor_value: Tensor = value
        # tensor_value: 임의 shape입니다. detach 후에도 shape은 그대로 유지됩니다.
        return tensor_value.detach()
    if isinstance(value, dict):
        return {key: detach_tensor_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [detach_tensor_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(detach_tensor_tree(child) for child in value)
    return value


def detach_training_rollout_state(rollout_state: Dict[str, object]) -> Dict[str, object]:
    """다음 0.5초 block 입력으로 넘어가는 rollout 상태의 gradient를 끊습니다.

    Args:
        rollout_state: closed-loop rollout 안에서 다음 block 입력으로 다시 쓰는 상태 사전입니다.
            예시는 ``pos_window``, ``head_window``, ``exec_pos_history_10hz`` 입니다.
            각 tensor shape은 key마다 다르며, 이 함수는 shape을 바꾸지 않습니다.

    Returns:
        Dict[str, object]: 같은 key 구조를 가진 새 상태 사전입니다.

    설명:
        self-forcing 학습은 모델이 자기 예측 상태를 다음 입력으로 받는 구조를 유지해야 합니다.
        다만 뒤쪽 block의 loss가 앞쪽 block의 실행 행동까지 거꾸로 바꾸면 학습 신호가 불안정해질 수 있습니다.
        이 함수는 다음 입력 상태만 detach합니다. 이미 loss가 직접 보는 committed 출력은 이 함수에 넣지 않습니다.
    """
    return {key: detach_tensor_tree(value) for key, value in rollout_state.items()}
