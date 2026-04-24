from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass(frozen=True)
class ValueTrainConfig:
    """Value training 설정을 한곳에 모읍니다.

    Attributes:
        enabled: value training 분기를 켤지 나타냅니다.
    """

    enabled: bool = False


def _read_config_value(config: Any, key: str, default: Any) -> Any:
    """dict 형태와 속성 형태를 모두 받아 같은 값을 꺼냅니다.

    Args:
        config: bool, dict, DictConfig처럼 키 접근 또는 속성 접근이 가능한 객체입니다.
        key: 읽을 이름입니다.
        default: 값이 없을 때 돌려줄 기본값입니다.

    Returns:
        Any: 읽은 값 또는 기본값입니다.
    """
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, key):
        return getattr(config, key)
    try:
        return config[key]
    except Exception:
        return default


def parse_valuetrain_config(valuetrain: Any) -> ValueTrainConfig:
    """입력 형태가 달라도 같은 value training 설정 객체로 바꿉니다.

    Args:
        valuetrain: bool 또는 설정 객체입니다.

    Returns:
        ValueTrainConfig: 통일된 value training 설정입니다.
    """
    if isinstance(valuetrain, bool):
        return ValueTrainConfig(enabled=bool(valuetrain))
    if valuetrain is None:
        return ValueTrainConfig(enabled=False)

    return ValueTrainConfig(
        enabled=bool(_read_config_value(valuetrain, "enabled", True)),
    )


def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    """모듈 안 모든 파라미터의 학습 여부를 한 번에 바꿉니다.

    Args:
        module: 대상 모듈입니다.
        requires_grad: 학습 여부입니다.

    Returns:
        None
    """
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def set_model_for_valuetraining(model: torch.nn.Module, valuetrain: Any) -> ValueTrainConfig:
    """현재 단계에 맞게 파라미터를 깔끔하게 얼리고 풉니다.

    Args:
        model: Value training을 위한 ``SMARTFlowDecoder`` 인스턴스입니다.
        valuetrain: bool 또는 value training 설정 객체입니다.

    Returns:
        ValueTrainConfig: 실제로 적용된 value training 설정입니다.
    """
    config = parse_valuetrain_config(valuetrain)

    if not config.enabled:
        _set_requires_grad(model, True)
        log.info("Value training mode: all parameters are trainable.")
        return config

    _set_requires_grad(model, False)
    # Flow velocity head만 학습합니다.
    _set_requires_grad(model.velocity_head, True)
    log.info("Value training mode: only flow velocity head is trainable.")
    return config
