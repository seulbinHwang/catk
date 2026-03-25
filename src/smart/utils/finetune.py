from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass(frozen=True)
class FinetuneConfig:
    """Adjoint Matching fine-tuning 설정을 한곳에 모읍니다.

    Attributes:
        enabled: fine-tuning 분기를 켤지 나타냅니다.
        mode: 현재 지원하는 fine-tuning 방식 이름입니다.
        rollout_steps: 학습 rollout step 수입니다.
        rollout_noise_scale:
            초기 무작위 상태에 곱하는 배율입니다.
            ``rollout_start_tau`` 를 따로 주지 않으면 예전과 같은 절대 크기로 동작합니다.
            ``rollout_start_tau`` 를 주고 ``rollout_init_noise_scale`` 을 비워 두면,
            학습 시작 시각의 ``sigma_t`` 에 이 값을 한 번 더 곱합니다.
        rollout_start_tau:
            학습용 AM rollout이 시작할 진행 시각입니다.
            ``None`` 이면 기존처럼 ``flow_ode.eps`` 에서 시작합니다.
        rollout_init_noise_scale:
            첫 rollout 상태의 무작위 크기를 직접 지정합니다.
            ``None`` 이면 ``rollout_start_tau`` 가 없을 때는 예전 scale을 그대로 쓰고,
            ``rollout_start_tau`` 가 있으면
            ``sigma_t(rollout_start_tau) * rollout_noise_scale`` 을 자동으로 사용합니다.
        feasible_weight: terminal feasible cost 가중치입니다.
        smooth_deadzone_epsilon: 정규화 gap dead-zone 크기입니다.
        smooth_deadzone_tau: smooth dead-zone의 매끈한 정도입니다.
    """

    enabled: bool = False
    mode: str = "adjoint_matching"
    rollout_steps: int = 4
    rollout_noise_scale: float = 1.0
    rollout_start_tau: float | None = None
    rollout_init_noise_scale: float | None = None
    feasible_weight: float = 1.0
    smooth_deadzone_epsilon: tuple[float, float, float] = (0.01, 0.01, 0.01)
    smooth_deadzone_tau: float = 0.002


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


def _read_optional_float_config_value(config: Any, key: str) -> float | None:
    """옵션으로 들어오는 실수 값을 읽어 옵니다.

    Args:
        config: 키 접근 또는 속성 접근이 가능한 설정 객체입니다.
        key: 읽을 이름입니다.

    Returns:
        float | None: 값이 없으면 ``None`` 이고, 있으면 ``float`` 로 바꾼 값입니다.
    """
    value = _read_config_value(config, key, None)
    if value is None:
        return None
    return float(value)


def parse_finetune_config(finetune: Any) -> FinetuneConfig:
    """입력 형태가 달라도 같은 fine-tuning 설정 객체로 바꿉니다.

    Args:
        finetune: bool 또는 설정 객체입니다.

    Returns:
        FinetuneConfig: 통일된 fine-tuning 설정입니다.
    """
    if isinstance(finetune, bool):
        return FinetuneConfig(enabled=bool(finetune))
    if finetune is None:
        return FinetuneConfig(enabled=False)

    epsilon = _read_config_value(finetune, "smooth_deadzone_epsilon", (0.01, 0.01, 0.01))
    epsilon_tuple = tuple(float(v) for v in epsilon)
    if len(epsilon_tuple) != 3:
        raise ValueError(
            "smooth_deadzone_epsilon must contain exactly 3 values for [vx, vy, omega]."
        )

    rollout_start_tau = _read_optional_float_config_value(finetune, "rollout_start_tau")
    rollout_init_noise_scale = _read_optional_float_config_value(
        finetune,
        "rollout_init_noise_scale",
    )

    return FinetuneConfig(
        enabled=bool(_read_config_value(finetune, "enabled", True)),
        mode=str(_read_config_value(finetune, "mode", "adjoint_matching")),
        rollout_steps=int(_read_config_value(finetune, "rollout_steps", 4)),
        rollout_noise_scale=float(_read_config_value(finetune, "rollout_noise_scale", 1.0)),
        rollout_start_tau=rollout_start_tau,
        rollout_init_noise_scale=rollout_init_noise_scale,
        feasible_weight=float(_read_config_value(finetune, "feasible_weight", 1.0)),
        smooth_deadzone_epsilon=epsilon_tuple,
        smooth_deadzone_tau=float(_read_config_value(finetune, "smooth_deadzone_tau", 0.002)),
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


def set_model_for_finetuning(model: torch.nn.Module, finetune: Any) -> FinetuneConfig:
    """현재 단계에 맞게 파라미터를 깔끔하게 얼리고 풉니다.

    Args:
        model: ``SMARTFlowDecoder`` 인스턴스입니다.
        finetune: bool 또는 fine-tuning 설정 객체입니다.

    Returns:
        FinetuneConfig: 실제로 적용된 fine-tuning 설정입니다.
    """
    config = parse_finetune_config(finetune)
    residual_head = model.agent_encoder.flow_decoder.residual_velocity_head

    if not config.enabled:
        _set_requires_grad(model, True)
        _set_requires_grad(residual_head, False)
        log.info("Pretraining mode: residual_velocity_head is frozen.")
        return config

    if config.mode != "adjoint_matching":
        raise ValueError(f"Unsupported finetune mode: {config.mode}")

    _set_requires_grad(model, False)
    _set_requires_grad(residual_head, True)
    log.info("Finetuning mode: only residual_velocity_head is trainable.")
    return config
