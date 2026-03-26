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
        rollout_steps: suffix-only AM에서 stochastic rollout을 적용할 마지막 step 수입니다.
        rollout_noise_scale: 초기 Gaussian 잡음 크기입니다.
        feasible_weight: terminal feasible cost 가중치입니다.
        smooth_deadzone_epsilon: 정규화 gap dead-zone 크기입니다.
        smooth_deadzone_tau: smooth dead-zone의 매끈한 정도입니다.
    """

    enabled: bool = False
    mode: str = "adjoint_matching"
    rollout_steps: int = 4
    rollout_noise_scale: float = 1.0
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

    return FinetuneConfig(
        enabled=bool(_read_config_value(finetune, "enabled", True)),
        mode=str(_read_config_value(finetune, "mode", "adjoint_matching")),
        rollout_steps=int(_read_config_value(finetune, "rollout_steps", 4)),
        rollout_noise_scale=float(_read_config_value(finetune, "rollout_noise_scale", 1.0)),
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
    """현재 단계에 맞게 teacher/student와 학습 파라미터를 설정합니다.

    Args:
        model: ``SMARTFlowDecoder`` 인스턴스입니다.
        finetune: bool 또는 fine-tuning 설정 객체입니다.

    Returns:
        FinetuneConfig: 실제로 적용된 fine-tuning 설정입니다.
    """
    config = parse_finetune_config(finetune)
    flow_agent_decoder = model.agent_encoder
    flow_decoder = flow_agent_decoder.flow_decoder
    total_solver_steps = int(flow_agent_decoder.flow_ode.solver_steps)

    if config.rollout_steps < 1:
        raise ValueError("finetune.rollout_steps must be at least 1.")
    if config.rollout_steps > total_solver_steps:
        raise ValueError(
            "finetune.rollout_steps must be smaller than or equal to flow_solver_steps. "
            f"Got rollout_steps={config.rollout_steps}, flow_solver_steps={total_solver_steps}."
        )

    if not config.enabled:
        _set_requires_grad(model, True)
        if hasattr(flow_agent_decoder, "disable_teacher_student_hybrid"):
            flow_agent_decoder.disable_teacher_student_hybrid()
        log.info("Pretraining mode: all base parameters are trainable.")
        return config

    if config.mode != "adjoint_matching":
        raise ValueError(f"Unsupported finetune mode: {config.mode}")

    teacher_prefix_steps = total_solver_steps - config.rollout_steps
    if not hasattr(flow_agent_decoder, "enable_teacher_student_hybrid"):
        raise AttributeError(
            "SMARTFlowAgentDecoder must provide enable_teacher_student_hybrid() for fine-tuning."
        )

    flow_agent_decoder.enable_teacher_student_hybrid(prefix_steps=teacher_prefix_steps)
    flow_agent_decoder.sync_teacher_flow_decoder_from_student()

    _set_requires_grad(model, False)
    _set_requires_grad(flow_decoder.step_refiner, True)
    _set_requires_grad(flow_decoder.velocity_head, True)

    log.info(
        "Finetuning mode: only flow_decoder.step_refiner and flow_decoder.velocity_head are trainable. "
        f"Teacher prefix steps={teacher_prefix_steps}, suffix AM steps={config.rollout_steps}."
    )
    return config
