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
    flow_reg_lambda: float = 0.0
    reward_huber_beta: float = 0.05
    # ── DICE / IQ-Learn ────────────────────────────────────────────────────
    dice_critic_hidden: int = 256       # critic MLP hidden dim
    dice_action_hidden: int = 128       # action encoder hidden dim
    dice_critic_lr: float = 3e-4        # critic (sidecar) optimizer LR
    dice_critic_updates_per_actor: int = 1  # critic steps per actor step
    dice_reward_enabled: bool = False   # on/off switch for external reward
    dice_reward_weight: float = 1.0     # scale of external reward term
    dice_bc_lambda: float = 0.0         # BC (flow-matching) regularization weight


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
        flow_reg_lambda=float(_read_config_value(finetune, "flow_reg_lambda", 0.0)),
        reward_huber_beta=float(_read_config_value(finetune, "reward_huber_beta", 0.05)),
        dice_critic_hidden=int(_read_config_value(finetune, "dice_critic_hidden", 256)),
        dice_action_hidden=int(_read_config_value(finetune, "dice_action_hidden", 128)),
        dice_critic_lr=float(_read_config_value(finetune, "dice_critic_lr", 3e-4)),
        dice_critic_updates_per_actor=int(_read_config_value(finetune, "dice_critic_updates_per_actor", 1)),
        dice_reward_enabled=bool(_read_config_value(finetune, "dice_reward_enabled", False)),
        dice_reward_weight=float(_read_config_value(finetune, "dice_reward_weight", 1.0)),
        dice_bc_lambda=float(_read_config_value(finetune, "dice_bc_lambda", 0.0)),
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
    # NOTE:
    # - Pretraining 모델(`SMART`)은 flow decoder 구조가 없을 수 있습니다.
    # - 그럼에도 불구하고 residual head를 무조건 참조하면,
    #   `finetune=False`여도 AttributeError로 크래시가 납니다.
    residual_head = None
    try:
        residual_head = model.agent_encoder.flow_decoder.residual_velocity_head
    except AttributeError:
        residual_head = None

    if not config.enabled:
        _set_requires_grad(model, True)
        if residual_head is not None:
            _set_requires_grad(residual_head, False)
            log.info("Pretraining mode: residual_velocity_head is frozen.")
        else:
            log.warning(
                "Pretraining mode: residual_velocity_head not found; skipping freeze."
            )
        return config

    if config.mode not in {
        "adjoint_matching",
        "terminal_cost_final_step",
        "terminal_cost_full_grad",
        "kinematic_proj_ft",    # ODE generate → KinematicProjection → FM target
        "kinematic_reward_ft",  # ODE full-grad → KinematicProjection as reward → reward grad
        "dice_ft",              # DICE / IQ-Learn imitation learning (Chi^2 f-divergence)
    }:
        raise ValueError(f"Unsupported finetune mode: {config.mode}")

    # 전체 모델 freeze 후 flow_decoder만 unfreeze
    _set_requires_grad(model, False)
    try:
        flow_decoder = model.agent_encoder.flow_decoder
    except AttributeError:
        raise AttributeError(
            "Finetuning enabled but flow_decoder not found. "
            "Use the flow-based model (e.g., SMARTFlow) or fix the model config."
        )
    _set_requires_grad(flow_decoder, True)

    # residual_velocity_head는 0으로 초기화 후 freeze (base velocity만 학습)
    if residual_head is not None:
        for p in residual_head.parameters():
            p.data.zero_()
        _set_requires_grad(residual_head, False)
        log.info("Finetuning mode: full flow_decoder is trainable, residual_velocity_head zeroed+frozen.")
    else:
        log.info("Finetuning mode: full flow_decoder is trainable.")
    return config
