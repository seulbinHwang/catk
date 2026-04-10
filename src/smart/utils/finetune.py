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
    dice_critic_hidden: int = 256
    dice_action_hidden: int = 128
    dice_critic_lr: float = 3e-4
    dice_critic_updates_per_actor: int = 1
    dice_reward_enabled: bool = False
    dice_reward_weight: float = 1.0
    dice_bc_lambda: float = 0.0
    # ── Flow-DPO ──────────────────────────────────────────────────────────────
    dpo_beta: float = 0.1
    dpo_n_samples: int = 8
    dpo_use_ref_model: bool = True
    dpo_bc_lambda: float = 0.0
    # ── Flow-EPG ──────────────────────────────────────────────────────────────
    epg_n_rollouts: int = 4            # G: number of rollouts per scenario
    epg_beta: float = 0.1             # KL regularisation weight β
    epg_n_samples: int = 8            # MC samples for ELBO log-prob estimation
    epg_use_ref_model: bool = True    # True → use frozen pretrained as reference
    epg_bc_lambda: float = 0.0        # optional BC regularization weight
    epg_ppo_epochs: int = 1           # K gradient steps per RMM evaluation (PPO-style)
    epg_head_only: bool = False       # True → only train residual_velocity_head (frozen trunk)
    #: G==1일 때만: RMM에서 뺄 baseline(스칼라). None이면 그룹 평균(=단일 rollout이면 R)을 쓰고
    #: 표준편차는 1로 두어 NaN을 피함 → advantage는 0. float를 주면 (R−baseline)/1 로 정규화.
    epg_single_rollout_baseline: float | None = None
    # ── Flow-RWR ──────────────────────────────────────────────────────────────
    rwr_n_rollouts: int = 4          # G: rollouts per scenario
    rwr_beta: float = 0.1            # temperature β for exp(R/β) weighting
    rwr_n_samples: int = 8           # MC samples for FM log-prob (ELBO)
    rwr_anchor_discount: float = 1.0  # γ: temporal discount per anchor step (1=uniform)
    rwr_head_only: bool = False      # True → only residual_velocity_head trained
    # ── RMM-BPTT ───────────────────────────────────────────────────────────────
    bptt_n_rollouts: int = 2
    rmm_bptt_use_ref_model: bool = False
    #: True → ``HierarchicalFlowDecoder.velocity_head`` 만 학습 (트렁크·residual 동결).
    #: ``flow_epg_ft``/``flow_rwr_ft`` 의 ``*_head_only``(residual 전용)보다 우선하지 않음.
    flow_velocity_head_only: bool = True


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

    epg_srb_raw = _read_config_value(finetune, "epg_single_rollout_baseline", None)
    epg_single_rollout_baseline = float(epg_srb_raw) if epg_srb_raw is not None else None

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
        dpo_beta=float(_read_config_value(finetune, "dpo_beta", 0.1)),
        dpo_n_samples=int(_read_config_value(finetune, "dpo_n_samples", 8)),
        dpo_use_ref_model=bool(_read_config_value(finetune, "dpo_use_ref_model", True)),
        dpo_bc_lambda=float(_read_config_value(finetune, "dpo_bc_lambda", 0.0)),
        epg_n_rollouts=int(_read_config_value(finetune, "epg_n_rollouts", 4)),
        epg_beta=float(_read_config_value(finetune, "epg_beta", 0.1)),
        epg_n_samples=int(_read_config_value(finetune, "epg_n_samples", 8)),
        epg_use_ref_model=bool(_read_config_value(finetune, "epg_use_ref_model", True)),
        epg_bc_lambda=float(_read_config_value(finetune, "epg_bc_lambda", 0.0)),
        epg_ppo_epochs=int(_read_config_value(finetune, "epg_ppo_epochs", 1)),
        epg_head_only=bool(_read_config_value(finetune, "epg_head_only", False)),
        epg_single_rollout_baseline=epg_single_rollout_baseline,
        rwr_n_rollouts=int(_read_config_value(finetune, "rwr_n_rollouts", 4)),
        rwr_beta=float(_read_config_value(finetune, "rwr_beta", 0.1)),
        rwr_n_samples=int(_read_config_value(finetune, "rwr_n_samples", 8)),
        rwr_anchor_discount=float(_read_config_value(finetune, "rwr_anchor_discount", 1.0)),
        rwr_head_only=bool(_read_config_value(finetune, "rwr_head_only", False)),
        bptt_n_rollouts=int(_read_config_value(finetune, "bptt_n_rollouts", 2)),
        rmm_bptt_use_ref_model=bool(_read_config_value(finetune, "rmm_bptt_use_ref_model", False)),
        flow_velocity_head_only=bool(_read_config_value(finetune, "flow_velocity_head_only", True)),
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
        "flow_epg_ft",          # Flow-EPG: Exact Policy Gradient with ELBO + RMM reward
        "flow_rwr_ft",          # Flow-RWR: Reward-Weighted Regression with GPU RMM
        "rmm_bptt_ft",          # RMM-BPTT: differentiable soft RMM through closed-loop rollout
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

    # ── head_only: residual_velocity_head만 학습 (트렁크 완전 동결) ──────────
    is_epg_head_only = (config.mode == "flow_epg_ft" and config.epg_head_only)
    is_rwr_head_only = (config.mode == "flow_rwr_ft" and config.rwr_head_only)
    if is_epg_head_only or is_rwr_head_only:
        if residual_head is None:
            raise AttributeError(
                "epg_head_only=True requires residual_velocity_head in flow_decoder."
            )
        # residual head only: zero-init + unfreeze only that head
        for p in residual_head.parameters():
            p.data.zero_()
        _set_requires_grad(residual_head, True)
        mode_tag = "EPG" if is_epg_head_only else "RWR"
        log.info(
            f"{mode_tag} head-only mode: flow_decoder trunk frozen, "
            "only residual_velocity_head is trainable (zero-initialized)."
        )
        return config

    # ── velocity_head만 학습 (트렁크·residual 동결) ─────────────────────────
    if config.flow_velocity_head_only:
        try:
            velocity_head = flow_decoder.velocity_head
        except AttributeError as exc:
            raise AttributeError(
                "flow_velocity_head_only=True requires velocity_head on flow_decoder "
                "(HierarchicalFlowDecoder)."
            ) from exc
        _set_requires_grad(flow_decoder, False)
        _set_requires_grad(velocity_head, True)
        if residual_head is not None:
            for p in residual_head.parameters():
                p.data.zero_()
            _set_requires_grad(residual_head, False)
            log.info(
                "Finetuning mode: only velocity_head trainable; flow_decoder trunk frozen; "
                "residual_velocity_head zeroed+frozen."
            )
        else:
            log.info(
                "Finetuning mode: only velocity_head trainable; flow_decoder trunk frozen "
                "(no residual_velocity_head)."
            )
        return config

    # ── 기본: 전체 flow_decoder 학습 ─────────────────────────────────────────
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
