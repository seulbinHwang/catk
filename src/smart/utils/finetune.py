from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass(frozen=True)
class FinetuneConfig:
    """OCSC (Open-Closed Self-Consistency) fine-tuning 설정을 한곳에 모읍니다.

    Attributes:
        enabled: fine-tuning 분기를 켤지 나타냅니다.
        mode: 현재 지원하는 fine-tuning 방식 이름입니다. ``"ocsc_ft"`` 만 허용합니다.
        flow_velocity_head_only: True 면 ``HierarchicalFlowDecoder.velocity_head`` 만
            학습 (트렁크·residual 동결).
        bptt_use_adjoint: Flow ODE generate() 안의 model_fn 호출을
            ``torch.utils.checkpoint`` 으로 감쌉니다 (BPTT adjoint method).
        bptt_last_n_solver_steps: Flow ODE solver 의 마지막 N step 에만 gradient 를
            흘립니다 (0 이하 = 모든 step).
        bptt_grad_clip_traj: pred_traj / pred_head_traj 에서 역전파되는 gradient
            L2 norm 의 상한 (0 이하 = 비활성).
        bptt_debug: True → ``_compute_soft_rmm`` 류에서 극값/likelihood WARNING.
        bptt_last_coarse_only: True → ``ocsc_pred_max_steps - 1`` coarse step 을
            no_grad warm-up 으로 처리하고 마지막 1 coarse step 만 gradient 를 흘립니다.
            ⚠ LR 민감 (lr=5e-6 + last_coarse_only=true 가 RMM 폭락 사례 있음).

        ocsc_n_rollouts: G — 시나리오당 closed-loop rollout 수.
        ocsc_n_ol_rollouts: M — open-loop sample 개수. ``-1`` 이면 G 와 동일.
            M > G 또는 nearest_match 면 ``ocsc_use_mmd`` 자동 False.
        ocsc_ol_nearest_match: 각 CL rollout g 에 대해 M 개 OL 중 argmin paired L2 target.
        ocsc_loss_type: "l2" | "smooth_l1" | "l1".
        ocsc_use_mmd: True → MMD². False → rollout 별 paired L2.
        ocsc_use_pretrained_ref: True → frozen pretrained ref decoder 로 OL 생성.
        ocsc_target_max_steps: open-loop target rollout 에서 실행할 coarse step 수.
        ocsc_pred_max_steps: closed-loop prediction rollout 에서 실행할 coarse step 수.
        ocsc_heading_weight: heading channel L2 가중치 (sin/cos).
        ocsc_position_weight: position channel L2 가중치.
        ocsc_rel_disp_weight: 상대변위 (delta-pos) L2 가중치 (paired L2 전용).
        ocsc_eval_hard_rmm: 매 training step 에서 hard RMM 계산 후 로깅.
        ocsc_eval_hard_rmm_interval: hard RMM 평가 주기 (N training step 마다 1 회).
        ocsc_fm_reg_lambda: GT FM regularization 가중치 (0 이면 비활성).
        ocsc_gt_target: True → OL sample 대신 GT 궤적을 target 으로 사용.
        ocsc_gt_resolution: "2hz" (기본, tokenized 2Hz GT) 또는 "10hz" (raw 10Hz).
        ocsc_nearest_include_gt: True → nearest-match candidate pool 에 GT 1 개 추가.
    """

    enabled: bool = False
    mode: str = "ocsc_ft"
    # ── 공통 학습 토글 ─────────────────────────────────────────────────────────
    gradient_clip_val: float = 0.0
    # ── BPTT 토글 (OCSC 에서 ODE solver / closed-loop rollout 제어) ────────────
    flow_velocity_head_only: bool = True
    bptt_use_adjoint: bool = False
    bptt_last_n_solver_steps: int = 0
    bptt_grad_clip_traj: float = 1.0
    bptt_debug: bool = False
    bptt_sequential_rollouts: bool = True
    bptt_warm_coarse_steps: int = 0
    bptt_last_n_coarse_steps: int = 0
    bptt_last_coarse_only: bool = False
    # ── OCSC (Open-Closed Self-Consistency) ───────────────────────────────────
    ocsc_n_rollouts: int = 2
    ocsc_n_ol_rollouts: int = -1
    ocsc_ol_nearest_match: bool = False
    ocsc_anchor_stride: int = 1
    ocsc_loss_type: str = "l2"
    ocsc_use_mmd: bool = True
    ocsc_use_pretrained_ref: bool = False
    ocsc_target_max_steps: int = 4
    ocsc_pred_max_steps: int = 4
    ocsc_heading_weight: float = 0.0
    ocsc_position_weight: float = 1.0
    ocsc_rel_disp_weight: float = 0.0
    ocsc_eval_hard_rmm: bool = True
    ocsc_eval_hard_rmm_interval: int = 1
    ocsc_fm_reg_lambda: float = 0.0
    ocsc_gt_target: bool = False
    ocsc_gt_resolution: str = "2hz"
    ocsc_nearest_include_gt: bool = False


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

    return FinetuneConfig(
        enabled=bool(_read_config_value(finetune, "enabled", True)),
        mode=str(_read_config_value(finetune, "mode", "ocsc_ft")),
        gradient_clip_val=float(_read_config_value(finetune, "gradient_clip_val", 0.0)),
        flow_velocity_head_only=bool(_read_config_value(finetune, "flow_velocity_head_only", True)),
        bptt_use_adjoint=bool(_read_config_value(finetune, "bptt_use_adjoint", False)),
        bptt_last_n_solver_steps=int(_read_config_value(finetune, "bptt_last_n_solver_steps", 0)),
        bptt_grad_clip_traj=float(_read_config_value(finetune, "bptt_grad_clip_traj", 1.0)),
        bptt_debug=bool(_read_config_value(finetune, "bptt_debug", False)),
        bptt_sequential_rollouts=bool(_read_config_value(finetune, "bptt_sequential_rollouts", True)),
        bptt_warm_coarse_steps=int(_read_config_value(finetune, "bptt_warm_coarse_steps", 0)),
        bptt_last_n_coarse_steps=int(_read_config_value(finetune, "bptt_last_n_coarse_steps", 0)),
        bptt_last_coarse_only=bool(_read_config_value(finetune, "bptt_last_coarse_only", False)),
        ocsc_n_rollouts=int(_read_config_value(finetune, "ocsc_n_rollouts", 2)),
        ocsc_n_ol_rollouts=int(_read_config_value(finetune, "ocsc_n_ol_rollouts", -1)),
        ocsc_ol_nearest_match=bool(_read_config_value(finetune, "ocsc_ol_nearest_match", False)),
        ocsc_anchor_stride=int(_read_config_value(finetune, "ocsc_anchor_stride", 1)),
        ocsc_loss_type=str(_read_config_value(finetune, "ocsc_loss_type", "l2")),
        ocsc_use_mmd=bool(_read_config_value(finetune, "ocsc_use_mmd", True)),
        ocsc_use_pretrained_ref=bool(_read_config_value(finetune, "ocsc_use_pretrained_ref", False)),
        ocsc_target_max_steps=int(_read_config_value(finetune, "ocsc_target_max_steps", 4)),
        ocsc_pred_max_steps=int(_read_config_value(finetune, "ocsc_pred_max_steps", 4)),
        ocsc_heading_weight=float(_read_config_value(finetune, "ocsc_heading_weight", 0.0)),
        ocsc_position_weight=float(_read_config_value(finetune, "ocsc_position_weight", 1.0)),
        ocsc_rel_disp_weight=float(_read_config_value(finetune, "ocsc_rel_disp_weight", 0.0)),
        ocsc_eval_hard_rmm=bool(_read_config_value(finetune, "ocsc_eval_hard_rmm", True)),
        ocsc_eval_hard_rmm_interval=int(_read_config_value(finetune, "ocsc_eval_hard_rmm_interval", 1)),
        ocsc_fm_reg_lambda=float(_read_config_value(finetune, "ocsc_fm_reg_lambda", 0.0)),
        ocsc_gt_target=bool(_read_config_value(finetune, "ocsc_gt_target", False)),
        ocsc_gt_resolution=str(_read_config_value(finetune, "ocsc_gt_resolution", "2hz")),
        ocsc_nearest_include_gt=bool(_read_config_value(finetune, "ocsc_nearest_include_gt", False)),
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

    if config.mode != "ocsc_ft":
        raise ValueError(
            f"Unsupported finetune mode: {config.mode}. Only 'ocsc_ft' is supported."
        )

    # 전체 모델 freeze 후 flow_decoder만 unfreeze
    _set_requires_grad(model, False)
    try:
        flow_decoder = model.agent_encoder.flow_decoder
    except AttributeError:
        raise AttributeError(
            "Finetuning enabled but flow_decoder not found. "
            "Use the flow-based model (e.g., SMARTFlow) or fix the model config."
        )

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
