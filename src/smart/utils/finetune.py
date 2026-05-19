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
        mode: fine-tuning 방식 이름입니다. ``"ocsc_ft"`` (Open-Closed Self-Consistency)
            또는 ``"road_ft"`` (RoaD CL-SFT baseline) 를 허용합니다.
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
        ocsc_loss_type: "l2" | "smooth_l1" | "l1" | "pwil".
            "pwil" 은 PWIL coupling (Wasserstein-1 upper bound) — ocsc_pwil_* 토글 사용.
        ocsc_use_mmd: True → MMD². False → rollout 별 paired L2.
        ocsc_pwil_coupling: PWIL 모드의 coupling 알고리즘.
            "hungarian" (G=M 필수, exact W_1) | "greedy" (PWIL 원논문, G≠M 허용) | "uniform".
        ocsc_pwil_use_exp_reward: True → per-CL transport cost 에 ``α(1 - exp(-β c))`` 변환.
            False → raw transport cost ``<d, γ>`` 직접 minimize.
        ocsc_pwil_alpha: bounded reward scale (use_exp_reward=True 전용; loss ∈ [0, α]).
        ocsc_pwil_beta: bounded reward decay 계수; ``β · typical(c) ≈ 1`` 으로 튜닝.
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
        ocsc_ol_resolution: OL target / CL prediction 시간 해상도.
            "10hz" (기본, native fine 20 step) 또는 "2hz" (fine→2Hz coarse 다운샘플).
            OL 분기 (ocsc_gt_target=False) 전용. GT 분기는 ocsc_gt_resolution 사용.
        ocsc_nearest_include_gt: True → nearest-match candidate pool 에 GT 1 개 추가.
    """

    enabled: bool = False
    mode: str = "ocsc_ft"
    # ── 공통 학습 토글 ─────────────────────────────────────────────────────────
    gradient_clip_val: float = 0.0
    # ── BPTT 토글 (OCSC 에서 ODE solver / closed-loop rollout 제어) ────────────
    flow_velocity_head_only: bool = True
    # 학습 대상 module 선택 (flow_velocity_head_only 보다 우선 적용).
    # "default": 기존 flow_velocity_head_only 토글 따름 (backward-compat).
    # "velocity_head": velocity_head 만 학습 (flow_velocity_head_only=true 와 동일).
    # "step_refiner_and_velocity_head": step_refiner + velocity_head 만 학습.
    # "chunk_mixers_and_velocity_head": chunk_mixers + velocity_head 만 학습.
    # "full": 전체 flow_decoder 학습 (flow_velocity_head_only=false 와 동일).
    flow_ft_target: str = "default"
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
    # ── PWIL (Primal Wasserstein Imitation Learning) coupling 토글 ────────────
    #: PWIL 모드 (loss_type="pwil") 의 coupling 알고리즘.
    #: "hungarian": scipy linear_sum_assignment, G=M 필수, exact W_1.
    #: "greedy": PWIL 원논문 faithful, G≠M 허용, nearest-first mass transport.
    #: "uniform": γ=1/(GM), ablation baseline (가장 느슨한 bound).
    ocsc_pwil_coupling: str = "hungarian"
    #: True → per-CL transport cost c_i 에 ``α(1 - exp(-β c_i))`` 변환 (bounded reward).
    #: False → raw transport cost ``<d, γ>`` 직접 minimize (W_1 upper bound 그대로).
    ocsc_pwil_use_exp_reward: bool = True
    #: bounded reward scale; use_exp_reward=True 일 때 loss ∈ [0, α].
    ocsc_pwil_alpha: float = 1.0
    #: bounded reward decay 계수; β · typical(c) ≈ 1 로 튜닝 (saturation 영역 회피).
    ocsc_pwil_beta: float = 5.0
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
    ocsc_ol_resolution: str = "10hz"
    ocsc_nearest_include_gt: bool = False
    #: True → OCSC active_mask 를 main training 과 동일하게 강화 (current_valid AND
    #: future pred_max_steps_raw*shift fine step 모두 valid). 부분 invalid agent
    #: (anchor 시점 valid 인데 future 일부 invalid) 는 OCSC anchor 에서 제외 →
    #: model 이 학습 안 한 영역의 hallucination self-consistency 학습 방지.
    ocsc_strict_active_mask: bool = False
    #: OL target 생성에 쓰는 ref_flow_decoder 갱신 방식.
    #: "frozen" (기본): 학습 시작 시점 pretrained 가중치로 고정.
    #: "periodic": ocsc_ref_refresh_interval step 마다 현재 flow_decoder 로 hard copy.
    #: "ema": 매 step ref = decay·ref + (1-decay)·current (mean-teacher 방식).
    ocsc_ref_refresh_mode: str = "frozen"
    #: periodic 모드의 갱신 주기 (training step). 0 이하면 갱신 안 함.
    ocsc_ref_refresh_interval: int = 0
    #: ema 모드의 decay 계수. 1 에 가까울수록 ref 가 느리게 따라옴.
    ocsc_ref_ema_decay: float = 0.999
    # ── RoaD (Rollouts as Demonstrations) CL-SFT baseline ─────────────────────
    # RoaD 논문 (NVIDIA, 2025) 의 closed-loop SFT 방법론을 OCSC 비교용 baseline 으로
    # 구현한 분기 (mode="road_ft").  알고리즘:
    #   1. Expert-guided closed-loop rollout: 매 coarse step 마다 정책에서 K 개 후보
    #      trajectory 를 i.i.d. 샘플 → GT continuation 에 weighted step-wise L2 (Eq.6)
    #      가 최소인 후보를 선택해 commit (Sample-K, Eq.4-5).
    #   2. BC loss: 선택된 후보를 clean target 으로 flow-matching loss.  RoaD loss 는
    #      -log π(a_t|o_<t) 이므로 conditioning (anchor_hidden) 은 detach, BPTT 없음.
    #: K — Sample-K 후보 개수 (논문 기본값 64).
    road_sample_k: int = 64
    #: 시나리오당 expert-guided rollout 수 (RoaD SFT dataset 의 N_roll).
    road_n_rollouts: int = 1
    #: expert-guided rollout 의 coarse step 수 (16 = 8초 full episode, WOSAC 기준).
    road_pred_max_steps: int = 16
    #: 후보 샘플링 noise scale (논문의 sampling temperature 0.8).
    road_temperature: float = 0.8
    #: d^g (Eq.6) position channel 가중치.
    road_position_weight: float = 1.0
    #: d^g (Eq.6) heading channel (cos/sin) 가중치.
    road_heading_weight: float = 0.1
    #: d^g 비교 horizon H_t — 후보당 비교할 fine(10Hz) step 수 (논문: first 20 = 2초).
    road_comparison_horizon: int = 20
    #: True → BC term 에 future GT 가 horizon 전체 valid 인 agent 만 포함.
    road_strict_active_mask: bool = True
    #: 매 training step free-running closed-loop hard RMM 모니터링.
    road_eval_hard_rmm: bool = False
    road_eval_hard_rmm_interval: int = 10
    # ── Self-Forcing DMD (Distribution Matching Distillation) ──────────────────
    # mode="self_forcing_dmd" 분기.  Generator (main flow_decoder) + frozen real_score
    # (ref_flow_decoder, OCSC ref 패턴 재사용) + trainable fake_score (별도 deepcopy).
    # 매 step DMD synthetic-gradient 와 fake_score FM loss 를 alternating 으로 update.
    #
    # DMD gradient (spec):
    #   ∇_θ J = E[(s_real(x_τ) − (1/β) · s_fake(x_τ)) · ∇_θ x_τ]
    # β=1 vanilla / β<1 diversity↑ / β>1 sharpening↑.
    #: entropy knob.  1.0 = vanilla DMD, <1 = diversity 강조, >1 = realism 강조.
    dmd_beta: float = 1.0
    #: 시나리오당 closed-loop rollout 수 (G).  Generator sample 다양화 용도.
    dmd_n_rollouts: int = 1
    #: closed-loop rollout 의 coarse(2Hz) step 수.
    dmd_pred_max_steps: int = 2
    #: True → frozen pretrained ref_flow_decoder 를 real_score teacher 로 사용.
    #: False → BC-style fake_score-only update (디버그용; DMD signal 사라짐).
    dmd_use_real_score: bool = True
    #: fake_score optimizer learning rate scale (lr_fake = lr_gen × scale).
    dmd_fake_lr_scale: float = 1.0
    #: True → Self-Forcing 의 abs-mean normalizer 적용 (synthetic grad stability).
    dmd_normalize: bool = True
    #: 매 N 번째 2Hz step 만 DMD anchor 로 사용.
    dmd_anchor_stride: int = 1
    #: True → future fine step 모두 valid 인 agent 만 DMD anchor 로 사용.
    dmd_strict_active_mask: bool = True
    #: 초기 N step 은 fake_score 만 update (generator no-op).  cold-start 안정성.
    dmd_warmup_fake_only_steps: int = 0
    #: generator backward 후 별도 gradient clip (0 = OCSC 의 bptt_grad_clip_traj 따름).
    dmd_gen_grad_clip: float = 0.0
    #: 매 training step hard RMM 모니터링.
    dmd_eval_hard_rmm: bool = True
    dmd_eval_hard_rmm_interval: int = 1


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
        flow_ft_target=str(_read_config_value(finetune, "flow_ft_target", "default")),
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
        ocsc_pwil_coupling=str(_read_config_value(finetune, "ocsc_pwil_coupling", "hungarian")),
        ocsc_pwil_use_exp_reward=bool(_read_config_value(finetune, "ocsc_pwil_use_exp_reward", True)),
        ocsc_pwil_alpha=float(_read_config_value(finetune, "ocsc_pwil_alpha", 1.0)),
        ocsc_pwil_beta=float(_read_config_value(finetune, "ocsc_pwil_beta", 5.0)),
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
        ocsc_ol_resolution=str(_read_config_value(finetune, "ocsc_ol_resolution", "10hz")),
        ocsc_nearest_include_gt=bool(_read_config_value(finetune, "ocsc_nearest_include_gt", False)),
        ocsc_strict_active_mask=bool(_read_config_value(finetune, "ocsc_strict_active_mask", False)),
        ocsc_ref_refresh_mode=str(_read_config_value(finetune, "ocsc_ref_refresh_mode", "frozen")),
        ocsc_ref_refresh_interval=int(_read_config_value(finetune, "ocsc_ref_refresh_interval", 0)),
        ocsc_ref_ema_decay=float(_read_config_value(finetune, "ocsc_ref_ema_decay", 0.999)),
        road_sample_k=int(_read_config_value(finetune, "road_sample_k", 64)),
        road_n_rollouts=int(_read_config_value(finetune, "road_n_rollouts", 1)),
        road_pred_max_steps=int(_read_config_value(finetune, "road_pred_max_steps", 16)),
        road_temperature=float(_read_config_value(finetune, "road_temperature", 0.8)),
        road_position_weight=float(_read_config_value(finetune, "road_position_weight", 1.0)),
        road_heading_weight=float(_read_config_value(finetune, "road_heading_weight", 0.1)),
        road_comparison_horizon=int(_read_config_value(finetune, "road_comparison_horizon", 20)),
        road_strict_active_mask=bool(_read_config_value(finetune, "road_strict_active_mask", True)),
        road_eval_hard_rmm=bool(_read_config_value(finetune, "road_eval_hard_rmm", False)),
        road_eval_hard_rmm_interval=int(_read_config_value(finetune, "road_eval_hard_rmm_interval", 10)),
        dmd_beta=float(_read_config_value(finetune, "dmd_beta", 1.0)),
        dmd_n_rollouts=int(_read_config_value(finetune, "dmd_n_rollouts", 1)),
        dmd_pred_max_steps=int(_read_config_value(finetune, "dmd_pred_max_steps", 2)),
        dmd_use_real_score=bool(_read_config_value(finetune, "dmd_use_real_score", True)),
        dmd_fake_lr_scale=float(_read_config_value(finetune, "dmd_fake_lr_scale", 1.0)),
        dmd_normalize=bool(_read_config_value(finetune, "dmd_normalize", True)),
        dmd_anchor_stride=int(_read_config_value(finetune, "dmd_anchor_stride", 1)),
        dmd_strict_active_mask=bool(_read_config_value(finetune, "dmd_strict_active_mask", True)),
        dmd_warmup_fake_only_steps=int(_read_config_value(finetune, "dmd_warmup_fake_only_steps", 0)),
        dmd_gen_grad_clip=float(_read_config_value(finetune, "dmd_gen_grad_clip", 0.0)),
        dmd_eval_hard_rmm=bool(_read_config_value(finetune, "dmd_eval_hard_rmm", True)),
        dmd_eval_hard_rmm_interval=int(_read_config_value(finetune, "dmd_eval_hard_rmm_interval", 1)),
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

    if config.mode not in ("ocsc_ft", "road_ft", "self_forcing_dmd"):
        raise ValueError(
            f"Unsupported finetune mode: {config.mode}. "
            "Supported: 'ocsc_ft', 'road_ft', 'self_forcing_dmd'."
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

    # ── flow_ft_target 우선 적용 ────────────────────────────────────────────
    # "default" 면 기존 flow_velocity_head_only 분기로 fallback.
    _target = str(getattr(config, "flow_ft_target", "default")).lower()
    if _target == "step_refiner_and_velocity_head":
        try:
            velocity_head = flow_decoder.velocity_head
            step_refiner = flow_decoder.step_refiner
        except AttributeError as exc:
            raise AttributeError(
                "flow_ft_target=step_refiner_and_velocity_head requires both "
                "velocity_head and step_refiner on flow_decoder."
            ) from exc
        _set_requires_grad(flow_decoder, False)
        _set_requires_grad(velocity_head, True)
        _set_requires_grad(step_refiner, True)
        if residual_head is not None:
            for p in residual_head.parameters():
                p.data.zero_()
            _set_requires_grad(residual_head, False)
        log.info(
            "Finetuning mode: step_refiner + velocity_head trainable; "
            "chunk_mixers / encoders / residual frozen."
        )
        return config
    if _target == "chunk_mixers_and_velocity_head":
        try:
            velocity_head = flow_decoder.velocity_head
            chunk_mixers = flow_decoder.chunk_mixers
        except AttributeError as exc:
            raise AttributeError(
                "flow_ft_target=chunk_mixers_and_velocity_head requires both "
                "velocity_head and chunk_mixers on flow_decoder."
            ) from exc
        _set_requires_grad(flow_decoder, False)
        _set_requires_grad(velocity_head, True)
        _set_requires_grad(chunk_mixers, True)
        if residual_head is not None:
            for p in residual_head.parameters():
                p.data.zero_()
            _set_requires_grad(residual_head, False)
        log.info(
            "Finetuning mode: chunk_mixers + velocity_head trainable; "
            "step_refiner / encoders / residual frozen."
        )
        return config
    if _target == "velocity_head":
        _velocity_head_only = True   # explicit alias
    elif _target == "full":
        _velocity_head_only = False  # explicit alias
    elif _target == "default":
        _velocity_head_only = bool(config.flow_velocity_head_only)
    else:
        log.warning(
            f"[finetune] unknown flow_ft_target={_target!r}; falling back to flow_velocity_head_only={config.flow_velocity_head_only}."
        )
        _velocity_head_only = bool(config.flow_velocity_head_only)

    # ── velocity_head만 학습 (트렁크·residual 동결) ─────────────────────────
    if _velocity_head_only:
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
