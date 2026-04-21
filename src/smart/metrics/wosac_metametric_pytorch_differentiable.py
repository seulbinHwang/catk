from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Mapping, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_metrics_pb2

_log = logging.getLogger(__name__)

# `compute_wosac_metametric_soft` 가중합 순서 (proto 필드명과 동일)
_METRIC_FIELD_NAMES = [
    "linear_speed",
    "linear_acceleration",
    "angular_speed",
    "angular_acceleration",
    "distance_to_nearest_object",
    "collision_indication",
    "time_to_collision",
    "distance_to_road_edge",
    "offroad_indication",
    "traffic_light_violation",
]
# tau 는 bin_width 에 비례하게 설정해야 gradient 가 흐른다.
# tau / bin_width ≈ 0.04 (linear_speed) 이면 softmax 가 완전히 포화 → gradient ≈ 0.
# 권장: tau ≈ bin_width / 2.
#   linear_speed:      bin_width=2.50  → tau=1.25
#   linear_accel:      bin_width=2.18  → tau=1.00
#   angular_speed:     bin_width=0.114 → tau=0.05  (과거 0.01은 grad explosion, 0.1은 양호했으나 조금 더 날카롭게)
#   angular_accel:     bin_width=0.571 → tau=0.25
# 주의: tau 를 키우면 gradient 가 살아나는 대신 metric 추정 자체가 softer 해짐.
#        bptt_grad_clip_traj 로 전체 clip 을 조절하는 것을 권장.
_DEFAULT_SOFT_CUSTOM_TAUS: Dict[str, float] = {
    "linear_speed": 1.25,        # bin_width=2.50 m/s
    "linear_acceleration": 1.00, # bin_width=2.18 m/s²
    "angular_speed": 0.05,       # bin_width=0.114 rad/s  (0.1에서 축소 — grad 크기 2x, explosion 주의)
    "angular_acceleration": 0.25, # bin_width=0.571 rad/s²
}


def _squeeze_log_sample(x: Tensor) -> Tensor:
    return x[0] if x.ndim > 0 and x.shape[0] == 1 else x


# ===========================================================================
# 1. 미분 가능한 유효성 및 평균 연산 (Soft Logic)
# ===========================================================================
def _central_logical_and_time_soft(valid: Tensor, pad_value: float = 0.0) -> Tensor:
    """AND 연산을 곱셈으로 대체하여 그라디언트 전파"""
    pad_shape = (*valid.shape[:-1], 1)
    pad_tensor = torch.full(pad_shape, pad_value, dtype=valid.dtype, device=valid.device)
    inner = valid[..., 2:] * valid[..., :-2]
    return torch.cat([pad_tensor, inner, pad_tensor], dim=-1)

def compute_kinematic_validity_soft(valid: Tensor) -> Tuple[Tensor, Tensor]:
    speed_validity = _central_logical_and_time_soft(valid, 0.0)
    acceleration_validity = _central_logical_and_time_soft(speed_validity, 0.0)
    return speed_validity, acceleration_validity

def _reduce_average_with_validity_soft(tensor: Tensor, validity: Tensor) -> Tensor:
    """Masking을 곱셈으로 처리하여 미분 가능하게 평균 산출"""
    cond_sum = (tensor * validity).sum()
    valid_sum = validity.sum().clamp(min=1e-5)
    return cond_sum / valid_sum


def _likelihood_from_log_ll(ll: Tensor) -> Tensor:
    """log-likelihood 합을 [0,1] 근처 likelihood로 변환. exp 오버플로·역전파 불안정 방지."""
    return torch.exp(torch.clamp(ll, min=-80.0, max=0.0))

# ===========================================================================
# 2. 미분 가능한 핵심 수학 함수 (Soft Any & Soft Binning)
# ===========================================================================
def soft_any(x: Tensor, dim: int = -1, beta: float = 10.0) -> Tensor:
    """Smooth Maximum (LogSumExp와 유사한 가중 합 방식).

    Non-finite 값(inf/NaN)을 0으로 치환해 softmax NaN을 방지합니다.
    """
    x_safe = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    weights = F.softmax(x_safe * beta, dim=dim)
    return (x_safe * weights).sum(dim=dim)

def _soft_bin_assignment(x: Tensor, edges: Tensor, tau: float) -> Tensor:
    """
    각 샘플을 빈(Bin)에 할당할 때 Softmax를 사용.
    tau가 작을수록 Hard 할당에 가까워지며, Angular Speed처럼 정밀한 값은 작은 tau 권장.
    """
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    # [..., 1, num_bins] 와 [..., num_samples, 1] 사이의 거리 계산
    dist = torch.abs(x.unsqueeze(-1) - bin_centers)
    return F.softmax(-dist / tau, dim=-1)

# ===========================================================================
# 3. 확률 분포 및 Likelihood 추정기
# ===========================================================================
def histogram_estimate_soft_torch(
    min_val: float, max_val: float, num_bins: int,
    pseudo: float, log_samples: Tensor, sim_samples: Tensor, tau: float
) -> Tensor:
    dtype = log_samples.dtype
    device = log_samples.device
    edges = torch.linspace(min_val, max_val, num_bins + 1, device=device, dtype=dtype)

    # Waymo Parity: Non-finite 값 처리
    def _sanitize(t):
        t = torch.where(torch.isfinite(t), t, torch.full_like(t, max_val))
        return t.clamp(min_val, max_val)

    log_c = _sanitize(log_samples)
    sim_c = _sanitize(sim_samples)

    # Sim samples로 확률 분포 생성
    sim_soft_assign = _soft_bin_assignment(sim_c, edges, tau)
    sim_counts = sim_soft_assign.sum(dim=1) + pseudo
    denom = sim_counts.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    probs = sim_counts / denom
    log_probs = torch.log(probs.clamp(min=1e-10))

    # Log samples의 Likelihood 산출
    log_soft_assign = _soft_bin_assignment(log_c, edges, tau)
    return (log_soft_assign * log_probs.unsqueeze(1)).sum(dim=-1)

def log_likelihood_estimate_soft(
    feature_config, log_values: Tensor, sim_values: Tensor, tau: float, is_timeseries: bool
) -> Tensor:
    if is_timeseries:
        n_rollouts, n_objects, n_steps = sim_values.shape
        sim_flat = sim_values.permute(1, 0, 2).reshape(n_objects, n_rollouts * n_steps)
        log_v = log_values
    else:
        # Scenario level: log [N], sim [G, N] — histogram 경로는 timeseries와 동일하게 [N,1] / [N,G]
        if log_values.dim() != 1 or sim_values.dim() != 2:
            raise ValueError(
                f"scenario level expects log [N], sim [G,N]; got {log_values.shape}, {sim_values.shape}"
            )
        sim_flat = sim_values.transpose(0, 1)
        log_v = log_values.unsqueeze(-1)

    which = feature_config.WhichOneof("estimator")
    if which == "histogram":
        h = feature_config.histogram
        ll = histogram_estimate_soft_torch(h.min_val, h.max_val, h.num_bins, h.additive_smoothing_pseudocount, log_v, sim_flat, tau)
    else: # bernoulli
        h = feature_config.bernoulli
        ll = histogram_estimate_soft_torch(-0.5, 1.5, 2, h.additive_smoothing_pseudocount, log_v, sim_flat, tau)
        
    return ll if is_timeseries else ll.squeeze(-1)

# ===========================================================================
# 4. 최종 통합 Metametric (Main Interface)
# ===========================================================================
@dataclass
class WosacMetametricSoftResult:
    metametric: Tensor
    likelihoods: Dict[str, Tensor]

def compute_wosac_metametric_soft(
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    log_features: Mapping[str, Tensor],
    sim_features: Mapping[str, Tensor],
    beta: float = 10.0,
    default_tau: float = 0.1,
    custom_taus: Dict[str, float] = None,
    debug: bool = False,
) -> WosacMetametricSoftResult:
    """
    Refined Differentiable Metametric.
    - ``default_tau``: 대부분의 시계열/시나리오 soft-bin 온도.
    - 기본으로 ``angular_speed`` 만 ``_DEFAULT_SOFT_CUSTOM_TAUS``(0.01)로 더 날카롭게 둠.
    - ``custom_taus``로 항목별 재정의 가능(예: ``{"angular_speed": 0.05}``).
    """
    dtype = log_features["linear_speed"].dtype
    device = log_features["linear_speed"].device
    taus = {fn: default_tau for fn in _METRIC_FIELD_NAMES}
    taus.update(_DEFAULT_SOFT_CUSTOM_TAUS)
    if custom_taus:
        taus.update(custom_taus)

    valid_log = _squeeze_log_sample(log_features["valid"]).to(dtype)
    speed_valid, accel_valid = compute_kinematic_validity_soft(valid_log)
    valid_expand = valid_log.unsqueeze(0)

    lik_dict = {}

    # --- 1. 시계열 기반 항목 (Timeseries) ---
    ts_fields = [
        ("linear_speed", config.linear_speed, speed_valid),
        ("angular_speed", config.angular_speed, speed_valid),
        ("linear_acceleration", config.linear_acceleration, accel_valid),
        ("angular_acceleration", config.angular_acceleration, accel_valid),
        ("distance_to_nearest_object", config.distance_to_nearest_object, valid_log),
        ("distance_to_road_edge", config.distance_to_road_edge, valid_log),
        ("time_to_collision", config.time_to_collision, None), # 하단 별도 처리
    ]

    for name, cfg, mask in ts_fields:
        if name == "time_to_collision":
            ot = _squeeze_log_sample(log_features["object_type"]).long()
            mask = valid_log * (ot == int(scenario_pb2.Track.ObjectType.TYPE_VEHICLE)).to(dtype).unsqueeze(-1)
        
        ll = log_likelihood_estimate_soft(cfg, _squeeze_log_sample(log_features[name]).to(dtype), sim_features[name].to(dtype), taus[name], True)
        lik_dict[f"{name}_likelihood"] = _likelihood_from_log_ll(
            _reduce_average_with_validity_soft(ll, mask)
        )

    # --- 2. 시나리오 기반 항목 (Scenario-level Any) ---
    sc_fields = [
        ("collision_indication", "collision_per_step", config.collision_indication, valid_log, valid_expand),
        ("offroad_indication", "offroad_per_step", config.offroad_indication, valid_log, valid_expand),
        ("traffic_light_violation", "traffic_light_violation_per_step", config.traffic_light_violation, None, None),
    ]

    for name, feat_key, cfg, v_log, v_exp in sc_fields:
        if name == "traffic_light_violation":
            ot = _squeeze_log_sample(log_features["object_type"]).long()
            v_log = valid_log * (ot == int(scenario_pb2.Track.ObjectType.TYPE_VEHICLE)).to(dtype).unsqueeze(-1)
            v_exp = v_log.unsqueeze(0)

        # Soft-Any 적용하여 시나리오당 0~1 값 추출
        log_any = soft_any(_squeeze_log_sample(log_features[feat_key]).to(dtype) * v_log, dim=-1, beta=beta)
        sim_any = soft_any(sim_features[feat_key].to(dtype) * v_exp, dim=-1, beta=beta)
        
        ll = log_likelihood_estimate_soft(cfg, log_any, sim_any, taus[name], False)
        lik_dict[f"{name}_likelihood"] = _likelihood_from_log_ll(ll.mean())

    # --- 3. 최종 Metametric 가중합 ---
    metametric = torch.tensor(0.0, dtype=dtype, device=device)
    for fn in _METRIC_FIELD_NAMES:
        weight = getattr(config, fn).metametric_weight
        metametric = metametric + weight * lik_dict[f"{fn}_likelihood"]

    if debug:
        parts = {fn: float(lik_dict[f"{fn}_likelihood"].detach()) for fn in _METRIC_FIELD_NAMES}
        _log.warning(
            "[soft_rmm_debug] metametric=%.4f | %s",
            float(metametric.detach()),
            " ".join(f"{k}={v:.3f}" for k, v in parts.items()),
        )

    return WosacMetametricSoftResult(metametric=metametric, likelihoods=lik_dict)


# =========================================================================
# 5. Batched Metametric — S scenarios in one fused GPU pass
# =========================================================================

def compute_wosac_metametric_soft_batched(
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    log_features_list: List[Mapping[str, Tensor]],
    sim_features_list: List[Mapping[str, Tensor]],
    beta: float = 10.0,
    default_tau: float = 0.1,
    custom_taus: Optional[Dict[str, float]] = None,
    debug: bool = False,
    train_weight_overrides: Optional[Dict[str, float]] = None,
) -> Tensor:
    """``train_weight_overrides``: 학습 loss 전용 weight 재정의 (로깅에는 영향 없음).
    예) 이미 포화된 safety 항 축소, 낮은 kinematic 항 증폭:
        {
            "linear_speed": 0.15, "linear_acceleration": 0.15, "angular_speed": 0.15,
            "angular_acceleration": 0.10,
            "collision_indication": 0.10, "offroad_indication": 0.10,
            "distance_to_nearest_object": 0.10, "distance_to_road_edge": 0.10,
            "time_to_collision": 0.10, "traffic_light_violation": 0.0,
        }
    """
    """Batched differentiable metametric for *S* scenarios in a single pass.

    Features are padded to ``A_max`` (the largest agent count across
    scenarios) and all histogram / soft-any / likelihood computations run
    as fused GPU kernels over the ``(S * A_max)`` flat batch.
    Padded agents are masked out via per-scenario validity so they never
    affect the per-scenario metametric values.

    Returns
    -------
    Tensor, shape ``(S,)``
        Per-scenario differentiable metametric.
    """
    S = len(log_features_list)
    if S == 0:
        return torch.tensor([], dtype=torch.float32)
    if S == 1:
        r = compute_wosac_metametric_soft(
            config, log_features_list[0], sim_features_list[0],
            beta, default_tau, custom_taus, debug,
        )
        return r.metametric.unsqueeze(0)

    dtype = _squeeze_log_sample(log_features_list[0]["linear_speed"]).dtype
    device = _squeeze_log_sample(log_features_list[0]["linear_speed"]).device

    taus = {fn: default_tau for fn in _METRIC_FIELD_NAMES}
    taus.update(_DEFAULT_SOFT_CUSTOM_TAUS)
    if custom_taus:
        taus.update(custom_taus)

    n_agents = [_squeeze_log_sample(lf["valid"]).shape[0] for lf in log_features_list]
    A = max(n_agents)
    T = _squeeze_log_sample(log_features_list[0]["valid"]).shape[-1]
    G = sim_features_list[0]["valid"].shape[0]

    agent_mask = torch.zeros(S, A, dtype=dtype, device=device)
    for i, na in enumerate(n_agents):
        agent_mask[i, :na] = 1.0
    agent_time_mask = agent_mask.unsqueeze(-1)  # (S, A, 1)

    # ── pad helpers ──────────────────────────────────────────────────────────

    def _pad_log(key: str) -> Tensor:
        """(A_i, T) → (S, A, T). No grad needed (ground-truth log features)."""
        out = torch.zeros(S, A, T, dtype=dtype, device=device)
        for i, lf in enumerate(log_features_list):
            t = _squeeze_log_sample(lf[key]).to(dtype).to(device)
            out[i, :t.shape[0]] = t
        return out

    def _pad_sim_flat(key: str) -> Tensor:
        """(G, A_i, T) → (S*A, G*T). Gradient-safe (F.pad + stack)."""
        parts: List[Tensor] = []
        for sf in sim_features_list:
            t = sf[key].to(dtype).to(device)
            flat = t.permute(1, 0, 2).reshape(t.shape[1], G * T)
            if flat.shape[0] < A:
                flat = F.pad(flat, (0, 0, 0, A - flat.shape[0]))
            parts.append(flat)
        return torch.stack(parts).reshape(S * A, G * T)

    def _pad_sim_3d(key: str) -> Tensor:
        """(G, A_i, T) → (S, G, A, T). Gradient-safe."""
        parts: List[Tensor] = []
        for sf in sim_features_list:
            t = sf[key].to(dtype).to(device)
            if t.shape[1] < A:
                t = F.pad(t, (0, 0, 0, A - t.shape[1]))
            parts.append(t)
        return torch.stack(parts)

    # ── histogram dispatch ───────────────────────────────────────────────────

    def _hist(cfg_f, log_flat: Tensor, sim_flat: Tensor, tau: float) -> Tensor:
        w = cfg_f.WhichOneof("estimator")
        if w == "histogram":
            h = cfg_f.histogram
            return histogram_estimate_soft_torch(
                h.min_val, h.max_val, h.num_bins,
                h.additive_smoothing_pseudocount, log_flat, sim_flat, tau,
            )
        h = cfg_f.bernoulli
        return histogram_estimate_soft_torch(
            -0.5, 1.5, 2, h.additive_smoothing_pseudocount,
            log_flat, sim_flat, tau,
        )

    # ── validity ─────────────────────────────────────────────────────────────

    log_valid = _pad_log("valid") * agent_time_mask
    speed_valid, accel_valid = compute_kinematic_validity_soft(log_valid)

    lik_dict: Dict[str, Tensor] = {}

    # ── 1. timeseries metrics ────────────────────────────────────────────────

    ts_fields = [
        ("linear_speed",               config.linear_speed,               speed_valid),
        ("angular_speed",              config.angular_speed,              speed_valid),
        ("linear_acceleration",        config.linear_acceleration,        accel_valid),
        ("angular_acceleration",       config.angular_acceleration,       accel_valid),
        ("distance_to_nearest_object", config.distance_to_nearest_object, log_valid),
        ("distance_to_road_edge",      config.distance_to_road_edge,      log_valid),
        ("time_to_collision",          config.time_to_collision,          None),
    ]

    _ot_padded: Optional[Tensor] = None

    for name, cfg_f, mask in ts_fields:
        if name == "time_to_collision":
            if _ot_padded is None:
                _ot_padded = torch.zeros(S, A, dtype=torch.long, device=device)
                for i, lf in enumerate(log_features_list):
                    ot = _squeeze_log_sample(lf["object_type"]).long().to(device)
                    _ot_padded[i, :ot.shape[0]] = ot
            mask = log_valid * (
                _ot_padded == int(scenario_pb2.Track.ObjectType.TYPE_VEHICLE)
            ).to(dtype).unsqueeze(-1)

        log_vals = _pad_log(name).reshape(S * A, T)
        sim_flat = _pad_sim_flat(name)

        ll = _hist(cfg_f, log_vals, sim_flat, taus[name]).reshape(S, A, T)

        masked_sum = (ll * mask).sum(dim=(1, 2))
        valid_sum  = mask.sum(dim=(1, 2)).clamp(min=1e-5)
        lik_dict[f"{name}_likelihood"] = _likelihood_from_log_ll(masked_sum / valid_sum)

    # ── 2. scenario-level metrics ────────────────────────────────────────────

    sc_fields = [
        ("collision_indication",    "collision_per_step",                config.collision_indication,   log_valid),
        ("offroad_indication",      "offroad_per_step",                  config.offroad_indication,     log_valid),
        ("traffic_light_violation", "traffic_light_violation_per_step",  config.traffic_light_violation, None),
    ]

    for name, feat_key, cfg_f, v_log in sc_fields:
        if name == "traffic_light_violation":
            if _ot_padded is None:
                _ot_padded = torch.zeros(S, A, dtype=torch.long, device=device)
                for i, lf in enumerate(log_features_list):
                    ot = _squeeze_log_sample(lf["object_type"]).long().to(device)
                    _ot_padded[i, :ot.shape[0]] = ot
            v_log = log_valid * (
                _ot_padded == int(scenario_pb2.Track.ObjectType.TYPE_VEHICLE)
            ).to(dtype).unsqueeze(-1)

        log_feat = _pad_log(feat_key)
        sim_feat = _pad_sim_3d(feat_key)

        log_any = soft_any(log_feat * v_log, dim=-1, beta=beta)
        v_exp   = v_log.unsqueeze(1)
        sim_any = soft_any(sim_feat * v_exp, dim=-1, beta=beta)

        log_v    = log_any.reshape(S * A, 1)
        sim_v    = sim_any.permute(0, 2, 1).reshape(S * A, G)
        ll_flat  = _hist(cfg_f, log_v, sim_v, taus[name])
        ll       = ll_flat.reshape(S, A)

        masked_sum  = (ll * agent_mask).sum(dim=1)
        valid_count = agent_mask.sum(dim=1).clamp(min=1e-5)
        lik_dict[f"{name}_likelihood"] = _likelihood_from_log_ll(masked_sum / valid_count)

    # ── 3. weighted sum ──────────────────────────────────────────────────────
    # train_weight_overrides 가 있으면 학습 loss 전용으로 재정의한 weight 사용.
    # 로깅(debug) 은 항상 원본 config 가중치 기준.

    metametric = torch.zeros(S, dtype=dtype, device=device)
    for fn in _METRIC_FIELD_NAMES:
        if train_weight_overrides is not None and fn in train_weight_overrides:
            w = train_weight_overrides[fn]
        else:
            w = getattr(config, fn).metametric_weight
        metametric = metametric + w * lik_dict[f"{fn}_likelihood"]

    if debug:
        with torch.no_grad():
            parts = {fn: lik_dict[f"{fn}_likelihood"].detach().cpu().tolist()
                     for fn in _METRIC_FIELD_NAMES}
            _log.warning(
                "[soft_rmm_batched] metametric=%s | %s",
                metametric.detach().cpu().tolist(),
                " ".join(f"{k}={v}" for k, v in parts.items()),
            )

    return metametric