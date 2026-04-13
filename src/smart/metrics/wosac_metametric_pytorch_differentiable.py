from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Tuple, Dict, Optional

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
# angular_speed 는 과거 tau=0.01 로 설정했으나, softmax Jacobian 이 1/tau 스케일이므로
# tau=0.01 → tau=0.1 대비 10배 그래디언트 증폭. BPTT 학습에서 grad explosion 원인.
# → custom tau 없이 default_tau(=0.1) 를 일괄 적용.
_DEFAULT_SOFT_CUSTOM_TAUS: Dict[str, float] = {}


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