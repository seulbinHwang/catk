# Not a contribution
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Pure PyTorch WOSAC realism metametric — **no** ``wdl_limited.sim_agents_metrics.metrics`` calls.

Replicates ``metrics.compute_scenario_metrics_for_features_bundle`` + histogram / Bernoulli
logic from ``waymo_open_dataset/wdl_limited/sim_agents_metrics/{estimators,trajectory_features}.py``.

Inputs are ``MetricFeatures``-aligned tensor dicts (same keys / shapes as TF).  Use
``metric_features.compute_scenario_rollouts_features`` elsewhere only to **build** tensors;
this module does not import Waymo metric *functions*.

**Parity:** On sample scenarios, metametric matches ``compute_scenario_metrics_for_bundle``
within ~``1e-7`` (float32); non-finite histogram inputs are mapped like TFP/Waymo (NaN →
upper bin). See ``scripts/verify_wosac_metametric_pytorch_parity.py``.

Dependencies: ``torch``, ``waymo_open_dataset.protos`` (for ``SimAgentMetricsConfig`` and
``scenario_pb2.Track.ObjectType.TYPE_VEHICLE`` only).

**코드 읽는 순서 (권장)**

1. ``compute_wosac_metametric_from_features_torch`` — 메인: 10개 likelihood → 가중합 metametric.
2. ``log_likelihood_estimate_timeseries_torch`` / ``..._scenario_level_torch`` — 공식 ``estimators`` reshape + 추정기 분기.
3. ``histogram_estimate_torch`` — 시뮬 샘플로 빈 확률 만들고, 로그 샘플의 log p.
4. ``compute_kinematic_validity`` / ``_reduce_average_with_validity`` — 공식 ``trajectory_features`` / ``metrics`` 와 동일 마스킹·평균.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

# proto: (1) SimAgentMetricsConfig = textproto 가중치·히스토그램 설정
#       (2) TYPE_VEHICLE = TTC·신호위반 마스크에만 사용 (공식 metrics.py 와 동일)
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_metrics_pb2


# ---------------------------------------------------------------------------
# trajectory_features.py — 중앙차분에 쓸 유효 시점 마스크
# ---------------------------------------------------------------------------
def _central_logical_and_time(valid: Tensor, pad_value: bool) -> Tensor:
    """``trajectory_features.central_logical_and`` on last axis."""
    pad_shape = (*valid.shape[:-1], 1)
    pad_tensor = torch.full(pad_shape, pad_value, dtype=torch.bool, device=valid.device)
    # 시각 t에서 유효하려면 t-1, t+1 둘 다 valid (중앙차분과 동일 조건)
    inner = valid[..., 2:] & valid[..., :-2]
    return torch.cat([pad_tensor, inner, pad_tensor], dim=-1)


def compute_kinematic_validity(valid: Tensor) -> Tuple[Tensor, Tensor]:
    """``trajectory_features.compute_kinematic_validity`` — [..., T] bool."""
    speed_validity = _central_logical_and_time(valid, False)
    # 가속도는 속도에 한 번 더 같은 연산 → 사실상 t±2 구간 valid 필요
    acceleration_validity = _central_logical_and_time(speed_validity, False)
    return speed_validity, acceleration_validity


# ---------------------------------------------------------------------------
# metrics.py — 유효한 (객체, 시각) 만 평균
# ---------------------------------------------------------------------------
def _reduce_average_with_validity(tensor: Tensor, validity: Tensor) -> Tensor:
    """``metrics._reduce_average_with_validity`` (both arguments broadcastable same shape)."""
    if tensor.shape != validity.shape:
        raise ValueError(f"shape mismatch {tensor.shape} vs {validity.shape}")
    z = torch.zeros_like(tensor)
    cond_sum = torch.where(validity, tensor, z).sum()
    valid_sum = validity.to(dtype=torch.float32).sum()
    return cond_sum / valid_sum.clamp(min=1.0)


# ---------------------------------------------------------------------------
# estimators.histogram_estimate — TFP uniform bin 과 동일하게 균일 구간 인덱스
# ---------------------------------------------------------------------------
def _bin_indices_uniform(x: Tensor, edges: Tensor, num_bins: int) -> Tensor:
    """TFP ``histogram`` half-open bins + last bin closed at ``edges[-1]`` (``quantiles.py``)."""
    h_min, h_max = edges[0], edges[-1]
    x = x.clamp(h_min, h_max)
    width = (h_max - h_min) / num_bins
    idx = ((x - h_min) / width).floor().to(torch.long)
    return idx.clamp(0, num_bins - 1)


def histogram_estimate_torch(
    min_val: float,
    max_val: float,
    num_bins: int,
    additive_smoothing_pseudocount: float,
    log_samples: Tensor,
    sim_samples: Tensor,
) -> Tensor:
    """``estimators.histogram_estimate`` — log_samples ``[B, L]``, sim_samples ``[B, S]``, float.

    배치 차원 B = 독립인 “모집단” 개수(보통 객체 수). 각 행 i에서:
    - 시뮬 샘플들로 히스토그램(의사카운트 스무딩) → 이산 확률 p_k
    - 로그 샘플 각각이 속한 빈 k에 대해 log p_k 반환 → shape [B, L]
    """
    if log_samples.shape[0] != sim_samples.shape[0]:
        raise ValueError("batch dim mismatch")
    device = log_samples.device
    dtype = log_samples.dtype

    edges = torch.linspace(min_val, max_val, num_bins + 1, device=device, dtype=dtype)
    # Match TFP ``find_bins`` / Waymo pipeline: non-finite values fall into the upper bin
    # (same effect as clipping NaN after replacement missing in ``clip_by_value``).
    log_c = torch.where(
        torch.isfinite(log_samples),
        log_samples,
        torch.full_like(log_samples, max_val),
    ).clamp(min_val, max_val)
    sim_c = torch.where(
        torch.isfinite(sim_samples),
        sim_samples,
        torch.full_like(sim_samples, max_val),
    ).clamp(min_val, max_val)

    sim_idx = _bin_indices_uniform(sim_c, edges, num_bins)
    sim_oh = F.one_hot(sim_idx, num_classes=num_bins).to(dtype)
    # dim=1: 각 배치 행 안의 모든 시뮬 샘플(S개)을 빈별로 합산 → 시뮬 경험분포
    # Accumulate counts in float64 (closer to TF/tfp numerics), then cast for log.
    sim_counts = sim_oh.sum(dim=1).to(torch.float64) + float(additive_smoothing_pseudocount)
    probs = sim_counts / sim_counts.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    probs = probs.to(dtype)
    log_probs = torch.log(probs.clamp(min=torch.finfo(dtype).tiny))

    log_idx = _bin_indices_uniform(log_c, edges, num_bins)
    # 로그의 각 샘플이 시뮬 분포에서 어느 빈의 log 확률인지
    out = torch.gather(log_probs, 1, log_idx)
    return out


# ---------------------------------------------------------------------------
# estimators.log_likelihood_estimate_timeseries (SIM_AGENTS, agent-level)
# ---------------------------------------------------------------------------
def log_likelihood_estimate_timeseries_torch(
    feature_config: sim_agents_metrics_pb2.SimAgentMetricsConfig.FeatureConfig,
    log_values: Tensor,
    sim_values: Tensor,
) -> Tensor:
    """``estimators.log_likelihood_estimate_timeseries`` for SIM_AGENTS (``aggregate_objects=False``)."""
    if feature_config.aggregate_objects:
        raise NotImplementedError("aggregate_objects=True (scenario_gen) not implemented")

    which = feature_config.WhichOneof("estimator")
    if which == "kernel_density":
        raise NotImplementedError("kernel_density estimator not implemented")
    if which not in ("histogram", "bernoulli"):
        raise ValueError(f"unknown estimator {which}")

    if log_values.dim() != 2:
        raise ValueError(f"log_values rank {log_values.dim()}, expected 2")
    if sim_values.dim() != 3:
        raise ValueError(f"sim_values rank {sim_values.dim()}, expected 3")

    n_rollouts, n_objects, n_steps = sim_values.shape
    if log_values.shape != (n_objects, n_steps):
        raise ValueError(
            f"log shape {log_values.shape} != ({n_objects}, {n_steps}) for sim {sim_values.shape}"
        )

    # independent_timesteps=True (Sim Agents 기본): 객체당 시뮬 샘플을 (G*T)개로 펴서
    # 한 객체당 하나의 히스토그램. False면 (객체*시각)마다 rollout만큼만 샘플.
    if feature_config.independent_timesteps:
        sim_flat = sim_values.permute(1, 0, 2).reshape(n_objects, n_rollouts * n_steps)
        log_v = log_values
    else:
        sim_flat = sim_values.permute(1, 2, 0).reshape(n_objects * n_steps, n_rollouts)
        log_v = log_values.reshape(n_objects * n_steps, 1)

    if which == "histogram":
        h = feature_config.histogram
        ll_flat = histogram_estimate_torch(
            h.min_val,
            h.max_val,
            h.num_bins,
            h.additive_smoothing_pseudocount,
            log_v,
            sim_flat,
        )
    else:
        # bernoulli_estimate = [-0.5, 1.5] 구간 2-bin 히스토그램 (0/1)
        h = feature_config.bernoulli
        pseudo = h.additive_smoothing_pseudocount
        lv = log_v.float()
        sv = sim_flat.float()
        ll_flat = histogram_estimate_torch(
            -0.5,
            1.5,
            2,
            pseudo,
            lv,
            sv,
        )

    if feature_config.independent_timesteps:
        return ll_flat
    return ll_flat.view(n_objects, n_steps)


# ---------------------------------------------------------------------------
# estimators.log_likelihood_estimate_scenario_level (시간 축 any 후 0/1만 남은 항목)
# ---------------------------------------------------------------------------
def log_likelihood_estimate_scenario_level_torch(
    feature_config: sim_agents_metrics_pb2.SimAgentMetricsConfig.FeatureConfig,
    log_values: Tensor,
    sim_values: Tensor,
) -> Tensor:
    """``estimators.log_likelihood_estimate_scenario_level`` — ``log`` [N], ``sim`` [G, N]."""
    if log_values.dim() != 1 or sim_values.dim() != 2:
        raise ValueError(f"shapes log={log_values.shape} sim={sim_values.shape}")
    # 공식: 길이 1짜리 가짜 시계열로 timeseries 경로 재사용 → (n_obj,) log-lik
    ll = log_likelihood_estimate_timeseries_torch(
        feature_config,
        log_values.unsqueeze(-1),
        sim_values.unsqueeze(-1),
    )
    return ll.squeeze(-1)


# metrics._compute_metametric 에 넣는 필드 순서 (proto 필드명과 동일)
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


def _squeeze_log_sample(x: Tensor) -> Tensor:
    """MetricFeatures 에서 로그 쪽은 n_samples=1 이라 앞 차원 [1, ...] 제거."""
    if x.shape[0] == 1:
        return x[0]
    return x


@dataclass
class WosacMetametricTorchResult:
    metametric: float
    linear_speed_likelihood: float
    linear_acceleration_likelihood: float
    angular_speed_likelihood: float
    angular_acceleration_likelihood: float
    distance_to_nearest_object_likelihood: float
    collision_indication_likelihood: float
    time_to_collision_likelihood: float
    distance_to_road_edge_likelihood: float
    offroad_indication_likelihood: float
    traffic_light_violation_likelihood: float


def compute_wosac_metametric_from_features_torch(
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    log_features: Mapping[str, Tensor],
    sim_features: Mapping[str, Tensor],
    *,
    dtype: torch.dtype = torch.float32,
) -> WosacMetametricTorchResult:
    """Port of ``metrics.compute_scenario_metrics_for_features_bundle`` (metametric + likelihoods only).

    Tensor keys (same as TF ``MetricFeatures``):

    - ``valid``: ``[1, n_obj, T]`` (log), ``[G, n_obj, T]`` (sim)
    - ``object_type``: ``[1, n_obj]`` or ``[G, n_obj]`` int (``Track.ObjectType`` enum)
    - ``linear_speed``, ``linear_acceleration``, ``angular_speed``, ``angular_acceleration``:
      ``[1,n,T]`` / ``[G,n,T]`` float
    - ``distance_to_nearest_object``, ``time_to_collision``, ``distance_to_road_edge``: idem
    - ``collision_per_step``, ``offroad_per_step``, ``traffic_light_violation_per_step``:
      ``[G, n, T]`` bool

    Tensors must live on the same device. ADE / sim rates are omitted (not part of metametric sum).
    """
    # --- 공통 마스크 ---
    valid_log = _squeeze_log_sample(log_features["valid"]).bool()
    speed_valid, accel_valid = compute_kinematic_validity(valid_log)

    # --- 시계열 히스토그램 항목: log p 그리드 → 유효 마스크로 평균 → exp → [0,1] likelihood ---
    ls_ll = log_likelihood_estimate_timeseries_torch(
        config.linear_speed,
        _squeeze_log_sample(log_features["linear_speed"]).to(dtype),
        sim_features["linear_speed"].to(dtype),
    )
    linear_speed_likelihood = torch.exp(_reduce_average_with_validity(ls_ll, speed_valid))

    as_ll = log_likelihood_estimate_timeseries_torch(
        config.angular_speed,
        _squeeze_log_sample(log_features["angular_speed"]).to(dtype),
        sim_features["angular_speed"].to(dtype),
    )
    angular_speed_likelihood = torch.exp(_reduce_average_with_validity(as_ll, speed_valid))

    la_ll = log_likelihood_estimate_timeseries_torch(
        config.linear_acceleration,
        _squeeze_log_sample(log_features["linear_acceleration"]).to(dtype),
        sim_features["linear_acceleration"].to(dtype),
    )
    linear_accel_likelihood = torch.exp(_reduce_average_with_validity(la_ll, accel_valid))

    aa_ll = log_likelihood_estimate_timeseries_torch(
        config.angular_acceleration,
        _squeeze_log_sample(log_features["angular_acceleration"]).to(dtype),
        sim_features["angular_acceleration"].to(dtype),
    )
    angular_accel_likelihood = torch.exp(_reduce_average_with_validity(aa_ll, accel_valid))

    dno_ll = log_likelihood_estimate_timeseries_torch(
        config.distance_to_nearest_object,
        _squeeze_log_sample(log_features["distance_to_nearest_object"]).to(dtype),
        sim_features["distance_to_nearest_object"].to(dtype),
    )
    # 최근접 거리 등은 공식과 같이 log_features.valid[0] (speed_valid 아님)
    distance_to_obj_likelihood = torch.exp(_reduce_average_with_validity(dno_ll, valid_log))

    ttc_ll = log_likelihood_estimate_timeseries_torch(
        config.time_to_collision,
        _squeeze_log_sample(log_features["time_to_collision"]).to(dtype),
        sim_features["time_to_collision"].to(dtype),
    )
    ot = _squeeze_log_sample(log_features["object_type"]).long()
    is_vehicle = ot == int(scenario_pb2.Track.ObjectType.TYPE_VEHICLE)
    # TTC: 차량만 (공식 metrics.py 의 is_vehicle)
    ttc_validity = valid_log & is_vehicle.unsqueeze(-1)
    ttc_likelihood = torch.exp(_reduce_average_with_validity(ttc_ll, ttc_validity))

    dre_ll = log_likelihood_estimate_timeseries_torch(
        config.distance_to_road_edge,
        _squeeze_log_sample(log_features["distance_to_road_edge"]).to(dtype),
        sim_features["distance_to_road_edge"].to(dtype),
    )
    distance_to_road_edge_likelihood = torch.exp(_reduce_average_with_validity(dre_ll, valid_log))

    # --- 베르누(시나리오 단위): 시간 any → 객체당 0/1, rollout×객체로 2-bin 히스토그램 ---
    valid_expand = valid_log.unsqueeze(0)  # [1, n, T] 브로드캐스트
    sim_col = (sim_features["collision_per_step"].bool() & valid_expand).any(dim=-1)
    log_col = (_squeeze_log_sample(log_features["collision_per_step"]).bool() & valid_log).any(
        dim=-1
    )
    collision_score = log_likelihood_estimate_scenario_level_torch(
        config.collision_indication,
        log_col.to(dtype),
        sim_col.to(dtype),
    )
    # collision_score: (n_obj,) → 공식은 reduce_mean 후 exp
    collision_likelihood = torch.exp(collision_score.mean())

    sim_off = (sim_features["offroad_per_step"].bool() & valid_expand).any(dim=-1)
    log_off = (_squeeze_log_sample(log_features["offroad_per_step"]).bool() & valid_log).any(
        dim=-1
    )
    off_score = log_likelihood_estimate_scenario_level_torch(
        config.offroad_indication,
        log_off.to(dtype),
        sim_off.to(dtype),
    )
    offroad_likelihood = torch.exp(off_score.mean())

    # 신호위반: 유효 & 차량 마스크를 로그/시뮬에 맞게 브로드캐스트 (metrics.py 와 동일)
    tl_validity = valid_log & is_vehicle.unsqueeze(-1)
    log_tl = (
        _squeeze_log_sample(log_features["traffic_light_violation_per_step"]).bool()
        & tl_validity
    ).any(dim=-1)
    sim_tl = (
        sim_features["traffic_light_violation_per_step"].bool()
        & tl_validity.unsqueeze(0)
    ).any(dim=-1)
    tl_score = log_likelihood_estimate_scenario_level_torch(
        config.traffic_light_violation,
        log_tl.to(dtype),
        sim_tl.to(dtype),
    )
    tl_likelihood = torch.exp(tl_score.mean())

    # --- metametric = Σ weight_i * likelihood_i (proto 의 metametric_weight) ---
    lik = {
        "linear_speed_likelihood": linear_speed_likelihood.item(),
        "linear_acceleration_likelihood": linear_accel_likelihood.item(),
        "angular_speed_likelihood": angular_speed_likelihood.item(),
        "angular_acceleration_likelihood": angular_accel_likelihood.item(),
        "distance_to_nearest_object_likelihood": distance_to_obj_likelihood.item(),
        "collision_indication_likelihood": collision_likelihood.item(),
        "time_to_collision_likelihood": ttc_likelihood.item(),
        "distance_to_road_edge_likelihood": distance_to_road_edge_likelihood.item(),
        "offroad_indication_likelihood": offroad_likelihood.item(),
        "traffic_light_violation_likelihood": tl_likelihood.item(),
    }

    metametric = sum(
        getattr(config, fn).metametric_weight * lik[f"{fn}_likelihood"]
        for fn in _METRIC_FIELD_NAMES
    )

    return WosacMetametricTorchResult(
        metametric=metametric,
        linear_speed_likelihood=lik["linear_speed_likelihood"],
        linear_acceleration_likelihood=lik["linear_acceleration_likelihood"],
        angular_speed_likelihood=lik["angular_speed_likelihood"],
        angular_acceleration_likelihood=lik["angular_acceleration_likelihood"],
        distance_to_nearest_object_likelihood=lik["distance_to_nearest_object_likelihood"],
        collision_indication_likelihood=lik["collision_indication_likelihood"],
        time_to_collision_likelihood=lik["time_to_collision_likelihood"],
        distance_to_road_edge_likelihood=lik["distance_to_road_edge_likelihood"],
        offroad_indication_likelihood=lik["offroad_indication_likelihood"],
        traffic_light_violation_likelihood=lik["traffic_light_violation_likelihood"],
    )


__all__ = [
    "WosacMetametricTorchResult",
    "compute_kinematic_validity",
    "compute_wosac_metametric_from_features_torch",
    "histogram_estimate_torch",
    "log_likelihood_estimate_scenario_level_torch",
    "log_likelihood_estimate_timeseries_torch",
]
