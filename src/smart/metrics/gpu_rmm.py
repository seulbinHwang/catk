"""GPU-accelerated reimplementation of Waymo's Realism Meta Metric (RMM).

**Likelihood estimators** (histogram / Bernoulli) follow
``wdl_limited.sim_agents_metrics.estimators.histogram_estimate`` (categorical
bin probabilities, per-bin pseudocount — not PDF/bin-width).

**Feature tensors** (map, interaction geometry, TTC, traffic lights) are still
approximations vs ``metric_features.compute_scenario_rollouts_features``; hitting
leaderboard-grade numeric parity (e.g. |Δmetametric| < 0.01) requires either
calling the official TensorFlow pipeline or porting those modules line-for-line.

Official metametric = weighted sum of 10 likelihood scores:
    linear_speed            w=0.05  histogram  independent_timesteps=True
    linear_acceleration     w=0.05  histogram  independent_timesteps=True
    angular_speed           w=0.05  histogram  independent_timesteps=True
    angular_acceleration    w=0.05  histogram  independent_timesteps=True
    distance_to_nearest_obj w=0.10  histogram  independent_timesteps=True
    collision_indication    w=0.25  bernoulli  (any collision over trajectory)
    time_to_collision       w=0.10  histogram  independent_timesteps=True
    distance_to_road_edge   w=0.05  histogram  independent_timesteps=True
    offroad_indication      w=0.25  bernoulli  (any offroad step)
    traffic_light_violation w=0.05  bernoulli  (vehicles only — set to 0 here)

References:
    Waymo Open Dataset challenge_2025_sim_agents_config.textproto (default)
    waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Config — parsed from official *.textproto (falls back if Waymo unavailable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RMMConfig:
    """`SimAgentMetricsConfig`: metametric weights + histogram bins."""

    weights: Dict[str, float]
    hist: Dict[str, Tuple[float, float, int, float]]
    #: `FeatureConfig.bernoulli.additive_smoothing_pseudocount` per field (0 if unset).
    bernoulli_smooth: Dict[str, float] = field(default_factory=dict)


def _fallback_rmm_config() -> RMMConfig:
    """challenge_2025_sim_agents_config (matches pip waymo_open_dataset)."""
    return RMMConfig(
        weights={
            "linear_speed": 0.05,
            "linear_acceleration": 0.05,
            "angular_speed": 0.05,
            "angular_acceleration": 0.05,
            "distance_to_nearest_object": 0.1,
            "collision_indication": 0.25,
            "time_to_collision": 0.1,
            "distance_to_road_edge": 0.05,
            "offroad_indication": 0.25,
            "traffic_light_violation": 0.05,
        },
        hist={
            "linear_speed": (0.0, 25.0, 10, 0.1),
            "linear_acceleration": (-12.0, 12.0, 11, 0.1),
            "angular_speed": (-0.628, 0.628, 11, 0.1),
            "angular_acceleration": (-3.14, 3.14, 11, 0.1),
            "distance_to_nearest_object": (-5.0, 40.0, 10, 0.1),
            "time_to_collision": (0.0, 5.0, 10, 0.1),
            "distance_to_road_edge": (-20.0, 40.0, 10, 0.1),
        },
        bernoulli_smooth={
            "collision_indication": 0.0,
            "offroad_indication": 0.0,
            "traffic_light_violation": 0.0,
        },
    )


def _rmm_config_from_proto(cfg: object) -> RMMConfig:
    weights: Dict[str, float] = {}
    hist: Dict[str, Tuple[float, float, int, float]] = {}
    bernoulli_smooth: Dict[str, float] = {}
    for field_desc, fc in cfg.ListFields():
        name: str = field_desc.name
        weights[name] = float(fc.metametric_weight)
        if fc.HasField("histogram"):
            h = fc.histogram
            hist[name] = (
                float(h.min_val),
                float(h.max_val),
                int(h.num_bins),
                float(h.additive_smoothing_pseudocount),
            )
        if fc.HasField("bernoulli"):
            bernoulli_smooth[name] = float(fc.bernoulli.additive_smoothing_pseudocount)
    return RMMConfig(weights=weights, hist=hist, bernoulli_smooth=bernoulli_smooth)


@lru_cache(maxsize=4)
def get_rmm_config(config: str = "challenge_2025_sim_agents") -> RMMConfig:
    """Load official textproto.

    Args:
        config: ``challenge_2025_sim_agents`` | ``challenge_2024``
    """
    try:
        import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as wm
        from google.protobuf import text_format
        from waymo_open_dataset.protos import sim_agents_metrics_pb2

        base = Path(wm.__file__).resolve().parent
        if config == "challenge_2024":
            path = base / "challenge_2024_config.textproto"
        else:
            path = base / "challenge_2025_sim_agents_config.textproto"
            if not path.is_file():
                path = base / "challenge_2024_config.textproto"
        if not path.is_file():
            return _fallback_rmm_config()
        cfg = sim_agents_metrics_pb2.SimAgentMetricsConfig()
        with open(path, "r") as f:
            text_format.Parse(f.read(), cfg)
        return _rmm_config_from_proto(cfg)
    except Exception:
        return _fallback_rmm_config()

# interaction_features.py — rounded-box core shrink
_CORNER_ROUNDING_FACTOR = 0.7

# Lane half-width assumption when only lane polylines exist (no road_edge proto)
_LANE_HALF_WIDTH = 2.0

# interaction_features.py — time-to-collision
_MAXIMUM_TIME_TO_COLLISION = 5.0
_TTC_MAX_HEADING_DIFF = math.radians(75.0)
_TTC_MAX_HEADING_SMALL_OVERLAP = math.radians(10.0)
_TTC_SMALL_OVERLAP_THRESHOLD = 0.5
_EXTREMELY_LARGE = 1e10

# submission_specs.SIM_AGENTS — metric time window starts after this index
_CURRENT_TIME_INDEX = 10


# ---------------------------------------------------------------------------
# Histogram / Bernoulli — match `estimators.histogram_estimate` (TFP Categorical)
# ---------------------------------------------------------------------------


def _histogram_log_likelihood(
    sim_values: Tensor,
    log_values: Tensor,
    log_valid: Tensor,
    hist_min: float,
    hist_max: float,
    n_bins: int,
    additive_smoothing_pseudocount: float,
) -> Tensor:
    """Per-(agent, time) log p under a smoothed empirical histogram.

    Mirrors Waymo ``histogram_estimate`` + agent-level timeseries layout:
    edges are ``linspace(min_val, max_val, num_bins+1)`` (uniform bins),
    samples are clipped to ``[min_val, max_val]``, counts get
    ``+ additive_smoothing_pseudocount`` **per bin**, then row-normalized to
    probabilities; log-likelihood is ``log P(bin(log))``, **not** density.
    """
    dtype = sim_values.dtype
    device = sim_values.device
    n_agents, n_sim = sim_values.shape
    _, T = log_values.shape
    eps = torch.finfo(dtype).tiny

    bin_width = (hist_max - hist_min) / n_bins
    sim_x = sim_values.clamp(hist_min, hist_max)
    log_x = log_values.clamp(hist_min, hist_max)
    sim_bin = ((sim_x - hist_min) / bin_width).floor().long().clamp(0, n_bins - 1)
    log_bin = ((log_x - hist_min) / bin_width).floor().long().clamp(0, n_bins - 1)

    sim_onehot = F.one_hot(sim_bin, num_classes=n_bins).to(dtype)
    counts = sim_onehot.sum(dim=1)
    smoothed = counts + additive_smoothing_pseudocount
    denom = smoothed.sum(dim=1, keepdim=True).clamp_min(eps)
    probs = smoothed / denom
    log_p_bin = torch.log(probs.clamp_min(eps))

    log_ll = log_p_bin.gather(1, log_bin)
    return log_ll.masked_fill(~log_valid, float("-inf"))


def _bernoulli_log_likelihood(
    sim_values: Tensor,
    log_values: Tensor,
    log_valid: Tensor,
    additive_smoothing_pseudocount: float,
) -> Tensor:
    """Same as Waymo: ``bernoulli_estimate`` → 2-bin histogram on [-0.5, 1.5]."""
    dtype = sim_values.dtype
    device = sim_values.device
    n_agents, n_samples = sim_values.shape
    eps = torch.finfo(dtype).tiny

    hist_min, hist_max = -0.5, 1.5
    n_bins = 2
    bin_width = 1.0

    sim_x = sim_values.clamp(hist_min, hist_max)
    log_x = log_values.clamp(hist_min, hist_max)
    sim_bin = ((sim_x - hist_min) / bin_width).floor().long().clamp(0, n_bins - 1)
    log_bin = ((log_x - hist_min) / bin_width).floor().long().clamp(0, n_bins - 1)

    sim_onehot = F.one_hot(sim_bin, num_classes=n_bins).to(dtype)
    counts = sim_onehot.sum(dim=1)
    smoothed = counts + additive_smoothing_pseudocount
    denom = smoothed.sum(dim=1, keepdim=True).clamp_min(eps)
    probs = smoothed / denom
    log_p_bin = torch.log(probs.clamp_min(eps))
    log_ll = log_p_bin.gather(1, log_bin.unsqueeze(1)).squeeze(1)
    return log_ll.masked_fill(~log_valid, float("-inf"))


# ---------------------------------------------------------------------------
# Reduction: mean of valid log-likelihoods → exp → scalar likelihood
# ---------------------------------------------------------------------------

def _reduce_to_likelihood(log_ll: Tensor, valid: Tensor) -> Tensor:
    """Compute mean log-likelihood over valid positions, return exp(mean).

    Matches Waymo's `_reduce_average_with_validity`.

    Args:
        log_ll: shape [n_agents, T] or [n_agents].
        valid:  bool mask, same shape as log_ll.

    Returns:
        Scalar tensor — the likelihood in [0, 1].
    """
    # Replace -inf / nan at invalid positions with 0 for summing
    ll_masked = log_ll.masked_fill(~valid | ~torch.isfinite(log_ll), 0.0)
    n_valid = valid.float().sum().clamp(min=1.0)
    mean_ll = ll_masked.sum() / n_valid
    return torch.exp(mean_ll)


def _histogram_likelihood_scores_per_rollout(
    sim_eval: Tensor,
    log_eval: Tensor,
    feat_valid_eval: Tensor,
    h_min: float,
    h_max: float,
    n_bins: int,
    smooth: float,
) -> Tensor:
    """Per-rollout likelihoods aligned with one Waymo ``JointScene`` (G_sim=1).

    Builds a separate histogram from each rollout's ``T`` timestep samples, then
    scores the same GT reference (`log_eval`) — matching
    ``compute_scenario_metrics_for_bundle`` when the submission has a single
    rollout.
    """
    _n_eval, G, T = sim_eval.shape
    sim_flat = sim_eval.permute(1, 0, 2).reshape(G * _n_eval, T)
    log_flat = log_eval.unsqueeze(0).expand(G, -1, -1).reshape(G * _n_eval, T)
    valid_flat = feat_valid_eval.unsqueeze(0).expand(G, -1, -1).reshape(G * _n_eval, T)
    log_ll = _histogram_log_likelihood(
        sim_flat, log_flat, valid_flat, h_min, h_max, n_bins, smooth
    )
    log_ll = log_ll.view(G, _n_eval, T)
    return torch.stack(
        [_reduce_to_likelihood(log_ll[g], feat_valid_eval) for g in range(G)],
        dim=0,
    )


def _bernoulli_likelihood_scores_per_rollout(
    sim_trajectory_flag: Tensor,
    log_flag: Tensor,
    agent_valid: Tensor,
    additive_smoothing_pseudocount: float,
) -> Tensor:
    """``sim_trajectory_flag``: [n_eval, G] (0/1 per rollout). One Bernoulli sample per rollout.

    Matches official per-rollout evaluation: histogram uses one simulated
    aggregate (any-collision / any-offroad) per rollout, not pooled across G.
    """
    _n_eval, G = sim_trajectory_flag.shape
    sim_flat = sim_trajectory_flag.transpose(0, 1).reshape(G * _n_eval, 1)
    log_flat = log_flag.unsqueeze(0).expand(G, -1).reshape(G * _n_eval)
    val_flat = agent_valid.unsqueeze(0).expand(G, -1).reshape(G * _n_eval)
    log_ll = _bernoulli_log_likelihood(
        sim_flat, log_flat, val_flat, additive_smoothing_pseudocount
    )
    log_ll = log_ll.view(G, _n_eval)
    out = []
    for g in range(G):
        out.append(
            _reduce_to_likelihood(
                log_ll[g].unsqueeze(-1), agent_valid.unsqueeze(-1)
            )
        )
    return torch.stack(out, dim=0)


# ---------------------------------------------------------------------------
# Kinematic feature extraction
# ---------------------------------------------------------------------------

def _central_diff(t: Tensor, pad_value: float = float("nan")) -> Tensor:
    """Central difference: (t[..., i+1] - t[..., i-1]) / 2.

    Returns a tensor with the same shape as t, with NaN padding at both ends.
    """
    pad = torch.full(
        (*t.shape[:-1], 1),
        pad_value,
        dtype=t.dtype,
        device=t.device,
    )
    diff = (t[..., 2:] - t[..., :-2]) / 2.0
    return torch.cat([pad, diff, pad], dim=-1)


def _wrap_angle(angle: Tensor) -> Tensor:
    """Wrap angle to [-pi, pi]."""
    return -math.pi + (angle + math.pi) % (2 * math.pi)


def _compute_kinematic_features(
    xy: Tensor,       # [n_agents, G, T, 2]
    z: Tensor,        # [n_agents, G, T]
    heading: Tensor,  # [n_agents, G, T]
    valid: Tensor,    # [n_agents, T] bool  (same for all rollouts)
    dt: float,
) -> dict:
    """Compute per-timestep kinematic features exactly matching Waymo's formula.

    Args:
        xy:      world XY positions, shape [n_agents, G, T, 2].
        z:       world Z positions, shape [n_agents, G, T].
        heading: heading in radians, shape [n_agents, G, T].
        valid:   validity mask, shape [n_agents, T].
        dt:      time step in seconds.

    Returns:
        Dictionary with tensors of shape [n_agents, G, T]:
            linear_speed, linear_accel, angular_speed, angular_accel
        and validity masks of shape [n_agents, T]:
            speed_valid, accel_valid
    """
    # xyz: [n_agents, G, T, 3]
    xyz = torch.cat([xy, z.unsqueeze(-1)], dim=-1)

    # Central diff of XYZ along T axis
    dx = _central_diff(xyz[..., 0])   # [n_agents, G, T]
    dy = _central_diff(xyz[..., 1])
    dz = _central_diff(xyz[..., 2])

    # Linear speed: norm of (dx, dy, dz) / dt
    speed = torch.sqrt(dx**2 + dy**2 + dz**2) / dt   # [n_agents, G, T]

    # Linear acceleration: central diff of speed / dt
    accel = _central_diff(speed) / dt   # [n_agents, G, T]

    # Angular speed from heading using double-wrapping trick
    # wrap(central_diff(heading) * 2) / 2  / dt
    dh = _central_diff(heading)
    angular_speed = _wrap_angle(dh * 2.0) / 2.0 / dt   # [n_agents, G, T]

    # Angular acceleration: wrap(central_diff(angular_speed * dt) * 2) / 2 / dt^2
    dw = _central_diff(angular_speed * dt)
    angular_accel = _wrap_angle(dw * 2.0) / 2.0 / (dt ** 2)   # [n_agents, G, T]

    # Validity masks (same for all rollouts — derived from GT valid)
    # speed valid: valid[i-1] AND valid[i+1]  (central diff of position)
    v = valid.float()   # [n_agents, T]
    # pad with zeros on both ends
    v_pad = F.pad(v, (1, 1), value=0.0)   # [n_agents, T+2]
    speed_valid = (v_pad[:, :-2] * v_pad[:, 2:]) > 0.5   # [n_agents, T]

    # accel valid: valid[i-2] AND valid[i] AND valid[i+2]  (central diff of speed)
    v2_pad = F.pad(v, (2, 2), value=0.0)   # [n_agents, T+4]
    accel_valid = (v2_pad[:, :-4] * v2_pad[:, 2:-2] * v2_pad[:, 4:]) > 0.5   # [n_agents, T]

    return {
        "linear_speed":    speed,
        "linear_accel":    accel,
        "angular_speed":   angular_speed,
        "angular_accel":   angular_accel,
        "speed_valid":     speed_valid,   # [n_agents, T]
        "accel_valid":     accel_valid,   # [n_agents, T]
    }


# ---------------------------------------------------------------------------
# OBB 2-D distance with corner rounding (Waymo formula)
# ---------------------------------------------------------------------------

def _obb_core_shrink(length: Tensor, width: Tensor) -> Tensor:
    """Compute per-agent shrink amount for core OBB.

    shrink = corner_rounding_factor * min(length, width) / 2
    """
    return _CORNER_ROUNDING_FACTOR * torch.minimum(length, width) / 2.0


def _point_to_obb_distance_2d(
    points: Tensor,       # [..., 2]  query points in local frame
    half_l: Tensor,       # [...] half-lengths of OBB
    half_w: Tensor,       # [...] half-widths of OBB
) -> Tensor:
    """Signed 2-D distance from a point to an axis-aligned OBB (in OBB local frame).

    Negative means the point is inside the box.
    """
    # Transform to box frame (box centred at origin, axes aligned)
    q = points.abs() - torch.stack([half_l, half_w], dim=-1)   # [..., 2]
    # Signed distance: outside = positive, inside = negative
    outside = torch.norm(q.clamp(min=0.0), dim=-1)
    inside = q.max(dim=-1).values.clamp(max=0.0)
    return outside + inside


def _obb_to_obb_distance_2d_vectorised(
    pos_i: Tensor,    # [*batch, 2]  position of agent i (world)
    head_i: Tensor,   # [*batch]     heading of agent i
    hl_i: Tensor,     # [*batch]     half-length of agent i (after shrink)
    hw_i: Tensor,     # [*batch]     half-width  of agent i (after shrink)
    pos_j: Tensor,    # [*batch, 2]  position of agent j (world)
    head_j: Tensor,   # [*batch]     heading of agent j
    hl_j: Tensor,     # [*batch]     half-length of agent j (after shrink)
    hw_j: Tensor,     # [*batch]     half-width  of agent j (after shrink)
) -> Tensor:
    """Approximate 2-D OBB–OBB distance using the GJK-lite approach.

    Strategy: compute distance from box-i's corners to box-j, and from
    box-j's corners to box-i, take the minimum. This over-estimates
    slightly but is differentiable and fast on GPU.

    Returns signed distance (negative = overlap).
    """
    device = pos_i.device
    dtype = pos_i.dtype
    batch_shape = pos_i.shape[:-1]

    # Build corners of box i in world coordinates
    cos_i, sin_i = head_i.cos(), head_i.sin()
    cos_j, sin_j = head_j.cos(), head_j.sin()

    def _box_corners(pos, cos, sin, hl, hw):
        # Local corners: (±hl, ±hw)
        lx = torch.stack([hl, hl, -hl, -hl], dim=-1)   # [*batch, 4]
        ly = torch.stack([hw, -hw, -hw, hw], dim=-1)
        # Rotate to world
        wx = pos[..., 0:1] + lx * cos.unsqueeze(-1) - ly * sin.unsqueeze(-1)
        wy = pos[..., 1:2] + lx * sin.unsqueeze(-1) + ly * cos.unsqueeze(-1)
        return torch.stack([wx, wy], dim=-1)   # [*batch, 4, 2]

    corners_i = _box_corners(pos_i, cos_i, sin_i, hl_i, hw_i)   # [*batch, 4, 2]
    corners_j = _box_corners(pos_j, cos_j, sin_j, hl_j, hw_j)

    def _to_local_frame(world_pts, ref_pos, ref_cos, ref_sin):
        # Transform world_pts into ref box local frame
        d = world_pts - ref_pos.unsqueeze(-2)   # [*batch, N, 2]
        lx = d[..., 0] * ref_cos.unsqueeze(-1) + d[..., 1] * ref_sin.unsqueeze(-1)
        ly = -d[..., 0] * ref_sin.unsqueeze(-1) + d[..., 1] * ref_cos.unsqueeze(-1)
        return torch.stack([lx, ly], dim=-1)    # [*batch, N, 2]

    # Distance from corners of i in box-j frame
    ci_in_j = _to_local_frame(corners_i, pos_j, cos_j, sin_j)   # [*batch, 4, 2]
    d_ci_to_j = _point_to_obb_distance_2d(
        ci_in_j,
        hl_j.unsqueeze(-1).expand(*batch_shape, 4),
        hw_j.unsqueeze(-1).expand(*batch_shape, 4),
    )  # [*batch, 4]

    # Distance from corners of j in box-i frame
    cj_in_i = _to_local_frame(corners_j, pos_i, cos_i, sin_i)   # [*batch, 4, 2]
    d_cj_to_i = _point_to_obb_distance_2d(
        cj_in_i,
        hl_i.unsqueeze(-1).expand(*batch_shape, 4),
        hw_i.unsqueeze(-1).expand(*batch_shape, 4),
    )  # [*batch, 4]

    # Min over corners — if any corner is inside, distance is negative
    d_ij = torch.cat([d_ci_to_j, d_cj_to_i], dim=-1).min(dim=-1).values   # [*batch]
    return d_ij


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

def _compute_distance_to_nearest_object(
    sim_xy: Tensor,       # [n_agents, G, T, 2]
    sim_head: Tensor,     # [n_agents, G, T]
    agent_shape: Tensor,  # [n_agents, 3]  (length, width, height)
    valid: Tensor,        # [n_agents, T] bool
    eval_mask: Tensor,    # [n_agents] bool
) -> Tuple[Tensor, Tensor]:
    """Compute distance-to-nearest-object and collision indicator per step.

    Uses 2-D rounded-rectangle (shrunk OBB) distances as in Waymo.

    Args:
        sim_xy:      positions, shape [n_agents, G, T, 2].
        sim_head:    headings,  shape [n_agents, G, T].
        agent_shape: (length, width, height), shape [n_agents, 3].
        valid:       GT validity, shape [n_agents, T].
        eval_mask:   which agents to compute metric for, shape [n_agents].

    Returns:
        dist_to_nearest: [n_eval, G, T]  — min distance to other agents
        collision_per_step: [n_eval, G, T]  — bool, True = collision
    """
    device = sim_xy.device
    dtype = sim_xy.dtype
    n_agents, G, T, _ = sim_xy.shape

    length = agent_shape[:, 0]   # [n_agents]
    width  = agent_shape[:, 1]   # [n_agents]

    shrink = _obb_core_shrink(length, width)   # [n_agents]
    # Shrunk half-extents
    hl = (length / 2.0 - shrink).clamp(min=1e-3)   # [n_agents]
    hw = (width  / 2.0 - shrink).clamp(min=1e-3)

    eval_idx = torch.where(eval_mask)[0]   # [n_eval]
    n_eval = eval_idx.shape[0]

    # All agent valid at each step: [n_agents, T]
    # We need other agents to be valid for them to be obstacles
    other_mask = valid   # [n_agents, T] — step-wise validity of obstacles

    if n_eval == 0:
        empty = torch.full((0, G, T), float("inf"), device=device, dtype=dtype)
        return empty, empty < 0

    # Pre-gather evaluated agents
    eval_xy   = sim_xy[eval_idx]     # [n_eval, G, T, 2]
    eval_head = sim_head[eval_idx]   # [n_eval, G, T]
    eval_hl   = hl[eval_idx]         # [n_eval]
    eval_hw   = hw[eval_idx]         # [n_eval]
    eval_shr  = shrink[eval_idx]     # [n_eval]

    # We iterate over obstacles in chunks to avoid O(n²) memory explosion
    # For each evaluated agent, compute distance to every other agent
    min_dist = torch.full((n_eval, G, T), float("inf"), device=device, dtype=dtype)

    for j in range(n_agents):
        if n_eval > 0 and eval_idx.numel() > 0:
            # Skip: evaluated agent against itself
            is_self = (eval_idx == j)   # [n_eval]
            if is_self.all():
                continue

        # obstacle validity at each step: [T]
        obs_valid_t = other_mask[j]   # [T]
        # Only compute where obstacle is valid
        if not obs_valid_t.any():
            continue

        obs_xy   = sim_xy[j]    # [G, T, 2]
        obs_head = sim_head[j]  # [G, T]
        obs_hl   = hl[j]        # scalar
        obs_hw   = hw[j]
        obs_shr  = shrink[j]

        # Broadcast to [n_eval, G, T]
        # eval:   [n_eval, G, T, 2],  obs: [G, T, 2] → broadcast to [n_eval, G, T, 2]
        ei_xy   = eval_xy                                    # [n_eval, G, T, 2]
        ej_xy   = obs_xy.unsqueeze(0).expand(n_eval, -1, -1, -1)   # [n_eval, G, T, 2]
        ei_head = eval_head                                  # [n_eval, G, T]
        ej_head = obs_head.unsqueeze(0).expand(n_eval, -1, -1)
        ei_hl   = eval_hl.view(n_eval, 1, 1).expand(n_eval, G, T)
        ei_hw   = eval_hw.view(n_eval, 1, 1).expand(n_eval, G, T)
        ej_hl   = obs_hl.view(1, 1, 1).expand(n_eval, G, T)
        ej_hw   = obs_hw.view(1, 1, 1).expand(n_eval, G, T)

        # Core OBB distance
        core_dist = _obb_to_obb_distance_2d_vectorised(
            ei_xy.reshape(-1, 2), ei_head.reshape(-1),
            ei_hl.reshape(-1),    ei_hw.reshape(-1),
            ej_xy.reshape(-1, 2), ej_head.reshape(-1),
            ej_hl.reshape(-1),    ej_hw.reshape(-1),
        ).reshape(n_eval, G, T)   # [n_eval, G, T]

        # Add back rounding: total_dist = core_dist - shrink_i - shrink_j
        total_dist = core_dist - eval_shr.view(n_eval, 1, 1) - obs_shr

        # Mask invalid obstacle steps
        obs_invalid_t = ~obs_valid_t.view(1, 1, T)   # [1, 1, T]
        total_dist = total_dist.masked_fill(obs_invalid_t, float("inf"))

        # Mask self (where eval_idx == j)
        is_self = (eval_idx == j)   # [n_eval]
        total_dist = total_dist.masked_fill(
            is_self.view(n_eval, 1, 1), float("inf")
        )

        min_dist = torch.minimum(min_dist, total_dist)

    collision_per_step = min_dist < 0.0   # [n_eval, G, T]
    return min_dist, collision_per_step


def _rotate_2d_points_torch(xys: Tensor, rotation_yaws: Tensor) -> Tensor:
    """``geometry_utils.rotate_2d_points`` — CCW. ``yaws`` broadcastable with ``xys[..., 0]``."""
    if rotation_yaws.shape[-1] == 1:
        rotation_yaws = rotation_yaws.squeeze(-1)
    c = torch.cos(rotation_yaws)
    s = torch.sin(rotation_yaws)
    x, y = xys[..., 0], xys[..., 1]
    return torch.stack([c * x - s * y, s * x + c * y], dim=-1)


def _planar_speed_central(cx: Tensor, cy: Tensor, dt: float) -> Tensor:
    """Speed norm with z=0; matches Waymo TTC auxiliary kinematics."""
    z = torch.zeros_like(cx)
    dx = _central_diff(cx)
    dy = _central_diff(cy)
    dz = _central_diff(z)
    return torch.sqrt(dx * dx + dy * dy + dz * dz) / dt


def _compute_time_to_collision(
    sim_xy: Tensor,
    sim_head: Tensor,
    agent_shape: Tensor,
    valid: Tensor,
    eval_mask: Tensor,
    dt: float,
) -> Tensor:
    """``interaction_features.compute_time_to_collision_with_object_in_front``."""
    device = sim_xy.device
    dtype = sim_xy.dtype
    n_agents, G, T, _ = sim_xy.shape
    eval_idx = torch.where(eval_mask)[0]
    n_eval = eval_idx.shape[0]
    if n_eval == 0:
        return torch.full(
            (0, G, T),
            _MAXIMUM_TIME_TO_COLLISION,
            device=device,
            dtype=dtype,
        )

    Len = agent_shape[:, 0]
    Wid = agent_shape[:, 1]
    rollouts = []
    for g in range(G):
        cx = sim_xy[:, g, :, 0]
        cy = sim_xy[:, g, :, 1]
        heading = sim_head[:, g, :]
        speed = _planar_speed_central(cx, cy, dt)
        Lexp = Len.unsqueeze(-1).expand(-1, T)
        Wexp = Wid.unsqueeze(-1).expand(-1, T)
        boxes = torch.stack([cx, cy, Lexp, Wexp, heading, speed], dim=-1)
        boxes = boxes.transpose(0, 1).contiguous()
        val_t = valid.transpose(0, 1)
        Te, n_obj, _ = boxes.shape

        eval_boxes = boxes[:, eval_idx, :]
        ego_xy, ego_sizes, ego_yaw, ego_speed = torch.split(
            eval_boxes, [2, 2, 1, 1], dim=-1
        )
        other_xy, other_sizes, other_yaw, _ = torch.split(
            boxes, [2, 2, 1, 1], dim=-1
        )

        yaw_diff = (other_yaw.unsqueeze(1) - ego_yaw.unsqueeze(2)).abs()
        yc = torch.cos(yaw_diff)
        ys = torch.sin(yaw_diff)
        o_half = other_sizes[:, None, :, :] / 2.0
        other_long_offset = (o_half * torch.cat([yc, ys], dim=-1).abs()).sum(
            dim=-1
        )
        other_lat_offset = (o_half * torch.cat([ys, yc], dim=-1).abs()).sum(
            dim=-1
        )

        rel = other_xy[:, None, :, :] - ego_xy[:, :, None, :]
        # ``ego_yaw``: [Te, n_eval, 1] — expand to each ``n_obj`` for broadcast.
        n_o = rel.shape[2]
        yaw_expand = (-ego_yaw).unsqueeze(2).expand(-1, -1, n_o, -1)
        rel_r = _rotate_2d_points_torch(rel, yaw_expand)

        long_distance = (
            rel_r[..., 0]
            - ego_sizes[:, :, None, 0] / 2.0
            - other_long_offset
        )
        lat_overlap = (
            rel_r[..., 1].abs()
            - ego_sizes[:, :, None, 1] / 2.0
            - other_lat_offset
        )

        fmask = long_distance > 0
        fmask = fmask & (yaw_diff.squeeze(-1) <= _TTC_MAX_HEADING_DIFF)
        fmask = fmask & (lat_overlap < 0)
        fmask = fmask & (
            (lat_overlap < -_TTC_SMALL_OVERLAP_THRESHOLD)
            | (yaw_diff.squeeze(-1) <= _TTC_MAX_HEADING_SMALL_OVERLAP)
        )
        v_eval = val_t[:, eval_idx]
        fmask = fmask & v_eval[:, :, None] & val_t[:, None, :]
        self_ok = eval_idx.view(1, n_eval, 1) != torch.arange(
            n_obj, device=device, dtype=torch.long
        ).view(1, 1, n_obj)
        fmask = fmask & self_ok

        masked_ld = long_distance + (~fmask).to(dtype) * _EXTREMELY_LARGE
        idx = masked_ld.argmin(dim=-1)
        dist_ahead = torch.gather(masked_ld, 2, idx.unsqueeze(-1)).squeeze(-1)

        t_ix = torch.arange(Te, device=device).unsqueeze(1).expand(Te, n_eval)
        sp_ahead = boxes[t_ix, idx, 5]

        rel_sp = ego_speed.squeeze(-1) - sp_ahead
        ttc_t = torch.where(
            rel_sp > 0.0,
            torch.minimum(
                dist_ahead / rel_sp.clamp(min=1e-8),
                torch.full_like(rel_sp, _MAXIMUM_TIME_TO_COLLISION),
            ),
            torch.full_like(rel_sp, _MAXIMUM_TIME_TO_COLLISION),
        )
        rollouts.append(ttc_t.transpose(0, 1))

    return torch.stack(rollouts, dim=1)


# ---------------------------------------------------------------------------
# Map-based features (distance to road edge approximation)
# ---------------------------------------------------------------------------

def _point_to_segment_distance_2d(
    pts: Tensor,   # [..., 2]  query points
    seg_a: Tensor, # [..., 2]  segment start
    seg_b: Tensor, # [..., 2]  segment end
) -> Tensor:
    """Compute distance from points to line segments."""
    ab = seg_b - seg_a                          # [..., 2]
    ap = pts - seg_a                            # [..., 2]
    ab_len2 = (ab ** 2).sum(dim=-1).clamp(min=1e-8)   # [...]
    t = (ap * ab).sum(dim=-1) / ab_len2         # [...]
    t = t.clamp(0.0, 1.0)
    closest = seg_a + t.unsqueeze(-1) * ab      # [..., 2]
    return torch.norm(pts - closest, dim=-1)    # [...]


def _compute_distance_to_road_edge(
    sim_xy: Tensor,        # [n_agents, G, T, 2]
    map_token_pos: Tensor, # [n_map, 3, 2]  (start, mid, end per token)
    valid: Tensor,         # [n_agents, T] bool
    eval_mask: Tensor,     # [n_agents] bool
) -> Tuple[Tensor, Tensor]:
    """Approximate distance to road edge from lane token positions.

    Approximation (same sign as Waymo ``map_metric_features``):
        signed_distance ≈ dist_to_lane_reference − lane_half_width
        Negative ⇒ inside / on drivable; **positive ⇒ off-road**.

    Args:
        sim_xy:        positions, shape [n_agents, G, T, 2].
        map_token_pos: map token positions, shape [n_map, 3, 2].
        valid:         GT validity, shape [n_agents, T].
        eval_mask:     which agents to compute metric for, shape [n_agents].

    Returns:
        dist_to_road_edge: [n_eval, G, T]
        offroad_per_step:  [n_eval, G, T] bool
    """
    device = sim_xy.device
    dtype = sim_xy.dtype
    n_agents, G, T, _ = sim_xy.shape

    eval_idx = torch.where(eval_mask)[0]
    n_eval = eval_idx.shape[0]

    if n_eval == 0:
        empty = torch.full((0, G, T), 0.0, device=device, dtype=dtype)
        return empty, empty > 0

    eval_xy = sim_xy[eval_idx]   # [n_eval, G, T, 2]

    if map_token_pos is None or map_token_pos.shape[0] == 0:
        # No map — neutral inside (negative signed distance).
        dist = torch.full(
            (n_eval, G, T), -_LANE_HALF_WIDTH, device=device, dtype=dtype
        )
        return dist, dist > 0

    n_map = map_token_pos.shape[0]
    map_token_pos = map_token_pos.to(device=device, dtype=dtype)

    # Use start-mid and mid-end segments (2 segments per token)
    seg_a = torch.cat([map_token_pos[:, 0], map_token_pos[:, 1]], dim=0)   # [2*n_map, 2]
    seg_b = torch.cat([map_token_pos[:, 1], map_token_pos[:, 2]], dim=0)   # [2*n_map, 2]
    n_seg = seg_a.shape[0]

    # Reshape eval_xy for vectorised distance computation
    # Process in chunks over map segments to bound memory
    flat_pts = eval_xy.reshape(-1, 2)   # [n_eval*G*T, 2]
    N = flat_pts.shape[0]

    chunk_size = 512   # segments per chunk
    min_dist = torch.full((N,), float("inf"), device=device, dtype=dtype)

    for s in range(0, n_seg, chunk_size):
        e = min(s + chunk_size, n_seg)
        sa = seg_a[s:e]   # [C, 2]
        sb = seg_b[s:e]   # [C, 2]
        C = sa.shape[0]

        # Broadcast: pts [N, 1, 2], segs [1, C, 2] → [N, C, 2]
        d = _point_to_segment_distance_2d(
            flat_pts.unsqueeze(1).expand(-1, C, -1),   # [N, C, 2]
            sa.unsqueeze(0).expand(N, -1, -1),
            sb.unsqueeze(0).expand(N, -1, -1),
        )   # [N, C]
        min_dist = torch.minimum(min_dist, d.min(dim=-1).values)

    dist_to_edge = min_dist.reshape(n_eval, G, T) - _LANE_HALF_WIDTH

    # Mask invalid steps — treat as inside lane (negative margin).
    eval_valid = valid[eval_idx]   # [n_eval, T]
    dist_to_edge = dist_to_edge.masked_fill(
        ~eval_valid.unsqueeze(1).expand(n_eval, G, T),
        -_LANE_HALF_WIDTH,
    )

    offroad = dist_to_edge > 0.0   # [n_eval, G, T]
    return dist_to_edge, offroad


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_gpu_rmm(
    pred_traj: Tensor,       # [n_agents, G, 80, 2]  world XY, future only
    pred_z: Tensor,          # [n_agents, G, 80]
    pred_head: Tensor,       # [n_agents, G, 80]
    gt_position: Tensor,     # [n_agents, 91, 3]  full traj (11 hist + 80 fut), world
    gt_heading: Tensor,      # [n_agents, 91]
    gt_valid: Tensor,        # [n_agents, 91] bool
    agent_shape: Tensor,     # [n_agents, 3]  (length, width, height)
    agent_type: Tensor,      # [n_agents] uint8  (0=vehicle, 1=ped, 2=cyclist)
    eval_mask: Tensor,       # [n_agents] bool  (tracks_to_predict)
    map_token_pos: Tensor,   # [n_map, 3, 2]  (3 pts per token, from map_save.traj_pos)
    dt: float = 0.1,
    *,
    rmm_config: Optional[RMMConfig] = None,
    return_subscores: bool = False,
) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
    """Compute GPU-based approximation of the Waymo Realism Meta Metric (RMM).

    One scenario at a time; returns one weighted metametric score per rollout.

    Per-rollout histograms use ``T`` samples from that rollout only (same as
    official metrics when the submission has one ``JointScene``). Bernoulli
    terms use one aggregate draw per rollout (not pooled across G).

    Args:
        rmm_config: Parsed ``SimAgentMetricsConfig`` (default: challenge 2025).
        return_subscores: If True, return ``(rmm, dict)`` of per-term likelihoods [G].
    """
    cfg = rmm_config if rmm_config is not None else get_rmm_config()
    device = pred_traj.device
    dtype = pred_traj.dtype
    n_agents, G, T_fut, _ = pred_traj.shape
    T_hist = 11
    T = T_hist + T_fut   # 91
    # ``metric_features``: likelihood uses simulation steps
    # ``current_time_index + 1 :`` (here 11…90, not history 0…10).
    t_met = _CURRENT_TIME_INDEX + 1

    # -----------------------------------------------------------------------
    # 1. Construct full simulated trajectory: prepend GT history
    # -----------------------------------------------------------------------
    hist_xy  = gt_position[:, :T_hist, :2]   # [n_agents, 11, 2]
    hist_z   = gt_position[:, :T_hist, 2]    # [n_agents, 11]
    hist_h   = gt_heading[:, :T_hist]        # [n_agents, 11]

    # Expand history across G rollouts
    hist_xy_g = hist_xy.unsqueeze(1).expand(-1, G, -1, -1)   # [n_agents, G, 11, 2]
    hist_z_g  = hist_z.unsqueeze(1).expand(-1, G, -1)
    hist_h_g  = hist_h.unsqueeze(1).expand(-1, G, -1)

    # Full sim trajectory: [n_agents, G, 91, 2/1]
    sim_xy   = torch.cat([hist_xy_g, pred_traj], dim=2)    # [n_agents, G, 91, 2]
    sim_z    = torch.cat([hist_z_g, pred_z], dim=2)        # [n_agents, G, 91]
    sim_head = torch.cat([hist_h_g, pred_head], dim=2)     # [n_agents, G, 91]

    # Full GT trajectory for reference (same XY as history, GT future)
    gt_xy  = gt_position[:, :, :2]   # [n_agents, 91, 2]
    gt_z   = gt_position[:, :, 2]    # [n_agents, 91]
    gt_h   = gt_heading              # [n_agents, 91]
    gt_val = gt_valid                # [n_agents, 91] bool

    # -----------------------------------------------------------------------
    # 2. Kinematic features
    # -----------------------------------------------------------------------
    # sim:  [n_agents, G, T, *]
    # log (gt): [n_agents, T, *] — treat as [n_agents, 1, T, *] broadcast

    kin = _compute_kinematic_features(sim_xy, sim_z, sim_head, gt_val, dt)
    gt_kin = _compute_kinematic_features(
        gt_xy.unsqueeze(1),   # [n_agents, 1, T, 2]
        gt_z.unsqueeze(1),
        gt_h.unsqueeze(1),
        gt_val,
        dt,
    )

    # -----------------------------------------------------------------------
    # Per-metric likelihoods (eval agents only)
    # -----------------------------------------------------------------------
    eval_idx = torch.where(eval_mask)[0]
    n_eval = eval_idx.shape[0]
    T_met = T - t_met
    eval_valid_fut = (
        gt_val[eval_idx][:, t_met:]
        if n_eval > 0
        else torch.zeros(0, T_met, dtype=torch.bool, device=device)
    )

    def _kinematic_per_rollout(feature_key: str, valid_key: str, hist_field: str) -> Tensor:
        h_min, h_max, n_bins, smooth = cfg.hist[hist_field]
        sim_feat = torch.nan_to_num(
            kin[feature_key][:, :, t_met:], nan=h_min
        )
        log_feat = torch.nan_to_num(
            gt_kin[feature_key][:, 0, t_met:], nan=h_min
        )
        feat_valid_eval = kin[valid_key][eval_idx][:, t_met:]
        if n_eval == 0:
            return torch.ones(G, device=device, dtype=dtype)
        return _histogram_likelihood_scores_per_rollout(
            sim_feat[eval_idx],
            log_feat[eval_idx],
            feat_valid_eval,
            h_min,
            h_max,
            n_bins,
            smooth,
        )

    ls_scores = _kinematic_per_rollout("linear_speed", "speed_valid", "linear_speed")
    la_scores = _kinematic_per_rollout("linear_accel", "accel_valid", "linear_acceleration")
    as_scores = _kinematic_per_rollout("angular_speed", "speed_valid", "angular_speed")
    aa_scores = _kinematic_per_rollout("angular_accel", "accel_valid", "angular_acceleration")

    # -----------------------------------------------------------------------
    # 5. Distance to nearest object and collision
    # -----------------------------------------------------------------------
    dist_to_obj, collision_step = _compute_distance_to_nearest_object(
        sim_xy, sim_head, agent_shape, gt_val, eval_mask
    )   # [n_eval, G, T], [n_eval, G, T]

    if n_eval > 0:
        gt_dist_nearest, _ = _compute_distance_to_nearest_object(
            gt_xy.unsqueeze(1).expand(-1, 1, -1, -1),
            gt_h.unsqueeze(1).expand(-1, 1, -1),
            agent_shape,
            gt_val,
            eval_mask,
        )
        gt_dist_ref = gt_dist_nearest[:, 0, t_met:].clamp(-100.0, 200.0)
    else:
        gt_dist_ref = torch.zeros(0, T - t_met, device=device, dtype=dtype)

    h_min, h_max, n_bins, smooth = cfg.hist["distance_to_nearest_object"]
    if n_eval == 0:
        dno_scores = torch.ones(G, device=device, dtype=dtype)
    else:
        dno_scores = _histogram_likelihood_scores_per_rollout(
            dist_to_obj[:, :, t_met:].clamp(-100.0, 200.0),
            gt_dist_ref,
            eval_valid_fut,
            h_min,
            h_max,
            n_bins,
            smooth,
        )

    if n_eval > 0:
        col_any_sim = collision_step[:, :, t_met:].any(dim=-1).float()
        gt_col_any = (gt_dist_nearest[:, 0, t_met:] < 0).any(dim=-1).float()
        agent_valid = eval_valid_fut.any(dim=-1)
        col_ind_scores = _bernoulli_likelihood_scores_per_rollout(
            col_any_sim,
            gt_col_any,
            agent_valid,
            float(cfg.bernoulli_smooth.get("collision_indication", 0.0)),
        )
    else:
        col_ind_scores = torch.ones(G, device=device, dtype=dtype)

    # -----------------------------------------------------------------------
    # 6. Time to collision (histogram uses vehicles only, future steps)
    # -----------------------------------------------------------------------
    ttc = _compute_time_to_collision(
        sim_xy, sim_head, agent_shape, gt_val, eval_mask, dt
    )   # [n_eval, G, T]

    gt_ttc = (
        _compute_time_to_collision(
            gt_xy.unsqueeze(1).expand(-1, 1, -1, -1),
            gt_h.unsqueeze(1).expand(-1, 1, -1),
            agent_shape,
            gt_val,
            eval_mask,
            dt,
        )
        if n_eval > 0
        else torch.full(
            (0, 1, T),
            _MAXIMUM_TIME_TO_COLLISION,
            device=device,
            dtype=dtype,
        )
    )
    veh_eval = agent_type[eval_idx].unsqueeze(-1) == 0
    ttc_valid_fut = eval_valid_fut & veh_eval

    h_min, h_max, n_bins, smooth = cfg.hist["time_to_collision"]
    if n_eval == 0:
        ttc_scores = torch.ones(G, device=device, dtype=dtype)
    else:
        ttc_scores = _histogram_likelihood_scores_per_rollout(
            ttc[:, :, t_met:],
            gt_ttc[:, 0, t_met:],
            ttc_valid_fut,
            h_min,
            h_max,
            n_bins,
            smooth,
        )

    # -----------------------------------------------------------------------
    # 7. Distance to road edge and offroad
    # -----------------------------------------------------------------------
    dist_edge, offroad_step = _compute_distance_to_road_edge(
        sim_xy, map_token_pos, gt_val, eval_mask
    )   # [n_eval, G, T], [n_eval, G, T]

    gt_dist_edge, _ = _compute_distance_to_road_edge(
        gt_xy.unsqueeze(1).expand(-1, 1, -1, -1),
        map_token_pos, gt_val, eval_mask
    ) if n_eval > 0 else (
        torch.zeros(0, 1, T, device=device, dtype=dtype), None
    )

    h_min, h_max, n_bins, smooth = cfg.hist["distance_to_road_edge"]
    if n_eval == 0:
        edge_scores = torch.ones(G, device=device, dtype=dtype)
        offroad_scores = torch.ones(G, device=device, dtype=dtype)
    else:
        edge_scores = _histogram_likelihood_scores_per_rollout(
            dist_edge[:, :, t_met:].clamp(-100.0, 200.0),
            gt_dist_edge[:, 0, t_met:].clamp(-100.0, 200.0),
            eval_valid_fut,
            h_min,
            h_max,
            n_bins,
            smooth,
        )
        offroad_flag_sim = offroad_step[:, :, t_met:].any(dim=-1).float()
        gt_offroad = (gt_dist_edge[:, 0, t_met:] > 0).any(dim=-1).float()
        agent_valid_or = eval_valid_fut.any(dim=-1)
        offroad_scores = _bernoulli_likelihood_scores_per_rollout(
            offroad_flag_sim,
            gt_offroad,
            agent_valid_or,
            float(cfg.bernoulli_smooth.get("offroad_indication", 0.0)),
        )

    w_tl = float(cfg.weights.get("traffic_light_violation", 0.0))
    tl_scores = torch.ones(G, device=device, dtype=dtype)

    w = cfg.weights
    rmm = (
        w["linear_speed"] * ls_scores
        + w["linear_acceleration"] * la_scores
        + w["angular_speed"] * as_scores
        + w["angular_acceleration"] * aa_scores
        + w["distance_to_nearest_object"] * dno_scores
        + w["collision_indication"] * col_ind_scores
        + w["time_to_collision"] * ttc_scores
        + w["distance_to_road_edge"] * edge_scores
        + w["offroad_indication"] * offroad_scores
        + w_tl * tl_scores
    )

    if return_subscores:
        sub = {
            "linear_speed": ls_scores,
            "linear_acceleration": la_scores,
            "angular_speed": as_scores,
            "angular_acceleration": aa_scores,
            "distance_to_nearest_object": dno_scores,
            "collision_indication": col_ind_scores,
            "time_to_collision": ttc_scores,
            "distance_to_road_edge": edge_scores,
            "offroad_indication": offroad_scores,
            "traffic_light_violation": tl_scores,
        }
        return rmm, sub
    return rmm
