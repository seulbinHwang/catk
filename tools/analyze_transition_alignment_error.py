from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TYPE_IDS = {
    "all": None,
    "vehicle": 0,
    "pedestrian": 1,
    "cyclist": 2,
}


@dataclass(frozen=True)
class AlignmentStatsConfig:
    current_step: int = 10
    commit_steps: int = 5
    flow_window_steps: int = 20
    num_anchors: int = 16
    max_future_steps: int = 80
    anchor_valid_mode: str = "prefix"
    use_holonomic_model_only: bool = False
    vehicle_no_slip_point_ratio: float = 0.2289518863
    cyclist_no_slip_point_ratio: float = 0.0495847873
    hist_max_error_m: float = 10.0
    hist_bins: int = 5_000
    thresholds_m: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0)
    warn_step_p99_m: float = 0.50
    warn_anchor_max_p99_m: float = 1.0
    warn_agent_max_p99_m: float = 2.0


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def safe_sinc(x: np.ndarray) -> np.ndarray:
    near_zero = np.abs(x) < 1.0e-6
    safe_x = np.where(near_zero, np.ones_like(x), x)
    x2 = x * x
    return np.where(near_zero, 1.0 - x2 / 6.0 + x2 * x2 / 120.0, np.sin(x) / safe_x)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_agent_record(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    agent = data["agent"]
    position = _as_numpy(agent["position"])
    try:
        velocity_value = agent["velocity"]
    except KeyError:
        velocity_value = np.zeros_like(position[..., :2])
    velocity = _as_numpy(velocity_value)
    return {
        "pos": position[..., :2].astype(np.float32, copy=False),
        "heading": _as_numpy(agent["heading"]).astype(np.float32, copy=False),
        "valid": _as_numpy(agent["valid_mask"]).astype(bool, copy=False),
        "vel": velocity[..., :2].astype(np.float32, copy=False),
        "type": _as_numpy(agent["type"]).astype(np.int16, copy=False),
        "length": _as_numpy(agent["shape"])[:, 0].astype(np.float32, copy=False),
    }


def clean_heading_np(valid: np.ndarray, heading: np.ndarray) -> np.ndarray:
    heading = heading.copy()
    if heading.shape[1] <= 1:
        return heading
    valid_pairs = valid[:, :-1] & valid[:, 1:]
    for step_idx in range(heading.shape[1] - 1):
        heading_diff = np.abs(wrap_angle(heading[:, step_idx] - heading[:, step_idx + 1]))
        change_needed = (heading_diff > 1.5) & valid_pairs[:, step_idx]
        heading[change_needed, step_idx + 1] = heading[change_needed, step_idx]
    return heading


def extrapolate_agent_to_prev_token_step_np(
    valid: np.ndarray,
    pos: np.ndarray,
    heading: np.ndarray,
    vel: np.ndarray,
    *,
    shift: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = valid.copy()
    pos = pos.copy()
    heading = heading.copy()
    vel = vel.copy()
    if valid.shape[1] == 0:
        return valid, pos, heading, vel

    first_valid_step = np.argmax(valid, axis=1)
    for agent_idx, first_step in enumerate(first_valid_step):
        step = int(first_step)
        n_step_to_extrapolate = step % int(shift)
        if step == 10 and step - int(shift) >= 0 and not bool(valid[agent_idx, step - int(shift)]):
            n_step_to_extrapolate = int(shift)
        if n_step_to_extrapolate <= 0:
            continue

        start = step - n_step_to_extrapolate
        vel[agent_idx, start:step] = vel[agent_idx, step]
        valid[agent_idx, start:step] = True
        heading[agent_idx, start:step] = heading[agent_idx, step]
        for offset in range(n_step_to_extrapolate):
            dst = step - offset - 1
            pos[agent_idx, dst] = pos[agent_idx, dst + 1] - vel[agent_idx, step] * 0.1

    return valid, pos, heading, vel


def preprocess_agent_record(record: dict[str, np.ndarray], cfg: AlignmentStatsConfig) -> dict[str, np.ndarray]:
    heading = clean_heading_np(record["valid"], record["heading"])
    valid, pos, heading, vel = extrapolate_agent_to_prev_token_step_np(
        valid=record["valid"],
        pos=record["pos"],
        heading=heading,
        vel=record["vel"],
        shift=cfg.commit_steps,
    )
    return {
        "pos": pos,
        "heading": heading,
        "valid": valid,
        "vel": vel,
        "type": record["type"],
        "length": record["length"],
    }


def _resolve_no_slip_offset_np(
    agent_type: np.ndarray,
    agent_length: np.ndarray,
    cfg: AlignmentStatsConfig,
) -> np.ndarray:
    ratio = np.zeros((agent_type.shape[0],), dtype=np.float32)
    ratio[agent_type == TYPE_IDS["vehicle"]] = float(cfg.vehicle_no_slip_point_ratio)
    ratio[agent_type == TYPE_IDS["cyclist"]] = float(cfg.cyclist_no_slip_point_ratio)
    if cfg.use_holonomic_model_only:
        ratio.fill(0.0)
    return ratio * agent_length.astype(np.float32, copy=False)


def _inverse_control_step_np(
    source_pos: np.ndarray,
    source_head: np.ndarray,
    target_pos: np.ndarray,
    target_head: np.ndarray,
    holonomic_mask: np.ndarray,
    no_slip_offset: np.ndarray,
) -> np.ndarray:
    delta_head = wrap_angle(target_head - source_head)
    delta_vec = target_pos - source_pos

    cos_head = np.cos(source_head)
    sin_head = np.sin(source_head)
    source_heading_vec = np.stack([cos_head, sin_head], axis=-1)
    target_heading_vec = np.stack([np.cos(target_head), np.sin(target_head)], axis=-1)

    ped_delta_s = delta_vec[:, 0] * cos_head + delta_vec[:, 1] * sin_head
    ped_delta_n = -delta_vec[:, 0] * sin_head + delta_vec[:, 1] * cos_head

    mid_head = source_head + 0.5 * delta_head
    h_mid = np.stack([np.cos(mid_head), np.sin(mid_head)], axis=-1)
    source_no_slip_pos = source_pos - no_slip_offset[:, None] * source_heading_vec
    target_no_slip_pos = target_pos - no_slip_offset[:, None] * target_heading_vec
    nonhol_delta_vec = target_no_slip_pos - source_no_slip_pos
    nonhol_proj = np.sum(nonhol_delta_vec * h_mid, axis=-1)
    nonhol_delta_s = nonhol_proj / safe_sinc(0.5 * delta_head)

    delta_s = np.where(holonomic_mask, ped_delta_s, nonhol_delta_s)
    delta_n = np.where(holonomic_mask, ped_delta_n, np.zeros_like(ped_delta_n))
    return np.stack([delta_s, delta_n, delta_head], axis=-1).astype(np.float32, copy=False)


def _decode_control_step_np(
    source_pos: np.ndarray,
    source_head: np.ndarray,
    control: np.ndarray,
    holonomic_mask: np.ndarray,
    no_slip_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta_s = control[:, 0]
    delta_n = control[:, 1]
    delta_head = control[:, 2]

    cos_head = np.cos(source_head)
    sin_head = np.sin(source_head)
    next_head = wrap_angle(source_head + delta_head)
    delta_pos_ped = np.stack(
        [
            delta_s * cos_head - delta_n * sin_head,
            delta_s * sin_head + delta_n * cos_head,
        ],
        axis=-1,
    )

    mid_head = source_head + 0.5 * delta_head
    arc_scale = delta_s * safe_sinc(0.5 * delta_head)
    delta_pos_nonhol = np.stack(
        [arc_scale * np.cos(mid_head), arc_scale * np.sin(mid_head)],
        axis=-1,
    )
    if np.any(no_slip_offset != 0.0):
        current_heading_vec = np.stack([cos_head, sin_head], axis=-1)
        next_heading_vec = np.stack([np.cos(next_head), np.sin(next_head)], axis=-1)
        delta_pos_nonhol = delta_pos_nonhol + no_slip_offset[:, None] * (
            next_heading_vec - current_heading_vec
        )

    delta_pos = np.where(holonomic_mask[:, None], delta_pos_ped, delta_pos_nonhol)
    return (
        (source_pos + delta_pos).astype(np.float32, copy=False),
        next_head.astype(np.float32, copy=False),
    )


def transition_aligned_position_error(
    pos: np.ndarray,
    heading: np.ndarray,
    agent_type: np.ndarray,
    agent_length: np.ndarray,
    cfg: AlignmentStatsConfig,
) -> np.ndarray:
    current_step = int(cfg.current_step)
    if pos.ndim != 3 or pos.shape[-1] != 2:
        raise ValueError(f"pos must have shape [N, T, 2], got {pos.shape}.")
    if heading.shape != pos.shape[:2]:
        raise ValueError(f"heading must have shape [N, T], got {heading.shape}.")
    if current_step < 0 or current_step >= pos.shape[1]:
        raise ValueError(f"current_step={current_step} is outside n_step={pos.shape[1]}.")
    commit_steps = int(cfg.commit_steps)
    if commit_steps <= 0:
        raise ValueError(f"commit_steps must be positive, got {commit_steps}.")

    num_agent = pos.shape[0]
    future_len = pos.shape[1] - current_step - 1
    if future_len <= 0:
        return np.zeros((num_agent, 0), dtype=np.float32)

    roll_pos = pos[:, current_step].astype(np.float32, copy=True)
    roll_head = heading[:, current_step].astype(np.float32, copy=True)
    holonomic_mask = agent_type == TYPE_IDS["pedestrian"]
    if cfg.use_holonomic_model_only:
        holonomic_mask = np.ones_like(holonomic_mask, dtype=bool)
    no_slip_offset = _resolve_no_slip_offset_np(
        agent_type=agent_type,
        agent_length=agent_length,
        cfg=cfg,
    )

    errors = np.zeros((num_agent, future_len), dtype=np.float32)

    for block_start in range(current_step, pos.shape[1] - 1, commit_steps):
        block_end = min(block_start + commit_steps, pos.shape[1] - 1)
        block_len = block_end - block_start
        target_pos = pos[:, block_end].astype(np.float32, copy=False)
        target_head = heading[:, block_end].astype(np.float32, copy=False)
        block_start_pos = roll_pos.copy()
        block_start_head = roll_head.copy()
        block_control = _inverse_control_step_np(
            source_pos=block_start_pos,
            source_head=block_start_head,
            target_pos=target_pos,
            target_head=target_head,
            holonomic_mask=holonomic_mask,
            no_slip_offset=no_slip_offset,
        )
        nonhol_sub_control = block_control / float(block_len)

        for sub_idx, raw_step in enumerate(range(block_start + 1, block_end + 1), start=1):
            step_control = nonhol_sub_control
            if np.any(holonomic_mask):
                fraction = float(sub_idx) / float(block_len)
                interp_pos = block_start_pos + fraction * (target_pos - block_start_pos)
                interp_head = wrap_angle(block_start_head + fraction * block_control[:, 2])
                holonomic_control = _inverse_control_step_np(
                    source_pos=roll_pos,
                    source_head=roll_head,
                    target_pos=interp_pos,
                    target_head=interp_head,
                    holonomic_mask=np.ones_like(holonomic_mask, dtype=bool),
                    no_slip_offset=np.zeros_like(no_slip_offset),
                )
                step_control = np.where(holonomic_mask[:, None], holonomic_control, nonhol_sub_control)

            roll_pos, roll_head = _decode_control_step_np(
                source_pos=roll_pos,
                source_head=roll_head,
                control=step_control,
                holonomic_mask=holonomic_mask,
                no_slip_offset=no_slip_offset,
            )
            future_idx = raw_step - current_step - 1
            errors[:, future_idx] = np.linalg.norm(roll_pos - pos[:, raw_step], axis=-1)

    return errors


def _empty_metric_stats(cfg: AlignmentStatsConfig) -> dict[str, Any]:
    return {
        "count": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "max": 0.0,
        "hist": np.zeros((cfg.hist_bins,), dtype=np.int64),
        "threshold_gt": np.zeros((len(cfg.thresholds_m),), dtype=np.int64),
    }


def _empty_group_stats(cfg: AlignmentStatsConfig) -> dict[str, Any]:
    return {
        "step_error": _empty_metric_stats(cfg),
        "agent_max_error": _empty_metric_stats(cfg),
        "anchor_window_max_error": _empty_metric_stats(cfg),
        "future_step_count": np.zeros((cfg.max_future_steps,), dtype=np.int64),
        "future_step_sum": np.zeros((cfg.max_future_steps,), dtype=np.float64),
        "future_step_max": np.zeros((cfg.max_future_steps,), dtype=np.float32),
    }


def empty_stats(cfg: AlignmentStatsConfig) -> dict[str, Any]:
    return {type_name: _empty_group_stats(cfg) for type_name in TYPE_IDS}


def _histogram(values: np.ndarray, cfg: AlignmentStatsConfig) -> np.ndarray:
    if values.size == 0:
        return np.zeros((cfg.hist_bins,), dtype=np.int64)
    clipped = np.clip(values, 0.0, float(cfg.hist_max_error_m))
    scaled = clipped * (int(cfg.hist_bins) / float(cfg.hist_max_error_m))
    indices = np.minimum(scaled.astype(np.int64), int(cfg.hist_bins) - 1)
    return np.bincount(indices, minlength=int(cfg.hist_bins)).astype(np.int64)


def _accumulate_metric(metric: dict[str, Any], values: np.ndarray, cfg: AlignmentStatsConfig) -> None:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    metric["count"] += int(values.size)
    metric["sum"] += float(values.sum(dtype=np.float64))
    metric["sum_sq"] += float(np.square(values, dtype=np.float32).sum(dtype=np.float64))
    metric["max"] = max(float(metric["max"]), float(values.max()))
    metric["hist"] += _histogram(values, cfg)
    thresholds = np.asarray(cfg.thresholds_m, dtype=np.float32)
    metric["threshold_gt"] += (values[:, None] > thresholds[None]).sum(axis=0).astype(np.int64)


def _accumulate_future_step(
    group_stats: dict[str, Any],
    values: np.ndarray,
    mask: np.ndarray,
    cfg: AlignmentStatsConfig,
) -> None:
    horizon = min(values.shape[1], int(cfg.max_future_steps))
    if horizon <= 0:
        return
    values = values[:, :horizon]
    mask = mask[:, :horizon]
    finite_mask = mask & np.isfinite(values)
    count = finite_mask.sum(axis=0).astype(np.int64)
    if not np.any(count):
        return

    masked_values = np.where(finite_mask, values, 0.0)
    group_stats["future_step_count"][:horizon] += count
    group_stats["future_step_sum"][:horizon] += masked_values.sum(axis=0, dtype=np.float64)
    max_values = np.where(finite_mask, values, -np.inf).max(axis=0)
    has_count = count > 0
    future_step_max = group_stats["future_step_max"][:horizon]
    future_step_max[has_count] = np.maximum(
        future_step_max[has_count],
        max_values[has_count].astype(np.float32, copy=False),
    )


def _build_anchor_window_max_errors(
    errors: np.ndarray,
    valid: np.ndarray,
    agent_type: np.ndarray,
    cfg: AlignmentStatsConfig,
) -> tuple[np.ndarray, np.ndarray]:
    current_step = int(cfg.current_step)
    flow_window_steps = int(cfg.flow_window_steps)
    anchor_values: list[np.ndarray] = []
    anchor_type_values: list[np.ndarray] = []
    if errors.size == 0 or flow_window_steps <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int16)

    for anchor_idx in range(int(cfg.num_anchors)):
        raw_step = int(cfg.commit_steps) * (anchor_idx + 2)
        if raw_step >= valid.shape[1]:
            continue
        future_start = raw_step + 1
        if future_start >= valid.shape[1]:
            continue

        error_start = future_start - current_step - 1
        if error_start < 0:
            continue

        remaining_valid_len = valid.shape[1] - future_start
        remaining_error_len = errors.shape[1] - error_start
        if cfg.anchor_valid_mode == "full" and (
            remaining_valid_len < flow_window_steps or remaining_error_len < flow_window_steps
        ):
            continue

        available_len = min(flow_window_steps, remaining_valid_len, remaining_error_len)
        if available_len <= 0:
            continue

        current_valid = valid[:, raw_step]
        future_valid = valid[:, future_start : future_start + available_len]
        if cfg.anchor_valid_mode == "prefix":
            loss_mask = np.cumprod(future_valid.astype(np.int8), axis=1).astype(bool)
        elif cfg.anchor_valid_mode == "full":
            loss_mask = np.broadcast_to(future_valid.all(axis=1, keepdims=True), future_valid.shape)
        else:
            raise ValueError(f"Unsupported anchor_valid_mode={cfg.anchor_valid_mode!r}.")

        anchor_mask = current_valid & loss_mask.any(axis=1)
        if not np.any(anchor_mask):
            continue

        window_errors = errors[:, error_start : error_start + available_len]
        masked_errors = np.where(loss_mask, window_errors, -np.inf)
        max_error = masked_errors.max(axis=1)
        keep = anchor_mask & np.isfinite(max_error)
        if np.any(keep):
            anchor_values.append(max_error[keep].astype(np.float32, copy=False))
            anchor_type_values.append(agent_type[keep].astype(np.int16, copy=False))

    if not anchor_values:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int16)
    return np.concatenate(anchor_values), np.concatenate(anchor_type_values)


def analyze_record(record: dict[str, np.ndarray], cfg: AlignmentStatsConfig) -> dict[str, Any]:
    record = preprocess_agent_record(record, cfg)
    errors = transition_aligned_position_error(
        pos=record["pos"],
        heading=record["heading"],
        agent_type=record["type"],
        agent_length=record["length"],
        cfg=cfg,
    )
    stats = empty_stats(cfg)
    if errors.shape[1] == 0:
        return stats

    current_valid = record["valid"][:, int(cfg.current_step)]
    future_valid = record["valid"][:, int(cfg.current_step) + 1 : int(cfg.current_step) + 1 + errors.shape[1]]
    step_mask = current_valid[:, None] & future_valid
    anchor_max_values, anchor_types = _build_anchor_window_max_errors(
        errors=errors,
        valid=record["valid"],
        agent_type=record["type"],
        cfg=cfg,
    )

    for type_name, type_id in TYPE_IDS.items():
        type_mask = np.ones((errors.shape[0],), dtype=bool) if type_id is None else record["type"] == type_id
        combined_step_mask = step_mask & type_mask[:, None]
        _accumulate_metric(stats[type_name]["step_error"], errors[combined_step_mask], cfg)
        _accumulate_future_step(stats[type_name], errors, combined_step_mask, cfg)

        agent_mask = type_mask & current_valid & step_mask.any(axis=1)
        if np.any(agent_mask):
            per_agent_error = np.where(step_mask[agent_mask], errors[agent_mask], -np.inf).max(axis=1)
            _accumulate_metric(stats[type_name]["agent_max_error"], per_agent_error, cfg)

        if anchor_max_values.size > 0:
            if type_id is None:
                _accumulate_metric(stats[type_name]["anchor_window_max_error"], anchor_max_values, cfg)
            else:
                _accumulate_metric(
                    stats[type_name]["anchor_window_max_error"],
                    anchor_max_values[anchor_types == type_id],
                    cfg,
                )

    return stats


def analyze_file(path: Path, cfg: AlignmentStatsConfig) -> dict[str, Any]:
    return analyze_record(_load_agent_record(path), cfg)


def merge_stats(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for type_name in TYPE_IDS:
        dst_group = dst[type_name]
        src_group = src[type_name]
        for metric_name in ["step_error", "agent_max_error", "anchor_window_max_error"]:
            dst_metric = dst_group[metric_name]
            src_metric = src_group[metric_name]
            dst_metric["count"] += int(src_metric["count"])
            dst_metric["sum"] += float(src_metric["sum"])
            dst_metric["sum_sq"] += float(src_metric["sum_sq"])
            dst_metric["max"] = max(float(dst_metric["max"]), float(src_metric["max"]))
            dst_metric["hist"] += src_metric["hist"]
            dst_metric["threshold_gt"] += src_metric["threshold_gt"]
        for key in ["future_step_count", "future_step_sum", "future_step_max"]:
            if key == "future_step_max":
                dst_group[key] = np.maximum(dst_group[key], src_group[key])
            else:
                dst_group[key] += src_group[key]
    return dst


def analyze_file_chunk(paths: list[Path], cfg: AlignmentStatsConfig) -> dict[str, Any]:
    merged = empty_stats(cfg)
    for path in paths:
        merge_stats(merged, analyze_file(path, cfg))
    return merged


def _chunked(items: list[Path], chunk_size: int) -> list[list[Path]]:
    return [items[start : start + chunk_size] for start in range(0, len(items), chunk_size)]


def _iter_chunk_results(
    files: list[Path],
    cfg: AlignmentStatsConfig,
    *,
    num_workers: int,
    files_per_task: int,
    progress_interval: int,
) -> Iterable[dict[str, Any]]:
    chunks = _chunked(files, max(1, int(files_per_task)))

    def maybe_report(index: int) -> None:
        processed = min(index * max(1, int(files_per_task)), len(files))
        if progress_interval > 0 and (processed == len(files) or processed % progress_interval < files_per_task):
            print(f"[alignment-error] processed {processed}/{len(files)} files", file=sys.stderr, flush=True)

    if num_workers <= 1:
        for index, chunk in enumerate(chunks, start=1):
            yield analyze_file_chunk(chunk, cfg)
            maybe_report(index)
        return

    worker_args = [(chunk, cfg) for chunk in chunks]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for index, result in enumerate(executor.map(_analyze_file_chunk_star, worker_args), start=1):
            yield result
            maybe_report(index)


def _analyze_file_chunk_star(args: tuple[list[Path], AlignmentStatsConfig]) -> dict[str, Any]:
    paths, cfg = args
    return analyze_file_chunk(paths, cfg)


def _resolve_split_dir(cache_root: Path, split: str) -> Path:
    split_dir = cache_root / split
    if split_dir.is_dir():
        return split_dir
    if cache_root.is_dir() and any(cache_root.glob("*.pkl")):
        return cache_root
    raise FileNotFoundError(f"Cannot find split directory: {split_dir}")


def _list_cache_files(cache_root: Path, split: str, max_files: int | None) -> list[Path]:
    split_dir = _resolve_split_dir(cache_root, split)
    files = sorted(split_dir.glob("*.pkl"))
    if max_files is not None:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No .pkl cache files found in {split_dir}")
    return files


def _histogram_percentile(hist: np.ndarray, percentile: float, cfg: AlignmentStatsConfig) -> float:
    total = int(hist.sum())
    if total <= 0:
        return float("nan")
    rank = max(1, int(math.ceil(float(percentile) / 100.0 * total)))
    index = int(np.searchsorted(np.cumsum(hist), rank, side="left"))
    return (index + 0.5) * float(cfg.hist_max_error_m) / int(cfg.hist_bins)


def _summarize_metric(metric: dict[str, Any], cfg: AlignmentStatsConfig) -> dict[str, Any]:
    count = int(metric["count"])
    threshold_gt = metric["threshold_gt"]
    return {
        "count": count,
        "mean_m": float(metric["sum"] / count) if count else float("nan"),
        "rms_m": float(math.sqrt(metric["sum_sq"] / count)) if count else float("nan"),
        "p50_m": _histogram_percentile(metric["hist"], 50.0, cfg),
        "p90_m": _histogram_percentile(metric["hist"], 90.0, cfg),
        "p95_m": _histogram_percentile(metric["hist"], 95.0, cfg),
        "p99_m": _histogram_percentile(metric["hist"], 99.0, cfg),
        "p99_9_m": _histogram_percentile(metric["hist"], 99.9, cfg),
        "max_m": float(metric["max"]) if count else float("nan"),
        "threshold_gt_rate": {
            f">{threshold:g}m": float(threshold_gt[index] / count) if count else float("nan")
            for index, threshold in enumerate(cfg.thresholds_m)
        },
        "threshold_gt_count": {
            f">{threshold:g}m": int(threshold_gt[index])
            for index, threshold in enumerate(cfg.thresholds_m)
        },
    }


def _summarize_future_steps(group: dict[str, Any], cfg: AlignmentStatsConfig) -> list[dict[str, Any]]:
    result = []
    count = group["future_step_count"]
    total = group["future_step_sum"]
    max_value = group["future_step_max"]
    for index in range(int(cfg.max_future_steps)):
        if int(count[index]) <= 0:
            continue
        result.append(
            {
                "raw_step": int(cfg.current_step + 1 + index),
                "horizon_s": round(0.1 * float(index + 1), 3),
                "count": int(count[index]),
                "mean_m": float(total[index] / count[index]),
                "max_m": float(max_value[index]),
            }
        )
    return result


def summarize_stats(stats: dict[str, Any], cfg: AlignmentStatsConfig) -> dict[str, Any]:
    summary = {}
    for type_name in TYPE_IDS:
        group = stats[type_name]
        summary[type_name] = {
            "step_error": _summarize_metric(group["step_error"], cfg),
            "agent_max_error": _summarize_metric(group["agent_max_error"], cfg),
            "anchor_window_max_error": _summarize_metric(group["anchor_window_max_error"], cfg),
            "future_step_mean": _summarize_future_steps(group, cfg),
        }
    summary["heuristic_flags"] = _build_heuristic_flags(summary, cfg)
    return summary


def _build_heuristic_flags(summary: dict[str, Any], cfg: AlignmentStatsConfig) -> dict[str, Any]:
    all_stats = summary["all"]
    flags = []
    if all_stats["step_error"]["p99_m"] > float(cfg.warn_step_p99_m):
        flags.append(
            f"step_error p99 {all_stats['step_error']['p99_m']:.3f}m > {cfg.warn_step_p99_m:.3f}m"
        )
    if all_stats["anchor_window_max_error"]["p99_m"] > float(cfg.warn_anchor_max_p99_m):
        flags.append(
            "anchor_window_max_error p99 "
            f"{all_stats['anchor_window_max_error']['p99_m']:.3f}m > {cfg.warn_anchor_max_p99_m:.3f}m"
        )
    if all_stats["agent_max_error"]["p99_m"] > float(cfg.warn_agent_max_p99_m):
        flags.append(
            f"agent_max_error p99 {all_stats['agent_max_error']['p99_m']:.3f}m > "
            f"{cfg.warn_agent_max_p99_m:.3f}m"
        )
    return {
        "status": "warn" if flags else "ok",
        "flags": flags,
        "note": (
            "These are conservative sanity flags, not a paper claim. "
            "Use threshold rates and type-specific tails to decide whether the alignment is acceptable."
        ),
    }


def analyze_transition_alignment(
    cache_root: Path,
    split: str,
    cfg: AlignmentStatsConfig,
    *,
    max_files: int | None = None,
    num_workers: int = 1,
    files_per_task: int = 512,
    progress_interval: int = 10_000,
) -> dict[str, Any]:
    files = _list_cache_files(cache_root=cache_root, split=split, max_files=max_files)
    merged = empty_stats(cfg)
    for result in _iter_chunk_results(
        files,
        cfg,
        num_workers=max(1, int(num_workers)),
        files_per_task=max(1, int(files_per_task)),
        progress_interval=max(0, int(progress_interval)),
    ):
        merge_stats(merged, result)
    return {
        "cache_root": str(cache_root),
        "split": split,
        "file_count": len(files),
        "config": asdict(cfg),
        "summary": summarize_stats(merged, cfg),
    }


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_sanitize(value.tolist())
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _format_metric_line(name: str, metric: dict[str, Any]) -> str:
    return (
        f"  {name:24s} count={metric['count']} "
        f"mean={metric['mean_m']:.4f}m p50={metric['p50_m']:.4f}m "
        f"p90={metric['p90_m']:.4f}m p99={metric['p99_m']:.4f}m "
        f"p99.9={metric['p99_9_m']:.4f}m max={metric['max_m']:.4f}m"
    )


def print_report(result: dict[str, Any]) -> None:
    print(f"cache_root: {result['cache_root']}")
    print(f"split: {result['split']} files={result['file_count']}")
    print(f"config: {result['config']}")
    print("")
    for type_name in TYPE_IDS:
        stats = result["summary"][type_name]
        print(f"{type_name}:")
        print(_format_metric_line("valid future step", stats["step_error"]))
        print(_format_metric_line("per-agent future max", stats["agent_max_error"]))
        print(_format_metric_line("per-anchor window max", stats["anchor_window_max_error"]))
        rates = stats["step_error"]["threshold_gt_rate"]
        print(
            "  step tail rates: "
            + ", ".join(f"{key}={100.0 * value:.3f}%" for key, value in rates.items())
        )
        print("")
    flags = result["summary"]["heuristic_flags"]
    print(f"heuristic_status: {flags['status']}")
    for flag in flags["flags"]:
        print(f"  - {flag}")


def _parse_thresholds(raw: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("threshold list must not be empty")
    if any(value < 0.0 for value in values):
        raise argparse.ArgumentTypeError("thresholds must be non-negative")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure raw-vs-transition-aligned position error over WOMD SMART cache. "
            "The tool mirrors the use_kinematic_control_flow=True 0.5s endpoint + "
            "kinematic substep preprocessing without modifying cache files."
        )
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ["CACHE_ROOT"]) if "CACHE_ROOT" in os.environ else None,
        help="SMART cache root. Defaults to the CACHE_ROOT environment variable.",
    )
    parser.add_argument("--split", default="training")
    parser.add_argument("--max-files", type=int, default=None, help="Debug: limit number of cache files.")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--files-per-task",
        type=int,
        default=512,
        help="Number of pkl files each worker merges before returning stats. Larger values reduce IPC.",
    )
    parser.add_argument("--progress-interval", type=int, default=10_000)
    parser.add_argument("--current-step", type=int, default=10)
    parser.add_argument("--commit-steps", type=int, default=5, help="Raw 10Hz steps per endpoint commit.")
    parser.add_argument("--flow-window-steps", type=int, default=20)
    parser.add_argument("--num-anchors", type=int, default=16)
    parser.add_argument("--max-future-steps", type=int, default=80)
    parser.add_argument("--anchor-valid-mode", choices=["prefix", "full"], default="prefix")
    parser.add_argument("--use-holonomic-model-only", action="store_true")
    parser.add_argument("--vehicle-no-slip-point-ratio", type=float, default=0.2289518863)
    parser.add_argument("--cyclist-no-slip-point-ratio", type=float, default=0.0495847873)
    parser.add_argument("--hist-max-error-m", type=float, default=10.0)
    parser.add_argument("--hist-bins", type=int, default=5_000)
    parser.add_argument("--thresholds-m", type=_parse_thresholds, default=(0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0))
    parser.add_argument("--warn-step-p99-m", type=float, default=0.50)
    parser.add_argument("--warn-anchor-max-p99-m", type=float, default=1.0)
    parser.add_argument("--warn-agent-max-p99-m", type=float, default=2.0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cache_root is None:
        raise SystemExit("Set CACHE_ROOT or pass --cache-root.")
    cfg = AlignmentStatsConfig(
        current_step=args.current_step,
        commit_steps=args.commit_steps,
        flow_window_steps=args.flow_window_steps,
        num_anchors=args.num_anchors,
        max_future_steps=args.max_future_steps,
        anchor_valid_mode=args.anchor_valid_mode,
        use_holonomic_model_only=bool(args.use_holonomic_model_only),
        vehicle_no_slip_point_ratio=args.vehicle_no_slip_point_ratio,
        cyclist_no_slip_point_ratio=args.cyclist_no_slip_point_ratio,
        hist_max_error_m=args.hist_max_error_m,
        hist_bins=args.hist_bins,
        thresholds_m=args.thresholds_m,
        warn_step_p99_m=args.warn_step_p99_m,
        warn_anchor_max_p99_m=args.warn_anchor_max_p99_m,
        warn_agent_max_p99_m=args.warn_agent_max_p99_m,
    )
    result = analyze_transition_alignment(
        cache_root=args.cache_root,
        split=args.split,
        cfg=cfg,
        max_files=args.max_files,
        num_workers=max(1, int(args.num_workers)),
        files_per_task=max(1, int(args.files_per_task)),
        progress_interval=max(0, int(args.progress_interval)),
    )
    print_report(result)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(_json_sanitize(result), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        print(f"wrote: {args.output_json}")


if __name__ == "__main__":
    main()
