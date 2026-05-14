from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TYPE_IDS = {"vehicle": 0, "cyclist": 2}


@dataclass(frozen=True)
class EstimatorConfig:
    segment_steps: int = 5
    min_displacement_m: float = 0.25
    min_abs_c_m: float = 0.10
    min_agent_segments: int = 5
    min_agent_info_m: float = 0.5
    rho_min: float = 0.0
    rho_max: float = 0.5
    hist_max_residual_m: float = 5.0
    hist_bins: int = 10_000


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return float("nan")
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cutoff = 0.5 * weights.sum()
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_agent_record(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    agent = data["agent"]
    return {
        "pos": _as_numpy(agent["position"])[..., :2].astype(np.float32, copy=False),
        "heading": _as_numpy(agent["heading"]).astype(np.float32, copy=False),
        "valid": _as_numpy(agent["valid_mask"]).astype(bool, copy=False),
        "type": _as_numpy(agent["type"]).astype(np.int16, copy=False),
        "length": _as_numpy(agent["shape"])[:, 0].astype(np.float32, copy=False),
    }


def _segment_lateral_equation(
    pos: np.ndarray,
    heading: np.ndarray,
    valid: np.ndarray,
    length: np.ndarray,
    cfg: EstimatorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k = int(cfg.segment_steps)
    if pos.shape[0] == 0 or pos.shape[1] <= k:
        empty = np.zeros((pos.shape[0], 0), dtype=np.float32)
        return empty.astype(bool), empty, empty, empty

    seg_valid = np.lib.stride_tricks.sliding_window_view(valid, window_shape=k + 1, axis=1).all(axis=2)
    delta_pos = pos[:, k:] - pos[:, :-k]
    delta_heading = wrap_angle(heading[:, k:] - heading[:, :-k])
    mid_heading = heading[:, :-k] + 0.5 * delta_heading
    lateral_mid = np.stack([-np.sin(mid_heading), np.cos(mid_heading)], axis=-1)
    b_value = np.sum(delta_pos * lateral_mid, axis=-1)
    c_value = 2.0 * length[:, None] * np.sin(0.5 * delta_heading)
    displacement = np.linalg.norm(delta_pos, axis=-1)

    informative = (
        seg_valid
        & np.isfinite(b_value)
        & np.isfinite(c_value)
        & np.isfinite(displacement)
        & np.isfinite(length[:, None])
        & (length[:, None] > 0.0)
        & (displacement >= float(cfg.min_displacement_m))
        & (np.abs(c_value) >= float(cfg.min_abs_c_m))
    )
    return informative, b_value.astype(np.float32, copy=False), c_value.astype(np.float32, copy=False), displacement


def _estimate_agent_rhos_for_type(
    pos: np.ndarray,
    heading: np.ndarray,
    valid: np.ndarray,
    length: np.ndarray,
    cfg: EstimatorConfig,
) -> dict[str, np.ndarray | int | float]:
    informative, b_value, c_value, _ = _segment_lateral_equation(pos, heading, valid, length, cfg)
    if informative.size == 0:
        return {
            "rho": np.zeros((0,), dtype=np.float32),
            "info": np.zeros((0,), dtype=np.float32),
            "segments": np.zeros((0,), dtype=np.int32),
            "informative_segments": 0,
        }

    weights = np.where(informative, np.abs(c_value), 0.0)
    q_value = np.divide(
        b_value,
        c_value,
        out=np.zeros_like(b_value, dtype=np.float32),
        where=informative,
    )
    segment_count = informative.sum(axis=1).astype(np.int32)
    info = weights.sum(axis=1).astype(np.float32)
    agent_keep = (segment_count >= int(cfg.min_agent_segments)) & (info >= float(cfg.min_agent_info_m))

    kept_indices = np.flatnonzero(agent_keep)
    agent_rhos: list[float] = []
    for agent_idx in kept_indices:
        seg_mask = informative[agent_idx]
        rho_i = weighted_median(q_value[agent_idx, seg_mask], weights[agent_idx, seg_mask])
        agent_rhos.append(float(np.clip(rho_i, cfg.rho_min, cfg.rho_max)))

    return {
        "rho": np.asarray(agent_rhos, dtype=np.float32),
        "info": info[kept_indices],
        "segments": segment_count[kept_indices],
        "informative_segments": int(informative.sum()),
    }


def estimate_file_agent_stats(path: Path, cfg: EstimatorConfig) -> dict[str, dict[str, np.ndarray | int]]:
    record = _load_agent_record(path)
    result: dict[str, dict[str, np.ndarray | int]] = {}
    for type_name, type_id in TYPE_IDS.items():
        type_mask = record["type"] == type_id
        result[type_name] = _estimate_agent_rhos_for_type(
            pos=record["pos"][type_mask],
            heading=record["heading"][type_mask],
            valid=record["valid"][type_mask],
            length=record["length"][type_mask],
            cfg=cfg,
        )
    return result


def _histogram_abs(values: np.ndarray, cfg: EstimatorConfig) -> np.ndarray:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros((int(cfg.hist_bins),), dtype=np.int64)
    clipped = np.clip(values, 0.0, float(cfg.hist_max_residual_m))
    scaled = clipped * (int(cfg.hist_bins) / float(cfg.hist_max_residual_m))
    indices = np.minimum(scaled.astype(np.int64), int(cfg.hist_bins) - 1)
    return np.bincount(indices, minlength=int(cfg.hist_bins)).astype(np.int64)


def residual_file_stats(path: Path, cfg: EstimatorConfig, rho_by_type: dict[str, float]) -> dict[str, dict[str, Any]]:
    record = _load_agent_record(path)
    result: dict[str, dict[str, Any]] = {}
    for type_name, type_id in TYPE_IDS.items():
        rho = float(rho_by_type[type_name])
        type_mask = record["type"] == type_id
        informative, b_value, c_value, _ = _segment_lateral_equation(
            pos=record["pos"][type_mask],
            heading=record["heading"][type_mask],
            valid=record["valid"][type_mask],
            length=record["length"][type_mask],
            cfg=cfg,
        )
        if informative.size == 0 or not np.any(informative) or not np.isfinite(rho):
            result[type_name] = {
                "count": 0,
                "hist_before": np.zeros((cfg.hist_bins,), dtype=np.int64),
                "hist_after": np.zeros((cfg.hist_bins,), dtype=np.int64),
                "sum_abs_before": 0.0,
                "sum_abs_after": 0.0,
            }
            continue

        before = np.abs(b_value[informative])
        after = np.abs(b_value[informative] - rho * c_value[informative])
        result[type_name] = {
            "count": int(before.size),
            "hist_before": _histogram_abs(before, cfg),
            "hist_after": _histogram_abs(after, cfg),
            "sum_abs_before": float(before.sum(dtype=np.float64)),
            "sum_abs_after": float(after.sum(dtype=np.float64)),
        }
    return result


def _iter_results(
    worker,
    files: list[Path],
    num_workers: int,
    chunksize: int,
    *,
    label: str,
    progress_interval: int,
) -> Iterable[Any]:
    def maybe_report(index: int) -> None:
        if progress_interval > 0 and (index == len(files) or index % progress_interval == 0):
            print(f"[{label}] processed {index}/{len(files)} files", file=sys.stderr, flush=True)

    if num_workers <= 1:
        for index, path in enumerate(files, start=1):
            yield worker(path)
            maybe_report(index)
        return
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for index, result in enumerate(executor.map(worker, files, chunksize=chunksize), start=1):
            yield result
            maybe_report(index)


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


def _merge_fit_stats(results: Iterable[dict[str, dict[str, np.ndarray | int]]]) -> dict[str, dict[str, Any]]:
    merged = {
        type_name: {
            "rho_parts": [],
            "info_parts": [],
            "segment_parts": [],
            "informative_segments": 0,
        }
        for type_name in TYPE_IDS
    }
    for result in results:
        for type_name in TYPE_IDS:
            stats = result[type_name]
            merged[type_name]["rho_parts"].append(stats["rho"])
            merged[type_name]["info_parts"].append(stats["info"])
            merged[type_name]["segment_parts"].append(stats["segments"])
            merged[type_name]["informative_segments"] += int(stats["informative_segments"])

    summary: dict[str, dict[str, Any]] = {}
    for type_name in TYPE_IDS:
        rho = np.concatenate(merged[type_name]["rho_parts"]) if merged[type_name]["rho_parts"] else np.zeros(0)
        info = np.concatenate(merged[type_name]["info_parts"]) if merged[type_name]["info_parts"] else np.zeros(0)
        segments = (
            np.concatenate(merged[type_name]["segment_parts"])
            if merged[type_name]["segment_parts"]
            else np.zeros(0, dtype=np.int32)
        )
        if rho.size == 0:
            type_rho = float("nan")
            cap = float("nan")
            weights = np.zeros_like(info)
        else:
            cap = float(np.percentile(info, 75))
            weights = np.minimum(info, cap)
            type_rho = weighted_median(rho, weights)
        summary[type_name] = {
            "rho": type_rho,
            "agent_count": int(rho.size),
            "informative_segment_count": int(merged[type_name]["informative_segments"]),
            "accepted_segment_count": int(segments.sum()) if segments.size else 0,
            "info_sum_m": float(info.sum(dtype=np.float64)) if info.size else 0.0,
            "agent_weight_cap_p75_m": cap,
            "agent_info_median_m": float(np.median(info)) if info.size else float("nan"),
            "agent_segments_median": float(np.median(segments)) if segments.size else float("nan"),
        }
    return summary


def _histogram_median(hist: np.ndarray, cfg: EstimatorConfig) -> float:
    total = int(hist.sum())
    if total <= 0:
        return float("nan")
    index = int(np.searchsorted(np.cumsum(hist), (total - 1) // 2 + 1, side="left"))
    return (index + 0.5) * float(cfg.hist_max_residual_m) / int(cfg.hist_bins)


def _merge_residual_stats(results: Iterable[dict[str, dict[str, Any]]], cfg: EstimatorConfig) -> dict[str, dict[str, Any]]:
    merged = {
        type_name: {
            "count": 0,
            "hist_before": np.zeros((cfg.hist_bins,), dtype=np.int64),
            "hist_after": np.zeros((cfg.hist_bins,), dtype=np.int64),
            "sum_abs_before": 0.0,
            "sum_abs_after": 0.0,
        }
        for type_name in TYPE_IDS
    }
    for result in results:
        for type_name in TYPE_IDS:
            stats = result[type_name]
            merged[type_name]["count"] += int(stats["count"])
            merged[type_name]["hist_before"] += stats["hist_before"]
            merged[type_name]["hist_after"] += stats["hist_after"]
            merged[type_name]["sum_abs_before"] += float(stats["sum_abs_before"])
            merged[type_name]["sum_abs_after"] += float(stats["sum_abs_after"])

    summary: dict[str, dict[str, Any]] = {}
    for type_name in TYPE_IDS:
        count = int(merged[type_name]["count"])
        median_before = _histogram_median(merged[type_name]["hist_before"], cfg)
        median_after = _histogram_median(merged[type_name]["hist_after"], cfg)
        improvement = (
            1.0 - (median_after / median_before)
            if count > 0 and np.isfinite(median_before) and median_before > 0.0
            else float("nan")
        )
        summary[type_name] = {
            "segment_count": count,
            "median_abs_before_m": median_before,
            "median_abs_after_m": median_after,
            "median_improvement": improvement,
            "mean_abs_before_m": merged[type_name]["sum_abs_before"] / count if count else float("nan"),
            "mean_abs_after_m": merged[type_name]["sum_abs_after"] / count if count else float("nan"),
        }
    return summary


def estimate_rho(
    cache_root: Path,
    fit_split: str,
    eval_split: str | None,
    cfg: EstimatorConfig,
    max_fit_files: int | None = None,
    max_eval_files: int | None = None,
    num_workers: int = 1,
    chunksize: int = 16,
    progress_interval: int = 1000,
) -> dict[str, Any]:
    fit_files = _list_cache_files(cache_root=cache_root, split=fit_split, max_files=max_fit_files)
    fit_worker = partial(estimate_file_agent_stats, cfg=cfg)
    fit_summary = _merge_fit_stats(
        _iter_results(
            fit_worker,
            fit_files,
            num_workers,
            chunksize,
            label=f"fit:{fit_split}",
            progress_interval=progress_interval,
        )
    )
    rho_by_type = {type_name: float(fit_summary[type_name]["rho"]) for type_name in TYPE_IDS}

    result: dict[str, Any] = {
        "cache_root": str(cache_root),
        "fit_split": fit_split,
        "fit_file_count": len(fit_files),
        "config": cfg.__dict__,
        "rho": rho_by_type,
        "fit": fit_summary,
    }
    if eval_split is not None:
        eval_files = _list_cache_files(cache_root=cache_root, split=eval_split, max_files=max_eval_files)
        residual_worker = partial(residual_file_stats, cfg=cfg, rho_by_type=rho_by_type)
        result["eval_split"] = eval_split
        result["eval_file_count"] = len(eval_files)
        result["residual"] = _merge_residual_stats(
            _iter_results(
                residual_worker,
                eval_files,
                num_workers,
                chunksize,
                label=f"eval:{eval_split}",
                progress_interval=progress_interval,
            ),
            cfg=cfg,
        )
    return result


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


def print_report(result: dict[str, Any]) -> None:
    print(f"cache_root: {result['cache_root']}")
    print(f"fit: split={result['fit_split']} files={result['fit_file_count']}")
    if "eval_split" in result:
        print(f"eval: split={result['eval_split']} files={result['eval_file_count']}")
    print("")
    print("estimated rho:")
    for type_name in TYPE_IDS:
        stats = result["fit"][type_name]
        print(
            f"  {type_name:7s}: rho={result['rho'][type_name]:.6f} "
            f"agents={stats['agent_count']} "
            f"accepted_segments={stats['accepted_segment_count']} "
            f"cap_p75={stats['agent_weight_cap_p75_m']:.3f}m"
        )
    if "residual" in result:
        print("")
        print("residual check:")
        for type_name in TYPE_IDS:
            stats = result["residual"][type_name]
            print(
                f"  {type_name:7s}: segments={stats['segment_count']} "
                f"median_before={stats['median_abs_before_m']:.4f}m "
                f"median_after={stats['median_abs_after_m']:.4f}m "
                f"improvement={100.0 * stats['median_improvement']:.2f}%"
            )
    print("")
    print("config candidates:")
    print(f"  vehicle rho: {result['rho']['vehicle']:.6f}")
    print(f"  cyclist rho: {result['rho']['cyclist']:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate effective no-slip point ratios from WOMD SMART cache with "
            "0.5s sliding segments and type-level capped weighted medians."
        )
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ["CACHE_ROOT"]) if "CACHE_ROOT" in os.environ else None,
        help="SMART cache root. Defaults to the CACHE_ROOT environment variable.",
    )
    parser.add_argument("--fit-split", default="training", help="Split used to fit rho.")
    parser.add_argument("--eval-split", default="validation", help="Split used for residual check. Use 'none' to skip.")
    parser.add_argument("--max-fit-files", type=int, default=None, help="Debug: limit number of fit cache files.")
    parser.add_argument("--max-eval-files", type=int, default=None, help="Debug: limit number of eval cache files.")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument("--segment-steps", type=int, default=5)
    parser.add_argument("--min-displacement-m", type=float, default=0.25)
    parser.add_argument("--min-abs-c-m", type=float, default=0.10)
    parser.add_argument("--min-agent-segments", type=int, default=5)
    parser.add_argument("--min-agent-info-m", type=float, default=0.5)
    parser.add_argument("--rho-min", type=float, default=0.0)
    parser.add_argument("--rho-max", type=float, default=0.5)
    parser.add_argument("--hist-max-residual-m", type=float, default=5.0)
    parser.add_argument("--hist-bins", type=int, default=10_000)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cache_root is None:
        raise SystemExit("Set CACHE_ROOT or pass --cache-root.")
    cfg = EstimatorConfig(
        segment_steps=args.segment_steps,
        min_displacement_m=args.min_displacement_m,
        min_abs_c_m=args.min_abs_c_m,
        min_agent_segments=args.min_agent_segments,
        min_agent_info_m=args.min_agent_info_m,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        hist_max_residual_m=args.hist_max_residual_m,
        hist_bins=args.hist_bins,
    )
    eval_split = None if str(args.eval_split).lower() in {"none", "false", "no"} else args.eval_split
    result = estimate_rho(
        cache_root=args.cache_root,
        fit_split=args.fit_split,
        eval_split=eval_split,
        cfg=cfg,
        max_fit_files=args.max_fit_files,
        max_eval_files=args.max_eval_files,
        num_workers=max(1, int(args.num_workers)),
        chunksize=max(1, int(args.chunksize)),
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
