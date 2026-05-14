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
import torch

from src.smart.modules.kinematic_control import (
    build_rolling_control_target_with_round_trip_error,
)


TYPE_IDS = {"all": None, "vehicle": 0, "pedestrian": 1, "cyclist": 2}


@dataclass(frozen=True)
class AnalysisConfig:
    flow_window_steps: int = 20
    shift: int = 5
    raw_start: int = 10
    raw_end: int = 70
    use_prefix_valid_future_loss_mask: bool = False
    control_pos_scale_m: float = 1.0
    control_vehicle_yaw_scale_rad: float = 0.025
    control_pedestrian_yaw_scale_rad: float = 0.20
    control_cyclist_yaw_scale_rad: float = 0.06
    control_vehicle_no_slip_point_ratio: float = 0.2289518863
    control_cyclist_no_slip_point_ratio: float = 0.0495847873
    use_holonomic_model_only: bool = False
    use_rolling_supervision: bool = True
    hist_max_error_m: float = 20.0
    hist_bins: int = 20_000
    recommend_quantile: float = 99.5
    recommend_rounding_m: float = 0.25


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def _load_agent_record(path: Path) -> dict[str, torch.Tensor]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    agent = data["agent"]
    record = {
        "pos": _as_tensor(agent["position"])[..., :2].to(dtype=torch.float32),
        "heading": _as_tensor(agent["heading"]).to(dtype=torch.float32),
        "valid": _as_tensor(agent["valid_mask"]).to(dtype=torch.bool),
        "type": _as_tensor(agent["type"]).to(dtype=torch.long),
        "length": _as_tensor(agent["shape"])[:, 0].to(dtype=torch.float32),
    }
    if "train_mask" in agent:
        record["train_mask"] = _as_tensor(agent["train_mask"]).to(dtype=torch.bool)
    else:
        record["train_mask"] = torch.ones(record["valid"].shape[0], dtype=torch.bool)
    return record


def _build_future_loss_mask(valid: torch.Tensor, raw_step: int, cfg: AnalysisConfig) -> torch.Tensor:
    future_start = raw_step + 1
    future_loss_mask = torch.zeros((valid.shape[0], cfg.flow_window_steps), dtype=torch.bool)
    available_len = min(cfg.flow_window_steps, max(0, valid.shape[1] - future_start))
    if available_len <= 0:
        return future_loss_mask

    available_future_valid = valid[:, future_start : future_start + available_len].bool()
    if not cfg.use_prefix_valid_future_loss_mask:
        if available_len != cfg.flow_window_steps:
            return future_loss_mask
        full_future_valid = available_future_valid.all(dim=1)
        future_loss_mask[full_future_valid] = True
        return future_loss_mask

    prefix_valid = available_future_valid.to(dtype=torch.long).cumprod(dim=1).bool()
    prefix_len = prefix_valid.long().sum(dim=1)
    usable_len = (prefix_len // cfg.shift) * cfg.shift
    step_index = torch.arange(cfg.flow_window_steps).unsqueeze(0)
    return step_index < usable_len.unsqueeze(1)


def _build_future_pose_with_loss_mask(
    *,
    pos: torch.Tensor,
    heading: torch.Tensor,
    current_pos: torch.Tensor,
    current_head: torch.Tensor,
    anchor_mask: torch.Tensor,
    raw_step: int,
    future_loss_mask: torch.Tensor,
    cfg: AnalysisConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_current_pos = current_pos[anchor_mask]
    selected_current_head = current_head[anchor_mask]
    future_start = raw_step + 1

    future_pos = selected_current_pos.unsqueeze(1).expand(-1, cfg.flow_window_steps, -1).clone()
    future_head = selected_current_head.unsqueeze(1).expand(-1, cfg.flow_window_steps).clone()

    available_len = min(cfg.flow_window_steps, max(0, pos.shape[1] - future_start))
    if available_len > 0:
        future_pos[:, :available_len] = pos[anchor_mask, future_start : future_start + available_len]
        future_head[:, :available_len] = heading[anchor_mask, future_start : future_start + available_len]

    valid_step_count = future_loss_mask.long().sum(dim=1)
    if bool((valid_step_count <= 0).any().item()):
        raise ValueError("future_loss_mask must contain at least one valid future step per selected anchor.")

    last_valid_index = valid_step_count - 1
    last_valid_pos = future_pos.gather(
        dim=1,
        index=last_valid_index.view(-1, 1, 1).expand(-1, 1, future_pos.shape[-1]),
    ).squeeze(1)
    last_valid_head = future_head.gather(
        dim=1,
        index=last_valid_index.view(-1, 1),
    ).squeeze(1)
    invalid_future_mask = ~future_loss_mask
    future_pos = torch.where(invalid_future_mask.unsqueeze(-1), last_valid_pos.unsqueeze(1), future_pos)
    future_head = torch.where(invalid_future_mask, last_valid_head.unsqueeze(1), future_head)
    return future_pos, future_head


def _empty_hist(cfg: AnalysisConfig) -> np.ndarray:
    return np.zeros((int(cfg.hist_bins),), dtype=np.int64)


def _histogram(values: np.ndarray, cfg: AnalysisConfig) -> np.ndarray:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return _empty_hist(cfg)
    clipped = np.clip(values, 0.0, float(cfg.hist_max_error_m))
    scaled = clipped * (int(cfg.hist_bins) / float(cfg.hist_max_error_m))
    indices = np.minimum(scaled.astype(np.int64), int(cfg.hist_bins) - 1)
    return np.bincount(indices, minlength=int(cfg.hist_bins)).astype(np.int64)


def _new_type_stats(cfg: AnalysisConfig, thresholds: np.ndarray) -> dict[str, Any]:
    return {
        "anchor_count": 0,
        "loss_step_count": 0,
        "hist_anchor_max": _empty_hist(cfg),
        "hist_step": _empty_hist(cfg),
        "threshold_keep_counts": np.zeros((thresholds.shape[0],), dtype=np.int64),
        "sum_anchor_max_error_m": 0.0,
        "sum_step_error_m": 0.0,
        "max_anchor_error_m": 0.0,
        "max_step_error_m": 0.0,
        "hist_clipped_anchor_count": 0,
        "hist_clipped_step_count": 0,
    }


def _update_stats(stats: dict[str, Any], anchor_max: torch.Tensor, step_error: torch.Tensor, thresholds: np.ndarray, cfg: AnalysisConfig) -> None:
    anchor_np = anchor_max.detach().cpu().numpy().astype(np.float64, copy=False)
    step_np = step_error.detach().cpu().numpy().astype(np.float64, copy=False)
    anchor_np = anchor_np[np.isfinite(anchor_np)]
    step_np = step_np[np.isfinite(step_np)]
    if anchor_np.size == 0:
        return

    stats["anchor_count"] += int(anchor_np.size)
    stats["loss_step_count"] += int(step_np.size)
    stats["hist_anchor_max"] += _histogram(anchor_np, cfg)
    stats["hist_step"] += _histogram(step_np, cfg)
    stats["threshold_keep_counts"] += np.asarray([(anchor_np <= threshold).sum() for threshold in thresholds], dtype=np.int64)
    stats["sum_anchor_max_error_m"] += float(anchor_np.sum(dtype=np.float64))
    stats["sum_step_error_m"] += float(step_np.sum(dtype=np.float64)) if step_np.size else 0.0
    stats["max_anchor_error_m"] = max(float(stats["max_anchor_error_m"]), float(anchor_np.max(initial=0.0)))
    stats["max_step_error_m"] = max(float(stats["max_step_error_m"]), float(step_np.max(initial=0.0)))
    stats["hist_clipped_anchor_count"] += int((anchor_np >= float(cfg.hist_max_error_m)).sum())
    stats["hist_clipped_step_count"] += int((step_np >= float(cfg.hist_max_error_m)).sum())


def analyze_cache_file(path: Path, cfg: AnalysisConfig, thresholds: np.ndarray) -> dict[str, dict[str, Any]]:
    record = _load_agent_record(path)
    pos = record["pos"]
    heading = record["heading"]
    valid = record["valid"]
    agent_type = record["type"]
    agent_length = record["length"]
    train_mask = record["train_mask"]

    result = {type_name: _new_type_stats(cfg, thresholds) for type_name in TYPE_IDS}
    raw_current_steps = range(int(cfg.raw_start), int(cfg.raw_end) + 1, int(cfg.shift))
    for raw_step in raw_current_steps:
        if raw_step >= valid.shape[1]:
            continue
        future_loss_mask_all = _build_future_loss_mask(valid=valid, raw_step=raw_step, cfg=cfg)
        anchor_mask = valid[:, raw_step] & future_loss_mask_all.any(dim=1) & train_mask
        if not bool(anchor_mask.any().item()):
            continue

        selected_loss_mask = future_loss_mask_all[anchor_mask]
        future_pos, future_head = _build_future_pose_with_loss_mask(
            pos=pos,
            heading=heading,
            current_pos=pos[:, raw_step],
            current_head=heading[:, raw_step],
            anchor_mask=anchor_mask,
            raw_step=raw_step,
            future_loss_mask=selected_loss_mask,
            cfg=cfg,
        )
        selected_type = agent_type[anchor_mask]
        selected_length = agent_length[anchor_mask]
        _, round_trip_error_m = build_rolling_control_target_with_round_trip_error(
            future_pos=future_pos,
            future_head=future_head,
            current_pos=pos[anchor_mask, raw_step],
            current_head=heading[anchor_mask, raw_step],
            agent_type=selected_type,
            agent_length=selected_length,
            pos_scale_m=cfg.control_pos_scale_m,
            vehicle_yaw_scale_rad=cfg.control_vehicle_yaw_scale_rad,
            pedestrian_yaw_scale_rad=cfg.control_pedestrian_yaw_scale_rad,
            cyclist_yaw_scale_rad=cfg.control_cyclist_yaw_scale_rad,
            use_holonomic_model_only=cfg.use_holonomic_model_only,
            use_rolling_supervision=cfg.use_rolling_supervision,
            vehicle_no_slip_point_ratio=cfg.control_vehicle_no_slip_point_ratio,
            cyclist_no_slip_point_ratio=cfg.control_cyclist_no_slip_point_ratio,
        )
        masked_error = torch.where(
            selected_loss_mask,
            round_trip_error_m,
            torch.zeros_like(round_trip_error_m),
        )
        anchor_max_error = masked_error.max(dim=1).values
        step_error = round_trip_error_m[selected_loss_mask]

        _update_stats(result["all"], anchor_max_error, step_error, thresholds, cfg)
        for type_name, type_id in TYPE_IDS.items():
            if type_id is None:
                continue
            type_mask = selected_type == int(type_id)
            if bool(type_mask.any().item()):
                _update_stats(
                    result[type_name],
                    anchor_max_error[type_mask],
                    round_trip_error_m[type_mask][selected_loss_mask[type_mask]],
                    thresholds,
                    cfg,
                )
    return result


def _iter_results(
    worker,
    files: list[Path],
    num_workers: int,
    chunksize: int,
    *,
    progress_interval: int,
) -> Iterable[Any]:
    def maybe_report(index: int) -> None:
        if progress_interval > 0 and (index == len(files) or index % progress_interval == 0):
            print(f"[round-trip] processed {index}/{len(files)} files", file=sys.stderr, flush=True)

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


def _merge_raw_stats(results: Iterable[dict[str, dict[str, Any]]], cfg: AnalysisConfig, thresholds: np.ndarray) -> dict[str, dict[str, Any]]:
    merged = {type_name: _new_type_stats(cfg, thresholds) for type_name in TYPE_IDS}
    for result in results:
        for type_name in TYPE_IDS:
            for key in [
                "anchor_count",
                "loss_step_count",
                "hist_clipped_anchor_count",
                "hist_clipped_step_count",
            ]:
                merged[type_name][key] += int(result[type_name][key])
            for key in ["sum_anchor_max_error_m", "sum_step_error_m"]:
                merged[type_name][key] += float(result[type_name][key])
            for key in ["hist_anchor_max", "hist_step", "threshold_keep_counts"]:
                merged[type_name][key] += result[type_name][key]
            for key in ["max_anchor_error_m", "max_step_error_m"]:
                merged[type_name][key] = max(float(merged[type_name][key]), float(result[type_name][key]))
    return merged


def _percentile_from_hist(hist: np.ndarray, percentile: float, cfg: AnalysisConfig) -> float:
    total = int(hist.sum())
    if total <= 0:
        return float("nan")
    percentile = min(max(float(percentile), 0.0), 100.0)
    rank = int(math.ceil((percentile / 100.0) * total))
    rank = min(max(rank, 1), total)
    index = int(np.searchsorted(np.cumsum(hist), rank, side="left"))
    return (index + 0.5) * float(cfg.hist_max_error_m) / int(cfg.hist_bins)


def _cdf_from_hist(hist: np.ndarray, threshold: float, cfg: AnalysisConfig) -> float:
    total = int(hist.sum())
    if total <= 0:
        return float("nan")
    index = int(math.floor(float(threshold) * int(cfg.hist_bins) / float(cfg.hist_max_error_m)))
    index = min(max(index, 0), int(cfg.hist_bins) - 1)
    return float(hist[: index + 1].sum()) / total


def _summarize_type_stats(raw: dict[str, Any], thresholds: np.ndarray, cfg: AnalysisConfig) -> dict[str, Any]:
    anchor_count = int(raw["anchor_count"])
    loss_step_count = int(raw["loss_step_count"])
    percentiles = [50, 75, 90, 95, 97, 99, 99.5, 99.9]
    anchor_percentiles = {
        f"p{str(percentile).replace('.', '_')}_m": _percentile_from_hist(raw["hist_anchor_max"], percentile, cfg)
        for percentile in percentiles
    }
    step_percentiles = {
        f"p{str(percentile).replace('.', '_')}_m": _percentile_from_hist(raw["hist_step"], percentile, cfg)
        for percentile in percentiles
    }
    threshold_table = []
    for threshold, keep_count in zip(thresholds.tolist(), raw["threshold_keep_counts"].tolist(), strict=True):
        keep_count = int(keep_count)
        threshold_table.append(
            {
                "threshold_m": float(threshold),
                "keep_count": keep_count,
                "drop_count": anchor_count - keep_count,
                "keep_pct": 100.0 * keep_count / anchor_count if anchor_count else float("nan"),
                "drop_pct": 100.0 * (anchor_count - keep_count) / anchor_count if anchor_count else float("nan"),
            }
        )

    recommended_base = anchor_percentiles[f"p{str(cfg.recommend_quantile).replace('.', '_')}_m"]
    if np.isfinite(recommended_base) and cfg.recommend_rounding_m > 0.0:
        recommended = math.ceil(recommended_base / cfg.recommend_rounding_m) * cfg.recommend_rounding_m
    else:
        recommended = float("nan")
    return {
        "anchor_count": anchor_count,
        "loss_step_count": loss_step_count,
        "anchor_max_error_mean_m": raw["sum_anchor_max_error_m"] / anchor_count if anchor_count else float("nan"),
        "step_error_mean_m": raw["sum_step_error_m"] / loss_step_count if loss_step_count else float("nan"),
        "anchor_max_error_max_m": float(raw["max_anchor_error_m"]),
        "step_error_max_m": float(raw["max_step_error_m"]),
        "anchor_hist_clipped_count": int(raw["hist_clipped_anchor_count"]),
        "step_hist_clipped_count": int(raw["hist_clipped_step_count"]),
        "anchor_max_error_percentiles": anchor_percentiles,
        "step_error_percentiles": step_percentiles,
        "threshold_table": threshold_table,
        "recommended_threshold_m": recommended,
        "recommended_quantile": float(cfg.recommend_quantile),
        "recommended_keep_pct_approx": 100.0 * _cdf_from_hist(raw["hist_anchor_max"], recommended, cfg)
        if np.isfinite(recommended)
        else float("nan"),
    }


def analyze_round_trip_distribution(
    *,
    cache_root: Path,
    split: str,
    cfg: AnalysisConfig,
    thresholds: np.ndarray,
    max_files: int | None = None,
    num_workers: int = 1,
    chunksize: int = 16,
    progress_interval: int = 1000,
) -> dict[str, Any]:
    files = _list_cache_files(cache_root=cache_root, split=split, max_files=max_files)
    worker = partial(analyze_cache_file, cfg=cfg, thresholds=thresholds)
    raw = _merge_raw_stats(
        _iter_results(
            worker,
            files,
            max(1, int(num_workers)),
            max(1, int(chunksize)),
            progress_interval=max(0, int(progress_interval)),
        ),
        cfg=cfg,
        thresholds=thresholds,
    )
    summary = {type_name: _summarize_type_stats(raw[type_name], thresholds, cfg) for type_name in TYPE_IDS}
    return {
        "cache_root": str(cache_root),
        "split": split,
        "file_count": len(files),
        "config": cfg.__dict__,
        "thresholds_m": thresholds.tolist(),
        "summary": summary,
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


def _parse_thresholds(text: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--thresholds must contain at least one value.")
    if any(value <= 0.0 for value in values):
        raise ValueError("--thresholds must be positive meter values.")
    return np.asarray(sorted(set(values)), dtype=np.float64)


def print_report(result: dict[str, Any]) -> None:
    print(f"cache_root: {result['cache_root']}")
    print(f"split: {result['split']} files={result['file_count']}")
    print("")
    for type_name in TYPE_IDS:
        stats = result["summary"][type_name]
        p = stats["anchor_max_error_percentiles"]
        print(
            f"{type_name}: anchors={stats['anchor_count']} "
            f"steps={stats['loss_step_count']} "
            f"mean={stats['anchor_max_error_mean_m']:.4f}m "
            f"p95={p['p95_m']:.4f}m p99={p['p99_m']:.4f}m "
            f"p99_5={p['p99_5_m']:.4f}m max={stats['anchor_max_error_max_m']:.4f}m"
        )
        print(
            f"  recommendation: {stats['recommended_threshold_m']:.3f}m "
            f"(p{stats['recommended_quantile']}, approx keep={stats['recommended_keep_pct_approx']:.2f}%)"
        )
        print("  thresholds:")
        for row in stats["threshold_table"]:
            print(
                f"    {row['threshold_m']:>6.2f}m: keep={row['keep_pct']:>6.2f}% "
                f"drop={row['drop_pct']:>6.2f}% ({row['drop_count']} anchors)"
            )
        print("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze control-space GT -> control -> pose round-trip position error "
            "distribution from SMART cache."
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
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--flow-window-steps", type=int, default=20)
    parser.add_argument("--shift", type=int, default=5)
    parser.add_argument("--raw-start", type=int, default=10)
    parser.add_argument("--raw-end", type=int, default=70)
    parser.add_argument("--use-prefix-valid-future-loss-mask", action="store_true")
    parser.add_argument("--control-pos-scale-m", type=float, default=1.0)
    parser.add_argument("--control-vehicle-yaw-scale-rad", type=float, default=0.025)
    parser.add_argument("--control-pedestrian-yaw-scale-rad", type=float, default=0.20)
    parser.add_argument("--control-cyclist-yaw-scale-rad", type=float, default=0.06)
    parser.add_argument("--control-vehicle-no-slip-point-ratio", type=float, default=0.2289518863)
    parser.add_argument("--control-cyclist-no-slip-point-ratio", type=float, default=0.0495847873)
    parser.add_argument("--use-holonomic-model-only", action="store_true")
    parser.add_argument("--no-use-rolling-supervision", dest="use_rolling_supervision", action="store_false")
    parser.set_defaults(use_rolling_supervision=True)
    parser.add_argument("--hist-max-error-m", type=float, default=20.0)
    parser.add_argument("--hist-bins", type=int, default=20_000)
    parser.add_argument("--thresholds", default="0.5,1,1.5,2,3,5,10")
    parser.add_argument("--recommend-quantile", type=float, default=99.5)
    parser.add_argument("--recommend-rounding-m", type=float, default=0.25)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cache_root is None:
        raise SystemExit("Set CACHE_ROOT or pass --cache-root.")
    cfg = AnalysisConfig(
        flow_window_steps=args.flow_window_steps,
        shift=args.shift,
        raw_start=args.raw_start,
        raw_end=args.raw_end,
        use_prefix_valid_future_loss_mask=bool(args.use_prefix_valid_future_loss_mask),
        control_pos_scale_m=args.control_pos_scale_m,
        control_vehicle_yaw_scale_rad=args.control_vehicle_yaw_scale_rad,
        control_pedestrian_yaw_scale_rad=args.control_pedestrian_yaw_scale_rad,
        control_cyclist_yaw_scale_rad=args.control_cyclist_yaw_scale_rad,
        control_vehicle_no_slip_point_ratio=args.control_vehicle_no_slip_point_ratio,
        control_cyclist_no_slip_point_ratio=args.control_cyclist_no_slip_point_ratio,
        use_holonomic_model_only=bool(args.use_holonomic_model_only),
        use_rolling_supervision=bool(args.use_rolling_supervision),
        hist_max_error_m=args.hist_max_error_m,
        hist_bins=args.hist_bins,
        recommend_quantile=args.recommend_quantile,
        recommend_rounding_m=args.recommend_rounding_m,
    )
    thresholds = _parse_thresholds(args.thresholds)
    result = analyze_round_trip_distribution(
        cache_root=args.cache_root,
        split=args.split,
        cfg=cfg,
        thresholds=thresholds,
        max_files=args.max_files,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
        progress_interval=args.progress_interval,
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
