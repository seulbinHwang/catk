from __future__ import annotations

import argparse
import json
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.smart.modules.kinematic_control import build_rolling_control_target_with_round_trip_error


TYPE_IDS = {"all": None, "vehicle": 0, "pedestrian": 1, "cyclist": 2}


@dataclass(frozen=True)
class AnalysisConfig:
    flow_window_steps: int = 20
    shift: int = 5
    raw_start: int = 10
    raw_end: int = 70
    use_prefix_valid_future_loss_mask: bool = False
    control_pos_scale_m: float = 1.0
    control_vehicle_yaw_scale_rad: float = 0.5
    control_pedestrian_yaw_scale_rad: float = 0.5
    control_cyclist_yaw_scale_rad: float = 0.5
    control_vehicle_no_slip_point_ratio: float = 0.0
    control_cyclist_no_slip_point_ratio: float = 0.0
    use_holonomic_model_only: bool = False
    use_rolling_supervision: bool = True
    hist_max_error_m: float = 20.0
    hist_bins: int = 20_000
    recommend_quantile: float = 99.5
    recommend_rounding_m: float = 0.25


def _available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def _default_num_workers() -> int:
    return min(64, max(1, 3 * _available_cpu_count()))


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def _load_agent_record(path: Path) -> dict[str, torch.Tensor]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    agent = data["agent"]
    pos = _as_tensor(agent["position"])[..., :2].to(dtype=torch.float32)
    valid = _as_tensor(agent["valid_mask"]).to(dtype=torch.bool)
    if "velocity" in agent:
        velocity = _as_tensor(agent["velocity"])[..., :2].to(dtype=torch.float32)
    else:
        velocity = torch.zeros_like(pos)
        if pos.shape[1] > 1:
            pair_valid = valid[:, 1:] & valid[:, :-1]
            diff_velocity = (pos[:, 1:] - pos[:, :-1]) / 0.1
            velocity[:, 1:] = torch.where(pair_valid.unsqueeze(-1), diff_velocity, torch.zeros_like(diff_velocity))
    record = {
        "pos": pos,
        "heading": _as_tensor(agent["heading"]).to(dtype=torch.float32),
        "velocity": velocity,
        "valid": valid,
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
    future_loss_mask[:, :available_len] = prefix_valid
    return future_loss_mask


def _build_future_window(
    *,
    pos: torch.Tensor,
    heading: torch.Tensor,
    velocity: torch.Tensor,
    current_pos: torch.Tensor,
    current_head: torch.Tensor,
    anchor_mask: torch.Tensor,
    raw_step: int,
    future_loss_mask: torch.Tensor,
    cfg: AnalysisConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_current_pos = current_pos[anchor_mask]
    selected_current_head = current_head[anchor_mask]
    future_start = raw_step + 1

    future_pos = selected_current_pos.unsqueeze(1).expand(-1, cfg.flow_window_steps, -1).clone()
    future_head = selected_current_head.unsqueeze(1).expand(-1, cfg.flow_window_steps).clone()
    future_velocity = velocity[anchor_mask, raw_step].unsqueeze(1).expand(
        -1,
        cfg.flow_window_steps,
        -1,
    ).clone()

    available_len = min(cfg.flow_window_steps, max(0, pos.shape[1] - future_start))
    if available_len > 0:
        future_pos[:, :available_len] = pos[anchor_mask, future_start : future_start + available_len]
        future_head[:, :available_len] = heading[anchor_mask, future_start : future_start + available_len]
        future_velocity[:, :available_len] = velocity[anchor_mask, future_start : future_start + available_len]

    valid_step_count = future_loss_mask.long().sum(dim=1)
    if bool((valid_step_count <= 0).any().item()):
        raise ValueError("future_loss_mask must contain at least one valid future step per selected anchor.")

    last_valid_index = valid_step_count - 1
    last_valid_pos = future_pos.gather(
        dim=1,
        index=last_valid_index.view(-1, 1, 1).expand(-1, 1, future_pos.shape[-1]),
    ).squeeze(1)
    last_valid_head = future_head.gather(dim=1, index=last_valid_index.view(-1, 1)).squeeze(1)
    last_valid_velocity = future_velocity.gather(
        dim=1,
        index=last_valid_index.view(-1, 1, 1).expand(-1, 1, future_velocity.shape[-1]),
    ).squeeze(1)
    invalid_future_mask = ~future_loss_mask
    future_pos = torch.where(invalid_future_mask.unsqueeze(-1), last_valid_pos.unsqueeze(1), future_pos)
    future_head = torch.where(invalid_future_mask, last_valid_head.unsqueeze(1), future_head)
    future_velocity = torch.where(
        invalid_future_mask.unsqueeze(-1),
        last_valid_velocity.unsqueeze(1),
        future_velocity,
    )
    return future_pos, future_head, future_velocity


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
    }


def _update_stats(
    stats: dict[str, Any],
    anchor_max: torch.Tensor,
    step_error: torch.Tensor,
    thresholds: np.ndarray,
    cfg: AnalysisConfig,
) -> None:
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
    stats["threshold_keep_counts"] += (anchor_np[:, None] <= thresholds[None, :]).sum(axis=0)
    stats["sum_anchor_max_error_m"] += float(anchor_np.sum())
    stats["sum_step_error_m"] += float(step_np.sum())
    stats["max_anchor_error_m"] = max(float(stats["max_anchor_error_m"]), float(anchor_np.max(initial=0.0)))
    stats["max_step_error_m"] = max(float(stats["max_step_error_m"]), float(step_np.max(initial=0.0)))


def analyze_cache_file(path: Path, cfg: AnalysisConfig, thresholds: np.ndarray) -> dict[str, Any]:
    record = _load_agent_record(path)
    stats = {type_name: _new_type_stats(cfg, thresholds) for type_name in TYPE_IDS}
    pos = record["pos"]
    heading = record["heading"]
    velocity = record["velocity"]
    valid = record["valid"]
    agent_type = record["type"]
    agent_length = record["length"]
    train_mask = record["train_mask"]

    max_raw_step = min(cfg.raw_end, valid.shape[1] - cfg.flow_window_steps - 1)
    for raw_step in range(cfg.raw_start, max_raw_step + 1, cfg.shift):
        current_valid = valid[:, raw_step]
        future_loss_mask_all = _build_future_loss_mask(valid=valid, raw_step=raw_step, cfg=cfg)
        anchor_mask = current_valid & future_loss_mask_all.any(dim=1) & train_mask
        if not bool(anchor_mask.any().item()):
            continue

        selected_future_loss_mask = future_loss_mask_all[anchor_mask]
        future_pos, future_head, future_velocity = _build_future_window(
            pos=pos,
            heading=heading,
            velocity=velocity,
            current_pos=pos[:, raw_step],
            current_head=heading[:, raw_step],
            anchor_mask=anchor_mask,
            raw_step=raw_step,
            future_loss_mask=selected_future_loss_mask,
            cfg=cfg,
        )
        selected_type = agent_type[anchor_mask]
        selected_length = agent_length[anchor_mask]
        current_speed = torch.linalg.vector_norm(velocity[anchor_mask, raw_step, :2], dim=-1)
        _, round_trip_error_m = build_rolling_control_target_with_round_trip_error(
            future_pos=future_pos,
            future_head=future_head,
            current_pos=pos[anchor_mask, raw_step],
            current_head=heading[anchor_mask, raw_step],
            agent_type=selected_type,
            agent_length=selected_length,
            current_speed=current_speed,
            future_velocity=future_velocity,
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
            selected_future_loss_mask,
            round_trip_error_m,
            torch.zeros_like(round_trip_error_m),
        )
        anchor_max = masked_error.max(dim=1).values
        step_error = masked_error[selected_future_loss_mask]
        _update_stats(stats["all"], anchor_max, step_error, thresholds, cfg)
        for type_name, type_id in TYPE_IDS.items():
            if type_id is None:
                continue
            type_mask = selected_type == int(type_id)
            if not bool(type_mask.any().item()):
                continue
            _update_stats(
                stats[type_name],
                anchor_max[type_mask],
                masked_error[type_mask][selected_future_loss_mask[type_mask]],
                thresholds,
                cfg,
            )

    return stats


def _merge_stats(target: dict[str, Any], source: dict[str, Any]) -> None:
    for type_name in TYPE_IDS:
        for key, value in source[type_name].items():
            if isinstance(value, np.ndarray):
                target[type_name][key] += value
            elif key.startswith("max_"):
                target[type_name][key] = max(target[type_name][key], value)
            else:
                target[type_name][key] += value


def _hist_percentile(hist: np.ndarray, cfg: AnalysisConfig, percentile: float) -> float:
    total = int(hist.sum())
    if total == 0:
        return 0.0
    rank = max(1, int(np.ceil(total * float(percentile) / 100.0)))
    idx = int(np.searchsorted(np.cumsum(hist), rank, side="left"))
    return (idx + 0.5) * float(cfg.hist_max_error_m) / float(cfg.hist_bins)


def _finalize_stats(stats: dict[str, Any], thresholds: np.ndarray, cfg: AnalysisConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for type_name, raw in stats.items():
        anchor_count = int(raw["anchor_count"])
        loss_step_count = int(raw["loss_step_count"])
        threshold_rows = []
        for threshold, keep_count in zip(thresholds, raw["threshold_keep_counts"], strict=True):
            keep_pct = 100.0 * float(keep_count) / float(anchor_count) if anchor_count else 0.0
            threshold_rows.append(
                {
                    "threshold_m": float(threshold),
                    "keep_count": int(keep_count),
                    "drop_count": int(anchor_count - int(keep_count)),
                    "keep_pct": keep_pct,
                    "drop_pct": 100.0 - keep_pct if anchor_count else 0.0,
                }
            )
        recommended = _hist_percentile(raw["hist_anchor_max"], cfg, cfg.recommend_quantile)
        rounding = float(cfg.recommend_rounding_m)
        if rounding > 0.0:
            recommended = float(np.ceil(recommended / rounding) * rounding)
        result[type_name] = {
            "anchor_count": anchor_count,
            "loss_step_count": loss_step_count,
            "anchor_max_error_mean_m": (
                float(raw["sum_anchor_max_error_m"]) / float(anchor_count) if anchor_count else 0.0
            ),
            "step_error_mean_m": (
                float(raw["sum_step_error_m"]) / float(loss_step_count) if loss_step_count else 0.0
            ),
            "max_anchor_error_m": float(raw["max_anchor_error_m"]),
            "max_step_error_m": float(raw["max_step_error_m"]),
            "anchor_max_error_percentiles": {
                "p50_m": _hist_percentile(raw["hist_anchor_max"], cfg, 50.0),
                "p95_m": _hist_percentile(raw["hist_anchor_max"], cfg, 95.0),
                "p99_m": _hist_percentile(raw["hist_anchor_max"], cfg, 99.0),
                "p99_5_m": _hist_percentile(raw["hist_anchor_max"], cfg, 99.5),
            },
            "threshold_table": threshold_rows,
            "recommended_threshold_m": recommended,
            "recommended_quantile": float(cfg.recommend_quantile),
            "recommended_keep_pct_approx": (
                100.0 * float((raw["hist_anchor_max"].cumsum()[
                    min(int(recommended / float(cfg.hist_max_error_m) * int(cfg.hist_bins)), int(cfg.hist_bins) - 1)
                ])) / float(anchor_count)
                if anchor_count
                else 0.0
            ),
        }
    return result


def list_cache_files(cache_root: Path, split: str, max_files: int | None) -> list[Path]:
    split_dir = cache_root / split
    files = sorted(path for path in split_dir.glob("*.pkl") if path.is_file() and not path.name.startswith("."))
    if max_files is not None:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No cache pickle files found under: {split_dir}")
    return files


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    thresholds = np.array([float(item) for item in str(args.thresholds).split(",") if item], dtype=np.float64)
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
    files = list_cache_files(args.cache_root, args.split, args.max_files)
    merged = {type_name: _new_type_stats(cfg, thresholds) for type_name in TYPE_IDS}
    worker = partial(analyze_cache_file, cfg=cfg, thresholds=thresholds)
    if int(args.num_workers) <= 1:
        for path in files:
            _merge_stats(merged, worker(path))
    else:
        with ProcessPoolExecutor(max_workers=int(args.num_workers)) as pool:
            for partial_stats in pool.map(worker, files, chunksize=int(args.chunksize)):
                _merge_stats(merged, partial_stats)
    return {
        "cache_root": str(args.cache_root),
        "split": args.split,
        "file_count": len(files),
        "summary": _finalize_stats(merged, thresholds, cfg),
    }


def print_report(result: dict[str, Any]) -> None:
    print(f"cache_root: {result['cache_root']}")
    print(f"split: {result['split']} files={result['file_count']}")
    for type_name in TYPE_IDS:
        stats = result["summary"][type_name]
        p = stats["anchor_max_error_percentiles"]
        print(
            f"{type_name}: anchors={stats['anchor_count']} steps={stats['loss_step_count']} "
            f"mean={stats['anchor_max_error_mean_m']:.4f}m "
            f"p95={p['p95_m']:.4f}m p99={p['p99_m']:.4f}m "
            f"p99_5={p['p99_5_m']:.4f}m max={stats['max_anchor_error_m']:.4f}m"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze MDG-style [acceleration, yaw_rate] target round-trip position error."
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ["CACHE_ROOT"]) if "CACHE_ROOT" in os.environ else None,
    )
    parser.add_argument("--split", default="training")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=_default_num_workers())
    parser.add_argument("--chunksize", type=int, default=512)
    parser.add_argument("--flow-window-steps", type=int, default=20)
    parser.add_argument("--shift", type=int, default=5)
    parser.add_argument("--raw-start", type=int, default=10)
    parser.add_argument("--raw-end", type=int, default=70)
    parser.add_argument("--use-prefix-valid-future-loss-mask", action="store_true")
    parser.add_argument("--control-pos-scale-m", type=float, default=1.0)
    parser.add_argument("--control-vehicle-yaw-scale-rad", type=float, default=0.5)
    parser.add_argument("--control-pedestrian-yaw-scale-rad", type=float, default=0.5)
    parser.add_argument("--control-cyclist-yaw-scale-rad", type=float, default=0.5)
    parser.add_argument("--control-vehicle-no-slip-point-ratio", type=float, default=0.0)
    parser.add_argument("--control-cyclist-no-slip-point-ratio", type=float, default=0.0)
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
    result = run_analysis(args)
    print_report(result)
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
