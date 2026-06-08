#!/usr/bin/env python3
"""Precompute deterministic semi_mdg token/flow targets as cache sidecars.

The sidecar is intentionally training-only. It stores deterministic map tokens,
coarse context tokens, and dense agent x anchor flow targets. Runtime training
packs those dense fields back to the exact anchor-major order produced by the
original on-the-fly token processor.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.smart.datamodules.target_builder import WaymoTargetBuilderVal  # noqa: E402
from src.smart.tokens.flow_token_processor import (  # noqa: E402
    FLOW_TRAIN_ANCHOR_COUNT,
    FlowTokenProcessor,
)


SIDECAR_VERSION = "semi_mdg_token_flow_sidecar_v2_accel_yawrate"


def list_cache_files(cache_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(
        path for path in cache_dir.glob("*.pkl") if path.is_file() and not path.name.startswith(".")
    )
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No cache pickle files found under: {cache_dir}")
    return files


def cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous()


def unpack_anchor_dense(
    packed: torch.Tensor,
    anchor_mask: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    dense = packed.new_zeros(shape)
    if packed.numel() == 0:
        return dense
    dense_anchor_first = dense.permute(1, 0, *range(2, dense.ndim))
    dense_anchor_first[anchor_mask.t()] = packed
    return dense


def build_sidecar(
    data: HeteroData,
    tokenized_map: dict[str, torch.Tensor],
    tokenized_agent: dict[str, torch.Tensor],
    flow_window_steps: int,
    config_summary: dict[str, Any],
) -> dict[str, Any]:
    flow_mask = tokenized_agent["flow_train_mask"].bool()
    num_agent = int(flow_mask.shape[0])
    num_anchor = int(flow_mask.shape[1])
    if num_anchor != FLOW_TRAIN_ANCHOR_COUNT:
        raise ValueError(f"Expected {FLOW_TRAIN_ANCHOR_COUNT} anchors, got {num_anchor}.")

    clean_norm = tokenized_agent["flow_train_clean_norm"]
    clean_metric_norm = tokenized_agent["flow_train_clean_metric_norm"]
    loss_mask = tokenized_agent["flow_train_loss_mask"].bool()
    current_speed = tokenized_agent["flow_train_current_speed"]
    clean_norm_dense = unpack_anchor_dense(
        packed=clean_norm,
        anchor_mask=flow_mask,
        shape=(num_agent, num_anchor, flow_window_steps, clean_norm.shape[-1]),
    )
    clean_metric_norm_dense = unpack_anchor_dense(
        packed=clean_metric_norm,
        anchor_mask=flow_mask,
        shape=(num_agent, num_anchor, flow_window_steps, clean_metric_norm.shape[-1]),
    )
    loss_mask_dense = unpack_anchor_dense(
        packed=loss_mask,
        anchor_mask=flow_mask,
        shape=(num_agent, num_anchor, flow_window_steps),
    ).bool()
    current_speed_dense = unpack_anchor_dense(
        packed=current_speed,
        anchor_mask=flow_mask,
        shape=(num_agent, num_anchor),
    )

    return {
        "version": SIDECAR_VERSION,
        "scenario_id": str(data["scenario_id"]),
        "config": config_summary,
        "agent": {
            "ctx_token": cpu_tensor(tokenized_agent["ctx_sampled_idx"]).long(),
            "ctx_pos": cpu_tensor(tokenized_agent["ctx_sampled_pos"]).float(),
            "ctx_heading": cpu_tensor(tokenized_agent["ctx_sampled_heading"]).float(),
            "ctx_valid": cpu_tensor(tokenized_agent["ctx_valid"]).bool(),
            "token_agent_shape": cpu_tensor(tokenized_agent["token_agent_shape"]).float(),
            "flow_mask": cpu_tensor(flow_mask).bool(),
            "flow_clean_norm_dense": cpu_tensor(clean_norm_dense).float(),
            "flow_clean_metric_norm_dense": cpu_tensor(clean_metric_norm_dense).float(),
            "flow_loss_mask_dense": cpu_tensor(loss_mask_dense).bool(),
            "flow_current_speed_dense": cpu_tensor(current_speed_dense).float(),
        },
        "pt_token": {
            "map_token": cpu_tensor(tokenized_map["token_idx"]).long(),
        },
    }


def load_processor(experiment: str, device: torch.device) -> FlowTokenProcessor:
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="run", overrides=[f"experiment={experiment}"])
    processor_cfg = OmegaConf.to_container(
        cfg.model.model_config.token_processor,
        resolve=True,
    )
    processor = FlowTokenProcessor(**processor_cfg)
    processor.train()
    processor.to(device)
    return processor


def processor_config_summary(processor: FlowTokenProcessor) -> dict[str, Any]:
    return {
        "map_token_file": "map_traj_token5.pkl",
        "agent_token_file": "agent_vocab_555_s2.pkl",
        "flow_window_steps": int(processor.flow_window_steps),
        "use_prefix_valid_future_loss_mask": bool(processor.use_prefix_valid_future_loss_mask),
        "use_kinematic_control_flow": bool(processor.use_kinematic_control_flow),
        "use_rolling_supervision": bool(processor.use_rolling_supervision),
        "control_pos_scale_m": float(processor.control_pos_scale_m),
        "control_vehicle_yaw_scale_rad": float(processor.control_vehicle_yaw_scale_rad),
        "control_pedestrian_yaw_scale_rad": float(processor.control_pedestrian_yaw_scale_rad),
        "control_cyclist_yaw_scale_rad": float(processor.control_cyclist_yaw_scale_rad),
        "control_vehicle_no_slip_point_ratio": float(processor.control_vehicle_no_slip_point_ratio),
        "control_cyclist_no_slip_point_ratio": float(processor.control_cyclist_no_slip_point_ratio),
        "control_round_trip_max_position_error_m": float(
            processor.control_round_trip_max_position_error_m
        ),
    }


def add_single_graph_batch_fields(data) -> None:
    data.num_graphs = 1
    if "batch" not in data["agent"]:
        num_agent = int(data["agent"]["position"].shape[0])
        data["agent"]["batch"] = torch.zeros(num_agent, dtype=torch.long)
    if "batch" not in data["pt_token"]:
        num_polyline = int(data["pt_token"]["type"].shape[0])
        data["pt_token"]["batch"] = torch.zeros(num_polyline, dtype=torch.long)


def iter_shard(files: list[Path], num_shards: int, shard_index: int) -> Iterable[Path]:
    if num_shards <= 1:
        return files
    return (path for idx, path in enumerate(files) if idx % num_shards == shard_index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment", default="mdg_pretrain_h100x3x2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    files = list_cache_files(args.cache_dir, args.limit)
    files = list(iter_shard(files, args.num_shards, args.shard_index))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    processor = load_processor(args.experiment, device=device)
    config_summary = processor_config_summary(processor)
    transform = WaymoTargetBuilderVal()

    written = 0
    skipped = 0
    for cache_path in tqdm(files, desc="semi_mdg sidecar", dynamic_ncols=True):
        with open(cache_path, "rb") as handle:
            raw = pickle.load(handle)
        scenario_id = str(raw["scenario_id"])
        output_path = args.output_dir / f"{scenario_id}.pkl"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        data = transform(raw)
        add_single_graph_batch_fields(data)
        data = data.to(device)
        with torch.no_grad():
            tokenized_map, tokenized_agent = processor(data)
        sidecar = build_sidecar(
            data=data,
            tokenized_map=tokenized_map,
            tokenized_agent=tokenized_agent,
            flow_window_steps=processor.flow_window_steps,
            config_summary=config_summary,
        )
        tmp_path = output_path.with_suffix(".tmp")
        with open(tmp_path, "wb") as handle:
            pickle.dump(sidecar, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(output_path)
        written += 1

    print(
        f"sidecar complete: written={written}, skipped={skipped}, "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
