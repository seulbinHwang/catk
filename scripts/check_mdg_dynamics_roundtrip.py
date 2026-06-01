#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.mdg.data import MDGDataset, collate_mdg_samples
from src.mdg.modules import KinematicDynamics


HISTORY_STEPS = 11


def wrap_abs_error(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle)).abs()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure MDG trajectory->action->dynamics round-trip error."
    )
    parser.add_argument("--cache-root", default="/workspace/womd_v1_3/MDG_cache")
    parser.add_argument("--split", default="validation", choices=["training", "validation", "testing"])
    parser.add_argument("--max-scenarios", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-agents", type=int, default=64)
    parser.add_argument(
        "--eval-all-agents",
        action="store_true",
        help="Use all current-valid cached agents instead of --max-agents.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = Path(args.cache_root) / args.split
    max_agents = None if args.eval_all_agents else args.max_agents
    dataset = MDGDataset(
        raw_dir=str(split_dir),
        max_agents=max_agents,
        max_map_polylines=320,
        map_waypoints=16,
        max_traffic_lights=16,
        training=args.split == "training",
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No cached scenarios found in {split_dir}")
    count = min(args.max_scenarios, len(dataset))
    loader = DataLoader(
        Subset(dataset, range(count)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_mdg_samples,
        pin_memory=False,
    )
    dynamics = KinematicDynamics(
        action_chunk=2,
        dt=0.1,
        action_mean=(0.0, 0.0),
        action_std=(1.0, 0.5),
    )
    ade_sum = torch.tensor(0.0)
    fde_sum = torch.tensor(0.0)
    heading_sum = torch.tensor(0.0)
    speed_sum = torch.tensor(0.0)
    neg_speed_sum = torch.tensor(0.0)
    denom_sum = torch.tensor(0.0)
    final_denom_sum = torch.tensor(0.0)

    with torch.no_grad():
        for batch in loader:
            current_pos = batch["agent_position"][:, :, HISTORY_STEPS - 1, :2]
            current_heading = batch["agent_heading"][:, :, HISTORY_STEPS - 1]
            current_speed = torch.linalg.norm(
                batch["agent_velocity"][:, :, HISTORY_STEPS - 1, :2],
                dim=-1,
            )
            current_velocity = batch["agent_velocity"][:, :, HISTORY_STEPS - 1, :2]
            future_pos = batch["agent_position"][:, :, HISTORY_STEPS:, :2]
            future_heading = batch["agent_heading"][:, :, HISTORY_STEPS:]
            future_velocity = batch["agent_velocity"][:, :, HISTORY_STEPS:, :2]
            valid = batch["agent_valid"].unsqueeze(-1) & batch["agent_valid_mask"][:, :, HISTORY_STEPS:]

            action = dynamics.trajectory_to_actions(
                current_pos=current_pos,
                current_heading=current_heading,
                current_speed=current_speed,
                future_pos=future_pos,
                future_heading=future_heading,
                future_velocity=future_velocity,
            )
            pred_pos, pred_heading, pred_speed, _, _ = dynamics(
                action,
                current_pos,
                current_heading,
                current_speed,
                current_velocity=current_velocity,
            )
            target_speed = torch.linalg.norm(future_velocity, dim=-1)
            valid_f = valid.to(dtype=pred_pos.dtype)
            denom = valid_f.sum().clamp_min(1.0)

            ade = torch.linalg.norm(pred_pos - future_pos, dim=-1)
            heading_error = wrap_abs_error(pred_heading - future_heading)
            speed_error = (pred_speed - target_speed).abs()
            ade_sum += (ade * valid_f).sum()
            heading_sum += (heading_error * valid_f).sum()
            speed_sum += (speed_error * valid_f).sum()
            neg_speed_sum += ((pred_speed < 0.0) & valid).to(dtype=pred_pos.dtype).sum()
            denom_sum += denom

            final_valid = valid[:, :, -1].to(dtype=pred_pos.dtype)
            final_denom = final_valid.sum().clamp_min(1.0)
            fde_sum += (ade[:, :, -1] * final_valid).sum()
            final_denom_sum += final_denom

    metrics = {
        "split": args.split,
        "scenarios": count,
        "agents": "all_current_valid" if args.eval_all_agents else str(args.max_agents),
        "roundtrip_ADE_m": float(ade_sum / denom_sum.clamp_min(1.0)),
        "roundtrip_FDE_m": float(fde_sum / final_denom_sum.clamp_min(1.0)),
        "heading_error_rad": float(heading_sum / denom_sum.clamp_min(1.0)),
        "speed_error_mps": float(speed_sum / denom_sum.clamp_min(1.0)),
        "negative_speed_rate": float(neg_speed_sum / denom_sum.clamp_min(1.0)),
    }
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
