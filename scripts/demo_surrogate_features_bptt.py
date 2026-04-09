#!/usr/bin/env python3
"""Tiny demo: gradients through surrogate per-step events (collision/offroad/TL).

This does NOT backprop through full map/interaction extraction; it demonstrates
that when surrogate outputs are used (float probs), the downstream soft metametric
is differentiable and can be optimized.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.smart.metrics.wosac_metrics import WOSACMetrics
from src.smart.metrics.wosac_metametric_pytorch_differentiable import compute_wosac_metametric_soft


def main() -> None:
    ap = argparse.ArgumentParser(description="Demo: optimize soft metametric through surrogate event probs")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=8, help="objects")
    ap.add_argument("--t", type=int, default=80, help="timesteps")
    ap.add_argument("--g", type=int, default=16, help="rollouts")
    args = ap.parse_args()

    device = torch.device("cpu")
    dtype = torch.float32
    config = WOSACMetrics.load_metrics_config()

    # Build minimal log/sim dicts expected by compute_wosac_metametric_soft.
    # We keep continuous histogram features fixed; we only learn event probs.
    N, T, G = args.n, args.t, args.g
    valid_log = torch.ones((1, N, T), dtype=dtype, device=device)
    object_type = torch.full((1, N), 1, dtype=torch.int64, device=device)

    def fixed_ts(minv, maxv):
        return torch.full((1, N, T), (minv + maxv) * 0.5, dtype=dtype, device=device)

    log = {
        "valid": valid_log,
        "object_type": object_type,
        "linear_speed": fixed_ts(config.linear_speed.histogram.min_val, config.linear_speed.histogram.max_val),
        "linear_acceleration": fixed_ts(config.linear_acceleration.histogram.min_val, config.linear_acceleration.histogram.max_val),
        "angular_speed": fixed_ts(config.angular_speed.histogram.min_val, config.angular_speed.histogram.max_val),
        "angular_acceleration": fixed_ts(config.angular_acceleration.histogram.min_val, config.angular_acceleration.histogram.max_val),
        "distance_to_nearest_object": fixed_ts(config.distance_to_nearest_object.histogram.min_val, config.distance_to_nearest_object.histogram.max_val),
        "time_to_collision": fixed_ts(config.time_to_collision.histogram.min_val, config.time_to_collision.histogram.max_val),
        "distance_to_road_edge": fixed_ts(config.distance_to_road_edge.histogram.min_val, config.distance_to_road_edge.histogram.max_val),
        "collision_per_step": torch.zeros((1, N, T), dtype=dtype, device=device),
        "offroad_per_step": torch.zeros((1, N, T), dtype=dtype, device=device),
        "traffic_light_violation_per_step": torch.zeros((1, N, T), dtype=dtype, device=device),
    }

    sim_cont = {
        k: torch.repeat_interleave(v, repeats=G, dim=0).contiguous()
        for k, v in log.items()
        if k
        in (
            "linear_speed",
            "linear_acceleration",
            "angular_speed",
            "angular_acceleration",
            "distance_to_nearest_object",
            "time_to_collision",
            "distance_to_road_edge",
        )
    }

    # Learnable logits for per-step event probabilities (surrogate outputs)
    col_logits = torch.nn.Parameter(torch.randn((G, N, T), dtype=dtype, device=device) * 0.5)
    off_logits = torch.nn.Parameter(torch.randn((G, N, T), dtype=dtype, device=device) * 0.5)
    tl_logits = torch.nn.Parameter(torch.randn((G, N, T), dtype=dtype, device=device) * 0.5)
    opt = torch.optim.Adam([col_logits, off_logits, tl_logits], lr=args.lr)

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        sim = dict(sim_cont)
        sim["collision_per_step"] = torch.sigmoid(col_logits)
        sim["offroad_per_step"] = torch.sigmoid(off_logits)
        sim["traffic_light_violation_per_step"] = torch.sigmoid(tl_logits)

        out = compute_wosac_metametric_soft(config, log, sim)
        loss = -out.metametric
        loss.backward()
        opt.step()

        if step == 1 or step % 50 == 0 or step == args.steps:
            print(f"step {step:4d}  soft_rmm={float(out.metametric.detach()):.6f}")


if __name__ == "__main__":
    main()

