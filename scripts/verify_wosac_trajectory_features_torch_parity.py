#!/usr/bin/env python3
"""Parity test: TF vs Torch `trajectory_features`.

This verifies our new implementation in:
`src/smart/metrics/wosac_metric_features_torch/trajectory_features_torch.py`
matches Waymo TF:
`waymo_open_dataset.wdl_limited.sim_agents_metrics.trajectory_features`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
import torch

from waymo_open_dataset.wdl_limited.sim_agents_metrics import trajectory_features as tf_traj

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.smart.metrics.wosac_metric_features_torch import trajectory_features_torch as pt_traj


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(a - b)
    # Ignore NaNs (both should be NaN in same places)
    diff = diff[np.isfinite(diff)]
    return float(diff.max()) if diff.size else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Parity: TF vs torch trajectory_features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shape", type=str, default="(8,91)", help="e.g. '(N,T)' or '(B,N,T)'")
    ap.add_argument("--step-seconds", type=float, default=0.1)
    ap.add_argument("--tol", type=float, default=2e-5)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    shape = tuple(int(x.strip()) for x in args.shape.strip()[1:-1].split(",") if x.strip())
    if not shape or len(shape) < 2:
        raise ValueError("shape must have at least 2 dims, e.g. (N,T)")

    # Random trajectories (heading in [-pi, pi])
    x = rng.normal(size=shape).astype(np.float32)
    y = rng.normal(size=shape).astype(np.float32)
    z = rng.normal(size=shape).astype(np.float32)
    heading = (rng.uniform(-np.pi, np.pi, size=shape)).astype(np.float32)
    valid = (rng.uniform(size=shape) > 0.2)

    # TF
    tf_ls, tf_la, tf_as, tf_aa = tf_traj.compute_kinematic_features(
        tf.constant(x), tf.constant(y), tf.constant(z), tf.constant(heading), args.step_seconds
    )
    tf_sv, tf_av = tf_traj.compute_kinematic_validity(tf.constant(valid))
    tf_de = tf_traj.compute_displacement_error(
        tf.constant(x), tf.constant(y), tf.constant(z),
        tf.constant(x * 0.9), tf.constant(y * 0.9), tf.constant(z * 0.9),
    )

    # Torch
    tx = torch.from_numpy(x)
    ty = torch.from_numpy(y)
    tz = torch.from_numpy(z)
    th = torch.from_numpy(heading)
    tv = torch.from_numpy(valid)

    pt_ls, pt_la, pt_as, pt_aa = pt_traj.compute_kinematic_features(tx, ty, tz, th, args.step_seconds)
    pt_sv, pt_av = pt_traj.compute_kinematic_validity(tv)
    pt_de = pt_traj.compute_displacement_error(tx, ty, tz, tx * 0.9, ty * 0.9, tz * 0.9)

    # Compare
    items = [
        ("linear_speed", tf_ls.numpy(), pt_ls.numpy()),
        ("linear_accel", tf_la.numpy(), pt_la.numpy()),
        ("angular_speed", tf_as.numpy(), pt_as.numpy()),
        ("angular_accel", tf_aa.numpy(), pt_aa.numpy()),
        ("disp_error", tf_de.numpy(), pt_de.numpy()),
    ]
    ok = True
    for name, a, b in items:
        d = _max_abs(a, b)
        flag = "ok" if d <= args.tol else "FAIL"
        print(f"max|Δ| {name:14s} {d:.3e}  {flag}")
        ok = ok and d <= args.tol

    # validity is bool exact match
    sv_ok = np.array_equal(tf_sv.numpy(), pt_sv.numpy())
    av_ok = np.array_equal(tf_av.numpy(), pt_av.numpy())
    print(f"exact speed_validity:        {'ok' if sv_ok else 'FAIL'}")
    print(f"exact accel_validity:        {'ok' if av_ok else 'FAIL'}")
    ok = ok and sv_ok and av_ok

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

