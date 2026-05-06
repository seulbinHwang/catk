#!/usr/bin/env python3
"""Benchmark end-to-end runtime: official TF pipeline vs fully PyTorch pipeline.

Definitions:
- **official_total**: Waymo TF `metrics.compute_scenario_metrics_for_bundle`
  (includes TF `compute_scenario_rollouts_features` + TF metametric).
- **torch_total**: torch port of stage① (`wosac_metric_features_torch.compute_scenario_rollouts_features`)
  + stage② metametric (`compute_wosac_metametric_from_features_torch`).

This uses the same scenario + same rollouts (default: GT rollout replicated G times).

Examples::

  python scripts/benchmark_official_vs_torch_pipeline.py --n 3 --g-rollouts 1
  python scripts/benchmark_official_vs_torch_pipeline.py --tfrecord /path/one.tfrecords --g-rollouts 32
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
tf.config.set_visible_devices([], "GPU")

from waymo_open_dataset.protos import scenario_pb2, sim_agents_submission_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics as wm

from src.smart.metrics.wosac_metrics import WOSACMetrics
from src.smart.metrics.wosac_metametric_pytorch import compute_wosac_metametric_from_features_torch
from src.smart.metrics.wosac_metric_features_torch.metric_features_torch import (
    compute_scenario_rollouts_features as pt_rollouts_features,
)


def _load_scenario(path: str) -> scenario_pb2.Scenario:
    s = scenario_pb2.Scenario()
    for data in tf.data.TFRecordDataset([path], compression_type=""):
        s.ParseFromString(bytes(data.numpy()))
        break
    return s


def _gt_joint_scene(s: scenario_pb2.Scenario) -> sim_agents_submission_pb2.JointScene:
    ct = submission_specs.ChallengeType.SIM_AGENTS
    cfg = submission_specs.get_submission_config(ct)
    sim_ids = submission_specs.get_sim_agent_ids(s, ct)
    tracks = {t.id: t for t in s.tracks}
    t0 = cfg.current_time_index + 1
    traj = []
    for oid in sim_ids:
        tr = tracks[int(oid)]
        traj.append(
            sim_agents_submission_pb2.SimulatedTrajectory(
                object_id=int(oid),
                center_x=[tr.states[ti].center_x for ti in range(t0, t0 + cfg.n_simulation_steps)],
                center_y=[tr.states[ti].center_y for ti in range(t0, t0 + cfg.n_simulation_steps)],
                center_z=[tr.states[ti].center_z for ti in range(t0, t0 + cfg.n_simulation_steps)],
                heading=[tr.states[ti].heading for ti in range(t0, t0 + cfg.n_simulation_steps)],
            )
        )
    return sim_agents_submission_pb2.JointScene(simulated_trajectories=traj)


def _default_data_dir() -> Path:
    env = os.environ.get("WOSAC_PARITY_TFRECORD_DIR")
    if env:
        return Path(env)
    return (
        ROOT.parent
        / "datasets"
        / "smart_data"
        / "waymo_processed_catk_rebuild_parallel_v1"
        / "validation_tfrecords_splitted"
    )


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark: official TF vs full PyTorch WOSAC pipeline")
    ap.add_argument("--tfrecord", type=Path, default=None)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--g-rollouts", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    if args.tfrecord is not None:
        paths = [args.tfrecord]
    else:
        data_dir = _default_data_dir()
        paths = sorted(data_dir.glob("*.tfrecords"))[: args.n]
        if not paths:
            print("No TFRecords under", data_dir, file=sys.stderr)
            sys.exit(2)

    config = WOSACMetrics.load_metrics_config()
    ct = submission_specs.ChallengeType.SIM_AGENTS

    print(f"scenarios={len(paths)}  G={args.g_rollouts}  warmup={args.warmup}  repeat={args.repeat}")
    print(f"{'scenario':<40}  official_total(s)  torch_total(s)  official_rmm  torch_rmm")
    print("-" * 110)

    all_off_t: List[float] = []
    all_pt_t: List[float] = []

    for p in paths:
        s = _load_scenario(str(p))
        j = _gt_joint_scene(s)
        roll = sim_agents_submission_pb2.ScenarioRollouts(
            joint_scenes=[j] * args.g_rollouts,
            scenario_id=s.scenario_id,
        )

        # Warmup (don’t record)
        for _ in range(args.warmup):
            wm.compute_scenario_metrics_for_bundle(config, s, roll, challenge_type=ct)
            pt_log, pt_sim = pt_rollouts_features(s, roll, challenge_type=ct)
            log_d = pt_log.as_dict()
            sim_d = pt_sim.as_dict()
            del log_d["object_id"], sim_d["object_id"]
            compute_wosac_metametric_from_features_torch(config, log_d, sim_d)

        off_times: List[float] = []
        pt_times: List[float] = []
        off_rmm = None
        pt_rmm = None
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            off = wm.compute_scenario_metrics_for_bundle(config, s, roll, challenge_type=ct)
            off_times.append(time.perf_counter() - t0)
            off_rmm = float(off.metametric)

            t0 = time.perf_counter()
            pt_log, pt_sim = pt_rollouts_features(s, roll, challenge_type=ct)
            log_d = pt_log.as_dict()
            sim_d = pt_sim.as_dict()
            del log_d["object_id"], sim_d["object_id"]
            pt = compute_wosac_metametric_from_features_torch(config, log_d, sim_d)
            pt_times.append(time.perf_counter() - t0)
            pt_rmm = float(pt.metametric)

        off_m = _mean(off_times)
        pt_m = _mean(pt_times)
        all_off_t.append(off_m)
        all_pt_t.append(pt_m)

        print(
            f"{p.name[:38]:<40}  {off_m:15.4f}  {pt_m:12.4f}  {off_rmm:11.6f}  {pt_rmm:9.6f}"
        )

    print("-" * 110)
    print(f"mean over scenarios:           {_mean(all_off_t):15.4f}  {_mean(all_pt_t):12.4f}")
    if all_off_t and all_pt_t:
        speedup = _mean(all_off_t) / max(_mean(all_pt_t), 1e-12)
        print(f"speedup (official / torch):   {speedup:8.2f}x")


if __name__ == "__main__":
    main()

