#!/usr/bin/env python3
"""Parity + timing: TF `compute_scenario_rollouts_features` vs Torch port.

This checks per-field max|Δ| for a few scenarios, and prints timing breakdown.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
tf.config.set_visible_devices([], "GPU")

from waymo_open_dataset.protos import scenario_pb2, sim_agents_submission_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features as tf_mf

from src.smart.metrics.wosac_metric_features_torch.metric_features_torch import (
    compute_scenario_rollouts_features as pt_compute_scenario_rollouts_features,
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


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    if a.dtype == np.bool_:
        return 0.0 if np.array_equal(a, b) else 1.0
    diff = np.abs(a - b)
    diff = diff[np.isfinite(diff)]
    return float(diff.max()) if diff.size else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Parity: TF vs torch MetricFeatures (stage ①)")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--g-rollouts", type=int, default=1)
    ap.add_argument("--tfrecord", type=Path, default=None)
    args = ap.parse_args()

    if args.tfrecord is not None:
        paths = [args.tfrecord]
    else:
        data_dir = _default_data_dir()
        paths = sorted(data_dir.glob("*.tfrecords"))[: args.n]
        if not paths:
            print("No TFRecords under", data_dir, file=sys.stderr)
            sys.exit(2)

    ct = submission_specs.ChallengeType.SIM_AGENTS
    worst: dict[str, float] = {}
    for p in paths:
        s = _load_scenario(str(p))
        j = _gt_joint_scene(s)
        roll = sim_agents_submission_pb2.ScenarioRollouts(
            joint_scenes=[j] * args.g_rollouts, scenario_id=s.scenario_id
        )

        t0 = time.perf_counter()
        tf_log, tf_sim = tf_mf.compute_scenario_rollouts_features(s, roll, ct)
        t_tf = time.perf_counter() - t0

        t0 = time.perf_counter()
        pt_log, pt_sim = pt_compute_scenario_rollouts_features(s, roll, challenge_type=ct)
        t_pt = time.perf_counter() - t0

        print(f"\nscenario={p.name}  G={args.g_rollouts}  tf={t_tf:.3f}s  pt={t_pt:.3f}s")

        for field in dataclasses.fields(tf_mf.MetricFeatures):
            name = field.name
            tf_a = getattr(tf_log, name).numpy()
            tf_b = getattr(tf_sim, name).numpy()
            pt_a = getattr(pt_log, name).detach().cpu().numpy()
            pt_b = getattr(pt_sim, name).detach().cpu().numpy()

            d_log = _max_abs(tf_a, pt_a)
            d_sim = _max_abs(tf_b, pt_b)
            worst[name + ":log"] = max(worst.get(name + ":log", 0.0), d_log)
            worst[name + ":sim"] = max(worst.get(name + ":sim", 0.0), d_sim)

        # show top few
        top = sorted(worst.items(), key=lambda kv: kv[1], reverse=True)[:8]
        print("top worst deltas:")
        for k, v in top:
            print(f"  {k:28s} {v:.3e}")

    print("\nworst per-field (max over checked scenarios):")
    for k, v in sorted(worst.items()):
        print(f"  {k:28s} {v:.3e}")


if __name__ == "__main__":
    main()

