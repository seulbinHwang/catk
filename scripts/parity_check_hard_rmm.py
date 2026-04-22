#!/usr/bin/env python3
"""Hard RMM (PyTorch) vs Official WOSAC TF — per-metric parity check.

Usage:
    python scripts/parity_check_hard_rmm.py --n 50 --g 32
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from waymo_open_dataset.protos import scenario_pb2, sim_agents_submission_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics as wm

from src.smart.metrics.wosac_metametric_pytorch import compute_wosac_metametric_from_features_torch
from src.smart.metrics.wosac_metrics import WOSACMetrics

DATA_DIR = (
    ROOT.parent
    / "datasets"
    / "smart_data"
    / "waymo_processed_catk_rebuild_parallel_v1"
    / "validation_tfrecords_splitted"
)

ATTRS = [
    "linear_speed_likelihood",
    "linear_acceleration_likelihood",
    "angular_speed_likelihood",
    "angular_acceleration_likelihood",
    "distance_to_nearest_object_likelihood",
    "collision_indication_likelihood",
    "time_to_collision_likelihood",
    "distance_to_road_edge_likelihood",
    "offroad_indication_likelihood",
    "traffic_light_violation_likelihood",
    "metametric",
]


def _mf_to_dict(mf) -> dict:
    d = {}
    for field in dataclasses.fields(mf):
        arr = getattr(mf, field.name).numpy()
        if arr.dtype == np.bool_:
            d[field.name] = torch.from_numpy(arr)
        else:
            d[field.name] = torch.from_numpy(arr.astype(np.float32))
    return d


def _gt_rollouts(s: scenario_pb2.Scenario, g: int) -> sim_agents_submission_pb2.ScenarioRollouts:
    ct = submission_specs.ChallengeType.SIM_AGENTS
    cfg = submission_specs.get_submission_config(ct)
    sim_ids = submission_specs.get_sim_agent_ids(s, ct)
    tracks = {t.id: t for t in s.tracks}
    traj = []
    t0 = cfg.current_time_index + 1
    for oid in sim_ids:
        tr = tracks[oid]
        traj.append(
            sim_agents_submission_pb2.SimulatedTrajectory(
                object_id=oid,
                center_x=[tr.states[ti].center_x for ti in range(t0, t0 + cfg.n_simulation_steps)],
                center_y=[tr.states[ti].center_y for ti in range(t0, t0 + cfg.n_simulation_steps)],
                center_z=[tr.states[ti].center_z for ti in range(t0, t0 + cfg.n_simulation_steps)],
                heading=[tr.states[ti].heading for ti in range(t0, t0 + cfg.n_simulation_steps)],
            )
        )
    joint = sim_agents_submission_pb2.JointScene(simulated_trajectories=traj)
    return sim_agents_submission_pb2.ScenarioRollouts(
        joint_scenes=[joint] * g, scenario_id=s.scenario_id
    )


def compare_one(path: Path, g: int, config) -> Dict[str, Tuple[float, float]]:
    """Returns {metric: (official_val, pytorch_val)} for one scenario."""
    ct = submission_specs.ChallengeType.SIM_AGENTS
    s = scenario_pb2.Scenario()
    for data in tf.data.TFRecordDataset([str(path)], compression_type=""):
        s.ParseFromString(bytes(data.numpy()))
        break

    roll = _gt_rollouts(s, g)

    official = wm.compute_scenario_metrics_for_bundle(config, s, roll, challenge_type=ct)
    log_mf, sim_mf = metric_features.compute_scenario_rollouts_features(s, roll, ct)
    log_d = _mf_to_dict(log_mf)
    sim_d = _mf_to_dict(sim_mf)
    del log_d["object_id"], sim_d["object_id"]

    pt = compute_wosac_metametric_from_features_torch(config, log_d, sim_d)

    result = {}
    for a in ATTRS:
        if a == "metametric":
            tf_v = float(official.metametric)
            pt_v = float(pt.metametric)
        else:
            tf_v = float(getattr(official, a))
            pt_v = float(getattr(pt, a))
        result[a] = (tf_v, pt_v)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="Number of scenarios")
    ap.add_argument("--g", type=int, default=32, help="Rollouts per scenario (G)")
    ap.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    args = ap.parse_args()

    config = WOSACMetrics.load_metrics_config()
    paths = sorted(Path(args.data_dir).glob("*.tfrecords"))[: args.n]
    if not paths:
        print("ERROR: No TFRecords found at", args.data_dir)
        sys.exit(1)

    print(f"Checking {len(paths)} scenarios, G={args.g} rollouts")
    print(f"Data: {args.data_dir}")
    print()

    # Accumulators
    all_tf: Dict[str, List[float]] = {a: [] for a in ATTRS}
    all_pt: Dict[str, List[float]] = {a: [] for a in ATTRS}
    all_diff: Dict[str, List[float]] = {a: [] for a in ATTRS}

    t_start = time.time()
    for i, p in enumerate(paths):
        try:
            res = compare_one(p, args.g, config)
        except Exception as e:
            print(f"  [{i+1:3d}/{len(paths)}] {p.name}  ERROR: {e}")
            continue
        for a in ATTRS:
            tf_v, pt_v = res[a]
            all_tf[a].append(tf_v)
            all_pt[a].append(pt_v)
            all_diff[a].append(abs(tf_v - pt_v))
        elapsed = time.time() - t_start
        print(f"  [{i+1:3d}/{len(paths)}] {p.name}  meta: official={res['metametric'][0]:.4f} pt={res['metametric'][1]:.4f}  elapsed={elapsed:.1f}s", flush=True)

    print()
    print("=" * 85)
    print(f"{'metric':<45}  {'mean_official':>13}  {'mean_pytorch':>12}  {'mean|d|':>9}  {'max|d|':>9}")
    print("-" * 85)
    for a in ATTRS:
        if not all_diff[a]:
            continue
        mo = np.mean(all_tf[a])
        mp = np.mean(all_pt[a])
        md = np.mean(all_diff[a])
        xd = np.max(all_diff[a])
        flag = "  <-- WARN" if xd > 1e-4 else ""
        print(f"{a:<45}  {mo:13.6f}  {mp:12.6f}  {md:9.2e}  {xd:9.2e}{flag}")
    print("=" * 85)
    print(f"Total time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
