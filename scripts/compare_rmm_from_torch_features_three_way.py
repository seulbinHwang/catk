#!/usr/bin/env python3
"""Compute RMM three-way using *torch-extracted* MetricFeatures.

1) Extract features with `wosac_metric_features_torch.compute_scenario_rollouts_features` (① torch).
2) Compute:
   - soft metametric (`compute_wosac_metametric_soft`)
   - port(pt) metametric (`compute_wosac_metametric_from_features_torch`)
   - official(tf) metametric by converting torch features back to TF MetricFeatures
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
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
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics as wm

from src.smart.metrics.wosac_metrics import WOSACMetrics
from src.smart.metrics.wosac_metametric_pytorch import compute_wosac_metametric_from_features_torch
from src.smart.metrics.wosac_metametric_pytorch_differentiable import compute_wosac_metametric_soft
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


def _to_tf_metric_features(pt) -> tf_mf.MetricFeatures:
    kwargs = {}
    for field in dataclasses.fields(tf_mf.MetricFeatures):
        name = field.name
        t = getattr(pt, name)
        arr = t.detach().cpu().numpy()
        if arr.dtype == np.bool_:
            kwargs[name] = tf.constant(arr)
        elif arr.dtype.kind in ("i", "u"):
            kwargs[name] = tf.constant(arr.astype(np.int32))
        else:
            kwargs[name] = tf.constant(arr.astype(np.float32))
    return tf_mf.MetricFeatures(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description="RMM three-way (torch-extracted features)")
    ap.add_argument("--tfrecord", type=Path, default=None)
    ap.add_argument("--g-rollouts", type=int, default=1)
    args = ap.parse_args()

    if args.tfrecord is None:
        data_dir = _default_data_dir()
        paths = sorted(data_dir.glob("*.tfrecords"))
        if not paths:
            print("No TFRecords under", data_dir, file=sys.stderr)
            sys.exit(2)
        path = paths[0]
    else:
        path = args.tfrecord

    s = _load_scenario(str(path))
    j = _gt_joint_scene(s)
    roll = sim_agents_submission_pb2.ScenarioRollouts(
        joint_scenes=[j] * args.g_rollouts, scenario_id=s.scenario_id
    )
    ct = submission_specs.ChallengeType.SIM_AGENTS
    config = WOSACMetrics.load_metrics_config()

    pt_log, pt_sim = pt_rollouts_features(s, roll, challenge_type=ct)
    log_d = pt_log.as_dict()
    sim_d = pt_sim.as_dict()
    del log_d["object_id"], sim_d["object_id"]

    soft = float(compute_wosac_metametric_soft(config, log_d, sim_d).metametric.detach())
    port = compute_wosac_metametric_from_features_torch(config, log_d, sim_d).metametric

    tf_log = _to_tf_metric_features(pt_log)
    tf_sim = _to_tf_metric_features(pt_sim)
    official = wm.compute_scenario_metrics_for_features_bundle(config, s.scenario_id, tf_log, tf_sim)
    off = float(official.metametric)

    print("scenario:", path.name, "G=", args.g_rollouts)
    print(f"soft:         {soft:.6f}")
    print(f"port(pt):     {port:.6f}")
    print(f"official(tf): {off:.6f}")
    print(f"|port-official|={abs(port-off):.3e}  |soft-official|={abs(soft-off):.3e}")


if __name__ == "__main__":
    main()

