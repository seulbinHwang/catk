#!/usr/bin/env python3
"""Parity test: TF vs Torch `map_metric_features.compute_distance_to_road_edge`."""

from __future__ import annotations

import argparse
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
from waymo_open_dataset.utils.sim_agents import converters, submission_specs
from waymo_open_dataset.wdl_limited.sim_agents_metrics import map_metric_features as tf_map

from src.smart.metrics.wosac_metric_features_torch import map_metric_features_torch as pt_map


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Parity: TF vs torch map_metric_features")
    ap.add_argument("--tfrecord", type=Path, default=None)
    ap.add_argument("--tol", type=float, default=5e-5)
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

    sim_traj = converters.joint_scene_to_trajectories(j, s, use_log_validity=False)
    eval_ids = tf.convert_to_tensor(
        submission_specs.get_evaluation_sim_agent_ids(s, submission_specs.ChallengeType.SIM_AGENTS)
    )
    eval_mask = tf.reduce_any(eval_ids[:, None] == sim_traj.object_id[None, :], axis=0)

    road_edges = []
    for mf in s.map_features:
        if mf.HasField("road_edge"):
            road_edges.append(list(mf.road_edge.polyline))

    tf_out = tf_map.compute_distance_to_road_edge(
        center_x=sim_traj.x,
        center_y=sim_traj.y,
        center_z=sim_traj.z,
        length=sim_traj.length,
        width=sim_traj.width,
        height=sim_traj.height,
        heading=sim_traj.heading,
        valid=sim_traj.valid,
        evaluated_object_mask=eval_mask,
        road_edge_polylines=road_edges,
    )

    pt_out = pt_map.compute_distance_to_road_edge(
        center_x=torch.from_numpy(sim_traj.x.numpy().astype(np.float32)),
        center_y=torch.from_numpy(sim_traj.y.numpy().astype(np.float32)),
        center_z=torch.from_numpy(sim_traj.z.numpy().astype(np.float32)),
        length=torch.from_numpy(sim_traj.length.numpy().astype(np.float32)),
        width=torch.from_numpy(sim_traj.width.numpy().astype(np.float32)),
        height=torch.from_numpy(sim_traj.height.numpy().astype(np.float32)),
        heading=torch.from_numpy(sim_traj.heading.numpy().astype(np.float32)),
        valid=torch.from_numpy(sim_traj.valid.numpy().astype(np.bool_)),
        evaluated_object_mask=torch.from_numpy(eval_mask.numpy().astype(np.bool_)),
        road_edge_polylines=road_edges,
        z_stretch=3.0,
    )

    a = tf_out.numpy()
    b = pt_out.numpy()
    diff = np.abs(a - b)
    diff = diff[np.isfinite(diff)]
    mx = float(diff.max()) if diff.size else 0.0
    print("scenario:", path.name)
    print(f"max|Δ| distance_to_road_edge: {mx:.3e}  (tol={args.tol:.3e})")
    if mx > args.tol:
        sys.exit(1)


if __name__ == "__main__":
    main()

