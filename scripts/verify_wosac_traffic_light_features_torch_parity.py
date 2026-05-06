#!/usr/bin/env python3
"""Parity test: TF vs Torch `traffic_light_features.compute_red_light_violation`.

Uses a real TFRecord scenario, builds a GT rollout, extracts trajectories via TF
converters (only for setting up inputs), then compares per-step violation masks.
"""

from __future__ import annotations

import argparse
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
from waymo_open_dataset.utils.sim_agents import converters
from waymo_open_dataset.wdl_limited.sim_agents_metrics import traffic_light_features as tf_tl

from src.smart.metrics.wosac_metric_features_torch import traffic_light_features_torch as pt_tl


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
    ap = argparse.ArgumentParser(description="Parity: TF vs torch traffic_light_features")
    ap.add_argument("--tfrecord", type=Path, default=None)
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

    # Build simulated trajectories (TF) with history prepended.
    sim_traj = converters.joint_scene_to_trajectories(j, s, use_log_validity=False)
    eval_ids = tf.convert_to_tensor(
        submission_specs.get_evaluation_sim_agent_ids(s, submission_specs.ChallengeType.SIM_AGENTS)
    )
    eval_mask = tf.reduce_any(eval_ids[:, None] == sim_traj.object_id[None, :], axis=0)

    # Extract lane polylines and traffic signals from scenario map/dynamics.
    lane_polylines = []
    lane_ids = []
    for mf in s.map_features:
        if mf.HasField("lane"):
            lane_polylines.append(list(mf.lane.polyline))
            lane_ids.append(int(mf.id))
    traffic_signals = [list(ts.lane_states) for ts in s.dynamic_map_states]

    tf_out = tf_tl.compute_red_light_violation(
        center_x=sim_traj.x,
        center_y=sim_traj.y,
        valid=sim_traj.valid,
        evaluated_object_mask=eval_mask,
        lane_polylines=lane_polylines,
        lane_ids=lane_ids,
        traffic_signals=traffic_signals,
    )

    pt_out = pt_tl.compute_red_light_violation(
        center_x=torch.from_numpy(sim_traj.x.numpy().astype(np.float32)),
        center_y=torch.from_numpy(sim_traj.y.numpy().astype(np.float32)),
        valid=torch.from_numpy(sim_traj.valid.numpy().astype(np.bool_)),
        evaluated_object_mask=torch.from_numpy(eval_mask.numpy().astype(np.bool_)),
        lane_polylines=lane_polylines,
        lane_ids=lane_ids,
        traffic_signals=traffic_signals,
    )

    eq = np.array_equal(tf_out.numpy(), pt_out.numpy())
    print("scenario:", path.name)
    print("exact match:", "ok" if eq else "FAIL")
    if not eq:
        # report mismatch count
        a = tf_out.numpy()
        b = pt_out.numpy()
        mismatch = np.count_nonzero(a != b)
        print("mismatch count:", mismatch, "/", a.size)
        sys.exit(1)


if __name__ == "__main__":
    import os

    main()

