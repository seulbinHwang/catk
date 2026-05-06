#!/usr/bin/env python3
# Not a contribution — compare TF official bundle vs pure PyTorch port.
"""Verify ``wosac_metametric_pytorch`` matches ``metrics.compute_scenario_metrics_for_bundle``.

Checks **metametric** and **all 10 likelihood scalars** that enter the metametric sum.
Option repeats the same ``JointScene`` ``G`` times to stress rollout pooling (histogram).

**장담하는 방법:** 통계적으로는 불가능하고, 아래만 **증거**로 삼을 수 있다.

- 동일한 ``MetricFeatures``를 TF / PyTorch가 받았다는 전제에서, 항목별 스칼라의
  최대 절대 오차가 허용치 이하인 시나리오 전수(또는 큰 표본) 검사
- 구성 변형: ``G=1`` vs ``G>1`` (동일 궤적 복제)로 reshape 경로 검증

이 스크립트는 TF ``metrics`` / ``metric_features``를 **참조값**으로만 쓰고,
PyTorch 구현은 ``wosac_metametric_pytorch``만 호출한다.

Examples::

    python scripts/verify_wosac_metametric_pytorch_parity.py --n 50
    python scripts/verify_wosac_metametric_pytorch_parity.py --n 100 --g-rollouts 4 --tol 1e-5

환경 변수 ``WOSAC_PARITY_TFRECORD_DIR`` 로 TFRecord 디렉터리를 바꿀 수 있다.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
tf.config.set_visible_devices([], "GPU")

from waymo_open_dataset.protos import scenario_pb2, sim_agents_submission_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics as wm

from src.smart.metrics.hard_sim_agents_metrics import HardSimAgentsMetrics
from src.smart.metrics.wosac_metametric_pytorch import compute_wosac_metametric_from_features_torch

_LIKELIHOOD_ATTRS = [
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
]


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
    traj, t0 = [], cfg.current_time_index + 1
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


def _mf_to_dict(mf: metric_features.MetricFeatures) -> dict:
    d = {}
    for field in dataclasses.fields(metric_features.MetricFeatures):
        name = field.name
        arr = getattr(mf, name).numpy()
        if arr.dtype == np.bool_:
            d[name] = torch.from_numpy(arr)
        else:
            d[name] = torch.from_numpy(arr.astype(np.float32))
    return d


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


def _compare_one(
    *,
    config,
    path: Path,
    g_rollouts: int,
    tol: float,
) -> Tuple[Dict[str, float], bool]:
    """Returns per-metric max abs-error for this scenario, and whether all <= tol."""
    ct = submission_specs.ChallengeType.SIM_AGENTS
    s = _load_scenario(str(path))
    j = _gt_joint_scene(s)
    scenes = [j] * g_rollouts
    roll = sim_agents_submission_pb2.ScenarioRollouts(
        joint_scenes=scenes,
        scenario_id=s.scenario_id,
    )
    official = wm.compute_scenario_metrics_for_bundle(config, s, roll, challenge_type=ct)
    log_mf, sim_mf = metric_features.compute_scenario_rollouts_features(s, roll, ct)
    log_d, sim_d = _mf_to_dict(log_mf), _mf_to_dict(sim_mf)
    del log_d["object_id"], sim_d["object_id"]
    pt = compute_wosac_metametric_from_features_torch(config, log_d, sim_d)

    deltas: Dict[str, float] = {}
    all_ok = True
    for attr in _LIKELIHOOD_ATTRS:
        tf_v = float(getattr(official, attr))
        pt_v = float(getattr(pt, attr))
        d = abs(tf_v - pt_v)
        deltas[attr] = d
        if d > tol:
            all_ok = False
    dm = abs(float(official.metametric) - pt.metametric)
    deltas["metametric"] = dm
    if dm > tol:
        all_ok = False
    return deltas, all_ok


def _aggregate_max(
    running: Dict[str, float], deltas: Dict[str, float]
) -> None:
    for k, v in deltas.items():
        running[k] = max(running.get(k, 0.0), v)


def main() -> None:
    ap = argparse.ArgumentParser(description="WOSAC PyTorch metametric parity vs TF reference")
    ap.add_argument("--n", type=int, default=50, help="Max number of TFRecord scenarios")
    ap.add_argument(
        "--g-rollouts",
        type=int,
        default=1,
        help="Repeat identical JointScene this many times (tests G>1 pooling)",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Fail if any |TF-PT| exceeds this (default 1e-5; use 5e-7 for tight float32)",
    )
    ap.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit with code 1 if any delta > tol",
    )
    args = ap.parse_args()

    data_dir = _default_data_dir()
    paths = sorted(data_dir.glob("*.tfrecords"))[: args.n]
    if not paths:
        print("No TFRecords under", data_dir, file=sys.stderr)
        print("Set WOSAC_PARITY_TFRECORD_DIR", file=sys.stderr)
        sys.exit(2)

    config = HardSimAgentsMetrics._load_config()
    worst: Dict[str, float] = {}
    n_fail = 0
    first_fail: Tuple[str, str, float] | None = None  # path, metric, delta

    for p in paths:
        deltas, ok = _compare_one(
            config=config, path=p, g_rollouts=args.g_rollouts, tol=args.tol
        )
        _aggregate_max(worst, deltas)
        if not ok:
            n_fail += 1
            if first_fail is None:
                bad = max(deltas.items(), key=lambda kv: kv[1])
                first_fail = (p.name, bad[0], bad[1])

    print(
        f"Checked {len(paths)} scenarios (G={args.g_rollouts}), data_dir={data_dir}"
    )
    print(f"Tolerance tol={args.tol}")
    for key in list(_LIKELIHOOD_ATTRS) + ["metametric"]:
        mx = worst.get(key, 0.0)
        flag = "FAIL" if mx > args.tol else "ok"
        print(f"  max |d| {key:45s} {mx:.3e}  {flag}")

    print(f"Scenarios with any delta > tol: {n_fail} / {len(paths)}")
    if first_fail:
        fn, mk, dv = first_fail
        print(f"First failure example: {fn}  metric={mk}  |d|={dv:.3e}")

    if args.fail_on_mismatch and n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()