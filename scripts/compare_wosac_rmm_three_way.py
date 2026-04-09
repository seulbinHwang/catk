#!/usr/bin/env python3
"""
① ``metric_features.compute_scenario_rollouts_features`` 로 **실제 맵·궤적 기반 특징**을 뽑은 뒤,
동일 입력으로 다음 세 스칼라를 비교한다.

- **soft**: ``compute_wosac_metametric_soft`` (미분 가능 근사)
- **port**: ``compute_wosac_metametric_from_features_torch`` (이산 포트, TF 수식 정합 목표)
- **official**: ``metrics.compute_scenario_metrics_for_features_bundle`` (Waymo TensorFlow)

데모 ``demo_soft_metametric_random_opt.py`` 와 달리 **합성 특징이 아니라** TFRecord 시나리오 +
GT 궤적 기반 rollout(``verify_wosac_metametric_pytorch_parity`` 와 동일한 roll 구성)을 쓴다.

Examples::

    python scripts/compare_wosac_rmm_three_way.py --tfrecord /path/to/one.tfrecords
    python scripts/compare_wosac_rmm_three_way.py --n 5
    WOSAC_PARITY_TFRECORD_DIR=/data/val python scripts/compare_wosac_rmm_three_way.py --n 10
    python scripts/compare_wosac_rmm_three_way.py --n 3 --no-timing   # 시간 출력 없이 값만
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
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics as wm

from src.smart.metrics.wosac_metametric_pytorch import compute_wosac_metametric_from_features_torch
from src.smart.metrics.wosac_metametric_pytorch_differentiable import compute_wosac_metametric_soft
from src.smart.metrics.wosac_metrics import WOSACMetrics


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


def _mf_to_torch_dict(mf: metric_features.MetricFeatures) -> dict:
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="① rollout features → soft / PT port / TF official metametric 비교"
    )
    ap.add_argument("--tfrecord", type=Path, default=None, help="단일 TFRecord (없으면 디렉터리에서 n개)")
    ap.add_argument("--n", type=int, default=1, help="tfrecord 미지정 시 사용할 최대 시나리오 수")
    ap.add_argument("--g-rollouts", type=int, default=1, help="동일 JointScene 반복 수")
    ap.add_argument(
        "--no-timing",
        action="store_true",
        help="특징 추출·각 RMM 계산 시간 출력 끔",
    )
    args = ap.parse_args()

    data_dir = _default_data_dir()
    if args.tfrecord is not None:
        paths = [args.tfrecord]
    else:
        paths = sorted(data_dir.glob("*.tfrecords"))[: args.n]
    if not paths:
        print("TFRecord 없음:", data_dir, file=sys.stderr)
        print("WOSAC_PARITY_TFRECORD_DIR 또는 --tfrecord 지정", file=sys.stderr)
        sys.exit(2)

    config = WOSACMetrics.load_metrics_config()
    ct = submission_specs.ChallengeType.SIM_AGENTS

    show_timing = not args.no_timing
    print(f"data: {paths[0] if len(paths)==1 else data_dir}  (n={len(paths)} scenario(s))  G={args.g_rollouts}")
    print(f"{'scenario':<40}  {'soft':>10}  {'port(pt)':>10}  {'official(tf)':>12}  |s-o|  |p-o|")
    print("-" * 100)

    acc_feat: list[float] = []
    acc_off: list[float] = []
    acc_conv: list[float] = []
    acc_port: list[float] = []
    acc_soft: list[float] = []

    for p in paths:
        s = _load_scenario(str(p))
        j = _gt_joint_scene(s)
        scenes = [j] * args.g_rollouts
        roll = sim_agents_submission_pb2.ScenarioRollouts(
            joint_scenes=scenes,
            scenario_id=s.scenario_id,
        )

        t0 = time.perf_counter()
        log_mf, sim_mf = metric_features.compute_scenario_rollouts_features(s, roll, ct)
        t_feat = time.perf_counter() - t0

        t0 = time.perf_counter()
        official = wm.compute_scenario_metrics_for_features_bundle(
            config, s.scenario_id, log_mf, sim_mf
        )
        t_off = time.perf_counter() - t0
        o = float(official.metametric)

        t0 = time.perf_counter()
        log_d, sim_d = _mf_to_torch_dict(log_mf), _mf_to_torch_dict(sim_mf)
        del log_d["object_id"], sim_d["object_id"]
        t_conv = time.perf_counter() - t0

        t0 = time.perf_counter()
        port = compute_wosac_metametric_from_features_torch(config, log_d, sim_d).metametric
        t_port = time.perf_counter() - t0

        t0 = time.perf_counter()
        soft = float(compute_wosac_metametric_soft(config, log_d, sim_d).metametric.detach())
        t_soft = time.perf_counter() - t0

        if show_timing:
            acc_feat.append(t_feat)
            acc_off.append(t_off)
            acc_conv.append(t_conv)
            acc_port.append(t_port)
            acc_soft.append(t_soft)
            print(
                f"{p.name[:38]:<40}  {soft:10.6f}  {port:10.6f}  {o:12.6f}  "
                f"{abs(soft-o):.2e}  {abs(port-o):.2e}"
            )
            print(
                f"  [sec] rollouts_features={t_feat:.4f}  official(tf)={t_off:.4f}  "
                f"np→torch={t_conv:.4f}  port(pt)={t_port:.4f}  soft={t_soft:.4f}"
            )
        else:
            print(
                f"{p.name[:38]:<40}  {soft:10.6f}  {port:10.6f}  {o:12.6f}  "
                f"{abs(soft-o):.2e}  {abs(port-o):.2e}"
            )

    print("-" * 100)
    print("|s-o| = |soft - official|  |p-o| = |port - official|  (port 는 보통 official ~1e-4)")
    if show_timing and acc_feat:
        n = len(acc_feat)
        if n > 1:
            def _mean(xs: list[float]) -> float:
                return sum(xs) / len(xs)

            print(
                f"\n[timing mean over {n} scenarios]  "
                f"rollouts_features={_mean(acc_feat):.4f}s  "
                f"official(tf)={_mean(acc_off):.4f}s  "
                f"np→torch={_mean(acc_conv):.4f}s  "
                f"port(pt)={_mean(acc_port):.4f}s  "
                f"soft={_mean(acc_soft):.4f}s"
            )
        print(
            "\n※ rollouts_features = ① 맵·궤적에서 특징 추출. "
            "나머지는 이미 뽑은 텐서로 metametric 만 계산."
        )


if __name__ == "__main__":
    main()
