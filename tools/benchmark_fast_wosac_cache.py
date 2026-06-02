from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from google.protobuf.descriptor import FieldDescriptor
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    pass

import torch


_NUMERIC_FIELD_TYPES = {
    FieldDescriptor.TYPE_DOUBLE,
    FieldDescriptor.TYPE_FLOAT,
    FieldDescriptor.TYPE_INT32,
    FieldDescriptor.TYPE_INT64,
    FieldDescriptor.TYPE_UINT32,
    FieldDescriptor.TYPE_UINT64,
    FieldDescriptor.TYPE_SINT32,
    FieldDescriptor.TYPE_SINT64,
    FieldDescriptor.TYPE_FIXED32,
    FieldDescriptor.TYPE_FIXED64,
    FieldDescriptor.TYPE_SFIXED32,
    FieldDescriptor.TYPE_SFIXED64,
    FieldDescriptor.TYPE_BOOL,
}


def _insert_source_root(source_root: Path) -> None:
    source_root = source_root.resolve()
    sys.path.insert(0, source_root.as_posix())


def _read_single_record_tfrecord(path: Path) -> scenario_pb2.Scenario:
    dataset = tf.data.TFRecordDataset(path.as_posix(), compression_type="")
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)
    for data in dataset.take(1):
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytes(data.numpy()))
        return scenario
    raise RuntimeError(f"TFRecord file is empty: {path}")


def _metric_field_names() -> list[str]:
    from waymo_open_dataset.protos import sim_agents_metrics_pb2

    names = []
    for field in sim_agents_metrics_pb2.SimAgentMetrics.DESCRIPTOR.fields:
        if field.name == "scenario_id":
            continue
        if field.label == FieldDescriptor.LABEL_REPEATED:
            continue
        if field.type in _NUMERIC_FIELD_TYPES:
            names.append(field.name)
    return names


def _build_prediction_payload(
    scenario_path: Path,
    *,
    num_rollouts: int,
    perturb: bool,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scenario = _read_single_record_tfrecord(scenario_path)
    challenge_type = submission_specs.ChallengeType.SIM_AGENTS
    sim_agent_ids = np.asarray(
        list(submission_specs.get_sim_agent_ids(scenario, challenge_type)),
        dtype=np.int32,
    )
    track_by_id = {int(track.id): track for track in scenario.tracks}
    first_track = track_by_id[int(sim_agent_ids[0])]
    start_index = int(scenario.current_time_index) + 1
    num_steps = len(first_track.states) - start_index
    num_agents = int(sim_agent_ids.shape[0])

    pred_traj = np.zeros((num_agents, num_rollouts, num_steps, 2), dtype=np.float32)
    pred_z = np.zeros((num_agents, num_rollouts, num_steps), dtype=np.float32)
    pred_head = np.zeros((num_agents, num_rollouts, num_steps), dtype=np.float32)
    for agent_idx, object_id in enumerate(sim_agent_ids):
        track = track_by_id[int(object_id)]
        for rollout_idx in range(num_rollouts):
            centered_rollout = rollout_idx - (num_rollouts - 1) / 2.0
            for step_idx, state in enumerate(track.states[start_index:]):
                phase = 0.17 * (agent_idx + 1) + 0.11 * (step_idx + 1)
                offset = 0.002 * centered_rollout if perturb else 0.0
                pred_traj[agent_idx, rollout_idx, step_idx, 0] = float(state.center_x) + offset * math.sin(phase)
                pred_traj[agent_idx, rollout_idx, step_idx, 1] = float(state.center_y) + offset * math.cos(phase)
                pred_z[agent_idx, rollout_idx, step_idx] = float(state.center_z)
                pred_head[agent_idx, rollout_idx, step_idx] = (
                    float(state.heading) + 0.0002 * centered_rollout
                    if perturb
                    else float(state.heading)
                )

    return scenario_path.as_posix(), sim_agent_ids, pred_traj, pred_z, pred_head


def _clear_target_caches() -> None:
    try:
        from src.smart.metrics.sim_agents_metrics import _clear_sim_agents_caches
    except ImportError:
        return
    _clear_sim_agents_caches()


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _metric_tensors_to_scalars(metric_dict: dict[str, Any]) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for key, value in metric_dict.items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                scalars[key] = float(value.detach().cpu().item())
        elif isinstance(value, (int, float)):
            scalars[key] = float(value)
    return scalars


def _max_abs_delta(reference: dict[str, float], candidate: dict[str, float]) -> float:
    shared_keys = sorted(set(reference) & set(candidate))
    if not shared_keys:
        return 0.0
    return max(abs(reference[key] - candidate[key]) for key in shared_keys)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    _insert_source_root(args.source_root)
    from src.smart.metrics.sim_agents_metrics import SimAgentsMetrics

    scenario_paths = sorted(args.scenario_dir.glob("*.tfrecords"))[: args.num_scenarios]
    if not scenario_paths:
        raise FileNotFoundError(f"No .tfrecords files found under {args.scenario_dir}")

    device = torch.device(args.device)
    payloads = [
        _build_prediction_payload(
            scenario_path,
            num_rollouts=args.num_rollouts,
            perturb=not args.no_perturb,
        )
        for scenario_path in scenario_paths
    ]

    metric = SimAgentsMetrics("bench", max_workers=args.max_workers)
    pass_seconds: list[float] = []
    pass_metrics: list[dict[str, float]] = []
    pass_cache_metrics: list[dict[str, float]] = []
    for pass_idx in range(args.passes):
        if args.clear_before_each_pass:
            _clear_target_caches()
        _sync_device(device)
        start = time.perf_counter()
        metric.update_from_prediction_payloads(payloads)
        metrics = _metric_tensors_to_scalars(metric.compute())
        cache_metrics = metric.get_cache_metrics(reset=False)
        _sync_device(device)
        pass_seconds.append(time.perf_counter() - start)
        pass_metrics.append(metrics)
        pass_cache_metrics.append(cache_metrics)
        metric.reset()

    first_metrics = pass_metrics[0]
    max_repeat_delta = max(
        (_max_abs_delta(first_metrics, metrics) for metrics in pass_metrics[1:]),
        default=0.0,
    )
    warm_seconds = pass_seconds[1:] if len(pass_seconds) > 1 else pass_seconds
    summary = {
        "source_root": args.source_root.resolve().as_posix(),
        "scenario_dir": args.scenario_dir.resolve().as_posix(),
        "num_scenarios": len(scenario_paths),
        "num_rollouts": int(args.num_rollouts),
        "passes": int(args.passes),
        "max_workers": int(args.max_workers),
        "device": str(device),
        "clear_before_each_pass": bool(args.clear_before_each_pass),
        "cache_env": {
            "CATK_FAST_WOSAC_GT_CACHE_MAX_SCENARIOS": os.environ.get(
                "CATK_FAST_WOSAC_GT_CACHE_MAX_SCENARIOS",
                "",
            ),
            "CATK_FAST_WOSAC_LOG_FEATURE_CACHE_MAX_SCENARIOS": os.environ.get(
                "CATK_FAST_WOSAC_LOG_FEATURE_CACHE_MAX_SCENARIOS",
                "",
            ),
        },
        "seconds_by_pass": pass_seconds,
        "first_pass_seconds": pass_seconds[0],
        "warm_mean_seconds": float(sum(warm_seconds) / len(warm_seconds)),
        "max_repeat_metric_delta": float(max_repeat_delta),
        "metric_keys": sorted(first_metrics),
        "metrics_first_pass": first_metrics,
        "cache_metrics_by_pass": pass_cache_metrics,
    }
    return summary


def main() -> None:
    default_cache_root = Path(
        os.environ.get(
            "CACHE_ROOT",
            "/home2/pnc2/repos_python/datasets/catk_cache",
        )
    )
    parser = argparse.ArgumentParser(
        description="Benchmark repeated Fast WOSAC metric scoring with and without cache reuse."
    )
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=default_cache_root / "validation_tfrecords_splitted",
    )
    parser.add_argument("--num-scenarios", type=int, default=4)
    parser.add_argument("--num-rollouts", type=int, default=32)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--clear-before-each-pass", action="store_true")
    parser.add_argument("--no-perturb", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    if args.num_scenarios < 1:
        raise ValueError("--num-scenarios must be >= 1")
    if args.num_rollouts < 1:
        raise ValueError("--num-rollouts must be >= 1")
    if args.passes < 1:
        raise ValueError("--passes must be >= 1")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")

    summary = run_benchmark(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
