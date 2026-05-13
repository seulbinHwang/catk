from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import torch
import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as official_metrics
from google.protobuf.descriptor import FieldDescriptor
from waymo_open_dataset.protos import (
    scenario_pb2,
    sim_agents_metrics_pb2,
    sim_agents_submission_pb2,
)
from waymo_open_dataset.utils.sim_agents import submission_specs

from src.smart.metrics.sim_agents_metrics import (
    _compute_scenario_metrics_from_fast_bundle,
    _load_waymo_sim_agents_2025_config,
    _scenario_rollout_proto_to_fast_bundle,
)

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
    names = []
    for field in sim_agents_metrics_pb2.SimAgentMetrics.DESCRIPTOR.fields:
        if field.name == "scenario_id":
            continue
        if field.label == FieldDescriptor.LABEL_REPEATED:
            continue
        if field.type in _NUMERIC_FIELD_TYPES:
            names.append(field.name)
    return names


def _build_logged_rollout(
    scenario: scenario_pb2.Scenario,
    *,
    num_rollouts: int,
    perturb: bool,
) -> sim_agents_submission_pb2.ScenarioRollouts:
    challenge_type = submission_specs.ChallengeType.SIM_AGENTS
    sim_agent_ids = list(submission_specs.get_sim_agent_ids(scenario, challenge_type))
    track_by_id = {int(track.id): track for track in scenario.tracks}
    first_track = track_by_id[int(sim_agent_ids[0])]
    start_index = int(scenario.current_time_index) + 1
    num_steps = len(first_track.states) - start_index

    joint_scenes = []
    for rollout_idx in range(num_rollouts):
        simulated_trajectories = []
        centered_rollout = rollout_idx - (num_rollouts - 1) / 2.0
        for agent_idx, object_id in enumerate(sim_agent_ids):
            track = track_by_id[int(object_id)]
            center_x = []
            center_y = []
            center_z = []
            heading = []
            for step_idx, state in enumerate(track.states[start_index:]):
                phase = 0.17 * (agent_idx + 1) + 0.11 * (step_idx + 1)
                offset = 0.002 * centered_rollout if perturb else 0.0
                center_x.append(float(state.center_x) + offset * math.sin(phase))
                center_y.append(float(state.center_y) + offset * math.cos(phase))
                center_z.append(float(state.center_z))
                heading.append(float(state.heading) + 0.0002 * centered_rollout if perturb else float(state.heading))
            if len(center_x) != num_steps:
                raise RuntimeError("Inconsistent future step count while building rollouts.")
            simulated_trajectories.append(
                sim_agents_submission_pb2.SimulatedTrajectory(
                    center_x=center_x,
                    center_y=center_y,
                    center_z=center_z,
                    heading=heading,
                    object_id=int(object_id),
                )
            )
        joint_scenes.append(
            sim_agents_submission_pb2.JointScene(
                simulated_trajectories=simulated_trajectories
            )
        )

    return sim_agents_submission_pb2.ScenarioRollouts(
        scenario_id=scenario.scenario_id,
        joint_scenes=joint_scenes,
    )


def _compute_official_metrics(
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    scenario: scenario_pb2.Scenario,
    scenario_rollout: sim_agents_submission_pb2.ScenarioRollouts,
) -> sim_agents_metrics_pb2.SimAgentMetrics:
    compute_fn = official_metrics.compute_scenario_metrics_for_bundle
    kwargs = {}
    if "challenge_type" in inspect.signature(compute_fn).parameters:
        kwargs["challenge_type"] = submission_specs.ChallengeType.SIM_AGENTS
    return compute_fn(config, scenario, scenario_rollout, **kwargs)


def _compare_scenario(
    scenario_path: Path,
    *,
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    device: torch.device,
    num_rollouts: int,
    perturb: bool,
) -> dict[str, object]:
    scenario = _read_single_record_tfrecord(scenario_path)
    scenario_rollout = _build_logged_rollout(
        scenario,
        num_rollouts=num_rollouts,
        perturb=perturb,
    )
    official = _compute_official_metrics(config, scenario, scenario_rollout)
    fast = _compute_scenario_metrics_from_fast_bundle(
        config=config,
        scenario_file=scenario_path.as_posix(),
        scenario_rollouts=_scenario_rollout_proto_to_fast_bundle(scenario_rollout, device=device),
        ego_only=False,
        device=device,
    )

    diffs = {
        field_name: abs(float(getattr(official, field_name)) - float(getattr(fast, field_name)))
        for field_name in _metric_field_names()
    }
    max_field = max(diffs, key=diffs.get)
    return {
        "scenario_id": scenario.scenario_id,
        "path": scenario_path.as_posix(),
        "max_field": max_field,
        "max_abs_error": diffs[max_field],
        "field_errors": diffs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare vendored TrajTok Fast WOSAC metrics against Waymo's official scorer."
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=Path("womd_v1_3/cache/SMART/validation_tfrecords_splitted"),
        help="Directory containing one-scenario .tfrecords files.",
    )
    parser.add_argument("--num-scenarios", type=int, default=3)
    parser.add_argument("--num-rollouts", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=1.0e-6)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--no-perturb", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    scenario_paths = sorted(args.scenario_dir.glob("*.tfrecords"))[: args.num_scenarios]
    if not scenario_paths:
        raise FileNotFoundError(f"No .tfrecords files found under {args.scenario_dir}")

    device = torch.device(args.device)
    config = _load_waymo_sim_agents_2025_config()
    results = [
        _compare_scenario(
            scenario_path,
            config=config,
            device=device,
            num_rollouts=int(args.num_rollouts),
            perturb=not args.no_perturb,
        )
        for scenario_path in scenario_paths
    ]
    max_result = max(results, key=lambda item: float(item["max_abs_error"]))
    summary = {
        "num_scenarios": len(results),
        "num_rollouts": int(args.num_rollouts),
        "device": str(device),
        "threshold": float(args.threshold),
        "max_abs_error": float(max_result["max_abs_error"]),
        "max_field": str(max_result["max_field"]),
        "max_scenario_id": str(max_result["scenario_id"]),
        "passed": float(max_result["max_abs_error"]) <= float(args.threshold),
        "scenarios": results,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
