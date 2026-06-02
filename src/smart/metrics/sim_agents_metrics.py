# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait as futures_wait
from collections import OrderedDict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import tensorflow as tf
import torch
from google.protobuf import text_format
from google.protobuf.descriptor import Descriptor, FieldDescriptor
from torch import Tensor, tensor
from torchmetrics import Metric
from waymo_open_dataset.protos import (
    scenario_pb2,
    sim_agents_metrics_pb2,
    sim_agents_submission_pb2,
)
from waymo_open_dataset.utils.sim_agents import submission_specs

from src.smart.metrics.wosac_fast_eval_tool.fast_sim_agents_metrics import (
    metric_features as fast_metric_features,
    metrics as fast_sim_agents_metrics,
)
from src.smart.metrics.wosac_fast_eval_tool.scenario_gt_converter import (
    extract_gt_scenario,
)

_SIM_AGENTS_2025_NAMESPACE = "sim_agents_2025"
_SIM_AGENTS_2025_VERSION = "2025"
_SIM_AGENTS_2025_CHALLENGE_TYPE = getattr(
    getattr(submission_specs, "ChallengeType", None),
    "SIM_AGENTS",
    None,
)
_FAST_WOSAC_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "wosac_fast_eval_tool"
    / "fast_sim_agents_metrics"
    / "challenge_2025_sim_agents_config.textproto"
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
_REQUIRED_2025_SCENARIO_FIELDS = {
    "traffic_light_violation_likelihood",
    "simulated_traffic_light_violation_rate",
}
_REQUIRED_2025_BUCKET_FIELDS = {
    "simulated_traffic_light_violation_rate",
}
_TF_RUNTIME_CONFIGURED = False
_TF_RUNTIME_LOCK = threading.Lock()
_GT_SCENARIO_CACHE: OrderedDict[tuple[str, bool], dict] = OrderedDict()
_GT_SCENARIO_CACHE_LOCK = threading.Lock()


def _clear_sim_agents_caches() -> None:
    with _GT_SCENARIO_CACHE_LOCK:
        _GT_SCENARIO_CACHE.clear()
    fast_metric_features.clear_log_feature_cache()


def _read_nonnegative_int_env(var_name: str, default: int) -> int:
    raw_value = os.environ.get(var_name, "").strip()
    if not raw_value:
        return default
    try:
        return max(0, int(raw_value))
    except ValueError as exc:
        raise RuntimeError(f"{var_name} must be an integer, got {raw_value!r}.") from exc


def _gt_scenario_cache_max_entries() -> int:
    return _read_nonnegative_int_env("CATK_FAST_WOSAC_GT_CACHE_MAX_SCENARIOS", 4096)


def _trim_gt_scenario_cache(max_entries: int) -> None:
    while len(_GT_SCENARIO_CACHE) > max_entries:
        _GT_SCENARIO_CACHE.popitem(last=False)


def _configure_tensorflow_runtime() -> None:
    """Keep TensorFlow as a lightweight TFRecord reader beside the torch metric."""
    global _TF_RUNTIME_CONFIGURED
    with _TF_RUNTIME_LOCK:
        if _TF_RUNTIME_CONFIGURED:
            return

        intra_op_threads = max(1, _read_nonnegative_int_env("CATK_TF_INTRA_OP_THREADS", 1))
        inter_op_threads = max(1, _read_nonnegative_int_env("CATK_TF_INTER_OP_THREADS", 1))

        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass

        try:
            tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
        except RuntimeError:
            pass

        try:
            tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
        except RuntimeError:
            pass

        _TF_RUNTIME_CONFIGURED = True


def _read_single_record_tfrecord(record_path: str) -> bytes:
    dataset = tf.data.TFRecordDataset(record_path, compression_type="")
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)
    for data in dataset.take(1):
        return bytes(data.numpy())
    raise RuntimeError(f"TFRecord file is empty: {record_path}")


def _get_waymo_version_string() -> str:
    try:
        return version("waymo-open-dataset-tf-2-12-0")
    except PackageNotFoundError:
        return "unknown"


def _get_scalar_field_names(
    message_descriptor: Descriptor,
    skip_names: Sequence[str] = (),
) -> Tuple[str, ...]:
    scalar_field_names = []
    for field in message_descriptor.fields:
        if field.name in skip_names:
            continue
        if field.label == FieldDescriptor.LABEL_REPEATED:
            continue
        if field.type not in _NUMERIC_FIELD_TYPES:
            continue
        scalar_field_names.append(field.name)
    return tuple(scalar_field_names)


def _validate_waymo_sim_agents_2025_runtime_support() -> None:
    scenario_field_names = set(
        _get_scalar_field_names(
            sim_agents_metrics_pb2.SimAgentMetrics.DESCRIPTOR,
            skip_names=("scenario_id",),
        )
    )
    bucket_field_names = set(
        _get_scalar_field_names(
            sim_agents_metrics_pb2.SimAgentsBucketedMetrics.DESCRIPTOR,
        )
    )
    missing_scenario_fields = sorted(
        _REQUIRED_2025_SCENARIO_FIELDS - scenario_field_names
    )
    missing_bucket_fields = sorted(_REQUIRED_2025_BUCKET_FIELDS - bucket_field_names)
    if missing_scenario_fields or missing_bucket_fields:
        raise RuntimeError(
            "설치된 waymo-open-dataset 패키지에 2025 Sim Agents 필드가 없습니다. "
            f"현재 버전={_get_waymo_version_string()}. "
            f"scenario missing={missing_scenario_fields}, "
            f"bucket missing={missing_bucket_fields}."
        )


def _load_waymo_sim_agents_2025_config(
) -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
    if _SIM_AGENTS_2025_CHALLENGE_TYPE is None:
        raise RuntimeError(
            "설치된 waymo-open-dataset 패키지가 2025 Sim Agents challenge type를 제공하지 않습니다. "
            f"현재 버전={_get_waymo_version_string()}. "
            "WOSAC 2024 평가는 허용되지 않으며, README 기준으로 "
            "waymo-open-dataset-tf-2-12-0==1.6.7 이상이 필요합니다."
        )
    if not _FAST_WOSAC_CONFIG_PATH.exists():
        raise RuntimeError(f"Fast WOSAC 2025 config 파일을 찾지 못했습니다: {_FAST_WOSAC_CONFIG_PATH}")

    config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
    with _FAST_WOSAC_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        text_format.Parse(handle.read(), config)

    traffic_light_weight = float(config.traffic_light_violation.metametric_weight)
    if traffic_light_weight <= 0.0:
        raise RuntimeError(
            "Fast WOSAC 설정을 읽었지만 traffic_light_violation 가중치가 0입니다. "
            "이 환경은 2025 Sim Agents 평가 환경이 아닙니다."
        )
    return config


def _prepare_ego_only_scenario(scenario: scenario_pb2.Scenario) -> None:
    for track_index, track in enumerate(scenario.tracks):
        if track_index == scenario.sdc_track_index:
            continue
        for state in track.states:
            state.valid = False

    del scenario.tracks_to_predict[:]
    required_prediction = scenario.tracks_to_predict.add()
    required_prediction.track_index = scenario.sdc_track_index


def _load_scenario_proto(scenario_file: str, ego_only: bool) -> scenario_pb2.Scenario:
    _configure_tensorflow_runtime()
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(_read_single_record_tfrecord(scenario_file))
    if ego_only:
        _prepare_ego_only_scenario(scenario)
    return scenario


def _clone_to_device(value, device: torch.device):
    if isinstance(value, Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _clone_to_device(child, device) for key, child in value.items()}
    if isinstance(value, list):
        return [_clone_to_device(child, device) for child in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_device(child, device) for child in value)
    return value


def _load_gt_scenario_for_device(
    scenario_file: str,
    ego_only: bool,
    device: torch.device,
) -> dict:
    cache_key = (scenario_file, ego_only)
    cache_max_entries = _gt_scenario_cache_max_entries()
    gt_scenario = None
    if cache_max_entries > 0:
        with _GT_SCENARIO_CACHE_LOCK:
            gt_scenario = _GT_SCENARIO_CACHE.get(cache_key)
            if gt_scenario is not None:
                _GT_SCENARIO_CACHE.move_to_end(cache_key)
    if gt_scenario is None:
        gt_scenario = extract_gt_scenario(
            _load_scenario_proto(scenario_file, ego_only),
            device="cpu",
        )
        gt_scenario["_cache_scenario_file"] = str(scenario_file)
        gt_scenario["_cache_ego_only"] = bool(ego_only)
        if cache_max_entries > 0:
            with _GT_SCENARIO_CACHE_LOCK:
                _GT_SCENARIO_CACHE[cache_key] = gt_scenario
                _trim_gt_scenario_cache(cache_max_entries)
    return _clone_to_device(gt_scenario, device)


def _resolve_metric_device(device=None, *values) -> torch.device:
    if device is not None:
        return torch.device(device)
    for value in values:
        if isinstance(value, Tensor):
            return value.device
    return torch.device("cpu")


def _as_tensor(value, *, device: torch.device, dtype: torch.dtype | None = None) -> Tensor:
    if isinstance(value, Tensor):
        out = value.detach()
    else:
        out = torch.as_tensor(value)
    if dtype is None:
        return out.to(device=device)
    return out.to(device=device, dtype=dtype)


def _prediction_arrays_to_fast_bundle(
    agent_id,
    pred_traj,
    pred_z,
    pred_head,
    device: torch.device,
) -> dict:
    agent_id_tensor = _as_tensor(agent_id, device=device, dtype=torch.int32)
    pred_traj_tensor = _as_tensor(pred_traj, device=device, dtype=torch.float32)
    pred_z_tensor = _as_tensor(pred_z, device=device, dtype=torch.float32)
    pred_head_tensor = _as_tensor(pred_head, device=device, dtype=torch.float32)

    if pred_traj_tensor.ndim != 4 or pred_traj_tensor.shape[-1] != 2:
        raise ValueError(
            "pred_traj must have shape [n_agent, n_rollout, n_step, 2], "
            f"got {tuple(pred_traj_tensor.shape)}."
        )
    if pred_z_tensor.shape != pred_traj_tensor.shape[:-1]:
        raise ValueError(
            "pred_z must have shape [n_agent, n_rollout, n_step], "
            f"got {tuple(pred_z_tensor.shape)} for pred_traj={tuple(pred_traj_tensor.shape)}."
        )
    if pred_head_tensor.shape != pred_traj_tensor.shape[:-1]:
        raise ValueError(
            "pred_head must have shape [n_agent, n_rollout, n_step], "
            f"got {tuple(pred_head_tensor.shape)} for pred_traj={tuple(pred_traj_tensor.shape)}."
        )

    simulated_states = torch.cat(
        [
            pred_traj_tensor,
            pred_z_tensor.unsqueeze(-1),
            pred_head_tensor.unsqueeze(-1),
        ],
        dim=-1,
    ).permute(1, 0, 2, 3).contiguous()
    return {
        "agent_id": agent_id_tensor,
        "simulated_states": simulated_states,
    }


def _scenario_rollout_proto_to_fast_bundle(
    scenario_rollout: sim_agents_submission_pb2.ScenarioRollouts,
    device: torch.device,
) -> dict:
    if len(scenario_rollout.joint_scenes) == 0:
        raise ValueError("ScenarioRollouts must contain at least one JointScene.")

    first_scene = scenario_rollout.joint_scenes[0]
    agent_ids = [int(traj.object_id) for traj in first_scene.simulated_trajectories]
    if len(agent_ids) == 0:
        raise ValueError("ScenarioRollouts JointScene must contain simulated trajectories.")

    n_rollout = len(scenario_rollout.joint_scenes)
    n_agent = len(agent_ids)
    n_step = len(first_scene.simulated_trajectories[0].center_x)
    simulated_states = torch.empty(
        (n_rollout, n_agent, n_step, 4),
        dtype=torch.float32,
        device=device,
    )
    for rollout_idx, joint_scene in enumerate(scenario_rollout.joint_scenes):
        scene_agent_ids = [int(traj.object_id) for traj in joint_scene.simulated_trajectories]
        if scene_agent_ids != agent_ids:
            raise ValueError("All JointScene objects must share the same agent order.")
        for agent_idx, trajectory in enumerate(joint_scene.simulated_trajectories):
            if len(trajectory.center_x) != n_step:
                raise ValueError("All simulated trajectories must share the same step count.")
            simulated_states[rollout_idx, agent_idx, :, 0] = torch.as_tensor(
                trajectory.center_x,
                dtype=torch.float32,
                device=device,
            )
            simulated_states[rollout_idx, agent_idx, :, 1] = torch.as_tensor(
                trajectory.center_y,
                dtype=torch.float32,
                device=device,
            )
            simulated_states[rollout_idx, agent_idx, :, 2] = torch.as_tensor(
                trajectory.center_z,
                dtype=torch.float32,
                device=device,
            )
            simulated_states[rollout_idx, agent_idx, :, 3] = torch.as_tensor(
                trajectory.heading,
                dtype=torch.float32,
                device=device,
            )

    return {
        "agent_id": torch.tensor(agent_ids, dtype=torch.int32, device=device),
        "simulated_states": simulated_states,
    }


def _fast_result_to_proto(
    scenario_id: str,
    fast_result,
) -> sim_agents_metrics_pb2.SimAgentMetrics:
    if isinstance(fast_result, sim_agents_metrics_pb2.SimAgentMetrics):
        return fast_result

    scenario_metrics = sim_agents_metrics_pb2.SimAgentMetrics(scenario_id=str(scenario_id))
    for field_name, value in dict(fast_result).items():
        if field_name == "scenario_id":
            scenario_metrics.scenario_id = str(value)
            continue
        if hasattr(scenario_metrics, field_name):
            setattr(scenario_metrics, field_name, float(value))
    return scenario_metrics


def _compute_scenario_metrics_from_fast_bundle(
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    scenario_file: str,
    scenario_rollouts: dict,
    ego_only: bool,
    device: torch.device,
) -> sim_agents_metrics_pb2.SimAgentMetrics:
    gt_scenario = _load_gt_scenario_for_device(
        scenario_file=scenario_file,
        ego_only=ego_only,
        device=device,
    )
    fast_result = fast_sim_agents_metrics.compute_scenario_metrics_for_bundle(
        config=config,
        gt_scenario=gt_scenario,
        scenario_rollouts=scenario_rollouts,
        version=_SIM_AGENTS_2025_VERSION,
    )
    return _fast_result_to_proto(str(gt_scenario["scenario_id"]), fast_result)


def _compute_scenario_metrics_from_arrays(
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    scenario_file: str,
    agent_id,
    pred_traj,
    pred_z,
    pred_head,
    ego_only: bool,
    device=None,
) -> sim_agents_metrics_pb2.SimAgentMetrics:
    metric_device = _resolve_metric_device(device, pred_traj, pred_z, pred_head)
    return _compute_scenario_metrics_from_fast_bundle(
        config=config,
        scenario_file=scenario_file,
        scenario_rollouts=_prediction_arrays_to_fast_bundle(
            agent_id=agent_id,
            pred_traj=pred_traj,
            pred_z=pred_z,
            pred_head=pred_head,
            device=metric_device,
        ),
        ego_only=ego_only,
        device=metric_device,
    )


def _compute_scenario_metrics(
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    scenario_file: str,
    scenario_rollout: sim_agents_submission_pb2.ScenarioRollouts,
    ego_only: bool,
    device=None,
) -> sim_agents_metrics_pb2.SimAgentMetrics:
    metric_device = _resolve_metric_device(device)
    return _compute_scenario_metrics_from_fast_bundle(
        config=config,
        scenario_file=scenario_file,
        scenario_rollouts=_scenario_rollout_proto_to_fast_bundle(
            scenario_rollout,
            device=metric_device,
        ),
        ego_only=ego_only,
        device=metric_device,
    )


class SimAgentsMetrics(Metric):
    """TrajTok Fast WOSAC 2025 evaluator wrapped as a torchmetrics Metric."""

    def __init__(self, prefix: str, ego_only: bool = False, max_workers: int = 8) -> None:
        super().__init__()
        self.prefix = prefix
        self.ego_only = ego_only
        self.metric_namespace = f"{self.prefix}/{_SIM_AGENTS_2025_NAMESPACE}"
        self.metric_mean_namespace = f"{self.prefix}/{_SIM_AGENTS_2025_NAMESPACE}_mean"

        _configure_tensorflow_runtime()
        _validate_waymo_sim_agents_2025_runtime_support()
        self.sim_agents_config = _load_waymo_sim_agents_2025_config()
        self.scenario_metric_field_names = _get_scalar_field_names(
            sim_agents_metrics_pb2.SimAgentMetrics.DESCRIPTOR,
            skip_names=("scenario_id",),
        )
        self.bucket_metric_field_names = _get_scalar_field_names(
            sim_agents_metrics_pb2.SimAgentsBucketedMetrics.DESCRIPTOR,
        )
        for field_name in self.scenario_metric_field_names:
            self.add_state(field_name, default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("scenario_counter", default=tensor(0.0), dist_reduce_fx="sum")
        self._max_workers = max(0, int(max_workers))
        self._executor: ThreadPoolExecutor | None = None
        self._pending_futures: list[Future] = []
        self._max_pending_futures = max(1, self._max_workers * 2)

    @staticmethod
    def _compute_scenario_metrics(
        config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
        scenario_file: str,
        scenario_rollout: sim_agents_submission_pb2.ScenarioRollouts,
        ego_only: bool,
    ) -> sim_agents_metrics_pb2.SimAgentMetrics:
        return _compute_scenario_metrics(
            config=config,
            scenario_file=scenario_file,
            scenario_rollout=scenario_rollout,
            ego_only=ego_only,
        )

    def _use_worker_pool(self) -> bool:
        return self._max_workers > 1

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="fast-wosac",
            )
        return self._executor

    def _submit_scenario_metrics_from_arrays(
        self,
        *,
        scenario_file: str,
        agent_id,
        pred_traj,
        pred_z,
        pred_head,
    ) -> None:
        future = self._ensure_executor().submit(
            _compute_scenario_metrics_from_arrays,
            config=self.sim_agents_config,
            scenario_file=scenario_file,
            agent_id=agent_id,
            pred_traj=pred_traj,
            pred_z=pred_z,
            pred_head=pred_head,
            ego_only=self.ego_only,
        )
        self._pending_futures.append(future)
        if len(self._pending_futures) >= self._max_pending_futures:
            self._drain_completed_futures(wait=True)

    def _submit_scenario_metrics_from_rollout(
        self,
        *,
        scenario_file: str,
        scenario_rollout: sim_agents_submission_pb2.ScenarioRollouts,
    ) -> None:
        future = self._ensure_executor().submit(
            self._compute_scenario_metrics,
            self.sim_agents_config,
            scenario_file,
            scenario_rollout,
            self.ego_only,
        )
        self._pending_futures.append(future)
        if len(self._pending_futures) >= self._max_pending_futures:
            self._drain_completed_futures(wait=True)

    def _drain_completed_futures(self, wait: bool, drain_all: bool = False) -> None:
        if not self._pending_futures:
            return

        if drain_all:
            futures_to_update = self._pending_futures
            self._pending_futures = []
        else:
            if wait and not self._pending_futures[0].done():
                futures_wait([self._pending_futures[0]])
            drain_count = 0
            for future in self._pending_futures:
                if not future.done():
                    break
                drain_count += 1
            if drain_count == 0:
                return
            futures_to_update = self._pending_futures[:drain_count]
            self._pending_futures = self._pending_futures[drain_count:]

        for future in futures_to_update:
            self._update_metric_states(future.result())

    def reset(self) -> None:
        self._drain_completed_futures(wait=True, drain_all=True)
        super().reset()

    def _update_metric_states(
        self,
        scenario_metrics: sim_agents_metrics_pb2.SimAgentMetrics,
    ) -> None:
        self.scenario_counter.add_(1.0)
        for field_name in self.scenario_metric_field_names:
            getattr(self, field_name).add_(float(getattr(scenario_metrics, field_name)))

    def _build_zero_output_dict(self) -> Dict[str, Tensor]:
        zero_value = self.scenario_counter * 0.0
        out_dict: Dict[str, Tensor] = {
            f"{self.metric_namespace}/scenario_counter": zero_value.clone(),
        }
        for field_name in self.bucket_metric_field_names:
            out_dict[f"{self.metric_namespace}/{field_name}"] = zero_value.clone()
        for field_name in self.scenario_metric_field_names:
            out_dict[f"{self.metric_mean_namespace}/{field_name}"] = zero_value.clone()
        return out_dict

    def get_state_tensor(self, device: torch.device) -> Tensor:
        self._drain_completed_futures(wait=True, drain_all=True)
        state_values = [self.scenario_counter.detach().to(device=device)]
        state_values.extend(
            getattr(self, field_name).detach().to(device=device)
            for field_name in self.scenario_metric_field_names
        )
        return torch.stack(state_values)

    def compute_from_state_tensor(self, state_tensor: Tensor) -> Dict[str, Tensor]:
        scenario_counter = state_tensor[0]
        if scenario_counter.item() == 0:
            zero_value = scenario_counter * 0.0
            out_dict: Dict[str, Tensor] = {
                f"{self.metric_namespace}/scenario_counter": zero_value.clone(),
            }
            for field_name in self.bucket_metric_field_names:
                out_dict[f"{self.metric_namespace}/{field_name}"] = zero_value.clone()
            for field_name in self.scenario_metric_field_names:
                out_dict[f"{self.metric_mean_namespace}/{field_name}"] = zero_value.clone()
            return out_dict

        mean_metric_tensors = {
            field_name: state_tensor[field_idx + 1] / scenario_counter
            for field_idx, field_name in enumerate(self.scenario_metric_field_names)
        }
        mean_metric_scalars = {
            field_name: float(metric_value.item())
            for field_name, metric_value in mean_metric_tensors.items()
        }
        mean_metrics = sim_agents_metrics_pb2.SimAgentMetrics(
            scenario_id="",
            **mean_metric_scalars,
        )
        bucket_metrics = fast_sim_agents_metrics.aggregate_metrics_to_buckets(
            self.sim_agents_config,
            mean_metrics,
        )

        out_dict: Dict[str, Tensor] = {
            f"{self.metric_namespace}/scenario_counter": scenario_counter.clone(),
        }
        for field_name in self.bucket_metric_field_names:
            out_dict[f"{self.metric_namespace}/{field_name}"] = scenario_counter.new_tensor(
                float(getattr(bucket_metrics, field_name))
            )
        for field_name, metric_value in mean_metric_tensors.items():
            out_dict[f"{self.metric_mean_namespace}/{field_name}"] = metric_value
        return out_dict

    def update(
        self,
        scenario_files: List[str],
        scenario_rollouts: List[sim_agents_submission_pb2.ScenarioRollouts],
    ) -> None:
        if len(scenario_rollouts) == 0:
            return

        self._computed = None
        self._update_count += 1

        for scenario_file, scenario_rollout in zip(scenario_files, scenario_rollouts):
            if self._use_worker_pool():
                self._submit_scenario_metrics_from_rollout(
                    scenario_file=scenario_file,
                    scenario_rollout=scenario_rollout,
                )
            else:
                scenario_metrics = self._compute_scenario_metrics(
                    self.sim_agents_config,
                    scenario_file,
                    scenario_rollout,
                    self.ego_only,
                )
                self._update_metric_states(scenario_metrics)

    @staticmethod
    def build_prediction_payloads(
        scenario_files: List[str],
        agent_id: Tensor,
        agent_batch: Tensor,
        pred_traj: Tensor,
        pred_z: Tensor,
        pred_head: Tensor,
    ) -> list[tuple[str, object, object, object, object]]:
        agent_batch_cpu = agent_batch.detach().to(device="cpu", dtype=torch.long)
        sizes = torch.bincount(agent_batch_cpu, minlength=len(scenario_files)).tolist()
        agent_id_cpu = agent_id.detach().cpu()
        pred_traj_cpu = pred_traj.detach().cpu()
        pred_z_cpu = pred_z.detach().cpu()
        pred_head_cpu = pred_head.detach().cpu()

        start = 0
        scenario_payloads = []
        for scenario_file, size in zip(scenario_files, sizes):
            end = start + int(size)
            scenario_payloads.append(
                (
                    scenario_file,
                    agent_id_cpu[start:end].numpy(),
                    pred_traj_cpu[start:end].numpy(),
                    pred_z_cpu[start:end].numpy(),
                    pred_head_cpu[start:end].numpy(),
                )
            )
            start = end
        return scenario_payloads

    def update_from_prediction_payloads(
        self,
        scenario_payloads: list[tuple[str, object, object, object, object]],
    ) -> None:
        if len(scenario_payloads) == 0:
            return

        self._computed = None
        self._update_count += 1

        for (
            scenario_file,
            scenario_agent_id,
            scenario_pred_traj,
            scenario_pred_z,
            scenario_pred_head,
        ) in scenario_payloads:
            if self._use_worker_pool():
                self._submit_scenario_metrics_from_arrays(
                    scenario_file=scenario_file,
                    agent_id=scenario_agent_id,
                    pred_traj=scenario_pred_traj,
                    pred_z=scenario_pred_z,
                    pred_head=scenario_pred_head,
                )
            else:
                scenario_metrics = _compute_scenario_metrics_from_arrays(
                    config=self.sim_agents_config,
                    scenario_file=scenario_file,
                    agent_id=scenario_agent_id,
                    pred_traj=scenario_pred_traj,
                    pred_z=scenario_pred_z,
                    pred_head=scenario_pred_head,
                    ego_only=self.ego_only,
                )
                self._update_metric_states(scenario_metrics)

    def update_from_prediction_tensors(
        self,
        scenario_files: List[str],
        agent_id: Tensor,
        agent_batch: Tensor,
        pred_traj: Tensor,
        pred_z: Tensor,
        pred_head: Tensor,
    ) -> None:
        if len(scenario_files) == 0:
            return

        self._computed = None
        self._update_count += 1

        if self._use_worker_pool():
            for (
                scenario_file,
                scenario_agent_id,
                scenario_pred_traj,
                scenario_pred_z,
                scenario_pred_head,
            ) in self.build_prediction_payloads(
                scenario_files=scenario_files,
                agent_id=agent_id,
                agent_batch=agent_batch,
                pred_traj=pred_traj,
                pred_z=pred_z,
                pred_head=pred_head,
            ):
                self._submit_scenario_metrics_from_arrays(
                    scenario_file=scenario_file,
                    agent_id=scenario_agent_id,
                    pred_traj=scenario_pred_traj,
                    pred_z=scenario_pred_z,
                    pred_head=scenario_pred_head,
                )
            return

        agent_batch_cpu = agent_batch.detach().to(device="cpu", dtype=torch.long)
        sizes = torch.bincount(agent_batch_cpu, minlength=len(scenario_files)).tolist()
        metric_device = pred_traj.device

        start = 0
        for scenario_file, size in zip(scenario_files, sizes):
            end = start + int(size)
            scenario_metrics = _compute_scenario_metrics_from_arrays(
                config=self.sim_agents_config,
                scenario_file=scenario_file,
                agent_id=agent_id[start:end],
                pred_traj=pred_traj[start:end],
                pred_z=pred_z[start:end],
                pred_head=pred_head[start:end],
                ego_only=self.ego_only,
                device=metric_device,
            )
            self._update_metric_states(scenario_metrics)
            start = end

    def compute(self) -> Dict[str, Tensor]:
        self._drain_completed_futures(wait=True, drain_all=True)
        return self.compute_from_state_tensor(
            self.get_state_tensor(device=self.scenario_counter.device)
        )
