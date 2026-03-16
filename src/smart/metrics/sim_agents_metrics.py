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

import concurrent.futures as cf
import inspect
import multiprocessing as mp
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as official_sim_agents_metrics
from google.protobuf.descriptor import Descriptor, FieldDescriptor
from google.protobuf import text_format
from torch import Tensor, tensor
from torchmetrics import Metric
from waymo_open_dataset.protos import (
    scenario_pb2,
    sim_agents_metrics_pb2,
    sim_agents_submission_pb2,
)
from waymo_open_dataset.utils.sim_agents import submission_specs

_SIM_AGENTS_2025_NAMESPACE = "sim_agents_2025"
_SIM_AGENTS_2025_CHALLENGE_TYPE = getattr(
    getattr(submission_specs, "ChallengeType", None),
    "SIM_AGENTS",
    None,
)
_WAYMO_SIM_AGENTS_METRICS_DIR = Path(official_sim_agents_metrics.__file__).resolve().parent
_SIM_AGENTS_2025_CONFIG_FILENAME = "challenge_2025_sim_agents_config.textproto"
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
_WORKER_SIM_AGENTS_CONFIG: sim_agents_metrics_pb2.SimAgentMetricsConfig | None = None
_WORKER_EGO_ONLY = False
_TF_RUNTIME_CONFIGURED = False


def _read_nonnegative_int_env(var_name: str, default: int) -> int:
    raw_value = os.environ.get(var_name, "").strip()
    if not raw_value:
        return default
    try:
        return max(0, int(raw_value))
    except ValueError as exc:
        raise RuntimeError(f"{var_name} must be an integer, got {raw_value!r}.") from exc


def _configure_tensorflow_runtime() -> None:
    global _TF_RUNTIME_CONFIGURED
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
    tf_record_iterator = getattr(tf.compat.v1.io, "tf_record_iterator", None)
    if tf_record_iterator is not None:
        for record in tf_record_iterator(record_path):
            return bytes(record)
        raise RuntimeError(f"TFRecord file is empty: {record_path}")

    dataset = tf.data.TFRecordDataset([record_path], compression_type="")
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)
    for data in dataset:
        return bytes(data.numpy())
    raise RuntimeError(f"TFRecord file is empty: {record_path}")


def _get_waymo_version_string() -> str:
    try:
        return version("waymo-open-dataset-tf-2-12-0")
    except PackageNotFoundError:
        return "unknown"


def _load_textproto_metrics_config(
    config_filename: str,
) -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
    config_path = _WAYMO_SIM_AGENTS_METRICS_DIR / config_filename
    if not config_path.exists():
        raise RuntimeError(
            "Waymo 2025 Sim Agents config 파일을 찾지 못했습니다. "
            f"expected={config_path}. "
            "README 기준으로 waymo-open-dataset-tf-2-12-0==1.6.7 이상을 다시 설치해야 합니다."
        )

    config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
    with config_path.open("r", encoding="utf-8") as handle:
        text_format.Parse(handle.read(), config)
    return config


def _get_scalar_field_names(
    message_descriptor: Descriptor,
    skip_names: Sequence[str] = (),
) -> Tuple[str, ...]:
    """프로토 메시지에서 숫자 하나짜리 필드 이름만 뽑습니다.

    Args:
        message_descriptor: 필드 구조가 담긴 프로토 설명 정보입니다.
        skip_names: 제외할 필드 이름 목록입니다.

    Returns:
        Tuple[str, ...]: 문자열이나 중첩 구조를 뺀 숫자 필드 이름들입니다.
    """
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


def _load_waymo_sim_agents_2025_config(
) -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
    """Waymo 공식 2025 Sim Agents 설정을 읽고 바로 검증합니다.

    Args:
        없음.

    Returns:
        sim_agents_metrics_pb2.SimAgentMetricsConfig:
            공식 2025 Sim Agents 채점 설정입니다.

    Raises:
        RuntimeError: 설치된 Waymo 패키지가 2025 Sim Agents 평가를 지원하지 않을 때 발생합니다.
    """
    if _SIM_AGENTS_2025_CHALLENGE_TYPE is None:
        raise RuntimeError(
            "설치된 waymo-open-dataset 패키지가 2025 Sim Agents challenge type를 제공하지 않습니다. "
            f"현재 버전={_get_waymo_version_string()}. "
            "WOSAC 2024 평가는 허용되지 않으며, README 기준으로 "
            "waymo-open-dataset-tf-2-12-0==1.6.7 이상이 필요합니다."
        )

    try:
        config = official_sim_agents_metrics.load_metrics_config(
            _SIM_AGENTS_2025_CHALLENGE_TYPE
        )
    except FileNotFoundError:
        # 1.6.7 wheel에서는 공식 helper가 상대경로로 textproto를 찾다가 실패할 수 있습니다.
        config = _load_textproto_metrics_config(_SIM_AGENTS_2025_CONFIG_FILENAME)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "설치된 waymo-open-dataset 패키지에서 공식 2025 Sim Agents 평가기를 찾지 못했습니다. "
            f"현재 버전={_get_waymo_version_string()}. "
            "README에 맞춰 waymo-open-dataset-tf-2-12-0==1.6.7 이상을 설치해야 합니다."
        ) from exc

    traffic_light_weight = float(config.traffic_light_violation.metametric_weight)
    if traffic_light_weight <= 0.0:
        raise RuntimeError(
            "공식 Sim Agents 설정을 읽었지만 traffic_light_violation 가중치가 0입니다. "
            "이 환경은 2025 Sim Agents 평가 환경이 아닙니다."
        )
    return config


def _validate_waymo_sim_agents_2025_runtime_support() -> None:
    """설치된 Waymo 패키지가 2025 출력 필드를 실제로 가지는지 확인합니다.

    Args:
        없음.

    Returns:
        없음.

    Raises:
        RuntimeError: 2025 전용 필드가 빠져 있을 때 발생합니다.
    """
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


def _compute_waymo_sim_agents_metrics_for_bundle(
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    scenario: scenario_pb2.Scenario,
    scenario_rollout: sim_agents_submission_pb2.ScenarioRollouts,
) -> sim_agents_metrics_pb2.SimAgentMetrics:
    compute_fn = official_sim_agents_metrics.compute_scenario_metrics_for_bundle
    signature = inspect.signature(compute_fn)

    if "challenge_type" not in signature.parameters:
        raise RuntimeError(
            "설치된 waymo-open-dataset 패키지가 2025 Sim Agents scorer 시그니처를 제공하지 않습니다. "
            f"현재 버전={_get_waymo_version_string()}. "
            "WOSAC 2024 평가는 허용되지 않습니다."
        )

    return compute_fn(
        config,
        scenario,
        scenario_rollout,
        challenge_type=_SIM_AGENTS_2025_CHALLENGE_TYPE,
    )


def _compute_scenario_metrics(
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    scenario_file: str,
    scenario_rollout: sim_agents_submission_pb2.ScenarioRollouts,
    ego_only: bool,
) -> sim_agents_metrics_pb2.SimAgentMetrics:
    _configure_tensorflow_runtime()
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(_read_single_record_tfrecord(scenario_file))
    if ego_only:
        for i in range(len(scenario.tracks)):
            if i != scenario.sdc_track_index:
                for t in range(91):
                    scenario.tracks[i].states[t].valid = False
        while len(scenario.tracks_to_predict) > 1:
            scenario.tracks_to_predict.pop()
        scenario.tracks_to_predict[0].track_index = scenario.sdc_track_index

    return _compute_waymo_sim_agents_metrics_for_bundle(
        config,
        scenario,
        scenario_rollout,
    )


def _init_sim_agents_metrics_worker(config_bytes: bytes, ego_only: bool) -> None:
    global _WORKER_SIM_AGENTS_CONFIG, _WORKER_EGO_ONLY
    _WORKER_SIM_AGENTS_CONFIG = sim_agents_metrics_pb2.SimAgentMetricsConfig()
    _WORKER_SIM_AGENTS_CONFIG.ParseFromString(config_bytes)
    _WORKER_EGO_ONLY = ego_only
    _configure_tensorflow_runtime()


def _compute_scenario_metrics_worker(
    scenario_file: str,
    scenario_rollout_bytes: bytes,
) -> bytes:
    if _WORKER_SIM_AGENTS_CONFIG is None:
        raise RuntimeError("Sim Agents metrics worker was used before it was initialized.")

    scenario_rollout = sim_agents_submission_pb2.ScenarioRollouts()
    scenario_rollout.ParseFromString(scenario_rollout_bytes)
    scenario_metrics = _compute_scenario_metrics(
        config=_WORKER_SIM_AGENTS_CONFIG,
        scenario_file=scenario_file,
        scenario_rollout=scenario_rollout,
        ego_only=_WORKER_EGO_ONLY,
    )
    return scenario_metrics.SerializeToString()


def _resolve_sim_agents_metric_workers() -> int:
    override = os.environ.get("CATK_SIM_AGENTS_METRIC_WORKERS", "").strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError as exc:
            raise RuntimeError(
                f"CATK_SIM_AGENTS_METRIC_WORKERS must be an integer, got {override!r}."
            ) from exc

    cpu_count = max(1, os.cpu_count() or 1)
    local_world_size = max(1, _read_nonnegative_int_env("LOCAL_WORLD_SIZE", 0) or 1)
    data_workers = _read_nonnegative_int_env("CATK_DATA_WORKERS", 0)

    reserved_cpu_budget = local_world_size * max(1, data_workers + 1)
    free_cpu_budget = max(1, cpu_count - reserved_cpu_budget)
    per_rank_budget = max(1, free_cpu_budget // local_world_size)
    worker_cap = 4 if local_world_size > 1 else 8
    return max(1, min(worker_cap, per_rank_budget))


class SimAgentsMetrics(Metric):
    """Waymo 공식 2025 Sim Agents 평가기를 torchmetrics 형태로 감싼 클래스입니다."""

    def __init__(self, prefix: str, ego_only: bool = False) -> None:
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
        self._max_workers = _resolve_sim_agents_metric_workers()
        self._max_pending_futures = max(self._max_workers * 4, self._max_workers)
        self._executor: cf.ProcessPoolExecutor | None = None
        self._pending_futures: Dict[cf.Future[bytes], int] = {}
        self._completed_results: Dict[int, bytes] = {}
        self._next_submission_index = 0
        self._next_result_index = 0
        self._worker_config_bytes = self.sim_agents_config.SerializeToString()

    @staticmethod
    def _compute_scenario_metrics(
        config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
        scenario_file: str,
        scenario_rollout: sim_agents_submission_pb2.ScenarioRollouts,
        ego_only: bool,
    ) -> sim_agents_metrics_pb2.SimAgentMetrics:
        return _compute_scenario_metrics(
            config,
            scenario_file,
            scenario_rollout,
            ego_only,
        )

    def _ensure_executor(self) -> None:
        if self._max_workers <= 1 or self._executor is not None:
            return

        self._executor = cf.ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_sim_agents_metrics_worker,
            initargs=(self._worker_config_bytes, self.ego_only),
        )

    def _shutdown_executor(self) -> None:
        executor = getattr(self, "_executor", None)
        if executor is None:
            return
        executor.shutdown(wait=False, cancel_futures=True)
        self._executor = None

    def _update_metric_states_from_bytes(self, metric_bytes: bytes) -> None:
        scenario_metrics = sim_agents_metrics_pb2.SimAgentMetrics()
        scenario_metrics.ParseFromString(metric_bytes)
        self._update_metric_states(scenario_metrics)

    def _drain_completed_futures(self, wait: bool, drain_all: bool = False) -> None:
        if not self._pending_futures:
            return

        if wait:
            done, not_done = cf.wait(
                tuple(self._pending_futures),
                return_when=cf.ALL_COMPLETED if drain_all else cf.FIRST_COMPLETED,
            )
        else:
            done = {future for future in self._pending_futures if future.done()}
            if not done:
                return

        for future in done:
            result_index = self._pending_futures.pop(future)
            self._completed_results[result_index] = future.result()

        while self._next_result_index in self._completed_results:
            self._update_metric_states_from_bytes(
                self._completed_results.pop(self._next_result_index)
            )
            self._next_result_index += 1

    def __del__(self) -> None:
        self._shutdown_executor()

    def reset(self) -> None:
        super().reset()
        self._shutdown_executor()
        if hasattr(self, "_pending_futures"):
            self._pending_futures.clear()
        if hasattr(self, "_completed_results"):
            self._completed_results.clear()
        self._next_submission_index = 0
        self._next_result_index = 0

    def _update_metric_states(
        self,
        scenario_metrics: sim_agents_metrics_pb2.SimAgentMetrics,
    ) -> None:
        """시나리오 하나의 점수를 누적 상태에 더합니다.

        Args:
            scenario_metrics: 공식 평가기가 돌려준 시나리오 단위 점수입니다.

        Returns:
            없음.
        """
        self.scenario_counter.add_(1.0)
        for field_name in self.scenario_metric_field_names:
            getattr(self, field_name).add_(float(getattr(scenario_metrics, field_name)))

    def _build_zero_output_dict(self) -> Dict[str, Tensor]:
        """채점된 시나리오가 없을 때도 같은 구조의 0점 결과를 만듭니다.

        Args:
            없음.

        Returns:
            Dict[str, Tensor]: 실제 compute 출력과 같은 키 구조의 0 텐서 사전입니다.
        """
        zero_value = self.scenario_counter * 0.0
        out_dict: Dict[str, Tensor] = {
            f"{self.metric_namespace}/scenario_counter": zero_value.clone(),
        }
        for field_name in self.bucket_metric_field_names:
            out_dict[f"{self.metric_namespace}/{field_name}"] = zero_value.clone()
        for field_name in self.scenario_metric_field_names:
            out_dict[f"{self.metric_mean_namespace}/{field_name}"] = zero_value.clone()
        return out_dict

    def update(
        self,
        scenario_files: List[str],
        scenario_rollouts: List[sim_agents_submission_pb2.ScenarioRollouts],
    ) -> None:
        if len(scenario_rollouts) == 0:
            return

        if self._max_workers <= 1 or len(scenario_rollouts) == 1:
            for scenario_file, scenario_rollout in zip(scenario_files, scenario_rollouts):
                scenario_metrics = self._compute_scenario_metrics(
                    self.sim_agents_config,
                    scenario_file,
                    scenario_rollout,
                    self.ego_only,
                )
                self._update_metric_states(scenario_metrics)
            return

        self._ensure_executor()
        for scenario_file, scenario_rollout in zip(scenario_files, scenario_rollouts):
            self._pending_futures[
                self._executor.submit(
                    _compute_scenario_metrics_worker,
                    scenario_file,
                    scenario_rollout.SerializeToString(),
                )
            ] = self._next_submission_index
            self._next_submission_index += 1

        self._drain_completed_futures(wait=False)
        while len(self._pending_futures) > self._max_pending_futures:
            self._drain_completed_futures(wait=True)

    def compute(self) -> Dict[str, Tensor]:
        self._drain_completed_futures(wait=True, drain_all=True)
        if self.scenario_counter.item() == 0:
            return self._build_zero_output_dict()

        mean_metric_tensors = {
            field_name: getattr(self, field_name) / self.scenario_counter
            for field_name in self.scenario_metric_field_names
        }
        mean_metric_scalars = {
            field_name: float(metric_value.item())
            for field_name, metric_value in mean_metric_tensors.items()
        }
        mean_metrics = sim_agents_metrics_pb2.SimAgentMetrics(
            scenario_id="",
            **mean_metric_scalars,
        )
        bucket_metrics = official_sim_agents_metrics.aggregate_metrics_to_buckets(
            self.sim_agents_config,
            mean_metrics,
        )

        out_dict: Dict[str, Tensor] = {
            f"{self.metric_namespace}/scenario_counter": self.scenario_counter.clone(),
        }
        for field_name in self.bucket_metric_field_names:
            out_dict[f"{self.metric_namespace}/{field_name}"] = (
                self.scenario_counter.new_tensor(float(getattr(bucket_metrics, field_name)))
            )
        for field_name, metric_value in mean_metric_tensors.items():
            out_dict[f"{self.metric_mean_namespace}/{field_name}"] = metric_value
        return out_dict
