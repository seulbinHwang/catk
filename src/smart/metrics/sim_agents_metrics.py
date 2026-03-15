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

import itertools
import multiprocessing as mp
import os
from typing import Dict, List, Sequence, Tuple

import tensorflow as tf
import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as official_sim_agents_metrics
from google.protobuf.descriptor import Descriptor, FieldDescriptor
from torch import Tensor, tensor
from torchmetrics import Metric
from waymo_open_dataset.protos import (
    scenario_pb2,
    sim_agents_metrics_pb2,
    sim_agents_submission_pb2,
)
from waymo_open_dataset.utils.sim_agents import submission_specs

_SIM_AGENTS_2025_NAMESPACE = "sim_agents_2025"
_SIM_AGENTS_2025_CHALLENGE_TYPE = submission_specs.ChallengeType.SIM_AGENTS
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
    try:
        config = official_sim_agents_metrics.load_metrics_config(
            _SIM_AGENTS_2025_CHALLENGE_TYPE
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "설치된 waymo-open-dataset 패키지에서 공식 2025 Sim Agents 평가기를 찾지 못했습니다. "
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
            f"scenario missing={missing_scenario_fields}, "
            f"bucket missing={missing_bucket_fields}."
        )


class SimAgentsMetrics(Metric):
    """Waymo 공식 2025 Sim Agents 평가기를 torchmetrics 형태로 감싼 클래스입니다."""

    def __init__(self, prefix: str, ego_only: bool = False) -> None:
        super().__init__()
        self.is_mp_init = False
        self.prefix = prefix
        self.ego_only = ego_only
        self.metric_namespace = f"{self.prefix}/{_SIM_AGENTS_2025_NAMESPACE}"
        self.metric_mean_namespace = f"{self.prefix}/{_SIM_AGENTS_2025_NAMESPACE}_mean"

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
        tf.config.set_visible_devices([], "GPU")

    @staticmethod
    def _compute_scenario_metrics(
        config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
        scenario_file: str,
        scenario_rollout: sim_agents_submission_pb2.ScenarioRollouts,
        ego_only: bool,
    ) -> sim_agents_metrics_pb2.SimAgentMetrics:
        scenario = scenario_pb2.Scenario()
        for data in tf.data.TFRecordDataset([scenario_file], compression_type=""):
            scenario.ParseFromString(bytes(data.numpy()))
            break
        if ego_only:
            for i in range(len(scenario.tracks)):
                if i != scenario.sdc_track_index:
                    for t in range(91):
                        scenario.tracks[i].states[t].valid = False
            while len(scenario.tracks_to_predict) > 1:
                scenario.tracks_to_predict.pop()
            scenario.tracks_to_predict[0].track_index = scenario.sdc_track_index

        return official_sim_agents_metrics.compute_scenario_metrics_for_bundle(
            config,
            scenario,
            scenario_rollout,
            challenge_type=_SIM_AGENTS_2025_CHALLENGE_TYPE,
        )

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

        use_single_process = os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ["", "0"]
        if use_single_process or len(scenario_rollouts) == 1:
            pool_scenario_metrics = []
            for scenario_file, scenario_rollout in zip(scenario_files, scenario_rollouts):
                pool_scenario_metrics.append(
                    self._compute_scenario_metrics(
                        self.sim_agents_config,
                        scenario_file,
                        scenario_rollout,
                        self.ego_only,
                    )
                )
        else:
            if not self.is_mp_init:
                self.is_mp_init = True
                mp.set_start_method("forkserver", force=True)
            with mp.Pool(processes=len(scenario_rollouts)) as pool:
                pool_scenario_metrics = pool.starmap(
                    self._compute_scenario_metrics,
                    zip(
                        itertools.repeat(self.sim_agents_config),
                        scenario_files,
                        scenario_rollouts,
                        itertools.repeat(self.ego_only),
                    ),
                )

        for scenario_metrics in pool_scenario_metrics:
            self._update_metric_states(scenario_metrics)

    def compute(self) -> Dict[str, Tensor]:
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
