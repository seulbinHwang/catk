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

"""
catk conda 환경에서 쓰는 Waymo Open Dataset **공식** Sim Agents 메트릭 래퍼.

**RMM(Realism Meta Metric)** 이 여기서 어떻게 도출되는지 (코드 추적 순서):

1. ``load_metrics_config()``  
   ``challenge_*_config.textproto`` 의 ``metametric_weight``·히스토그램 구간 등을 읽음.
2. ``wosac_metrics.compute_scenario_metrics_for_bundle(config, scenario, scenario_rollout)``  
   내부에서 ``metric_features.compute_scenario_rollouts_features`` 로 로그/시뮬 특징 텐서를 만든 뒤,
   ``compute_scenario_metrics_for_features_bundle`` 에서 항목별 likelihood를 계산하고
   ``_compute_metametric`` 으로 가중합 → ``scenario_metrics.metametric``.
3. 이 클래스의 ``update`` 는 시나리오마다 그 ``metametric`` 등을 **누적 합**.
4. ``compute`` 에서 ``누적값 / scenario_counter`` 로 평균낸 뒤,
   ``aggregate_metrics_to_buckets`` 로 kinematic / interactive / map_based 버킷과
   ``realism_meta_metric`` (버킷이 아니라 **전체 메타메트릭 평균과 동일 스칼라**)를 만듦.

세부 수학(히스토그램 빈·베르누리·``independent_timesteps`` 풀링)은
``waymo_open_dataset/wdl_limited/sim_agents_metrics/{metrics,estimators}.py`` 참고.
"""

import itertools
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List

import tensorflow as tf
import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as wosac_metrics
from google.protobuf import text_format
from torch import Tensor, tensor
from torchmetrics import Metric
from waymo_open_dataset.protos import (
    scenario_pb2,
    sim_agents_metrics_pb2,
    sim_agents_submission_pb2,
)


class WOSACMetrics(Metric):
    """
    Waymo Sim Agents 공식 메트릭(리더보드와 동일한 TF 파이프라인).

    반환 요약:
      - ``{prefix}/wosac/realism_meta_metric``: 시나리오 평균 **RMM**(= metametric 평균).
      - ``kinematic_metrics`` / ``interactive_metrics`` / ``map_based_metrics``:
        각 버킷 안 likelihood들의 **가중 평균**(버킷 내 weight 정규화).
      - ``{prefix}/wosac_likelihood/*``: 10개 부분 likelihood + ADE 등 시나리오 평균.
    """

    def __init__(self, prefix: str, ego_only: bool = False) -> None:
        super().__init__()
        self.is_mp_init = False
        self.prefix = prefix
        self.ego_only = ego_only
        self.wosac_config = self.load_metrics_config()

        # compute_scenario_metrics_for_bundle 이 채우는 SimAgentMetrics 필드 중
        # 시나리오별로 평균 내어 로깅할 항목 (metametric = RMM 스칼라).
        self.field_names = [
            "metametric",
            "average_displacement_error",
            "linear_speed_likelihood",
            "linear_acceleration_likelihood",
            "angular_speed_likelihood",
            "angular_acceleration_likelihood",
            "distance_to_nearest_object_likelihood",
            "collision_indication_likelihood",
            "time_to_collision_likelihood",
            "distance_to_road_edge_likelihood",
            "offroad_indication_likelihood",
            "min_average_displacement_error",
            "simulated_collision_rate",
            "simulated_offroad_rate",
        ]
        for k in self.field_names:
            self.add_state(k, default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("scenario_counter", default=tensor(0.0), dist_reduce_fx="sum")
        # WOSAC 메트릭은 TensorFlow CPU에서만 돌리도록 고정 (GPU와 메모리 충돌 방지).
        tf.config.set_visible_devices([], "GPU")

    @staticmethod
    def _compute_scenario_metrics(
        config, scenario_file, scenario_rollout, ego_only
    ) -> sim_agents_metrics_pb2.SimAgentMetrics:
        # TFRecord 하나 → scenario_pb2: 맵·전 트랙·tracks_to_predict 가 전부 들어 있음.
        # 공식 RMM은 이 protobuf와 제출 rollout을 같이 넣어 특징을 계산함.
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

        # challenge_type 기본 SIM_AGENTS. 반환 protobuf의 metametric 가 RMM.
        return wosac_metrics.compute_scenario_metrics_for_bundle(
            config, scenario, scenario_rollout
        )

    def update(
        self,
        scenario_files: List[str],
        scenario_rollouts: List[sim_agents_submission_pb2.ScenarioRollouts],
    ) -> None:

        # 배치 내 시나리오별로 TF 그래프 실행이 무겁기 때문에 멀티프로세스로 병렬화.
        # forkserver: CUDA 초기화 이후 fork 이슈를 줄이기 위한 일반적인 선택.
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") in ["", "0"]:
            if not self.is_mp_init:
                self.is_mp_init = True
                mp.set_start_method("forkserver", force=True)
            with mp.Pool(processes=len(scenario_rollouts)) as pool:
                pool_scenario_metrics = pool.starmap(
                    self._compute_scenario_metrics,
                    zip(
                        itertools.repeat(self.wosac_config),
                        scenario_files,
                        scenario_rollouts,
                        itertools.repeat(self.ego_only),
                    ),
                )
                pool.close()
                pool.join()
        else:
            pool_scenario_metrics = []
            for _scenario, _scenario_rollout in zip(scenario_files, scenario_rollouts):
                pool_scenario_metrics.append(
                    self._compute_scenario_metrics(
                        self.wosac_config, _scenario, _scenario_rollout, self.ego_only
                    )
                )

        # Distributed 학습 시 dist_reduce_fx="sum" 이므로, 여기서는 시나리오당 스칼라를 그대로 더함.
        # 최종 평균은 compute() 에서 scenario_counter 로 나눔.
        for scenario_metrics in pool_scenario_metrics:
            self.scenario_counter += 1
            self.metametric += scenario_metrics.metametric
            self.average_displacement_error += (
                scenario_metrics.average_displacement_error
            )
            self.linear_speed_likelihood += scenario_metrics.linear_speed_likelihood
            self.linear_acceleration_likelihood += (
                scenario_metrics.linear_acceleration_likelihood
            )
            self.angular_speed_likelihood += scenario_metrics.angular_speed_likelihood
            self.angular_acceleration_likelihood += (
                scenario_metrics.angular_acceleration_likelihood
            )
            self.distance_to_nearest_object_likelihood += (
                scenario_metrics.distance_to_nearest_object_likelihood
            )
            self.collision_indication_likelihood += (
                scenario_metrics.collision_indication_likelihood
            )
            self.time_to_collision_likelihood += (
                scenario_metrics.time_to_collision_likelihood
            )
            self.distance_to_road_edge_likelihood += (
                scenario_metrics.distance_to_road_edge_likelihood
            )
            self.offroad_indication_likelihood += (
                scenario_metrics.offroad_indication_likelihood
            )
            self.min_average_displacement_error += (
                scenario_metrics.min_average_displacement_error
            )
            self.simulated_collision_rate += scenario_metrics.simulated_collision_rate
            self.simulated_offroad_rate += scenario_metrics.simulated_offroad_rate

    def compute(self) -> Dict[str, Tensor]:
        # 시나리오 산술 평균 → aggregate_metrics_to_buckets 는
        # 설정의 weight 로 kinematic/interactive/map 버킷 점수를 따로 계산.
        # realism_meta_metric 는 mean_metrics.metametric 과 동일(원본 메타메트릭 평균).
        metrics_dict = {}
        for k in self.field_names:
            metrics_dict[k] = getattr(self, k) / self.scenario_counter

        mean_metrics = sim_agents_metrics_pb2.SimAgentMetrics(
            scenario_id="", **metrics_dict
        )
        final_metrics = wosac_metrics.aggregate_metrics_to_buckets(
            self.wosac_config, mean_metrics
        )

        out_dict = {
            f"{self.prefix}/wosac/realism_meta_metric": final_metrics.realism_meta_metric,
            f"{self.prefix}/wosac/kinematic_metrics": final_metrics.kinematic_metrics,
            f"{self.prefix}/wosac/interactive_metrics": final_metrics.interactive_metrics,
            f"{self.prefix}/wosac/map_based_metrics": final_metrics.map_based_metrics,
            f"{self.prefix}/wosac/min_ade": final_metrics.min_ade,
            f"{self.prefix}/wosac/scenario_counter": self.scenario_counter,
        }
        for k in self.field_names:
            out_dict[f"{self.prefix}/wosac_likelihood/{k}"] = metrics_dict[k]

        return out_dict

    @staticmethod
    def load_metrics_config() -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
        # pip 패키지에 포함된 textproto. 리더보드 연도와 맞추려면
        # challenge_2025_sim_agents_config.textproto 등으로 교체 검토
        # (가중치·히스토그램 범위가 연도별로 다를 수 있음).
        config_path = (
            Path(wosac_metrics.__file__).parent / "challenge_2024_config.textproto"
        )
        with open(config_path, "r") as f:
            config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
            text_format.Parse(f.read(), config)
        return config
