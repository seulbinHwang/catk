# Hard-RMM per-scenario 헬퍼.
#
# `HardSimAgentsMetrics.update_from_prediction_tensors` 는 batch-aggregated
# scalar 만 누적/반환하므로, per-scenario 단위로 metametric + 10 개 likelihood
# 를 분리해서 wandb 로깅에 쓰려면 같은 내부 logic 을 그대로 재사용하면서
# `hard_per_scenario` 누적 단계만 분기시켜야 한다.
#
# 본 모듈은 그 inner-loop 만 떼어내 `compute_hard_rmm_per_scenario` 로 노출한다.
# 코드 흐름은 src/smart/metrics/__init__.py 의 update_from_prediction_tensors
# (line 434~) 와 1:1 매칭이다 — log feature load 는 `_log_feat_cache` 와
# `_disk_cache_dir` 를 그대로 재활용해 동일 epoch 의 두 metric 인스턴스 (OL/CL)
# 가 캐시 hit 으로 빠르게 동작한다.

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch_geometric.utils import degree as _tg_degree

# Hard RMM 본 클래스 (cache, config, pool 활용을 위해 import)
from src.smart.metrics import HardSimAgentsMetrics, _LIKELIHOOD_NAMES


def compute_hard_rmm_per_scenario(
    metric: HardSimAgentsMetrics,
    *,
    scenario_files: List[str],
    agent_id: torch.Tensor,
    agent_batch: torch.Tensor,
    pred_traj: torch.Tensor,    # [n_agents, G, 80, 2]
    pred_z: torch.Tensor,       # [n_agents, G, 80]
    pred_head: torch.Tensor,    # [n_agents, G, 80]
    update_running: bool = True,
) -> List[Dict[str, Any]]:
    """Hard RMM 을 시나리오별로 분리해서 metametric + 10 likelihoods 반환.

    Args:
        metric: 기존 ``HardSimAgentsMetrics`` 인스턴스. cache + pool 재활용.
        scenario_files: TFRecord 파일 경로 리스트, 길이 ``n_scenarios``.
        agent_id: ``[n_agents]`` 객체 ID. ``data["agent"]["id"]`` 그대로 가능.
        agent_batch: ``[n_agents]`` graph 배정 (PyG style).
        pred_traj/pred_z/pred_head: WOSAC 8 초 (80 step, 10Hz) world-frame 예측.
        update_running: ``True`` 면 ``metric._metametric_sum`` / ``_per_metric_sums``
            / ``_count`` 누적도 함께 수행 (epoch 끝에서 ``metric.compute()``
            가 정상 동작하도록).

    Returns:
        ``len == n_scenarios`` 의 dict 리스트. 각 dict 의 키는
        ``scenario_file`` (str), ``metametric`` (float), ``_LIKELIHOOD_NAMES`` 의
        10 개 항목 (각 float). 시나리오 순서는 입력 ``scenario_files`` 와 동일.
    """
    # 아래 import 는 metric 본체가 사용하는 inner module 들. circular 위험 없음.
    from src.smart.metrics.wosac_metric_features_torch.metric_features_torch_differentiable import (
        PredictedSimTrajectories,
        compute_metric_features_batched_scenes,
    )
    from src.smart.metrics.wosac_metric_features_torch.metric_features_torch import (
        compute_metric_features,
        scenario_to_joint_scene,
    )
    from src.smart.metrics.wosac_metametric_pytorch import (
        compute_wosac_metametric_from_features_torch,
    )
    from waymo_open_dataset.protos import scenario_pb2 as _scenario_pb2
    from waymo_open_dataset.utils.sim_agents import submission_specs

    n_scenarios = len(scenario_files)
    if n_scenarios == 0:
        return []

    sizes = [int(s) for s in _tg_degree(agent_batch, dtype=torch.long).tolist()]
    device = pred_traj.device
    G = int(pred_traj.shape[1])
    T_pred = int(pred_traj.shape[2])
    _challenge = submission_specs.ChallengeType.SIM_AGENTS

    id_splits = agent_id.split(sizes)
    traj_splits = pred_traj.split(sizes)
    z_splits = pred_z.split(sizes)
    head_splits = pred_head.split(sizes)

    # ── log_feat 로드 (HardSimAgentsMetrics 의 mem/disk 캐시 재사용) ──────────
    pool = metric._get_pool()
    _mem_cache = HardSimAgentsMetrics._log_feat_cache
    _disk_dir = HardSimAgentsMetrics._disk_cache_dir
    os.makedirs(_disk_dir, exist_ok=True)

    def _disk_path(sf: str) -> str:
        bn = os.path.basename(sf).replace(os.sep, "_")
        return os.path.join(_disk_dir, f"{bn}.pkl")

    slot: list = [None] * n_scenarios
    miss_idx: list = []
    miss_files: list = []
    for j, sf in enumerate(scenario_files):
        cached = _mem_cache.get(sf)
        if cached is not None:
            slot[j] = cached
            continue
        dp = _disk_path(sf)
        if os.path.exists(dp):
            try:
                with open(dp, "rb") as _f:
                    res = pickle.load(_f)
                _mem_cache[sf] = res
                slot[j] = res
                continue
            except Exception:
                pass
        miss_idx.append(j)
        miss_files.append(sf)

    if miss_files:
        if pool is not None and len(miss_files) > 1:
            from src.smart.metrics import _hard_load_and_log_feat_worker
            miss_results = pool.starmap(
                _hard_load_and_log_feat_worker,
                [(sf, _challenge) for sf in miss_files],
            )
        else:
            # serial fallback
            from src.smart.metrics import _hard_load_and_log_feat_worker
            miss_results = [
                _hard_load_and_log_feat_worker(sf, _challenge) for sf in miss_files
            ]
        for j, sf, res in zip(miss_idx, miss_files, miss_results):
            _mem_cache[sf] = res
            slot[j] = res
            try:
                dp = _disk_path(sf)
                tmp = dp + ".tmp"
                with open(tmp, "wb") as _f:
                    pickle.dump(res, _f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, dp)
            except Exception:
                pass

    # 시나리오 protobuf 와 log_feat dict 분리
    scenarios = []
    log_feat_dicts: List[dict] = []
    for sc_bytes, lf_dict in slot:
        sc = _scenario_pb2.Scenario()
        sc.ParseFromString(sc_bytes)
        scenarios.append(sc)
        log_feat_dicts.append(lf_dict)

    # ── Sim feature 계산 (rollout 별로 모든 시나리오 batched) ────────────────
    sim_feat_per_g: List[list] = []
    with torch.no_grad():
        for g in range(G):
            preds_g = [
                PredictedSimTrajectories(
                    object_id=id_splits[i].cpu(),
                    center_x=traj_splits[i][:, g, :, 0],
                    center_y=traj_splits[i][:, g, :, 1],
                    center_z=z_splits[i][:, g, :],
                    heading=head_splits[i][:, g, :],
                    valid=torch.ones(sizes[i], T_pred, dtype=torch.bool, device=device),
                )
                for i in range(n_scenarios)
            ]
            feat_list_g = compute_metric_features_batched_scenes(
                scenarios=scenarios, preds=preds_g, surrogate=None,
            )
            sim_feat_per_g.append(feat_list_g)

    # G rollout 을 시나리오별로 stack 해서 (G, E_i, T) 형태로 만듦
    sim_feat_dicts: List[dict] = []
    for i in range(n_scenarios):
        feats_i = [sim_feat_per_g[g][i] for g in range(G)]

        def _cat(field: str) -> torch.Tensor:
            return torch.cat([getattr(f, field) for f in feats_i], dim=0)

        sim_feat_dicts.append({
            "object_id": feats_i[0].object_id,
            "object_type": _cat("object_type"),
            "valid": _cat("valid"),
            "average_displacement_error": _cat("average_displacement_error"),
            "linear_speed": _cat("linear_speed"),
            "linear_acceleration": _cat("linear_acceleration"),
            "angular_speed": _cat("angular_speed"),
            "angular_acceleration": _cat("angular_acceleration"),
            "distance_to_nearest_object": _cat("distance_to_nearest_object"),
            "collision_per_step": _cat("collision_per_step"),
            "time_to_collision": _cat("time_to_collision"),
            "distance_to_road_edge": _cat("distance_to_road_edge"),
            "offroad_per_step": _cat("offroad_per_step"),
            "traffic_light_violation_per_step": _cat("traffic_light_violation_per_step"),
        })

    # ── 시나리오별 metametric + likelihood 계산 ──────────────────────────────
    out: List[Dict[str, Any]] = []
    for i in range(n_scenarios):
        result = compute_wosac_metametric_from_features_torch(
            metric._config, log_feat_dicts[i], sim_feat_dicts[i]
        )
        d: Dict[str, Any] = {
            "scenario_file": scenario_files[i],
            "metametric": float(result.metametric),
        }
        for name in _LIKELIHOOD_NAMES:
            d[name] = float(getattr(result, name))
        out.append(d)
        if update_running:
            metric._metametric_sum += d["metametric"]
            metric._count += 1
            for name in _LIKELIHOOD_NAMES:
                metric._per_metric_sums[name] += d[name]
    return out
