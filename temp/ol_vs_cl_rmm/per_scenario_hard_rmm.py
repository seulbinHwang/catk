# Hard-RMM 2초 horizon per-scenario 헬퍼.
#
# 의도: ``HardSimAgentsMetrics.update_from_prediction_tensors`` 와 동일한
# **non-differentiable** hard RMM 경로 (`compute_metric_features` 가
# JointScene proto 를 받아 feature 추출 → ``compute_wosac_metametric_from_features_torch``)
# 를 사용하되, GT future 와 sim trajectory 모두 ``T_pred`` (2초 = 20 step) 로
# 잘라서 정합성을 깨지 않으면서 short-horizon RMM 을 측정한다.
#
# 정합성 (vs 8초 hard RMM):
#   - feature extractor / metametric 호출 코드는 동일 — 단지 시간축 길이만 달라짐.
#   - histogram bin 범위는 8초 분포 기준 (config.linear_speed.histogram 등) 이므로
#     절대값 likelihood 는 8초와 직접 비교 불가, OL/CL 간 상대 비교 (Δ) 만 의미.
#   - cache (`_log_feat_cache` / `_disk_cache_dir`) 는 8초 log_feat 만 저장하므로
#     2초 변형은 매 시나리오 새로 계산 (caching 은 future work).
from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import torch
from torch_geometric.utils import degree as _tg_degree

# 2초 변형 hard RMM 본체 (temp/ol_vs_cl_rmm/hard_rmm_2s/) 호출
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from temp.ol_vs_cl_rmm.hard_rmm_2s.metric_features_torch_2s import (
    compute_scenario_rollouts_features as _compute_scenario_rollouts_features_2s,
)
from temp.ol_vs_cl_rmm.hard_rmm_2s.wosac_metametric_pytorch_2s import (
    compute_wosac_metametric_from_features_torch as _compute_metametric_2s,
)
from src.smart.metrics import HardSimAgentsMetrics, _LIKELIHOOD_NAMES


def compute_hard_rmm_per_scenario(
    metric: HardSimAgentsMetrics,
    *,
    scenario_files: List[str],
    agent_id: torch.Tensor,
    agent_batch: torch.Tensor,
    pred_traj: torch.Tensor,    # [n_agents, G, T_pred, 2]
    pred_z: torch.Tensor,       # [n_agents, G, T_pred]
    pred_head: torch.Tensor,    # [n_agents, G, T_pred]
    update_running: bool = True,
) -> List[Dict[str, Any]]:
    """2초 horizon hard-RMM 을 시나리오별로 분리해서 metametric + 10 likelihoods 반환.

    이 함수는 **2초 변형** ``compute_scenario_rollouts_features_2s`` 를 사용해
    sim 길이 (T_pred) 로 GT future 를 잘라 log/sim feature 시간축을 맞춘다.
    원본 hard RMM (8초) 파이프라인 (``HardSimAgentsMetrics.update_from_prediction_tensors``)
    와 코드 흐름은 동일하지만, ``T_pred`` 가 짧을 때만 (예: 20=2초) 의미 있는
    경로다.  ``metric._scenario_cache`` 만 재활용 (TFRecord proto 파싱 캐시).

    Args:
        metric: 기존 ``HardSimAgentsMetrics`` 인스턴스.  scenario proto cache
            (``_load_scenario``) 와 누적 ``_metametric_sum`` / ``_count`` 만 사용.
        pred_traj/pred_z/pred_head: world-frame 짧은 (``T_pred``) prediction.

    Returns:
        ``len == n_scenarios`` 의 dict 리스트.  키: ``scenario_file``,
        ``metametric``, ``_LIKELIHOOD_NAMES`` 의 10 항목.
    """
    from waymo_open_dataset.protos import sim_agents_submission_pb2

    n_scenarios = len(scenario_files)
    if n_scenarios == 0:
        return []

    sizes = [int(s) for s in _tg_degree(agent_batch, dtype=torch.long).tolist()]
    G = int(pred_traj.shape[1])
    T_pred = int(pred_traj.shape[2])

    id_splits = [t.cpu().numpy() for t in agent_id.cpu().split(sizes)]
    traj_splits = [t.cpu().numpy() for t in pred_traj.cpu().split(sizes)]
    z_splits = [t.cpu().numpy() for t in pred_z.cpu().split(sizes)]
    head_splits = [t.cpu().numpy() for t in pred_head.cpu().split(sizes)]

    out: List[Dict[str, Any]] = []
    for i in range(n_scenarios):
        sf = scenario_files[i]
        scenario = metric._load_scenario(sf)   # proto cache hit (HardSimAgentsMetrics 의 LRU)

        # ── 시나리오별 ScenarioRollouts proto 빌드 (G rollouts) ───────────────
        n_agents = int(traj_splits[i].shape[0])
        joint_scenes = []
        for g in range(G):
            sim_trajs = []
            for a in range(n_agents):
                sim_trajs.append(
                    sim_agents_submission_pb2.SimulatedTrajectory(
                        center_x=traj_splits[i][a, g, :, 0],
                        center_y=traj_splits[i][a, g, :, 1],
                        center_z=z_splits[i][a, g],
                        heading=head_splits[i][a, g],
                        object_id=int(id_splits[i][a]),
                    )
                )
            joint_scenes.append(
                sim_agents_submission_pb2.JointScene(simulated_trajectories=sim_trajs)
            )
        scenario_rollouts = sim_agents_submission_pb2.ScenarioRollouts(
            joint_scenes=joint_scenes,
            scenario_id=scenario.scenario_id,
        )

        # ── 2초 변형 hard RMM feature extraction ─────────────────────────────
        log_feat, sim_feat = _compute_scenario_rollouts_features_2s(
            scenario,
            scenario_rollouts,
            n_steps_override=T_pred,    # 핵심: GT future 도 T_pred 로 잘라 log T 매치
        )
        # ── metametric (dim-dynamic) ─────────────────────────────────────────
        result = _compute_metametric_2s(
            metric._config, log_feat.as_dict(), sim_feat.as_dict(),
        )

        d: Dict[str, Any] = {
            "scenario_file": sf,
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
