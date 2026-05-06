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

"""Pure-PyTorch Hard-RMM evaluator for WOSAC 2025.

`SimAgentsMetrics` 와 동일한 외부 인터페이스를 갖되, official TF metric 호출을
완전히 제거한 in-process 구현입니다.  forkserver pool 로 시나리오별 TFRecord
로딩 + log feature 계산만 병렬화하고, RMM 본 계산은 PyTorch 로 수행하여
TF/protobuf 가 GPU CUDA context 와 충돌하지 않도록 격리합니다.

`smart_flow.py` 등에서는 `SimAgentsMetrics` 와 드롭인 교체 가능합니다 (단,
출력 dict 의 정확한 구조는 다를 수 있습니다 — Phase 2 통합 시 정합 확인).

Worker 함수 (`_hard_load_and_log_feat_worker`) 는 multiprocessing pickling 을
위해 모듈 top-level 에 정의됩니다.
"""

import multiprocessing as mp
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch_geometric.utils import degree as _tg_degree


_LIKELIHOOD_NAMES: List[str] = [
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



def _hard_load_and_log_feat_worker(scenario_file: str, challenge):
    """Worker for HardSimAgentsMetrics: load TFRecord scenario + compute log features.

    challenge: ChallengeType enum (pickled from main).
    Returns (scenario_serialized_bytes, log_feat_dict). All tensors CPU.
    """
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2
    from src.smart.metrics.wosac_metric_features_torch.metric_features_torch import (
        compute_metric_features,
        scenario_to_joint_scene,
    )

    tf.config.set_visible_devices([], "GPU")

    scenario = scenario_pb2.Scenario()
    for tfdata in tf.data.TFRecordDataset([scenario_file], compression_type=""):
        scenario.ParseFromString(bytes(tfdata.numpy()))
        break

    log_joint = scenario_to_joint_scene(scenario, challenge)
    lf = compute_metric_features(
        scenario, log_joint, challenge_type=challenge, use_log_validity=True
    )
    return scenario.SerializeToString(), lf.as_dict()




class HardSimAgentsMetrics:
    """Pure-PyTorch hard RMM 계산기 — subprocess·TF metric computation 없음.

    SimAgentsMetrics 와 동일한 인터페이스를 제공하므로 smart_flow.py 에서
    드롭인 교체가 가능합니다.

    속도 개선 포인트 (SimAgentsMetrics 대비):
    - subprocess(forkserver) 생성 오버헤드 없음 — 배치당 proc spawn 비용 제거.
    - TF metric/feature computation 없음 — PyTorch 로 인-프로세스 계산.
    - Scenario proto LRU 캐시(기본 256) — 같은 epoch 내 동일 시나리오
      TFRecord 재파싱 방지.
    - WOSAC_TORCH_COMPILE=1 env var 로 dno/ttc/d_road kernel torch.compile 활성화.
    """

    _SCENARIO_CACHE_MAX: int = 256
    # Module-level (process-local) cache: scenario_file_path -> (sc_bytes, lf_dict).
    _log_feat_cache: Dict[str, Any] = {}
    # Disk cache root (persistent across processes). Override with WOSAC_HARD_LOG_CACHE_DIR.
    _disk_cache_dir: str = os.environ.get("WOSAC_HARD_LOG_CACHE_DIR", "/tmp/wosac_hard_log_feat_cache")

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self._metric_key = f"{prefix}/sim_agents_2025/realism_meta_metric"
        self._config = self._load_config()
        self._metametric_sum: float = 0.0
        self._count: int = 0
        self._scenario_cache: Dict[str, Any] = {}
        self._per_metric_sums: Dict[str, float] = {n: 0.0 for n in _LIKELIHOOD_NAMES}
        # Persistent forkserver pool for parallel scenario load + log feature compute.
        # WOSAC_HARD_POOL_WORKERS env var (default = min(16, ncpu//2)). 0 disables pool.
        self._pool: Any = None
        _pw = int(os.environ.get("WOSAC_HARD_POOL_WORKERS", "-1"))
        self._pool_workers = (
            _pw if _pw >= 0 else max(1, min(16, (os.cpu_count() or 4) // 2))
        )

    def _get_pool(self):
        if self._pool_workers <= 0:
            return None
        if self._pool is None:
            try:
                ctx = mp.get_context("forkserver")
            except ValueError:
                ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(processes=self._pool_workers)
        return self._pool

    def __getstate__(self):
        # Pool 객체는 pickle 불가 — DDP/checkpoint 호환성 위해 제외.
        state = self.__dict__.copy()
        state["_pool"] = None
        return state

    def close_pool(self) -> None:
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            except Exception:
                pass
            self._pool = None

    @staticmethod
    def _load_config():
        from google.protobuf import text_format
        from waymo_open_dataset.protos import sim_agents_metrics_pb2
        import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as wm

        config_path = Path(wm.__file__).parent / "challenge_2025_sim_agents_config.textproto"
        with open(config_path) as f:
            cfg = sim_agents_metrics_pb2.SimAgentMetricsConfig()
            text_format.Parse(f.read(), cfg)
        return cfg

    def _load_scenario(self, scenario_file: str):
        cached = self._scenario_cache.get(scenario_file)
        if cached is not None:
            return cached
        import tensorflow as tf
        from waymo_open_dataset.protos import scenario_pb2

        tf.config.set_visible_devices([], "GPU")
        scenario = scenario_pb2.Scenario()
        for tfdata in tf.data.TFRecordDataset([scenario_file], compression_type=""):
            scenario.ParseFromString(bytes(tfdata.numpy()))
            break
        if len(self._scenario_cache) >= self._SCENARIO_CACHE_MAX:
            self._scenario_cache.pop(next(iter(self._scenario_cache)))
        self._scenario_cache[scenario_file] = scenario
        return scenario

    def _compute_one(
        self,
        scenario_file: str,
        agent_ids: np.ndarray,
        pred_traj_np: np.ndarray,
        pred_z_np: np.ndarray,
        pred_head_np: np.ndarray,
    ) -> float:
        from waymo_open_dataset.protos import sim_agents_submission_pb2
        from src.smart.metrics.wosac_metric_features_torch.metric_features_torch import (
            compute_scenario_rollouts_features,
        )
        from src.smart.metrics.wosac_metametric_pytorch import (
            compute_wosac_metametric_from_features_torch,
        )

        scenario = self._load_scenario(scenario_file)

        n_agents, n_rollout = pred_traj_np.shape[:2]
        joint_scenes = []
        for i_rollout in range(n_rollout):
            simulated_trajectories = []
            for i_agent in range(n_agents):
                simulated_trajectories.append(
                    sim_agents_submission_pb2.SimulatedTrajectory(
                        center_x=pred_traj_np[i_agent, i_rollout, :, 0],
                        center_y=pred_traj_np[i_agent, i_rollout, :, 1],
                        center_z=pred_z_np[i_agent, i_rollout],
                        heading=pred_head_np[i_agent, i_rollout],
                        object_id=int(agent_ids[i_agent]),
                    )
                )
            joint_scenes.append(
                sim_agents_submission_pb2.JointScene(simulated_trajectories=simulated_trajectories)
            )

        scenario_rollouts = sim_agents_submission_pb2.ScenarioRollouts(
            joint_scenes=joint_scenes,
            scenario_id=scenario.scenario_id,
        )

        log_feat, sim_feat = compute_scenario_rollouts_features(scenario, scenario_rollouts)
        result = compute_wosac_metametric_from_features_torch(
            self._config, log_feat.as_dict(), sim_feat.as_dict()
        )
        return float(result.metametric)

    def update_from_prediction_tensors(
        self,
        *,
        scenario_files: List[str],
        agent_id: torch.Tensor,
        agent_batch: torch.Tensor,
        pred_traj: torch.Tensor,
        pred_z: torch.Tensor,
        pred_head: torch.Tensor,
    ) -> None:
        from src.smart.metrics.wosac_metric_features_torch.metric_features_torch_differentiable import (
            PredictedSimTrajectories,
            compute_metric_features_batched_scenes,
        )
        from src.smart.metrics.wosac_metric_features_torch.metric_features_torch import (
            compute_metric_features,
            scenario_to_joint_scene,
        )
        from waymo_open_dataset.utils.sim_agents import submission_specs

        n_scenarios = len(scenario_files)
        if n_scenarios == 0:
            return

        sizes = [int(s) for s in _tg_degree(agent_batch, dtype=torch.long).tolist()]
        device = pred_traj.device
        G = pred_traj.shape[1]  # (n_agents, G, T, 2)
        T_pred = pred_traj.shape[2]
        _challenge = submission_specs.ChallengeType.SIM_AGENTS

        id_splits = agent_id.split(sizes)
        traj_splits = pred_traj.split(sizes)
        z_splits = pred_z.split(sizes)
        head_splits = pred_head.split(sizes)

        # Parallel: load scenario protobuf + log feature compute.
        # mp.Pool worker 가 TFRecord 파싱 + log feat (TF/CPU) 를 시나리오별 동시 처리.
        # cache 활용은 포기하지만 (worker 격리), val 데이터셋은 시나리오 unique 라 cache hit 율 낮아 net 이득.
        pool = self._get_pool()
        from waymo_open_dataset.protos import scenario_pb2 as _scenario_pb2

        if pool is not None and n_scenarios > 1:
            if bool(int(os.environ.get("WOSAC_PROFILE", "0"))):
                import time as _time
                _pool_t0 = _time.perf_counter()
            # Two-level cache: in-memory (process-local) + disk (persistent across processes).
            # Disk hits skip TF compute_metric_features entirely; Run 1 → Run 2 free path.
            mem_cache = HardSimAgentsMetrics._log_feat_cache
            disk_dir = HardSimAgentsMetrics._disk_cache_dir
            os.makedirs(disk_dir, exist_ok=True)

            def _disk_path(sf: str) -> str:
                # basename only — assumes scenario_id unique. Safer than path-hashed.
                bn = os.path.basename(sf).replace(os.sep, "_")
                return os.path.join(disk_dir, f"{bn}.pkl")

            miss_idx: list = []
            miss_files: list = []
            slot: list = [None] * n_scenarios
            n_mem_hits = 0
            n_disk_hits = 0
            for j, sf in enumerate(scenario_files):
                cached = mem_cache.get(sf)
                if cached is not None:
                    slot[j] = cached
                    n_mem_hits += 1
                    continue
                dp = _disk_path(sf)
                if os.path.exists(dp):
                    try:
                        with open(dp, "rb") as _f:
                            res = pickle.load(_f)
                        mem_cache[sf] = res
                        slot[j] = res
                        n_disk_hits += 1
                        continue
                    except Exception:
                        pass  # fallthrough to recompute
                miss_idx.append(j)
                miss_files.append(sf)
            if miss_files:
                miss_results = pool.starmap(
                    _hard_load_and_log_feat_worker,
                    [(sf, _challenge) for sf in miss_files],
                )
                for j, sf, res in zip(miss_idx, miss_files, miss_results):
                    mem_cache[sf] = res
                    slot[j] = res
                    # Atomic write: tmp file + rename to prevent partial reads.
                    dp = _disk_path(sf)
                    try:
                        tmp = dp + ".tmp"
                        with open(tmp, "wb") as _f:
                            pickle.dump(res, _f, protocol=pickle.HIGHEST_PROTOCOL)
                        os.replace(tmp, dp)
                    except Exception:
                        pass  # cache write failures are non-fatal
            results = slot
            if bool(int(os.environ.get("WOSAC_PROFILE", "0"))):
                print(
                    f"[hard-rmm-profile] pool_starmap={_time.perf_counter()-_pool_t0:.2f}s "
                    f"n_scenes={n_scenarios} mem_hits={n_mem_hits} disk_hits={n_disk_hits} "
                    f"misses={len(miss_files)}",
                    flush=True,
                )
            scenarios = []
            log_feat_dicts: List[dict] = []
            for sc_bytes, lf_dict in results:
                sc = _scenario_pb2.Scenario()
                sc.ParseFromString(sc_bytes)
                scenarios.append(sc)
                log_feat_dicts.append(lf_dict)
        else:
            # Fallback: 기존 serial 경로 (n_scenarios=1 또는 pool disabled).
            scenarios = [self._load_scenario(sf) for sf in scenario_files]
            log_feat_dicts = []
            for sc in scenarios:
                log_joint = scenario_to_joint_scene(sc, _challenge)
                lf = compute_metric_features(sc, log_joint, challenge_type=_challenge, use_log_validity=True)
                log_feat_dicts.append(lf.as_dict())

        # Sim features: for each rollout g, batch across all scenarios
        sim_feat_per_g: List[list] = []  # G lists, each of length n_scenarios
        _PROFILE = bool(int(os.environ.get("WOSAC_PROFILE", "0")))
        if _PROFILE:
            import time as _time
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            _t0 = _time.perf_counter()
            _per_g_times: list = []
        with torch.no_grad():
            for g in range(G):
                if _PROFILE:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    _g_start = _time.perf_counter()
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
                if _PROFILE:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    _per_g_times.append(_time.perf_counter() - _g_start)
        if _PROFILE:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            _t1 = _time.perf_counter()
            print(f"[hard-rmm-profile] G_loop_total={_t1-_t0:.2f}s mean_per_g={sum(_per_g_times)/len(_per_g_times):.3f}s n_scenes={n_scenarios} G={G}", flush=True)

        # Stack G rollouts per scenario → (G, E_i, T) then batched hard RMM
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

        from src.smart.metrics.wosac_metametric_pytorch import (
            compute_wosac_metametric_from_features_torch,
        )

        # Per-scenario meta-metric (Hard) and optional verification against official Real.
        hard_per_scenario: List[float] = []
        for i in range(n_scenarios):
            result = compute_wosac_metametric_from_features_torch(
                self._config, log_feat_dicts[i], sim_feat_dicts[i]
            )
            hm = float(result.metametric)
            hard_per_scenario.append(hm)
            self._metametric_sum += hm
            for name in _LIKELIHOOD_NAMES:
                self._per_metric_sums[name] += float(getattr(result, name))
            self._count += 1

        # Verify mode: per-sub-metric Hard vs Real comparison via subprocess pool.
        if os.environ.get("WOSAC_VERIFY") == "1":
            # Lazy persistent pool reuse from SimAgentsMetrics (cheaper than a new pool).
            if not hasattr(self, "_verify_pool"):
                try:
                    _ctx = mp.get_context("forkserver")
                except ValueError:
                    _ctx = mp.get_context("spawn")
                self._verify_pool = _ctx.Pool(
                    processes=int(os.environ.get("WOSAC_VERIFY_POOL", "4"))
                )
                self._verify_config_bytes = SimAgentsMetrics._load_config_bytes()

            # Build Real call args per scenario.
            real_args = []
            for i in range(n_scenarios):
                real_args.append((
                    self._verify_config_bytes,
                    scenario_files[i],
                    id_splits[i].cpu().numpy(),
                    pred_traj.split(sizes)[i].cpu().numpy(),
                    pred_z.split(sizes)[i].cpu().numpy(),
                    pred_head.split(sizes)[i].cpu().numpy(),
                ))
            from src.smart.metrics._verify_workers import real_full_metrics_worker as _vw
            real_results = self._verify_pool.starmap(_vw, real_args)

            _SHORT_NAMES = [
                "linear_speed", "linear_acceleration", "angular_speed", "angular_acceleration",
                "distance_to_nearest_object", "collision_indication", "time_to_collision",
                "distance_to_road_edge", "offroad_indication", "traffic_light_violation",
            ]
            for i in range(n_scenarios):
                hard_result = compute_wosac_metametric_from_features_torch(
                    self._config, log_feat_dicts[i], sim_feat_dicts[i]
                )
                rr = real_results[i]
                line_parts = [
                    f"[VERIFY i={i} sc={rr['scenario_id'][:10]}]",
                    f"meta H={hard_result.metametric:.4f} R={rr['metametric']:.4f} d={abs(hard_result.metametric-rr['metametric']):.4f}",
                ]
                for short in _SHORT_NAMES:
                    h = float(getattr(hard_result, short + "_likelihood"))
                    r = rr[short]
                    d = abs(h - r)
                    mark = "**" if d > 0.05 else ("*" if d > 0.01 else "")
                    line_parts.append(f"{short[:8]}:H={h:.3f}/R={r:.3f}{mark}")
                print(" | ".join(line_parts), flush=True)

    def _drain_completed_futures(self, wait: bool = True, drain_all: bool = True) -> None:
        return None

    def get_state_tensor(self, device: torch.device) -> torch.Tensor:
        """DDP all_reduce 용으로 [metametric_sum, count, *per_metric_sums] tensor 반환."""
        vals = [self._metametric_sum, float(self._count)]
        for name in _LIKELIHOOD_NAMES:
            vals.append(self._per_metric_sums[name])
        return torch.tensor(vals, device=device, dtype=torch.float64)

    def compute_from_state_tensor(self, reduced_metric_state: torch.Tensor) -> Dict[str, Any]:
        """all_reduce 이후 tensor로부터 metametric + per-metric 평균 계산."""
        total_count = reduced_metric_state[1].clamp_min(1.0)
        result = {
            self._metric_key: (reduced_metric_state[0] / total_count).to(torch.float32)
        }
        for j, name in enumerate(_LIKELIHOOD_NAMES):
            key = f"{self.prefix}/sim_agents_2025/{name}"
            result[key] = (reduced_metric_state[2 + j] / total_count).to(torch.float32)
        return result

    def compute(self) -> Dict[str, Any]:
        count = max(self._count, 1)
        result = {
            self._metric_key: torch.tensor(self._metametric_sum / count, dtype=torch.float32)
        }
        for name in _LIKELIHOOD_NAMES:
            key = f"{self.prefix}/sim_agents_2025/{name}"
            result[key] = torch.tensor(self._per_metric_sums[name] / count, dtype=torch.float32)
        return result

    def reset(self) -> None:
        self._metametric_sum = 0.0
        self._count = 0
        self._per_metric_sums = {n: 0.0 for n in _LIKELIHOOD_NAMES}


