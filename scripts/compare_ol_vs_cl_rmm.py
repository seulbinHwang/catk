#!/usr/bin/env python3
"""
Open-loop vs Closed-loop RMM 비교 (T=20 / 2초 horizon).

목적
====
``train_flow_ref_nll`` 의 가정을 검증한다:
  "OL 한 번 (2초 single-shot ODE) 으로 뽑은 trajectory 의 RMM" 이
  "CL (0.5초 commit × 4 = 2초 AR rollout) 으로 뽑은 trajectory 의 RMM" 보다
  실제로 우수한가?

방법
====
같은 pretrained ckpt 에서 시작해서, 같은 scenario·같은 noise 로 OL/CL 을 만들고
같은 torch RMM (HardSimAgentsMetrics 와 동일한 _features_torch_differentiable +
wosac_metametric_pytorch 경로) 로 점수를 계산.

  - OL: ``rollout_from_cache(max_steps=1, return_per_step_x1=True)`` 의 첫 coarse step
        ODE 출력 ``y_hat_norm[:, :20, :]`` 를 통째로 world frame 으로 풀어 사용 (CL 의
        ContinuousCommitBridge.commit 이 ``[:, :5]`` 만 푸는 부분을 ``[:, :20]`` 으로 확장).
  - CL: ``_run_closed_loop_rollouts`` 의 80-step 출력에서 앞 20 step 슬라이스
        (= 4 coarse step × 5 fine = 정확히 2초; coarse 5..15 는 사용 안 함).
  - 두 경로 모두 같은 ``scenario_sampling_seeds`` → 첫 coarse step 노이즈 동일.
  - Log features 는 한 번 계산해 ``_slice_log_feat_dict_to_pred_horizon(..., 20)`` 으로
    잘라 OL/CL 모두에 재사용.

진행 상황은 ``logs/<task>/compare_ol_vs_cl_rmm.log`` 에 라인 단위로 기록됩니다.
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import argparse
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

# ── GPU/TF 환경 ──────────────────────────────────────────────────────────────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Hydra config 흉내 (SMARTFlow.__init__ 가 출력 디렉터리를 요구) ─────────
import hydra.core.hydra_config  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

_OUTPUT_DIR_DEFAULT = ROOT / "logs" / "compare_ol_vs_cl_rmm"
_OUTPUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)


def _make_fake_hydra(output_dir: Path):
    class _FakeHydraConfig:
        @staticmethod
        def get():
            return OmegaConf.create({"runtime": {"output_dir": str(output_dir)}})

    hydra.core.hydra_config.HydraConfig = _FakeHydraConfig


# 위 fake 가 SMARTFlow import 시점의 hydra 호출을 가로채야 함.
_make_fake_hydra(_OUTPUT_DIR_DEFAULT)

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from torch_geometric.utils import degree as tg_degree  # noqa: E402

import tensorflow as tf  # noqa: E402

tf.config.set_visible_devices([], "GPU")

from waymo_open_dataset.protos import scenario_pb2, sim_agents_metrics_pb2  # noqa: E402
import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as wm_metrics  # noqa: E402
from waymo_open_dataset.utils.sim_agents import submission_specs  # noqa: E402
from google.protobuf import text_format  # noqa: E402

from src.smart.model.smart_flow import (  # noqa: E402
    SMARTFlow,
    _slice_log_feat_dict_to_pred_horizon,
)
from src.smart.datasets import MultiDataset  # noqa: E402
from src.smart.datamodules.target_builder import WaymoTargetBuilderVal  # noqa: E402
from src.smart.utils.geometry import wrap_angle  # noqa: E402
from src.smart.utils.rollout import transform_to_global  # noqa: E402
from src.smart.metrics.wosac_metric_features_torch.metric_features_torch import (  # noqa: E402
    compute_metric_features,
    scenario_to_joint_scene,
)
from src.smart.metrics.wosac_metric_features_torch.metric_features_torch_differentiable import (  # noqa: E402
    PredictedSimTrajectories,
    compute_metric_features_batched_scenes,
)
from src.smart.metrics.wosac_metametric_pytorch import (  # noqa: E402
    compute_wosac_metametric_from_features_torch,
)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
KST = timezone(timedelta(hours=9))


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ol_vs_cl_rmm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(str(log_path), mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False

    def _ts_kst() -> str:
        return datetime.now(tz=KST).strftime("%Y-%m-%d %H:%M:%S KST")

    logger.info(f"Log file: {log_path}")
    logger.info(f"Started at: {_ts_kst()}")
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Model load
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: Path, device: torch.device, logger: logging.Logger) -> SMARTFlow:
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    mc = ckpt["hyper_parameters"]["model_config"]
    if "eval_sampling_noise" not in mc:
        old = mc.get("validation_rollout_sampling", {"noise_scale": 1.0})
        mc["eval_sampling_noise"] = {"noise_scale": float(old.get("noise_scale", 1.0))}
        logger.info("  patched: added eval_sampling_noise=1.0 to hparams")

    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        tmp = f.name
    try:
        torch.save(ckpt, tmp)
        model = SMARTFlow.load_from_checkpoint(tmp, map_location="cpu", strict=False)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
    model.eval()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {n_params:,} params, device={device}")
    return model


def load_waymo_metric_config() -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
    config_path = (
        Path(wm_metrics.__file__).parent / "challenge_2025_sim_agents_config.textproto"
    )
    cfg = sim_agents_metrics_pb2.SimAgentMetricsConfig()
    with open(config_path) as f:
        text_format.Parse(f.read(), cfg)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# OL trajectory extraction (한 번의 ODE call → 2초 world frame)
# ─────────────────────────────────────────────────────────────────────────────
def _ol_commit_full(
    y_hat_norm: Tensor,        # [n_active, 20, 4]
    current_pos: Tensor,       # [n_active, 2]
    current_head: Tensor,      # [n_active]
) -> Tuple[Tensor, Tensor]:
    """ContinuousCommitBridge.commit 의 ``[:, :5]`` → ``[:, :20]`` 확장 버전."""
    chunk = y_hat_norm.clone()
    chunk[..., :2] = chunk[..., :2] * 20.0  # denormalize xy
    cos_sin = F.normalize(chunk[..., 2:4], dim=-1)
    delta_head = torch.atan2(cos_sin[..., 1], cos_sin[..., 0])  # [n_active, 20]
    pos_world, _ = transform_to_global(
        pos_local=chunk[..., :2],
        head_local=None,
        pos_now=current_pos,
        head_now=current_head,
    )  # [n_active, 20, 2]
    head_world = wrap_angle(current_head.unsqueeze(1) + delta_head)  # [n_active, 20]
    return pos_world, head_world


@torch.no_grad()
def run_open_loop_world_2s(
    model: SMARTFlow,
    tokenized_agent: Dict[str, Tensor],
    map_feature: Dict[str, Tensor],
    rollout_cache: Dict[str, object],
    scenario_sampling_seeds: Tensor,
) -> Tuple[Tensor, Tensor]:
    """단일 rollout(=하나의 noise) 의 OL 2초 trajectory.

    Returns:
        pred_traj_world: [n_agent, 20, 2]
        pred_head_world: [n_agent, 20]
    inactive 한 agent (flow ODE 가 호출되지 않은) 는 history 마지막 pose 를
    20-step 동안 정지시킨 채로 채워넣고, valid 마스크는 별도로 반환하지 않습니다
    (downstream 에서 valid 는 일괄 True 로 사용).
    """
    enc = model.encoder.agent_encoder

    pred = enc.rollout_from_cache(
        rollout_cache=rollout_cache,
        tokenized_agent=tokenized_agent,
        map_feature=map_feature,
        sampling_noise=model.eval_sampling_noise,
        scenario_sampling_seeds=scenario_sampling_seeds,
        max_steps=1,
        return_per_step_x1=True,
    )
    per_step_x1: List[Tensor] = pred["per_step_x1"]
    per_step_active: List[Tensor] = pred["per_step_active_mask"]

    n_agent = int(tokenized_agent["batch"].shape[0])
    device = tokenized_agent["batch"].device

    if len(per_step_x1) == 0 or per_step_x1[0] is None:
        # 매우 드문 경우 — active agent 가 0 인 batch.
        last_pos = rollout_cache["pos_window"][:, -1].to(device)  # [n_agent, 2]
        last_head = rollout_cache["head_window"][:, -1].to(device)  # [n_agent]
        pos_world = last_pos.unsqueeze(1).expand(n_agent, 20, 2).contiguous()
        head_world = last_head.unsqueeze(1).expand(n_agent, 20).contiguous()
        return pos_world, head_world

    y_hat_norm_active = per_step_x1[0]  # [n_active, 20, 4]
    active_mask: Tensor = per_step_active[0]  # [n_agent] bool

    last_pos_full = rollout_cache["pos_window"][:, -1].to(device)   # [n_agent, 2]
    last_head_full = rollout_cache["head_window"][:, -1].to(device)  # [n_agent]
    cur_pos_active = last_pos_full[active_mask]
    cur_head_active = last_head_full[active_mask]

    pos_world_active, head_world_active = _ol_commit_full(
        y_hat_norm_active, cur_pos_active, cur_head_active
    )  # [n_active, 20, 2/]

    # Scatter back to full agent axis.
    pos_world = last_pos_full.unsqueeze(1).expand(n_agent, 20, 2).contiguous().clone()
    head_world = last_head_full.unsqueeze(1).expand(n_agent, 20).contiguous().clone()
    pos_world[active_mask] = pos_world_active
    head_world[active_mask] = head_world_active
    return pos_world, head_world


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default=str(ROOT / "logs/pretrained/epoch_last.ckpt"),
        help="pretrained checkpoint (train_flow_ref_nll 의 CKPT_PATH 와 동일)",
    )
    p.add_argument(
        "--val_dir",
        default="/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1/validation",
    )
    p.add_argument(
        "--tfrecord_dir",
        default="/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1/validation_tfrecords_splitted",
    )
    p.add_argument("--n_scenarios", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_rollouts", type=int, default=8, help="시나리오당 rollout 수 G")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--task_name", default=None)
    return p.parse_args()


def _now_kst() -> str:
    return datetime.now(tz=KST).strftime("%Y-%m-%d %H:%M:%S KST")


def main():
    args = parse_args()
    device = torch.device(args.device)

    task_name = args.task_name or datetime.now(tz=KST).strftime("run_%Y%m%d_%H%M%S")
    out_dir = _OUTPUT_DIR_DEFAULT / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _make_fake_hydra(out_dir)

    log_path = out_dir / "compare_ol_vs_cl_rmm.log"
    summary_jsonl = out_dir / "per_scenario.jsonl"
    logger = _setup_logging(log_path)

    logger.info("=" * 78)
    logger.info("Open-loop vs Closed-loop RMM @ T=20 (2s)")
    logger.info("=" * 78)
    logger.info(f"args: {vars(args)}")
    logger.info(f"output dir: {out_dir}")

    # ── 1. Model ────────────────────────────────────────────────────────────
    model = load_model(Path(args.ckpt), device, logger)
    cfg = load_waymo_metric_config()
    challenge = submission_specs.ChallengeType.SIM_AGENTS

    # ── 2. Data ─────────────────────────────────────────────────────────────
    logger.info(f"Loading validation data: {args.val_dir}")
    dataset = MultiDataset(args.val_dir, WaymoTargetBuilderVal(), args.tfrecord_dir)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # ── 3. Iterate ──────────────────────────────────────────────────────────
    G = int(args.n_rollouts)
    T_pred = 20  # 2초 = 20 fine steps

    n_done = 0
    ol_rmms: List[float] = []
    cl_rmms: List[float] = []
    ol_lik_sums: Dict[str, float] = {}
    cl_lik_sums: Dict[str, float] = {}
    n_lik_count = 0
    t_start = time.time()

    likelihood_keys = [
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
    for k in likelihood_keys:
        ol_lik_sums[k] = 0.0
        cl_lik_sums[k] = 0.0

    summary_fp = open(summary_jsonl, "w")

    try:
        for i_batch, batch in enumerate(loader):
            if n_done >= args.n_scenarios:
                break
            batch = batch.to(device)
            scenario_ids: List[str] = list(batch["scenario_id"])
            tfrecord_paths: List[str] = list(batch["tfrecord_path"])
            n_sc = len(scenario_ids)

            logger.info(
                f"[batch {i_batch}] scenarios={n_sc}  done_so_far={n_done}/"
                f"{args.n_scenarios}  elapsed={time.time()-t_start:.1f}s"
            )

            # 토큰 + 맵 인코딩 + cache 준비 (한 번)
            tokenized_map, tokenized_agent = model.token_processor(batch)
            map_feature = model.encoder.encode_map(tokenized_map)
            rollout_cache = model.encoder.agent_encoder.prepare_inference_cache(
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
            )

            # ─── CL: 80-step rollout, 앞 20 step 만 사용 ─────────────────────
            orig_n = model.n_rollout_closed_val
            model.n_rollout_closed_val = G
            t_cl0 = time.time()
            try:
                cl_traj, cl_z, cl_head = model._run_closed_loop_rollouts(
                    data=batch,
                    tokenized_agent=tokenized_agent,
                    map_feature=map_feature,
                )
            finally:
                model.n_rollout_closed_val = orig_n
            t_cl = time.time() - t_cl0
            # cl_traj: [A, G, 80, 2] / cl_z: [A, G, 80] / cl_head: [A, G, 80]
            cl_traj_2s = cl_traj[..., :T_pred, :].contiguous()
            cl_z_2s = cl_z[..., :T_pred].contiguous()
            cl_head_2s = cl_head[..., :T_pred].contiguous()

            # ─── OL: G개의 1-step rollout (같은 seed table) ─────────────────
            seed_table = model._build_closed_loop_seed_table(
                scenario_ids=batch["scenario_id"],
                rollout_indices=list(range(G)),
                device=device,
            )  # [G, n_scenario]

            ol_traj_list: List[Tensor] = []
            ol_head_list: List[Tensor] = []
            t_ol0 = time.time()
            for g in range(G):
                seeds_g = seed_table[g]
                pos_world_g, head_world_g = run_open_loop_world_2s(
                    model=model,
                    tokenized_agent=tokenized_agent,
                    map_feature=map_feature,
                    rollout_cache=rollout_cache,
                    scenario_sampling_seeds=seeds_g,
                )
                ol_traj_list.append(pos_world_g)
                ol_head_list.append(head_world_g)
            t_ol = time.time() - t_ol0

            # [A, G, 20, 2] / [A, G, 20]
            ol_traj_2s = torch.stack(ol_traj_list, dim=1)
            ol_head_2s = torch.stack(ol_head_list, dim=1)
            ol_z_2s = cl_z_2s  # z 는 history last z 를 expand → OL/CL 동일

            # Sanity: OL 과 CL 의 첫 5 fine step (0.5초) 은 같은 ODE call 의 commit 이라
            # 부동소수 오차 수준에서 일치해야 함 (transform_to_global 정합 검증).
            with torch.no_grad():
                first_chunk_diff = (
                    ol_traj_2s[..., :5, :] - cl_traj_2s[..., :5, :]
                ).abs().max().item()
                first_head_diff = (
                    wrap_angle(ol_head_2s[..., :5] - cl_head_2s[..., :5])
                ).abs().max().item()
            logger.info(
                f"  sanity[batch {i_batch}] first-0.5s OL≈CL  "
                f"max|Δxy|={first_chunk_diff:.4e}m  max|Δhead|={first_head_diff:.4e}rad"
            )

            # ─── 시나리오별 unbatch ─────────────────────────────────────────
            sizes = [int(s) for s in tg_degree(batch["agent"]["batch"], dtype=torch.long).tolist()]
            agent_id = batch["agent"]["id"]
            id_splits = agent_id.split(sizes)

            ol_traj_splits = ol_traj_2s.split(sizes, dim=0)
            ol_head_splits = ol_head_2s.split(sizes, dim=0)
            cl_traj_splits = cl_traj_2s.split(sizes, dim=0)
            cl_head_splits = cl_head_2s.split(sizes, dim=0)
            z_splits = ol_z_2s.split(sizes, dim=0)

            # ─── per-scenario log feature & RMM ─────────────────────────────
            for i_sc in range(n_sc):
                if n_done >= args.n_scenarios:
                    break
                scen_id = scenario_ids[i_sc]
                tfr = tfrecord_paths[i_sc]

                # Scenario proto
                scenario = scenario_pb2.Scenario()
                for tfdata in tf.data.TFRecordDataset([tfr], compression_type=""):
                    scenario.ParseFromString(bytes(tfdata.numpy()))
                    break

                # Log features (compute once on full T, slice to T_pred)
                log_joint = scenario_to_joint_scene(scenario, challenge)
                lf = compute_metric_features(
                    scenario, log_joint, challenge_type=challenge, use_log_validity=True
                )
                log_feat_full = lf.as_dict()
                log_feat_dict = _slice_log_feat_dict_to_pred_horizon(log_feat_full, T_pred)
                # Move log to device
                log_feat_dict = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in log_feat_dict.items()
                }

                ids_i = id_splits[i_sc]
                z_i = z_splits[i_sc]

                # Build PredictedSimTrajectories per rollout (G개) — batched_scenes 가
                # scenario 1개 + G rollout 을 한 번에 처리하지 못하므로 rollout 별로 호출.
                # (compute_metric_features_batched_scenes 는 (n_scenes, ...) 이고
                #  scene 당 single rollout 가정. 우리는 동일 scenario 를 G번 넣어 batched.)
                # ▶ OL
                ol_sim_feat_list: List[dict] = []
                cl_sim_feat_list: List[dict] = []
                with torch.no_grad():
                    for g in range(G):
                        ol_pred = PredictedSimTrajectories(
                            object_id=ids_i.cpu(),
                            center_x=ol_traj_splits[i_sc][:, g, :, 0],
                            center_y=ol_traj_splits[i_sc][:, g, :, 1],
                            center_z=z_i[:, g, :],
                            heading=ol_head_splits[i_sc][:, g, :],
                            valid=torch.ones(
                                ids_i.shape[0], T_pred, dtype=torch.bool, device=device
                            ),
                        )
                        ol_feat_list = compute_metric_features_batched_scenes(
                            scenarios=[scenario], preds=[ol_pred], surrogate=None,
                        )
                        ol_sim_feat_list.append(ol_feat_list[0].as_dict())

                        cl_pred = PredictedSimTrajectories(
                            object_id=ids_i.cpu(),
                            center_x=cl_traj_splits[i_sc][:, g, :, 0],
                            center_y=cl_traj_splits[i_sc][:, g, :, 1],
                            center_z=z_i[:, g, :],
                            heading=cl_head_splits[i_sc][:, g, :],
                            valid=torch.ones(
                                ids_i.shape[0], T_pred, dtype=torch.bool, device=device
                            ),
                        )
                        cl_feat_list = compute_metric_features_batched_scenes(
                            scenarios=[scenario], preds=[cl_pred], surrogate=None,
                        )
                        cl_sim_feat_list.append(cl_feat_list[0].as_dict())

                # G rollout 합쳐 (G, n_obj, T) 형태로 만들어 한 번에 RMM 계산.
                def _stack(field, lst):
                    return torch.cat([d[field] for d in lst], dim=0)

                ol_sim_dict = {
                    "object_id": ol_sim_feat_list[0]["object_id"],
                    "object_type": _stack("object_type", ol_sim_feat_list),
                    "valid": _stack("valid", ol_sim_feat_list),
                    "average_displacement_error": _stack(
                        "average_displacement_error", ol_sim_feat_list
                    ),
                    "linear_speed": _stack("linear_speed", ol_sim_feat_list),
                    "linear_acceleration": _stack("linear_acceleration", ol_sim_feat_list),
                    "angular_speed": _stack("angular_speed", ol_sim_feat_list),
                    "angular_acceleration": _stack("angular_acceleration", ol_sim_feat_list),
                    "distance_to_nearest_object": _stack(
                        "distance_to_nearest_object", ol_sim_feat_list
                    ),
                    "collision_per_step": _stack("collision_per_step", ol_sim_feat_list),
                    "time_to_collision": _stack("time_to_collision", ol_sim_feat_list),
                    "distance_to_road_edge": _stack("distance_to_road_edge", ol_sim_feat_list),
                    "offroad_per_step": _stack("offroad_per_step", ol_sim_feat_list),
                    "traffic_light_violation_per_step": _stack(
                        "traffic_light_violation_per_step", ol_sim_feat_list
                    ),
                }
                cl_sim_dict = {
                    "object_id": cl_sim_feat_list[0]["object_id"],
                    "object_type": _stack("object_type", cl_sim_feat_list),
                    "valid": _stack("valid", cl_sim_feat_list),
                    "average_displacement_error": _stack(
                        "average_displacement_error", cl_sim_feat_list
                    ),
                    "linear_speed": _stack("linear_speed", cl_sim_feat_list),
                    "linear_acceleration": _stack("linear_acceleration", cl_sim_feat_list),
                    "angular_speed": _stack("angular_speed", cl_sim_feat_list),
                    "angular_acceleration": _stack("angular_acceleration", cl_sim_feat_list),
                    "distance_to_nearest_object": _stack(
                        "distance_to_nearest_object", cl_sim_feat_list
                    ),
                    "collision_per_step": _stack("collision_per_step", cl_sim_feat_list),
                    "time_to_collision": _stack("time_to_collision", cl_sim_feat_list),
                    "distance_to_road_edge": _stack("distance_to_road_edge", cl_sim_feat_list),
                    "offroad_per_step": _stack("offroad_per_step", cl_sim_feat_list),
                    "traffic_light_violation_per_step": _stack(
                        "traffic_light_violation_per_step", cl_sim_feat_list
                    ),
                }

                ol_result = compute_wosac_metametric_from_features_torch(
                    cfg, log_feat_dict, ol_sim_dict
                )
                cl_result = compute_wosac_metametric_from_features_torch(
                    cfg, log_feat_dict, cl_sim_dict
                )

                ol_meta = float(ol_result.metametric)
                cl_meta = float(cl_result.metametric)
                delta = ol_meta - cl_meta

                ol_rmms.append(ol_meta)
                cl_rmms.append(cl_meta)
                for k in likelihood_keys:
                    ol_lik_sums[k] += float(getattr(ol_result, k))
                    cl_lik_sums[k] += float(getattr(cl_result, k))
                n_lik_count += 1
                n_done += 1

                row = {
                    "i": n_done,
                    "scenario_id": scen_id,
                    "ol_rmm": ol_meta,
                    "cl_rmm": cl_meta,
                    "delta_ol_minus_cl": delta,
                    "G": G,
                    "T_pred": T_pred,
                    "n_eval_agents": int(ids_i.shape[0]),
                    "ol_likelihoods": {
                        k: float(getattr(ol_result, k)) for k in likelihood_keys
                    },
                    "cl_likelihoods": {
                        k: float(getattr(cl_result, k)) for k in likelihood_keys
                    },
                }
                summary_fp.write(json.dumps(row) + "\n")
                summary_fp.flush()

                running_ol = sum(ol_rmms) / len(ol_rmms)
                running_cl = sum(cl_rmms) / len(cl_rmms)
                running_delta = running_ol - running_cl
                logger.info(
                    f"  [sc {n_done:>3}/{args.n_scenarios}] {scen_id[:14]} "
                    f"OL={ol_meta:.4f}  CL={cl_meta:.4f}  Δ={delta:+.4f}  "
                    f"║ avg OL={running_ol:.4f} CL={running_cl:.4f} Δ={running_delta:+.4f}  "
                    f"(t_cl={t_cl:.1f}s t_ol={t_ol:.1f}s)"
                )

            # 메모리 정리
            del cl_traj, cl_z, cl_head, ol_traj_2s, ol_head_2s
            torch.cuda.empty_cache()
    finally:
        summary_fp.close()

    # ── 4. Final summary ────────────────────────────────────────────────────
    logger.info("=" * 78)
    logger.info(f"DONE. processed n={n_done} scenarios, G={G} rollouts each, T={T_pred}")
    if n_done == 0:
        logger.info("no scenarios processed.")
        return
    avg_ol = sum(ol_rmms) / n_done
    avg_cl = sum(cl_rmms) / n_done
    logger.info(f"  AVG OL RMM = {avg_ol:.6f}")
    logger.info(f"  AVG CL RMM = {avg_cl:.6f}")
    logger.info(f"  Δ (OL - CL) = {avg_ol - avg_cl:+.6f}")
    logger.info(f"  win_count: OL > CL = {sum(1 for o, c in zip(ol_rmms, cl_rmms) if o > c)}/"
                f"{n_done}")
    logger.info("Per-likelihood (avg):")
    logger.info(f"  {'metric':<40}  {'OL':>8}  {'CL':>8}  {'delta':>10}  {'w*delta':>10}")
    weights = {
        "linear_speed_likelihood": 0.05,
        "linear_acceleration_likelihood": 0.05,
        "angular_speed_likelihood": 0.05,
        "angular_acceleration_likelihood": 0.05,
        "distance_to_nearest_object_likelihood": 0.10,
        "collision_indication_likelihood": 0.25,
        "time_to_collision_likelihood": 0.10,
        "distance_to_road_edge_likelihood": 0.05,
        "offroad_indication_likelihood": 0.25,
        "traffic_light_violation_likelihood": 0.05,
    }
    total_wd = 0.0
    for k in likelihood_keys:
        ol_v = ol_lik_sums[k] / n_lik_count
        cl_v = cl_lik_sums[k] / n_lik_count
        d = ol_v - cl_v
        wd = float(weights.get(k, 0.0)) * d
        total_wd += wd
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "=")
        logger.info(f"  {k:<40}  {ol_v:>8.4f}  {cl_v:>8.4f}  {d:>+10.5f}  {wd:>+10.5f} {arrow}")
    logger.info(f"  {'weighted-sum (≈ Δ RMM)':<40}  {'':>8}  {'':>8}  {'':>10}  {total_wd:>+10.5f}")

    # Stats: SE / t / 95% CI on Δ across scenarios
    deltas = [o - c for o, c in zip(ol_rmms, cl_rmms)]
    n_d = len(deltas)
    if n_d >= 2:
        mu = sum(deltas) / n_d
        var = sum((d - mu) ** 2 for d in deltas) / (n_d - 1)
        sd = var ** 0.5
        se = sd / (n_d ** 0.5)
        t = mu / se if se > 0 else float("nan")
        logger.info(f"Δ stats (n={n_d}): mean={mu:+.5f}  sd={sd:.5f}  SE={se:.5f}  t={t:+.3f}  "
                    f"95%CI=[{mu-1.96*se:+.5f}, {mu+1.96*se:+.5f}]")
    logger.info(f"per-scenario JSONL: {summary_jsonl}")
    logger.info(f"Finished at: {_now_kst()}")


if __name__ == "__main__":
    main()
