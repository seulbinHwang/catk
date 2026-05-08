#!/usr/bin/env python3
# Open-Loop vs Closed-Loop Hard-RMM 시나리오별 비교.
#
# 의도:
#   - 같은 모델, 같은 batch 에서 Open-Loop (Flow ODE 2 초 단발 ODE) 와
#     Closed-Loop (4 × 0.5 초 autoregressive coarse step = 2 초) 두 방식으로
#     2 초 미래를 만들고, 나머지 6 초는 GT 로 padding 한 8 초 trajectory
#     (WOSAC 표준 80-step) 에 대해 Hard-RMM 을 시나리오별로 분리 측정한다.
#   - 같은 noise 를 OL 과 CL 이 공유하도록 ``_build_rollout_noise_tape`` /
#     ``noise_tape_override`` 로 paired evaluation 을 보장한다.
#   - OL/CL 각각의 metametric 과 10 개 likelihood, 그리고 delta = CL − OL 을
#     wandb 에 step-by-step 으로 로깅하고, scenario-level table 에 기록한다.
#
# 설계 정합성:
#   - SMART-Flow 모델의 OL eval 경로 (``sample_open_loop_future``) 와
#     CL eval 경로 (``rollout_from_cache(max_steps=4)``) 를 그대로 호출.
#   - Per-scenario hard-RMM 추출은 ``temp/ol_vs_cl_rmm/per_scenario_hard_rmm.py``
#     의 ``compute_hard_rmm_per_scenario`` (HardSimAgentsMetrics 의 inner loop
#     를 시나리오별로 분기시킨 thin wrapper) 사용.
#   - 모델/datamodule 인스턴스화는 ``configs/run.yaml`` 을 그대로 따르고,
#     ``configs/experiment/local_val_flow.yaml`` 와 같은 validate 셋팅을 base
#     로 한다.  단, 우리는 직접 val_loader 를 돌려 OL/CL 평가를 수행하므로
#     ``trainer.validate`` 는 호출하지 않는다 — model.validation_step 의 표준
#     OL/CL 메트릭과 충돌하지 않도록 ``val_open_loop=False, val_closed_loop=False``
#     로 강제한다.
#
# Run example (KST 2026-05-07 기준 launcher 통해 띄움):
#   bash temp/ol_vs_cl_rmm/launch_compare.sh
#
# 환경 변수 (launcher 가 export):
#   OLCL_LIMIT_VAL_BATCHES   (default 0.01)
#   OLCL_G_ROLLOUTS          (default 16)
#   OLCL_PRED_2S_COARSE      (default 4 = 2 초)
#   OLCL_PAD_MODE            (gt | last | zero, default gt)
#   OLCL_TFRECORD_DIR        validation tfrecord dir (override)
#   OLCL_WANDB_PROJECT       (default project_3-ol-vs-cl-rmm)
#   OLCL_WANDB_RUN_NAME      (default ocsc-ol-vs-cl-<timestamp_kst>)
#   OLCL_CKPT_PATH           (default fix-hard-rmm 호환 ckpt path)
#
# 결과:
#   - wandb panel:
#       ol/realism_meta_metric, cl/realism_meta_metric, delta/realism_meta_metric
#       ol/<likelihood>, cl/<likelihood>, delta/<likelihood> (10 개)
#       running 평균 (running/ol_*, running/cl_*, running/delta_*)
#       per-scenario wandb.Table (scenario_id, ol_meta, cl_meta, delta, …)
#   - stdout: KST timestamp prefix 와 함께 batch 별 진행/요약 print
#   - artifacts/: NPZ 와 CSV 로 per-scenario 기록 저장 (반복 분석용)

from __future__ import annotations

import csv
import datetime as _dt
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import hydra
import lightning as L
import numpy as np
import torch
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig, OmegaConf, open_dict

# 본 스크립트 디렉토리 (project_3) 를 sys.path 에 넣어 src/ 를 import 가능하게.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.smart.metrics import HardSimAgentsMetrics, SimAgentsMetrics, _LIKELIHOOD_NAMES  # noqa: E402
from src.smart.utils.rollout import transform_to_global  # noqa: E402

# 로컬 헬퍼
sys.path.insert(0, str(Path(__file__).resolve().parent))
from per_scenario_hard_rmm import compute_hard_rmm_per_scenario  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Logging / KST timestamp helpers
# ──────────────────────────────────────────────────────────────────────────────

_KST = ZoneInfo("Asia/Seoul")


def _kst_now() -> str:
    return _dt.datetime.now(_KST).strftime("%Y-%m-%d %H:%M:%S KST")


def _kst_compact() -> str:
    return _dt.datetime.now(_KST).strftime("%Y%m%d-%H%M%S")


class _KstFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return _dt.datetime.fromtimestamp(record.created, tz=_KST).strftime(
            "%Y-%m-%d %H:%M:%S"
        )


def _setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = _KstFormatter(
        fmt="[%(asctime)s KST] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("ol_vs_cl_rmm")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_dir / f"compare_ol_vs_cl_{_kst_compact()}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# OL → world frame helper (정규화 4-channel 출력을 world XY/heading 으로 복귀)
# ──────────────────────────────────────────────────────────────────────────────


def ol_norm_to_world(
    ol_norm: torch.Tensor,       # [n_active, 20, 4] = [x/20, y/20, cos_h, sin_h]
    current_pos: torch.Tensor,    # [n_active, 2]
    current_head: torch.Tensor,   # [n_active]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """OL normalized prediction 을 world frame 으로 변환합니다.

    OL 은 anchor frame (현재 위치/heading 기준) 에서 ``[x/20, y/20, cos, sin]``
    로 정규화된 2 초 미래를 출력하므로,

      1) 위치: ``pos_local = ol[:, :, :2] * 20`` 후 ``transform_to_global``.
      2) heading: ``head_local = atan2(sin, cos)`` 후 ``current_head`` 더해 world 로.

    Args:
        ol_norm: ``[n_active, 20, 4]`` (active mask 적용된 OL 출력).
        current_pos: ``[n_active, 2]`` anchor 시점 world 위치 (rollout cache
            ``pos_window[:, -1]`` 의 active 샘플).
        current_head: ``[n_active]`` anchor 시점 world heading.

    Returns:
        (pos_global, head_global): 각각 ``[n_active, 20, 2]`` / ``[n_active, 20]``.
    """
    pos_local = ol_norm[..., :2] * 20.0  # [n_active, 20, 2]
    head_local = torch.atan2(ol_norm[..., 3], ol_norm[..., 2])  # [n_active, 20]
    pos_global, head_global = transform_to_global(
        pos_local=pos_local,
        head_local=head_local,
        pos_now=current_pos,
        head_now=current_head,
    )
    return pos_global, head_global


# ──────────────────────────────────────────────────────────────────────────────
# Padding to 80-step (WOSAC 8 초 표준) — OL 과 CL 둘 다 동일 GT 로 채워서
# RMM 차이가 오직 0:T_pred (앞 2 초) 에서만 발생하도록 한다.
# ──────────────────────────────────────────────────────────────────────────────


def pad_pred_to_80(
    pred_traj_short: torch.Tensor,   # [n_agents, G, T_pred, 2]
    pred_z_short: torch.Tensor,      # [n_agents, G, T_pred]
    pred_head_short: torch.Tensor,   # [n_agents, G, T_pred]
    gt_pos_future: torch.Tensor,     # [n_agents, 80, 2]   (data["agent"]["position"][:, hist:hist+80, :2])
    gt_z_future: torch.Tensor,       # [n_agents, 80]      (data["agent"]["position"][:, hist:hist+80, 2])
    gt_head_future: torch.Tensor,    # [n_agents, 80]      (data["agent"]["heading"][:, hist:hist+80])
    pad_mode: str = "gt",
    valid_active_2hz: Optional[torch.Tensor] = None,  # [n_agents] active mask
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """앞 ``T_pred`` step 은 model 예측, 뒤는 ``pad_mode`` 에 따라 채운다.

    NOTE: ``pad_mode="none"`` 인 경우 caller 는 이 함수를 호출하지 않고 짧은
    예측 그대로 hard-RMM 에 흘려보낸다.  ``compute_metric_features_batched_scenes``
    가 sim 쪽 GT alignment 를 자동 처리하고, ``per_scenario_hard_rmm`` 에서
    log_feat 의 시간 축도 ``T_pred`` 로 잘라주므로 8초 padding 은 불필요.

    Args:
        pred_*_short: 모델이 만든 짧은 (예: 2 초 = 20 step) 예측들.
        gt_*_future: GT 8 초 미래 (10Hz, 80 step). 비활성 (active_mask False)
            agent 도 GT 가 들어있어야 하므로 caller 가 ``data["agent"]["position"]``
            full slice 를 그대로 넘긴다고 가정.
        pad_mode: ``"gt"`` (default) — 뒤 60 step 을 GT 로 채움 (OL/CL 차이가
            앞 2 초에서만 발생하도록).  ``"last"`` — 마지막 예측 step 을 hold,
            ``"zero"`` — 0 으로 채움 (debug 용; RMM 비현실적).
        valid_active_2hz: ``[n_agents]`` 비활성 agent (anchor t=0 에서 valid_window
            False) 는 OL 출력이 없으므로 8 초 전체를 GT 로 채워 RMM 영향 0.

    Returns:
        (full_traj, full_z, full_head): ``[n_agents, G, 80, 2/1/1]``.
    """
    n_agents, G, T_pred, _ = pred_traj_short.shape
    assert T_pred <= 80, f"T_pred={T_pred} 가 80 보다 크면 자르기를 caller 가 결정"

    device = pred_traj_short.device
    dtype = pred_traj_short.dtype

    # GT 를 G dim 으로 확장 (모든 rollout 이 같은 GT 로 패딩)
    gt_pos_g = gt_pos_future.unsqueeze(1).expand(n_agents, G, 80, 2).to(dtype)
    gt_z_g = gt_z_future.unsqueeze(1).expand(n_agents, G, 80).to(dtype)
    gt_head_g = gt_head_future.unsqueeze(1).expand(n_agents, G, 80).to(dtype)

    # 비활성 agent: 전체 80 step GT 사용 (OL 예측 없음)
    if valid_active_2hz is not None:
        active_idx = valid_active_2hz.bool()
    else:
        active_idx = torch.ones(n_agents, dtype=torch.bool, device=device)

    # 활성 agent 에 대해 앞 T_pred 는 model pred, 뒤는 pad_mode
    full_traj = gt_pos_g.clone()
    full_z = gt_z_g.clone()
    full_head = gt_head_g.clone()

    # active 부분에 prediction 덮기 (앞 T_pred)
    full_traj[active_idx, :, :T_pred, :] = pred_traj_short[active_idx]
    full_z[active_idx, :, :T_pred] = pred_z_short[active_idx]
    full_head[active_idx, :, :T_pred] = pred_head_short[active_idx]

    # 뒤 (T_pred:80) padding 분기
    if pad_mode == "gt":
        # 이미 gt 로 초기화돼 있으니 그대로
        pass
    elif pad_mode == "last":
        last_traj = pred_traj_short[:, :, -1:, :].expand(-1, G, 80 - T_pred, 2)
        last_z = pred_z_short[:, :, -1:].expand(-1, G, 80 - T_pred)
        last_head = pred_head_short[:, :, -1:].expand(-1, G, 80 - T_pred)
        full_traj[active_idx, :, T_pred:, :] = last_traj[active_idx]
        full_z[active_idx, :, T_pred:] = last_z[active_idx]
        full_head[active_idx, :, T_pred:] = last_head[active_idx]
    elif pad_mode == "zero":
        full_traj[active_idx, :, T_pred:, :] = 0.0
        full_z[active_idx, :, T_pred:] = 0.0
        full_head[active_idx, :, T_pred:] = 0.0
    else:
        raise ValueError(f"unknown pad_mode={pad_mode}")

    return full_traj, full_z, full_head


# ──────────────────────────────────────────────────────────────────────────────
# 1 batch 평가: OL 과 CL 의 80-step world prediction 을 paired noise 로 생성
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_ol_cl_2s_for_batch(
    model: LightningModule,
    data: dict,
    G: int,
    pred_max_steps: int,
    logger: logging.Logger,
    ol_use_gt: bool = False,
) -> Dict[str, torch.Tensor]:
    """한 batch 에 대해 OL/CL 의 ``T_pred = pred_max_steps * shift`` step
    (default 20 = 2 초) world prediction 텐서를 반환한다.

    OL: ``_sample_open_loop_future_from_hidden`` 으로 active anchor t=0 에서
    2 초 단발 ODE 호출.  CL: ``_run_parallel_rollout_chunk(max_steps=
    pred_max_steps, full_grad=True)`` 로 4 × 0.5 초 autoregressive rollout.
    같은 ``scenario_sampling_seeds`` + ``noise_tape_override`` 로 noise 공유.
    원본 hard RMM (8 초) 과 정합성을 깨지 않으면서 short horizon 으로 측정하기
    위해 padding 없이 짧은 텐서 그대로 반환 — caller (per_scenario_hard_rmm)
    가 ``compute_scenario_rollouts_features_2s(n_steps_override=T_pred)`` 로
    GT 도 같은 길이로 잘라 feature 추출을 한다.
    """
    encoder = model.encoder            # SMARTFlowDecoder wrapper
    agent_enc = encoder.agent_encoder  # SMARTFlowAgentDecoder
    shift = int(getattr(agent_enc, "shift", 5))   # 5: 2Hz → 10Hz
    sample_win = 20                                # OL 단발 ODE = 20 fine step
    n_step_future_10hz = int(agent_enc.num_future_steps)  # 80

    # 1) Map / agent token 인코딩 (validation_step 과 동일)
    tokenized_map, tokenized_agent = model.token_processor(data)
    map_feature = encoder.encode_map(tokenized_map)

    # 2) 평가용 rollout cache (t=0 anchor 기준)
    rollout_cache = agent_enc.prepare_inference_cache(
        tokenized_agent=tokenized_agent,
        map_feature=map_feature,
    )
    active_mask = rollout_cache["valid_window"][:, -1]   # [n_agent]
    n_agent_full = int(rollout_cache["valid_window"].shape[0])

    if not bool(active_mask.any()):
        logger.warning("active_mask 가 전부 False — 모든 agent invalid at t=0.")
        return {}

    current_pos_active = rollout_cache["pos_window"][:, -1][active_mask]   # [n_act, 2]
    current_head_active = rollout_cache["head_window"][:, -1][active_mask]
    active_hidden = rollout_cache["feat_a_now"][active_mask]               # [n_act, H]

    # 3) GT future (10Hz, 80 step) for padding
    hist = int(model.num_historical_steps)  # 11
    pos_full = data["agent"]["position"]   # [n_agent, T_total, 3]  T_total = 91
    head_full = data["agent"]["heading"]   # [n_agent, T_total]
    gt_pos_future = pos_full[:, hist:hist + 80, :2]
    gt_z_future = pos_full[:, hist:hist + 80, 2]
    gt_head_future = head_full[:, hist:hist + 80]

    # 4) Per-rollout: noise tape → OL & CL paired
    tape_steps = n_step_future_10hz + sample_win - shift   # 95
    device = active_hidden.device
    dtype = active_hidden.dtype

    pred_max_steps = int(pred_max_steps)
    T_pred_fine = pred_max_steps * shift  # 20 for 2 초
    assert T_pred_fine <= 80, f"T_pred_fine={T_pred_fine} > 80"

    # OL/CL 두 buffer (full 80 step world frame)
    ol_traj_list: List[torch.Tensor] = []   # G × [n_agent, 80, 2]
    ol_z_list: List[torch.Tensor] = []
    ol_head_list: List[torch.Tensor] = []

    cl_traj_list: List[torch.Tensor] = []
    cl_z_list: List[torch.Tensor] = []
    cl_head_list: List[torch.Tensor] = []

    t_ol = 0.0
    t_cl = 0.0
    for g in range(G):
        # 4a) Per-scenario seed (OL/CL 동일 — fair noise)
        seeds_g = model._get_closed_loop_scenario_seeds(
            scenario_ids=data["scenario_id"],
            rollout_idx=g,
            device=device,
        )

        # 4b) noise tape (OCSC 와 동일 spec)
        tape_g = agent_enc._build_rollout_noise_tape(
            num_agent=n_agent_full,
            tape_steps=tape_steps,
            device=device,
            dtype=dtype,
            sampling_noise=model.eval_sampling_noise,
            scenario_sampling_seeds=seeds_g,
            agent_batch=tokenized_agent["batch"],
            share_noise_across_time=False,
        )

        # 4c) "OL" slot:
        #   ol_use_gt=False (default): 기존 OL = `_sample_open_loop_future_from_hidden`
        #     를 같은 noise 의 앞 20 step 으로 호출 → world frame 변환
        #   ol_use_gt=True : OL 자리에 GT future trajectory (next T_pred_fine step
        #     in world frame) 를 직접 사용.  Sanity / ceiling 비교용 (sim≡log
        #     이라 metametric ≈ 1.0 가 나와야 metric pipeline 정상.  CL 의 metametric
        #     과 비교하면 "GT 대비 CL 이 얼마나 떨어지는가" 가 보임).
        _t0 = time.perf_counter()
        if ol_use_gt:
            # GT 직접 사용 — active mask 무관하게 모든 agent 의 GT slice
            ol_traj_short = gt_pos_future[:, :T_pred_fine].unsqueeze(1).to(dtype)
            ol_head_short = gt_head_future[:, :T_pred_fine].unsqueeze(1).to(dtype)
            ol_z_short = gt_z_future[:, :T_pred_fine].unsqueeze(1).to(dtype)
        else:
            x_init_ol = tape_g[active_mask, :sample_win, :].clone()
            ol_norm = agent_enc._sample_open_loop_future_from_hidden(
                anchor_hidden_valid=active_hidden,
                sampling_noise=model.eval_sampling_noise,
                x_init_override=x_init_ol,
            )   # [n_active, 20, 4]
            ol_pos_world, ol_head_world = ol_norm_to_world(
                ol_norm=ol_norm,
                current_pos=current_pos_active,
                current_head=current_head_active,
            )
            ol_traj_short = torch.zeros(n_agent_full, 1, T_pred_fine, 2,
                                        device=device, dtype=dtype)
            ol_head_short = torch.zeros(n_agent_full, 1, T_pred_fine,
                                        device=device, dtype=dtype)
            ol_z_short = gt_z_future[:, :T_pred_fine].unsqueeze(1).to(dtype)
            ol_traj_short[active_mask, 0] = ol_pos_world
            ol_head_short[active_mask, 0] = ol_head_world

        # 비활성 agent 는 OL prediction 이 없으므로 GT 의 첫 T_pred step 으로
        # 채운다 (CL 도 동일 처리 → 두 path 모두 비활성 agent 기여는 동일 = 0).
        if not bool(active_mask.all()):
            inactive = ~active_mask
            ol_traj_short[inactive, 0] = gt_pos_future[inactive, :T_pred_fine].to(dtype)
            ol_head_short[inactive, 0] = gt_head_future[inactive, :T_pred_fine].to(dtype)

        # native 2 초 (T_pred_fine = 20 step) RMM 을 직접 흘려보낸다 — 짧은
        # horizon 에 맞게 hard RMM 코드를 복사/수정한 ``hard_rmm_2s/`` 본체가
        # GT future 도 T_pred 로 자르므로 padding 불필요.
        ol_traj_list.append(ol_traj_short.squeeze(1))    # [n_agent, T_pred, 2]
        ol_z_list.append(ol_z_short.squeeze(1))
        ol_head_list.append(ol_head_short.squeeze(1))
        t_ol += time.perf_counter() - _t0

        # 4d) CL 2 초: rollout_from_cache(max_steps=pred_max_steps,
        #                                 noise_tape_override=tape_g)
        _t0 = time.perf_counter()
        # _run_parallel_rollout_chunk 가 chunk_size=1 일 때 noise_tape_override
        # 를 그대로 전달.  full_grad=True 로 max_steps 가 인자로 들어가도록 하되
        # outer @torch.no_grad() 로 gradient 는 안 만들어진다.
        cl_traj_g, cl_z_g, cl_head_g = model._run_parallel_rollout_chunk(
            data=data,
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            rollout_cache=rollout_cache,
            rollout_indices=[g],
            return_anchor_hidden=False,
            full_grad=True,
            max_steps=pred_max_steps,
            warm_coarse_steps=0,
            share_noise_across_time=False,
            noise_tape_override=tape_g,
        )
        # cl_*_g shapes: [n_agent, 1, T_pred_fine, 2/1/1]
        # T_pred_fine = pred_max_steps * shift (= 20 일 때)
        T_cl_actual = int(cl_traj_g.shape[-2])
        cl_traj_short = cl_traj_g.to(dtype)              # [n_agent, 1, T, 2]
        cl_z_short = cl_z_g.to(dtype)                    # [n_agent, 1, T]
        cl_head_short = cl_head_g.to(dtype)              # [n_agent, 1, T]

        # CL native 2 초: rollout_from_cache(max_steps=4) 가 이미 [n_agent, 1,
        # T_pred_fine, 2/h/z] 로 출력하므로 padding 없이 그대로 흘려보낸다.
        # 비활성 agent 는 OL 과 동일하게 GT 의 앞 T_pred 로 채워 비교 isolated.
        if not bool(active_mask.all()):
            inactive = ~active_mask
            cl_traj_short[inactive, 0] = gt_pos_future[inactive, :T_pred_fine].to(dtype)
            cl_head_short[inactive, 0] = gt_head_future[inactive, :T_pred_fine].to(dtype)
            cl_z_short[inactive, 0] = gt_z_future[inactive, :T_pred_fine].to(dtype)
        cl_traj_list.append(cl_traj_short.squeeze(1))
        cl_z_list.append(cl_z_short.squeeze(1))
        cl_head_list.append(cl_head_short.squeeze(1))
        t_cl += time.perf_counter() - _t0

    # 5) Stack G rollouts → [n_agent, G, 80, 2/1/1]
    ol_traj = torch.stack(ol_traj_list, dim=1)
    ol_z = torch.stack(ol_z_list, dim=1)
    ol_head = torch.stack(ol_head_list, dim=1)
    cl_traj = torch.stack(cl_traj_list, dim=1)
    cl_z = torch.stack(cl_z_list, dim=1)
    cl_head = torch.stack(cl_head_list, dim=1)

    logger.info(
        f"  ↳ rollout times (G={G}): OL_total={t_ol:.2f}s "
        f"CL_total={t_cl:.2f}s  active_agents={int(active_mask.sum())}/"
        f"{n_agent_full}"
    )
    return {
        "ol_traj": ol_traj,
        "ol_z": ol_z,
        "ol_head": ol_head,
        "cl_traj": cl_traj,
        "cl_z": cl_z,
        "cl_head": cl_head,
        "active_mask": active_mask,
        "n_agent": n_agent_full,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Wandb logging utilities
# ──────────────────────────────────────────────────────────────────────────────


def _build_metric_keys(prefix: str) -> List[str]:
    """``ol`` / ``cl`` / ``delta`` prefix 에 대한 metric 키 13 개 (1 metametric
    + 10 likelihoods + 1 ade-like 자리 + future-proof) 를 만든다."""
    keys = [f"{prefix}/realism_meta_metric"]
    for name in _LIKELIHOOD_NAMES:
        keys.append(f"{prefix}/{name}")
    return keys


def _per_scenario_to_flat(prefix: str, ps_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """Per-scenario list 의 평균을 ``prefix/...`` flat dict 로 만든다."""
    if not ps_list:
        return {}
    out: Dict[str, float] = {}
    out[f"{prefix}/realism_meta_metric"] = float(
        np.mean([d["metametric"] for d in ps_list])
    )
    for name in _LIKELIHOOD_NAMES:
        out[f"{prefix}/{name}"] = float(np.mean([d[name] for d in ps_list]))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


@hydra.main(version_base="1.3", config_path="../../configs", config_name="run")
def main(cfg: DictConfig) -> None:
    # ── env override (launcher 가 지정하는 모든 토글) ────────────────────────
    G = int(os.environ.get("OLCL_G_ROLLOUTS", "16"))
    pred_max_steps = int(os.environ.get("OLCL_PRED_2S_COARSE", "4"))
    limit_val_batches = float(os.environ.get("OLCL_LIMIT_VAL_BATCHES", "0.01"))
    # OL slot 을 model OL prediction 대신 GT future 로 교체할지 (sanity / ceiling 비교)
    ol_use_gt = os.environ.get("OLCL_OL_USE_GT", "false").strip().lower() in ("1", "true", "yes")
    # WOSAC official metric (TF subprocess + 공식 metrics 라이브러리) 사용 여부.
    # True 면 HardSimAgentsMetrics 대신 SimAgentsMetrics, 항상 8초 (80 step) 가정.
    use_official = os.environ.get("OLCL_USE_OFFICIAL_RMM", "false").strip().lower() in ("1", "true", "yes")
    wandb_project = os.environ.get("OLCL_WANDB_PROJECT", "project_3-ol-vs-cl-rmm")
    wandb_run_name = os.environ.get(
        "OLCL_WANDB_RUN_NAME", f"ol-vs-cl-{_kst_compact()}"
    )
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    ckpt_path_override = os.environ.get("OLCL_CKPT_PATH", "")

    # ── 결과 저장 디렉토리 (KST 타임스탬프 기반) ──────────────────────────
    out_dir = Path(__file__).resolve().parent / "artifacts" / _kst_compact()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(out_dir / "logs")
    logger.info(f"=== OL vs CL Hard-RMM 비교 시작 @ {_kst_now()} ===")
    logger.info(
        f"config: G={G} pred_max_steps={pred_max_steps} "
        f"(={pred_max_steps*0.5}초 horizon) "
        f"limit_val={limit_val_batches}  ol_use_gt={ol_use_gt}  use_official={use_official}"
    )
    if ol_use_gt:
        logger.info(
            "★ ol_use_gt=True — 'OL' slot 은 model OL 이 아니라 GT future trajectory."
            " 'OL' 컬럼은 실은 'GT' (sanity/ceiling), Δ = CL − GT (음수가 정상)."
        )
    if use_official:
        logger.info(
            "★ use_official=True — WOSAC official TF metric (SimAgentsMetrics) 사용. "
            "per-scenario 분리 없음, batch-level metametric 만 보고."
        )
        if pred_max_steps != 16:
            logger.warning(
                f"WOSAC official 은 8초 (80 step) 기준 — pred_max_steps={pred_max_steps} "
                f"권장값 16 으로 강제 override."
            )
            pred_max_steps = 16

    # ── Hydra cfg 강제 셋팅 (validate-only, 표준 OL/CL 메트릭 비활성) ─────
    with open_dict(cfg):
        cfg.action = "validate"
        if ckpt_path_override:
            cfg.ckpt_path = ckpt_path_override
        # 표준 validation_step 의 OL/CL eval 은 비활성 (우리 직접 실행)
        cfg.model.model_config.val_open_loop = False
        cfg.model.model_config.val_closed_loop = False
        # 본 비교에서 G 는 별도로 관리하므로 standard val 의 G 는 1 로 minimal
        cfg.model.model_config.n_rollout_closed_val = max(
            1, int(cfg.model.model_config.get("n_rollout_closed_val", 8))
        )
        # GT-padded 짧은 예측 → 비교에서 의미 있는 sim_agents_metric 은 우리 직접
        cfg.model.model_config.n_batch_sim_agents_metric = 0
        # trainer 지표
        cfg.trainer.limit_val_batches = limit_val_batches
        cfg.trainer.precision = "bf16-mixed"
        # 이 스크립트는 manual loop 라 logger/callbacks 자동 주입 막기
        cfg.logger = {}
        cfg.callbacks = {}

    if cfg.get("seed"):
        L.seed_everything(int(cfg.seed), workers=True)

    logger.info(f"Hydra cfg overrides 완료. ckpt_path={cfg.ckpt_path}")

    # ── Datamodule / model 인스턴스화 (run.py 와 동일) ─────────────────────
    logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)

    # ── Ckpt 로드 (manual: trainer.validate 안 씀) ─────────────────────────
    if cfg.get("ckpt_path"):
        logger.info(f"Loading ckpt: {cfg.ckpt_path}")
        ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        miss, unex = model.load_state_dict(sd, strict=False)
        logger.info(f"  state_dict loaded.  missing={len(miss)} unexpected={len(unex)}")
        if len(miss) > 0:
            logger.warning(f"  missing[:5]={miss[:5]}")
        if len(unex) > 0:
            logger.warning(f"  unexpected[:5]={unex[:5]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Moving model to {device} and setting eval()")
    model = model.to(device).eval()

    # ── Datamodule setup → val_loader ──────────────────────────────────────
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()
    n_total_batches = len(val_loader)
    # 0 < x < 1 → 비율 (e.g. 0.01 = 1%).  >=1 → 절대 batch 개수.  1.0 == 1 batch.
    if 0 < limit_val_batches < 1.0:
        n_use_batches = max(1, int(round(n_total_batches * limit_val_batches)))
    else:
        n_use_batches = max(1, min(int(limit_val_batches), n_total_batches))
    logger.info(
        f"val_loader: total_batches={n_total_batches}  use={n_use_batches} "
        f"(limit={limit_val_batches})"
    )

    # ── Wandb 초기화 ───────────────────────────────────────────────────────
    try:
        import wandb  # noqa: WPS433
    except ImportError:
        logger.error("wandb 가 설치되어 있지 않다.  pip install wandb 후 재실행.")
        raise

    wandb_run = wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        mode=wandb_mode,
        dir=str(out_dir / "wandb"),
        config={
            "G_rollouts": G,
            "pred_max_steps_coarse": pred_max_steps,
            "pred_horizon_seconds": pred_max_steps * 0.5,
            "rmm_path": "non_diff_2s_native",
            "ol_use_gt": ol_use_gt,
            "comparison_mode": "gt-vs-cl" if ol_use_gt else "ol-vs-cl",
            "limit_val_batches": limit_val_batches,
            "n_use_batches": n_use_batches,
            "n_total_batches": n_total_batches,
            "ckpt_path": cfg.get("ckpt_path", None),
            "kst_started": _kst_now(),
            "device": str(device),
        },
    )
    # wandb tag (구분용)
    wandb_run.tags = wandb_run.tags + ("ol-vs-cl-2s",) if wandb_run.tags else ("ol-vs-cl-2s",)
    logger.info(f"wandb run 시작: project={wandb_project} name={wandb_run_name} mode={wandb_mode}")

    # ── Metric 인스턴스 (OL/CL 별도) ───────────────────────────────────────
    if use_official:
        # WOSAC official: TF subprocess + 공식 metrics 라이브러리. 8초 standard.
        _ol_label = "gt_official_8s" if ol_use_gt else "ol_official_8s"
        _cl_label = "cl_official_8s"
        ol_metric = SimAgentsMetrics(_ol_label, max_workers=0)
        cl_metric = SimAgentsMetrics(_cl_label, max_workers=0)
    else:
        # PyTorch hard RMM (2s native, custom-modified)
        _ol_label = "gt_2s" if ol_use_gt else "ol_2s"
        _cl_label = "cl_2s"
        ol_metric = HardSimAgentsMetrics(_ol_label)
        cl_metric = HardSimAgentsMetrics(_cl_label)

    # ── Per-scenario CSV / 누적 buffer ─────────────────────────────────────
    csv_path = out_dir / "per_scenario.csv"
    csv_f = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_f)
    _csv_header = ["batch_idx", "scenario_idx_in_batch", "scenario_file"]
    for prefix in ("ol", "cl", "delta"):
        _csv_header.append(f"{prefix}_metametric")
        for name in _LIKELIHOOD_NAMES:
            _csv_header.append(f"{prefix}_{name}")
    csv_writer.writerow(_csv_header)
    csv_f.flush()

    # wandb table for per-scenario tracking (running)
    table_columns = (
        ["batch_idx", "scenario_idx_in_batch", "scenario_file"]
        + [f"ol_{k}" for k in (["metametric"] + list(_LIKELIHOOD_NAMES))]
        + [f"cl_{k}" for k in (["metametric"] + list(_LIKELIHOOD_NAMES))]
        + [f"delta_{k}" for k in (["metametric"] + list(_LIKELIHOOD_NAMES))]
    )
    per_scenario_table = wandb.Table(columns=table_columns)

    # 누적 평균용
    accum: Dict[str, List[float]] = {}
    n_scen_total = 0

    # ── 메인 루프 ──────────────────────────────────────────────────────────
    t_loop_start = time.time()
    for batch_idx, data in enumerate(val_loader):
        if batch_idx >= n_use_batches:
            break
        t_batch_start = time.time()
        # device 이동 (PyG Batch 타입은 .to() 가 동작; dict 가 아니면 알아서)
        data = data.to(device) if hasattr(data, "to") else data

        n_scen_in_batch = len(data["scenario_id"]) if "scenario_id" in data else 0
        logger.info(
            f"=== batch {batch_idx+1}/{n_use_batches} "
            f"n_scenarios={n_scen_in_batch} "
            f"@ {_kst_now()} ==="
        )

        # ── Predict ───────────────────────────────────────────────────────
        out = run_ol_cl_2s_for_batch(
            model=model,
            data=data,
            G=G,
            pred_max_steps=pred_max_steps,
            logger=logger,
            ol_use_gt=ol_use_gt,
        )
        if not out:
            logger.warning(f"batch {batch_idx} 건너뜀 (active 0).")
            continue

        # ── RMM (per-scenario or official) ────────────────────────────────
        if use_official:
            # 공식 metric 은 per-scenario 분리 미제공.  단순 batch-level update
            # 후 epoch 끝에서 compute() 한번에 평균값.  이번 batch 의 metametric
            # 만 보고 싶으면 reset → update → compute → reset 패턴.
            t0 = time.perf_counter()
            ol_metric.reset()
            ol_metric.update_from_prediction_tensors(
                scenario_files=list(data["tfrecord_path"]),
                agent_id=data["agent"]["id"],
                agent_batch=data["agent"]["batch"],
                pred_traj=out["ol_traj"],
                pred_z=out["ol_z"],
                pred_head=out["ol_head"],
            )
            ol_compute = ol_metric.compute()
            t_ol_rmm = time.perf_counter() - t0
            t0 = time.perf_counter()
            cl_metric.reset()
            cl_metric.update_from_prediction_tensors(
                scenario_files=list(data["tfrecord_path"]),
                agent_id=data["agent"]["id"],
                agent_batch=data["agent"]["batch"],
                pred_traj=out["cl_traj"],
                pred_z=out["cl_z"],
                pred_head=out["cl_head"],
            )
            cl_compute = cl_metric.compute()
            t_cl_rmm = time.perf_counter() - t0
            # 가짜 per-scenario 리스트 만들어서 기존 로깅 흐름 재사용
            def _flatten(compute_dict, prefix):
                out = {}
                for k, v in compute_dict.items():
                    short = k.split("/", 2)[-1]   # "realism_meta_metric" or "<name>"
                    out[short] = float(v.item() if torch.is_tensor(v) else v)
                # convert to list-of-1-scenario format used downstream
                d = {"scenario_file": "<batch_aggregated>",
                     "metametric": out.get("realism_meta_metric", float("nan"))}
                for name in _LIKELIHOOD_NAMES:
                    d[name] = out.get(name, float("nan"))
                return [d]
            ol_per = _flatten(ol_compute, _ol_label)
            cl_per = _flatten(cl_compute, _cl_label)
            logger.info(
                f"  ↳ official-RMM compute: OL={t_ol_rmm:.1f}s  CL={t_cl_rmm:.1f}s "
                f"(batch-aggregated)"
            )
        else:
            t0 = time.perf_counter()
            ol_per = compute_hard_rmm_per_scenario(
                metric=ol_metric,
                scenario_files=list(data["tfrecord_path"]),
                agent_id=data["agent"]["id"],
                agent_batch=data["agent"]["batch"],
                pred_traj=out["ol_traj"],
                pred_z=out["ol_z"],
                pred_head=out["ol_head"],
                update_running=True,
            )
            t_ol_rmm = time.perf_counter() - t0
            t0 = time.perf_counter()
            cl_per = compute_hard_rmm_per_scenario(
                metric=cl_metric,
                scenario_files=list(data["tfrecord_path"]),
                agent_id=data["agent"]["id"],
                agent_batch=data["agent"]["batch"],
                pred_traj=out["cl_traj"],
                pred_z=out["cl_z"],
                pred_head=out["cl_head"],
                update_running=True,
            )
            t_cl_rmm = time.perf_counter() - t0
            logger.info(
                f"  ↳ hard-RMM compute: OL={t_ol_rmm:.2f}s  CL={t_cl_rmm:.2f}s "
                f"per_scenario_count={len(ol_per)}"
            )

        # ── Per-scenario logging ──────────────────────────────────────────
        for s_i, (ol_d, cl_d) in enumerate(zip(ol_per, cl_per)):
            row = [batch_idx, s_i, ol_d["scenario_file"]]
            ol_meta = ol_d["metametric"]
            cl_meta = cl_d["metametric"]
            delta_meta = cl_meta - ol_meta
            row += [ol_meta] + [ol_d[n] for n in _LIKELIHOOD_NAMES]
            row += [cl_meta] + [cl_d[n] for n in _LIKELIHOOD_NAMES]
            row += [delta_meta] + [cl_d[n] - ol_d[n] for n in _LIKELIHOOD_NAMES]
            csv_writer.writerow(row)
            per_scenario_table.add_data(*row)
            n_scen_total += 1
        csv_f.flush()

        # ── Batch-level wandb log ─────────────────────────────────────────
        ol_flat = _per_scenario_to_flat("ol", ol_per)
        cl_flat = _per_scenario_to_flat("cl", cl_per)
        # ol/<name> 와 cl/<name> 짝 맞춰 delta/<name> 산출
        delta_flat: Dict[str, float] = {}
        for k_cl, v_cl in cl_flat.items():
            _name = k_cl.split("/", 1)[1]
            _k_ol = f"ol/{_name}"
            if _k_ol in ol_flat:
                delta_flat[f"delta/{_name}"] = v_cl - ol_flat[_k_ol]
        # running mean
        for d in (ol_flat, cl_flat, delta_flat):
            for k, v in d.items():
                accum.setdefault(k, []).append(v)
        running = {f"running/{k}": float(np.mean(vs)) for k, vs in accum.items()}

        wandb.log(
            {
                **ol_flat,
                **cl_flat,
                **delta_flat,
                **running,
                "progress/batch_idx": batch_idx,
                "progress/n_scenarios_seen": n_scen_total,
                "progress/wall_seconds": time.time() - t_loop_start,
            },
            step=batch_idx,
        )

        # 콘솔에 핵심 수치 정리
        logger.info(
            f"  ↳ OL meta={ol_flat['ol/realism_meta_metric']:.4f}  "
            f"CL meta={cl_flat['cl/realism_meta_metric']:.4f}  "
            f"Δ={delta_flat['delta/realism_meta_metric']:+.4f}  "
            f"(running OL={running['running/ol/realism_meta_metric']:.4f} "
            f"CL={running['running/cl/realism_meta_metric']:.4f} "
            f"Δ={running['running/delta/realism_meta_metric']:+.4f})"
        )
        # 가장 informative 한 likelihood 5 개도 verbose 로깅
        _spotlight = [
            "linear_speed_likelihood",
            "linear_acceleration_likelihood",
            "collision_indication_likelihood",
            "distance_to_road_edge_likelihood",
            "offroad_indication_likelihood",
        ]
        spot_str = []
        for k in _spotlight:
            ov = ol_flat.get(f"ol/{k}", float("nan"))
            cv = cl_flat.get(f"cl/{k}", float("nan"))
            spot_str.append(f"{k.split('_likelihood')[0]}: OL={ov:.3f} CL={cv:.3f} Δ={cv-ov:+.3f}")
        logger.info("    sub:  " + "  |  ".join(spot_str))
        logger.info(
            f"  ↳ batch wall={time.time()-t_batch_start:.1f}s  "
            f"loop wall={time.time()-t_loop_start:.0f}s"
        )

    # ── Epoch-level summary ───────────────────────────────────────────────
    csv_f.close()
    epoch_ol = ol_metric.compute()
    epoch_cl = cl_metric.compute()
    summary: Dict[str, float] = {}
    for k_ol, k_cl in zip(epoch_ol.keys(), epoch_cl.keys()):
        # k_ol prefix is "ol_2s/sim_agents_2025/<name>"
        nm = k_ol.split("/", 2)[-1]
        ol_v = float(epoch_ol[k_ol].item())
        cl_v = float(epoch_cl[k_cl].item())
        summary[f"epoch/ol/{nm}"] = ol_v
        summary[f"epoch/cl/{nm}"] = cl_v
        summary[f"epoch/delta/{nm}"] = cl_v - ol_v

    summary["progress/n_scenarios_total"] = n_scen_total
    summary["progress/total_wall_seconds"] = time.time() - t_loop_start
    wandb.log(summary, step=n_use_batches)
    wandb.log({"per_scenario_table": per_scenario_table}, step=n_use_batches)

    logger.info(f"=== Epoch summary @ {_kst_now()} ===")
    logger.info(
        f"  OL meta={summary['epoch/ol/realism_meta_metric']:.4f}  "
        f"CL meta={summary['epoch/cl/realism_meta_metric']:.4f}  "
        f"Δ={summary['epoch/delta/realism_meta_metric']:+.4f}"
    )
    logger.info(f"  per-scenario CSV: {csv_path}  ({n_scen_total} rows)")
    logger.info(f"  artifacts dir:    {out_dir}")

    # ── Pool 정리 ──────────────────────────────────────────────────────────
    try:
        ol_metric.close_pool()
        cl_metric.close_pool()
    except Exception:
        pass
    wandb.finish()
    logger.info(f"=== 종료 @ {_kst_now()} ===")


if __name__ == "__main__":
    main()
