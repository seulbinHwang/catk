#!/usr/bin/env python3
"""
Gradient Ascent on RMM — standalone test script.

Pipeline (per scenario):
  ① N-rollout flow model    →  Official RMM (N rollouts, baseline)
  ② Rollout-0 (no ascent)   →  Official RMM (1 rollout, fair baseline)
  ③ Gradient ascent on rollout-0 trajectory  (soft RMM objective)
  ④ Optimised rollout        →  Official RMM (1 rollout, after)

"Official RMM" = wm.compute_scenario_metrics_for_bundle (Waymo TF 구현, not pytorch)
"Soft RMM"     = compute_wosac_metametric_soft (differentiable PyTorch surrogate)

Comparison ②→④ is apples-to-apples (same single rollout, before/after ascent).

Usage:
    CUDA_VISIBLE_DEVICES=2,3 python scripts/test_grad_ascent_rmm.py

Configurable constants below.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── GPU guard ───────────────────────────────────────────────────────────────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Fake HydraConfig so SMARTFlow.__init__ doesn't crash ────────────────────
_OUTPUT_DIR = Path("/tmp/grad_ascent_rmm_test")
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import hydra.core.hydra_config  # noqa: E402
from omegaconf import OmegaConf


class _FakeHydraConfig:
    @staticmethod
    def get():
        return OmegaConf.create({"runtime": {"output_dir": str(_OUTPUT_DIR)}})


hydra.core.hydra_config.HydraConfig = _FakeHydraConfig

# ── Imports ──────────────────────────────────────────────────────────────────
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree as tg_degree

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")   # TF uses CPU only

from waymo_open_dataset.protos import scenario_pb2, sim_agents_metrics_pb2
import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as wm_metrics
from google.protobuf import text_format

from src.smart.model.smart_flow import SMARTFlow
from src.smart.datasets import MultiDataset
from src.smart.datamodules.target_builder import WaymoTargetBuilderVal
from src.utils.wosac_utils import get_scenario_rollouts, get_scenario_id_int_tensor

from src.smart.metrics.wosac_metric_features_torch.metric_features_torch import (
    compute_scenario_rollouts_features,
)
from src.smart.metrics.wosac_metric_features_torch.metric_features_torch_differentiable import (
    PredictedSimTrajectories,
    compute_metric_features_from_predicted_sim_trajectories,
)
from src.smart.metrics.wosac_metric_features_torch.surrogate import SurrogateConfig
from src.smart.metrics.wosac_metametric_pytorch_differentiable import (
    compute_wosac_metametric_soft,
    WosacMetametricSoftResult,
)
from src.utils.vis_waymo import VisWaymo

# ─────────────────────────────────────────────────────────────────────────────
# ▶  Configuration — edit here
# ─────────────────────────────────────────────────────────────────────────────
CKPT_PATH    = ROOT / "logs/pretrained/epoch_last.ckpt"
DATA_VAL_DIR = "/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1/validation"
TFRECORD_DIR = "/home2/pnc2/repos_python/datasets/smart_data/waymo_processed_catk_rebuild_parallel_v1/validation_tfrecords_splitted"

N_SCENARIOS    = 6    # number of scenarios to process
N_ROLLOUTS     = 32   # closed-loop rollouts for the N-rollout baseline
N_ASCENT_STEPS = 100  # gradient ascent iterations
ASCENT_LR      = 3e-3
VERBOSE_EVERY  = 10   # print full sub-metric breakdown every N steps

DEVICE = torch.device("cuda:0")   # physical GPU 2

# Surrogate temperatures for differentiable collision / offroad
SURROGATE = SurrogateConfig(
    collision_temperature=0.15,
    offroad_temperature=0.15,
    red_light_crossing_temperature=0.05,
)

VIDEO_DIR = _OUTPUT_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Sub-metric weights (for display)
_METRIC_WEIGHTS = {
    "linear_speed": 0.05,
    "linear_acceleration": 0.05,
    "angular_speed": 0.05,
    "angular_acceleration": 0.05,
    "distance_to_nearest_object": 0.10,
    "collision_indication": 0.25,
    "time_to_collision": 0.10,
    "distance_to_road_edge": 0.05,
    "offroad_indication": 0.25,
    "traffic_light_violation": 0.05,
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper: load model (patch old-format hparams)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device) -> SMARTFlow:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    mc = ckpt["hyper_parameters"]["model_config"]
    if "eval_sampling_noise" not in mc:
        old = mc.get("validation_rollout_sampling", {"noise_scale": 1.0})
        mc["eval_sampling_noise"] = {"noise_scale": float(old.get("noise_scale", 1.0))}
        print("    [patch] added eval_sampling_noise to hparams")
    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        tmp = f.name
    torch.save(ckpt, tmp)
    model = SMARTFlow.load_from_checkpoint(tmp, map_location="cpu", strict=False)
    os.unlink(tmp)
    model.eval()
    model.to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load Waymo official metric config
# ─────────────────────────────────────────────────────────────────────────────

def load_waymo_metric_config() -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
    config_path = Path(wm_metrics.__file__).parent / "challenge_2025_sim_agents_config.textproto"
    cfg = sim_agents_metrics_pb2.SimAgentMetricsConfig()
    with open(config_path) as f:
        text_format.Parse(f.read(), cfg)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Official RMM via Waymo TF (compute_scenario_metrics_for_bundle)
# ─────────────────────────────────────────────────────────────────────────────

def compute_official_rmm_waymo(
    scenario: scenario_pb2.Scenario,
    pred_traj: Tensor,    # [n_agent, n_rollout, 80, 2]
    pred_z:    Tensor,    # [n_agent, n_rollout, 80]
    pred_head: Tensor,    # [n_agent, n_rollout, 80]
    agent_ids: Tensor,    # [n_agent] int64
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
) -> float:
    """Official Waymo wm.compute_scenario_metrics_for_bundle."""
    from waymo_open_dataset.protos import sim_agents_submission_pb2

    pt  = pred_traj.detach().cpu().numpy()   # [A, R, 80, 2]
    pz  = pred_z.detach().cpu().numpy()      # [A, R, 80]
    ph  = pred_head.detach().cpu().numpy()   # [A, R, 80]
    ids = agent_ids.detach().cpu().numpy()   # [A]

    n_agents, n_rollout = pt.shape[:2]
    joint_scenes = []
    for r in range(n_rollout):
        sims = []
        for a in range(n_agents):
            sims.append(
                sim_agents_submission_pb2.SimulatedTrajectory(
                    center_x=pt[a, r, :, 0],
                    center_y=pt[a, r, :, 1],
                    center_z=pz[a, r],
                    heading=ph[a, r],
                    object_id=int(ids[a]),
                )
            )
        joint_scenes.append(sim_agents_submission_pb2.JointScene(simulated_trajectories=sims))

    sr = sim_agents_submission_pb2.ScenarioRollouts(
        joint_scenes=joint_scenes,
        scenario_id=scenario.scenario_id,
    )
    result = wm_metrics.compute_scenario_metrics_for_bundle(config, scenario, sr)
    return float(result.metametric)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Soft RMM for a single rollout  (differentiable)
# ─────────────────────────────────────────────────────────────────────────────

def compute_soft_rmm(
    scenario: scenario_pb2.Scenario,
    x:    Tensor,  # [A, 80]
    y:    Tensor,
    z:    Tensor,
    head: Tensor,
    agent_ids: Tensor,
    valid: Tensor,
    log_feat_dict: dict,
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
) -> WosacMetametricSoftResult:
    pred = PredictedSimTrajectories(
        object_id=agent_ids.cpu(),
        center_x=x, center_y=y, center_z=z, heading=head, valid=valid,
    )
    sim_feat = compute_metric_features_from_predicted_sim_trajectories(
        scenario=scenario, pred=pred, surrogate=SURROGATE,
    )
    return compute_wosac_metametric_soft(
        config=config,
        log_features=log_feat_dict,
        sim_features=sim_feat.as_dict(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gradient ascent
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_likelihoods(lik: Dict[str, Tensor]) -> str:
    parts = []
    for name, w in _METRIC_WEIGHTS.items():
        key = f"{name}_likelihood"
        v = float(lik[key])
        parts.append(f"{name[:7]}={v:.4f}(×{w})")
    return "  ".join(parts)


def run_gradient_ascent(
    scenario: scenario_pb2.Scenario,
    init_x:    Tensor,  # [A, 80]
    init_y:    Tensor,
    init_z:    Tensor,
    init_head: Tensor,
    agent_ids: Tensor,
    valid:     Tensor,
    log_feat_dict: dict,
    config: sim_agents_metrics_pb2.SimAgentMetricsConfig,
    n_steps: int = N_ASCENT_STEPS,
    lr: float = ASCENT_LR,
    verbose_every: int = VERBOSE_EVERY,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[dict]]:
    """Gradient ascent to maximise soft RMM on a single trajectory."""
    x    = init_x.clone().detach().requires_grad_(True)
    y    = init_y.clone().detach().requires_grad_(True)
    z    = init_z.clone().detach().requires_grad_(True)
    head = init_head.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([x, y, z, head], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr * 0.1)

    history: List[dict] = []

    for step in range(1, n_steps + 1):
        optimizer.zero_grad()

        result = compute_soft_rmm(
            scenario=scenario,
            x=x, y=y, z=z, head=head,
            agent_ids=agent_ids, valid=valid,
            log_feat_dict=log_feat_dict,
            config=config,
        )
        loss = -result.metametric
        loss.backward()

        grad_norm = float(
            torch.stack([p.grad.norm() for p in [x, y, z, head] if p.grad is not None]).norm()
        )
        torch.nn.utils.clip_grad_norm_([x, y, z, head], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # displacement from init (how much traj has moved)
        disp_xy = float(torch.stack([
            (x - init_x).abs().mean(),
            (y - init_y).abs().mean(),
        ]).mean())

        row = {
            "step": step,
            "soft_rmm": float(result.metametric.item()),
            "loss": float(loss.item()),
            "grad_norm": grad_norm,
            "lr": scheduler.get_last_lr()[0],
            "disp_xy_m": disp_xy,
            "likelihoods": {k: float(v.item()) for k, v in result.likelihoods.items()},
        }
        history.append(row)

        should_print = (step == 1 or step % verbose_every == 0 or step == n_steps)
        if should_print:
            lik = result.likelihoods
            weighted_str = "  ".join(
                f"{n[:6]}={float(lik[f'{n}_likelihood']):.4f}(w={w})"
                for n, w in _METRIC_WEIGHTS.items()
            )
            print(
                f"  step {step:4d}/{n_steps}"
                f"  soft_rmm={row['soft_rmm']:.6f}"
                f"  loss={row['loss']:+.6f}"
                f"  |grad|={row['grad_norm']:.4f}"
                f"  Δxy={row['disp_xy_m']:.3f}m"
                f"  lr={row['lr']:.2e}"
            )
            print(f"    ↳ {weighted_str}")

    return x.detach(), y.detach(), z.detach(), head.detach(), history


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Device : {DEVICE}")
    print(f"Ckpt   : {CKPT_PATH}")
    print(f"Scenarios: {N_SCENARIOS}  |  N_ROLLOUTS: {N_ROLLOUTS}  |  ascent steps: {N_ASCENT_STEPS}")

    # 1. Model
    print("\n[1] Loading model ...")
    model = load_model(CKPT_PATH, DEVICE)
    print(f"    {sum(p.numel() for p in model.parameters()):,} params")

    # 2. Waymo metric config
    wm_cfg = load_waymo_metric_config()

    # 3. Data
    print("\n[2] Loading validation data ...")
    dataset = MultiDataset(DATA_VAL_DIR, WaymoTargetBuilderVal(), TFRECORD_DIR)
    loader  = DataLoader(dataset, batch_size=N_SCENARIOS, shuffle=False, num_workers=2)
    batch   = next(iter(loader)).to(DEVICE)
    scenario_ids: List[str]  = batch["scenario_id"]
    tfrecord_paths: List[str] = batch["tfrecord_path"]
    print(f"    Loaded {len(scenario_ids)} scenarios")

    # 4. Closed-loop rollouts (N_ROLLOUTS)
    print(f"\n[3] Closed-loop rollouts (N={N_ROLLOUTS}) ...")
    with torch.no_grad():
        tokenized_map, tokenized_agent = model.token_processor(batch)
        map_feature   = model.encoder.encode_map(tokenized_map)
        orig_n = model.n_rollout_closed_val
        model.n_rollout_closed_val = N_ROLLOUTS
        pred_traj, pred_z, pred_head = model._run_closed_loop_rollouts(
            data=batch, tokenized_agent=tokenized_agent, map_feature=map_feature,
        )
        model.n_rollout_closed_val = orig_n
    print(f"    pred_traj: {tuple(pred_traj.shape)}  (n_agent, n_rollout, 80, 2)")

    # Unbatch by scenario
    sizes      = tg_degree(batch["agent"]["batch"], dtype=torch.long).tolist()
    pt_list    = pred_traj.split([int(s) for s in sizes], dim=0)
    pz_list    = pred_z.split([int(s) for s in sizes], dim=0)
    ph_list    = pred_head.split([int(s) for s in sizes], dim=0)
    aids_list  = batch["agent"]["id"].split([int(s) for s in sizes], dim=0)

    summary_rows = []

    for i_sc, (scen_id, tfr) in enumerate(zip(scenario_ids, tfrecord_paths)):
        print(f"\n{'='*70}")
        print(f"[Scenario {i_sc}/{N_SCENARIOS-1}]  id={scen_id}")

        pt   = pt_list[i_sc]   # [A, R, 80, 2]
        pz   = pz_list[i_sc]   # [A, R, 80]
        ph   = ph_list[i_sc]   # [A, R, 80]
        aids = aids_list[i_sc] # [A]

        # ── Load scenario proto ───────────────────────────────────────────────
        scenario = scenario_pb2.Scenario()
        for tfdata in tf.data.TFRecordDataset([tfr], compression_type=""):
            scenario.ParseFromString(bytes(tfdata.numpy()))
            break

        # ── ① Official RMM: N rollouts (normal evaluation baseline) ──────────
        print(f"  ① Official RMM (N={N_ROLLOUTS} rollouts)  ...", end=" ", flush=True)
        rmm_n_rollout = compute_official_rmm_waymo(scenario, pt, pz, ph, aids, wm_cfg)
        print(f"{rmm_n_rollout:.6f}")

        # ── ② Official RMM: rollout-0 only (fair 1-rollout baseline) ─────────
        print(f"  ② Official RMM (1 rollout, rollout-0, before) ...", end=" ", flush=True)
        rmm_1_before = compute_official_rmm_waymo(
            scenario, pt[:, :1], pz[:, :1], ph[:, :1], aids, wm_cfg
        )
        print(f"{rmm_1_before:.6f}")

        # ── Pre-compute log features (fixed, from GT) ─────────────────────────
        sr_for_log = get_scenario_rollouts(
            scenario_id=get_scenario_id_int_tensor([scen_id], DEVICE),
            agent_id=aids,
            agent_batch=torch.zeros(aids.shape[0], dtype=torch.long, device=DEVICE),
            pred_traj=pt[:, :1], pred_z=pz[:, :1], pred_head=ph[:, :1],
        )
        log_feat, _ = compute_scenario_rollouts_features(scenario, sr_for_log[0])
        log_feat_dict = {k: v.to(DEVICE) for k, v in log_feat.as_dict().items()}

        # ── Soft RMM before (rollout-0) ───────────────────────────────────────
        valid_t = torch.ones(pt.shape[0], 80, dtype=torch.bool, device=DEVICE)
        with torch.no_grad():
            soft_before = compute_soft_rmm(
                scenario=scenario,
                x=pt[:, 0, :, 0], y=pt[:, 0, :, 1],
                z=pz[:, 0, :], head=ph[:, 0, :],
                agent_ids=aids, valid=valid_t,
                log_feat_dict=log_feat_dict, config=wm_cfg,
            )
        print(f"     Soft RMM (1 rollout, before) = {float(soft_before.metametric):.6f}")

        # ── Video before ──────────────────────────────────────────────────────
        save_before = VIDEO_DIR / f"sc{i_sc:02d}_{scen_id[:8]}_before"
        sr_vis = get_scenario_rollouts(
            scenario_id=get_scenario_id_int_tensor([scen_id], DEVICE),
            agent_id=aids,
            agent_batch=torch.zeros(aids.shape[0], dtype=torch.long, device=DEVICE),
            pred_traj=pt, pred_z=pz, pred_head=ph,
        )
        vis = VisWaymo(scenario_path=tfr, save_dir=save_before)
        vis.save_video_scenario_rollout(sr_vis[0], n_vis_rollout=min(4, N_ROLLOUTS))
        print(f"     Video before → {save_before.name}/")

        # ── ③ Gradient ascent (on rollout-0) ─────────────────────────────────
        print(f"\n  ③ Gradient ascent  ({N_ASCENT_STEPS} steps, lr={ASCENT_LR})")
        opt_x, opt_y, opt_z, opt_head, ga_history = run_gradient_ascent(
            scenario=scenario,
            init_x=pt[:, 0, :, 0].clone(),
            init_y=pt[:, 0, :, 1].clone(),
            init_z=pz[:, 0, :].clone(),
            init_head=ph[:, 0, :].clone(),
            agent_ids=aids, valid=valid_t,
            log_feat_dict=log_feat_dict, config=wm_cfg,
        )

        # ── ④ Official RMM after ascent (1 rollout, fair comparison with ②) ──
        opt_traj = torch.stack([opt_x, opt_y], dim=-1).unsqueeze(1)  # [A,1,80,2]
        opt_z_t  = opt_z.unsqueeze(1)                                  # [A,1,80]
        opt_h_t  = opt_head.unsqueeze(1)                               # [A,1,80]

        print(f"\n  ④ Official RMM (1 rollout, after) ...", end=" ", flush=True)
        rmm_1_after = compute_official_rmm_waymo(scenario, opt_traj, opt_z_t, opt_h_t, aids, wm_cfg)
        print(f"{rmm_1_after:.6f}")

        with torch.no_grad():
            soft_after = compute_soft_rmm(
                scenario=scenario,
                x=opt_x, y=opt_y, z=opt_z, head=opt_head,
                agent_ids=aids, valid=valid_t,
                log_feat_dict=log_feat_dict, config=wm_cfg,
            )
        print(f"     Soft RMM (1 rollout, after)  = {float(soft_after.metametric):.6f}")

        delta_official = rmm_1_after - rmm_1_before
        delta_soft     = float(soft_after.metametric) - float(soft_before.metametric)
        print(f"\n  ΔOfficial (②→④) : {delta_official:+.6f}  "
              f"({'↑ improved' if delta_official > 0 else '↓ degraded'})")
        print(f"  ΔSoft            : {delta_soft:+.6f}  "
              f"({'↑ improved' if delta_soft > 0 else '↓ degraded'})")

        # ── Video after ───────────────────────────────────────────────────────
        save_after = VIDEO_DIR / f"sc{i_sc:02d}_{scen_id[:8]}_after"
        sr_after = get_scenario_rollouts(
            scenario_id=get_scenario_id_int_tensor([scen_id], DEVICE),
            agent_id=aids,
            agent_batch=torch.zeros(aids.shape[0], dtype=torch.long, device=DEVICE),
            pred_traj=opt_traj, pred_z=opt_z_t, pred_head=opt_h_t,
        )
        vis_after = VisWaymo(scenario_path=tfr, save_dir=save_after)
        vis_after.save_video_scenario_rollout(sr_after[0], n_vis_rollout=1)
        print(f"     Video after  → {save_after.name}/")

        # ── Sub-metric breakdown before/after ─────────────────────────────────
        print("\n  Sub-metric likelihoods (before → after):")
        for name, w in _METRIC_WEIGHTS.items():
            key = f"{name}_likelihood"
            v_b = float(soft_before.likelihoods[key])
            v_a = float(soft_after.likelihoods[key])
            arrow = "↑" if v_a > v_b else ("↓" if v_a < v_b else "=")
            print(f"    {name:<35}  {v_b:.4f} → {v_a:.4f}  {arrow}  (w={w})")

        summary_rows.append({
            "scenario":     scen_id[:14],
            "rmm_n_rollout": rmm_n_rollout,
            "rmm_1_before": rmm_1_before,
            "rmm_1_after":  rmm_1_after,
            "delta_off":    delta_official,
            "soft_before":  float(soft_before.metametric),
            "soft_after":   float(soft_after.metametric),
            "delta_soft":   delta_soft,
            "ga_history":   ga_history,
        })

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    hdr = (
        f"{'Scenario':<16}  {'RMM(N-roll)':>11}  {'RMM(1,bfr)':>10}  "
        f"{'RMM(1,aft)':>10}  {'Δ(1-roll)':>9}  {'Δsoft':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in summary_rows:
        sign = "+" if r["delta_off"] >= 0 else ""
        ssign = "+" if r["delta_soft"] >= 0 else ""
        print(
            f"{r['scenario']:<16}  "
            f"{r['rmm_n_rollout']:>11.6f}  "
            f"{r['rmm_1_before']:>10.6f}  "
            f"{r['rmm_1_after']:>10.6f}  "
            f"{sign}{r['delta_off']:>8.6f}  "
            f"{ssign}{r['delta_soft']:.6f}"
        )

    # Gradient ascent curve for first scenario
    if summary_rows:
        print(f"\nGradient ascent curve  (scenario 0):")
        print(f"  {'step':>5}  {'soft_rmm':>10}  {'loss':>10}  {'|grad|':>8}  {'Δxy(m)':>8}  {'lr':>8}")
        for row in summary_rows[0]["ga_history"]:
            s = row["step"]
            if s == 1 or s % VERBOSE_EVERY == 0 or s == N_ASCENT_STEPS:
                print(
                    f"  {s:>5}  {row['soft_rmm']:>10.6f}  "
                    f"{row['loss']:>+10.6f}  {row['grad_norm']:>8.4f}  "
                    f"{row['disp_xy_m']:>8.4f}  {row['lr']:>8.2e}"
                )

    print(f"\nVideos → {VIDEO_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
