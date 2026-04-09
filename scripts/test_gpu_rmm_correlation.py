"""Correlation test: GPU RMM vs official Waymo RMM.

**Typical workflow (what you want):**
  1. **Once:** fill per-scenario official scores on disk (slow TF), with deterministic
     rollouts — same ``stem|G|noise`` always yields the same ``pred`` and the same
     labels for correlation.
  2. **After changing ``gpu_rmm``:** run again **without** recomputing official: only
     ``compute_gpu_rmm`` is executed; correlation is Spearman/Pearson vs the **stored**
     official row. Use ``--require-official-cache`` so TF never starts by accident.

Rollouts: fixed seed from ``(stem, G, noise, ROLLOUT_SCHEME_ID)``.

Usage:
    # One-time (or when cache stale): compute official + save cache
    python scripts/test_gpu_rmm_correlation.py --n-scenarios 20 --G 4 --noise 2.0

    # Iterating on gpu_rmm.py: GPU only, fail if cache missing
    python scripts/test_gpu_rmm_correlation.py --require-official-cache ...

    python scripts/test_gpu_rmm_correlation.py --refresh-official-cache   # force TF recompute
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ── project root on path ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── dataset paths ──────────────────────────────────────────────────────────────
VAL_PKL_DIR = Path(
    "/home2/pnc2/repos_python/datasets/smart_data/"
    "waymo_processed_catk_rebuild_parallel_v1/validation"
)
VAL_TF_DIR = VAL_PKL_DIR.parent / "validation_tfrecords_splitted"

# Official RMM cache (per scenario / G / noise / Waymo config / rollout scheme)
OFFICIAL_CACHE_DIR = REPO_ROOT / ".cache" / "gpu_rmm_correlation_official"

# Bump this if ``build_rollouts`` / seed definition changes (invalidates old caches).
ROLLOUT_SCHEME_ID = "deterministic_seed_v1"


def _official_config_tag() -> str:
    """Fingerprint the same textproto ``_sim_agents_worker`` uses."""
    try:
        import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as wm
        from google.protobuf import text_format
        from waymo_open_dataset.protos import sim_agents_metrics_pb2

        base = Path(wm.__file__).resolve().parent
        p = base / "challenge_2025_sim_agents_config.textproto"
        if not p.is_file():
            p = base / "challenge_2024_config.textproto"
        raw = p.read_bytes()
        cfg = sim_agents_metrics_pb2.SimAgentMetricsConfig()
        text_format.Parse(raw.decode(), cfg)
        digest = hashlib.sha256(raw).hexdigest()[:16]
        return f"{p.name}_{digest}"
    except Exception:
        return "unknown_config"


def _rollout_seed(stem: str, G: int, noise: float) -> int:
    h = hashlib.sha256(
        f"{stem}|{G}|{noise:.10f}|{ROLLOUT_SCHEME_ID}".encode()
    ).hexdigest()
    return int(h[:12], 16) % (2**31)


def _official_cache_path(
    cache_dir: Path, stem: str, G: int, noise: float, config_tag: str
) -> Path:
    payload = (
        f"{stem}\0{G}\0{noise:.10f}\0{config_tag}\0{ROLLOUT_SCHEME_ID}".encode()
    )
    h = hashlib.sha256(payload).hexdigest()[:20]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:72]
    return cache_dir / f"{safe}__G{G}__n{noise:g}__{h}.json"


def _load_official_cache(path: Path, G: int) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    try:
        with open(path, "r") as f:
            obj: Dict[str, Any] = json.load(f)
        if obj.get("rollout_scheme_id") != ROLLOUT_SCHEME_ID:
            return None
        arr = np.asarray(obj["official_scores"], dtype=np.float64)
        if arr.shape != (G,):
            return None
        return arr.astype(np.float32)
    except Exception:
        return None


def _save_official_cache(
    path: Path,
    stem: str,
    G: int,
    noise: float,
    config_tag: str,
    scores: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(
            {
                "stem": stem,
                "G": G,
                "noise_std": noise,
                "config_tag": config_tag,
                "rollout_scheme_id": ROLLOUT_SCHEME_ID,
                "official_scores": scores.astype(float).tolist(),
            },
            f,
        )
    tmp.replace(path)


# ── helpers ───────────────────────────────────────────────────────────────────


def load_pkls(n: int) -> List[Tuple[str, dict]]:
    """Return up to n (stem, data) pairs from the validation directory."""
    import pickle

    pairs = []
    for pkl_path in sorted(VAL_PKL_DIR.glob("*.pkl")):
        stem = pkl_path.stem
        tf_path = VAL_TF_DIR / f"{stem}.tfrecords"
        if not tf_path.exists():
            continue
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        pairs.append((stem, data, str(tf_path)))
        if len(pairs) >= n:
            break
    return pairs


def build_rollouts(
    data: dict,
    G: int,
    noise_std: float,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create G perturbed future rollouts from GT (reproducible via ``seed``)."""
    agent = data["agent"]
    pos = agent["position"].to(device, dtype=torch.float32)
    head = agent["heading"].to(device, dtype=torch.float32)
    n_agents = pos.shape[0]

    gt_fut_xy = pos[:, 11:, :2]
    gt_fut_z = pos[:, 11:, 2]
    gt_fut_h = head[:, 11:]

    pred_traj = gt_fut_xy.unsqueeze(1).expand(-1, G, -1, -1).clone()
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    noise_np = rng.standard_normal(pred_traj.shape, dtype=np.float32)
    noise = torch.from_numpy(noise_np).to(device)
    pred_traj = pred_traj + noise * noise_std

    pred_z = gt_fut_z.unsqueeze(1).expand(-1, G, -1).clone()
    pred_head = gt_fut_h.unsqueeze(1).expand(-1, G, -1).clone()

    return pred_traj, pred_z, pred_head


def call_official_rmm(
    stem: str,
    tf_path: str,
    data: dict,
    pred_traj: torch.Tensor,
    pred_z: torch.Tensor,
    pred_head: torch.Tensor,
) -> np.ndarray:
    """Run official RMM via ``_sim_agents_worker`` for each rollout."""
    from src.smart.metrics import _sim_agents_worker
    from google.protobuf import text_format
    from waymo_open_dataset.protos import sim_agents_metrics_pb2
    import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as wm

    config_path = Path(wm.__file__).parent / "challenge_2025_sim_agents_config.textproto"
    if not config_path.exists():
        config_path = Path(wm.__file__).parent / "challenge_2024_config.textproto"
    with open(config_path, "r") as f:
        cfg = sim_agents_metrics_pb2.SimAgentMetricsConfig()
        text_format.Parse(f.read(), cfg)
    config_bytes = cfg.SerializeToString()

    agent = data["agent"]
    agent_ids = agent["id"].cpu().numpy()

    pred_traj_np = pred_traj.cpu().numpy()
    pred_z_np = pred_z.cpu().numpy()
    pred_head_np = pred_head.cpu().numpy()

    G = pred_traj_np.shape[1]
    scores = np.zeros(G, dtype=np.float32)

    for g in range(G):
        score_g = _sim_agents_worker(
            config_bytes,
            tf_path,
            agent_ids,
            pred_traj_np[:, g : g + 1],
            pred_z_np[:, g : g + 1],
            pred_head_np[:, g : g + 1],
        )
        scores[g] = score_g

    return scores


def call_gpu_rmm(
    data: dict,
    pred_traj: torch.Tensor,
    pred_z: torch.Tensor,
    pred_head: torch.Tensor,
    device: torch.device,
    return_subscores: bool = False,
):
    from src.smart.metrics.gpu_rmm import compute_gpu_rmm

    agent = data["agent"]
    gt_pos = agent["position"].to(device, dtype=torch.float32)
    gt_head = agent["heading"].to(device, dtype=torch.float32)
    gt_valid = agent["valid_mask"].to(device)
    shape = agent["shape"].to(device, dtype=torch.float32)
    atype = agent["type"].to(device)
    role = agent["role"].to(device)
    eval_mask = role[:, 2]

    map_save = data["map_save"]
    map_pos = map_save["traj_pos"].to(device, dtype=torch.float32)

    # Pass road-edge token types if available (pt_token["type"])
    map_token_type = None
    pt_token = data.get("pt_token", None)
    if pt_token is not None and "type" in pt_token:
        map_token_type = pt_token["type"].to(device)

    return compute_gpu_rmm(
        pred_traj=pred_traj,
        pred_z=pred_z,
        pred_head=pred_head,
        gt_position=gt_pos,
        gt_heading=gt_head,
        gt_valid=gt_valid,
        agent_shape=shape,
        agent_type=atype,
        eval_mask=eval_mask,
        map_token_pos=map_pos,
        dt=0.1,
        map_token_type=map_token_type,
        return_subscores=return_subscores,
    )


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU RMM vs official RMM correlation test")
    parser.add_argument("--n-scenarios", type=int, default=20, help="Number of scenarios")
    parser.add_argument("--G", type=int, default=4, help="Rollouts per scenario")
    parser.add_argument("--noise", type=float, default=2.0, help="Gaussian noise std (m)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-official", action="store_true", help="GPU-only; no correlation vs official")
    parser.add_argument("--subscores", action="store_true", help="Print GPU sub-scores per scenario")
    parser.add_argument(
        "--refresh-official-cache",
        action="store_true",
        help="Ignore cache and recompute official RMM (still saves cache after)",
    )
    parser.add_argument(
        "--require-official-cache",
        action="store_true",
        help="Never run TF official: load scores from cache only (for gpu_rmm iterations)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help=f"Override official cache dir (default: {OFFICIAL_CACHE_DIR})",
    )
    args = parser.parse_args()
    if args.require_official_cache and args.refresh_official_cache:
        print("ERROR: use only one of --require-official-cache and --refresh-official-cache")
        sys.exit(2)

    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else OFFICIAL_CACHE_DIR
    )

    device = torch.device(args.device)
    G = args.G
    noise_std = args.noise
    config_tag = _official_config_tag()

    print(f"Device: {device}")
    print(f"Scenarios: {args.n_scenarios}, G={G}, noise_std={noise_std:.1f}m")
    print(f"Official config fingerprint: {config_tag}")
    print(f"Rollout scheme: {ROLLOUT_SCHEME_ID}")
    print(f"Official cache dir: {cache_dir}")
    if args.require_official_cache:
        print("Mode: GPU RMM only — official from cache (TF disabled)")
    print(f"Val PKL dir:  {VAL_PKL_DIR}")
    print(f"Val TFR dir:  {VAL_TF_DIR}")
    print()

    print("Loading scenarios...")
    scenarios = load_pkls(args.n_scenarios)
    if len(scenarios) == 0:
        print("ERROR: No matching (pkl, tfrecords) pairs found.")
        sys.exit(1)
    print(f"Loaded {len(scenarios)} scenarios.\n")

    official_all: List[float] = []
    gpu_all: List[float] = []
    total_gpu_time = 0.0
    total_official_time = 0.0
    n_cache_hit = 0
    n_cache_miss = 0

    for idx, (stem, data, tf_path) in enumerate(scenarios):
        print(f"[{idx+1}/{len(scenarios)}] {stem}", end=" ", flush=True)

        seed = _rollout_seed(stem, G, noise_std)
        pred_traj, pred_z, pred_head = build_rollouts(data, G, noise_std, device, seed)

        t0 = time.time()
        with torch.no_grad():
            if args.subscores:
                gpu_scores, gpu_sub = call_gpu_rmm(data, pred_traj, pred_z, pred_head, device, return_subscores=True)
            else:
                gpu_scores = call_gpu_rmm(data, pred_traj, pred_z, pred_head, device)
                gpu_sub = None
        total_gpu_time += time.time() - t0
        gpu_scores_np = gpu_scores.cpu().numpy()
        print(f"GPU={gpu_scores_np}", end=" ", flush=True)
        if args.subscores and gpu_sub is not None:
            print()
            for k, v in gpu_sub.items():
                print(f"  {k:35s}: {v.cpu().numpy().mean():.4f}")

        if args.no_official:
            official_scores = np.full(G, float("nan"), dtype=np.float32)
            print()
        else:
            cpath = _official_cache_path(cache_dir, stem, G, noise_std, config_tag)
            cached = None if args.refresh_official_cache else _load_official_cache(cpath, G)
            if cached is not None:
                official_scores = cached
                n_cache_hit += 1
                print(f"Official={official_scores} (cache)")
            elif args.require_official_cache:
                print(f"\nERROR: missing official cache for {stem}: {cpath}")
                print("Run once without --require-official-cache (or copy cache files here).")
                sys.exit(1)
            else:
                n_cache_miss += 1
                t0 = time.time()
                try:
                    official_scores = call_official_rmm(
                        stem, tf_path, data, pred_traj, pred_z, pred_head
                    )
                    total_official_time += time.time() - t0
                    _save_official_cache(
                        cpath, stem, G, noise_std, config_tag, official_scores
                    )
                    print(f"Official={official_scores} (computed)")
                except Exception as e:
                    warnings.warn(f"Official RMM failed for {stem}: {e}")
                    official_scores = np.full(G, float("nan"), dtype=np.float32)
                    print("Official=FAILED")

        official_all.extend(official_scores.tolist())
        gpu_all.extend(gpu_scores_np.tolist())

    print()
    print("=" * 60)
    if not args.no_official:
        print(f"Official cache hits: {n_cache_hit}  misses (computed): {n_cache_miss}")
    print()

    official_arr = np.array(official_all, dtype=np.float64)
    gpu_arr = np.array(gpu_all, dtype=np.float64)
    mask = np.isfinite(official_arr) & np.isfinite(gpu_arr)
    n_pairs = int(mask.sum())

    print(f"Total (scenario, rollout) pairs: {len(official_arr)}")
    print(f"Valid pairs (both finite):       {n_pairs}")
    print()

    if n_pairs >= 2:
        from scipy.stats import pearsonr, spearmanr

        off_valid = official_arr[mask]
        gpu_valid = gpu_arr[mask]

        pearson_r, pearson_p = pearsonr(off_valid, gpu_valid)
        spearman_r, spearman_p = spearmanr(off_valid, gpu_valid)
        mae = float(np.mean(np.abs(off_valid - gpu_valid)))

        print(f"Pearson  r = {pearson_r:+.4f}  (p={pearson_p:.3e})")
        print(f"Spearman r = {spearman_r:+.4f}  (p={spearman_p:.3e})")
        print(f"MAE        = {mae:.4f}")
        print()
        print(f"Official RMM  mean={off_valid.mean():.4f}  std={off_valid.std():.4f}")
        print(f"GPU RMM       mean={gpu_valid.mean():.4f}  std={gpu_valid.std():.4f}")
    else:
        print("Not enough valid pairs for correlation analysis.")

    print()
    print(f"Avg GPU time / scenario:      {total_gpu_time / len(scenarios) * 1000:.1f} ms")
    if not args.no_official and (n_cache_miss > 0 or total_official_time > 0):
        denom = max(1, n_cache_miss)
        print(
            f"Avg Official time (computed only): {total_official_time / denom * 1000:.1f} ms  "
            f"over {n_cache_miss} miss(es)"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
