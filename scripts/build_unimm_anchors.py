from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import sys
from collections import defaultdict
from collections.abc import Iterable
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import transform_to_local, wrap_angle
from src.unimm.anchors import AGENT_TYPE_NAMES, NUM_AGENT_TYPES, anchor_distance

DEFAULT_NUM_WORKERS = min(16, os.cpu_count() or 1)


def _iter_cache_files(cache_dir: Path):
    yield from sorted(p for p in cache_dir.glob("*.pkl") if p.is_file() and not p.name.startswith("."))


def _chunked(items: list[Path], chunk_size: int) -> Iterable[tuple[str, ...]]:
    for start in range(0, len(items), chunk_size):
        yield tuple(str(path) for path in items[start : start + chunk_size])


def _extract_one_file_trajectory_groups(path_str: str, horizon_steps: int, start_step: int) -> dict[int, np.ndarray]:
    with Path(path_str).open("rb") as handle:
        data = pickle.load(handle)

    agent = data["agent"]
    pos = agent["position"][..., :2].clone().contiguous()
    head = TokenProcessor._clean_heading(agent["valid_mask"].clone(), agent["heading"].clone())
    valid = agent["valid_mask"]
    agent_type = agent["type"].long()

    if start_step + horizon_steps >= pos.shape[1]:
        return {}
    future_valid = valid[:, start_step] & valid[:, start_step + 1 : start_step + horizon_steps + 1].all(dim=1)
    rows = torch.nonzero(future_valid, as_tuple=False).flatten()
    if rows.numel() == 0:
        return {}

    row_types = agent_type[rows]
    type_valid = (row_types >= 0) & (row_types < NUM_AGENT_TYPES)
    rows = rows[type_valid]
    row_types = row_types[type_valid]
    if rows.numel() == 0:
        return {}

    fut_pos = pos[rows, start_step + 1 : start_step + horizon_steps + 1]
    fut_head = head[rows, start_step + 1 : start_step + horizon_steps + 1]
    local_pos, local_head = transform_to_local(
        pos_global=fut_pos,
        head_global=fut_head,
        pos_now=pos[rows, start_step],
        head_now=head[rows, start_step],
    )
    traj = torch.cat([local_pos, wrap_angle(local_head).unsqueeze(-1)], dim=-1).cpu()

    groups: dict[int, np.ndarray] = {}
    for type_idx in range(NUM_AGENT_TYPES):
        mask = row_types == type_idx
        if bool(mask.any()):
            groups[type_idx] = traj[mask].contiguous().numpy()
    return groups


def _extract_file_trajectory_groups(args: tuple[tuple[str, ...], int, int]) -> dict[int, np.ndarray]:
    path_chunk, horizon_steps, start_step = args
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for path_str in path_chunk:
        groups = _extract_one_file_trajectory_groups(path_str, horizon_steps, start_step)
        for type_idx, trajectories in groups.items():
            grouped[type_idx].append(trajectories)

    output: dict[int, np.ndarray] = {}
    for type_idx, parts in grouped.items():
        output[type_idx] = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
    return output


def _worker_init() -> None:
    torch.set_num_threads(1)


def _update_reservoir(
    reservoirs: dict[int, torch.Tensor],
    filled: dict[int, int],
    seen: dict[int, int],
    rng: random.Random,
    groups: dict[int, np.ndarray],
    max_per_type: int,
) -> None:
    for type_idx, trajectories_np in groups.items():
        trajectories = torch.from_numpy(trajectories_np)
        bucket = reservoirs[type_idx]
        for traj in trajectories:
            seen[type_idx] += 1
            if filled[type_idx] < max_per_type:
                bucket[filled[type_idx]].copy_(traj)
                filled[type_idx] += 1
            else:
                replace_idx = rng.randrange(seen[type_idx])
                if replace_idx < max_per_type:
                    bucket[replace_idx].copy_(traj)


def collect_training_trajectories(
    train_cache_dir: Path,
    max_per_type: int | None,
    horizon_steps: int,
    start_step: int,
    seed: int,
    num_workers: int,
    collect_file_chunk_size: int,
) -> dict[str, torch.Tensor]:
    rng = random.Random(seed)
    seen = defaultdict(int)
    use_all_trajectories = max_per_type is None
    reservoirs: dict[int, torch.Tensor] = {}
    filled = defaultdict(int)
    blocks: dict[int, list[torch.Tensor]] = defaultdict(list)
    if not use_all_trajectories:
        reservoirs = {
            type_idx: torch.empty((max_per_type, horizon_steps, 3), dtype=torch.float32)
            for type_idx in range(NUM_AGENT_TYPES)
        }

    files = list(_iter_cache_files(train_cache_dir))
    file_chunks = list(_chunked(files, collect_file_chunk_size))
    worker_args = ((chunk, horizon_steps, start_step) for chunk in file_chunks)

    def consume(groups: dict[int, np.ndarray]) -> None:
        if use_all_trajectories:
            for type_idx, trajectories_np in groups.items():
                seen[type_idx] += int(trajectories_np.shape[0])
                blocks[type_idx].append(torch.from_numpy(trajectories_np))
        else:
            _update_reservoir(reservoirs, filled, seen, rng, groups, max_per_type)

    if num_workers > 0:
        ctx = get_context("fork")
        with ctx.Pool(processes=num_workers, initializer=_worker_init) as pool:
            iterator = pool.imap(_extract_file_trajectory_groups, worker_args, chunksize=32)
            for groups in tqdm(iterator, total=len(file_chunks), desc="collect trajectory chunks"):
                consume(groups)
    else:
        for groups in tqdm(
            map(_extract_file_trajectory_groups, worker_args),
            total=len(file_chunks),
            desc="collect trajectory chunks",
        ):
            consume(groups)

    output: dict[str, torch.Tensor] = {}
    for type_idx, name in enumerate(AGENT_TYPE_NAMES):
        if use_all_trajectories:
            if len(blocks[type_idx]) == 0:
                raise RuntimeError(f"no valid {name} trajectories found in {train_cache_dir}")
            output[name] = torch.cat(blocks[type_idx], dim=0).contiguous()
            print(f"{name}: collected {output[name].shape[0]} valid trajectories")
            continue

        if filled[type_idx] == 0:
            raise RuntimeError(f"no valid {name} trajectories found in {train_cache_dir}")
        output[name] = reservoirs[type_idx][: filled[type_idx]].clone()
        print(f"{name}: sampled {output[name].shape[0]} of {seen[type_idx]} valid trajectories")
    return output


@torch.no_grad()
def nearest_anchor_assignment(
    trajectories: torch.Tensor,
    anchors: torch.Tensor,
    heading_weight: float,
    row_chunk_size: int,
    anchor_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign rows to anchors with the same distance used for UniMM matching."""

    if trajectories.ndim != 3 or trajectories.shape[-1] != 3:
        raise ValueError(f"trajectories must be [N, H, 3], got {tuple(trajectories.shape)}")
    if anchors.ndim != 3 or anchors.shape[-1] != 3:
        raise ValueError(f"anchors must be [K, H, 3], got {tuple(anchors.shape)}")
    if trajectories.shape[1] != anchors.shape[1]:
        raise ValueError(
            "trajectories and anchors must use the same horizon, "
            f"got {trajectories.shape[1]} and {anchors.shape[1]}"
        )
    if row_chunk_size < 1 or anchor_chunk_size < 1:
        raise ValueError("row_chunk_size and anchor_chunk_size must be positive")

    n_traj, horizon_steps = trajectories.shape[:2]
    num_anchors = anchors.shape[0]
    best_idx = torch.zeros(n_traj, dtype=torch.long, device=trajectories.device)
    best_dist = torch.full(
        (n_traj,),
        float("inf"),
        dtype=trajectories.dtype,
        device=trajectories.device,
    )

    anchor_pos = anchors[..., :2].flatten(1)
    anchor_pos_norm = anchor_pos.square().sum(dim=-1)
    anchor_head = anchors[..., 2]
    denom = float(horizon_steps)
    for row_start in range(0, n_traj, row_chunk_size):
        row_end = min(row_start + row_chunk_size, n_traj)
        rows = trajectories[row_start:row_end]
        row_pos = rows[..., :2].flatten(1)
        row_pos_norm = row_pos.square().sum(dim=-1)
        row_head = rows[..., 2]
        chunk_best = torch.full(
            (rows.shape[0],),
            float("inf"),
            dtype=trajectories.dtype,
            device=trajectories.device,
        )
        chunk_idx = torch.zeros(rows.shape[0], dtype=torch.long, device=trajectories.device)

        for anchor_start in range(0, num_anchors, anchor_chunk_size):
            anchor_end = min(anchor_start + anchor_chunk_size, num_anchors)
            pos_sum = (
                row_pos_norm[:, None]
                + anchor_pos_norm[anchor_start:anchor_end][None, :]
                - 2.0 * row_pos @ anchor_pos[anchor_start:anchor_end].T
            ).clamp_min_(0.0)
            head_diff = wrap_angle(anchor_head[anchor_start:anchor_end].unsqueeze(0) - row_head.unsqueeze(1))
            dist = (pos_sum + float(heading_weight) * head_diff.square().sum(dim=-1)) / denom
            anchor_best, anchor_idx = dist.min(dim=1)
            improved = anchor_best < chunk_best
            chunk_best = torch.where(improved, anchor_best, chunk_best)
            chunk_idx = torch.where(improved, anchor_idx + anchor_start, chunk_idx)

        best_dist[row_start:row_end] = chunk_best
        best_idx[row_start:row_end] = chunk_idx

    return best_idx, best_dist


@torch.no_grad()
def minibatch_kmeans(
    trajectories: torch.Tensor,
    num_clusters: int,
    num_iters: int,
    batch_size: int,
    heading_weight: float,
    seed: int,
    anchor_chunk_size: int = 256,
) -> torch.Tensor:
    generator = torch.Generator(device=trajectories.device)
    generator.manual_seed(seed)
    n_traj = trajectories.shape[0]
    if n_traj < num_clusters:
        repeats = (num_clusters + n_traj - 1) // n_traj
        trajectories = trajectories.repeat(repeats, 1, 1)[:num_clusters]
        n_traj = trajectories.shape[0]

    init_idx = torch.randperm(n_traj, generator=generator, device=trajectories.device)[:num_clusters]
    centroids = trajectories[init_idx].clone()
    counts = torch.zeros(num_clusters, dtype=torch.float32, device=trajectories.device)

    for _ in tqdm(range(num_iters), desc="mini-batch k-means"):
        batch_idx = torch.randint(
            0,
            n_traj,
            (min(batch_size, n_traj),),
            generator=generator,
            device=trajectories.device,
        )
        batch = trajectories[batch_idx]
        assign, _ = nearest_anchor_assignment(
            trajectories=batch,
            anchors=centroids,
            heading_weight=heading_weight,
            row_chunk_size=min(batch.shape[0], batch_size),
            anchor_chunk_size=min(anchor_chunk_size, num_clusters),
        )

        batch_counts = torch.bincount(assign, minlength=num_clusters).to(dtype=counts.dtype)
        active = batch_counts > 0
        if not bool(active.any()):
            continue

        pos_sum = torch.zeros(
            (num_clusters, trajectories.shape[1], 2),
            dtype=trajectories.dtype,
            device=trajectories.device,
        )
        sin_sum = torch.zeros(
            (num_clusters, trajectories.shape[1]),
            dtype=trajectories.dtype,
            device=trajectories.device,
        )
        cos_sum = torch.zeros_like(sin_sum)
        pos_sum.index_add_(0, assign, batch[:, :, :2])
        sin_sum.index_add_(0, assign, batch[:, :, 2].sin())
        cos_sum.index_add_(0, assign, batch[:, :, 2].cos())

        old_count = counts[active]
        new_count = old_count + batch_counts[active]
        centroids[active, :, :2] = (
            centroids[active, :, :2] * old_count[:, None, None]
            + pos_sum[active]
        ) / new_count[:, None, None].clamp_min(1.0)
        sin_mean = (
            centroids[active, :, 2].sin() * old_count[:, None]
            + sin_sum[active]
        ) / new_count[:, None].clamp_min(1.0)
        cos_mean = (
            centroids[active, :, 2].cos() * old_count[:, None]
            + cos_sum[active]
        ) / new_count[:, None].clamp_min(1.0)
        centroids[active, :, 2] = torch.atan2(sin_mean, cos_mean)
        counts[active] = new_count

    return centroids.contiguous()


@torch.no_grad()
def lloyd_refine_kmeans(
    trajectories: torch.Tensor,
    centroids: torch.Tensor,
    num_iters: int,
    heading_weight: float,
    row_chunk_size: int,
    anchor_chunk_size: int,
    tol: float,
    reinit_empty_clusters: bool = True,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Run full-data Lloyd updates with chunked assignment to keep memory bounded."""

    history: list[dict[str, float]] = []
    if num_iters <= 0:
        return centroids.contiguous(), history

    num_clusters, horizon_steps = centroids.shape[:2]
    prev_objective = math.inf
    for iter_idx in tqdm(range(num_iters), desc="full Lloyd refinement"):
        counts = torch.zeros(num_clusters, dtype=torch.float32, device=trajectories.device)
        pos_sum = torch.zeros(
            (num_clusters, horizon_steps, 2),
            dtype=trajectories.dtype,
            device=trajectories.device,
        )
        sin_sum = torch.zeros(
            (num_clusters, horizon_steps),
            dtype=trajectories.dtype,
            device=trajectories.device,
        )
        cos_sum = torch.zeros_like(sin_sum)

        assign_all, dist_all = nearest_anchor_assignment(
            trajectories=trajectories,
            anchors=centroids,
            heading_weight=heading_weight,
            row_chunk_size=row_chunk_size,
            anchor_chunk_size=anchor_chunk_size,
        )
        for start in range(0, trajectories.shape[0], row_chunk_size):
            chunk = trajectories[start : start + row_chunk_size]
            assign = assign_all[start : start + row_chunk_size]
            counts.index_add_(0, assign, torch.ones_like(assign, dtype=counts.dtype))
            pos_sum.index_add_(0, assign, chunk[..., :2])
            sin_sum.index_add_(0, assign, chunk[..., 2].sin())
            cos_sum.index_add_(0, assign, chunk[..., 2].cos())

        active = counts > 0
        if bool(active.any()):
            centroids[active, :, :2] = pos_sum[active] / counts[active, None, None].clamp_min(1.0)
            centroids[active, :, 2] = torch.atan2(
                sin_sum[active] / counts[active, None].clamp_min(1.0),
                cos_sum[active] / counts[active, None].clamp_min(1.0),
            )
        empty = torch.nonzero(~active, as_tuple=False).flatten()
        reinitialized = 0
        if reinit_empty_clusters and empty.numel() > 0:
            reinitialized = min(int(empty.numel()), int(trajectories.shape[0]))
            farthest = torch.topk(dist_all, k=reinitialized, largest=True).indices
            centroids[empty[:reinitialized]] = trajectories[farthest]

        objective = float(dist_all.double().mean().item())
        empty_clusters = int((~active).sum().item())
        if math.isfinite(prev_objective):
            rel_improvement = max(0.0, prev_objective - objective) / max(abs(prev_objective), 1e-12)
        else:
            rel_improvement = math.inf
        history.append(
            {
                "iter": float(iter_idx + 1),
                "objective": objective,
                "rel_improvement": rel_improvement,
                "empty_clusters": float(empty_clusters),
                "reinitialized_empty_clusters": float(reinitialized),
            }
        )
        print(
            "full Lloyd iter "
            f"{iter_idx + 1}: objective={objective:.8f}, "
            f"rel_improvement={rel_improvement:.6g}, empty_clusters={empty_clusters}, "
            f"reinitialized={reinitialized}",
            flush=True,
        )

        if iter_idx > 0 and empty_clusters == 0 and rel_improvement <= tol:
            break
        prev_objective = objective

    return centroids.contiguous(), history


def compute_threshold(
    trajectories: torch.Tensor,
    anchors: torch.Tensor,
    match_steps: int,
    quantile: float,
    heading_weight: float,
    row_chunk_size: int,
) -> float:
    errors = []
    for start in range(0, trajectories.shape[0], row_chunk_size):
        chunk = trajectories[start : start + row_chunk_size, :match_steps]
        dist = anchor_distance(
            anchors=anchors[:, :match_steps],
            target_local=chunk,
            valid=torch.ones(chunk.shape[:2], dtype=torch.bool, device=chunk.device),
            heading_weight=heading_weight,
        )
        errors.append(dist.min(dim=1).values.cpu())
    return float(torch.cat(errors).quantile(quantile).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UniMM 8s category anchors from WOMD training cache.")
    parser.add_argument("--train-cache-dir", required=True)
    parser.add_argument("--output", default="src/unimm/anchors/unimm_anchors_8s_k2048.pkl")
    parser.add_argument("--num-anchors", type=int, default=2048)
    parser.add_argument("--horizon-steps", type=int, default=80)
    parser.add_argument("--match-steps", type=int, default=5)
    parser.add_argument("--start-step", type=int, default=10)
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=0,
        help="Maximum trajectories per agent type for k-means input. Use 0 to keep all valid training trajectories.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Parallel pickle readers for trajectory collection.",
    )
    parser.add_argument(
        "--collect-file-chunk-size",
        type=int,
        default=64,
        help="Number of cache pickle files processed per worker task.",
    )
    parser.add_argument("--kmeans-iters", type=int, default=200)
    parser.add_argument("--kmeans-batch-size", type=int, default=8192)
    parser.add_argument(
        "--lloyd-iters",
        type=int,
        default=20,
        help="Full-training-data Lloyd refinement sweeps after mini-batch initialization. Use 0 to disable.",
    )
    parser.add_argument(
        "--lloyd-row-chunk-size",
        type=int,
        default=2048,
        help="Trajectory rows per exact full-data assignment chunk.",
    )
    parser.add_argument(
        "--lloyd-anchor-chunk-size",
        type=int,
        default=256,
        help="Anchor columns per exact full-data assignment chunk.",
    )
    parser.add_argument(
        "--lloyd-tol",
        type=float,
        default=1e-4,
        help="Stop full Lloyd refinement when relative objective improvement is below this value.",
    )
    parser.add_argument(
        "--no-reinit-empty-clusters",
        action="store_true",
        help="Do not refill empty Lloyd clusters from high-error trajectories.",
    )
    parser.add_argument("--heading-weight", type=float, default=1.0)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for k-means and threshold computation: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_anchors < 1:
        raise ValueError("--num-anchors must be positive")
    if args.horizon_steps < 1 or args.match_steps < 1:
        raise ValueError("--horizon-steps and --match-steps must be positive")
    if args.match_steps > args.horizon_steps:
        raise ValueError("--match-steps cannot exceed --horizon-steps")
    if args.max_per_type < 0:
        raise ValueError("--max-per-type cannot be negative")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.collect_file_chunk_size < 1:
        raise ValueError("--collect-file-chunk-size must be positive")
    if args.kmeans_batch_size < 1:
        raise ValueError("--kmeans-batch-size must be positive")
    if args.lloyd_iters < 0:
        raise ValueError("--lloyd-iters cannot be negative")
    if args.lloyd_row_chunk_size < 1 or args.lloyd_anchor_chunk_size < 1:
        raise ValueError("--lloyd chunk sizes must be positive")
    if args.lloyd_tol < 0:
        raise ValueError("--lloyd-tol cannot be negative")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"using k-means device: {device}")
    max_per_type = None if args.max_per_type == 0 else args.max_per_type

    train_cache_dir = Path(args.train_cache_dir)
    trajectories = collect_training_trajectories(
        train_cache_dir=train_cache_dir,
        max_per_type=max_per_type,
        horizon_steps=args.horizon_steps,
        start_step=args.start_step,
        seed=args.seed,
        num_workers=args.num_workers,
        collect_file_chunk_size=args.collect_file_chunk_size,
    )
    anchors: dict[str, torch.Tensor] = {}
    thresholds: dict[str, float] = {}
    lloyd_history: dict[str, list[dict[str, float]]] = {}
    for type_idx, name in enumerate(AGENT_TYPE_NAMES):
        print(f"clustering {name}")
        trajectories_for_type = trajectories[name].to(device=device, non_blocking=True)
        init_anchors = minibatch_kmeans(
            trajectories=trajectories_for_type,
            num_clusters=args.num_anchors,
            num_iters=args.kmeans_iters,
            batch_size=args.kmeans_batch_size,
            heading_weight=args.heading_weight,
            seed=args.seed + type_idx,
            anchor_chunk_size=args.lloyd_anchor_chunk_size,
        )
        anchors[name], lloyd_history[name] = lloyd_refine_kmeans(
            trajectories=trajectories_for_type,
            centroids=init_anchors,
            num_iters=args.lloyd_iters,
            heading_weight=args.heading_weight,
            row_chunk_size=args.lloyd_row_chunk_size,
            anchor_chunk_size=args.lloyd_anchor_chunk_size,
            tol=args.lloyd_tol,
            reinit_empty_clusters=not args.no_reinit_empty_clusters,
        )
        thresholds[name] = compute_threshold(
            trajectories=trajectories_for_type,
            anchors=anchors[name],
            match_steps=args.match_steps,
            quantile=args.threshold_quantile,
            heading_weight=args.heading_weight,
            row_chunk_size=args.kmeans_batch_size,
        )
        anchors[name] = anchors[name].cpu()
        print(f"{name}: posterior_error_threshold={thresholds[name]:.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "unimm_anchor_bank_v1",
        "anchors": {name: anchors[name].numpy() for name in AGENT_TYPE_NAMES},
        "posterior_error_threshold": thresholds,
        "metadata": {
            **vars(args),
            "distance_metric": "mean(pos_sq + heading_weight * wrap_angle(heading_diff)^2)",
            "kmeans_init": "mini_batch",
            "kmeans_refinement": "full_training_data_lloyd",
            "empty_cluster_policy": (
                "reinitialize_from_high_error_trajectories"
                if not args.no_reinit_empty_clusters
                else "keep_previous_centroid"
            ),
            "lloyd_history": lloyd_history,
        },
    }
    with output.open("wb") as handle:
        pickle.dump(payload, handle)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
