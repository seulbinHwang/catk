#!/usr/bin/env python
"""Precompute deterministic TrajTok agent token targets as sidecar files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.smart.datamodules.target_builder import WaymoTargetBuilderTrain
from src.smart.datasets import MultiDataset
from src.smart.tokens.token_processor import (
    AGENT_TOKEN_SIDECAR_FIELDS,
    AGENT_TOKEN_SIDECAR_VERSION,
    TokenProcessor,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scenario_ids_from_batch(batch) -> list[str]:
    scenario_ids = getattr(batch, "scenario_id", None)
    if scenario_ids is None:
        scenario_ids = batch.get("scenario_id", None)
    if scenario_ids is None:
        raise RuntimeError("Batch is missing scenario_id; cannot write sidecars.")
    if isinstance(scenario_ids, str):
        return [scenario_ids]
    if isinstance(scenario_ids, tuple):
        return [str(x) for x in scenario_ids]
    if isinstance(scenario_ids, list):
        return [str(x) for x in scenario_ids]
    return [str(x) for x in list(scenario_ids)]


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", default="training")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-max-num", type=int, default=32)
    parser.add_argument("--agent-token-file", default="trajtok_vocab.pkl")
    parser.add_argument("--map-token-file", default="map_traj_token5.pkl")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    cache_root = Path(args.cache_root)
    raw_dir = cache_root / args.split
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else cache_root / "trajtok_agent_token_sidecar" / args.split
    )

    module_dir = Path(__file__).resolve().parents[1] / "src" / "smart" / "tokens"
    agent_token_path = module_dir / args.agent_token_file
    metadata = {
        "version": AGENT_TOKEN_SIDECAR_VERSION,
        "split": args.split,
        "agent_token_file": args.agent_token_file,
        "agent_token_sha256": sha256_file(agent_token_path),
        "train_max_num": int(args.train_max_num),
        "target_builder": "WaymoTargetBuilderTrain",
        "random_scene_scale_config": None,
        "random_time_shift_config": None,
        "fields": list(AGENT_TOKEN_SIDECAR_FIELDS),
    }
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / "metadata.json"
        tmp_path = metadata_path.with_name(f"{metadata_path.name}.tmp.{os.getpid()}")
        tmp_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, metadata_path)
        print(f"[sidecar] metadata: {metadata_path}", flush=True)
    if distributed:
        dist.barrier()

    transform = WaymoTargetBuilderTrain(max_num=args.train_max_num)
    dataset = MultiDataset(str(raw_dir), transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    processor = TokenProcessor(
        map_token_file=args.map_token_file,
        agent_token_file=args.agent_token_file,
    ).to(device)
    processor.train()

    start_time = time.time()
    written = 0
    skipped = 0
    for batch_idx, batch in enumerate(loader):
        if args.limit_batches > 0 and batch_idx >= args.limit_batches:
            break
        scenario_ids = scenario_ids_from_batch(batch)
        output_paths = [output_dir / f"{scenario_id}.pt" for scenario_id in scenario_ids]
        if not args.overwrite and all(path.exists() for path in output_paths):
            skipped += len(output_paths)
            continue

        batch = batch.to(device, non_blocking=True)
        with torch.no_grad():
            tokenized_agent = processor.tokenize_agent(batch)
        agent_batch = batch["agent"]["batch"].detach().cpu()
        for graph_idx, scenario_id in enumerate(scenario_ids):
            output_path = output_dir / f"{scenario_id}.pt"
            if output_path.exists() and not args.overwrite:
                skipped += 1
                continue
            mask = agent_batch == graph_idx
            agent_payload = {
                field: tokenized_agent[field].detach().cpu()[mask].contiguous()
                for field in AGENT_TOKEN_SIDECAR_FIELDS
            }
            payload = {
                "metadata": metadata,
                "scenario_id": scenario_id,
                "num_agents": int(mask.sum().item()),
                "agent": agent_payload,
            }
            atomic_torch_save(payload, output_path)
            written += 1

        if rank == 0 and (batch_idx + 1) % 50 == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                "[sidecar] "
                f"rank={rank} batch={batch_idx + 1}/{len(loader)} "
                f"written={written} skipped={skipped} "
                f"rate={(written + skipped) / elapsed:.1f} scenarios/s",
                flush=True,
            )

    if distributed:
        dist.barrier()
    elapsed = max(time.time() - start_time, 1e-6)
    print(
        f"[sidecar] rank={rank} done written={written} skipped={skipped} "
        f"elapsed={elapsed:.1f}s rate={(written + skipped) / elapsed:.1f} scenarios/s",
        flush=True,
    )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
