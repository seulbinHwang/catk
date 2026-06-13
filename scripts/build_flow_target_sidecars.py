#!/usr/bin/env python3
"""Precompute deterministic Flow token/target sidecars for SMART cache samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch_geometric.data import Batch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.smart.datamodules.scalable_datamodule import build_train_agent_target_builder
from src.smart.datasets import MultiDataset
from src.smart.tokens.flow_token_processor import FlowTokenProcessor

SIDECAR_MANIFEST_VERSION = 1
SIDECAR_MANIFEST_NAME = "manifest.json"
SIDECAR_SHARD_MARKER_DIR = ".manifest_shards"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", default="training")
    parser.add_argument("--sidecar-dir", required=True)
    parser.add_argument("--experiment", default="pre_bc_flow_2x4_h100")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for this worker. 'auto' maps torchrun LOCAL_RANK to cuda:<rank>.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=0,
        help="Total dataset shards. Defaults to torchrun WORLD_SIZE, or 1 outside torchrun.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=-1,
        help="Shard index for this worker. Defaults to torchrun RANK, or 0 outside torchrun.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--status-every", type=int, default=100)
    return parser.parse_known_args()


def raw_path_list_hash(raw_paths: Sequence[str]) -> str:
    """Fingerprint the exact raw SMART cache path list used for prebuild."""

    digest = hashlib.sha1()
    for path in raw_paths:
        resolved_path = Path(path).expanduser().resolve(strict=False)
        digest.update(str(resolved_path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def build_signature(
    *,
    sidecar_root: Path,
    split: str,
    sample_count: int,
    raw_paths_hash: str,
    fingerprint: str,
    num_shards: int,
    max_samples: int,
) -> str:
    payload = {
        "sidecar_root": sidecar_root.expanduser().resolve(strict=False).as_posix(),
        "split": str(split),
        "sample_count": int(sample_count),
        "raw_path_list_hash": str(raw_paths_hash),
        "fingerprint": str(fingerprint),
        "num_shards": int(num_shards),
        "max_samples": int(max_samples),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def manifest_matches(
    manifest: dict[str, Any] | None,
    *,
    split: str,
    sample_count: int,
    raw_paths_hash: str,
    fingerprint: str,
    max_samples: int,
) -> bool:
    if not manifest:
        return False
    if manifest.get("version") != SIDECAR_MANIFEST_VERSION:
        return False
    if manifest.get("status") != "complete":
        return False
    if manifest.get("split") != str(split):
        return False
    if int(manifest.get("sample_count", -1)) != int(sample_count):
        return False
    if manifest.get("raw_path_list_hash") != str(raw_paths_hash):
        return False
    if manifest.get("fingerprint") != str(fingerprint):
        return False
    if int(manifest.get("max_samples", -1)) != int(max_samples):
        return False
    completed = manifest.get("completed_shards")
    if not isinstance(completed, list) or not completed:
        return False
    covered = sum(int(item.get("processed", 0)) for item in completed if isinstance(item, dict))
    return covered == int(sample_count)


def shard_marker_path(sidecar_root: Path, signature: str, shard_index: int) -> Path:
    return (
        sidecar_root
        / SIDECAR_SHARD_MARKER_DIR
        / str(signature)
        / f"shard_{int(shard_index):05d}.json"
    )


def clear_shard_markers(sidecar_root: Path, signature: str) -> None:
    marker_dir = sidecar_root / SIDECAR_SHARD_MARKER_DIR / str(signature)
    if not marker_dir.exists():
        return
    for path in marker_dir.glob("shard_*.json"):
        try:
            path.unlink()
        except OSError:
            pass


def read_completed_shards(
    *,
    sidecar_root: Path,
    signature: str,
    num_shards: int,
) -> list[dict[str, Any]] | None:
    completed: list[dict[str, Any]] = []
    for shard_idx in range(int(num_shards)):
        marker = read_json(shard_marker_path(sidecar_root, signature, shard_idx))
        if not marker:
            return None
        if marker.get("build_signature") != signature:
            return None
        if int(marker.get("num_shards", -1)) != int(num_shards):
            return None
        if int(marker.get("shard_index", -1)) != int(shard_idx):
            return None
        completed.append(marker)
    return completed


def try_write_complete_manifest(
    *,
    sidecar_root: Path,
    split: str,
    sample_count: int,
    raw_paths_hash: str,
    fingerprint: str,
    max_samples: int,
    num_shards: int,
    signature: str,
) -> bool:
    completed_shards = read_completed_shards(
        sidecar_root=sidecar_root,
        signature=signature,
        num_shards=num_shards,
    )
    if completed_shards is None:
        return False
    covered = sum(int(item.get("processed", 0)) for item in completed_shards)
    if covered != int(sample_count):
        return False

    manifest = {
        "version": SIDECAR_MANIFEST_VERSION,
        "status": "complete",
        "split": str(split),
        "sample_count": int(sample_count),
        "raw_path_list_hash": str(raw_paths_hash),
        "fingerprint": str(fingerprint),
        "max_samples": int(max_samples),
        "num_shards": int(num_shards),
        "build_signature": str(signature),
        "covered_sample_count": int(covered),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_shards": completed_shards,
    }
    write_json_atomic(sidecar_root / SIDECAR_MANIFEST_NAME, manifest)
    return True


def maybe_init_process_group() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1 or not dist.is_available():
        return False
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    return True


def barrier_if_needed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.barrier()


def scenario_id_from_data(data) -> str:
    scenario_id = getattr(data, "scenario_id", None)
    if scenario_id is None and isinstance(data, dict):
        scenario_id = data.get("scenario_id")
    if isinstance(scenario_id, (list, tuple)):
        if len(scenario_id) != 1:
            raise ValueError(f"Expected one scenario id, got {scenario_id!r}")
        scenario_id = scenario_id[0]
    if scenario_id is None:
        raise KeyError("Sample does not contain scenario_id.")
    return str(scenario_id)


def resolve_shard(args: argparse.Namespace) -> tuple[int, int]:
    num_shards = int(args.num_shards)
    shard_index = int(args.shard_index)
    if num_shards <= 0:
        num_shards = int(os.environ.get("WORLD_SIZE", "1"))
    if shard_index < 0:
        shard_index = int(os.environ.get("RANK", "0"))
    if num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}.")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"--shard-index must be in [0, {num_shards}), got {shard_index}."
        )
    return num_shards, shard_index


def resolve_device(requested: str) -> torch.device:
    requested = str(requested)
    if requested == "auto":
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            local_rank = os.environ.get("LOCAL_RANK")
            if local_rank is not None:
                return torch.device(f"cuda:{int(local_rank)}")
            return torch.device("cuda")
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    args, extra_overrides = parse_args()
    num_shards, shard_index = resolve_shard(args)
    distributed = maybe_init_process_group()
    config_dir = (Path(__file__).resolve().parents[1] / "configs").as_posix()
    overrides = [
        f"experiment={args.experiment}",
        f"paths.cache_root={args.cache_root}",
        f"model.model_config.token_processor.flow_target_sidecar_dir={args.sidecar_dir}",
        "model.model_config.token_processor.flow_target_sidecar_read=false",
        "model.model_config.token_processor.flow_target_sidecar_write=true",
    ]
    overrides.extend(extra_overrides)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="run", overrides=overrides)

    token_processor_cfg = OmegaConf.to_container(
        cfg.model.model_config.token_processor,
        resolve=True,
    )
    processor = FlowTokenProcessor(**token_processor_cfg)
    processor.train()

    raw_dir = Path(args.cache_root) / args.split
    transform = build_train_agent_target_builder(
        train_max_num=int(cfg.data.train_max_num),
        train_use_eval_agent_selection=bool(cfg.data.train_use_eval_agent_selection),
    )
    dataset = MultiDataset(raw_dir.as_posix(), transform)
    total = len(dataset) if args.max_samples <= 0 else min(int(args.max_samples), len(dataset))
    paths_hash = raw_path_list_hash(dataset.raw_paths[:total])
    shard_indices = range(shard_index, total, num_shards)
    sidecar_root = processor._flow_target_sidecar_root()
    sidecar_root.mkdir(parents=True, exist_ok=True)
    signature = build_signature(
        sidecar_root=sidecar_root,
        split=args.split,
        sample_count=total,
        raw_paths_hash=paths_hash,
        fingerprint=processor._flow_target_sidecar_fingerprint,
        num_shards=num_shards,
        max_samples=int(args.max_samples),
    )
    manifest_path = sidecar_root / SIDECAR_MANIFEST_NAME
    if not args.overwrite and manifest_matches(
        read_json(manifest_path),
        split=args.split,
        sample_count=total,
        raw_paths_hash=paths_hash,
        fingerprint=processor._flow_target_sidecar_fingerprint,
        max_samples=int(args.max_samples),
    ):
        print(
            f"[sidecar] skip complete manifest shard={shard_index}/{num_shards} "
            f"total={total} root={sidecar_root} manifest={manifest_path}",
            flush=True,
        )
        barrier_if_needed(distributed)
        return

    if shard_index == 0:
        clear_shard_markers(sidecar_root, signature)
    barrier_if_needed(distributed)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    processor.to(device)

    started = time.time()
    processed = 0
    written = 0
    skipped = 0
    print(
        f"[sidecar] shard={shard_index}/{num_shards} total={total} "
        f"device={device} root={sidecar_root}",
        flush=True,
    )
    for idx in shard_indices:
        data = dataset[idx]
        scenario_id = scenario_id_from_data(data)
        sidecar_path = processor._sidecar_path_for_scenario(scenario_id)
        if sidecar_path.exists() and not args.overwrite:
            skipped += 1
        else:
            data = Batch.from_data_list([data]).to(device)
            with torch.no_grad():
                processor(data)
            written += 1
        processed += 1
        if args.status_every > 0 and processed % int(args.status_every) == 0:
            elapsed = time.time() - started
            rate = processed / max(elapsed, 1e-6)
            print(
                f"[sidecar] shard={shard_index}/{num_shards} processed={processed} "
                f"last_index={idx + 1}/{total} written={written} skipped={skipped} "
                f"rate={rate:.2f} samples/s root={sidecar_root}",
                flush=True,
            )

    elapsed = time.time() - started
    marker = {
        "version": SIDECAR_MANIFEST_VERSION,
        "build_signature": signature,
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "processed": int(processed),
        "written": int(written),
        "skipped": int(skipped),
        "elapsed_sec": float(elapsed),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_atomic(shard_marker_path(sidecar_root, signature, shard_index), marker)
    barrier_if_needed(distributed)
    if shard_index == 0 and try_write_complete_manifest(
        sidecar_root=sidecar_root,
        split=args.split,
        sample_count=total,
        raw_paths_hash=paths_hash,
        fingerprint=processor._flow_target_sidecar_fingerprint,
        max_samples=int(args.max_samples),
        num_shards=num_shards,
        signature=signature,
    ):
        print(
            f"[sidecar] complete manifest written root={sidecar_root} "
            f"manifest={manifest_path}",
            flush=True,
        )
    barrier_if_needed(distributed)
    print(
        f"[sidecar] done shard={shard_index}/{num_shards} processed={processed} "
        f"total={total} written={written} skipped={skipped} elapsed={elapsed:.1f}s "
        f"root={sidecar_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
