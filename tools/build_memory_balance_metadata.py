#!/usr/bin/env python3
"""Prebuild the train memory-balance metadata cache.

This tool reads SMART cache pkl files and writes only the lightweight counts
used by MemoryBalancedDistributedBatchSampler. It does not modify dataset cache
files.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.smart.datamodules.memory_balanced_batch_sampler import (  # noqa: E402
    load_or_build_memory_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the metadata cache used by memory-balanced train batches."
    )
    parser.add_argument(
        "--raw-dir",
        required=True,
        help="Directory containing train SMART cache pkl files, e.g. $CACHE_ROOT/training.",
    )
    parser.add_argument(
        "--cache-path",
        required=True,
        help="Output metadata cache path used by data.train_memory_balance_metadata_cache.",
    )
    parser.add_argument("--pattern", default="*.pkl")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing metadata cache before rebuilding it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir).expanduser()
    cache_path = Path(args.cache_path).expanduser()
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"raw dir does not exist: {raw_dir}")
    raw_paths = sorted(str(path) for path in raw_dir.glob(args.pattern))
    if not raw_paths:
        raise FileNotFoundError(f"no files matched {args.pattern!r} under {raw_dir}")

    if args.force and cache_path.exists():
        cache_path.unlink()

    start_time = time.perf_counter()
    metadata = load_or_build_memory_metadata(
        raw_paths,
        cache_path=str(cache_path),
        num_workers=args.num_workers,
        build_on_missing=True,
    )
    elapsed_sec = time.perf_counter() - start_time
    print(
        "memory-balance metadata ready: "
        f"samples={len(metadata)} cache={cache_path} elapsed_sec={elapsed_sec:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
