from __future__ import annotations

import hashlib
import math
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
from torch.utils.data import Sampler


METADATA_VERSION = 1


@dataclass(frozen=True)
class SampleMemoryMetadata:
    """Lightweight per-cache-file fields used only for train batch ordering."""

    agent_count: int
    current_valid_agent_count: int
    valid_agent_step_count: int
    map_count: int


def _shape0(value) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    try:
        return int(len(value))
    except TypeError:
        return 0


def _sum_bool(value) -> int:
    if value is None:
        return 0
    try:
        return int(value.bool().sum().item())
    except AttributeError:
        try:
            return int(value.sum())
        except TypeError:
            return 0


def _get_storage(data, key: str):
    try:
        return data[key]
    except (KeyError, TypeError):
        return None


def _get_field(storage, key: str):
    if storage is None:
        return None
    try:
        return storage[key]
    except (KeyError, TypeError):
        return getattr(storage, key, None)


def _extract_sample_metadata(path: str) -> SampleMemoryMetadata:
    with open(path, "rb") as handle:
        data = pickle.load(handle)

    agent = _get_storage(data, "agent")
    agent_type = _get_field(agent, "type")
    valid_mask = _get_field(agent, "valid_mask")
    position = _get_field(agent, "position")

    agent_count = _shape0(agent_type) or _shape0(valid_mask) or _shape0(position)
    valid_agent_step_count = _sum_bool(valid_mask)
    current_valid_agent_count = 0
    if valid_mask is not None:
        try:
            step_current = min(10, int(valid_mask.shape[1]) - 1)
            if step_current >= 0:
                current_valid_agent_count = _sum_bool(valid_mask[:, step_current])
        except (AttributeError, IndexError, TypeError):
            current_valid_agent_count = 0

    map_count = 0
    map_save = _get_storage(data, "map_save")
    traj_pos = _get_field(map_save, "traj_pos")
    if traj_pos is not None:
        map_count = _shape0(traj_pos)
    else:
        for storage_name in ("map_polygon", "map_point"):
            storage = _get_storage(data, storage_name)
            pos = _get_field(storage, "position")
            map_count = max(map_count, _shape0(pos))

    return SampleMemoryMetadata(
        agent_count=int(agent_count),
        current_valid_agent_count=int(current_valid_agent_count),
        valid_agent_step_count=int(valid_agent_step_count),
        map_count=int(map_count),
    )


def fingerprint_raw_paths(raw_paths: Sequence[str]) -> str:
    """Fingerprint dataset identity without baking machine-specific prefixes."""

    digest = hashlib.sha1()
    for path in raw_paths:
        digest.update(Path(path).name.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _default_metadata_cache_path(raw_paths: Sequence[str]) -> Path:
    raw_dir = Path(raw_paths[0]).parent if raw_paths else Path(".")
    filename = f"{raw_dir.name}_memory_balance_v{METADATA_VERSION}.pt"
    return raw_dir.parent / ".catk_metadata" / filename


def _build_metadata(raw_paths: Sequence[str], num_workers: int) -> list[SampleMemoryMetadata]:
    if num_workers <= 1:
        return [_extract_sample_metadata(path) for path in raw_paths]
    with ThreadPoolExecutor(max_workers=int(num_workers)) as executor:
        return list(executor.map(_extract_sample_metadata, raw_paths, chunksize=128))


def _load_metadata_file(
    cache_path: Path, raw_paths: Sequence[str]
) -> list[SampleMemoryMetadata] | None:
    if not cache_path.is_file():
        return None
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(cache_path, map_location="cpu")
    expected_fingerprint = fingerprint_raw_paths(raw_paths)
    if payload.get("version") != METADATA_VERSION:
        return None
    if payload.get("num_samples") != len(raw_paths):
        return None
    if payload.get("fingerprint") != expected_fingerprint:
        return None

    return [
        SampleMemoryMetadata(
            agent_count=int(agent_count),
            current_valid_agent_count=int(current_valid_agent_count),
            valid_agent_step_count=int(valid_agent_step_count),
            map_count=int(map_count),
        )
        for agent_count, current_valid_agent_count, valid_agent_step_count, map_count in zip(
            payload["agent_count"],
            payload["current_valid_agent_count"],
            payload["valid_agent_step_count"],
            payload["map_count"],
        )
    ]


def load_or_build_memory_metadata(
    raw_paths: Sequence[str],
    *,
    cache_path: str | None,
    num_workers: int,
    build_on_missing: bool,
    lock_timeout_sec: int = 7200,
) -> list[SampleMemoryMetadata]:
    """Load cached sample-size metadata, building it once when allowed.

    The cache intentionally stores only counts and a basename fingerprint. It
    never mutates the dataset cache files themselves.
    """

    resolved_cache_path = (
        Path(cache_path).expanduser()
        if cache_path not in (None, "")
        else _default_metadata_cache_path(raw_paths)
    )
    loaded = _load_metadata_file(resolved_cache_path, raw_paths)
    if loaded is not None:
        return loaded
    if not build_on_missing:
        raise FileNotFoundError(
            "Train memory-balance metadata cache is missing or stale: "
            f"{resolved_cache_path}. Set train_memory_balance_build_on_missing=true "
            "or build the cache once before training."
        )

    resolved_cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = resolved_cache_path.with_suffix(resolved_cache_path.suffix + ".lock")
    start_time = time.monotonic()
    owns_lock = False
    while True:
        try:
            os.mkdir(lock_dir)
            owns_lock = True
            break
        except FileExistsError:
            loaded = _load_metadata_file(resolved_cache_path, raw_paths)
            if loaded is not None:
                return loaded
            if time.monotonic() - start_time > lock_timeout_sec:
                raise TimeoutError(
                    "Timed out waiting for train memory-balance metadata cache: "
                    f"{resolved_cache_path}"
                )
            time.sleep(10)

    try:
        loaded = _load_metadata_file(resolved_cache_path, raw_paths)
        if loaded is not None:
            return loaded

        metadata = _build_metadata(raw_paths, num_workers=num_workers)
        payload = {
            "version": METADATA_VERSION,
            "num_samples": len(raw_paths),
            "fingerprint": fingerprint_raw_paths(raw_paths),
            "agent_count": [item.agent_count for item in metadata],
            "current_valid_agent_count": [item.current_valid_agent_count for item in metadata],
            "valid_agent_step_count": [item.valid_agent_step_count for item in metadata],
            "map_count": [item.map_count for item in metadata],
        }
        tmp_path = resolved_cache_path.with_suffix(
            resolved_cache_path.suffix + f".tmp.{os.getpid()}"
        )
        torch.save(payload, tmp_path)
        os.replace(tmp_path, resolved_cache_path)
        return metadata
    finally:
        if owns_lock:
            try:
                os.rmdir(lock_dir)
            except OSError:
                pass


def memory_balance_weights(
    metadata: Iterable[SampleMemoryMetadata],
    *,
    agent_weight: float,
    current_valid_agent_weight: float,
    valid_agent_step_weight: float,
    map_weight: float,
) -> torch.Tensor:
    values = []
    for item in metadata:
        values.append(
            float(agent_weight) * item.agent_count
            + float(current_valid_agent_weight) * item.current_valid_agent_count
            + float(valid_agent_step_weight) * item.valid_agent_step_count
            + float(map_weight) * item.map_count
        )
    return torch.as_tensor(values, dtype=torch.float64)


class MemoryBalancedDistributedBatchSampler(Sampler[list[int]]):
    """Distributed batch sampler that spreads high-memory scenes across batches."""

    def __init__(
        self,
        *,
        sample_weight: torch.Tensor,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int = 0,
        fraction: float = 1.0,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}.")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}.")
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
        if sample_weight.dim() != 1:
            raise ValueError("sample_weight must be a 1D tensor.")
        if sample_weight.numel() == 0:
            raise ValueError("sample_weight must contain at least one sample.")

        self.sample_weight = sample_weight.detach().cpu().to(torch.float64)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.fraction = float(fraction)
        self.epoch = 0

        self.dataset_size = int(self.sample_weight.numel())
        self.subset_size = max(1, int(math.floor(self.dataset_size * self.fraction)))
        self.global_batch_size = self.batch_size * self.num_replicas
        self.num_global_batches = int(math.ceil(self.subset_size / self.global_batch_size))
        self.total_size = self.num_global_batches * self.global_batch_size

    def __len__(self) -> int:
        return self.num_global_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _selected_indices(self, generator: torch.Generator) -> torch.Tensor:
        if self.shuffle:
            indices = torch.randperm(self.dataset_size, generator=generator)
        else:
            indices = torch.arange(self.dataset_size)
        indices = indices[: self.subset_size]
        if indices.numel() < self.total_size:
            pad_count = self.total_size - int(indices.numel())
            pad_source = indices[
                torch.randint(indices.numel(), (pad_count,), generator=generator)
            ]
            indices = torch.cat([indices, pad_source], dim=0)
        return indices

    def _balanced_bins(self, generator: torch.Generator) -> torch.Tensor:
        indices = self._selected_indices(generator)
        weights = self.sample_weight[indices]
        if self.shuffle:
            jitter = torch.rand(weights.shape, generator=generator, dtype=weights.dtype)
            order = torch.argsort(weights + jitter * 1.0e-6, descending=True, stable=True)
        else:
            order = torch.argsort(weights, descending=True, stable=True)
        indices = indices[order]

        num_bins = self.num_global_batches * self.num_replicas
        strata = indices.view(self.batch_size, num_bins)
        for row_idx in range(self.batch_size):
            if self.shuffle:
                perm = torch.randperm(num_bins, generator=generator)
            else:
                perm = torch.arange(num_bins)
            strata[row_idx] = strata[row_idx, perm]
        return strata.t().contiguous()

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        bins = self._balanced_bins(generator)
        start = self.rank
        stop = self.num_global_batches * self.num_replicas
        for bin_idx in range(start, stop, self.num_replicas):
            yield bins[bin_idx].tolist()
