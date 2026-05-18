import math
import os
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator, Sequence

from torch.utils.data import Sampler


_METADATA_VERSION = 1


def _get_nested_shape0(data: dict, *keys: str) -> int:
    value = data
    for key in keys:
        value = value[key]
    return int(value.shape[0])


def _read_raw_sample_metadata(raw_path: str) -> dict[str, int | str]:
    with open(raw_path, "rb") as handle:
        data = pickle.load(handle)

    agent_count = _get_nested_shape0(data, "agent", "position")
    valid_agent_steps = int(data["agent"]["valid_mask"].sum().item())
    map_point_count = 0
    if "pt_token" in data and "position" in data["pt_token"]:
        map_point_count = _get_nested_shape0(data, "pt_token", "position")
    elif "map_save" in data and "traj_pos" in data["map_save"]:
        map_point_count = _get_nested_shape0(data, "map_save", "traj_pos")

    return {
        "name": Path(raw_path).name,
        "agent_count": agent_count,
        "valid_agent_steps": valid_agent_steps,
        "map_point_count": map_point_count,
        "file_size": int(os.path.getsize(raw_path)),
    }


def _metadata_matches_paths(metadata: dict, raw_paths: Sequence[str]) -> bool:
    entries = metadata.get("entries")
    if metadata.get("version") != _METADATA_VERSION:
        return False
    if not isinstance(entries, list) or len(entries) != len(raw_paths):
        return False
    if not entries:
        return len(raw_paths) == 0
    return (
        entries[0].get("name") == Path(raw_paths[0]).name
        and entries[-1].get("name") == Path(raw_paths[-1]).name
    )


def _metadata_cache_path(raw_paths: Sequence[str], metadata_path: str | None) -> Path:
    if metadata_path:
        return Path(metadata_path)
    if not raw_paths:
        return Path(".catk_memory_balanced_metadata_v1.pkl")
    return Path(raw_paths[0]).parent / ".catk_memory_balanced_metadata_v1.pkl"


def _load_metadata_cache(cache_path: Path, raw_paths: Sequence[str]) -> list[dict] | None:
    if not cache_path.exists():
        return None
    with open(cache_path, "rb") as handle:
        metadata = pickle.load(handle)
    if not isinstance(metadata, dict) or not _metadata_matches_paths(metadata, raw_paths):
        return None
    return metadata["entries"]


def _build_metadata_entries(
    raw_paths: Sequence[str],
    num_workers: int,
) -> list[dict]:
    if num_workers <= 0:
        return [_read_raw_sample_metadata(path) for path in raw_paths]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(_read_raw_sample_metadata, raw_paths, chunksize=64))


def load_or_build_memory_metadata(
    raw_paths: Sequence[str],
    metadata_path: str | None = None,
    num_workers: int = 0,
    lock_timeout_seconds: float = 14400.0,
) -> list[dict]:
    """Load or build per-scenario metadata used for memory-aware batching."""
    raw_paths = list(raw_paths)
    cache_path = _metadata_cache_path(raw_paths, metadata_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached = _load_metadata_cache(cache_path, raw_paths)
    if cached is not None:
        return cached

    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    start_time = time.monotonic()
    while True:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            cached = _load_metadata_cache(cache_path, raw_paths)
            if cached is not None:
                return cached
            if time.monotonic() - start_time > lock_timeout_seconds:
                raise TimeoutError(
                    f"Timed out waiting for memory-balanced metadata cache: {cache_path}"
                )
            time.sleep(5.0)

    try:
        with os.fdopen(lock_fd, "w") as handle:
            handle.write(f"pid={os.getpid()} time={time.time()}\n")
        cached = _load_metadata_cache(cache_path, raw_paths)
        if cached is not None:
            return cached
        entries = _build_metadata_entries(raw_paths, num_workers=num_workers)
        metadata = {
            "version": _METADATA_VERSION,
            "created_at": time.time(),
            "entries": entries,
        }
        tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
        with open(tmp_path, "wb") as handle:
            pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
        return entries
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _metadata_weight(entry: dict, weight_key: str) -> float:
    agent_count = float(entry.get("agent_count", 1))
    valid_agent_steps = float(entry.get("valid_agent_steps", 0))
    map_point_count = float(entry.get("map_point_count", 0))
    if weight_key == "agent_count":
        return max(agent_count, 1.0)
    if weight_key == "agent_quadratic":
        return max(
            agent_count * agent_count
            + 0.05 * valid_agent_steps
            + 0.25 * map_point_count,
            1.0,
        )
    if weight_key == "valid_agent_steps":
        return max(valid_agent_steps, 1.0)
    if weight_key == "file_size":
        return max(float(entry.get("file_size", 1)), 1.0)
    raise ValueError(
        "weight_key must be one of "
        "{'agent_count', 'agent_quadratic', 'valid_agent_steps', 'file_size'}, "
        f"got {weight_key!r}."
    )


class MemoryBalancedDistributedBatchSampler(Sampler[list[int]]):
    """Distributed batch sampler that spreads memory-heavy scenarios across batches.

    The sampler keeps the same per-rank batch size and number of samples as a
    standard distributed sampler. It only changes the order of samples so dense
    scenarios are not randomly concentrated in the same rank-local batch.
    """

    def __init__(
        self,
        raw_paths: Sequence[str],
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        metadata_path: str | None = None,
        metadata_num_workers: int = 0,
        weight_key: str = "agent_quadratic",
        bucket_size_multiplier: int = 50,
        metadata_entries: Sequence[dict] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}.")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}.")

        self.raw_paths = list(raw_paths)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.bucket_size_multiplier = max(1, int(bucket_size_multiplier))

        if metadata_entries is None:
            metadata_entries = load_or_build_memory_metadata(
                self.raw_paths,
                metadata_path=metadata_path,
                num_workers=int(metadata_num_workers),
            )
        if len(metadata_entries) != len(self.raw_paths):
            raise ValueError(
                "metadata_entries length must match raw_paths length, "
                f"got {len(metadata_entries)} and {len(self.raw_paths)}."
            )
        self.weights = [_metadata_weight(entry, weight_key) for entry in metadata_entries]

        self.dataset_size = len(self.raw_paths)
        self.num_samples = int(math.ceil(self.dataset_size / self.num_replicas))
        self.num_batches = int(math.ceil(self.num_samples / self.batch_size))
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _epoch_indices(self) -> list[int]:
        indices = list(range(self.dataset_size))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(indices)
        if len(indices) < self.total_size:
            indices.extend(indices[: self.total_size - len(indices)])
        return indices[: self.total_size]

    def _rank_batch_capacities(self) -> list[int]:
        full_batches = self.num_samples // self.batch_size
        last_batch_size = self.num_samples - full_batches * self.batch_size
        capacities = [self.batch_size] * (full_batches * self.num_replicas)
        if last_batch_size > 0:
            capacities.extend([last_batch_size] * self.num_replicas)
        return capacities

    def _pack_window(
        self,
        window_indices: list[int],
        capacities: list[int],
        rng: random.Random,
    ) -> list[list[int]]:
        bins: list[list[int]] = [[] for _ in capacities]
        loads = [0.0 for _ in capacities]
        remaining = capacities[:]
        active = list(range(len(capacities)))
        weighted_indices = [
            (self.weights[index], rng.random(), index) for index in window_indices
        ]
        weighted_indices.sort(reverse=True)

        for weight, _, index in weighted_indices:
            bin_idx = min(active, key=lambda candidate: (loads[candidate], candidate))
            bins[bin_idx].append(index)
            loads[bin_idx] += weight
            remaining[bin_idx] -= 1
            if remaining[bin_idx] == 0:
                active.remove(bin_idx)
        return bins

    def _shuffle_rank_batch_steps(
        self,
        rank_batches: list[list[int]],
        rng: random.Random,
    ) -> list[list[int]]:
        steps = [
            rank_batches[start : start + self.num_replicas]
            for start in range(0, len(rank_batches), self.num_replicas)
        ]
        rng.shuffle(steps)
        for step in steps:
            rng.shuffle(step)
        return [batch for step in steps for batch in step]

    def _build_rank_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch + 1_000_003)
        epoch_indices = self._epoch_indices()
        capacities = self._rank_batch_capacities()
        rank_batches: list[list[int]] = []
        cursor = 0
        rank_batch_window = self.num_replicas * self.bucket_size_multiplier

        for start in range(0, len(capacities), rank_batch_window):
            window_capacities = capacities[start : start + rank_batch_window]
            window_size = sum(window_capacities)
            window_indices = epoch_indices[cursor : cursor + window_size]
            cursor += window_size
            window_batches = self._pack_window(window_indices, window_capacities, rng)
            rank_batches.extend(self._shuffle_rank_batch_steps(window_batches, rng))

        return rank_batches

    def __iter__(self) -> Iterator[list[int]]:
        rank_batches = self._build_rank_batches()
        for batch_idx in range(self.rank, len(rank_batches), self.num_replicas):
            yield rank_batches[batch_idx]
