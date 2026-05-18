from __future__ import annotations

import hashlib
import math
import os
import pickle
import shutil
import socket
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
from torch.utils.data import Sampler


METADATA_VERSION = 1
DEFAULT_LOCK_STALE_SEC = 30.0
DEFAULT_LOCK_POLL_SEC = 1.0
LOCK_HEARTBEAT_INTERVAL_SEC = 5.0


@dataclass(frozen=True)
class SampleMemoryMetadata:
    """Lightweight per-cache-file fields used only for train batch ordering."""

    agent_count: int
    current_valid_agent_count: int
    valid_agent_step_count: int
    map_count: int


@dataclass(frozen=True)
class MemoryBalanceMetadata:
    """Tensorized per-cache-file fields used only for train batch ordering."""

    agent_count: torch.Tensor
    current_valid_agent_count: torch.Tensor
    valid_agent_step_count: torch.Tensor
    map_count: torch.Tensor

    def __len__(self) -> int:
        return int(self.agent_count.numel())

    @classmethod
    def from_samples(cls, samples: Iterable[SampleMemoryMetadata]) -> "MemoryBalanceMetadata":
        agent_count = []
        current_valid_agent_count = []
        valid_agent_step_count = []
        map_count = []
        for item in samples:
            agent_count.append(int(item.agent_count))
            current_valid_agent_count.append(int(item.current_valid_agent_count))
            valid_agent_step_count.append(int(item.valid_agent_step_count))
            map_count.append(int(item.map_count))
        return cls(
            agent_count=torch.as_tensor(agent_count, dtype=torch.int64),
            current_valid_agent_count=torch.as_tensor(
                current_valid_agent_count, dtype=torch.int64
            ),
            valid_agent_step_count=torch.as_tensor(valid_agent_step_count, dtype=torch.int64),
            map_count=torch.as_tensor(map_count, dtype=torch.int64),
        )


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


def _init_metadata_worker() -> None:
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def fingerprint_raw_paths(raw_paths: Sequence[str]) -> str:
    """Fingerprint the exact SMART cache path list used by the dataloader."""

    digest = hashlib.sha1()
    for path in raw_paths:
        resolved_path = Path(path).expanduser().resolve(strict=False)
        digest.update(str(resolved_path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _default_metadata_cache_path(raw_paths: Sequence[str]) -> Path:
    raw_dir = Path(raw_paths[0]).parent if raw_paths else Path(".")
    filename = f"{raw_dir.name}_memory_balance_v{METADATA_VERSION}.pt"
    return raw_dir.parent / ".catk_metadata" / filename


def memory_metadata_lock_path(cache_path: str | Path) -> Path:
    cache_path = Path(cache_path).expanduser()
    return cache_path.with_suffix(cache_path.suffix + ".lock")


def _metadata_lock_owner_path(lock_dir: Path) -> Path:
    return lock_dir / "owner.txt"


def _is_lock_stale(lock_dir: Path, stale_sec: float) -> bool:
    if stale_sec <= 0:
        return False
    try:
        lock_mtime = lock_dir.stat().st_mtime
    except FileNotFoundError:
        return False
    return time.time() - lock_mtime > stale_sec


def _remove_lock_path(lock_dir: Path) -> bool:
    try:
        if lock_dir.is_dir():
            shutil.rmtree(lock_dir)
        else:
            lock_dir.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _remove_stale_lock_if_needed(lock_dir: Path, stale_sec: float) -> bool:
    if not _is_lock_stale(lock_dir, stale_sec):
        return False
    return _remove_lock_path(lock_dir)


def _start_lock_heartbeat(
    lock_dir: Path,
    stop_event: threading.Event,
    stale_sec: float,
) -> threading.Thread | None:
    if stale_sec <= 0:
        return None
    interval = min(LOCK_HEARTBEAT_INTERVAL_SEC, max(1.0, stale_sec / 3.0))

    def _heartbeat() -> None:
        while not stop_event.wait(interval):
            try:
                os.utime(lock_dir, None)
            except FileNotFoundError:
                return

    thread = threading.Thread(
        target=_heartbeat,
        name="memory-balance-metadata-lock-heartbeat",
        daemon=True,
    )
    thread.start()
    return thread


def _as_int64_tensor(value) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.int64).cpu().flatten().contiguous()


def _payload_to_metadata(payload: dict) -> MemoryBalanceMetadata:
    return MemoryBalanceMetadata(
        agent_count=_as_int64_tensor(payload["agent_count"]),
        current_valid_agent_count=_as_int64_tensor(payload["current_valid_agent_count"]),
        valid_agent_step_count=_as_int64_tensor(payload["valid_agent_step_count"]),
        map_count=_as_int64_tensor(payload["map_count"]),
    )


def _metadata_payload(
    metadata: MemoryBalanceMetadata, raw_paths: Sequence[str]
) -> dict[str, object]:
    return {
        "version": METADATA_VERSION,
        "num_samples": len(raw_paths),
        "fingerprint": fingerprint_raw_paths(raw_paths),
        "agent_count": metadata.agent_count,
        "current_valid_agent_count": metadata.current_valid_agent_count,
        "valid_agent_step_count": metadata.valid_agent_step_count,
        "map_count": metadata.map_count,
    }


def _build_metadata(raw_paths: Sequence[str], num_workers: int) -> MemoryBalanceMetadata:
    if num_workers <= 1:
        return MemoryBalanceMetadata.from_samples(
            _extract_sample_metadata(path) for path in raw_paths
        )
    with ProcessPoolExecutor(
        max_workers=int(num_workers), initializer=_init_metadata_worker
    ) as executor:
        samples = executor.map(_extract_sample_metadata, raw_paths, chunksize=128)
        return MemoryBalanceMetadata.from_samples(samples)


def _load_metadata_file(
    cache_path: Path, raw_paths: Sequence[str]
) -> MemoryBalanceMetadata | None:
    if not cache_path.is_file():
        return None
    try:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
    except (EOFError, OSError, RuntimeError, ValueError, pickle.UnpicklingError):
        return None
    expected_fingerprint = fingerprint_raw_paths(raw_paths)
    if payload.get("version") != METADATA_VERSION:
        return None
    if payload.get("num_samples") != len(raw_paths):
        return None
    if payload.get("fingerprint") != expected_fingerprint:
        return None

    try:
        metadata = _payload_to_metadata(payload)
    except KeyError:
        return None
    if len(metadata) != len(raw_paths):
        return None
    return metadata


def load_or_build_memory_metadata(
    raw_paths: Sequence[str],
    *,
    cache_path: str | None,
    num_workers: int,
    build_on_missing: bool,
    lock_timeout_sec: int = 7200,
    lock_stale_sec: float = DEFAULT_LOCK_STALE_SEC,
    lock_poll_sec: float = DEFAULT_LOCK_POLL_SEC,
) -> MemoryBalanceMetadata:
    """Load cached sample-size metadata, building it once when allowed.

    The cache intentionally stores only counts and a path-list fingerprint. It
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
    lock_dir = memory_metadata_lock_path(resolved_cache_path)
    lock_stale_sec = float(lock_stale_sec)
    lock_poll_sec = max(0.1, float(lock_poll_sec))
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
            if _remove_stale_lock_if_needed(lock_dir, lock_stale_sec):
                continue
            if time.monotonic() - start_time > lock_timeout_sec:
                raise TimeoutError(
                    "Timed out waiting for train memory-balance metadata cache: "
                    f"{resolved_cache_path}"
                )
            time.sleep(lock_poll_sec)

    stop_heartbeat = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    try:
        _metadata_lock_owner_path(lock_dir).write_text(
            f"pid={os.getpid()} host={socket.gethostname()} time={time.time()}\n"
        )
        heartbeat_thread = _start_lock_heartbeat(
            lock_dir=lock_dir,
            stop_event=stop_heartbeat,
            stale_sec=lock_stale_sec,
        )
        loaded = _load_metadata_file(resolved_cache_path, raw_paths)
        if loaded is not None:
            return loaded

        metadata = _build_metadata(raw_paths, num_workers=num_workers)
        payload = _metadata_payload(metadata, raw_paths)
        tmp_path = resolved_cache_path.with_suffix(
            resolved_cache_path.suffix + f".tmp.{os.getpid()}"
        )
        torch.save(payload, tmp_path)
        os.replace(tmp_path, resolved_cache_path)
        return metadata
    finally:
        stop_heartbeat.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)
        if owns_lock:
            _remove_lock_path(lock_dir)


def memory_balance_weights(
    metadata: MemoryBalanceMetadata | Iterable[SampleMemoryMetadata],
    *,
    agent_weight: float,
    current_valid_agent_weight: float,
    valid_agent_step_weight: float,
    map_weight: float,
) -> torch.Tensor:
    if isinstance(metadata, MemoryBalanceMetadata):
        return (
            float(agent_weight) * metadata.agent_count.to(torch.float64)
            + float(current_valid_agent_weight)
            * metadata.current_valid_agent_count.to(torch.float64)
            + float(valid_agent_step_weight)
            * metadata.valid_agent_step_count.to(torch.float64)
            + float(map_weight) * metadata.map_count.to(torch.float64)
        )

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

    @property
    def sampler(self):
        return self

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
