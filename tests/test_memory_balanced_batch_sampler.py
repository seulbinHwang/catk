import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import torch

from src.smart.datamodules.memory_balanced_batch_sampler import (
    _build_metadata,
    fingerprint_raw_paths,
    load_or_build_memory_metadata,
    memory_balance_weights,
    memory_metadata_lock_path,
)


def _write_sample(path: Path, *, agent_count: int, current_valid: int, map_count: int) -> None:
    valid_mask = torch.zeros(agent_count, 12, dtype=torch.bool)
    valid_mask[:, :5] = True
    valid_mask[:current_valid, 10] = True
    data = {
        "agent": {
            "type": torch.zeros(agent_count, dtype=torch.long),
            "valid_mask": valid_mask,
            "position": torch.zeros(agent_count, 12, 3),
        },
        "map_save": {
            "traj_pos": torch.zeros(map_count, 2),
        },
    }
    with path.open("wb") as handle:
        pickle.dump(data, handle)


def test_process_metadata_build_matches_serial(tmp_path: Path) -> None:
    specs = [(3, 2, 11), (5, 4, 7), (1, 1, 0), (8, 3, 19)]
    raw_paths = []
    for idx, (agent_count, current_valid, map_count) in enumerate(specs):
        path = tmp_path / f"sample_{idx}.pkl"
        _write_sample(
            path,
            agent_count=agent_count,
            current_valid=current_valid,
            map_count=map_count,
        )
        raw_paths.append(str(path))

    serial = _build_metadata(raw_paths, num_workers=1)
    parallel = _build_metadata(raw_paths, num_workers=2)

    assert torch.equal(parallel.agent_count, serial.agent_count)
    assert torch.equal(
        parallel.current_valid_agent_count, serial.current_valid_agent_count
    )
    assert torch.equal(parallel.valid_agent_step_count, serial.valid_agent_step_count)
    assert torch.equal(parallel.map_count, serial.map_count)


def test_metadata_cache_loads_tensor_payload_without_rebuilding(tmp_path: Path) -> None:
    raw_paths = []
    for idx, agent_count in enumerate([2, 4]):
        path = tmp_path / f"cached_{idx}.pkl"
        _write_sample(path, agent_count=agent_count, current_valid=1, map_count=idx + 3)
        raw_paths.append(str(path))

    cache_path = tmp_path / "metadata.pt"
    built = load_or_build_memory_metadata(
        raw_paths,
        cache_path=str(cache_path),
        num_workers=1,
        build_on_missing=True,
    )
    loaded = load_or_build_memory_metadata(
        raw_paths,
        cache_path=str(cache_path),
        num_workers=1,
        build_on_missing=False,
    )

    assert cache_path.is_file()
    assert torch.equal(loaded.agent_count, built.agent_count)
    assert torch.equal(loaded.map_count, built.map_count)
    weights = memory_balance_weights(
        loaded,
        agent_weight=1.0,
        current_valid_agent_weight=1.0,
        valid_agent_step_weight=0.0,
        map_weight=0.02,
    )
    assert torch.allclose(weights, torch.tensor([3.06, 5.08], dtype=torch.float64))


def test_metadata_cache_rejects_same_filenames_from_different_root(tmp_path: Path) -> None:
    raw_dir_a = tmp_path / "cache_a" / "training"
    raw_dir_b = tmp_path / "cache_b" / "training"
    raw_dir_a.mkdir(parents=True)
    raw_dir_b.mkdir(parents=True)
    for raw_dir in (raw_dir_a, raw_dir_b):
        _write_sample(raw_dir / "same_name.pkl", agent_count=2, current_valid=1, map_count=3)

    cache_path = tmp_path / "metadata.pt"
    load_or_build_memory_metadata(
        [str(raw_dir_a / "same_name.pkl")],
        cache_path=str(cache_path),
        num_workers=1,
        build_on_missing=True,
    )

    try:
        load_or_build_memory_metadata(
            [str(raw_dir_b / "same_name.pkl")],
            cache_path=str(cache_path),
            num_workers=1,
            build_on_missing=False,
        )
    except FileNotFoundError as exc:
        assert "missing or stale" in str(exc)
    else:
        raise AssertionError("metadata cache should be stale for a different cache root")


def test_corrupt_metadata_cache_rebuilds_when_allowed(tmp_path: Path) -> None:
    raw_path = tmp_path / "sample.pkl"
    _write_sample(raw_path, agent_count=2, current_valid=1, map_count=3)
    cache_path = tmp_path / "metadata.pt"
    cache_path.write_bytes(b"")

    metadata = load_or_build_memory_metadata(
        [str(raw_path)],
        cache_path=str(cache_path),
        num_workers=1,
        build_on_missing=True,
    )

    assert metadata.agent_count.tolist() == [2]
    assert cache_path.stat().st_size > 0


def test_old_list_payload_still_loads(tmp_path: Path) -> None:
    raw_paths = [str(tmp_path / "a.pkl"), str(tmp_path / "b.pkl")]
    cache_path = tmp_path / "old_metadata.pt"
    torch.save(
        {
            "version": 1,
            "num_samples": 2,
            "fingerprint": fingerprint_raw_paths(raw_paths),
            "agent_count": [3, 5],
            "current_valid_agent_count": [2, 4],
            "valid_agent_step_count": [15, 25],
            "map_count": [7, 9],
        },
        cache_path,
    )

    loaded = load_or_build_memory_metadata(
        raw_paths,
        cache_path=str(cache_path),
        num_workers=1,
        build_on_missing=False,
    )

    assert loaded.agent_count.tolist() == [3, 5]
    assert loaded.current_valid_agent_count.tolist() == [2, 4]
    assert loaded.valid_agent_step_count.tolist() == [15, 25]
    assert loaded.map_count.tolist() == [7, 9]


def test_prebuild_force_removes_stale_lock_dir(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_sample(raw_dir / "sample.pkl", agent_count=2, current_valid=1, map_count=3)

    cache_path = tmp_path / "metadata.pt"
    lock_path = memory_metadata_lock_path(cache_path)
    lock_path.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "build_memory_balance_metadata.py"),
            "--raw-dir",
            str(raw_dir),
            "--cache-path",
            str(cache_path),
            "--num-workers",
            "1",
            "--force",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "memory-balance metadata ready" in result.stdout
    assert cache_path.is_file()
    assert not lock_path.exists()


def test_metadata_cache_reclaims_stale_lock_dir(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "sample.pkl"
    _write_sample(raw_path, agent_count=2, current_valid=1, map_count=3)

    cache_path = tmp_path / "metadata.pt"
    lock_path = memory_metadata_lock_path(cache_path)
    lock_path.mkdir()
    (lock_path / "owner.txt").write_text("pid=999999 host=dead-builder time=0\n")
    old_mtime = time.time() - 60.0
    # Directory mtime is the stale-lock signal used by the loader.
    os.utime(lock_path, (old_mtime, old_mtime))

    metadata = load_or_build_memory_metadata(
        [str(raw_path)],
        cache_path=str(cache_path),
        num_workers=1,
        build_on_missing=True,
        lock_timeout_sec=2,
        lock_stale_sec=0.01,
        lock_poll_sec=0.1,
    )

    assert metadata.agent_count.tolist() == [2]
    assert cache_path.is_file()
    assert not lock_path.exists()
