from __future__ import annotations

import pickle

import torch

from src.smart.datasets.scalable_dataset import MultiDataset
from src.smart.datamodules.memory_balanced_sampler import (
    MemoryBalancedDistributedBatchSampler,
    load_or_build_memory_metadata,
)


def _fake_paths(n_samples: int) -> list[str]:
    return [f"/tmp/scenario_{idx:05d}.pkl" for idx in range(n_samples)]


def _metadata_from_agent_counts(agent_counts: list[int]) -> list[dict]:
    return [
        {
            "name": f"scenario_{idx:05d}.pkl",
            "agent_count": count,
            "valid_agent_steps": count * 91,
            "map_point_count": 100,
            "file_size": count * 1000,
        }
        for idx, count in enumerate(agent_counts)
    ]


def _collect_rank_batches(
    agent_counts: list[int],
    rank: int,
    epoch: int = 0,
) -> list[list[int]]:
    sampler = MemoryBalancedDistributedBatchSampler(
        raw_paths=_fake_paths(len(agent_counts)),
        batch_size=2,
        num_replicas=2,
        rank=rank,
        shuffle=True,
        seed=7,
        metadata_entries=_metadata_from_agent_counts(agent_counts),
        weight_key="agent_count",
        bucket_size_multiplier=4,
    )
    sampler.set_epoch(epoch)
    return list(iter(sampler))


def test_memory_balanced_sampler_keeps_distributed_batch_shape() -> None:
    agent_counts = [100, 95, 90, 85, 20, 18, 16, 14, 12, 10, 8]
    rank0_batches = _collect_rank_batches(agent_counts, rank=0)
    rank1_batches = _collect_rank_batches(agent_counts, rank=1)

    assert len(rank0_batches) == len(rank1_batches)
    assert all(0 < len(batch) <= 2 for batch in rank0_batches)
    assert all(0 < len(batch) <= 2 for batch in rank1_batches)
    assert sum(len(batch) for batch in rank0_batches) == 6
    assert sum(len(batch) for batch in rank1_batches) == 6


def test_memory_balanced_sampler_spreads_heavy_scenarios() -> None:
    agent_counts = [120, 110, 100, 90, 16, 15, 14, 13]
    rank0_batches = _collect_rank_batches(agent_counts, rank=0)
    rank1_batches = _collect_rank_batches(agent_counts, rank=1)
    all_batches = rank0_batches + rank1_batches

    max_batch_agents = max(sum(agent_counts[idx] for idx in batch) for batch in all_batches)

    assert max_batch_agents < 120 + 110


def test_memory_balanced_sampler_epoch_changes_order() -> None:
    agent_counts = [50, 45, 40, 35, 30, 25, 20, 15]

    epoch0 = _collect_rank_batches(agent_counts, rank=0, epoch=0)
    epoch1 = _collect_rank_batches(agent_counts, rank=0, epoch=1)

    assert epoch0 != epoch1


def test_memory_metadata_cache_reads_agent_valid_and_map_counts(tmp_path) -> None:
    raw_dir = tmp_path / "training"
    raw_dir.mkdir()
    raw_path = raw_dir / "sample.pkl"
    data = {
        "agent": {
            "position": torch.zeros(3, 91, 3),
            "valid_mask": torch.ones(3, 91, dtype=torch.bool),
        },
        "pt_token": {
            "position": torch.zeros(17, 2),
        },
    }
    with open(raw_path, "wb") as handle:
        pickle.dump(data, handle)

    entries = load_or_build_memory_metadata(
        [raw_path.as_posix()],
        metadata_path=(tmp_path / "metadata.pkl").as_posix(),
        num_workers=0,
    )

    assert entries[0]["name"] == "sample.pkl"
    assert entries[0]["agent_count"] == 3
    assert entries[0]["valid_agent_steps"] == 273
    assert entries[0]["map_point_count"] == 17
    assert entries[0]["file_size"] > 0


def test_multidataset_ignores_hidden_metadata_cache(tmp_path) -> None:
    raw_dir = tmp_path / "training"
    raw_dir.mkdir()
    with open(raw_dir / "sample.pkl", "wb") as handle:
        pickle.dump({"scenario_id": "sample"}, handle)
    with open(raw_dir / ".catk_memory_balanced_metadata_v1.pkl", "wb") as handle:
        pickle.dump({"not": "a scenario"}, handle)

    dataset = MultiDataset(raw_dir.as_posix(), transform=lambda data: data)

    assert len(dataset.raw_paths) == 1
    assert dataset.raw_paths[0].endswith("sample.pkl")
