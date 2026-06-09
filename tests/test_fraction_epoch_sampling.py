import pytest
import torch
from torch.utils.data import TensorDataset

from src.smart.datamodules.memory_balanced_batch_sampler import (
    MemoryBalancedDistributedBatchSampler,
)
from src.smart.datamodules.random_fraction_distributed_sampler import (
    RandomFractionDistributedSampler,
)


def _epoch_indices(sampler, epoch: int) -> list[int]:
    sampler.set_epoch(epoch)
    return list(iter(sampler))


def _epoch_batch_indices(sampler, epoch: int) -> list[int]:
    sampler.set_epoch(epoch)
    return [index for batch in sampler for index in batch]


def test_random_fraction_default_mode_allows_non_integral_reciprocal() -> None:
    dataset = TensorDataset(torch.arange(10))
    sampler = RandomFractionDistributedSampler(
        dataset=dataset,
        fraction=0.3,
        num_replicas=1,
        rank=0,
        shuffle_fraction_each_epoch=True,
    )

    assert len(_epoch_indices(sampler, 0)) == 3


def test_random_fraction_partition_mode_rejects_non_integral_reciprocal() -> None:
    dataset = TensorDataset(torch.arange(10))

    with pytest.raises(ValueError, match="1 / train_epoch_sample_fraction"):
        RandomFractionDistributedSampler(
            dataset=dataset,
            fraction=0.3,
            num_replicas=1,
            rank=0,
            shuffle_fraction_each_epoch=False,
        )


def test_random_fraction_partition_mode_covers_full_dataset_once_per_cycle() -> None:
    dataset = TensorDataset(torch.arange(12))
    sampler = RandomFractionDistributedSampler(
        dataset=dataset,
        fraction=0.25,
        num_replicas=1,
        rank=0,
        seed=17,
        shuffle_fraction_each_epoch=False,
    )

    epoch_sets = [set(_epoch_indices(sampler, epoch)) for epoch in range(4)]

    assert set.union(*epoch_sets) == set(range(12))
    for left in range(4):
        for right in range(left + 1, 4):
            assert epoch_sets[left].isdisjoint(epoch_sets[right])


def test_random_fraction_partition_mode_shards_same_global_cycle_across_ranks() -> None:
    dataset = TensorDataset(torch.arange(12))
    samplers = [
        RandomFractionDistributedSampler(
            dataset=dataset,
            fraction=0.25,
            num_replicas=2,
            rank=rank,
            seed=19,
            shuffle_fraction_each_epoch=False,
        )
        for rank in range(2)
    ]

    epoch_sets = []
    for epoch in range(4):
        combined = set()
        for sampler in samplers:
            combined.update(_epoch_indices(sampler, epoch))
        epoch_sets.append(combined)

    assert set.union(*epoch_sets) == set(range(12))
    for left in range(4):
        for right in range(left + 1, 4):
            assert epoch_sets[left].isdisjoint(epoch_sets[right])


def test_memory_balanced_partition_mode_covers_full_dataset_before_balancing() -> None:
    sampler = MemoryBalancedDistributedBatchSampler(
        sample_weight=torch.arange(12, dtype=torch.float64),
        batch_size=3,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=23,
        fraction=0.25,
        shuffle_fraction_each_epoch=False,
    )

    epoch_sets = [set(_epoch_batch_indices(sampler, epoch)) for epoch in range(4)]

    assert set.union(*epoch_sets) == set(range(12))
    for left in range(4):
        for right in range(left + 1, 4):
            assert epoch_sets[left].isdisjoint(epoch_sets[right])


def test_memory_balanced_partition_mode_shards_same_global_cycle_across_ranks() -> None:
    samplers = [
        MemoryBalancedDistributedBatchSampler(
            sample_weight=torch.arange(12, dtype=torch.float64),
            batch_size=2,
            num_replicas=2,
            rank=rank,
            shuffle=True,
            seed=29,
            fraction=0.25,
            shuffle_fraction_each_epoch=False,
        )
        for rank in range(2)
    ]

    epoch_sets = []
    for epoch in range(4):
        combined = set()
        for sampler in samplers:
            combined.update(_epoch_batch_indices(sampler, epoch))
        epoch_sets.append(combined)

    assert set.union(*epoch_sets) == set(range(12))
    for left in range(4):
        for right in range(left + 1, 4):
            assert epoch_sets[left].isdisjoint(epoch_sets[right])
