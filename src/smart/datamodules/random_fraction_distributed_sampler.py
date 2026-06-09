from __future__ import annotations

import math
from typing import Iterator

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


def validate_fraction_cycle_length(fraction: float) -> int:
    """Return cycle length for non-reshuffled fractional epochs.

    ``fraction=0.25`` means a 4-epoch cycle. The sequential partition mode
    relies on an integral cycle length so each cycle can cover the full train
    dataset exactly once before padding for distributed sharding.
    """

    reciprocal = 1.0 / float(fraction)
    cycle_length = int(round(reciprocal))
    if not math.isclose(reciprocal, cycle_length, rel_tol=1.0e-9, abs_tol=1.0e-9):
        raise ValueError(
            "train_epoch_sample_fraction_shuffle_flag=false requires "
            "1 / train_epoch_sample_fraction to be an integer; "
            f"got fraction={fraction} and reciprocal={reciprocal}."
        )
    return cycle_length


def fraction_epoch_partition_indices(
    *,
    dataset_size: int,
    fraction: float,
    epoch: int,
    seed: int,
) -> torch.Tensor:
    cycle_length = validate_fraction_cycle_length(fraction)
    if int(dataset_size) < cycle_length:
        raise ValueError(
            "train_epoch_sample_fraction_shuffle_flag=false requires at least "
            "1 / train_epoch_sample_fraction train samples; "
            f"got dataset_size={dataset_size} and cycle_length={cycle_length}."
        )
    cycle_index = int(epoch) // cycle_length
    cycle_epoch = int(epoch) % cycle_length

    generator = torch.Generator()
    generator.manual_seed(int(seed) + cycle_index)
    indices = torch.randperm(int(dataset_size), generator=generator)
    return torch.tensor_split(indices, cycle_length)[cycle_epoch]


class RandomFractionDistributedSampler(DistributedSampler):
    """Sample one global random dataset fraction per epoch and shard it by rank."""

    def __init__(
        self,
        dataset: Dataset,
        fraction: float,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        shuffle_fraction_each_epoch: bool = True,
    ) -> None:
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        dataset_len = len(self.dataset)
        self.fraction = float(fraction)
        self.shuffle_fraction_each_epoch = bool(shuffle_fraction_each_epoch)
        self.cycle_length = (
            None
            if self.shuffle_fraction_each_epoch
            else validate_fraction_cycle_length(self.fraction)
        )
        if self.shuffle_fraction_each_epoch:
            self.subset_size = max(1, int(math.floor(dataset_len * self.fraction)))
        else:
            self.subset_size = max(1, int(math.ceil(dataset_len / self.cycle_length)))
        self.num_samples = int(math.ceil(self.subset_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        if self.shuffle_fraction_each_epoch:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator)
            indices = indices[: self.subset_size]
        else:
            indices = fraction_epoch_partition_indices(
                dataset_size=len(self.dataset),
                fraction=self.fraction,
                epoch=self.epoch,
                seed=self.seed,
            )
        indices = indices.tolist()

        if len(indices) < self.total_size:
            repeat_count = int(math.ceil((self.total_size - len(indices)) / len(indices)))
            indices += (indices * repeat_count)[: self.total_size - len(indices)]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)
