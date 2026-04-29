from __future__ import annotations

import math
from typing import Iterator

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class RandomFractionDistributedSampler(DistributedSampler):
    """Sample one global random dataset fraction per epoch and shard it by rank."""

    def __init__(
        self,
        dataset: Dataset,
        fraction: float,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
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
        self.subset_size = max(1, int(math.floor(dataset_len * self.fraction)))
        self.num_samples = int(math.ceil(self.subset_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        indices = indices[: self.subset_size]

        if len(indices) < self.total_size:
            repeat_count = int(math.ceil((self.total_size - len(indices)) / len(indices)))
            indices += (indices * repeat_count)[: self.total_size - len(indices)]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)
