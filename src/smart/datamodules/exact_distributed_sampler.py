from __future__ import annotations

import torch
from torch.utils.data.distributed import DistributedSampler


class ExactDistributedSampler(DistributedSampler):
    """Shard eval/test datasets across DDP ranks without padding duplicates."""

    def __init__(
        self,
        dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
        dataset_len = len(self.dataset)
        shard_size, remainder = divmod(dataset_len, self.num_replicas)
        self.num_samples = shard_size + int(self.rank < remainder)
        self.total_size = dataset_len

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)
