# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
from typing import Optional

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform

from src.smart.datasets import MultiDataset

from .exact_distributed_sampler import ExactDistributedSampler
from .target_builder import WaymoTargetBuilderTrain, WaymoTargetBuilderVal


def build_train_agent_target_builder(
    train_max_num: int, train_use_eval_agent_selection: bool
) -> BaseTransform:
    if train_use_eval_agent_selection:
        return WaymoTargetBuilderVal()
    return WaymoTargetBuilderTrain(train_max_num)


class FractionDistributedSampler(DistributedSampler):
    """Shard a per-epoch dataset fraction across ranks."""

    def __init__(
        self,
        dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        fraction: float = 1.0,
    ) -> None:
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
        self.fraction = float(fraction)
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
        dataset_len = len(self.dataset)
        if dataset_len <= 0:
            raise ValueError("fractional training sampler requires a non-empty dataset.")
        self.fraction_size = max(1, int(math.floor(dataset_len * self.fraction)))
        if self.fraction >= 1.0:
            self.fraction_size = dataset_len
        self.num_samples = int(math.ceil(self.fraction_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))

        indices = indices[: self.fraction_size]
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            repeat_count = math.ceil(padding_size / len(indices))
            indices += (indices * repeat_count)[:padding_size]
        else:
            indices = indices[: self.total_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)


class MultiDataModule(LightningDataModule):
    def __init__(
        self,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        train_raw_dir: str,
        val_raw_dir: str,
        test_raw_dir: str,
        val_tfrecords_splitted: str,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        train_max_num: int,
        train_use_eval_agent_selection: bool = False,
        road_num_rollouts_per_scenario: int = 1,
        train_epoch_sample_fraction: float = 1.0,
    ) -> None:
        super(MultiDataModule, self).__init__()
        if not 0.0 < float(train_epoch_sample_fraction) <= 1.0:
            raise ValueError(
                "train_epoch_sample_fraction must be in (0, 1], "
                f"got {train_epoch_sample_fraction}."
            )
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.val_tfrecords_splitted = val_tfrecords_splitted
        self.train_use_eval_agent_selection = train_use_eval_agent_selection
        self.road_num_rollouts_per_scenario = road_num_rollouts_per_scenario
        self.train_epoch_sample_fraction = float(train_epoch_sample_fraction)
        self._train_dataset_raw_dir: Optional[str] = None
        self._train_dataset_road_group_size: Optional[int] = None

        self.train_transform = build_train_agent_target_builder(
            train_max_num, train_use_eval_agent_selection
        )
        self.val_transform = WaymoTargetBuilderVal()
        self.test_transform = WaymoTargetBuilderVal()

    def set_train_raw_dir(
        self, train_raw_dir: str, road_num_rollouts_per_scenario: int = 1
    ) -> None:
        """다음 train dataloader가 읽을 cache 위치를 바꾼다.

        Args:
            train_raw_dir: 다음 epoch에서 사용할 pickle cache 디렉터리이다.
            road_num_rollouts_per_scenario: scenario 하나당 저장된 RoaD rollout 개수이다.
                1이면 기존 WOMD cache처럼 파일 하나를 sample 하나로 본다.
        """
        self.train_raw_dir = train_raw_dir
        self.road_num_rollouts_per_scenario = road_num_rollouts_per_scenario
        self.train_dataset = None
        self._train_dataset_raw_dir = None
        self._train_dataset_road_group_size = None

    def _build_train_dataset(self) -> None:
        """현재 설정된 train cache에서 학습 dataset을 만든다.

        RoaD fine-tuning에서는 epoch마다 cache 디렉터리가 바뀐다. 이 함수는
        dataloader가 새로 만들어질 때 현재 cache 위치를 기준으로 dataset을 다시 만든다.
        """
        self.train_dataset = MultiDataset(
            self.train_raw_dir,
            self.train_transform,
            road_num_rollouts_per_scenario=self.road_num_rollouts_per_scenario,
        )
        self._train_dataset_raw_dir = self.train_raw_dir
        self._train_dataset_road_group_size = self.road_num_rollouts_per_scenario

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self._build_train_dataset()
            self.val_dataset = MultiDataset(
                self.val_raw_dir,
                self.val_transform,
                tfrecord_dir=self.val_tfrecords_splitted,
            )
        elif stage == "validate":
            self.val_dataset = MultiDataset(
                self.val_raw_dir,
                self.val_transform,
                tfrecord_dir=self.val_tfrecords_splitted,
            )
        elif stage == "test":
            self.test_dataset = MultiDataset(self.test_raw_dir, self.test_transform)
        else:
            raise ValueError(f"{stage} should be one of [fit, validate, test]")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        needs_rebuild = (
            not hasattr(self, "train_dataset")
            or self.train_dataset is None
            or self._train_dataset_raw_dir != self.train_raw_dir
            or self._train_dataset_road_group_size != self.road_num_rollouts_per_scenario
        )
        if needs_rebuild:
            self._build_train_dataset()

        sampler = self._build_train_sampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def _get_trainer_world_info(self) -> tuple[int, int]:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return 1, 0
        world_size = int(getattr(trainer, "world_size", 1) or 1)
        global_rank = int(getattr(trainer, "global_rank", 0) or 0)
        return max(1, world_size), global_rank

    def _build_train_sampler(self, dataset):
        world_size, global_rank = self._get_trainer_world_info()
        if self.train_epoch_sample_fraction < 1.0:
            return FractionDistributedSampler(
                dataset=dataset,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=self.shuffle,
                fraction=self.train_epoch_sample_fraction,
            )
        if world_size <= 1:
            return None
        return DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=self.shuffle,
            drop_last=False,
        )

    def _build_eval_sampler(self, dataset):
        world_size, global_rank = self._get_trainer_world_info()
        if world_size <= 1:
            return None
        return ExactDistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        sampler = self._build_eval_sampler(self.val_dataset)
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        sampler = self._build_eval_sampler(self.test_dataset)
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )
