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

from typing import Optional

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform

from src.smart.datasets import MultiDataset

from .exact_distributed_sampler import ExactDistributedSampler
from .memory_balanced_sampler import MemoryBalancedDistributedBatchSampler
from .target_builder import WaymoTargetBuilderTrain, WaymoTargetBuilderVal


def build_train_agent_target_builder(
    train_max_num: int,
    train_use_eval_agent_selection: bool,
    train_agent_token_sidecar_dir: Optional[str],
    train_agent_token_sidecar_required: bool,
) -> BaseTransform:
    if train_use_eval_agent_selection:
        return WaymoTargetBuilderVal()
    return WaymoTargetBuilderTrain(
        train_max_num,
        agent_token_sidecar_dir=train_agent_token_sidecar_dir,
        agent_token_sidecar_required=train_agent_token_sidecar_required,
    )


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
        train_agent_token_sidecar_dir: Optional[str] = None,
        train_agent_token_sidecar_required: bool = False,
        road_num_rollouts_per_scenario: int = 1,
        train_memory_balanced_batching: bool = True,
        train_memory_balance_metadata_path: Optional[str] = None,
        train_memory_balance_metadata_num_workers: int = 0,
        train_memory_balance_weight_key: str = "agent_quadratic",
        train_memory_balance_bucket_size_multiplier: int = 50,
        train_memory_balance_seed: int = 0,
        random_scene_scale_config: Optional[dict] = None,
        random_time_shift_config: Optional[dict] = None,
    ) -> None:
        super(MultiDataModule, self).__init__()
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
        self.train_agent_token_sidecar_dir = train_agent_token_sidecar_dir
        self.train_agent_token_sidecar_required = bool(train_agent_token_sidecar_required)
        self.road_num_rollouts_per_scenario = road_num_rollouts_per_scenario
        self.train_memory_balanced_batching = train_memory_balanced_batching
        self.train_memory_balance_metadata_path = train_memory_balance_metadata_path
        self.train_memory_balance_metadata_num_workers = (
            train_memory_balance_metadata_num_workers
        )
        self.train_memory_balance_weight_key = train_memory_balance_weight_key
        self.train_memory_balance_bucket_size_multiplier = (
            train_memory_balance_bucket_size_multiplier
        )
        self.train_memory_balance_seed = int(train_memory_balance_seed)
        self.random_scene_scale_config = random_scene_scale_config
        self.random_time_shift_config = random_time_shift_config
        self._train_dataset_raw_dir: Optional[str] = None
        self._train_dataset_road_group_size: Optional[int] = None
        self._train_batch_sampler: Optional[MemoryBalancedDistributedBatchSampler] = None

        self.train_transform = build_train_agent_target_builder(
            train_max_num,
            train_use_eval_agent_selection,
            train_agent_token_sidecar_dir,
            self.train_agent_token_sidecar_required,
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
        self._train_batch_sampler = None

    def _build_train_dataset(self) -> None:
        """현재 설정된 train cache에서 학습 dataset을 만든다.

        RoaD fine-tuning에서는 epoch마다 cache 디렉터리가 바뀐다. 이 함수는
        dataloader가 새로 만들어질 때 현재 cache 위치를 기준으로 dataset을 다시 만든다.
        """
        self.train_dataset = MultiDataset(
            self.train_raw_dir,
            self.train_transform,
            road_num_rollouts_per_scenario=self.road_num_rollouts_per_scenario,
            random_scene_scale_config=self.random_scene_scale_config,
            random_time_shift_config=self.random_time_shift_config,
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

        if self.train_memory_balanced_batching:
            if self.road_num_rollouts_per_scenario != 1:
                raise ValueError(
                    "train_memory_balanced_batching currently supports one pickle "
                    "per dataset sample. Set road_num_rollouts_per_scenario=1."
                )
            world_size, global_rank = self._get_trainer_world_info()
            batch_sampler = MemoryBalancedDistributedBatchSampler(
                raw_paths=self.train_dataset.raw_paths,
                batch_size=self.train_batch_size,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=self.shuffle,
                seed=self.train_memory_balance_seed,
                metadata_path=self.train_memory_balance_metadata_path,
                metadata_num_workers=self.train_memory_balance_metadata_num_workers,
                weight_key=self.train_memory_balance_weight_key,
                bucket_size_multiplier=self.train_memory_balance_bucket_size_multiplier,
            )
            self._train_batch_sampler = batch_sampler
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )

        self._train_batch_sampler = None
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

    def set_train_epoch(self, epoch: int) -> None:
        """Explicitly propagate the current training epoch to custom train samplers."""
        batch_sampler = getattr(self, "_train_batch_sampler", None)
        set_epoch = getattr(batch_sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(int(epoch))

    def _get_trainer_world_info(self) -> tuple[int, int]:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return 1, 0
        world_size = int(getattr(trainer, "world_size", 1) or 1)
        global_rank = int(getattr(trainer, "global_rank", 0) or 0)
        return max(1, world_size), global_rank

    def _build_train_sampler(self, dataset):
        world_size, global_rank = self._get_trainer_world_info()
        if world_size <= 1:
            return None
        return DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=self.shuffle,
            seed=self.train_memory_balance_seed,
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
