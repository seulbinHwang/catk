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
from .memory_balanced_batch_sampler import (
    MemoryBalancedDistributedBatchSampler,
    load_or_build_memory_metadata,
    memory_balance_weights,
)
from .random_fraction_distributed_sampler import RandomFractionDistributedSampler
from .target_builder import WaymoTargetBuilderTrain, WaymoTargetBuilderVal


def build_train_agent_target_builder(
    train_max_num: int,
    train_use_eval_agent_selection: bool,
    map_pt2pt_radius: Optional[float],
    map_pt2pt_max_num_neighbors: int,
) -> BaseTransform:
    """학습용 agent 선택 규칙에 맞는 transform을 고릅니다.

    Args:
        train_max_num: 기존 학습 규칙에서 사용할 최대 학습 대상 agent 수입니다.
        train_use_eval_agent_selection: ``True``면 학습에서도 validation/추론과 같은
            agent 기준을 그대로 씁니다. 이 경우 입력 agent를 150m로 자르지 않고
            추가 ``train_mask``도 만들지 않습니다. ``False``면 기존 학습 규칙을
            그대로 사용합니다.

    Returns:
        BaseTransform: 학습 데이터셋에 붙일 transform 객체입니다.
    """
    if train_use_eval_agent_selection:
        return WaymoTargetBuilderVal(
            map_pt2pt_radius=map_pt2pt_radius,
            map_pt2pt_max_num_neighbors=map_pt2pt_max_num_neighbors,
        )
    return WaymoTargetBuilderTrain(
        train_max_num,
        map_pt2pt_radius=map_pt2pt_radius,
        map_pt2pt_max_num_neighbors=map_pt2pt_max_num_neighbors,
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
        prefetch_factor: Optional[int],
        pin_memory: bool,
        persistent_workers: bool,
        train_max_num: int,
        train_use_eval_agent_selection: bool = False,
        train_epoch_sample_fraction: float = 1.0,
        train_memory_balanced_batches: bool = False,
        train_memory_balance_metadata_cache: Optional[str] = None,
        train_memory_balance_metadata_num_workers: int = 8,
        train_memory_balance_build_on_missing: bool = True,
        train_memory_balance_agent_weight: float = 1.0,
        train_memory_balance_current_valid_agent_weight: float = 1.0,
        train_memory_balance_valid_agent_step_weight: float = 0.0,
        train_memory_balance_map_weight: float = 0.02,
        train_memory_balance_seed: int = 0,
        map_pt2pt_cache_radius: Optional[float] = None,
        map_pt2pt_cache_max_num_neighbors: int = 100,
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
        self.train_epoch_sample_fraction = float(train_epoch_sample_fraction)
        self.train_memory_balanced_batches = bool(train_memory_balanced_batches)
        self.train_memory_balance_metadata_cache = train_memory_balance_metadata_cache
        self.train_memory_balance_metadata_num_workers = int(
            train_memory_balance_metadata_num_workers
        )
        self.train_memory_balance_build_on_missing = bool(
            train_memory_balance_build_on_missing
        )
        self.train_memory_balance_agent_weight = float(train_memory_balance_agent_weight)
        self.train_memory_balance_current_valid_agent_weight = float(
            train_memory_balance_current_valid_agent_weight
        )
        self.train_memory_balance_valid_agent_step_weight = float(
            train_memory_balance_valid_agent_step_weight
        )
        self.train_memory_balance_map_weight = float(train_memory_balance_map_weight)
        self.train_memory_balance_seed = int(train_memory_balance_seed)
        self.map_pt2pt_cache_radius = (
            None if map_pt2pt_cache_radius is None else float(map_pt2pt_cache_radius)
        )
        self.map_pt2pt_cache_max_num_neighbors = int(
            map_pt2pt_cache_max_num_neighbors
        )
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.val_tfrecords_splitted = val_tfrecords_splitted

        self.train_transform = build_train_agent_target_builder(
            train_max_num=train_max_num,
            train_use_eval_agent_selection=train_use_eval_agent_selection,
            map_pt2pt_radius=self.map_pt2pt_cache_radius,
            map_pt2pt_max_num_neighbors=self.map_pt2pt_cache_max_num_neighbors,
        )
        self.val_transform = WaymoTargetBuilderVal(
            map_pt2pt_radius=self.map_pt2pt_cache_radius,
            map_pt2pt_max_num_neighbors=self.map_pt2pt_cache_max_num_neighbors,
        )
        self.test_transform = WaymoTargetBuilderVal(
            map_pt2pt_radius=self.map_pt2pt_cache_radius,
            map_pt2pt_max_num_neighbors=self.map_pt2pt_cache_max_num_neighbors,
        )

    def refresh_train_dataset(self, train_raw_dir: Optional[str] = None) -> None:
        """학습용 cache 폴더를 바꾼 뒤 train dataset을 다시 만듭니다.

        Args:
            train_raw_dir: 새로 읽을 학습 cache 폴더입니다. 값이 없으면 현재
                ``self.train_raw_dir`` 값을 그대로 사용합니다.

        Returns:
            None
        """
        if train_raw_dir is not None:
            self.train_raw_dir = str(train_raw_dir)
        self.train_dataset = MultiDataset(self.train_raw_dir, self.train_transform)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.refresh_train_dataset()
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

    def _get_trainer_world_info(self) -> tuple[int, int]:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return 1, 0
        world_size = int(getattr(trainer, "world_size", 1) or 1)
        global_rank = int(getattr(trainer, "global_rank", 0) or 0)
        return max(1, world_size), global_rank

    def _build_train_fraction_sampler(self):
        if self.train_epoch_sample_fraction >= 1.0:
            return None
        world_size, global_rank = self._get_trainer_world_info()
        return RandomFractionDistributedSampler(
            dataset=self.train_dataset,
            fraction=self.train_epoch_sample_fraction,
            num_replicas=world_size,
            rank=global_rank,
        )

    def _build_memory_balanced_batch_sampler(self):
        world_size, global_rank = self._get_trainer_world_info()
        metadata = load_or_build_memory_metadata(
            self.train_dataset.raw_paths,
            cache_path=self.train_memory_balance_metadata_cache,
            num_workers=self.train_memory_balance_metadata_num_workers,
            build_on_missing=self.train_memory_balance_build_on_missing,
        )
        sample_weight = memory_balance_weights(
            metadata,
            agent_weight=self.train_memory_balance_agent_weight,
            current_valid_agent_weight=self.train_memory_balance_current_valid_agent_weight,
            valid_agent_step_weight=self.train_memory_balance_valid_agent_step_weight,
            map_weight=self.train_memory_balance_map_weight,
        )
        return MemoryBalancedDistributedBatchSampler(
            sample_weight=sample_weight,
            batch_size=self.train_batch_size,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=self.shuffle,
            seed=self.train_memory_balance_seed,
            fraction=self.train_epoch_sample_fraction,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if not hasattr(self, "train_dataset"):
            self.refresh_train_dataset()

        base_loader_kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }
        if self.prefetch_factor is not None:
            base_loader_kwargs["prefetch_factor"] = self.prefetch_factor

        if self.train_memory_balanced_batches:
            batch_sampler = self._build_memory_balanced_batch_sampler()
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                **base_loader_kwargs,
            )

        sampler = self._build_train_fraction_sampler()
        if sampler is None:
            world_size, global_rank = self._get_trainer_world_info()
            if world_size > 1:
                sampler = DistributedSampler(
                    dataset=self.train_dataset,
                    num_replicas=world_size,
                    rank=global_rank,
                    shuffle=self.shuffle,
                    seed=self.train_memory_balance_seed,
                    drop_last=False,
                )
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle if sampler is None else False,
            sampler=sampler,
            drop_last=False,
            **base_loader_kwargs,
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
        loader_kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "drop_last": False,
        }
        if self.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            sampler=sampler,
            **loader_kwargs,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        sampler = self._build_eval_sampler(self.test_dataset)
        loader_kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "drop_last": False,
        }
        if self.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            sampler=sampler,
            **loader_kwargs,
        )
