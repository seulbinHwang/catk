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
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform

from src.smart.datasets import MultiDataset

from .exact_distributed_sampler import ExactDistributedSampler
from .random_fraction_distributed_sampler import RandomFractionDistributedSampler
from .target_builder import WaymoTargetBuilderTrain, WaymoTargetBuilderVal


def build_train_agent_target_builder(
    train_max_num: int,
    train_use_eval_agent_selection: bool,
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
        return WaymoTargetBuilderVal()
    return WaymoTargetBuilderTrain(train_max_num)


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
        train_tfrecords_splitted: Optional[str] = None,
        train_use_val_transform: bool = False,
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
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.val_tfrecords_splitted = val_tfrecords_splitted
        # project_3 train pipeline 은 pickle cache 기반이라 TFRecord 입력은
        # 사용하지 않습니다. OCSC launcher 는 호환성을 위해 이 인자를 inject
        # 하므로 받기만 하고 별도 사용은 안 합니다.
        self.train_tfrecords_splitted = train_tfrecords_splitted

        self.val_transform = WaymoTargetBuilderVal()
        self.test_transform = WaymoTargetBuilderVal()
        if train_use_val_transform:
            # OCSC 학습은 train/val transform 을 동일하게 두어 covariate shift
            # 비교를 안정시킨다. 이 분기에선 val transform 을 train transform
            # 으로 그대로 쓴다.
            self.train_transform = self.val_transform
        else:
            self.train_transform = build_train_agent_target_builder(
                train_max_num=train_max_num,
                train_use_eval_agent_selection=train_use_eval_agent_selection,
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
        # OCSC fine-tuning 은 dataset[i] 에 tfrecord_path / scenario_id 가 필요해
        # train_tfrecords_splitted 가 set 된 경우 train dataset 에도 tfrecord_dir
        # 를 넘긴다. 일반 pretraining 은 train_tfrecords_splitted=None 이라
        # 기존 동작과 동일하다.
        self.train_dataset = MultiDataset(
            self.train_raw_dir,
            self.train_transform,
            tfrecord_dir=self.train_tfrecords_splitted,
        )

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

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if not hasattr(self, "train_dataset"):
            self.refresh_train_dataset()

        loader_kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "drop_last": False,
        }
        if self.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        sampler = self._build_train_fraction_sampler()
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle if sampler is None else False,
            sampler=sampler,
            **loader_kwargs,
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
