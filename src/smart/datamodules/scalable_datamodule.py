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

import hashlib
from pathlib import Path
from typing import Optional

import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Batch
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


AUTO_FLOW_TARGET_SIDECAR_ROOT = "auto"
FLOW_TARGET_SIDECAR_ROW_ORDER_ANCHOR_MAJOR = "anchor_major_v1"


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


class SequentialTransform(BaseTransform):
    def __init__(self, *transforms: BaseTransform) -> None:
        super().__init__()
        self.transforms = tuple(transform for transform in transforms if transform is not None)

    def forward(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class FlowTargetSidecarPayloadTransform(BaseTransform):
    """Load precomputed Flow training targets in DataLoader workers.

    The model-side token processor still owns validation and collation. This
    transform only moves deterministic sidecar file IO out of the training step
    so worker prefetch can overlap it with GPU compute.
    """

    def __init__(self, sidecar_root: str, required: bool = False) -> None:
        super().__init__()
        self.sidecar_root = Path(sidecar_root) if sidecar_root else None
        self.required = bool(required)

    def _sidecar_path(self, scenario_id: str) -> Path:
        safe_hash = hashlib.sha1(str(scenario_id).encode("utf-8")).hexdigest()
        return self.sidecar_root / f"{safe_hash}.pt"

    @staticmethod
    def _scenario_id_from_data(data) -> str:
        scenario_id = getattr(data, "scenario_id", None)
        if scenario_id is None and isinstance(data, dict):
            scenario_id = data.get("scenario_id")
        if isinstance(scenario_id, (list, tuple)):
            if len(scenario_id) != 1:
                raise ValueError(f"Expected one scenario id, got {scenario_id!r}")
            scenario_id = scenario_id[0]
        if scenario_id is None:
            raise KeyError("Sample does not contain scenario_id.")
        return str(scenario_id)

    def forward(self, data):
        if self.sidecar_root is None:
            return data
        scenario_id = self._scenario_id_from_data(data)
        path = self._sidecar_path(scenario_id)
        if not path.exists():
            if self.required:
                raise FileNotFoundError(f"Missing flow target sidecar: {path}")
            return data
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        metadata = payload.get("metadata", {})
        if str(metadata.get("scenario_id")) != scenario_id:
            if self.required:
                raise ValueError(
                    f"Flow target sidecar scenario mismatch: expected={scenario_id}, "
                    f"actual={metadata.get('scenario_id')}"
                )
            return data
        data.flow_target_sidecar_payload = payload
        return data


class FlowTargetSidecarCollater:
    """Collate sidecar payload rows into the order consumed by FlowTokenProcessor."""

    map_keys = ("position", "orientation", "token_idx", "type", "pl_type", "light_type")
    per_agent_keys = (
        "type",
        "shape",
        "ego_mask",
        "token_agent_shape",
        "ctx_sampled_idx",
        "ctx_sampled_pos",
        "ctx_sampled_heading",
        "ctx_valid",
        "flow_train_mask",
    )
    row_keys = (
        "flow_train_clean_norm",
        "flow_train_clean_metric_norm",
        "flow_train_loss_mask",
        "flow_train_agent_type",
        "flow_train_agent_length",
    )

    def __init__(self, required: bool = True) -> None:
        self.required = bool(required)

    def __call__(self, data_list):
        payloads = [
            getattr(data, "flow_target_sidecar_payload", None)
            for data in data_list
        ]
        if not any(payload is not None for payload in payloads):
            return Batch.from_data_list(data_list)
        if any(payload is None for payload in payloads):
            if self.required:
                raise RuntimeError("Missing flow target sidecar payload during batch collation.")
            return Batch.from_data_list(data_list)

        batch = Batch.from_data_list(
            data_list,
            exclude_keys=["flow_target_sidecar_payload"],
        )
        batch.flow_target_sidecar_payload = self._collate_payloads(payloads)
        return batch

    @staticmethod
    def _metadata_list(payloads, key: str) -> list:
        return [payload.get("metadata", {}).get(key) for payload in payloads]

    @staticmethod
    def _cat_present(payloads, section: str, keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
        values: dict[str, torch.Tensor] = {}
        for key in keys:
            if all(key in payload[section] for payload in payloads):
                values[key] = torch.cat([payload[section][key] for payload in payloads], dim=0)
        return values

    @staticmethod
    def _split_flow_rows(agent_payload: dict, key: str) -> list[torch.Tensor]:
        counts = agent_payload["flow_train_mask"].long().sum(dim=0).tolist()
        value = agent_payload[key]
        chunks: list[torch.Tensor] = []
        cursor = 0
        for count in counts:
            next_cursor = cursor + int(count)
            chunks.append(value[cursor:next_cursor])
            cursor = next_cursor
        if cursor != int(value.shape[0]):
            raise ValueError(
                f"Flow target sidecar row count mismatch for {key}: "
                f"mask_count={cursor}, value_rows={int(value.shape[0])}."
            )
        return chunks

    def _collate_agent_rows(self, payloads) -> dict[str, torch.Tensor]:
        agent_payloads = [payload["agent"] for payload in payloads]
        collated = self._cat_present(payloads, "agent", self.per_agent_keys)
        if not agent_payloads:
            return collated
        num_anchor = int(agent_payloads[0]["flow_train_mask"].shape[1])
        for key in self.row_keys:
            if not all(key in item for item in agent_payloads):
                continue
            split_rows = [
                self._split_flow_rows(agent_payload, key)
                for agent_payload in agent_payloads
            ]
            ordered_parts: list[torch.Tensor] = []
            for anchor_idx in range(num_anchor):
                for sample_idx in range(len(agent_payloads)):
                    part = split_rows[sample_idx][anchor_idx]
                    if int(part.shape[0]) > 0:
                        ordered_parts.append(part)
            if ordered_parts:
                collated[key] = torch.cat(ordered_parts, dim=0).contiguous()
            else:
                example = agent_payloads[0][key]
                collated[key] = example.new_zeros((0,) + tuple(example.shape[1:]))
        return collated

    def _collate_payloads(self, payloads) -> dict:
        return {
            "metadata": {
                "version": 1,
                "scenario_id": self._metadata_list(payloads, "scenario_id"),
                "fingerprint": self._metadata_list(payloads, "fingerprint"),
                "flow_row_order": FLOW_TARGET_SIDECAR_ROW_ORDER_ANCHOR_MAJOR,
            },
            "map": self._cat_present(payloads, "map", self.map_keys),
            "agent": self._collate_agent_rows(payloads),
        }


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
        train_epoch_sample_fraction_shuffle_flag: bool = False,
        train_memory_balanced_batches: bool = False,
        train_memory_balance_metadata_cache: Optional[str] = None,
        train_memory_balance_metadata_num_workers: int = 8,
        train_memory_balance_build_on_missing: bool = True,
        train_memory_balance_agent_weight: float = 1.0,
        train_memory_balance_current_valid_agent_weight: float = 1.0,
        train_memory_balance_valid_agent_step_weight: float = 0.0,
        train_memory_balance_map_weight: float = 0.02,
        train_memory_balance_seed: int = 0,
        train_flow_target_sidecar_root: Optional[str] = AUTO_FLOW_TARGET_SIDECAR_ROOT,
        train_flow_target_sidecar_required: bool = True,
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
        self.train_epoch_sample_fraction_shuffle_flag = bool(
            train_epoch_sample_fraction_shuffle_flag
        )
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
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.val_tfrecords_splitted = val_tfrecords_splitted

        self.train_flow_target_sidecar_root = self._normalize_train_flow_target_sidecar_root(
            train_flow_target_sidecar_root
        )
        self.train_flow_target_sidecar_required = bool(train_flow_target_sidecar_required)

        self.train_target_transform = build_train_agent_target_builder(
            train_max_num=train_max_num,
            train_use_eval_agent_selection=train_use_eval_agent_selection,
        )
        self.train_transform = self._build_train_transform()
        self.val_transform = WaymoTargetBuilderVal()
        self.test_transform = WaymoTargetBuilderVal()

    @staticmethod
    def _normalize_train_flow_target_sidecar_root(value: Optional[str]) -> str:
        if value in (None, ""):
            return ""
        text = str(value)
        if text.strip().lower() == AUTO_FLOW_TARGET_SIDECAR_ROOT:
            return ""
        return text

    def _build_train_transform(self) -> BaseTransform:
        if self.train_flow_target_sidecar_root:
            return SequentialTransform(
                self.train_target_transform,
                FlowTargetSidecarPayloadTransform(
                    sidecar_root=self.train_flow_target_sidecar_root,
                    required=self.train_flow_target_sidecar_required,
                ),
            )
        return self.train_target_transform

    def configure_train_flow_target_sidecar(
        self,
        sidecar_root: str,
        *,
        required: Optional[bool] = None,
    ) -> None:
        """Enable DataLoader-worker sidecar preload with a resolved fingerprint root."""

        resolved_root = self._normalize_train_flow_target_sidecar_root(sidecar_root)
        if not resolved_root:
            return
        previous_root = self.train_flow_target_sidecar_root
        previous_required = self.train_flow_target_sidecar_required
        if required is not None:
            self.train_flow_target_sidecar_required = bool(required)
        if (
            previous_root == resolved_root
            and previous_required == self.train_flow_target_sidecar_required
        ):
            return
        self.train_flow_target_sidecar_root = resolved_root
        self.train_transform = self._build_train_transform()
        if hasattr(self, "train_dataset"):
            self.refresh_train_dataset()

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
            shuffle_fraction_each_epoch=self.train_epoch_sample_fraction_shuffle_flag,
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
            shuffle_fraction_each_epoch=self.train_epoch_sample_fraction_shuffle_flag,
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
            if self.train_flow_target_sidecar_root:
                return TorchDataLoader(
                    self.train_dataset,
                    batch_sampler=batch_sampler,
                    collate_fn=FlowTargetSidecarCollater(
                        required=self.train_flow_target_sidecar_required,
                    ),
                    **base_loader_kwargs,
                )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                **base_loader_kwargs,
            )

        sampler = self._build_train_fraction_sampler()
        if self.train_flow_target_sidecar_root:
            return TorchDataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                shuffle=self.shuffle if sampler is None else False,
                sampler=sampler,
                drop_last=False,
                collate_fn=FlowTargetSidecarCollater(
                    required=self.train_flow_target_sidecar_required,
                ),
                **base_loader_kwargs,
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
