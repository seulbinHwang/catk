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

import pickle
from pathlib import Path
from typing import Callable, List, Optional

from torch_geometric.data import Dataset

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

_REQUIRED_CACHE_KEYS = (
    "agent",
    "map_save",
    "pt_token",
    "polygon_token",
    "scenario_id",
)
_REQUIRED_POLYGON_TOKEN_KEYS = (
    "position",
    "orientation",
    "size",
    "type",
    "boundary",
    "num_nodes",
)


class MultiDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        transform: Callable,
        tfrecord_dir: Optional[str] = None,
    ) -> None:
        raw_dir = Path(raw_dir)
        if not raw_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {raw_dir}")
        if not raw_dir.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {raw_dir}")
        self._raw_paths = [p.as_posix() for p in sorted(raw_dir.glob("*"))]
        self._num_samples = len(self._raw_paths)

        self._tfrecord_dir = Path(tfrecord_dir) if tfrecord_dir is not None else None
        if self._tfrecord_dir is not None and not self._tfrecord_dir.exists():
            raise FileNotFoundError(
                f"TFRecord directory does not exist: {self._tfrecord_dir}"
            )
        if self._num_samples == 0:
            raise FileNotFoundError(f"No cached samples found under: {raw_dir}")

        log.info("Length of {} dataset is ".format(raw_dir) + str(self._num_samples))
        super(MultiDataset, self).__init__(
            transform=transform, pre_transform=None, pre_filter=None
        )

    @property
    def raw_paths(self) -> List[str]:
        return self._raw_paths

    def len(self) -> int:
        return self._num_samples

    def _validate_cache_schema(self, data, raw_path: str) -> None:
        missing_top_level = [key for key in _REQUIRED_CACHE_KEYS if key not in data]
        if missing_top_level:
            missing_str = ", ".join(missing_top_level)
            raise RuntimeError(
                "Cached sample is missing required keys "
                f"({missing_str}) at {raw_path}. "
                "This usually means the cache was generated before the current polygon-map "
                "patch. Regenerate every split with the current src.data_preprocess."
            )

        polygon_token = data["polygon_token"]
        if not isinstance(polygon_token, dict):
            raise RuntimeError(
                f"Cached sample has invalid polygon_token store at {raw_path}. "
                "Regenerate every split with the current src.data_preprocess."
            )

        missing_polygon_keys = [
            key for key in _REQUIRED_POLYGON_TOKEN_KEYS if key not in polygon_token
        ]
        if missing_polygon_keys:
            missing_str = ", ".join(missing_polygon_keys)
            raise RuntimeError(
                "Cached sample has incomplete polygon_token store "
                f"({missing_str}) at {raw_path}. "
                "Regenerate every split with the current src.data_preprocess."
            )

    def get(self, idx: int):
        with open(self.raw_paths[idx], "rb") as handle:
            data = pickle.load(handle)

        self._validate_cache_schema(data, self.raw_paths[idx])

        if self._tfrecord_dir is not None:
            data["tfrecord_path"] = (
                self._tfrecord_dir / (data["scenario_id"] + ".tfrecords")
            ).as_posix()
        return data
