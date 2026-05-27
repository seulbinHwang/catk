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
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, List, Optional

from torch_geometric.data import Dataset

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def is_cache_sample_path(path: Path) -> bool:
    """Return true only for visible SMART scenario cache pickle files."""

    return path.is_file() and path.suffix == ".pkl" and not path.name.startswith(".")


class MultiDataset(Dataset):
    def __init__(
        self,
        raw_dir: str | Sequence[str],
        transform: Callable,
        tfrecord_dir: Optional[str] = None,
    ) -> None:
        raw_dirs = [raw_dir] if isinstance(raw_dir, str) else list(raw_dir)
        if not raw_dirs:
            raise ValueError("At least one dataset directory must be provided.")

        self._raw_dirs = []
        self._raw_paths = []
        for raw_dir_item in raw_dirs:
            raw_dir_path = Path(raw_dir_item)
            if not raw_dir_path.exists():
                raise FileNotFoundError(f"Dataset directory does not exist: {raw_dir_path}")
            if not raw_dir_path.is_dir():
                raise NotADirectoryError(
                    f"Dataset path is not a directory: {raw_dir_path}"
                )
            self._raw_dirs.append(raw_dir_path)
            self._raw_paths.extend(
                p.as_posix()
                for p in sorted(raw_dir_path.glob("*.pkl"))
                if is_cache_sample_path(p)
            )
        self._num_samples = len(self._raw_paths)

        self._tfrecord_dir = Path(tfrecord_dir) if tfrecord_dir is not None else None
        if self._tfrecord_dir is not None and not self._tfrecord_dir.exists():
            raise FileNotFoundError(
                f"TFRecord directory does not exist: {self._tfrecord_dir}"
            )
        if self._num_samples == 0:
            raise FileNotFoundError(f"No cached samples found under: {self._raw_dirs}")

        log.info(
            "Length of {} dataset is ".format(
                ",".join(path.as_posix() for path in self._raw_dirs)
            )
            + str(self._num_samples)
        )
        super(MultiDataset, self).__init__(
            transform=transform, pre_transform=None, pre_filter=None
        )

    @property
    def raw_paths(self) -> List[str]:
        return self._raw_paths

    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int):
        with open(self.raw_paths[idx], "rb") as handle:
            data = pickle.load(handle)

        if self._tfrecord_dir is not None:
            data["tfrecord_path"] = (
                self._tfrecord_dir / (data["scenario_id"] + ".tfrecords")
            ).as_posix()
        return data
