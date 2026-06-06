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


def is_cache_sample_path(path: Path) -> bool:
    """Return true only for visible SMART scenario cache pickle files."""

    return path.is_file() and path.suffix == ".pkl" and not path.name.startswith(".")


class MultiDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        transform: Callable,
        tfrecord_dir: Optional[str] = None,
        sidecar_dir: Optional[str] = None,
    ) -> None:
        raw_dir = Path(raw_dir)
        if not raw_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {raw_dir}")
        if not raw_dir.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {raw_dir}")
        self._raw_paths = [
            p.as_posix()
            for p in sorted(raw_dir.glob("*.pkl"))
            if is_cache_sample_path(p)
        ]
        self._num_samples = len(self._raw_paths)

        self._tfrecord_dir = Path(tfrecord_dir) if tfrecord_dir is not None else None
        if self._tfrecord_dir is not None and not self._tfrecord_dir.exists():
            raise FileNotFoundError(
                f"TFRecord directory does not exist: {self._tfrecord_dir}"
            )
        self._sidecar_dir = Path(sidecar_dir) if sidecar_dir is not None else None
        if self._sidecar_dir is not None and not self._sidecar_dir.exists():
            raise FileNotFoundError(
                f"Sidecar directory does not exist: {self._sidecar_dir}"
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

    def _sidecar_path(self, data: dict, raw_path: str) -> Path:
        scenario_id = str(data.get("scenario_id") or Path(raw_path).stem)
        candidates = [
            self._sidecar_dir / f"{scenario_id}.pkl",
            self._sidecar_dir / Path(raw_path).name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Semi-MDG sidecar is enabled but no sidecar file was found for "
            f"scenario_id={scenario_id}. Checked: "
            + ", ".join(str(path) for path in candidates)
        )

    @staticmethod
    def _attach_sidecar(data: dict, sidecar: dict) -> None:
        if sidecar.get("version") != "semi_mdg_token_flow_sidecar_v1":
            raise ValueError(
                "Unsupported Semi-MDG sidecar version: "
                f"{sidecar.get('version')!r}."
            )
        for store_name, prefix in [("agent", "semi_mdg_sidecar_"), ("pt_token", "semi_mdg_sidecar_")]:
            store = sidecar.get(store_name, {})
            if not store:
                continue
            if store_name not in data:
                raise KeyError(f"Cache sample is missing store required by sidecar: {store_name}")
            for key, value in store.items():
                data[store_name][f"{prefix}{key}"] = value

    def get(self, idx: int):
        raw_path = self.raw_paths[idx]
        with open(raw_path, "rb") as handle:
            data = pickle.load(handle)

        if self._sidecar_dir is not None:
            with open(self._sidecar_path(data, raw_path), "rb") as handle:
                sidecar = pickle.load(handle)
            self._attach_sidecar(data, sidecar)

        if self._tfrecord_dir is not None:
            data["tfrecord_path"] = (
                self._tfrecord_dir / (data["scenario_id"] + ".tfrecords")
            ).as_posix()
        return data
