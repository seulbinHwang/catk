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


class MultiDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        transform: Callable,
        tfrecord_dir: Optional[str] = None,
    ) -> None:
        raw_dir = Path(raw_dir)
        all_paths = [p for p in sorted(raw_dir.glob("*")) if p.is_file()]
        # 0-byte 파일은 pickle.load에서 EOFError를 내므로 시작 시점에 제외합니다.
        zero_size_paths = [p for p in all_paths if p.stat().st_size == 0]
        if len(zero_size_paths) > 0:
            preview = ", ".join(p.name for p in zero_size_paths[:5])
            log.warning(
                f"Skipping {len(zero_size_paths)} zero-byte samples under {raw_dir} (e.g., {preview})"
            )
        self._raw_paths = [p.as_posix() for p in all_paths if p.stat().st_size > 0]
        self._num_samples = len(self._raw_paths)
        self._bad_indices: set[int] = set()

        self._tfrecord_dir = Path(tfrecord_dir) if tfrecord_dir is not None else None

        log.info("Length of {} dataset is ".format(raw_dir) + str(self._num_samples))
        super(MultiDataset, self).__init__(
            transform=transform, pre_transform=None, pre_filter=None
        )

    @property
    def raw_paths(self) -> List[str]:
        return self._raw_paths

    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int):
        if self._num_samples == 0:
            raise RuntimeError("No valid samples found in dataset.")

        max_retry = min(self._num_samples, 32)
        for retry in range(max_retry):
            cur_idx = (idx + retry) % self._num_samples
            if cur_idx in self._bad_indices:
                continue
            sample_path = self.raw_paths[cur_idx]
            try:
                with open(sample_path, "rb") as handle:
                    data = pickle.load(handle)
                # 일부 캐시는 top-level에 `city` 키가 있고, 일부는 없어
                # PyG collate에서 KeyError를 유발할 수 있으므로 통일 제거합니다.
                data.pop("city", None)
                if self._tfrecord_dir is not None:
                    scenario_id = data["scenario_id"]
                    data["tfrecord_path"] = (
                        self._tfrecord_dir / (scenario_id + ".tfrecords")
                    ).as_posix()
                return data
            except Exception as error:
                self._bad_indices.add(cur_idx)
                log.warning(
                    f"Skipping corrupted sample idx={cur_idx} path={sample_path} "
                    f"due to {type(error).__name__}: {error}"
                )
                continue

        raise RuntimeError(
            "Failed to fetch a valid sample after retries. "
            f"requested_idx={idx}, num_samples={self._num_samples}, bad_cached={len(self._bad_indices)}"
        )
