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
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

from torch_geometric.data import Dataset

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def get_road_group_key(raw_path: str) -> str:
    """RoaD rollout 파일 이름에서 원본 scenario 이름을 꺼낸다.

    Args:
        raw_path: pickle 파일 경로이다. RoaD 파일은
            ``<scenario_id>__road_rXX.pkl`` 형태를 따른다.

    Returns:
        같은 원본 scenario에서 나온 rollout들이 공유하는 이름이다.
    """
    stem = Path(raw_path).stem
    marker = "__road_r"
    if marker not in stem:
        return stem
    return stem.split(marker)[0]


def group_road_raw_paths(raw_paths: List[str], group_size: int) -> List[List[str]]:
    """scenario별 RoaD rollout 파일을 하나의 묶음으로 정리한다.

    Args:
        raw_paths: RoaD cache 디렉터리 안의 pickle 파일 경로 목록이다.
        group_size: scenario 하나당 저장된 rollout 개수이다.

    Returns:
        원본 scenario 단위로 묶은 파일 경로 목록이다. 바깥 list 길이는 원본
        scenario 수이고, 안쪽 list 길이는 ``group_size``이다.
    """
    grouped_paths: Dict[str, List[str]] = defaultdict(list)
    for raw_path in raw_paths:
        grouped_paths[get_road_group_key(raw_path)].append(raw_path)

    path_groups = []
    for scenario_key in sorted(grouped_paths.keys()):
        paths = sorted(grouped_paths[scenario_key])
        if len(paths) != group_size:
            raise ValueError(
                f"RoaD cache group '{scenario_key}' has {len(paths)} files, "
                f"but road_num_rollouts_per_scenario={group_size}."
            )
        path_groups.append(paths)
    return path_groups


class MultiDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        transform: Callable,
        tfrecord_dir: Optional[str] = None,
        road_num_rollouts_per_scenario: int = 1,
    ) -> None:
        raw_dir = Path(raw_dir)
        self._raw_paths = [
            p.as_posix()
            for p in sorted(raw_dir.glob("*"))
            if p.is_file() and not p.name.startswith(".")
        ]
        self._road_num_rollouts_per_scenario = road_num_rollouts_per_scenario

        if road_num_rollouts_per_scenario > 1:
            group_road_raw_paths(self._raw_paths, road_num_rollouts_per_scenario)
        self._num_samples = len(self._raw_paths)

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

    def _select_raw_path(self, idx: int) -> str:
        """학습에 사용할 pickle 하나를 고른다.

        Args:
            idx: dataset index이다. RoaD cache에서는 rollout 파일 index를 뜻한다.

        Returns:
            실제로 열 pickle 파일 경로이다. RoaD cache에서는 같은 scenario의
            rollout 3개가 각각 독립 학습 sample이 된다.
        """
        return self._raw_paths[idx]

    def get(self, idx: int):
        raw_path = self._select_raw_path(idx)
        with open(raw_path, "rb") as handle:
            data = pickle.load(handle)

        if self._tfrecord_dir is not None:
            data["tfrecord_path"] = (
                self._tfrecord_dir / (data["scenario_id"] + ".tfrecords")
            ).as_posix()
        return data
