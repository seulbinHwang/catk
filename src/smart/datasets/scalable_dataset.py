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
import random
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from torch_geometric.data import Dataset

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

RawDir = Union[str, PathLike[str], Sequence[Union[str, PathLike[str]]]]


def normalize_raw_dirs(raw_dir: RawDir) -> List[Path]:
    if isinstance(raw_dir, (str, PathLike)):
        return [Path(raw_dir)]
    return [Path(path) for path in raw_dir]


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
        raw_dir: RawDir,
        transform: Callable,
        tfrecord_dir: Optional[str] = None,
        road_num_rollouts_per_scenario: int = 1,
        random_scene_scale_config: Optional[dict] = None,
        random_time_shift_config: Optional[dict] = None,
    ) -> None:
        raw_dirs = normalize_raw_dirs(raw_dir)
        self._raw_paths = []
        for raw_dir_path in raw_dirs:
            self._raw_paths.extend(
                p.as_posix()
                for p in sorted(raw_dir_path.glob("*"))
                if p.is_file() and not p.name.startswith(".")
            )
        self._road_num_rollouts_per_scenario = road_num_rollouts_per_scenario
        self._raw_path_groups: Optional[List[List[str]]] = None

        if road_num_rollouts_per_scenario > 1:
            self._raw_path_groups = group_road_raw_paths(
                self._raw_paths, road_num_rollouts_per_scenario
            )
            self._num_samples = len(self._raw_path_groups)
        else:
            self._num_samples = len(self._raw_paths)

        self._tfrecord_dir = Path(tfrecord_dir) if tfrecord_dir is not None else None
        self.random_scene_scale_config = random_scene_scale_config
        self.random_time_shift_config = random_time_shift_config

        raw_dir_label = ", ".join(path.as_posix() for path in raw_dirs)
        log.info(
            "Length of {} dataset is ".format(raw_dir_label) + str(self._num_samples)
        )
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
            idx: dataset index이다. RoaD cache에서는 원본 scenario index를 뜻한다.

        Returns:
            실제로 열 pickle 파일 경로이다. RoaD cache에서는 같은 scenario의 3개
            rollout 중 하나를 균등하게 고른다.
        """
        if self._raw_path_groups is None:
            return self._raw_paths[idx]
        rollout_paths = self._raw_path_groups[idx]
        return rollout_paths[random.randrange(len(rollout_paths))]

    def get(self, idx: int):
        raw_path = self._select_raw_path(idx)
        with open(raw_path, "rb") as handle:
            data = pickle.load(handle)

        if self._tfrecord_dir is not None:
            data["tfrecord_path"] = (
                self._tfrecord_dir / (data["scenario_id"] + ".tfrecords")
            ).as_posix()

        if self.random_scene_scale_config is not None:
            data = self.random_scene_scale(self.random_scene_scale_config, data)
        if self.random_time_shift_config is not None:
            data = self.random_time_shift(self.random_time_shift_config, data)
        return data

    @staticmethod
    def random_scene_scale(config: dict, data):
        """공식 TrajTok train recipe와 같은 scene scale augmentation이다."""
        scale_range = config["SCALE_RANGE"]
        scale = np.random.uniform(scale_range[0], scale_range[1])
        data["map_save"]["traj_pos"] *= scale
        data["agent"]["position"][:, :, 0:2] *= scale
        data["agent"]["velocity"][:, :, 0:2] *= scale
        return data

    @staticmethod
    def random_time_shift(config: dict, data):
        """예측 대상 agent가 유효한 current 주변 시점으로 train sample을 이동한다."""
        max_time_shift = int(config["MAX_TIME_SHIFT"])
        if max_time_shift <= 0:
            return data

        track_to_predict = data["agent"]["role"][:, 2]
        if not bool(track_to_predict.any()):
            return data

        valid_time_mask = data["agent"]["valid_mask"][track_to_predict][
            :, 10 - max_time_shift : 10 + max_time_shift
        ]
        valid_time_offset = valid_time_mask.all(dim=0).nonzero().reshape(-1)
        if valid_time_offset.numel() == 0:
            return data

        choice = np.random.choice(valid_time_offset.detach().cpu().numpy())
        time_shift = int(choice) - max_time_shift
        if time_shift > 0:
            data["agent"]["position"][:, :-time_shift, :] = data["agent"]["position"][
                :, time_shift:, :
            ].clone()
            data["agent"]["velocity"][:, :-time_shift, :] = data["agent"]["velocity"][
                :, time_shift:, :
            ].clone()
            data["agent"]["heading"][:, :-time_shift] = data["agent"]["heading"][
                :, time_shift:
            ].clone()
            data["agent"]["position"][:, -time_shift:, :] = 0
            data["agent"]["velocity"][:, -time_shift:, :] = 0
            data["agent"]["heading"][:, -time_shift:] = 0
            data["agent"]["valid_mask"][:, -time_shift:] = False
        elif time_shift < 0:
            time_shift = abs(time_shift)
            data["agent"]["position"][:, time_shift:, :] = data["agent"]["position"][
                :, :-time_shift, :
            ].clone()
            data["agent"]["velocity"][:, time_shift:, :] = data["agent"]["velocity"][
                :, :-time_shift, :
            ].clone()
            data["agent"]["heading"][:, time_shift:] = data["agent"]["heading"][
                :, :-time_shift
            ].clone()
            data["agent"]["position"][:, :time_shift, :] = 0
            data["agent"]["velocity"][:, :time_shift, :] = 0
            data["agent"]["heading"][:, :time_shift] = 0
            data["agent"]["valid_mask"][:, :time_shift] = False
        return data
