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

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class WaymoTargetBuilderTrain(BaseTransform):
    def __init__(self, max_num: int) -> None:
        super(WaymoTargetBuilderTrain, self).__init__()
        self.step_current = 10
        self.max_num = max_num

    def forward(self, data) -> HeteroData:
        pos = data["agent"]["position"]
        try:
            av_index = torch.where(data["agent"]["role"][:, 0])[0].item()
        except KeyError:
            # `role` 필드가 없으면, step_current에서 valid한 agent 중 첫 번째를 기준으로 잡습니다.
            # (스모크/단순 실행 목적이므로 정확도보다 실행 가능성을 우선합니다.)
            valid_mask = data["agent"]["valid_mask"]  # [n_agent, n_step]
            candidates = torch.where(valid_mask[:, self.step_current])[0]
            av_index = candidates[0].item() if candidates.numel() > 0 else 0
        distance = torch.norm(pos - pos[av_index], dim=-1)

        # we do not believe the perception out of range of 150 meters
        data["agent"]["valid_mask"] = data["agent"]["valid_mask"] & (distance < 150)

        # we do not predict vehicle too far away from ego car
        # 일부 데이터/전처리 경로에서는 `role` 필드가 없을 수 있으므로 방어적으로 처리합니다.
        try:
            role_train_mask = data["agent"]["role"].any(-1)
        except KeyError:
            role_train_mask = torch.zeros(
                (pos.shape[0],), dtype=torch.bool, device=pos.device
            )
        extra_train_mask = (distance[:, self.step_current] < 100) & (
            data["agent"]["valid_mask"][:, self.step_current + 1 :].sum(-1) >= 5
        )

        train_mask = extra_train_mask | role_train_mask
        if train_mask.sum() > self.max_num:  # too many vehicle
            _indices = torch.where(extra_train_mask & ~role_train_mask)[0]
            selected_indices = _indices[
                torch.randperm(_indices.size(0))[: self.max_num - role_train_mask.sum()]
            ]
            data["agent"]["train_mask"] = role_train_mask
            data["agent"]["train_mask"][selected_indices] = True
        else:
            data["agent"]["train_mask"] = train_mask  # [n_agent]

        return HeteroData(data)

    # torch_geometric BaseTransform이 forward를 요구하지만,
    # 기존 코드 호환을 위해 __call__도 유지합니다.
    def __call__(self, data) -> HeteroData:
        return self.forward(data)


class WaymoTargetBuilderVal(BaseTransform):
    def __init__(self) -> None:
        super(WaymoTargetBuilderVal, self).__init__()

    def forward(self, data) -> HeteroData:
        return HeteroData(data)

    def __call__(self, data) -> HeteroData:
        return self.forward(data)
