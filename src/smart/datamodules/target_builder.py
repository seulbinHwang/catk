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

from pathlib import Path
from typing import Optional

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from src.smart.tokens.token_processor import (
    AGENT_TOKEN_SIDECAR_FIELDS,
    AGENT_TOKEN_SIDECAR_VERSION,
)


class WaymoTargetBuilderTrain(BaseTransform):
    def __init__(
        self,
        max_num: int,
        agent_token_sidecar_dir: Optional[str] = None,
        agent_token_sidecar_required: bool = False,
        agent_token_sidecar_version: str = AGENT_TOKEN_SIDECAR_VERSION,
    ) -> None:
        super(WaymoTargetBuilderTrain, self).__init__()
        self.step_current = 10
        self.max_num = max_num
        self.agent_token_sidecar_dir = (
            Path(agent_token_sidecar_dir) if agent_token_sidecar_dir else None
        )
        self.agent_token_sidecar_required = bool(agent_token_sidecar_required)
        self.agent_token_sidecar_version = agent_token_sidecar_version

    def forward(self, data) -> HeteroData:
        pos = data["agent"]["position"]
        av_index = torch.where(data["agent"]["role"][:, 0])[0].item()
        distance = torch.norm(pos - pos[av_index], dim=-1)

        # we do not believe the perception out of range of 150 meters
        data["agent"]["valid_mask"] = data["agent"]["valid_mask"] & (distance < 150)

        # we do not predict vehicle too far away from ego car
        role_train_mask = data["agent"]["role"].any(-1)
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

        self._attach_agent_token_sidecar(data)
        return HeteroData(data)

    def _attach_agent_token_sidecar(self, data) -> None:
        if self.agent_token_sidecar_dir is None:
            return
        scenario_id = str(data["scenario_id"])
        sidecar_path = self.agent_token_sidecar_dir / f"{scenario_id}.pt"
        if not sidecar_path.exists():
            if self.agent_token_sidecar_required:
                raise FileNotFoundError(f"Missing agent token sidecar: {sidecar_path}")
            return

        payload = self._torch_load(sidecar_path)
        metadata = payload.get("metadata", {})
        if metadata.get("version") != self.agent_token_sidecar_version:
            raise RuntimeError(
                f"Agent token sidecar version mismatch for {scenario_id}: "
                f"{metadata.get('version')!r} != {self.agent_token_sidecar_version!r}"
            )
        n_agent = int(data["agent"]["position"].shape[0])
        if int(payload.get("num_agents", -1)) != n_agent:
            raise RuntimeError(
                f"Agent token sidecar agent count mismatch for {scenario_id}: "
                f"{payload.get('num_agents')} != {n_agent}"
            )

        agent_payload = payload["agent"]
        for field in AGENT_TOKEN_SIDECAR_FIELDS:
            if field not in agent_payload:
                raise RuntimeError(
                    f"Agent token sidecar for {scenario_id} is missing field {field!r}."
                )
            data["agent"][f"token_sidecar_{field}"] = agent_payload[field]

    @staticmethod
    def _torch_load(path: Path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


class WaymoTargetBuilderVal(BaseTransform):
    def __init__(self) -> None:
        super(WaymoTargetBuilderVal, self).__init__()

    def forward(self, data) -> HeteroData:
        return HeteroData(data)
