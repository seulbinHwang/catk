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
from torch_cluster import radius_graph
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from src.smart.utils import angle_between_2d_vectors, wrap_angle


MAP_PT2PT_EDGE_TYPE = ("map_save", "pt2pt", "map_save")


def _attach_map_pt2pt_cache(
    data: HeteroData,
    radius: float | None,
    max_num_neighbors: int,
) -> HeteroData:
    """Attach static map point-to-point geometry used by the map encoder.

    The cached values depend only on map geometry and the configured radius.
    Model weights, dropout, and learned map features are intentionally not
    cached.
    """
    if radius is None:
        return data
    if max_num_neighbors < 1:
        raise ValueError(
            f"map_pt2pt_max_num_neighbors must be >= 1, got {max_num_neighbors}."
        )

    pos_pt = data["map_save"]["traj_pos"][:, 0, :2].contiguous()
    orient_pt = data["map_save"]["traj_theta"].contiguous()
    num_map_nodes = int(pos_pt.shape[0])
    data["map_save"].num_nodes = num_map_nodes
    if num_map_nodes == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=pos_pt.device)
        r_raw = pos_pt.new_empty((0, 3))
    else:
        batch = torch.zeros(num_map_nodes, dtype=torch.long, device=pos_pt.device)
        edge_index = radius_graph(
            x=pos_pt,
            r=float(radius),
            batch=batch,
            loop=False,
            max_num_neighbors=int(max_num_neighbors),
        )
        rel_pos = pos_pt[edge_index[0]] - pos_pt[edge_index[1]]
        rel_orient = wrap_angle(orient_pt[edge_index[0]] - orient_pt[edge_index[1]])
        orient_vector = torch.stack([orient_pt.cos(), orient_pt.sin()], dim=-1)
        r_raw = torch.stack(
            [
                torch.norm(rel_pos[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=orient_vector[edge_index[1]],
                    nbr_vector=rel_pos[:, :2],
                ),
                rel_orient,
            ],
            dim=-1,
        )

    data[MAP_PT2PT_EDGE_TYPE]["edge_index"] = edge_index.contiguous()
    data[MAP_PT2PT_EDGE_TYPE]["r_raw"] = r_raw.contiguous()
    data[MAP_PT2PT_EDGE_TYPE]["radius"] = pos_pt.new_tensor(float(radius))
    data[MAP_PT2PT_EDGE_TYPE]["max_num_neighbors"] = torch.tensor(
        int(max_num_neighbors), dtype=torch.long, device=pos_pt.device
    )
    return data


class WaymoTargetBuilderTrain(BaseTransform):
    def __init__(
        self,
        max_num: int,
        map_pt2pt_radius: float | None = None,
        map_pt2pt_max_num_neighbors: int = 100,
    ) -> None:
        super(WaymoTargetBuilderTrain, self).__init__()
        self.step_current = 10
        self.max_num = max_num
        self.map_pt2pt_radius = map_pt2pt_radius
        self.map_pt2pt_max_num_neighbors = int(map_pt2pt_max_num_neighbors)

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

        return _attach_map_pt2pt_cache(
            HeteroData(data),
            radius=self.map_pt2pt_radius,
            max_num_neighbors=self.map_pt2pt_max_num_neighbors,
        )


class WaymoTargetBuilderVal(BaseTransform):
    def __init__(
        self,
        map_pt2pt_radius: float | None = None,
        map_pt2pt_max_num_neighbors: int = 100,
    ) -> None:
        super(WaymoTargetBuilderVal, self).__init__()
        self.map_pt2pt_radius = map_pt2pt_radius
        self.map_pt2pt_max_num_neighbors = int(map_pt2pt_max_num_neighbors)

    def forward(self, data) -> HeteroData:
        return _attach_map_pt2pt_cache(
            HeteroData(data),
            radius=self.map_pt2pt_radius,
            max_num_neighbors=self.map_pt2pt_max_num_neighbors,
        )
