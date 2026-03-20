# Not a contribution
#
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

from typing import Any, Dict

import numpy as np
import torch
from scipy.interpolate import interp1d


def get_polylines_from_polygon(polygon: np.ndarray) -> np.ndarray:
    # polygon: [4, 3]
    l1 = np.linalg.norm(polygon[1, :2] - polygon[0, :2])
    l2 = np.linalg.norm(polygon[2, :2] - polygon[1, :2])

    def _pl_interp_start_end(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        length = np.linalg.norm(start - end)
        unit_vec = (end - start) / length
        pl = []
        for i in range(int(length) + 1):  # 4.5 -> 5 [0, 1, 2, 3, 4]
            x, y, z = start + unit_vec * i
            pl.append([x, y, z])
        pl.append([end[0], end[1], end[2]])
        return np.array(pl)

    if l1 > l2:
        pl1 = _pl_interp_start_end(polygon[0], polygon[1])
        pl2 = _pl_interp_start_end(polygon[2], polygon[3])
    else:
        pl1 = _pl_interp_start_end(polygon[0], polygon[3])
        pl2 = _pl_interp_start_end(polygon[2], polygon[1])
    return np.concatenate([pl1, pl1[::-1], pl2, pl2[::-1]], axis=0)


def _interplating_polyline(polylines, distance=0.5, split_distace=5):
    # Calculate the cumulative distance along the path, up-sample the polyline to 0.5 meter
    dist_along_path_list = []
    polylines_list = []
    euclidean_dists = np.linalg.norm(polylines[1:, :2] - polylines[:-1, :2], axis=-1)
    euclidean_dists = np.concatenate([[0], euclidean_dists])
    breakpoints = np.where(euclidean_dists > 3)[0]
    breakpoints = np.concatenate([[0], breakpoints, [polylines.shape[0]]])
    for i in range(1, breakpoints.shape[0]):
        start = breakpoints[i - 1]
        end = breakpoints[i]
        dist_along_path_list.append(
            np.cumsum(euclidean_dists[start:end]) - euclidean_dists[start]
        )
        polylines_list.append(polylines[start:end])

    multi_polylines_list = []
    for idx in range(len(dist_along_path_list)):
        if len(dist_along_path_list[idx]) < 2:
            continue
        dist_along_path = dist_along_path_list[idx]
        polylines_cur = polylines_list[idx]

        # Create interpolation functions for x and y coordinates
        fxy = interp1d(dist_along_path, polylines_cur, axis=0)

        # Create an array of distances at which to interpolate
        new_dist_along_path = np.arange(0, dist_along_path[-1], distance)
        new_dist_along_path = np.concatenate([new_dist_along_path, dist_along_path[[-1]]])

        # Combine the new x and y coordinates into a single array
        new_polylines = fxy(new_dist_along_path)
        polyline_size = int(split_distace / distance)
        if new_polylines.shape[0] >= (polyline_size + 1):
            padding_size = (new_polylines.shape[0] - (polyline_size + 1)) % polyline_size
            final_index = (new_polylines.shape[0] - (polyline_size + 1)) // polyline_size + 1
        else:
            padding_size = new_polylines.shape[0]
            final_index = 0

        multi_polylines = None
        new_polylines = torch.from_numpy(new_polylines)
        new_heading = torch.atan2(
            new_polylines[1:, 1] - new_polylines[:-1, 1],
            new_polylines[1:, 0] - new_polylines[:-1, 0],
        )
        new_heading = torch.cat([new_heading, new_heading[-1:]], -1)[..., None]
        new_polylines = torch.cat([new_polylines, new_heading], -1)
        if new_polylines.shape[0] >= (polyline_size + 1):
            multi_polylines = new_polylines.unfold(
                dimension=0,
                size=polyline_size + 1,
                step=polyline_size,
            )
            multi_polylines = multi_polylines.transpose(1, 2)
            multi_polylines = multi_polylines[:, ::5, :]
        if padding_size >= 3:
            last_polyline = new_polylines[final_index * polyline_size :]
            last_polyline = last_polyline[
                torch.linspace(0, last_polyline.shape[0] - 1, steps=3).long()
            ]
            if multi_polylines is not None:
                multi_polylines = torch.cat([multi_polylines, last_polyline.unsqueeze(0)], dim=0)
            else:
                multi_polylines = last_polyline.unsqueeze(0)
        if multi_polylines is None:
            continue
        multi_polylines_list.append(multi_polylines)

    if len(multi_polylines_list) > 0:
        multi_polylines_list = torch.cat(multi_polylines_list, dim=0).to(torch.float32)
    else:
        multi_polylines_list = None
    return multi_polylines_list


def _resample_closed_boundary(boundary_xy: np.ndarray, num_samples: int) -> torch.Tensor:
    """닫힌 경계를 둘레 기준으로 고정 개수 점으로 다시 뽑습니다.

    Args:
        boundary_xy: 원래 경계점입니다. shape은 ``[num_vertex, 2]`` 입니다.
        num_samples: 다시 뽑을 점 개수입니다.

    Returns:
        torch.Tensor: 균일 간격으로 다시 뽑은 경계점입니다.
            shape은 ``[num_samples, 2]`` 입니다.
    """
    if boundary_xy.shape[0] == 0:
        return torch.zeros((num_samples, 2), dtype=torch.float32)

    if boundary_xy.shape[0] > 1 and np.allclose(boundary_xy[0], boundary_xy[-1]):
        boundary_xy = boundary_xy[:-1]

    if boundary_xy.shape[0] == 1:
        return torch.from_numpy(np.repeat(boundary_xy, num_samples, axis=0)).to(torch.float32)

    closed_boundary = np.concatenate([boundary_xy, boundary_xy[:1]], axis=0)
    segment_length = np.linalg.norm(closed_boundary[1:] - closed_boundary[:-1], axis=-1)
    total_length = float(segment_length.sum())
    if total_length < 1e-6:
        center_xy = boundary_xy.mean(axis=0, keepdims=True)
        return torch.from_numpy(np.repeat(center_xy, num_samples, axis=0)).to(torch.float32)

    cumulative_length = np.concatenate([[0.0], np.cumsum(segment_length)])
    sample_distance = np.linspace(0.0, total_length, num=num_samples + 1)[:-1]
    sampled_boundary = np.zeros((num_samples, 2), dtype=np.float32)
    segment_idx = 0
    for point_idx, target_dist in enumerate(sample_distance):
        while segment_idx < len(segment_length) - 1 and cumulative_length[segment_idx + 1] < target_dist:
            segment_idx += 1
        current_length = max(segment_length[segment_idx], 1e-6)
        ratio = (target_dist - cumulative_length[segment_idx]) / current_length
        sampled_boundary[point_idx] = (
            closed_boundary[segment_idx]
            + ratio * (closed_boundary[segment_idx + 1] - closed_boundary[segment_idx])
        )
    return torch.from_numpy(sampled_boundary).to(torch.float32)


def _to_local_boundary(
    boundary_global: torch.Tensor,
    center_xy: torch.Tensor,
    heading: torch.Tensor,
) -> torch.Tensor:
    """전역 좌표 경계를 중심 기준 local 좌표로 바꿉니다.

    Args:
        boundary_global: 다시 뽑은 전역 경계점입니다.
            shape은 ``[num_boundary, 2]`` 입니다.
        center_xy: polygon 중심점입니다. shape은 ``[2]`` 입니다.
        heading: polygon 대표 방향입니다. shape은 ``[]`` 입니다.

    Returns:
        torch.Tensor: 중심 기준 local 경계점입니다.
            shape은 ``[num_boundary, 2]`` 입니다.
    """
    rel_xy = boundary_global - center_xy.unsqueeze(0)
    cos_val = torch.cos(heading)
    sin_val = torch.sin(heading)
    rot_mat = torch.stack(
        [
            torch.stack([cos_val, sin_val]),
            torch.stack([-sin_val, cos_val]),
        ],
        dim=0,
    )
    return rel_xy @ rot_mat.T


def _build_polygon_token_store(
    semantic_polygon: Dict[str, Any] | None,
    num_boundary_samples: int,
) -> Dict[str, Any]:
    """polygon branch가 바로 읽을 수 있는 고정 길이 입력을 만듭니다.

    Args:
        semantic_polygon: `get_map_features`에서 넘긴 polygon 정보입니다.
            중심은 ``[n_poly, 2]``, 방향은 ``[n_poly]``, 크기는 ``[n_poly, 2]`` 입니다.
            raw_boundary는 길이가 ``n_poly`` 인 리스트이고 각 원소 shape은 ``[num_vertex_i, 2]`` 입니다.
        num_boundary_samples: 경계에서 다시 뽑을 점 개수입니다.

    Returns:
        Dict[str, Any]: HeteroData node store로 바로 넣을 수 있는 polygon token 정보입니다.
        - ``position``: ``[n_poly, 2]``
        - ``orientation``: ``[n_poly]``
        - ``size``: ``[n_poly, 2]``
        - ``type``: ``[n_poly]``
        - ``boundary``: ``[n_poly, num_boundary_samples, 2]``
    """
    if semantic_polygon is None or semantic_polygon["num_nodes"] == 0:
        return {
            "position": torch.zeros((0, 2), dtype=torch.float32),
            "orientation": torch.zeros((0,), dtype=torch.float32),
            "size": torch.zeros((0, 2), dtype=torch.float32),
            "type": torch.zeros((0,), dtype=torch.uint8),
            "boundary": torch.zeros((0, num_boundary_samples, 2), dtype=torch.float32),
            "num_nodes": 0,
        }

    centers = semantic_polygon["position"].to(torch.float32)
    headings = semantic_polygon["orientation"].to(torch.float32)
    sizes = semantic_polygon["size"].to(torch.float32)
    types = semantic_polygon["type"].to(torch.uint8)

    boundary_local_list = []
    for poly_idx, raw_boundary in enumerate(semantic_polygon["raw_boundary"]):
        if isinstance(raw_boundary, torch.Tensor):
            raw_boundary_np = raw_boundary.cpu().numpy()
        else:
            raw_boundary_np = np.asarray(raw_boundary, dtype=np.float32)
        sampled_boundary_global = _resample_closed_boundary(
            boundary_xy=raw_boundary_np,
            num_samples=num_boundary_samples,
        )
        sampled_boundary_local = _to_local_boundary(
            boundary_global=sampled_boundary_global,
            center_xy=centers[poly_idx],
            heading=headings[poly_idx],
        )
        boundary_local_list.append(sampled_boundary_local)

    return {
        "position": centers,
        "orientation": headings,
        "size": sizes,
        "type": types,
        "boundary": torch.stack(boundary_local_list, dim=0),
        "num_nodes": centers.shape[0],
    }


def preprocess_map(
    map_data: Dict[str, Any],
    polygon_boundary_samples: int = 8,
) -> Dict[str, Any]:
    pt2pl = map_data[("map_point", "to", "map_polygon")]["edge_index"]
    split_polyline_type = []
    split_polyline_pos = []
    split_polyline_theta = []
    split_polygon_type = []
    split_light_type = []
    for i in sorted(torch.unique(pt2pl[1])):
        index = pt2pl[0, pt2pl[1] == i]
        if len(index) <= 2:
            continue
        polygon_type = map_data["map_polygon"]["type"][i]
        light_type = map_data["map_polygon"]["light_type"][i]
        cur_type = map_data["map_point"]["type"][index]
        cur_pos = map_data["map_point"]["position"][index, :2]
        split_polyline = _interplating_polyline(cur_pos.numpy())
        if split_polyline is None:
            continue
        split_polyline_pos.append(split_polyline[..., :2])
        split_polyline_theta.append(split_polyline[..., 2])
        split_polyline_type.append(cur_type[0].repeat(split_polyline.shape[0]))
        split_polygon_type.append(polygon_type.repeat(split_polyline.shape[0]))
        split_light_type.append(light_type.repeat(split_polyline.shape[0]))

    data = {}
    if len(split_polyline_pos) == 0:  # add dummy empty map
        data["map_save"] = {
            # 6e4 such that it's within the range of float16.
            "traj_pos": torch.zeros([1, 3, 2], dtype=torch.float32) + 6e4,
            "traj_theta": torch.zeros([1], dtype=torch.float32),
        }
        data["pt_token"] = {
            "type": torch.tensor([0], dtype=torch.uint8),
            "pl_type": torch.tensor([0], dtype=torch.uint8),
            "light_type": torch.tensor([0], dtype=torch.uint8),
            "num_nodes": 1,
        }
    else:
        data["map_save"] = {
            "traj_pos": torch.cat(split_polyline_pos, dim=0),  # [num_nodes, 3, 2]
            "traj_theta": torch.cat(split_polyline_theta, dim=0)[:, 0],  # [num_nodes]
        }
        data["pt_token"] = {
            "type": torch.cat(split_polyline_type, dim=0),  # [num_nodes], uint8
            "pl_type": torch.cat(split_polygon_type, dim=0),  # [num_nodes], uint8
            "light_type": torch.cat(split_light_type, dim=0),  # [num_nodes], uint8
            "num_nodes": data["map_save"]["traj_pos"].shape[0],
        }

    data["polygon_token"] = _build_polygon_token_store(
        semantic_polygon=map_data.get("semantic_polygon"),
        num_boundary_samples=polygon_boundary_samples,
    )
    return data
