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

import os
import pickle
from typing import Dict, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.smart.tokens.agent_token_matching import (
    build_agent_type_masks,
    match_token_idx_from_local_contour,
)
from src.smart.utils import (
    cal_polygon_contour,
    merge_by_type,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)


def _clean_heading_dense_impl(valid: Tensor, heading: Tensor) -> Tensor:
    valid_pairs = valid[:, :-1] & valid[:, 1:]
    cleaned_steps = [heading[:, 0]]
    prev_heading = heading[:, 0]
    for i in range(heading.shape[1] - 1):
        raw_next_heading = heading[:, i + 1]
        heading_diff = torch.abs(wrap_angle(prev_heading - raw_next_heading))
        change_needed = (heading_diff > 1.5) & valid_pairs[:, i]
        next_heading = torch.where(change_needed, prev_heading, raw_next_heading)
        cleaned_steps.append(next_heading)
        prev_heading = next_heading
    return torch.stack(cleaned_steps, dim=1)


class TokenProcessor(torch.nn.Module):

    def __init__(
        self,
        map_token_file: str,
        agent_token_file: str,
    ) -> None:
        super(TokenProcessor, self).__init__()
        self.shift = 5

        module_dir = os.path.dirname(__file__)
        self.init_agent_token(os.path.join(module_dir, agent_token_file))
        self.init_map_token(os.path.join(module_dir, map_token_file))
        self.n_token_agent = {
            "veh": self.agent_token_all_veh.shape[0],
            "ped": self.agent_token_all_ped.shape[0],
            "cyc": self.agent_token_all_cyc.shape[0],
        }
        self.register_buffer(
            "token_heading",
            torch.arange(-179, 180, dtype=torch.float32) / 180 * torch.pi,
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        tokenized_map = self.tokenize_map(data)
        tokenized_agent = self.tokenize_agent(data)
        return tokenized_map, tokenized_agent

    def init_map_token(self, map_token_traj_path, argmin_sample_len=3) -> None:
        map_token_traj = pickle.load(open(map_token_traj_path, "rb"))["traj_src"]
        indices = torch.linspace(
            0, map_token_traj.shape[1] - 1, steps=argmin_sample_len
        ).long()

        self.register_buffer(
            "map_token_traj_src",
            torch.tensor(map_token_traj, dtype=torch.float32).flatten(1, 2),
            persistent=False,
        )  # [n_token, 11*2]

        self.register_buffer(
            "map_token_sample_pt",
            torch.tensor(map_token_traj[:, indices], dtype=torch.float32).unsqueeze(0),
            persistent=False,
        )  # [1, n_token, 3, 2]

    def init_agent_token(self, agent_token_path) -> None:
        agent_token_data = pickle.load(open(agent_token_path, "rb"))
        for k, v in agent_token_data["token_all"].items():
            v = torch.tensor(v, dtype=torch.float32)
            # [n_token, 6, 4, 2], countour, 10 hz
            self.register_buffer(f"agent_token_all_{k}", v, persistent=False)

    def tokenize_map(self, data: HeteroData) -> Dict[str, Tensor]:
        traj_pos = data["map_save"]["traj_pos"]  # [n_pl, 3, 2]
        traj_theta = data["map_save"]["traj_theta"]  # [n_pl]

        traj_pos_local, _ = transform_to_local(
            pos_global=traj_pos,  # [n_pl, 3, 2]
            head_global=None,  # [n_pl, 1]
            pos_now=traj_pos[:, 0],  # [n_pl, 2]
            head_now=traj_theta,  # [n_pl]
        )
        # [1, n_token, 3, 2] - [n_pl, 1, 3, 2]
        dist = torch.sum(
            (self.map_token_sample_pt - traj_pos_local.unsqueeze(1)) ** 2,
            dim=(-2, -1),
        )  # [n_pl, n_token]

        token_idx = torch.argmin(dist, dim=-1)

        tokenized_map = {
            "position": traj_pos[:, 0].contiguous(),  # [n_pl, 2]
            "orientation": traj_theta,  # [n_pl]
            "token_idx": token_idx,  # [n_pl]
            "token_traj_src": self.map_token_traj_src,  # [n_token, 11*2]
            "type": data["pt_token"]["type"].long(),  # [n_pl]
            "pl_type": data["pt_token"]["pl_type"].long(),  # [n_pl]
            "light_type": data["pt_token"]["light_type"].long(),  # [n_pl]
            "batch": data["pt_token"]["batch"],  # [n_pl]
        }
        return tokenized_map

    def tokenize_agent(self, data: HeteroData) -> Dict[str, Tensor]:
        """
        Args: data["agent"]: Dict
            "valid_mask": [n_agent, n_step], bool
            "role": [n_agent, 3], bool
            "id": [n_agent], int64
            "type": [n_agent], uint8
            "position": [n_agent, n_step, 3], float32
            "heading": [n_agent, n_step], float32
            "velocity": [n_agent, n_step, 2], float32
            "shape": [n_agent, 3], float32
        """
        # ! collate width/length, traj tokens for current batch
        agent_shape, token_traj_all, token_traj = self._get_agent_shape_and_token_traj(
            data["agent"]["type"]
        )

        # ! get raw trajectory data
        valid = data["agent"]["valid_mask"].clone()  # [n_agent, n_step]
        heading = data["agent"]["heading"].clone()  # [n_agent, n_step]
        pos = data["agent"]["position"][..., :2].clone().contiguous()  # [n_agent, n_step, 2]
        vel = data["agent"]["velocity"].clone()  # [n_agent, n_step, 2]

        # ! agent, specifically vehicle's heading can be 180 degree off. We fix it here.
        heading = self._clean_heading(valid, heading)
        # ! extrapolate to previous 5th step.
        valid, pos, heading, vel = self._extrapolate_agent_to_prev_token_step(
            valid, pos, heading, vel
        )

        # ! prepare output dict
        tokenized_agent = {
            "num_graphs": data.num_graphs,
            "type": data["agent"]["type"],
            "type_mask": build_agent_type_masks(data["agent"]["type"]),
            "shape": data["agent"]["shape"],
            "ego_mask": data["agent"]["role"][:, 0],  # [n_agent]
            "role_mask": data["agent"]["role"].any(-1),  # [n_agent]
            "token_agent_shape": agent_shape,  # [n_agent, 2]
            "batch": data["agent"]["batch"],
            "token_traj_all": token_traj_all,  # type -> [n_agent_type, n_token_type, 6, 4, 2]
            "token_heading": self.token_heading.to(pos.device),
            "token_traj": token_traj,  # type -> [n_agent_type, n_token_type, 4, 2]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": pos[:, self.shift :: self.shift],  # [n_agent, n_step=18, 2]
            "gt_head_raw": heading[:, self.shift :: self.shift],  # [n_agent, n_step=18]
            "gt_valid_raw": valid[:, self.shift :: self.shift],  # [n_agent, n_step=18]
        }
        # [n_token, 8]
        for k in ["veh", "ped", "cyc"]:
            tokenized_agent[f"trajectory_token_{k}"] = getattr(
                self, f"agent_token_all_{k}"
            )[:, -1].flatten(1, 2)

        # ! match token for each agent
        if not self.training:
            # [n_agent]
            tokenized_agent["gt_z_raw"] = data["agent"]["position"][:, 10, 2]

        token_dict_by_type = {}
        for agent_type, type_mask in tokenized_agent["type_mask"].items():
            token_dict_by_type[agent_type] = self._match_agent_token(
                valid=valid[type_mask],
                pos=pos[type_mask],
                heading=heading[type_mask],
                agent_shape=agent_shape[type_mask],
                token_traj=token_traj[agent_type],
            )

        for key in ("valid_mask", "token_idx", "tokenized_pos", "tokenized_heading"):
            tokenized_agent[key] = merge_by_type(
                {agent_type: data[key] for agent_type, data in token_dict_by_type.items()},
                tokenized_agent["type_mask"],
            )
        tokenized_agent["gt_idx"] = tokenized_agent["token_idx"]
        tokenized_agent["gt_pos"] = tokenized_agent["tokenized_pos"]
        tokenized_agent["gt_heading"] = tokenized_agent["tokenized_heading"]
        tokenized_agent["sampled_idx"] = tokenized_agent["token_idx"]
        tokenized_agent["sampled_pos"] = tokenized_agent["tokenized_pos"]
        tokenized_agent["sampled_heading"] = tokenized_agent["tokenized_heading"]
        return tokenized_agent

    def _match_agent_token(
        self,
        valid: Tensor,  # [n_agent, n_step]
        pos: Tensor,  # [n_agent, n_step, 2]
        heading: Tensor,  # [n_agent, n_step]
        agent_shape: Tensor,  # [n_agent, 2]
        token_traj: Tensor,  # [n_agent, n_token, 4, 2]
    ) -> Dict[str, Tensor]:
        """n_step_token=n_step//5
        n_step_token=18 for train with BC.
        n_step_token=2 for val/test and train with closed-loop rollout.
        Returns: Dict
            # ! action that goes from [(0->5), (5->10), ..., (85->90)]
            "valid_mask": [n_agent, n_step_token]
            "gt_idx": [n_agent, n_step_token]
            # ! at step [5, 10, 15, ..., 90]
            "gt_pos": [n_agent, n_step_token, 2]
            "gt_heading": [n_agent, n_step_token]
            # ! deterministic rollout state used by open-loop training
            "sampled_idx": [n_agent, n_step_token]
            "sampled_pos": [n_agent, n_step_token, 2]
            "sampled_heading": [n_agent, n_step_token]
        """
        n_agent, n_step = valid.shape
        range_a = torch.arange(n_agent, device=valid.device)
        prev_pos, prev_head = pos[:, 0], heading[:, 0]  # [n_agent, 2], [n_agent]

        out_dict = {
            "valid_mask": [],
            "token_idx": [],
            "tokenized_pos": [],
            "tokenized_heading": [],
        }

        for i in range(self.shift, n_step, self.shift):  # [5, 10, 15, ..., 90]
            _valid_mask = valid[:, i - self.shift] & valid[:, i]  # [n_agent]
            _invalid_mask = ~_valid_mask
            out_dict["valid_mask"].append(_valid_mask)

            #! gt_contour: [n_agent, 4, 2] in global coord
            gt_contour = cal_polygon_contour(pos[:, i], heading[:, i], agent_shape)

            gt_contour_local, _ = transform_to_local(
                pos_global=gt_contour,
                head_global=None,
                pos_now=prev_pos,  # [n_agent, 2]
                head_now=prev_head,  # [n_agent]
            )
            # ! tokenize without sampling. Matching in the previous-token local
            # frame is equivalent to transforming every token to global coords,
            # but avoids materializing [n_agent, n_token, 4, 2] every step.
            token_idx_gt = torch.argmin(
                torch.norm(
                    token_traj - gt_contour_local.unsqueeze(1),
                    dim=-1,
                ).sum(-1),
                dim=-1,
            )
            token_contour_gt_local = token_traj[range_a, token_idx_gt]
            token_pos_gt, token_head_gt = self._token_pose_from_local_contour(
                token_contour_local=token_contour_gt_local,
                ref_pos=prev_pos,
                ref_head=prev_head,
            )

            # udpate prev_pos, prev_head
            prev_head = heading[:, i].clone()
            prev_head[_valid_mask] = token_head_gt[_valid_mask]
            prev_pos = pos[:, i].clone()
            prev_pos[_valid_mask] = token_pos_gt[_valid_mask]
            # add to output dict
            out_dict["token_idx"].append(token_idx_gt)
            out_dict["tokenized_pos"].append(
                prev_pos.masked_fill(_invalid_mask.unsqueeze(1), 0)
            )
            out_dict["tokenized_heading"].append(prev_head.masked_fill(_invalid_mask, 0))
        out_dict = {k: torch.stack(v, dim=1) for k, v in out_dict.items()}
        return out_dict

    def _match_token_idx_from_local_contour(
        self,
        agent_type: Tensor,
        contour_local: Tensor,
        reduction: str,
    ) -> Tensor:
        """Choose nearest agent token directly in the previous-token local frame."""
        return match_token_idx_from_local_contour(
            agent_type=agent_type,
            contour_local=contour_local,
            token_bank_all_veh=self.agent_token_all_veh[:, -1],
            token_bank_all_ped=self.agent_token_all_ped[:, -1],
            token_bank_all_cyc=self.agent_token_all_cyc[:, -1],
            reduction=reduction,
        )

    def _token_pose_from_local_contour(
        self,
        token_contour_local: Tensor,
        ref_pos: Tensor,
        ref_head: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Convert selected local token contours back to global center/heading."""
        token_center_local = token_contour_local.mean(dim=1)
        token_center_global, _ = transform_to_global(
            pos_local=token_center_local.unsqueeze(1),
            head_local=None,
            pos_now=ref_pos,
            head_now=ref_head,
        )

        token_dxy_local = token_contour_local[:, 0] - token_contour_local[:, 3]
        token_head_local = torch.arctan2(
            token_dxy_local[:, 1],
            token_dxy_local[:, 0],
        )
        return token_center_global.squeeze(1), wrap_angle(ref_head + token_head_local)

    @staticmethod
    def _clean_heading(valid: Tensor, heading: Tensor) -> Tensor:
        return _clean_heading_dense_impl(valid=valid, heading=heading)

    def _extrapolate_agent_to_prev_token_step(
        self,
        valid: Tensor,  # [n_agent, n_step]
        pos: Tensor,  # [n_agent, n_step, 2]
        heading: Tensor,  # [n_agent, n_step]
        vel: Tensor,  # [n_agent, n_step, 2]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # [n_agent], max will give the first True step
        first_valid_step = torch.max(valid, dim=1).indices
        n_step_to_extrapolate = first_valid_step.remainder(self.shift)

        prev_token_step = 10 - self.shift
        if 0 <= prev_token_step < valid.shape[1]:
            needs_history_token = (first_valid_step == 10) & (~valid[:, prev_token_step])
            n_step_to_extrapolate = torch.where(
                needs_history_token,
                torch.full_like(n_step_to_extrapolate, self.shift),
                n_step_to_extrapolate,
            )

        step_index = torch.arange(valid.shape[1], device=valid.device).unsqueeze(0)
        fill_start = first_valid_step - n_step_to_extrapolate
        fill_mask = (
            (n_step_to_extrapolate > 0).unsqueeze(1)
            & (step_index >= fill_start.unsqueeze(1))
            & (step_index < first_valid_step.unsqueeze(1))
        )
        if not bool(fill_mask.any().item()):
            return valid, pos, heading, vel

        agent_index, step_index_flat = fill_mask.nonzero(as_tuple=True)
        source_step = first_valid_step[agent_index]
        source_vel = vel[agent_index, source_step]

        valid[agent_index, step_index_flat] = True
        vel[agent_index, step_index_flat] = source_vel
        heading[agent_index, step_index_flat] = heading[agent_index, source_step]
        delta_step = (source_step - step_index_flat).to(dtype=pos.dtype).unsqueeze(-1)
        pos[agent_index, step_index_flat] = (
            pos[agent_index, source_step] - source_vel * (0.1 * delta_step)
        )

        return valid, pos, heading, vel

    def _get_agent_shape_and_token_traj(
        self, agent_type: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        agent_shape: [n_agent, 2]
        token_traj_all: [n_agent, n_token, 6, 4, 2]
        token_traj: [n_agent, n_token, 4, 2]
        """
        agent_type_masks = build_agent_type_masks(agent_type)
        agent_shape = torch.zeros(len(agent_type), 2, dtype=torch.float32, device=agent_type.device)
        token_traj_all = {}
        token_traj = {}
        for k, mask in agent_type_masks.items():
            if k == "veh":
                width = 2.0
                length = 4.8
            elif k == "cyc":
                width = 1.0
                length = 2.0
            else:
                width = 1.0
                length = 1.0
            agent_shape[mask] = torch.tensor([width, length], dtype=agent_shape.dtype, device=agent_shape.device)
            token_traj_all[k] = getattr(self, f"agent_token_all_{k}").unsqueeze(0).repeat(
                int(mask.sum().item()), 1, 1, 1, 1
            )
            token_traj[k] = token_traj_all[k][:, :, -1, :, :].contiguous()
        return agent_shape, token_traj_all, token_traj
