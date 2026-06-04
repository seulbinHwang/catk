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

from src.smart.tokens.agent_token_matching import build_agent_type_masks
from src.smart.utils import (
    merge_by_type,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)


DEFAULT_AGENT_TOKEN_MATCH_CHUNK_SIZE = 384


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
        self.agent_token_match_chunk_size = DEFAULT_AGENT_TOKEN_MATCH_CHUNK_SIZE
        self.n_token_agent = {
            "veh": self.agent_token_all_veh.shape[0],
            "ped": self.agent_token_all_ped.shape[0],
            "cyc": self.agent_token_all_cyc.shape[0],
        }
        self.register_buffer(
            "agent_shape_by_type",
            torch.tensor(
                (
                    (2.0, 4.8),  # veh
                    (1.0, 1.0),  # ped
                    (1.0, 2.0),  # cyc
                ),
                dtype=torch.float32,
            ),
            persistent=False,
        )
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
            self.register_buffer(
                f"agent_token_contour_trajectory_{k}",
                v[:, 1:].contiguous(),
                persistent=False,
            )
        for k, v in agent_token_data["traj"].items():
            v = torch.tensor(v[:, 1:], dtype=torch.float32)
            # [n_token, 5, 3], x/y/yaw trajectory over the 0.5s action interval
            self.register_buffer(f"agent_token_trajectory_{k}", v, persistent=False)

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
        (
            agent_shape,
            token_traj_all,
            token_traj,
            token_trajectory,
            token_contour_trajectory,
            agent_type_masks,
        ) = self._get_agent_shape_and_token_traj(data["agent"]["type"])

        # ! get raw trajectory data
        valid = data["agent"]["valid_mask"].clone()  # [n_agent, n_step]
        heading = wrap_angle(data["agent"]["heading"].clone())  # [n_agent, n_step]
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
            "type_mask": agent_type_masks,
            "shape": data["agent"]["shape"],
            "ego_mask": data["agent"]["role"][:, 0],  # [n_agent]
            "role_mask": data["agent"]["role"].any(-1),  # [n_agent]
            "token_agent_shape": agent_shape,  # [n_agent, 2]
            "batch": data["agent"]["batch"],
            "token_traj_all": token_traj_all,  # type -> [n_agent_type, n_token_type, 6, 4, 2]
            "token_heading": self.token_heading.to(pos.device),
            "token_traj": token_traj,  # type -> [n_agent_type, n_token_type, 4, 2]
            "token_trajectory": token_trajectory,  # type -> [n_token_type, 5, 3]
            "token_contour_trajectory": token_contour_trajectory,  # type -> [n_token_type, 5, 4, 2]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": pos[:, self.shift :: self.shift],  # [n_agent, n_step=18, 2]
            "gt_head_raw": heading[:, self.shift :: self.shift],  # [n_agent, n_step=18]
            "gt_valid_raw": valid[:, self.shift :: self.shift],  # [n_agent, n_step=18]
            # raw 10 Hz segments for actions [(0->5), (5->10), ..., (85->90)]
            "gt_pos_segment_raw": pos[:, 1:].reshape(
                pos.shape[0], -1, self.shift, pos.shape[-1]
            ),  # [n_agent, 18, 5, 2]
            "gt_head_segment_raw": heading[:, 1:].reshape(
                heading.shape[0], -1, self.shift
            ),  # [n_agent, 18, 5]
            "gt_valid_segment_raw": valid[:, 1:].reshape(
                valid.shape[0], -1, self.shift
            ),  # [n_agent, 18, 5]
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
                token_trajectory=token_trajectory[agent_type],
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
        token_trajectory: Tensor,  # [n_token, 5, 3]
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
        prev_pos, prev_head = pos[:, 0], heading[:, 0]  # [n_agent, 2], [n_agent]

        out_dict = {
            "valid_mask": [],
            "token_idx": [],
            "tokenized_pos": [],
            "tokenized_heading": [],
        }

        for i in range(self.shift, n_step, self.shift):  # [5, 10, 15, ..., 90]
            _valid_mask = valid[:, i - self.shift : i + 1].all(dim=1)  # [n_agent]
            _invalid_mask = ~_valid_mask
            out_dict["valid_mask"].append(_valid_mask)

            token_idx_gt = torch.zeros(
                n_agent, dtype=torch.long, device=valid.device
            )

            match_prev_pos, match_prev_head = prev_pos, prev_head

            # udpate prev_pos, prev_head
            prev_head = heading[:, i].clone()
            prev_pos = pos[:, i].clone()
            valid_idx = _valid_mask.nonzero(as_tuple=True)[0]
            token_idx_valid, token_endpoint_valid = (
                self._match_full_trajectory_agent_token(
                    token_trajectory=token_trajectory,
                    gt_pos=pos[valid_idx, i - self.shift + 1 : i + 1],
                    gt_heading=heading[valid_idx, i - self.shift + 1 : i + 1],
                    prev_pos=match_prev_pos[valid_idx],
                    prev_head=match_prev_head[valid_idx],
                )
            )
            token_idx_gt[valid_idx] = token_idx_valid
            prev_head[valid_idx] = token_endpoint_valid[:, 2]
            prev_pos[valid_idx] = token_endpoint_valid[:, :2]
            # add to output dict
            out_dict["token_idx"].append(token_idx_gt)
            out_dict["tokenized_pos"].append(
                prev_pos.masked_fill(_invalid_mask.unsqueeze(1), 0)
            )
            out_dict["tokenized_heading"].append(prev_head.masked_fill(_invalid_mask, 0))
        out_dict = {k: torch.stack(v, dim=1) for k, v in out_dict.items()}
        return out_dict

    def _match_full_trajectory_agent_token(
        self,
        token_trajectory: Tensor,
        gt_pos: Tensor,
        gt_heading: Tensor,
        prev_pos: Tensor,
        prev_head: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Match tokens using the full 0.5s trajectory described by TrajTok.

        The paper defines each token as L points of (x, y, yaw). We transform the
        GT segment into the current token frame and compare it with the local
        token trajectories. This preserves the rigid-transform distance while
        avoiding a per-agent global materialization of every token.
        """
        n_agent = gt_pos.shape[0]
        if n_agent == 0:
            empty_idx = torch.empty(0, dtype=torch.long, device=token_trajectory.device)
            empty_endpoint = token_trajectory.new_empty((0, 3))
            return empty_idx, empty_endpoint

        chunk_size = min(self.agent_token_match_chunk_size, n_agent)
        token_idx_chunks = []
        token_endpoint_chunks = []
        for start in range(0, n_agent, chunk_size):
            end = min(start + chunk_size, n_agent)

            gt_pos_local, gt_head_local = transform_to_local(
                pos_global=gt_pos[start:end],
                head_global=gt_heading[start:end],
                pos_now=prev_pos[start:end],
                head_now=prev_head[start:end],
            )
            gt_head_local = wrap_angle(gt_head_local)

            pos_delta = (
                token_trajectory[:, :, :2].unsqueeze(0)
                - gt_pos_local.unsqueeze(1)
            )
            head_delta = wrap_angle(
                token_trajectory[:, :, 2].unsqueeze(0)
                - gt_head_local.unsqueeze(1)
            )
            token_dist = torch.sqrt(pos_delta.square().sum(-1) + head_delta.square()).mean(-1)
            token_idx = token_dist.argmin(dim=-1)
            token_idx_chunks.append(token_idx)

            matched_local = token_trajectory[token_idx, -1]
            endpoint_pos, endpoint_head = transform_to_global(
                pos_local=matched_local[:, :2].unsqueeze(1),
                head_local=matched_local[:, 2].unsqueeze(1),
                pos_now=prev_pos[start:end],
                head_now=prev_head[start:end],
            )
            token_endpoint_chunks.append(
                torch.cat(
                    (
                        endpoint_pos.squeeze(1),
                        wrap_angle(endpoint_head.squeeze(1)).unsqueeze(-1),
                    ),
                    dim=-1,
                )
            )

        return torch.cat(token_idx_chunks, dim=0), torch.cat(token_endpoint_chunks, dim=0)

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
        n_step_to_extrapolate = first_valid_step % self.shift
        if valid.shape[1] > 10:
            force_history_token = (first_valid_step == 10) & (
                ~valid[:, 10 - self.shift]
            )
            n_step_to_extrapolate = torch.where(
                force_history_token,
                n_step_to_extrapolate.new_full((), self.shift),
                n_step_to_extrapolate,
            )

        offsets = torch.arange(self.shift, device=valid.device)
        start_step = first_valid_step - n_step_to_extrapolate
        fill_times = start_step[:, None] + offsets[None, :]
        fill_mask = offsets[None, :] < n_step_to_extrapolate[:, None]

        agent_idx = torch.arange(valid.shape[0], device=valid.device)[:, None]
        agent_idx = agent_idx.expand_as(fill_times)[fill_mask]
        fill_times = fill_times[fill_mask]
        source_times = first_valid_step[agent_idx]
        source_vel = vel[agent_idx, source_times]
        source_heading = heading[agent_idx, source_times]

        valid[agent_idx, fill_times] = True
        vel[agent_idx, fill_times] = source_vel
        heading[agent_idx, fill_times] = source_heading
        all_agents = torch.arange(valid.shape[0], device=valid.device)
        for offset in range(self.shift):
            active_mask = n_step_to_extrapolate > offset
            active_agents = all_agents[active_mask]
            if active_agents.numel() == 0:
                continue
            active_source_times = first_valid_step[active_agents]
            source_pos_step = active_source_times - offset
            target_pos_step = source_pos_step - 1
            pos[active_agents, target_pos_step] = (
                pos[active_agents, source_pos_step]
                - vel[active_agents, active_source_times] * 0.1
            )

        return valid, pos, heading, vel

    def _get_agent_shape_and_token_traj(
        self, agent_type: Tensor
    ) -> Tuple[
        Tensor,
        Dict[str, Tensor],
        Dict[str, Tensor],
        Dict[str, Tensor],
        Dict[str, Tensor],
        Dict[str, Tensor],
    ]:
        """
        agent_shape: [n_agent, 2]
        token_traj_all: [n_agent, n_token, 6, 4, 2]
        token_traj: [n_agent, n_token, 4, 2]
        token_trajectory: [n_token, 5, 3]
        token_contour_trajectory: [n_token, 5, 4, 2]
        """
        agent_type_masks = build_agent_type_masks(agent_type)
        agent_shape = self.agent_shape_by_type.new_zeros((len(agent_type), 2))
        valid_type_mask = (agent_type >= 0) & (agent_type < self.agent_shape_by_type.shape[0])
        agent_shape[valid_type_mask] = self.agent_shape_by_type[
            agent_type[valid_type_mask].long()
        ]
        token_traj_all = {}
        token_traj = {}
        token_trajectory = {}
        token_contour_trajectory = {}
        for k, mask in agent_type_masks.items():
            n_agent_type = int(mask.sum().item())
            token_bank = getattr(self, f"agent_token_all_{k}")
            token_traj_all[k] = token_bank.unsqueeze(0).expand(
                n_agent_type, -1, -1, -1, -1
            )
            token_traj[k] = token_bank[:, -1].unsqueeze(0).expand(
                n_agent_type, -1, -1, -1
            )
            token_trajectory[k] = getattr(self, f"agent_token_trajectory_{k}")
            token_contour_trajectory[k] = getattr(
                self,
                f"agent_token_contour_trajectory_{k}",
            )
        return (
            agent_shape,
            token_traj_all,
            token_traj,
            token_trajectory,
            token_contour_trajectory,
            agent_type_masks,
        )
