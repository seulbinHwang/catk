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
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Categorical
from torch_geometric.data import HeteroData

from src.smart.tokens.agent_token_matching import (
    build_agent_type_masks,
    match_token_idx_from_local_contour,
)
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)


class TokenProcessor(torch.nn.Module):

    def __init__(
        self,
        map_token_file: str,
        agent_token_file: str,
        map_token_sampling: DictConfig,
        agent_token_sampling: DictConfig,
    ) -> None:
        super(TokenProcessor, self).__init__()
        self.map_token_sampling = map_token_sampling
        self.agent_token_sampling = agent_token_sampling
        self.shift = 5

        module_dir = os.path.dirname(__file__)
        self.init_agent_token(os.path.join(module_dir, agent_token_file))
        self.init_map_token(os.path.join(module_dir, map_token_file))
        self.n_token_agent = self.agent_token_all_veh.shape[0]

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

    def _empty_polygon_tokenized_map(self, device: torch.device) -> Dict[str, Tensor]:
        """polygon branch가 없을 때 쓰는 빈 입력을 만듭니다.

        Args:
            device: 반환 tensor를 올릴 장치입니다. shape은 scalar 입니다.

        Returns:
            Dict[str, Tensor]: polygon branch 기본 입력입니다.
            - ``polygon_position``: ``[0, 2]``
            - ``polygon_orientation``: ``[0]``
            - ``polygon_boundary``: ``[0, 8, 2]``
            - ``polygon_size``: ``[0, 2]``
            - ``polygon_type``: ``[0]``
            - ``polygon_batch``: ``[0]``
        """
        return {
            "polygon_position": torch.zeros((0, 2), dtype=torch.float32, device=device),
            "polygon_orientation": torch.zeros((0,), dtype=torch.float32, device=device),
            "polygon_boundary": torch.zeros(
                (0, 8, 2), dtype=torch.float32, device=device
            ),
            "polygon_size": torch.zeros((0, 2), dtype=torch.float32, device=device),
            "polygon_type": torch.zeros((0,), dtype=torch.long, device=device),
            "polygon_batch": torch.zeros((0,), dtype=torch.long, device=device),
        }

    def _tokenize_polygon_map(self, data: HeteroData) -> Dict[str, Tensor]:
        """전처리된 polygon store를 map decoder용 입력으로 옮깁니다.

        Args:
            data: 배치된 입력입니다.

        Returns:
            Dict[str, Tensor]: polygon branch 입력 사전입니다.
            - ``polygon_position``: ``[n_poly, 2]``
            - ``polygon_orientation``: ``[n_poly]``
            - ``polygon_boundary``: ``[n_poly, 8, 2]``
            - ``polygon_size``: ``[n_poly, 2]``
            - ``polygon_type``: ``[n_poly]``
            - ``polygon_batch``: ``[n_poly]``
        """
        device = data["map_save"]["traj_pos"].device
        if "polygon_token" not in data.node_types:
            raise RuntimeError(
                "Batched input is missing polygon_token. "
                "Regenerate every cache split with the current src.data_preprocess "
                "so training and inference use the same polygon-map schema."
            )

        polygon_store = data["polygon_token"]
        required_keys = {"position", "orientation", "boundary", "size", "type"}
        missing_keys = sorted(required_keys.difference(polygon_store.keys()))
        if missing_keys:
            missing_str = ", ".join(missing_keys)
            raise RuntimeError(
                f"polygon_token store is missing required fields ({missing_str}). "
                "Regenerate every cache split with the current src.data_preprocess."
            )

        if polygon_store["position"].numel() == 0:
            return self._empty_polygon_tokenized_map(device=device)

        if (
            "batch" in polygon_store
            and polygon_store["batch"].numel() == polygon_store["position"].shape[0]
        ):
            polygon_batch = polygon_store["batch"].long()
        else:
            polygon_batch = torch.zeros(
                polygon_store["position"].shape[0],
                dtype=torch.long,
                device=polygon_store["position"].device,
            )

        return {
            "polygon_position": polygon_store["position"].to(torch.float32).contiguous(),
            "polygon_orientation": polygon_store["orientation"]
            .to(torch.float32)
            .contiguous(),
            "polygon_boundary": polygon_store["boundary"].to(torch.float32).contiguous(),
            "polygon_size": polygon_store["size"].to(torch.float32).contiguous(),
            "polygon_type": polygon_store["type"].long().contiguous(),
            "polygon_batch": polygon_batch.contiguous(),
        }

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

        if self.training and (self.map_token_sampling.num_k > 1):
            topk_dists, topk_indices = torch.topk(
                dist,
                self.map_token_sampling.num_k,
                dim=-1,
                largest=False,
                sorted=False,
            )  # [n_pl, K]
            topk_logits = (-1e-6 - topk_dists) / self.map_token_sampling.temp
            _samples = Categorical(logits=topk_logits).sample()  # [n_pl] in K
            token_idx = topk_indices[torch.arange(len(_samples)), _samples].contiguous()
        else:
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
        tokenized_map.update(self._tokenize_polygon_map(data))
        return tokenized_map

    def tokenize_agent(
        self,
        data: HeteroData,
        return_preprocessed: bool = False,
    ) -> Dict[str, Tensor] | tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Args:
            data: 원본 에이전트 정보가 들어있는 배치입니다.
            return_preprocessed: True면 토큰 결과와 함께 전처리된 시계열을 같이 돌려줍니다.

        Returns:
            Dict[str, Tensor] | tuple[Dict[str, Tensor], Dict[str, Tensor]]:
                - 기본 반환: 학습/평가에 쓰는 토큰화 결과 사전.
                - 추가 반환: ``valid [n_agent, n_step]``, ``pos [n_agent, n_step, 2]``,
                  ``heading [n_agent, n_step]``를 담은 전처리 결과 사전.
        """
        agent_type = data["agent"]["type"]
        agent_shape = self._get_agent_shape(agent_type)

        valid = data["agent"]["valid_mask"]
        heading = data["agent"]["heading"]
        pos = data["agent"]["position"][..., :2].contiguous()
        vel = data["agent"]["velocity"]

        heading = self._clean_heading(valid, heading)
        valid, pos, heading, vel = self._extrapolate_agent_to_prev_token_step(
            valid,
            pos,
            heading,
            vel,
        )

        tokenized_agent = {
            "num_graphs": data.num_graphs,
            "type": agent_type,
            "shape": data["agent"]["shape"],
            "ego_mask": data["agent"]["role"][:, 0],  # [n_agent]
            "token_agent_shape": agent_shape,  # [n_agent, 2]
            "batch": data["agent"]["batch"],
        }
        for k in ["veh", "ped", "cyc"]:
            tokenized_agent[f"trajectory_token_{k}"] = getattr(
                self, f"agent_token_all_{k}"
            )[:, -1].flatten(1, 2)
            tokenized_agent[f"token_bank_all_{k}"] = getattr(self, f"agent_token_all_{k}")

        if not self.training:
            tokenized_agent.update(
                {
                    "gt_pos_raw": pos[:, self.shift :: self.shift],
                    "gt_head_raw": heading[:, self.shift :: self.shift],
                    "gt_valid_raw": valid[:, self.shift :: self.shift],
                    "gt_z_raw": data["agent"]["position"][:, 10, 2],
                }
            )

        token_dict = self._match_agent_token(
            valid=valid,
            pos=pos,
            heading=heading,
            agent_type=agent_type,
            agent_shape=agent_shape,
        )
        tokenized_agent.update(token_dict)

        if return_preprocessed:
            return tokenized_agent, {
                "valid": valid,
                "pos": pos,
                "heading": heading,
            }
        return tokenized_agent

    def _match_agent_token(
        self,
        valid: Tensor,  # [n_agent, n_step]
        pos: Tensor,  # [n_agent, n_step, 2]
        heading: Tensor,  # [n_agent, n_step]
        agent_type: Tensor,  # [n_agent]
        agent_shape: Tensor,  # [n_agent, 2]
    ) -> Dict[str, Tensor]:
        """에이전트별 큰 토큰 묶음을 만들지 않고 마지막 토큰만 바로 찾습니다.

        Args:
            valid: 유효 여부입니다. shape은 ``[n_agent, n_step]`` 입니다.
            pos: 중심점 좌표입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
            heading: 각 시점 진행 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.
            agent_type: 차종 종류입니다. shape은 ``[n_agent]`` 입니다.
            agent_shape: 토큰화에 쓰는 가로, 세로 크기입니다. shape은 ``[n_agent, 2]`` 입니다.

        Returns:
            Dict[str, Tensor]:
                coarse 간격 기준의 정답 토큰과 샘플 토큰을 담은 사전입니다.
                모든 항목의 첫 차원은 ``n_agent`` 이고 두 번째 차원은 ``n_step_token`` 입니다.
        """
        num_k = self.agent_token_sampling.num_k if self.training else 1
        n_agent, n_step = valid.shape

        prev_pos = pos[:, 0].clone()  # [n_agent, 2]
        prev_head = heading[:, 0].clone()  # [n_agent]
        prev_pos_sample = prev_pos.clone()
        prev_head_sample = prev_head.clone()

        out_dict = {
            "valid_mask": [],
            "gt_idx": [],
            "gt_pos": [],
            "gt_heading": [],
            "sampled_idx": [],
            "sampled_pos": [],
            "sampled_heading": [],
        }

        for i in range(self.shift, n_step, self.shift):
            valid_mask = valid[:, i - self.shift] & valid[:, i]
            invalid_mask = ~valid_mask
            out_dict["valid_mask"].append(valid_mask)

            gt_contour = cal_polygon_contour(pos[:, i], heading[:, i], agent_shape)
            gt_contour_local, _ = transform_to_local(
                pos_global=gt_contour,
                head_global=None,
                pos_now=prev_pos,
                head_now=prev_head,
            )
            token_idx_gt = self._match_token_idx_from_local_contour(
                agent_type=agent_type,
                contour_local=gt_contour_local,
                reduction="sum",
                num_k=1,
                sample_topk=False,
            )
            token_pos_gt, token_head_gt = self._token_pose_from_index(
                agent_type=agent_type,
                token_idx=token_idx_gt,
                ref_pos=prev_pos,
                ref_head=prev_head,
            )

            prev_head = heading[:, i].clone()
            prev_head[valid_mask] = token_head_gt[valid_mask]
            prev_pos = pos[:, i].clone()
            prev_pos[valid_mask] = token_pos_gt[valid_mask]

            out_dict["gt_idx"].append(token_idx_gt)
            out_dict["gt_pos"].append(prev_pos.masked_fill(invalid_mask.unsqueeze(1), 0.0))
            out_dict["gt_heading"].append(prev_head.masked_fill(invalid_mask, 0.0))

            if num_k == 1:
                out_dict["sampled_idx"].append(out_dict["gt_idx"][-1])
                out_dict["sampled_pos"].append(out_dict["gt_pos"][-1])
                out_dict["sampled_heading"].append(out_dict["gt_heading"][-1])
                continue

            gt_contour_local_sample, _ = transform_to_local(
                pos_global=gt_contour,
                head_global=None,
                pos_now=prev_pos_sample,
                head_now=prev_head_sample,
            )
            token_idx_sample = self._match_token_idx_from_local_contour(
                agent_type=agent_type,
                contour_local=gt_contour_local_sample,
                reduction="mean",
                num_k=num_k,
                sample_topk=True,
            )
            token_pos_sample, token_head_sample = self._token_pose_from_index(
                agent_type=agent_type,
                token_idx=token_idx_sample,
                ref_pos=prev_pos_sample,
                ref_head=prev_head_sample,
            )

            prev_head_sample = heading[:, i].clone()
            prev_head_sample[valid_mask] = token_head_sample[valid_mask]
            prev_pos_sample = pos[:, i].clone()
            prev_pos_sample[valid_mask] = token_pos_sample[valid_mask]

            out_dict["sampled_idx"].append(token_idx_sample)
            out_dict["sampled_pos"].append(
                prev_pos_sample.masked_fill(invalid_mask.unsqueeze(1), 0.0)
            )
            out_dict["sampled_heading"].append(
                prev_head_sample.masked_fill(invalid_mask, 0.0)
            )

        return {k: torch.stack(v, dim=1) for k, v in out_dict.items()}

    def _build_agent_type_masks(self, agent_type: Tensor) -> Dict[str, Tensor]:
        """차종별 마스크를 한 번에 만듭니다.

        Args:
            agent_type: 차종 번호입니다. shape은 ``[n_agent]`` 입니다.

        Returns:
            Dict[str, Tensor]:
                ``veh``, ``ped``, ``cyc`` 키를 가지는 bool 마스크 사전입니다.
                각 마스크 shape은 ``[n_agent]`` 입니다.
        """
        return build_agent_type_masks(agent_type)

    def _get_agent_shape(self, agent_type: Tensor) -> Tensor:
        """토큰화에 쓰는 고정 가로, 세로 크기를 차종별로 붙입니다.

        Args:
            agent_type: 차종 번호입니다. shape은 ``[n_agent]`` 입니다.

        Returns:
            Tensor:
                토큰화 기준 가로, 세로 크기입니다. shape은 ``[n_agent, 2]`` 입니다.
                마지막 차원은 ``[width, length]`` 순서입니다.
        """
        n_agent = agent_type.shape[0]
        agent_shape = torch.zeros(
            (n_agent, 2),
            device=agent_type.device,
            dtype=torch.float32,
        )
        agent_type_masks = self._build_agent_type_masks(agent_type)
        agent_shape[agent_type_masks["veh"]] = agent_shape.new_tensor([2.0, 4.8])
        agent_shape[agent_type_masks["ped"]] = agent_shape.new_tensor([1.0, 1.0])
        agent_shape[agent_type_masks["cyc"]] = agent_shape.new_tensor([1.0, 2.0])
        return agent_shape

    def _match_token_idx_from_local_contour(
        self,
        agent_type: Tensor,
        contour_local: Tensor,
        reduction: str,
        num_k: int,
        sample_topk: bool,
    ) -> Tensor:
        """로컬 좌표에서 바로 토큰 번호를 고릅니다.

        Args:
            agent_type: 차종 번호입니다. shape은 ``[n_agent]`` 입니다.
            contour_local: 현재 기준 좌표로 옮긴 사각형 점들입니다.
                shape은 ``[n_agent, 4, 2]`` 입니다.
            reduction: 점별 거리를 ``sum`` 또는 ``mean`` 으로 줄이는 방법입니다.
            num_k: 샘플 후보 개수입니다.
            sample_topk: True면 top-k 안에서 하나를 뽑고, False면 가장 가까운 하나만 고릅니다.

        Returns:
            Tensor:
                선택된 토큰 번호입니다. shape은 ``[n_agent]`` 입니다.
        """
        return match_token_idx_from_local_contour(
            agent_type=agent_type,
            contour_local=contour_local,
            token_bank_all_veh=self.agent_token_all_veh,
            token_bank_all_ped=self.agent_token_all_ped,
            token_bank_all_cyc=self.agent_token_all_cyc,
            reduction=reduction,
            num_k=num_k,
            sample_topk=sample_topk,
            sampling_temp=float(self.agent_token_sampling.temp),
        )

    def _token_pose_from_index(
        self,
        agent_type: Tensor,
        token_idx: Tensor,
        ref_pos: Tensor,
        ref_head: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """고른 토큰 번호를 다시 중심점과 방향으로 바꿉니다.

        Args:
            agent_type: 차종 번호입니다. shape은 ``[n_agent]`` 입니다.
            token_idx: 토큰 번호입니다. shape은 ``[n_agent]`` 입니다.
            ref_pos: 현재 기준 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            ref_head: 현재 기준 방향입니다. shape은 ``[n_agent]`` 입니다.

        Returns:
            Tuple[Tensor, Tensor]:
                - token_pos: 고른 토큰의 전역 중심점. shape은 ``[n_agent, 2]`` 입니다.
                - token_head: 고른 토큰의 전역 방향. shape은 ``[n_agent]`` 입니다.
        """
        token_pos = ref_pos.clone()
        token_head = ref_head.clone()

        for token_key, mask in self._build_agent_type_masks(agent_type).items():
            if not mask.any():
                continue

            token_bank = getattr(self, f"agent_token_all_{token_key}")[:, -1]
            token_contour_local = token_bank[token_idx[mask]]
            token_center_local = token_contour_local.mean(dim=1)
            token_center_global, _ = transform_to_global(
                pos_local=token_center_local.unsqueeze(1),
                head_local=None,
                pos_now=ref_pos[mask],
                head_now=ref_head[mask],
            )
            token_pos[mask] = token_center_global.squeeze(1)

            token_dxy_local = token_contour_local[:, 0] - token_contour_local[:, 3]
            token_head_local = torch.arctan2(token_dxy_local[:, 1], token_dxy_local[:, 0])
            token_head[mask] = wrap_angle(ref_head[mask] + token_head_local)

        return token_pos, token_head

    @staticmethod
    def _clean_heading(valid: Tensor, heading: Tensor) -> Tensor:
        valid_pairs = valid[:, :-1] & valid[:, 1:]
        for i in range(heading.shape[1] - 1):
            heading_diff = torch.abs(wrap_angle(heading[:, i] - heading[:, i + 1]))
            change_needed = (heading_diff > 1.5) & valid_pairs[:, i]
            heading[:, i + 1][change_needed] = heading[:, i][change_needed]
        return heading

    def _extrapolate_agent_to_prev_token_step(
        self,
        valid: Tensor,  # [n_agent, n_step]
        pos: Tensor,  # [n_agent, n_step, 2]
        heading: Tensor,  # [n_agent, n_step]
        vel: Tensor,  # [n_agent, n_step, 2]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # [n_agent], max will give the first True step
        first_valid_step = torch.max(valid, dim=1).indices

        for i, t in enumerate(first_valid_step):  # extrapolate to previous 5th step.
            n_step_to_extrapolate = t % self.shift
            if (t == 10) and (not valid[i, 10 - self.shift]):
                # such that at least one token is valid in the history.
                n_step_to_extrapolate = self.shift

            if n_step_to_extrapolate > 0:
                vel[i, t - n_step_to_extrapolate : t] = vel[i, t]
                valid[i, t - n_step_to_extrapolate : t] = True
                heading[i, t - n_step_to_extrapolate : t] = heading[i, t]

                for j in range(n_step_to_extrapolate):
                    pos[i, t - j - 1] = pos[i, t - j] - vel[i, t] * 0.1

        return valid, pos, heading, vel
