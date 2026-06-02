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
        agent_type = self._normalize_agent_type(data["agent"]["type"])
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
            ).flatten(1, 3)
            tokenized_agent[f"token_bank_all_{k}"] = getattr(self, f"agent_token_all_{k}")

        if not self.training:
            (
                rollout_init_fine_pos_pair,
                rollout_init_fine_head_pair,
                rollout_init_fine_valid_pair,
            ) = self._build_rollout_init_fine_pair(
                valid=valid,
                pos=pos,
                heading=heading,
            )
            (
                rollout_init_fine_pos_history,
                rollout_init_fine_head_history,
                rollout_init_fine_valid_history,
            ) = self._build_rollout_init_fine_history(
                valid=valid,
                pos=pos,
                heading=heading,
            )
            tokenized_agent.update(
                {
                    "gt_pos_raw": pos[:, self.shift :: self.shift],
                    "gt_head_raw": heading[:, self.shift :: self.shift],
                    "gt_valid_raw": valid[:, self.shift :: self.shift],
                    "gt_z_raw": data["agent"]["position"][:, 10, 2],
                    "rollout_init_fine_pos_pair": rollout_init_fine_pos_pair,
                    "rollout_init_fine_head_pair": rollout_init_fine_head_pair,
                    "rollout_init_fine_valid_pair": rollout_init_fine_valid_pair,
                    "rollout_init_fine_pos_history": rollout_init_fine_pos_history,
                    "rollout_init_fine_head_history": rollout_init_fine_head_history,
                    "rollout_init_fine_valid_history": rollout_init_fine_valid_history,
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

    def _build_rollout_init_fine_pair(
        self,
        valid: Tensor,
        pos: Tensor,
        heading: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """closed-loop 첫 block에서 쓸 10Hz 마지막 두 실제 상태를 만듭니다.

        현재 semi-continuous closed-loop는 raw step 10을 현재 coarse 시점으로 씁니다.
        그래서 첫 dynamics-aware commit은 raw step 9와 10 사이의 실제 변화량을
        바로 시작 상태로 쓰는 것이 가장 자연스럽습니다.

        Args:
            valid: 전체 유효 여부입니다. shape은 ``[n_agent, n_step]`` 입니다.
            pos: 전체 중심점입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
            heading: 전체 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - 최근 fine 중심점 2개 ``[n_agent, 2, 2]``
                - 최근 fine 방향 2개 ``[n_agent, 2]``
                - 최근 fine 유효 여부 2개 ``[n_agent, 2]``
        """
        current_raw_step = min(self.shift * 2, pos.shape[1] - 1)
        start_step = max(current_raw_step - 1, 0)
        pos_pair = pos[:, start_step : current_raw_step + 1].contiguous()
        head_pair = heading[:, start_step : current_raw_step + 1].contiguous()
        valid_pair = valid[:, start_step : current_raw_step + 1].contiguous()
        if pos_pair.shape[1] == 2:
            return pos_pair, head_pair, valid_pair

        pos_pair = torch.cat([pos_pair, pos_pair], dim=1)
        head_pair = torch.cat([head_pair, head_pair], dim=1)
        valid_pair = torch.cat([valid_pair, valid_pair], dim=1)
        return pos_pair, head_pair, valid_pair

    def _build_rollout_init_fine_history(
        self,
        valid: Tensor,
        pos: Tensor,
        heading: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """closed-loop LQR bridge가 쓸 최근 0.5초 실제 10Hz 상태 6개를 만듭니다.

        현재 semi-continuous closed-loop는 raw step 10을 현재 coarse 시점으로 씁니다.
        그래서 raw step 5~10 전체를 그대로 넘기면, 현재 시점과 직전 0.5초 실제
        실행 이력을 함께 사용할 수 있습니다. 기록 길이가 부족하면 가장 앞 상태를
        반복해 길이를 6으로 맞춥니다.

        Args:
            valid: 전체 유효 여부입니다. shape은 ``[n_agent, n_step]`` 입니다.
            pos: 전체 중심점입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
            heading: 전체 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - 최근 fine 중심점 6개 ``[n_agent, 6, 2]``
                - 최근 fine 방향 6개 ``[n_agent, 6]``
                - 최근 fine 유효 여부 6개 ``[n_agent, 6]``
        """
        current_raw_step = min(self.shift * 2, pos.shape[1] - 1)
        start_step = max(current_raw_step - self.shift, 0)
        pos_history = pos[:, start_step : current_raw_step + 1].contiguous()
        head_history = heading[:, start_step : current_raw_step + 1].contiguous()
        valid_history = valid[:, start_step : current_raw_step + 1].contiguous()

        history_len = self.shift + 1
        if pos_history.shape[1] >= history_len:
            return pos_history[:, -history_len:], head_history[:, -history_len:], valid_history[:, -history_len:]

        pad_len = history_len - pos_history.shape[1]
        pos_pad = pos_history[:, :1].expand(-1, pad_len, -1)
        head_pad = head_history[:, :1].expand(-1, pad_len)
        valid_pad = valid_history[:, :1].expand(-1, pad_len)
        return (
            torch.cat([pos_pad, pos_history], dim=1),
            torch.cat([head_pad, head_history], dim=1),
            torch.cat([valid_pad, valid_history], dim=1),
        )

    def _match_agent_token(
        self,
        valid: Tensor,  # [n_agent, n_step]
        pos: Tensor,  # [n_agent, n_step, 2]
        heading: Tensor,  # [n_agent, n_step]
        agent_type: Tensor,  # [n_agent]
        agent_shape: Tensor,  # [n_agent, 2]
    ) -> Dict[str, Tensor]:
        """6개 점 경로 전체를 기준으로 토큰 번호를 찾고 실제 coarse 상태를 보존합니다.

        Args:
            valid: 유효 여부입니다. shape은 ``[n_agent, n_step]`` 입니다.
            pos: 중심점 좌표입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
            heading: 각 시점 진행 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.
            agent_type: 차종 종류입니다. shape은 ``[n_agent]`` 입니다.
            agent_shape: 토큰화에 쓰는 가로, 세로 크기입니다. shape은 ``[n_agent, 2]`` 입니다.

        Returns:
            Dict[str, Tensor]:
                coarse 간격 기준의 정답 토큰과 샘플 토큰, 그리고 실제 coarse 상태를 담은 사전입니다.
                모든 항목의 첫 차원은 ``n_agent`` 이고 두 번째 차원은 ``n_step_token`` 입니다.
        """
        n_agent, n_step = valid.shape
        device = pos.device

        coarse_end_steps = torch.arange(self.shift, n_step, self.shift, device=device)
        n_token_step = int(coarse_end_steps.numel())
        if n_token_step == 0:
            empty_bool = valid.new_zeros((n_agent, 0))
            empty_idx = torch.zeros((n_agent, 0), device=device, dtype=torch.long)
            empty_pos = pos.new_zeros((n_agent, 0, 2))
            empty_heading = heading.new_zeros((n_agent, 0))
            return {
                "valid_mask": empty_bool,
                "gt_idx": empty_idx,
                "gt_pos": empty_pos,
                "gt_heading": empty_heading,
                "sampled_idx": empty_idx,
                "sampled_pos": empty_pos,
                "sampled_heading": empty_heading,
            }

        coarse_start_steps = coarse_end_steps - self.shift
        window_offsets = torch.arange(self.shift + 1, device=device)
        segment_step_index = coarse_start_steps.unsqueeze(1) + window_offsets.unsqueeze(0)

        segment_valid_mask = valid[:, segment_step_index].all(dim=-1)
        invalid_mask = ~segment_valid_mask

        token_idx_gt = torch.zeros(
            (n_agent, n_token_step),
            device=device,
            dtype=torch.long,
        )
        valid_flat = segment_valid_mask.reshape(-1)
        if bool(valid_flat.any().item()):
            gt_contour_local = self._build_local_contour_windows(
                pos=pos,
                heading=heading,
                agent_shape=agent_shape,
                coarse_start_steps=coarse_start_steps,
                segment_step_index=segment_step_index,
            )
            flat_agent_type = (
                agent_type.unsqueeze(1)
                .expand(-1, n_token_step)
                .reshape(-1)
            )
            token_idx_gt_flat = token_idx_gt.reshape(-1)
            token_idx_gt_flat[valid_flat] = self._match_token_idx_from_local_contour(
                agent_type=flat_agent_type[valid_flat],
                contour_local=gt_contour_local.reshape(
                    n_agent * n_token_step,
                    self.shift + 1,
                    4,
                    2,
                )[valid_flat],
                reduction="sum",
            )
            token_idx_gt = token_idx_gt_flat.view(n_agent, n_token_step)

        gt_pos = pos[:, coarse_end_steps].masked_fill(invalid_mask.unsqueeze(-1), 0.0)
        gt_heading = heading[:, coarse_end_steps].masked_fill(invalid_mask, 0.0)

        return {
            "valid_mask": segment_valid_mask,
            "gt_idx": token_idx_gt,
            "gt_pos": gt_pos,
            "gt_heading": gt_heading,
            "sampled_idx": token_idx_gt,
            "sampled_pos": gt_pos,
            "sampled_heading": gt_heading,
        }

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

    def _normalize_agent_type(self, agent_type: Tensor) -> Tensor:
        if agent_type.dim() <= 1:
            return agent_type
        if agent_type.shape[-1] == 1:
            return agent_type.reshape(-1)
        return agent_type.argmax(dim=-1)

    def _get_agent_shape(self, agent_type: Tensor) -> Tensor:
        """토큰화에 쓰는 고정 가로, 세로 크기를 차종별로 붙입니다.

        Args:
            agent_type: 차종 번호입니다. shape은 ``[n_agent]`` 입니다.

        Returns:
            Tensor:
                토큰화 기준 가로, 세로 크기입니다. shape은 ``[n_agent, 2]`` 입니다.
                마지막 차원은 ``[width, length]`` 순서입니다.
        """
        agent_type = self._normalize_agent_type(agent_type)
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


    def _build_local_contour_sequence(
        self,
        pos_seq: Tensor,
        heading_seq: Tensor,
        ref_pos: Tensor,
        ref_head: Tensor,
        agent_shape: Tensor,
    ) -> Tensor:
        """현재 coarse 시작 상태를 기준으로 경로 전체 사각형을 local 좌표로 만듭니다.

        Args:
            pos_seq: 이번 coarse 구간의 중심점 시퀀스입니다.
                shape은 ``[n_agent, n_seq, 2]`` 입니다.
            heading_seq: 이번 coarse 구간의 진행 방향 시퀀스입니다.
                shape은 ``[n_agent, n_seq]`` 입니다.
            ref_pos: local 좌표의 원점으로 쓸 실제 중심점입니다.
                shape은 ``[n_agent, 2]`` 입니다.
            ref_head: local 좌표의 원점으로 쓸 실제 방향입니다.
                shape은 ``[n_agent]`` 입니다.
            agent_shape: 토큰화에 쓰는 고정 가로, 세로 크기입니다.
                shape은 ``[n_agent, 2]`` 입니다.

        Returns:
            Tensor:
                local 좌표의 사각형 경로입니다. shape은 ``[n_agent, n_seq, 4, 2]`` 입니다.
        """
        contour_global = cal_polygon_contour(
            pos=pos_seq,
            head=heading_seq,
            width_length=agent_shape.unsqueeze(1),
        )
        contour_local_flat, _ = transform_to_local(
            pos_global=contour_global.flatten(1, 2),
            head_global=None,
            pos_now=ref_pos,
            head_now=ref_head,
        )
        return contour_local_flat.view(pos_seq.shape[0], pos_seq.shape[1], 4, 2)

    def _build_local_contour_windows(
        self,
        pos: Tensor,
        heading: Tensor,
        agent_shape: Tensor,
        coarse_start_steps: Tensor,
        segment_step_index: Tensor,
    ) -> Tensor:
        """모든 coarse segment의 local contour를 한 번에 만듭니다.

        Args:
            pos: 전체 중심점입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
            heading: 전체 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.
            agent_shape: 토큰화에 쓰는 고정 가로, 세로 크기입니다.
                shape은 ``[n_agent, 2]`` 입니다.
            coarse_start_steps: 각 coarse segment의 시작 raw step입니다.
                shape은 ``[n_token_step]`` 입니다.
            segment_step_index: 각 coarse segment가 참조하는 raw step index입니다.
                shape은 ``[n_token_step, shift + 1]`` 입니다.

        Returns:
            Tensor:
                local 좌표의 사각형 경로입니다.
                shape은 ``[n_agent, n_token_step, shift + 1, 4, 2]`` 입니다.
        """
        n_agent = pos.shape[0]
        n_token_step = int(coarse_start_steps.numel())
        n_seq = int(segment_step_index.shape[1])

        pos_seq = pos[:, segment_step_index]
        heading_seq = heading[:, segment_step_index]
        contour_global = cal_polygon_contour(
            pos=pos_seq,
            head=heading_seq,
            width_length=agent_shape[:, None, None],
        )

        ref_pos = pos[:, coarse_start_steps].reshape(n_agent * n_token_step, 2)
        ref_head = heading[:, coarse_start_steps].reshape(n_agent * n_token_step)
        contour_local_flat, _ = transform_to_local(
            pos_global=contour_global.reshape(n_agent * n_token_step, n_seq * 4, 2),
            head_global=None,
            pos_now=ref_pos,
            head_now=ref_head,
        )
        return contour_local_flat.view(n_agent, n_token_step, n_seq, 4, 2)

    def _match_token_idx_from_local_contour(
        self,
        agent_type: Tensor,
        contour_local: Tensor,
        reduction: str,
    ) -> Tensor:
        """로컬 좌표에서 바로 토큰 번호를 고릅니다.

        Args:
            agent_type: 차종 번호입니다. shape은 ``[n_agent]`` 입니다.
            contour_local: 현재 기준 좌표로 옮긴 사각형 경로입니다.
                기본 shape은 ``[n_agent, 6, 4, 2]`` 이고, 호환을 위해
                ``[n_agent, 4, 2]`` 도 받을 수 있습니다.
            reduction: 점별 거리를 ``sum`` 또는 ``mean`` 으로 줄이는 방법입니다.

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
