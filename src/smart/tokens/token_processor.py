from __future__ import annotations

import os
import pickle
from typing import Dict, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Categorical
from torch_geometric.data import HeteroData

from src.smart.utils import (
    cal_polygon_contour,
    transform_to_local,
    wrap_angle,
)


class TokenProcessor(torch.nn.Module):
    """맵 토큰화와 flow 학습용 agent 정리를 함께 수행한다.

    기존 SMART의 맵 토큰과 agent coarse token은 유지하되, 학습 목표는
    token id가 아니라 2초 길이의 연속 미래가 되도록 필요한 항목을 더 만든다.
    """

    def __init__(
        self,
        map_token_file: str,
        agent_token_file: str,
        map_token_sampling: DictConfig,
        agent_token_sampling: Optional[DictConfig] = None,
        step_current: int = 10,
        flow_num_future_steps: int = 20,
        flow_anchor_stride: int = 5,
        flow_num_anchors: int = 13,
    ) -> None:
        super().__init__()
        self.map_token_sampling = map_token_sampling
        self.agent_token_sampling = agent_token_sampling
        self.shift = 5
        self.step_current = step_current
        self.flow_num_future_steps = flow_num_future_steps
        self.flow_anchor_stride = flow_anchor_stride
        self.flow_num_anchors = flow_num_anchors

        module_dir = os.path.dirname(__file__)
        self.init_agent_token(os.path.join(module_dir, agent_token_file))
        self.init_map_token(os.path.join(module_dir, map_token_file))
        self.n_token_agent = self.agent_token_all_veh.shape[0]

    @torch.no_grad()
    def forward(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """한 배치를 SMART-flow 입력 형태로 바꾼다.

        Args:
            data: Waymo 장면 배치이다.

        Returns:
            맵 토큰 딕셔너리와 agent 토큰/flow 목표 딕셔너리를 돌려준다.
        """
        tokenized_map = self.tokenize_map(data)
        tokenized_agent = self.tokenize_agent(data)
        return tokenized_map, tokenized_agent

    def init_map_token(self, map_token_traj_path: str, argmin_sample_len: int = 3) -> None:
        """맵 토큰 파일을 읽어 메모리에 올린다."""
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

    def init_agent_token(self, agent_token_path: str) -> None:
        """agent 토큰 파일을 읽어 메모리에 올린다."""
        agent_token_data = pickle.load(open(agent_token_path, "rb"))
        for k, v in agent_token_data["token_all"].items():
            v = torch.tensor(v, dtype=torch.float32)
            self.register_buffer(f"agent_token_all_{k}", v, persistent=False)

    def tokenize_map(self, data: HeteroData) -> Dict[str, Tensor]:
        """맵을 기존 SMART 방식의 토큰으로 바꾼다."""
        traj_pos = data["map_save"]["traj_pos"]  # [n_pl, 3, 2]
        traj_theta = data["map_save"]["traj_theta"]  # [n_pl]

        traj_pos_local, _ = transform_to_local(
            pos_global=traj_pos,
            head_global=None,
            pos_now=traj_pos[:, 0],
            head_now=traj_theta,
        )
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
            )
            topk_logits = (-1e-6 - topk_dists) / self.map_token_sampling.temp
            samples = Categorical(logits=topk_logits).sample()
            token_idx = topk_indices[
                torch.arange(len(samples), device=samples.device),
                samples,
            ].contiguous()
        else:
            token_idx = torch.argmin(dist, dim=-1)

        return {
            "position": traj_pos[:, 0].contiguous(),
            "orientation": traj_theta,
            "token_idx": token_idx,
            "token_traj_src": self.map_token_traj_src,
            "type": data["pt_token"]["type"].long(),
            "pl_type": data["pt_token"]["pl_type"].long(),
            "light_type": data["pt_token"]["light_type"].long(),
            "batch": data["pt_token"]["batch"],
        }

    def tokenize_agent(self, data: HeteroData) -> Dict[str, Tensor]:
        """agent를 coarse token + flow target 형태로 정리한다.

        Returns:
            Dict[str, Tensor]:
                - `gt_idx`: [n_agent, 18] coarse token id.
                - `valid_mask`: [n_agent, 18] coarse step 유효 여부.
                - `coarse_pos`: [n_agent, 18, 2] coarse step 실제 위치.
                - `coarse_head`: [n_agent, 18] coarse step 실제 heading.
                - `flow_future_local`: [n_agent, 13, 20, 4] local GT 미래.
                - `flow_anchor_valid`: [n_agent, 13] full 2초 GT 유효 여부.
        """
        agent_shape, token_traj_all, token_traj = self._get_agent_shape_and_token_traj(
            data["agent"]["type"]
        )

        valid = data["agent"]["valid_mask"].clone()  # [n_agent, n_step]
        heading = data["agent"]["heading"].clone()  # [n_agent, n_step]
        pos = data["agent"]["position"][..., :2].contiguous()  # [n_agent, n_step, 2]
        vel = data["agent"]["velocity"].clone()  # [n_agent, n_step, 2]

        heading = self._clean_heading(valid, heading)
        valid, pos, heading, vel = self._extrapolate_agent_to_prev_token_step(
            valid, pos, heading, vel
        )

        tokenized_agent: Dict[str, Tensor] = {
            "num_graphs": data.num_graphs,
            "type": data["agent"]["type"],
            "shape": data["agent"]["shape"],
            "ego_mask": data["agent"]["role"][:, 0],
            "token_agent_shape": agent_shape,
            "batch": data["agent"]["batch"],
            "token_traj_all": token_traj_all,
            "token_traj": token_traj,
            "gt_pos_raw": pos[:, self.shift :: self.shift],
            "gt_head_raw": heading[:, self.shift :: self.shift],
            "gt_valid_raw": valid[:, self.shift :: self.shift],
            "gt_z_raw": data["agent"]["position"][:, self.step_current, 2],
        }
        for k in ["veh", "ped", "cyc"]:
            tokenized_agent[f"trajectory_token_{k}"] = getattr(
                self, f"agent_token_all_{k}"
            )[:, -1].flatten(1, 2)

        tokenized_agent["gt_idx"] = self._match_agent_token(
            valid=valid,
            pos=pos,
            heading=heading,
            agent_shape=agent_shape,
            token_traj=token_traj,
        )
        tokenized_agent["sampled_idx"] = tokenized_agent["gt_idx"]
        tokenized_agent["valid_mask"] = valid[:, self.shift :: self.shift]
        tokenized_agent["coarse_pos"] = pos[:, self.shift :: self.shift]
        tokenized_agent["coarse_head"] = heading[:, self.shift :: self.shift]

        tokenized_agent.update(
            self._build_flow_targets(
                valid=valid,
                pos=pos,
                heading=heading,
            )
        )
        return tokenized_agent

    def _build_flow_targets(
        self,
        valid: Tensor,
        pos: Tensor,
        heading: Tensor,
    ) -> Dict[str, Tensor]:
        """2초 future flow 학습 목표를 만든다.

        Args:
            valid: [n_agent, n_step] 모양의 유효 마스크이다.
            pos: [n_agent, n_step, 2] 모양의 위치이다.
            heading: [n_agent, n_step] 모양의 heading이다.

        Returns:
            Dict[str, Tensor]:
                - `flow_anchor_step`: [13] 10Hz 기준 anchor step.
                - `flow_anchor_token_idx`: [13] coarse token index.
                - `flow_anchor_pos`: [n_agent, 13, 2] anchor 위치.
                - `flow_anchor_head`: [n_agent, 13] anchor heading.
                - `flow_future_valid`: [n_agent, 13, 20] step별 유효 여부.
                - `flow_anchor_valid`: [n_agent, 13] 2초 전체 유효 여부.
                - `flow_future_local`: [n_agent, 13, 20, 4] local GT 미래.
                - `flow_future_pos`: [n_agent, 13, 20, 2] global GT 미래 위치.
                - `flow_future_head`: [n_agent, 13, 20] global GT 미래 heading.
        """
        device = pos.device
        dtype = pos.dtype

        anchor_steps = self.step_current + torch.arange(
            self.flow_num_anchors, device=device
        ) * self.flow_anchor_stride  # [13], [10, 15, ..., 70]
        future_offsets = torch.arange(
            1, self.flow_num_future_steps + 1, device=device
        )  # [20], [1, ..., 20]
        future_steps = anchor_steps.unsqueeze(-1) + future_offsets.unsqueeze(0)

        anchor_pos = pos[:, anchor_steps]  # [n_agent, 13, 2]
        anchor_head = heading[:, anchor_steps]  # [n_agent, 13]
        future_pos = pos[:, future_steps]  # [n_agent, 13, 20, 2]
        future_head = heading[:, future_steps]  # [n_agent, 13, 20]

        future_valid = valid[:, future_steps]  # [n_agent, 13, 20]
        anchor_valid = valid[:, anchor_steps] & future_valid.all(dim=-1)  # [n_agent, 13]

        future_pos_local = self._future_to_local(
            future_pos=future_pos,
            anchor_pos=anchor_pos,
            anchor_head=anchor_head,
        )  # [n_agent, 13, 20, 2]
        delta_yaw = wrap_angle(future_head - anchor_head.unsqueeze(-1))
        future_local = torch.cat(
            [
                future_pos_local,
                delta_yaw.cos().unsqueeze(-1),
                delta_yaw.sin().unsqueeze(-1),
            ],
            dim=-1,
        )
        future_local = future_local.masked_fill(~future_valid.unsqueeze(-1), 0.0)

        return {
            "flow_anchor_step": anchor_steps,
            "flow_anchor_token_idx": anchor_steps // self.shift - 1,
            "flow_anchor_pos": anchor_pos,
            "flow_anchor_head": anchor_head,
            "flow_future_valid": future_valid,
            "flow_anchor_valid": anchor_valid,
            "flow_future_local": future_local.to(dtype),
            "flow_future_pos": future_pos,
            "flow_future_head": future_head,
        }

    @staticmethod
    def _future_to_local(
        future_pos: Tensor,
        anchor_pos: Tensor,
        anchor_head: Tensor,
    ) -> Tensor:
        """future 위치를 anchor 기준 local 좌표로 바꾼다.

        Args:
            future_pos: [n_agent, n_anchor, n_future_step, 2] 모양의 global 위치이다.
            anchor_pos: [n_agent, n_anchor, 2] 모양의 anchor 위치이다.
            anchor_head: [n_agent, n_anchor] 모양의 anchor heading이다.

        Returns:
            [n_agent, n_anchor, n_future_step, 2] 모양의 local 위치이다.
        """
        n_agent, n_anchor, n_future_step, _ = future_pos.shape
        local_pos, _ = transform_to_local(
            pos_global=future_pos.reshape(-1, n_future_step, 2),
            head_global=None,
            pos_now=anchor_pos.reshape(-1, 2),
            head_now=anchor_head.reshape(-1),
        )
        return local_pos.view(n_agent, n_anchor, n_future_step, 2)

    def _match_agent_token(
        self,
        valid: Tensor,
        pos: Tensor,
        heading: Tensor,
        agent_shape: Tensor,
        token_traj: Tensor,
    ) -> Tensor:
        """각 0.5초 anchor를 가장 가까운 coarse token으로 바꾼다.

        Args:
            valid: [n_agent, n_step] 모양의 유효 마스크이다.
            pos: [n_agent, n_step, 2] 모양의 위치이다.
            heading: [n_agent, n_step] 모양의 heading이다.
            agent_shape: [n_agent, 2] 모양의 width/length이다.
            token_traj: [n_agent, n_token, 4, 2] 모양의 마지막 contour 토큰이다.

        Returns:
            [n_agent, 18] 모양의 coarse token index이다.
        """
        n_agent, n_step = valid.shape
        range_a = torch.arange(n_agent, device=valid.device)
        prev_pos = pos[:, 0].clone()
        prev_head = heading[:, 0].clone()
        token_idx_list = []

        for i in range(self.shift, n_step, self.shift):
            valid_mask = valid[:, i - self.shift] & valid[:, i]
            gt_contour = cal_polygon_contour(pos[:, i], heading[:, i], agent_shape)
            gt_contour = gt_contour.unsqueeze(1)  # [n_agent, 1, 4, 2]

            cos, sin = prev_head.cos(), prev_head.sin()
            rot_mat = torch.zeros((n_agent, 2, 2), device=prev_head.device, dtype=pos.dtype)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            token_world = torch.bmm(
                token_traj.flatten(1, 2), rot_mat
            ).view(*token_traj.shape) + prev_pos.unsqueeze(1).unsqueeze(1)

            token_idx = torch.argmin(
                torch.norm(token_world - gt_contour, dim=-1).sum(-1), dim=-1
            )
            token_idx_list.append(token_idx)

            token_contour = token_world[range_a, token_idx]
            contour_center = token_contour.mean(dim=1)
            contour_diff = token_contour[:, 0] - token_contour[:, 3]
            contour_head = torch.atan2(contour_diff[:, 1], contour_diff[:, 0])
            prev_pos[valid_mask] = contour_center[valid_mask]
            prev_head[valid_mask] = contour_head[valid_mask]
            prev_pos[~valid_mask] = pos[:, i][~valid_mask]
            prev_head[~valid_mask] = heading[:, i][~valid_mask]

        return torch.stack(token_idx_list, dim=1)

    @staticmethod
    def _clean_heading(valid: Tensor, heading: Tensor) -> Tensor:
        """갑자기 180도 뒤집힌 heading을 부드럽게 정리한다."""
        valid_pairs = valid[:, :-1] & valid[:, 1:]
        for i in range(heading.shape[1] - 1):
            heading_diff = torch.abs(wrap_angle(heading[:, i] - heading[:, i + 1]))
            change_needed = (heading_diff > 1.5) & valid_pairs[:, i]
            heading[:, i + 1][change_needed] = heading[:, i][change_needed]
        return heading

    def _extrapolate_agent_to_prev_token_step(
        self,
        valid: Tensor,
        pos: Tensor,
        heading: Tensor,
        vel: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """첫 coarse token 앞의 빈 구간을 짧게 메운다.

        Args:
            valid: [n_agent, n_step] 모양의 유효 마스크이다.
            pos: [n_agent, n_step, 2] 모양의 위치이다.
            heading: [n_agent, n_step] 모양의 heading이다.
            vel: [n_agent, n_step, 2] 모양의 속도이다.

        Returns:
            보정된 `valid`, `pos`, `heading`, `vel`을 순서대로 돌려준다.
        """
        first_valid_step = torch.max(valid, dim=1).indices

        for i, t in enumerate(first_valid_step):
            n_step_to_extrapolate = t % self.shift
            if (t == self.step_current) and (not valid[i, self.step_current - self.shift]):
                n_step_to_extrapolate = self.shift

            if n_step_to_extrapolate > 0:
                vel[i, t - n_step_to_extrapolate : t] = vel[i, t]
                valid[i, t - n_step_to_extrapolate : t] = True
                heading[i, t - n_step_to_extrapolate : t] = heading[i, t]
                for j in range(n_step_to_extrapolate):
                    pos[i, t - j - 1] = pos[i, t - j] - vel[i, t] * 0.1

        return valid, pos, heading, vel

    def _get_agent_shape_and_token_traj(
        self,
        agent_type: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """agent 종류에 맞는 폭/길이와 coarse token 사전을 만든다.

        Args:
            agent_type: [n_agent] 모양의 agent 종류이다.

        Returns:
            - agent_shape: [n_agent, 2]
            - token_traj_all: [n_agent, n_token, 6, 4, 2]
            - token_traj: [n_agent, n_token, 4, 2]
        """
        agent_type_masks = {
            "veh": agent_type == 0,
            "ped": agent_type == 1,
            "cyc": agent_type == 2,
        }
        agent_shape = 0.0
        token_traj_all = 0.0
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
            agent_shape += torch.stack([width * mask, length * mask], dim=-1)
            token_traj_all += mask[:, None, None, None, None] * (
                getattr(self, f"agent_token_all_{k}").unsqueeze(0)
            )

        token_traj = token_traj_all[:, :, -1, :, :].contiguous()
        return agent_shape, token_traj_all, token_traj
