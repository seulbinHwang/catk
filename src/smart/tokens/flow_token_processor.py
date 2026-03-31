from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.smart.modules.continuous_motion_history import build_context_from_raw
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import transform_to_local, wrap_angle


class FlowTokenProcessor(TokenProcessor):
    """연속 5-point agent history만 쓰는 flow 전용 토큰 처리기입니다.

    map token은 기존과 같이 유지하지만, agent 움직임은 더 이상 2048개 어휘집과
    매칭하지 않습니다. 대신 실제 10Hz 좌표를 바로 정리해서 0.5초 구간 5점 문맥과
    flow supervision을 만듭니다.
    """

    def __init__(
        self,
        map_token_file: str,
        agent_token_file: str,
        map_token_sampling,
        agent_token_sampling,
    ) -> None:
        """map token만 초기화하고 agent vocab 로드는 생략합니다.

        Args:
            map_token_file: map token 파일 경로입니다.
            agent_token_file: 기존 설정 호환용 인자입니다. 더 이상 쓰지 않습니다.
            map_token_sampling: map token 샘플링 설정입니다.
            agent_token_sampling: 기존 설정 호환용 인자입니다. 더 이상 쓰지 않습니다.
        """
        del agent_token_file
        torch.nn.Module.__init__(self)
        self.map_token_sampling = map_token_sampling
        self.agent_token_sampling = agent_token_sampling
        self.shift = 5
        self.num_context_steps = 14
        self.num_anchor_steps = 13
        module_dir = os.path.dirname(__file__)
        self.init_map_token(os.path.join(module_dir, map_token_file))
        self.n_token_agent = 0

    @torch.no_grad()
    def forward(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """지도 토큰과 연속 agent 문맥/목표를 만듭니다.

        Args:
            data: 원본 장면 배치입니다.

        Returns:
            Tuple[Dict[str, Tensor], Dict[str, Tensor]]: 지도 토큰 사전과 agent 사전입니다.
        """
        tokenized_map = self.tokenize_map(data)
        tokenized_agent, processed_agent = self._tokenize_agent_continuous(data)
        tokenized_agent = self._build_flow_targets(
            data=data,
            tokenized_agent=tokenized_agent,
            processed_agent=processed_agent,
        )
        return tokenized_map, tokenized_agent

    def _tokenize_agent_continuous(
        self,
        data: HeteroData,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """agent 원본 시계열을 연속 좌표 기반 사전으로 바꿉니다.

        Args:
            data: 원본 장면 배치입니다.

        Returns:
            Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
                - tokenized_agent: 학습/추론 공통 메타데이터 사전입니다.
                - processed_agent: 10Hz 전처리 결과 사전입니다.
        """
        valid = data["agent"]["valid_mask"].clone()
        heading = data["agent"]["heading"].clone()
        pos = data["agent"]["position"][..., :2].clone().contiguous()
        vel = data["agent"]["velocity"].clone()

        heading = self._clean_heading(valid=valid, heading=heading)
        valid, pos, heading, vel = self._extrapolate_agent_to_prev_token_step(
            valid=valid,
            pos=pos,
            heading=heading,
            vel=vel,
        )

        coarse_pos = pos[:, self.shift :: self.shift].contiguous()
        coarse_heading = heading[:, self.shift :: self.shift].contiguous()
        coarse_valid = valid[:, self.shift :: self.shift].contiguous()

        current_step_10hz = min(self.num_context_steps * self.shift, data["agent"]["position"].shape[1] - 1)
        gt_z_raw = data["agent"]["position"][:, current_step_10hz, 2].contiguous()

        tokenized_agent = {
            "num_graphs": data.num_graphs,
            "type": data["agent"]["type"],
            "shape": data["agent"]["shape"],
            "ego_mask": data["agent"]["role"][:, 0],
            "batch": data["agent"]["batch"],
            # full 10Hz trajectory for continuous rollout and target construction
            "gt_pos_raw": pos,
            "gt_head_raw": heading,
            "gt_valid_raw": valid,
            # coarse view kept for downstream reporting compatibility
            "gt_pos": coarse_pos,
            "gt_heading": coarse_heading,
            "valid_mask": coarse_valid,
            "gt_z_raw": gt_z_raw,
            # legacy placeholder only; there is no motion vocab anymore.
            "trajectory_token_veh": pos.new_zeros((0, 8)),
            "trajectory_token_ped": pos.new_zeros((0, 8)),
            "trajectory_token_cyc": pos.new_zeros((0, 8)),
        }
        processed_agent = {
            "valid": valid,
            "pos": pos,
            "heading": heading,
            "vel": vel,
        }
        return tokenized_agent, processed_agent

    def _build_flow_targets(
        self,
        data: HeteroData,
        tokenized_agent: Dict[str, Tensor],
        processed_agent: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """학습/평가에 필요한 anchor별 미래와 문맥을 만듭니다.

        Args:
            data: 원본 장면 배치입니다.
            tokenized_agent: 공통 agent 메타데이터 사전입니다.
            processed_agent: 10Hz 전처리 결과 사전입니다.

        Returns:
            Dict[str, Tensor]: flow 관련 필드가 추가된 agent 사전입니다.
        """
        valid = processed_agent["valid"]
        pos = processed_agent["pos"]
        heading = processed_agent["heading"]

        ctx_motion_local, ctx_pos, ctx_heading, ctx_valid = build_context_from_raw(
            pos_raw=pos,
            head_raw=heading,
            valid_raw=valid,
            shift=self.shift,
            num_context_steps=self.num_context_steps,
        )

        num_agent = pos.shape[0]
        device = pos.device
        dtype = pos.dtype
        raw_current_steps = list(range(10, 71, self.shift))

        if "train_mask" in data["agent"]:
            train_mask = data["agent"]["train_mask"].bool()
        else:
            train_mask = torch.ones(num_agent, device=device, dtype=torch.bool)

        tokenized_agent.update(
            {
                "ctx_motion_local": ctx_motion_local,
                "ctx_pos": ctx_pos,
                "ctx_heading": ctx_heading,
                "ctx_valid": ctx_valid,
            }
        )

        if self.training:
            flow_train_mask = torch.zeros(
                num_agent,
                self.num_anchor_steps,
                device=device,
                dtype=torch.bool,
            )
            flow_train_chunks: List[Tensor] = []
            flow_train_agent_type_chunks: List[Tensor] = []
            flow_train_prev_control_chunks: List[Tensor] = []
            flow_train_prev_control_valid_chunks: List[Tensor] = []

            for anchor_offset, raw_step in enumerate(raw_current_steps):
                current_valid = ctx_valid[:, anchor_offset + 1]
                future_valid = valid[:, raw_step + 1 : raw_step + 21].all(dim=1)
                anchor_mask = current_valid & future_valid
                train_anchor_mask = anchor_mask & train_mask
                flow_train_mask[:, anchor_offset] = train_anchor_mask
                if not train_anchor_mask.any():
                    continue

                current_pos = ctx_pos[:, anchor_offset + 1]
                current_head = ctx_heading[:, anchor_offset + 1]
                flow_train_chunks.append(
                    self._build_anchor_clean_norm(
                        pos=pos,
                        heading=heading,
                        current_pos=current_pos,
                        current_head=current_head,
                        anchor_mask=train_anchor_mask,
                        raw_step=raw_step,
                    )
                )
                prev_control, prev_control_valid = self._build_anchor_prev_control(
                    pos=pos,
                    heading=heading,
                    valid=valid,
                    current_pos=current_pos,
                    current_head=current_head,
                    anchor_mask=train_anchor_mask,
                    raw_step=raw_step,
                )
                flow_train_agent_type_chunks.append(tokenized_agent["type"][train_anchor_mask])
                flow_train_prev_control_chunks.append(prev_control)
                flow_train_prev_control_valid_chunks.append(prev_control_valid)

            tokenized_agent.update(
                {
                    "flow_train_mask": flow_train_mask,
                    "flow_train_clean_norm": self._concat_flow_chunks(
                        chunks=flow_train_chunks,
                        dtype=dtype,
                        device=device,
                    ),
                    "flow_train_agent_type": self._concat_vector_chunks(
                        chunks=flow_train_agent_type_chunks,
                        dtype=tokenized_agent["type"].dtype,
                        device=device,
                    ),
                    "flow_train_prev_control": self._concat_matrix_chunks(
                        chunks=flow_train_prev_control_chunks,
                        width=3,
                        dtype=dtype,
                        device=device,
                    ),
                    "flow_train_prev_control_valid": self._concat_vector_chunks(
                        chunks=flow_train_prev_control_valid_chunks,
                        dtype=torch.bool,
                        device=device,
                    ),
                }
            )
            return tokenized_agent

        flow_eval_mask = torch.zeros(
            num_agent,
            self.num_anchor_steps,
            device=device,
            dtype=torch.bool,
        )
        flow_eval_chunks: List[Tensor] = []
        for anchor_offset, raw_step in enumerate(raw_current_steps):
            current_valid = ctx_valid[:, anchor_offset + 1]
            future_valid = valid[:, raw_step + 1 : raw_step + 21].all(dim=1)
            anchor_mask = current_valid & future_valid
            flow_eval_mask[:, anchor_offset] = anchor_mask
            if not anchor_mask.any():
                continue

            flow_eval_chunks.append(
                self._build_anchor_clean_norm(
                    pos=pos,
                    heading=heading,
                    current_pos=ctx_pos[:, anchor_offset + 1],
                    current_head=ctx_heading[:, anchor_offset + 1],
                    anchor_mask=anchor_mask,
                    raw_step=raw_step,
                )
            )

        tokenized_agent.update(
            {
                "flow_eval_mask": flow_eval_mask,
                "flow_eval_clean_norm": self._concat_flow_chunks(
                    chunks=flow_eval_chunks,
                    dtype=dtype,
                    device=device,
                ),
            }
        )
        return tokenized_agent

    def _build_anchor_clean_norm(
        self,
        pos: Tensor,
        heading: Tensor,
        current_pos: Tensor,
        current_head: Tensor,
        anchor_mask: Tensor,
        raw_step: int,
    ) -> Tensor:
        """한 anchor에서 실제로 쓰는 agent만 골라 2초 미래 목표를 만듭니다.

        Args:
            pos: 전처리된 10Hz 중심점입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
            heading: 전처리된 10Hz 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.
            current_pos: 현재 coarse anchor 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: 현재 coarse anchor 방향입니다. shape은 ``[n_agent]`` 입니다.
            anchor_mask: 이번 anchor를 실제로 쓰는지 나타냅니다. shape은 ``[n_agent]`` 입니다.
            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.

        Returns:
            Tensor: 정규화된 2초 미래 목표입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
        """
        future_pos = pos[anchor_mask, raw_step + 1 : raw_step + 21]
        future_head = heading[anchor_mask, raw_step + 1 : raw_step + 21]
        future_pos_local, future_head_local = transform_to_local(
            pos_global=future_pos,
            head_global=future_head,
            pos_now=current_pos[anchor_mask],
            head_now=current_head[anchor_mask],
        )
        return torch.stack(
            [
                future_pos_local[..., 0] / 20.0,
                future_pos_local[..., 1] / 20.0,
                future_head_local.cos(),
                future_head_local.sin(),
            ],
            dim=-1,
        )

    def _build_anchor_prev_control(
        self,
        pos: Tensor,
        heading: Tensor,
        valid: Tensor,
        current_pos: Tensor,
        current_head: Tensor,
        anchor_mask: Tensor,
        raw_step: int,
    ) -> Tuple[Tensor, Tensor]:
        """anchor 직전 구간의 단순 제어를 local frame 기준으로 만듭니다.

        Args:
            pos: 전처리된 10Hz 중심점입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
            heading: 전처리된 10Hz 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.
            valid: 각 시점 유효 여부입니다. shape은 ``[n_agent, n_step]`` 입니다.
            current_pos: 현재 coarse anchor 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: 현재 coarse anchor 방향입니다. shape은 ``[n_agent]`` 입니다.
            anchor_mask: 이번 anchor를 실제로 쓰는 agent입니다. shape은 ``[n_agent]`` 입니다.
            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.

        Returns:
            Tuple[Tensor, Tensor]: 직전 제어 ``[v_x^b, v_y^b, omega]`` 와 유효 마스크입니다.
            shape은 각각 ``[n_valid_anchor, 3]`` 과 ``[n_valid_anchor]`` 입니다.
        """
        num_valid_anchor = int(anchor_mask.sum().item())
        if num_valid_anchor == 0:
            return (
                pos.new_zeros((0, 3)),
                torch.zeros((0,), device=pos.device, dtype=torch.bool),
            )

        prev_control_valid = valid[anchor_mask, raw_step] & valid[anchor_mask, raw_step - 1]
        prev_control = pos.new_zeros((num_valid_anchor, 3))
        if not prev_control_valid.any():
            return prev_control, prev_control_valid

        pos_pair = pos[anchor_mask, raw_step - 1 : raw_step + 1]
        head_pair = heading[anchor_mask, raw_step - 1 : raw_step + 1]
        pos_pair_local, head_pair_local = transform_to_local(
            pos_global=pos_pair,
            head_global=head_pair,
            pos_now=current_pos[anchor_mask],
            head_now=current_head[anchor_mask],
        )
        delta_pos = pos_pair_local[:, 1] - pos_pair_local[:, 0]
        prev_head_local = head_pair_local[:, 0]
        delta_head = self._wrap_angle(head_pair_local[:, 1] - head_pair_local[:, 0])
        cos_prev = prev_head_local.cos()
        sin_prev = prev_head_local.sin()
        prev_control[:, 0] = (delta_pos[:, 0] * cos_prev + delta_pos[:, 1] * sin_prev) / 0.1
        prev_control[:, 1] = (-delta_pos[:, 0] * sin_prev + delta_pos[:, 1] * cos_prev) / 0.1
        prev_control[:, 2] = delta_head / 0.1
        prev_control[~prev_control_valid] = 0.0
        return prev_control, prev_control_valid

    @staticmethod
    def _concat_flow_chunks(
        chunks: List[Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """flow 목표 조각을 하나로 합칩니다."""
        if len(chunks) == 0:
            return torch.zeros((0, 20, 4), device=device, dtype=dtype)
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _concat_vector_chunks(
        chunks: List[Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """1차원 조각 목록을 하나의 벡터로 잇습니다."""
        if len(chunks) == 0:
            return torch.zeros((0,), device=device, dtype=dtype)
        return torch.cat([chunk.to(device=device, dtype=dtype) for chunk in chunks], dim=0)

    @staticmethod
    def _concat_matrix_chunks(
        chunks: List[Tensor],
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """2차원 조각 목록을 하나의 행렬로 잇습니다."""
        if len(chunks) == 0:
            return torch.zeros((0, width), device=device, dtype=dtype)
        return torch.cat([chunk.to(device=device, dtype=dtype) for chunk in chunks], dim=0)

    @staticmethod
    def _clean_heading(valid: Tensor, heading: Tensor) -> Tensor:
        """갑자기 180도 뒤집히는 heading 값을 완만하게 정리합니다."""
        valid_pairs = valid[:, :-1] & valid[:, 1:]
        heading = heading.clone()
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
        """첫 유효 시점 앞쪽을 0.5초 경계까지 짧게 메꿉니다.

        Args:
            valid: 유효 여부입니다. shape은 ``[n_agent, n_step]`` 입니다.
            pos: 중심점입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
            heading: 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.
            vel: 속도입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: 보정된 ``valid, pos, heading, vel`` 입니다.
        """
        valid = valid.clone()
        pos = pos.clone()
        heading = heading.clone()
        vel = vel.clone()
        first_valid_step = torch.max(valid, dim=1).indices
        for agent_idx, step_idx in enumerate(first_valid_step.tolist()):
            n_step_to_extrapolate = step_idx % self.shift
            if (step_idx == 10) and (not bool(valid[agent_idx, 10 - self.shift])):
                n_step_to_extrapolate = self.shift
            if n_step_to_extrapolate <= 0:
                continue
            vel[agent_idx, step_idx - n_step_to_extrapolate : step_idx] = vel[agent_idx, step_idx]
            valid[agent_idx, step_idx - n_step_to_extrapolate : step_idx] = True
            heading[agent_idx, step_idx - n_step_to_extrapolate : step_idx] = heading[agent_idx, step_idx]
            for j in range(n_step_to_extrapolate):
                pos[agent_idx, step_idx - j - 1] = pos[agent_idx, step_idx - j] - vel[agent_idx, step_idx] * 0.1
        return valid, pos, heading, vel

    @staticmethod
    def _wrap_angle(angle: Tensor) -> Tensor:
        """각도를 ``[-pi, pi]`` 범위로 접습니다."""
        return torch.atan2(angle.sin(), angle.cos())
