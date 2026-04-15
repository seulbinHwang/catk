from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import transform_to_local


class FlowTokenProcessor(TokenProcessor):
    """Flow 학습용 목표와 추가 penalty용 보조 메타데이터를 만듭니다."""

    def forward(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """지도 토큰과 에이전트 토큰을 만들고 flow 목표를 붙입니다.

        Args:
            data: 원본 장면 배치입니다.

        Returns:
            Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
                지도 토큰 사전과 에이전트 토큰 사전입니다.
        """
        tokenized_map = self.tokenize_map(data)
        tokenized_agent, processed_agent = self.tokenize_agent(
            data,
            return_preprocessed=True,
        )
        tokenized_agent = self._build_flow_targets(
            data=data,
            tokenized_agent=tokenized_agent,
            processed_agent=processed_agent,
        )
        return tokenized_map, tokenized_agent

    def _build_flow_targets(
        self,
        data: HeteroData,
        tokenized_agent: Dict[str, Tensor],
        processed_agent: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """학습/평가에 필요한 anchor별 미래와 메타데이터를 만듭니다.

        Args:
            data: 원본 장면 배치입니다.
            tokenized_agent: coarse token 기반 에이전트 토큰 사전입니다.
            processed_agent: 전처리된 실제 좌표와 방향 사전입니다.

        Returns:
            Dict[str, Tensor]:
                flow 관련 필드가 추가된 에이전트 토큰 사전입니다.
        """
        valid = processed_agent["valid"]
        pos = processed_agent["pos"]
        heading = processed_agent["heading"]

        ctx_sampled_idx = tokenized_agent["sampled_idx"][:, :14].contiguous()
        ctx_sampled_pos = tokenized_agent["sampled_pos"][:, :14].contiguous()
        ctx_sampled_heading = tokenized_agent["sampled_heading"][:, :14].contiguous()
        ctx_valid = tokenized_agent["valid_mask"][:, :14].contiguous()

        num_agent = pos.shape[0]
        device = pos.device
        dtype = pos.dtype
        num_anchor = 13
        raw_current_steps = list(range(10, 71, self.shift))

        if "train_mask" in data["agent"]:
            train_mask = data["agent"]["train_mask"].bool()
        else:
            train_mask = torch.ones(num_agent, device=device, dtype=torch.bool)

        tokenized_agent.update(
            {
                "ctx_sampled_idx": ctx_sampled_idx,
                "ctx_sampled_pos": ctx_sampled_pos,
                "ctx_sampled_heading": ctx_sampled_heading,
                "ctx_valid": ctx_valid,
            }
        )

        if self.training:
            flow_train_mask = torch.zeros(num_agent, num_anchor, device=device, dtype=torch.bool)
            flow_train_chunks: List[Tensor] = []
            flow_train_agent_type_chunks: List[Tensor] = []
            flow_train_prev_control_chunks: List[Tensor] = []
            flow_train_prev_control_valid_chunks: List[Tensor] = []
            flow_train_current_pos_chunks: List[Tensor] = []
            flow_train_current_head_chunks: List[Tensor] = []
            flow_train_exec_pos_history_chunks: List[Tensor] = []
            flow_train_exec_head_history_chunks: List[Tensor] = []
            flow_train_exec_valid_history_chunks: List[Tensor] = []

            for anchor_offset, raw_step in enumerate(raw_current_steps):
                current_valid = valid[:, raw_step]
                future_valid = valid[:, raw_step + 1 : raw_step + 21].all(dim=1)
                anchor_mask = current_valid & future_valid
                train_anchor_mask = anchor_mask & train_mask
                flow_train_mask[:, anchor_offset] = train_anchor_mask
                if not train_anchor_mask.any():
                    continue

                current_pos = pos[:, raw_step]
                current_head = heading[:, raw_step]
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
                exec_pos_history, exec_head_history, exec_valid_history = self._build_anchor_exec_history(
                    pos=pos,
                    heading=heading,
                    valid=valid,
                    anchor_mask=train_anchor_mask,
                    raw_step=raw_step,
                    history_steps=6,
                )
                flow_train_agent_type_chunks.append(tokenized_agent["type"][train_anchor_mask])
                flow_train_prev_control_chunks.append(prev_control)
                flow_train_prev_control_valid_chunks.append(prev_control_valid)
                flow_train_current_pos_chunks.append(current_pos[train_anchor_mask])
                flow_train_current_head_chunks.append(current_head[train_anchor_mask])
                flow_train_exec_pos_history_chunks.append(exec_pos_history)
                flow_train_exec_head_history_chunks.append(exec_head_history)
                flow_train_exec_valid_history_chunks.append(exec_valid_history)

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
                    "flow_train_current_pos": self._concat_matrix_chunks(
                        chunks=flow_train_current_pos_chunks,
                        width=2,
                        dtype=dtype,
                        device=device,
                    ),
                    "flow_train_current_head": self._concat_vector_chunks(
                        chunks=flow_train_current_head_chunks,
                        dtype=dtype,
                        device=device,
                    ),
                    "flow_train_exec_pos_history": self._concat_rank3_chunks(
                        chunks=flow_train_exec_pos_history_chunks,
                        dim1=6,
                        dim2=2,
                        dtype=dtype,
                        device=device,
                    ),
                    "flow_train_exec_head_history": self._concat_matrix_chunks(
                        chunks=flow_train_exec_head_history_chunks,
                        width=6,
                        dtype=dtype,
                        device=device,
                    ),
                    "flow_train_exec_valid_history": self._concat_matrix_chunks(
                        chunks=flow_train_exec_valid_history_chunks,
                        width=6,
                        dtype=torch.bool,
                        device=device,
                    ),
                }
            )
            for key in [
                "valid_mask",
                "gt_idx",
                "gt_pos",
                "gt_heading",
                "sampled_idx",
                "sampled_pos",
                "sampled_heading",
            ]:
                tokenized_agent.pop(key, None)
            return tokenized_agent

        flow_eval_mask = torch.zeros(num_agent, num_anchor, device=device, dtype=torch.bool)
        flow_eval_chunks: List[Tensor] = []
        for anchor_offset, raw_step in enumerate(raw_current_steps):
            current_valid = valid[:, raw_step]
            future_valid = valid[:, raw_step + 1 : raw_step + 21].all(dim=1)
            anchor_mask = current_valid & future_valid
            flow_eval_mask[:, anchor_offset] = anchor_mask
            if not anchor_mask.any():
                continue

            flow_eval_chunks.append(
                self._build_anchor_clean_norm(
                    pos=pos,
                    heading=heading,
                    current_pos=pos[:, raw_step],
                    current_head=heading[:, raw_step],
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
        """한 anchor에서 실제로 쓰는 에이전트만 골라 목표를 만듭니다.

        Args:
            pos: 전처리된 중심점입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
            heading: 전처리된 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.
            current_pos: 현재 coarse anchor 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: 현재 coarse anchor 방향입니다. shape은 ``[n_agent]`` 입니다.
            anchor_mask: 이번 anchor를 실제로 학습 또는 평가에 쓰는지 나타냅니다.
                shape은 ``[n_agent]`` 입니다.
            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.

        Returns:
            Tensor:
                정규화된 2초 미래 목표입니다. shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
                마지막 차원은 ``[x, y, cos, sin]`` 순서입니다.
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
            pos: 전처리된 중심점입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
            heading: 전처리된 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.
            valid: 각 시점 유효 여부입니다. shape은 ``[n_agent, n_step]`` 입니다.
            current_pos: 현재 coarse anchor 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: 현재 coarse anchor 방향입니다. shape은 ``[n_agent]`` 입니다.
            anchor_mask: 이번 anchor를 실제로 쓰는 에이전트입니다. shape은 ``[n_agent]`` 입니다.
            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.

        Returns:
            Tuple[Tensor, Tensor]:
                직전 제어 ``[v_x^b, v_y^b, omega]`` 와 유효 마스크입니다.
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

    def _build_anchor_exec_history(
        self,
        pos: Tensor,
        heading: Tensor,
        valid: Tensor,
        anchor_mask: Tensor,
        raw_step: int,
        history_steps: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """anchor 현재 시점까지의 실제 fine history를 길이 6으로 만듭니다.

        Args:
            pos: 전처리된 중심점입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
            heading: 전처리된 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.
            valid: 각 시점 유효 여부입니다. shape은 ``[n_agent, n_step]`` 입니다.
            anchor_mask: 이번 anchor를 실제로 쓰는 에이전트입니다. shape은 ``[n_agent]`` 입니다.
            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.
            history_steps: 만들 history 길이입니다.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                현재 시점을 포함한 실제 fine history 위치, 방향, valid 입니다.
                shape은 각각 ``[n_valid_anchor, history_steps, 2]``,
                ``[n_valid_anchor, history_steps]``, ``[n_valid_anchor, history_steps]`` 입니다.
        """
        num_valid_anchor = int(anchor_mask.sum().item())
        if num_valid_anchor == 0:
            return (
                pos.new_zeros((0, history_steps, 2)),
                heading.new_zeros((0, history_steps)),
                torch.zeros((0, history_steps), device=valid.device, dtype=torch.bool),
            )

        history_start = max(0, raw_step - history_steps + 1)
        pos_history = pos[anchor_mask, history_start : raw_step + 1].clone()
        head_history = heading[anchor_mask, history_start : raw_step + 1].clone()
        valid_history = valid[anchor_mask, history_start : raw_step + 1].clone()
        if pos_history.shape[1] == history_steps:
            return pos_history, head_history, valid_history

        pad_len = history_steps - pos_history.shape[1]
        pad_pos = pos_history[:, :1].expand(-1, pad_len, -1)
        pad_head = head_history[:, :1].expand(-1, pad_len)
        pad_valid = torch.zeros((num_valid_anchor, pad_len), device=valid.device, dtype=torch.bool)
        return (
            torch.cat([pad_pos, pos_history], dim=1),
            torch.cat([pad_head, head_history], dim=1),
            torch.cat([pad_valid, valid_history], dim=1),
        )

    def _concat_flow_chunks(
        self,
        chunks: List[Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """빈 경우까지 포함해서 flow 목표 조각을 하나로 합칩니다.

        Args:
            chunks: 각 anchor에서 만든 목표 조각 목록입니다.
                각 원소 shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
            dtype: 반환 텐서 자료형입니다.
            device: 반환 텐서 장치입니다.

        Returns:
            Tensor:
                이어 붙인 목표입니다. shape은 ``[n_total_valid_anchor, 20, 4]`` 입니다.
                유효한 anchor가 없으면 ``[0, 20, 4]`` 빈 텐서를 돌려줍니다.
        """
        if len(chunks) == 0:
            return torch.zeros((0, 20, 4), device=device, dtype=dtype)
        return torch.cat(chunks, dim=0)

    def _concat_vector_chunks(
        self,
        chunks: List[Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """1차원 조각 목록을 하나의 벡터로 잇습니다.

        Args:
            chunks: 각 조각은 ``[n_valid_anchor]`` 입니다.
            dtype: 반환 텐서 자료형입니다.
            device: 반환 텐서 장치입니다.

        Returns:
            Tensor:
                이어 붙인 벡터입니다. shape은 ``[n_total_valid_anchor]`` 입니다.
        """
        if len(chunks) == 0:
            return torch.zeros((0,), device=device, dtype=dtype)
        return torch.cat([chunk.to(device=device, dtype=dtype) for chunk in chunks], dim=0)

    def _concat_matrix_chunks(
        self,
        chunks: List[Tensor],
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """2차원 조각 목록을 하나의 행렬로 잇습니다.

        Args:
            chunks: 각 조각은 ``[n_valid_anchor, width]`` 입니다.
            width: 마지막 축 너비입니다.
            dtype: 반환 텐서 자료형입니다.
            device: 반환 텐서 장치입니다.

        Returns:
            Tensor:
                이어 붙인 행렬입니다. shape은 ``[n_total_valid_anchor, width]`` 입니다.
        """
        if len(chunks) == 0:
            return torch.zeros((0, width), device=device, dtype=dtype)
        return torch.cat([chunk.to(device=device, dtype=dtype) for chunk in chunks], dim=0)

    def _concat_rank3_chunks(
        self,
        chunks: List[Tensor],
        dim1: int,
        dim2: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """3차원 조각 목록을 하나의 텐서로 잇습니다.

        Args:
            chunks: 각 조각은 ``[n_valid_anchor, dim1, dim2]`` 입니다.
            dim1: 두 번째 축 길이입니다.
            dim2: 마지막 축 길이입니다.
            dtype: 반환 텐서 자료형입니다.
            device: 반환 텐서 장치입니다.

        Returns:
            Tensor:
                이어 붙인 3차원 텐서입니다.
                shape은 ``[n_total_valid_anchor, dim1, dim2]`` 입니다.
        """
        if len(chunks) == 0:
            return torch.zeros((0, dim1, dim2), device=device, dtype=dtype)
        return torch.cat([chunk.to(device=device, dtype=dtype) for chunk in chunks], dim=0)

    def _wrap_angle(self, angle: Tensor) -> Tensor:
        """각도를 ``[-pi, pi]`` 범위로 접습니다.

        Args:
            angle: 각도 텐서입니다. shape은 임의입니다.

        Returns:
            Tensor: 같은 shape의 접힌 각도입니다.
        """
        return torch.atan2(angle.sin(), angle.cos())
