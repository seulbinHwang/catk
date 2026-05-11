from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.smart.tokens.token_processor_v2 import TokenProcessor
from src.smart.utils import transform_to_local, validate_flow_window_steps


class FlowTokenProcessor(TokenProcessor):
    """Flow 학습용 목표와 DRaFT용 보조 메타데이터를 만듭니다."""

    def __init__(
        self,
        map_token_file: str,
        agent_token_file: str,
        map_token_sampling,
        agent_token_sampling,
        flow_window_steps: int = 20,
    ) -> None:
        super().__init__(
            map_token_file=map_token_file,
            agent_token_file=agent_token_file,
            map_token_sampling=map_token_sampling,
            agent_token_sampling=agent_token_sampling,
        )
        self.flow_window_steps = validate_flow_window_steps(
            flow_window_steps=flow_window_steps,
            commit_steps=self.shift,
        )

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
            agent_length = self._get_agent_box_length(tokenized_agent)
            flow_train_mask = torch.zeros(num_agent, num_anchor, device=device, dtype=torch.bool)
            flow_train_chunks: List[Tensor] = []
            flow_train_loss_mask_chunks: List[Tensor] = []
            flow_train_agent_type_chunks: List[Tensor] = []
            flow_train_agent_length_chunks: List[Tensor] = []
            flow_train_prev_control_chunks: List[Tensor] = []
            flow_train_prev_control_valid_chunks: List[Tensor] = []

            for anchor_offset, raw_step in enumerate(raw_current_steps):
                current_valid = valid[:, raw_step]
                future_loss_mask = self._build_anchor_future_loss_mask(valid=valid, raw_step=raw_step)
                anchor_mask = current_valid & future_loss_mask.any(dim=1)
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
                        future_loss_mask=future_loss_mask[train_anchor_mask],
                    )
                )
                flow_train_loss_mask_chunks.append(future_loss_mask[train_anchor_mask])
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
                flow_train_agent_length_chunks.append(agent_length[train_anchor_mask])
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
                    "flow_train_loss_mask": self._concat_mask_chunks(
                        chunks=flow_train_loss_mask_chunks,
                        device=device,
                    ),
                    "flow_train_agent_type": self._concat_vector_chunks(
                        chunks=flow_train_agent_type_chunks,
                        dtype=tokenized_agent["type"].dtype,
                        device=device,
                    ),
                    "flow_train_agent_length": self._concat_vector_chunks(
                        chunks=flow_train_agent_length_chunks,
                        dtype=dtype,
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
            future_valid = self._build_anchor_future_valid(valid=valid, raw_step=raw_step)
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

    def _build_anchor_future_valid(self, valid: Tensor, raw_step: int) -> Tensor:
        future_loss_mask = self._build_anchor_future_loss_mask(valid=valid, raw_step=raw_step)
        return future_loss_mask.all(dim=1)

    def _build_anchor_future_loss_mask(self, valid: Tensor, raw_step: int) -> Tensor:
        """현재 anchor 뒤 전체 flow window가 유효한 경우에만 미래 mask를 만듭니다.

        Args:
            valid: 각 agent와 시점의 유효 여부입니다.
                shape은 ``[n_agent, n_step]`` 입니다.
            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.

        Returns:
            Tensor:
                미래 step별 loss 사용 여부입니다.
                shape은 ``[n_agent, flow_window_steps]`` 입니다.
                전체 미래 window가 유효한 agent만 모든 step이 ``True`` 입니다.
        """
        future_start = raw_step + 1
        # future_mask: [n_agent, flow_window_steps]
        future_mask = torch.zeros(
            (valid.shape[0], self.flow_window_steps),
            device=valid.device,
            dtype=torch.bool,
        )
        available_len = min(self.flow_window_steps, max(0, valid.shape[1] - future_start))
        if available_len <= 0:
            return future_mask

        # available_future_valid: [n_agent, available_len]
        available_future_valid = valid[:, future_start : future_start + available_len].bool()
        if available_len != self.flow_window_steps:
            return future_mask

        full_future_valid = available_future_valid.all(dim=1)
        future_mask[full_future_valid] = True
        return future_mask

    def _build_anchor_clean_norm(
        self,
        pos: Tensor,
        heading: Tensor,
        current_pos: Tensor,
        current_head: Tensor,
        anchor_mask: Tensor,
        raw_step: int,
        future_loss_mask: Tensor | None = None,
    ) -> Tensor:
        """한 anchor에서 실제로 쓰는 agent만 골라 미래 목표를 만듭니다.

        Args:
            pos: 전처리된 중심점입니다. shape은 ``[n_agent, n_step, 2]`` 입니다.
            heading: 전처리된 방향입니다. shape은 ``[n_agent, n_step]`` 입니다.
            current_pos: 현재 coarse anchor 중심점입니다. shape은 ``[n_agent, 2]`` 입니다.
            current_head: 현재 coarse anchor 방향입니다. shape은 ``[n_agent]`` 입니다.
            anchor_mask: 이번 anchor를 실제로 학습 또는 평가에 쓰는지 나타냅니다.
                shape은 ``[n_agent]`` 입니다.
            raw_step: 현재 coarse anchor가 가리키는 10Hz 시점 번호입니다.
            future_loss_mask: loss에 포함할 미래 step입니다.
                shape은 ``[n_valid_anchor, flow_window_steps]`` 입니다.
                값이 없으면 전체 window를 모두 사용합니다.

        Returns:
            Tensor:
                정규화된 미래 목표입니다.
                shape은 ``[n_valid_anchor, flow_window_steps, 4]`` 입니다.
                마지막 차원은 ``[x, y, cos, sin]`` 순서입니다.
        """
        num_valid_anchor = int(anchor_mask.sum().item())
        if num_valid_anchor == 0:
            return pos.new_zeros((0, self.flow_window_steps, 4))

        selected_current_pos = current_pos[anchor_mask]
        selected_current_head = current_head[anchor_mask]
        future_start = raw_step + 1
        future_end = future_start + self.flow_window_steps

        if future_loss_mask is None:
            if future_end > pos.shape[1]:
                raise ValueError(
                    "Requested flow future window exceeds the available sequence length: "
                    f"raw_step={raw_step}, flow_window_steps={self.flow_window_steps}, "
                    f"n_step={pos.shape[1]}."
                )
            # future_pos: [n_valid_anchor, flow_window_steps, 2]
            future_pos = pos[anchor_mask, future_start:future_end]
            # future_head: [n_valid_anchor, flow_window_steps]
            future_head = heading[anchor_mask, future_start:future_end]
        else:
            expected_shape = (num_valid_anchor, self.flow_window_steps)
            if tuple(future_loss_mask.shape) != expected_shape:
                raise ValueError(
                    "future_loss_mask shape must match selected anchors and flow_window_steps: "
                    f"expected={expected_shape}, actual={tuple(future_loss_mask.shape)}."
                )
            future_loss_mask = future_loss_mask.to(device=pos.device, dtype=torch.bool)
            valid_step_count = future_loss_mask.long().sum(dim=1)
            if bool((valid_step_count <= 0).any().item()):
                raise ValueError("future_loss_mask must contain at least one valid future step per anchor.")

            # future_pos: [n_valid_anchor, flow_window_steps, 2]
            future_pos = selected_current_pos.unsqueeze(1).expand(-1, self.flow_window_steps, -1).clone()
            # future_head: [n_valid_anchor, flow_window_steps]
            future_head = selected_current_head.unsqueeze(1).expand(-1, self.flow_window_steps).clone()

            available_len = min(self.flow_window_steps, max(0, pos.shape[1] - future_start))
            if available_len > 0:
                future_pos[:, :available_len] = pos[anchor_mask, future_start : future_start + available_len]
                future_head[:, :available_len] = heading[anchor_mask, future_start : future_start + available_len]

            last_valid_index = valid_step_count - 1
            # last_valid_pos: [n_valid_anchor, 2]
            last_valid_pos = future_pos.gather(
                dim=1,
                index=last_valid_index.view(-1, 1, 1).expand(-1, 1, future_pos.shape[-1]),
            ).squeeze(1)
            # last_valid_head: [n_valid_anchor]
            last_valid_head = future_head.gather(
                dim=1,
                index=last_valid_index.view(-1, 1),
            ).squeeze(1)
            invalid_future_mask = ~future_loss_mask
            future_pos = torch.where(
                invalid_future_mask.unsqueeze(-1),
                last_valid_pos.unsqueeze(1),
                future_pos,
            )
            future_head = torch.where(
                invalid_future_mask,
                last_valid_head.unsqueeze(1),
                future_head,
            )

        future_pos_local, future_head_local = transform_to_local(
            pos_global=future_pos,
            head_global=future_head,
            pos_now=selected_current_pos,
            head_now=selected_current_head,
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

    def _get_agent_box_length(self, tokenized_agent: Dict[str, Tensor]) -> Tensor:
        """DRaFT inverse feasibility에 쓸 box length를 고릅니다.

        raw data의 shape 순서가 ``[length, width]`` 인지 ``[width, length]`` 인지
        전처리 경로에 따라 달라질 수 있어서,
        첫 두 축 가운데 더 큰 값을 length로 사용합니다.
        raw shape가 없으면 토큰화용 고정 ``[width, length]`` 값을 대신 사용합니다.

        Args:
            tokenized_agent: 에이전트 토큰 사전입니다.
                ``shape`` 는 보통 ``[n_agent, 2]`` 또는 ``[n_agent, 3]`` 이고,
                ``token_agent_shape`` 는 ``[n_agent, 2]`` 입니다.

        Returns:
            Tensor:
                anchor 전 공통으로 쓸 box length입니다. shape은 ``[n_agent]`` 입니다.
        """
        if "shape" in tokenized_agent:
            shape = tokenized_agent["shape"].to(dtype=torch.float32)
            if shape.dim() >= 2 and shape.shape[-1] >= 2:
                return shape[..., :2].amax(dim=-1)

        token_shape = tokenized_agent["token_agent_shape"].to(dtype=torch.float32)
        return token_shape[..., 1]

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
            return torch.zeros((0, self.flow_window_steps, 4), device=device, dtype=dtype)
        return torch.cat(chunks, dim=0)

    def _concat_mask_chunks(
        self,
        chunks: List[Tensor],
        device: torch.device,
    ) -> Tensor:
        """미래 step별 loss mask 조각을 하나로 잇습니다.

        Args:
            chunks: 각 anchor에서 고른 mask 조각 목록입니다.
                각 원소 shape은 ``[n_valid_anchor, flow_window_steps]`` 입니다.
            device: 반환 텐서 장치입니다.

        Returns:
            Tensor:
                이어 붙인 mask입니다.
                shape은 ``[n_total_valid_anchor, flow_window_steps]`` 입니다.
        """
        if len(chunks) == 0:
            return torch.zeros((0, self.flow_window_steps), device=device, dtype=torch.bool)
        return torch.cat([chunk.to(device=device, dtype=torch.bool) for chunk in chunks], dim=0)

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

    def _wrap_angle(self, angle: Tensor) -> Tensor:
        """각도를 ``[-pi, pi]`` 범위로 접습니다.

        Args:
            angle: 각도 텐서입니다. shape은 임의입니다.

        Returns:
            Tensor: 같은 shape의 접힌 각도입니다.
        """
        return torch.atan2(angle.sin(), angle.cos())
