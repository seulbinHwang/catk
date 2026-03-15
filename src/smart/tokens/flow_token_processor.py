from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import transform_to_local


class FlowTokenProcessor(TokenProcessor):
    """Extends the original token processor with 14-slot context packs and
    packed flow-matching targets.
    """

    def forward(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
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
            for anchor_offset, raw_step in enumerate(raw_current_steps):
                current_valid = valid[:, raw_step]
                future_valid = valid[:, raw_step + 1 : raw_step + 21].all(dim=1)
                anchor_mask = current_valid & future_valid
                train_anchor_mask = anchor_mask & train_mask
                flow_train_mask[:, anchor_offset] = train_anchor_mask
                if not train_anchor_mask.any():
                    continue

                anchor_clean_norm = self._build_anchor_clean_norm(
                    pos=pos,
                    heading=heading,
                    current_pos=tokenized_agent["sampled_pos"][:, anchor_offset + 1],
                    current_head=tokenized_agent["sampled_heading"][:, anchor_offset + 1],
                    anchor_mask=anchor_mask,
                    raw_step=raw_step,
                )
                train_keep_local = train_mask[anchor_mask]
                flow_train_chunks.append(anchor_clean_norm[train_keep_local])

            tokenized_agent.update(
                {
                    "flow_train_mask": flow_train_mask,
                    "flow_train_clean_norm": self._concat_flow_chunks(
                        chunks=flow_train_chunks,
                        dtype=dtype,
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

            anchor_clean_norm = self._build_anchor_clean_norm(
                pos=pos,
                heading=heading,
                current_pos=tokenized_agent["sampled_pos"][:, anchor_offset + 1],
                current_head=tokenized_agent["sampled_heading"][:, anchor_offset + 1],
                anchor_mask=anchor_mask,
                raw_step=raw_step,
            )
            flow_eval_chunks.append(anchor_clean_norm)

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
