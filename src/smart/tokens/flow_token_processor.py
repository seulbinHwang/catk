from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import transform_to_local


class FlowTokenProcessor(TokenProcessor):
    """Extends the original token processor with 14-slot context packs and
    13-anchor flow-matching targets.
    """

    def forward(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        tokenized_map, tokenized_agent = super().forward(data)
        tokenized_agent = self._build_flow_targets(data, tokenized_agent)
        return tokenized_map, tokenized_agent

    def _build_flow_targets(
        self,
        data: HeteroData,
        tokenized_agent: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        valid = data["agent"]["valid_mask"].clone()
        heading = data["agent"]["heading"].clone()
        pos = data["agent"]["position"][..., :2].contiguous().clone()
        vel = data["agent"]["velocity"].clone()

        heading = self._clean_heading(valid, heading)
        valid, pos, heading, vel = self._extrapolate_agent_to_prev_token_step(
            valid,
            pos,
            heading,
            vel,
        )

        ctx_sampled_idx = tokenized_agent["sampled_idx"][:, :14].contiguous()
        ctx_sampled_pos = tokenized_agent["sampled_pos"][:, :14].contiguous()
        ctx_sampled_heading = tokenized_agent["sampled_heading"][:, :14].contiguous()
        ctx_valid = tokenized_agent["valid_mask"][:, :14].contiguous()

        num_agent = pos.shape[0]
        device = pos.device
        dtype = pos.dtype

        flow_clean_norm = torch.zeros(num_agent, 13, 20, 4, device=device, dtype=dtype)
        flow_anchor_mask = torch.zeros(num_agent, 13, device=device, dtype=torch.bool)

        # Current anchors are {10, 15, ..., 70} in raw 10 Hz time.
        raw_current_steps = list(range(10, 71, self.shift))
        for anchor_offset, raw_step in enumerate(raw_current_steps):
            current_valid = valid[:, raw_step]
            future_valid = valid[:, raw_step + 1 : raw_step + 21].all(dim=1)
            anchor_mask = current_valid & future_valid
            flow_anchor_mask[:, anchor_offset] = anchor_mask

            current_pos = tokenized_agent["sampled_pos"][:, anchor_offset + 1]
            current_head = tokenized_agent["sampled_heading"][:, anchor_offset + 1]
            future_pos = pos[:, raw_step + 1 : raw_step + 21]
            future_head = heading[:, raw_step + 1 : raw_step + 21]
            future_pos_local, future_head_local = transform_to_local(
                pos_global=future_pos,
                head_global=future_head,
                pos_now=current_pos,
                head_now=current_head,
            )
            flow_clean_norm[:, anchor_offset, :, 0] = future_pos_local[..., 0] / 20.0
            flow_clean_norm[:, anchor_offset, :, 1] = future_pos_local[..., 1] / 20.0
            flow_clean_norm[:, anchor_offset, :, 2] = future_head_local.cos()
            flow_clean_norm[:, anchor_offset, :, 3] = future_head_local.sin()

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
                "flow_clean_norm": flow_clean_norm,
                "flow_anchor_mask": flow_anchor_mask,
                "flow_train_mask": flow_anchor_mask & train_mask.unsqueeze(1),
                "flow_eval_mask": flow_anchor_mask,
            }
        )
        return tokenized_agent
