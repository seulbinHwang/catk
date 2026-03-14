# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

from __future__ import annotations

import torch

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def set_model_for_finetuning(model: torch.nn.Module, finetune: bool) -> None:
    """Closed-loop fine-tuning용으로 필요한 부분만 학습 가능하게 바꾼다."""

    def _unfreeze(module: torch.nn.Module, name: str) -> None:
        for p in module.parameters():
            p.requires_grad = True
        log.info(f"Unfreezing {name}")

    if not finetune:
        return

    for p in model.parameters():
        p.requires_grad = False

    agent = model.agent_encoder
    _unfreeze(agent.current_anchor_emb, "current_anchor_emb")
    _unfreeze(agent.future_segment_emb, "future_segment_emb")
    _unfreeze(agent.segment_out_head, "segment_out_head")
    _unfreeze(agent.flow_time_emb, "flow_time_emb")
    _unfreeze(agent.hist2f_attn_layers, "hist2f_attn_layers")
    _unfreeze(agent.t_attn_layers, "t_attn_layers")
    _unfreeze(agent.pt2a_attn_layers, "pt2a_attn_layers")
    _unfreeze(agent.a2a_attn_layers, "a2a_attn_layers")
    _unfreeze(agent.r_t_emb, "r_t_emb")
    _unfreeze(agent.r_pt2a_emb, "r_pt2a_emb")
    _unfreeze(agent.r_a2a_emb, "r_a2a_emb")
    _unfreeze(agent.r_hist2f_emb, "r_hist2f_emb")
