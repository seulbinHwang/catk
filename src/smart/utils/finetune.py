# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import torch

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)



def set_model_for_finetuning(model: torch.nn.Module, finetune: bool) -> None:
    """선택적 short closed-loop fine-tuning 때 학습할 모듈만 풉니다."""

    def _unfreeze(module: torch.nn.Module, name: str) -> None:
        for p in module.parameters():
            p.requires_grad = True
        log.info(f"Unfreezing {name}")

    if not finetune:
        return

    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model.agent_encoder, "current_anchor_emb"):
        _unfreeze(model.agent_encoder.current_anchor_emb, "current_anchor_emb")
    if hasattr(model.agent_encoder, "future_segment_emb"):
        _unfreeze(model.agent_encoder.future_segment_emb, "future_segment_emb")
    if hasattr(model.agent_encoder, "flow_time_emb"):
        _unfreeze(model.agent_encoder.flow_time_emb, "flow_time_emb")
    if hasattr(model.agent_encoder, "segment_index_emb"):
        _unfreeze(model.agent_encoder.segment_index_emb, "segment_index_emb")
    if hasattr(model.agent_encoder, "segment_out_head"):
        _unfreeze(model.agent_encoder.segment_out_head, "segment_out_head")
    if hasattr(model.agent_encoder, "context_fusion"):
        _unfreeze(model.agent_encoder.context_fusion, "context_fusion")
    if hasattr(model.agent_encoder, "t_attn_layers"):
        _unfreeze(model.agent_encoder.t_attn_layers, "t_attn_layers")
    if hasattr(model.agent_encoder, "pt2a_attn_layers"):
        _unfreeze(model.agent_encoder.pt2a_attn_layers, "pt2a_attn_layers")
    if hasattr(model.agent_encoder, "a2a_attn_layers"):
        _unfreeze(model.agent_encoder.a2a_attn_layers, "a2a_attn_layers")