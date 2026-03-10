# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

from typing import Dict, Optional

import torch.nn as nn
from torch import Tensor

from .agent_flow_decoder import SMARTAgentFlowDecoder
from .map_decoder import SMARTMapDecoder


class SMARTDecoder(nn.Module):
    """SMART map encoder와 flow agent decoder를 묶는 얇은 래퍼입니다."""

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        future_window_steps: int,
        anchor_chunk_k: int,
        pl2pl_radius: float,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_map_layers: int,
        num_agent_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        n_token_agent: int,
        history_slots: int = 6,
    ) -> None:
        super().__init__()
        self.map_encoder = SMARTMapDecoder(
            hidden_dim=hidden_dim,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.agent_encoder = SMARTAgentFlowDecoder(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            future_window_steps=future_window_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            hist_drop_prob=hist_drop_prob,
            n_token_agent=n_token_agent,
            history_slots=history_slots,
        )

    def encode_map(self, tokenized_map: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """static map feature를 한 번만 인코딩해 재사용합니다."""
        return self.map_encoder(tokenized_map)

    def forward(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        data,
        anchor_10hz: int,
        sampling_cfg,
        map_feature: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """open-loop anchor 하나를 학습할 때 사용합니다."""
        map_feature = map_feature if map_feature is not None else self.encode_map(tokenized_map)
        return self.agent_encoder(tokenized_agent, map_feature, data, anchor_10hz, sampling_cfg)

    def rollout(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        sampling_cfg,
        data=None,
        rollout_steps: Optional[int] = None,
        return_targets: bool = False,
        map_feature: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """closed-loop rollout을 수행합니다."""
        map_feature = map_feature if map_feature is not None else self.encode_map(tokenized_map)
        return self.agent_encoder.rollout(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            sampling_cfg=sampling_cfg,
            data=data,
            rollout_steps=rollout_steps,
            return_targets=return_targets,
        )

    def inference(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        sampling_cfg,
        data=None,
        rollout_steps: Optional[int] = None,
        map_feature: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """기존 호출과의 호환을 위한 alias입니다."""
        return self.rollout(
            tokenized_map=tokenized_map,
            tokenized_agent=tokenized_agent,
            sampling_cfg=sampling_cfg,
            data=data,
            rollout_steps=rollout_steps,
            return_targets=False,
            map_feature=map_feature,
        )
