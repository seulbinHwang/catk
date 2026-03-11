# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from .agent_decoder import SMARTAgentDecoder
from .map_decoder import SMARTMapDecoder


class SMARTDecoder(nn.Module):
    """RoadNet + Flow-based Agent decoder wrapper."""

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
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
        history_steps: int = 6,
        future_window_steps: int = 20,
        future_num_segments: int = 4,
        future_segment_points: int = 6,
        ode_steps: int = 4,
        hist2f_radius: Optional[float] = None,
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
        self.agent_encoder = SMARTAgentDecoder(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
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
            history_steps=history_steps,
            future_window_steps=future_window_steps,
            future_num_segments=future_num_segments,
            future_segment_points=future_segment_points,
            ode_steps=ode_steps,
            hist2f_radius=hist2f_radius,
        )

    def encode_map(self, tokenized_map: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """도로 token을 한 번만 인코딩한다."""
        return self.map_encoder(tokenized_map)

    def forward(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        agent_raw: Dict[str, Tensor],
        anchor_step: int,
    ) -> Dict[str, Tensor]:
        map_feature = self.encode_map(tokenized_map)
        return self.agent_encoder(tokenized_agent, map_feature, agent_raw, anchor_step)

    def forward_from_map(
        self,
        map_feature: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        agent_raw: Dict[str, Tensor],
        anchor_step: int,
    ) -> Dict[str, Tensor]:
        """이미 계산된 map feature를 재사용한다."""
        return self.agent_encoder(tokenized_agent, map_feature, agent_raw, anchor_step)

    def forward_anchor_batch_from_map(
        self,
        map_feature: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        agent_raw: Dict[str, Tensor],
        anchor_steps: Sequence[int] | Tensor,
        return_full_outputs: bool = True,
    ) -> Dict[str, Tensor]:
        """이미 계산된 map feature로 여러 anchor를 한 번에 처리한다.

        Args:
            map_feature: 인코딩된 map dict.
            tokenized_agent: tokenized agent dict.
            agent_raw: raw ``data['agent']`` dict.
            anchor_steps: 길이 ``K`` 인 raw 10Hz anchor step 목록.
            return_full_outputs: ``True`` 이면 validation/분석용 auxiliary 출력까지
                모두 반환하고, ``False`` 이면 train loss에 필요한 텐서만 반환한다.

        Returns:
            각 텐서가 ``[K, N, ...]`` shape인 batched prediction dict.
        """
        return self.agent_encoder.forward_anchor_batch(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            agent_raw=agent_raw,
            anchor_steps=anchor_steps,
            return_full_outputs=return_full_outputs,
        )

    def closed_loop_train(
        self,
        map_feature: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        agent_raw: Dict[str, Tensor],
        unroll_steps: int,
    ) -> list[Dict[str, Tensor]]:
        """짧은 closed-loop fine-tuning을 수행한다."""
        return self.agent_encoder.closed_loop_train(tokenized_agent, map_feature, agent_raw, unroll_steps)

    def inference(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        agent_raw: Dict[str, Tensor],
        sampling_scheme: DictConfig,
    ) -> Dict[str, Tensor]:
        map_feature = self.encode_map(tokenized_map)
        return self.agent_encoder.inference(tokenized_agent, map_feature, agent_raw, sampling_scheme)
