from __future__ import annotations

from typing import Dict, Optional

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from .agent_decoder import SMARTAgentDecoder
from .map_decoder import SMARTMapDecoder


class SMARTDecoder(nn.Module):
    """맵 인코더와 agent flow decoder를 묶는 얇은 래퍼이다."""

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
        flow_num_future_steps: int,
        flow_num_anchors: int,
        flow_anchor_stride: int,
        commit_num_future_steps: int,
        flow_tau_eps: float,
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
            flow_num_future_steps=flow_num_future_steps,
            flow_num_anchors=flow_num_anchors,
            flow_anchor_stride=flow_anchor_stride,
            commit_num_future_steps=commit_num_future_steps,
            flow_tau_eps=flow_tau_eps,
        )

    def forward(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """학습용 forward이다."""
        map_feature = self.map_encoder(tokenized_map)
        return self.agent_encoder(tokenized_agent, map_feature)

    def inference(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        sampling_scheme: DictConfig,
    ) -> Dict[str, Tensor]:
        """flow matching 기반 closed-loop inference이다."""
        map_feature = self.map_encoder(tokenized_map)
        return self.agent_encoder.inference(tokenized_agent, map_feature, sampling_scheme)
