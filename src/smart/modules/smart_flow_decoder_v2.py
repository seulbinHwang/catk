from __future__ import annotations

from typing import Dict, Optional

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from .flow_agent_decoder_v2 import SMARTFlowAgentDecoder
from .map_decoder import SMARTMapDecoder


class SMARTFlowDecoder(nn.Module):

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
        flow_dim: int,
        flow_num_chunk_heads: int,
        flow_num_chunk_layers: int,
        flow_solver_steps: int,
        flow_solver_method: str,
        flow_solver_eps: float,
        closed_loop_rollout_mode: str = "raw_fm",
        flow_window_steps: int = 20,
        use_lqr: bool = False,
        use_stop_motion: bool = False,
        lqr_commit: DictConfig | None = None,
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
        self.agent_encoder = SMARTFlowAgentDecoder(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            flow_window_steps=flow_window_steps,
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
            flow_dim=flow_dim,
            flow_num_chunk_heads=flow_num_chunk_heads,
            flow_num_chunk_layers=flow_num_chunk_layers,
            flow_solver_steps=flow_solver_steps,
            flow_solver_method=flow_solver_method,
            flow_solver_eps=flow_solver_eps,
            closed_loop_rollout_mode=closed_loop_rollout_mode,
            use_lqr=use_lqr,
            use_stop_motion=use_stop_motion,
            lqr_commit=lqr_commit,
        )

    def encode_map(self, tokenized_map: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.map_encoder(tokenized_map)

    def forward_from_map_feature(
        self,
        map_feature: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        anchor_mask_key: str = "flow_eval_mask",
    ) -> Dict[str, Tensor]:
        flow_clean_norm_key = {
            "flow_train_mask": "flow_train_clean_norm",
            "flow_eval_mask": "flow_eval_clean_norm",
        }[anchor_mask_key]
        flow_loss_mask = (
            tokenized_agent["flow_train_loss_mask"]
            if anchor_mask_key == "flow_train_mask"
            else None
        )
        return self.agent_encoder(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            anchor_mask=tokenized_agent[anchor_mask_key],
            flow_clean_norm=tokenized_agent[flow_clean_norm_key],
            flow_loss_mask=flow_loss_mask,
        )

    def forward(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        anchor_mask_key: str = "flow_eval_mask",
    ) -> Dict[str, Tensor]:
        map_feature = self.encode_map(tokenized_map)
        return self.forward_from_map_feature(
            map_feature=map_feature,
            tokenized_agent=tokenized_agent,
            anchor_mask_key=anchor_mask_key,
        )

    def prepare_inference_cache(
        self,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
    ) -> Dict[str, object]:
        return self.agent_encoder.prepare_inference_cache(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
        )

    def rollout_from_cache(
        self,
        rollout_cache: Dict[str, object],
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        sampling_scheme: DictConfig,
        sampling_seed: int | None = None,
        scenario_sampling_seeds: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        return self.agent_encoder.rollout_from_cache(
            rollout_cache=rollout_cache,
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            sampling_scheme=sampling_scheme,
            sampling_seed=sampling_seed,
            scenario_sampling_seeds=scenario_sampling_seeds,
        )

    def sample_open_loop_future(
        self,
        anchor_hidden: Tensor,
        anchor_mask: Tensor,
        sampling_scheme: DictConfig,
        sampling_seed: int | None = None,
        backprop_last_k: int | None = None,
    ) -> Tensor:
        """고정된 문맥에서 실제 생성 경로로 2초 미래를 만듭니다.

        Args:
            anchor_hidden: 모든 anchor 문맥입니다.
                shape은 ``[n_agent, 13, hidden_dim]`` 입니다.
            anchor_mask: 실제로 평가할 anchor 여부입니다.
                shape은 ``[n_agent, 13]`` 입니다.
            sampling_scheme: 샘플링 단계 수, 방법, 잡음 크기 설정입니다.
            sampling_seed: validation마다 같은 샘플을 만들기 위한 고정 seed입니다.

        Returns:
            Tensor: 생성된 정규화 2초 미래입니다.
                shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
        """
        return self.agent_encoder.sample_open_loop_future(
            anchor_hidden=anchor_hidden,
            anchor_mask=anchor_mask,
            sampling_scheme=sampling_scheme,
            sampling_seed=sampling_seed,
            backprop_last_k=backprop_last_k,
        )

    def inference(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        sampling_scheme: DictConfig,
    ) -> Dict[str, Tensor]:
        map_feature = self.encode_map(tokenized_map)
        rollout_cache = self.prepare_inference_cache(tokenized_agent, map_feature)
        return self.rollout_from_cache(
            rollout_cache=rollout_cache,
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            sampling_scheme=sampling_scheme,
        )
