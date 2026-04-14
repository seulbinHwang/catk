from __future__ import annotations

from typing import Dict, Optional

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from .flow_agent_decoder import SMARTFlowAgentDecoder
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
        )

    def encode_map(self, tokenized_map: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.map_encoder(tokenized_map)

    def encode_anchor_context_from_map_feature(
        self,
        map_feature: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        anchor_mask_key: str = "flow_eval_mask",
    ) -> tuple[Tensor, Tensor, Tensor]:
        """한 번 인코딩한 지도 특징에서 anchor 문맥만 뽑습니다.

        Args:
            map_feature: 지도 인코더 출력입니다.
            tokenized_agent: agent 토큰 사전입니다.
            anchor_mask_key: 어떤 anchor 마스크를 쓸지 나타내는 키입니다.

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                - ``ctx_hidden_pack``: context encoder 전체 출력입니다.
                  shape은 ``[n_agent, 14, hidden_dim]`` 입니다.
                - ``anchor_hidden``: 13개 anchor 문맥입니다.
                  shape은 ``[n_agent, 13, hidden_dim]`` 입니다.
                - ``anchor_hidden_valid``: 유효 anchor만 모은 문맥입니다.
                  shape은 ``[n_valid_anchor, hidden_dim]`` 입니다.
        """
        return self.agent_encoder.encode_anchor_context(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            anchor_mask=tokenized_agent[anchor_mask_key],
        )

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
        return self.agent_encoder(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            anchor_mask=tokenized_agent[anchor_mask_key],
            flow_clean_norm=tokenized_agent[flow_clean_norm_key],
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
        sampling_noise: DictConfig,
        sampling_seed: int | None = None,
        scenario_sampling_seeds: Tensor | None = None,
        max_steps: int | None = None,
        warm_coarse_steps: int = 0,
    ) -> Dict[str, Tensor]:
        return self.agent_encoder.rollout_from_cache(
            rollout_cache=rollout_cache,
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            sampling_noise=sampling_noise,
            sampling_seed=sampling_seed,
            scenario_sampling_seeds=scenario_sampling_seeds,
            max_steps=max_steps,
            warm_coarse_steps=warm_coarse_steps,
        )

    def rollout_from_cache_no_grad(
        self,
        rollout_cache: Dict[str, object],
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        sampling_noise: DictConfig,
        sampling_seed: int | None = None,
        scenario_sampling_seeds: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        return self.agent_encoder.rollout_from_cache_no_grad(
            rollout_cache=rollout_cache,
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            sampling_noise=sampling_noise,
            sampling_seed=sampling_seed,
            scenario_sampling_seeds=scenario_sampling_seeds,
        )

    def sample_open_loop_future(
        self,
        anchor_hidden: Tensor,
        anchor_mask: Tensor,
        sampling_noise: DictConfig,
        sampling_seed: int | None = None,
        agent_type: Tensor | None = None,
        v_init: Tensor | None = None,
        delta_init: Tensor | None = None,
        current_control: Tensor | None = None,
        current_control_valid: Tensor | None = None,
    ) -> Tensor:
        """고정된 문맥에서 실제 생성 경로로 2초 미래를 만듭니다.

        Args:
            anchor_hidden: 모든 anchor 문맥입니다.
                shape은 ``[n_agent, 13, hidden_dim]`` 입니다.
            anchor_mask: 실제로 평가할 anchor 여부입니다.
                shape은 ``[n_agent, 13]`` 입니다.
            sampling_noise: 평가 시 샘플링 초기 잡음 설정입니다.
            sampling_seed: 평가마다 같은 샘플을 만들기 위한 고정 seed입니다.
            agent_type: 유효 anchor의 agent type입니다. shape ``[n_valid_anchor]``.
            current_control: 유효 anchor의 직전 body-frame control입니다.
                shape은 ``[n_valid_anchor, 3]`` 입니다.
            current_control_valid: 위 control의 신뢰도 마스크입니다.
                shape은 ``[n_valid_anchor]`` 입니다.

        Returns:
            Tensor: 생성된 정규화 2초 미래입니다.
                shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
        """
        return self.agent_encoder.sample_open_loop_future(
            anchor_hidden=anchor_hidden,
            anchor_mask=anchor_mask,
            sampling_noise=sampling_noise,
            sampling_seed=sampling_seed,
            agent_type=agent_type,
            v_init=v_init,
            delta_init=delta_init,
            current_control=current_control,
            current_control_valid=current_control_valid,
        )

    def inference(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        sampling_noise: DictConfig,
    ) -> Dict[str, Tensor]:
        map_feature = self.encode_map(tokenized_map)
        rollout_cache = self.prepare_inference_cache(tokenized_agent, map_feature)
        return self.rollout_from_cache(
            rollout_cache=rollout_cache,
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            sampling_noise=sampling_noise,
        )
