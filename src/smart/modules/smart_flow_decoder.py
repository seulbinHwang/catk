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
        flow_solver_path_type: str,
        flow_solver_sigma_min: float,
        flow_use_residual_velocity_head: bool,
        flow_residual_bottleneck_dim: int | None,
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
            flow_solver_path_type=flow_solver_path_type,
            flow_solver_sigma_min=flow_solver_sigma_min,
            flow_use_residual_velocity_head=flow_use_residual_velocity_head,
            flow_residual_bottleneck_dim=flow_residual_bottleneck_dim,
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
        sampling_noise: DictConfig | None = None,
        sampling_seed: int | None = None,
        scenario_sampling_seeds: Tensor | None = None,
        sampling_scheme: DictConfig | None = None,
    ) -> Dict[str, Tensor]:
        return self.agent_encoder.rollout_from_cache(
            rollout_cache=rollout_cache,
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            sampling_noise=sampling_noise,
            sampling_seed=sampling_seed,
            scenario_sampling_seeds=scenario_sampling_seeds,
            sampling_scheme=sampling_scheme,
        )


    def sample_open_loop_future(
        self,
        anchor_hidden: Tensor,
        anchor_mask: Tensor,
        sampling_noise: DictConfig | None = None,
        sampling_seed: int | None = None,
        sampling_scheme: DictConfig | None = None,
    ) -> Tensor:
        """고정된 문맥에서 실제 생성 경로로 2초 미래를 만듭니다.

        Args:
            anchor_hidden: 모든 anchor 문맥입니다.
                shape은 ``[n_agent, 13, hidden_dim]`` 입니다.
            anchor_mask: 실제로 평가할 anchor 여부입니다.
                shape은 ``[n_agent, 13]`` 입니다.
            sampling_noise: 평가 rollout용 초기 잡음 설정입니다.
            sampling_seed: validation마다 같은 샘플을 만들기 위한 고정 seed입니다.

        Returns:
            Tensor: 생성된 정규화 2초 미래입니다.
                shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
        """
        return self.agent_encoder.sample_open_loop_future(
            anchor_hidden=anchor_hidden,
            anchor_mask=anchor_mask,
            sampling_noise=sampling_noise,
            sampling_seed=sampling_seed,
            sampling_scheme=sampling_scheme,
        )

    def pack_anchor_hidden(
        self,
        anchor_hidden: Tensor,
        anchor_mask: Tensor,
    ) -> Tensor:
        """유효 anchor 문맥만 압축해 돌려줍니다.

        Args:
            anchor_hidden: 모든 anchor 문맥입니다.
                shape은 ``[n_agent, 13, hidden_dim]`` 입니다.
            anchor_mask: 유효 anchor 여부입니다.
                shape은 ``[n_agent, 13]`` 입니다.

        Returns:
            Tensor:
                유효 anchor만 모은 문맥입니다.
                shape은 ``[n_valid_anchor, hidden_dim]`` 입니다.
        """
        return self.agent_encoder.pack_anchor_hidden(anchor_hidden=anchor_hidden, anchor_mask=anchor_mask)

    def sample_open_loop_future_from_hidden(
        self,
        anchor_hidden_valid: Tensor,
        sampling_noise: DictConfig | None = None,
        sampling_seed: int | None = None,
        sampling_scheme: DictConfig | None = None,
    ) -> Tensor:
        """압축된 anchor 문맥에서 바로 2초 미래를 생성합니다.

        Args:
            anchor_hidden_valid: 유효 anchor만 모은 문맥입니다.
                shape은 ``[n_valid_anchor, hidden_dim]`` 입니다.
            sampling_noise: 평가 rollout용 초기 잡음 설정입니다.
            sampling_seed: 같은 seed에서 같은 출발 잡음을 만들기 위한 값입니다.

        Returns:
            Tensor:
                생성된 정규화 2초 미래입니다.
                shape은 ``[n_valid_anchor, 20, 4]`` 입니다.
        """
        return self.agent_encoder.sample_open_loop_future_from_hidden(
            anchor_hidden_valid=anchor_hidden_valid,
            sampling_noise=sampling_noise,
            sampling_seed=sampling_seed,
            sampling_scheme=sampling_scheme,
        )

    def inference(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        sampling_noise: DictConfig | None = None,
        sampling_scheme: DictConfig | None = None,
    ) -> Dict[str, Tensor]:
        map_feature = self.encode_map(tokenized_map)
        rollout_cache = self.prepare_inference_cache(tokenized_agent, map_feature)
        return self.rollout_from_cache(
            rollout_cache=rollout_cache,
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            sampling_noise=sampling_noise,
            sampling_scheme=sampling_scheme,
        )
