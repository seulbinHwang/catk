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
        closed_loop_rollout_mode: str = "raw_fm",
        flow_window_steps: int = 20,
        use_kinematic_control_flow: bool = False,
        use_holonomic_model_only: bool = False,
        use_rolling_supervision: bool = True,
        control_pos_scale_m: float = 1.0,
        control_vehicle_no_slip_point_ratio: float = 0.0,
        control_cyclist_no_slip_point_ratio: float = 0.0,
        control_vehicle_yaw_scale_rad: float | None = None,
        control_pedestrian_yaw_scale_rad: float | None = None,
        control_cyclist_yaw_scale_rad: float | None = None,
        use_lqr: bool = False,
        use_stop_motion: bool = False,
        lqr_commit: DictConfig | None = None,
        detach_train_metric_clean: bool = False,
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
            use_kinematic_control_flow=use_kinematic_control_flow,
            control_pos_scale_m=control_pos_scale_m,
            control_vehicle_no_slip_point_ratio=control_vehicle_no_slip_point_ratio,
            control_cyclist_no_slip_point_ratio=control_cyclist_no_slip_point_ratio,
            control_vehicle_yaw_scale_rad=control_vehicle_yaw_scale_rad,
            control_pedestrian_yaw_scale_rad=control_pedestrian_yaw_scale_rad,
            control_cyclist_yaw_scale_rad=control_cyclist_yaw_scale_rad,
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
            use_stop_motion=False,
            lqr_commit=lqr_commit,
            use_holonomic_model_only=use_holonomic_model_only,
            use_rolling_supervision=use_rolling_supervision,
            detach_train_metric_clean=detach_train_metric_clean,
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
        flow_clean_metric_norm_key = {
            "flow_train_mask": "flow_train_clean_metric_norm",
            "flow_eval_mask": "flow_eval_clean_metric_norm",
        }[anchor_mask_key]
        flow_loss_mask = (
            tokenized_agent["flow_train_loss_mask"]
            if anchor_mask_key == "flow_train_mask"
            else None
        )
        flow_agent_type_key = {
            "flow_train_mask": "flow_train_agent_type",
            "flow_eval_mask": "flow_eval_agent_type",
        }[anchor_mask_key]
        flow_agent_length_key = {
            "flow_train_mask": "flow_train_agent_length",
            "flow_eval_mask": "flow_eval_agent_length",
        }[anchor_mask_key]
        return self.agent_encoder(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            anchor_mask=tokenized_agent[anchor_mask_key],
            flow_clean_norm=tokenized_agent[flow_clean_norm_key],
            flow_agent_type=tokenized_agent.get(flow_agent_type_key),
            flow_agent_length=tokenized_agent.get(flow_agent_length_key),
            flow_loss_mask=flow_loss_mask,
            flow_clean_metric_norm=tokenized_agent.get(flow_clean_metric_norm_key),
        )

    def build_anchor_context_from_map_feature(
        self,
        map_feature: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        anchor_mask_key: str = "flow_eval_mask",
    ) -> Dict[str, Tensor]:
        flow_clean_norm_key = {
            "flow_train_mask": "flow_train_clean_norm",
            "flow_eval_mask": "flow_eval_clean_norm",
        }[anchor_mask_key]
        flow_clean_metric_norm_key = {
            "flow_train_mask": "flow_train_clean_metric_norm",
            "flow_eval_mask": "flow_eval_clean_metric_norm",
        }[anchor_mask_key]
        flow_loss_mask = (
            tokenized_agent["flow_train_loss_mask"]
            if anchor_mask_key == "flow_train_mask"
            else None
        )
        flow_agent_type_key = {
            "flow_train_mask": "flow_train_agent_type",
            "flow_eval_mask": "flow_eval_agent_type",
        }[anchor_mask_key]
        flow_agent_length_key = {
            "flow_train_mask": "flow_train_agent_length",
            "flow_eval_mask": "flow_eval_agent_length",
        }[anchor_mask_key]
        return self.agent_encoder.build_anchor_context(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            anchor_mask=tokenized_agent[anchor_mask_key],
            flow_clean_norm=tokenized_agent[flow_clean_norm_key],
            flow_agent_type=tokenized_agent.get(flow_agent_type_key),
            flow_agent_length=tokenized_agent.get(flow_agent_length_key),
            flow_loss_mask=flow_loss_mask,
            flow_clean_metric_norm=tokenized_agent.get(flow_clean_metric_norm_key),
        )

    def build_anchor_context(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        anchor_mask_key: str = "flow_eval_mask",
    ) -> Dict[str, Tensor]:
        map_feature = self.encode_map(tokenized_map)
        return self.build_anchor_context_from_map_feature(
            map_feature=map_feature,
            tokenized_agent=tokenized_agent,
            anchor_mask_key=anchor_mask_key,
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

    def prepare_training_rollout_cache(
        self,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
    ) -> Dict[str, object]:
        """self-forced 학습에서 gradient를 유지한 rollout cache를 만듭니다.

        Args:
            tokenized_agent: 평가 모드 기준 agent token 사전입니다.
            map_feature: 현재 decoder가 만든 map feature입니다.

        Returns:
            Dict[str, object]: N초 self-rollout에 사용할 초기 cache입니다.
        """
        return self.agent_encoder.prepare_training_rollout_cache(
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
        scenario_sampling_signs: Tensor | None = None,
        return_flow_2s_preview: bool = False,
        rollout_steps_2hz: int | None = None,
    ) -> Dict[str, Tensor]:
        return self.agent_encoder.rollout_from_cache(
            rollout_cache=rollout_cache,
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            sampling_scheme=sampling_scheme,
            sampling_seed=sampling_seed,
            scenario_sampling_seeds=scenario_sampling_seeds,
            scenario_sampling_signs=scenario_sampling_signs,
            return_flow_2s_preview=return_flow_2s_preview,
            rollout_steps_2hz=rollout_steps_2hz,
        )

    def training_rollout_from_cache(
        self,
        rollout_cache: Dict[str, object],
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        sampling_scheme: DictConfig,
        sampling_seed: int | None = None,
        scenario_sampling_seeds: Tensor | None = None,
        scenario_sampling_signs: Tensor | None = None,
        rollout_steps_2hz: int | None = None,
        self_forced_epoch: int | None = None,
        detach_block_transition: bool = False,
        use_stop_motion: bool | None = None,
    ) -> Dict[str, Tensor]:
        """self-forced 학습에서 gradient를 유지한 closed-loop rollout을 실행합니다.

        Args:
            rollout_cache: ``prepare_training_rollout_cache`` 가 만든 초기 상태입니다.
            tokenized_agent: 평가 모드 기준 agent token 사전입니다.
            map_feature: 현재 decoder가 만든 map feature입니다.
            sampling_scheme: flow sampling 설정입니다.
            sampling_seed: batch 공통 seed입니다.
            scenario_sampling_seeds: scenario별 seed입니다. shape은 ``[n_scenario]`` 입니다.
            rollout_steps_2hz: 실행할 0.5초 block 수입니다. ``None`` 이면 전체 평가 길이를 실행합니다.
            use_stop_motion: self-forced 학습 rollout 전용 stop-motion 사용 여부입니다.

        Returns:
            Dict[str, Tensor]: committed self-rollout 결과입니다.
        """
        return self.agent_encoder.training_rollout_from_cache(
            rollout_cache=rollout_cache,
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            sampling_scheme=sampling_scheme,
            sampling_seed=sampling_seed,
            scenario_sampling_seeds=scenario_sampling_seeds,
            scenario_sampling_signs=scenario_sampling_signs,
            rollout_steps_2hz=rollout_steps_2hz,
            self_forced_epoch=self_forced_epoch,
            detach_block_transition=detach_block_transition,
            use_stop_motion=False,
        )

    def path_flow_velocity_for_anchor0(
        self,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        path_noisy_norm: Tensor,
        tau: Tensor,
        anchor_mask: Tensor,
    ) -> Dict[str, Tensor]:
        """첫 flow anchor의 noisy N초 path에 대한 velocity와 clean estimate를 계산합니다.

        Args:
            tokenized_agent: 평가 모드 기준 agent token 사전입니다.
            map_feature: 이 decoder가 직접 만든 map feature입니다.
            path_noisy_norm: noisy path입니다. shape은 ``[n_valid_agent, flow_window_steps, 4]`` 입니다.
            tau: flow interpolation time입니다. shape은 ``[n_valid_agent]`` 입니다.
            anchor_mask: 첫 anchor에서 사용할 agent mask입니다. shape은 ``[n_agent]`` 입니다.

        Returns:
            Dict[str, Tensor]: ``velocity`` 와 ``clean`` 을 담은 사전입니다.
        """
        return self.agent_encoder.path_flow_velocity_for_anchor0(
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            path_noisy_norm=path_noisy_norm,
            tau=tau,
            anchor_mask=anchor_mask,
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
                shape은 ``[n_agent, n_anchor, hidden_dim]`` 입니다.
            anchor_mask: 실제로 평가할 anchor 여부입니다.
                shape은 ``[n_agent, n_anchor]`` 입니다.
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

    def flow_norm_to_pose_metric_norm(
        self,
        value: Tensor,
        agent_type: Tensor | None,
        agent_length: Tensor | None = None,
    ) -> Tensor:
        return self.agent_encoder.flow_norm_to_pose_metric_norm(
            value=value,
            agent_type=agent_type,
            agent_length=agent_length,
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
