from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.smart.modules.flow_adjoint_matching import SmoothControlProjector
from src.smart.utils import angle_between_2d_vectors


@dataclass
class TerminalCostFinalStepResult:
    loss: Tensor
    terminal_cost: Tensor
    projection_gap: Tensor
    final_count: Tensor
    flow_reg_loss: Tensor = None


class TerminalCostFinalStepLoss(nn.Module):
    """
    Feasibility 기반 projector terminal_cost를 연속값 reward(최소화 loss)로 쓰고,
    memoryless Euler-Maruyama rollout에서 마지막 diffusion step에서만 gradient가 흐르도록 합니다.
    """

    def __init__(
        self,
        rollout_steps: int = 4,
        rollout_noise_scale: float = 1.0,
        feasible_weight: float = 1.0,
        smooth_deadzone_epsilon: Tuple[float, float, float] = (0.01, 0.01, 0.01),
        smooth_deadzone_tau: float = 0.002,
        flow_reg_lambda: float = 0.0,
    ) -> None:
        super().__init__()
        self.rollout_steps = int(rollout_steps)
        self.rollout_noise_scale = float(rollout_noise_scale)
        self.flow_reg_lambda = float(flow_reg_lambda)
        self.projector = SmoothControlProjector(
            feasible_weight=feasible_weight,
            smooth_deadzone_epsilon=smooth_deadzone_epsilon,
            smooth_deadzone_tau=smooth_deadzone_tau,
        )

    def _build_step_times(
        self,
        flow_ode: nn.Module,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[Tensor]:
        """
        t0=tau_start=flow_ode.eps로부터 rollout_steps 구간을 나눕니다.
        """
        t0 = float(flow_ode.eps)
        dt = (1.0 - t0) / float(self.rollout_steps)
        return [
            torch.full((batch_size,), t0 + step_idx * dt, device=device, dtype=dtype)
            for step_idx in range(self.rollout_steps)
        ]

    def _rollout_memoryless_sde_last_step_grad(
        self,
        *,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        x_init_norm: Optional[Tensor] = None,
    ) -> Tuple[Tensor, list[Tensor]]:
        """
        memoryless Euler-Maruyama SDE 롤아웃을 만들되,
        중간 step은 `detach`해서 마지막 step에서만 autograd graph를 유지합니다.
        """
        batch_size = int(anchor_hidden_valid.shape[0])
        device = anchor_hidden_valid.device
        dtype = anchor_hidden_valid.dtype

        if x_init_norm is None:
            x_init_norm = torch.randn(
                batch_size,
                20,
                4,
                device=device,
                dtype=dtype,
            ) * self.rollout_noise_scale
        else:
            if x_init_norm.dim() != 3 or tuple(x_init_norm.shape[-2:]) != (20, 4):
                raise ValueError(
                    f"x_init_norm must have shape [N,20,4], got {tuple(x_init_norm.shape)}"
                )
            x_init_norm = x_init_norm.to(device=device, dtype=dtype)

        dt = (1.0 - float(flow_ode.eps)) / float(self.rollout_steps)
        times = self._build_step_times(
            flow_ode=flow_ode,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        current_state = x_init_norm
        for step_idx in range(self.rollout_steps):
            tau = times[step_idx]

            # DDP unused-parameter를 피하기 위해 매 step에서 파라미터가 그래프에 연결되도록
            # forward는 항상 autograd를 켠 채로 수행합니다.
            # 그래프 폭증은 아래에서 state detach로 제어합니다.
            velocity_dict = flow_decoder.forward_components(
                anchor_hidden=anchor_hidden_valid,
                x_t_norm=current_state,
                tau=tau,
            )
            drift = flow_ode.drift_from_velocity(
                x_t=current_state,
                velocity=velocity_dict["velocity"],
                tau=tau,
            )

            noise = torch.randn_like(current_state)
            sigma = flow_ode.memoryless_sigma(tau).view(-1, 1, 1)

            next_state = current_state + dt * drift + (dt**0.5) * sigma * noise

            if step_idx < self.rollout_steps - 1:
                current_state = next_state.detach()
            else:
                current_state = next_state

        self._assert_finite_tensor("final_state", current_state)
        return current_state, times

    def _rollout_ode_last_step_grad(
        self,
        *,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """결정론적 ODE rollout. 마지막 step에서만 autograd graph를 유지합니다.

        SDE 노이즈 없이 순수 drift로 적분하므로 trajectory가 안정적입니다.
        gradient는 마지막 step의 forward_components를 통해서만 전파됩니다.

        Returns:
            Tuple[Tensor, Tensor]:
                - final_state: [n_anchor, 20, 4]
                - residual_velocity: 마지막 step의 residual_velocity_head 출력 [n_anchor, 20, 4]
        """
        batch_size = int(anchor_hidden_valid.shape[0])
        device = anchor_hidden_valid.device
        dtype = anchor_hidden_valid.dtype

        dt = (1.0 - float(flow_ode.eps)) / float(self.rollout_steps)
        times = self._build_step_times(flow_ode, batch_size, device, dtype)

        x_t = torch.randn(batch_size, 20, 4, device=device, dtype=dtype)

        # OT flow ODE: dx = v_θ(x_t, tau) * dt
        # drift_from_velocity (SDE-equivalent)를 쓰지 않고 velocity를 직접 적분합니다.
        # DDP unused-parameter 방지: 매 step에서 autograd 활성 상태로 forward합니다.
        velocity_dict: dict = {}
        for step_idx in range(self.rollout_steps):
            tau = times[step_idx]
            velocity_dict = flow_decoder.forward_components(
                anchor_hidden=anchor_hidden_valid,
                x_t_norm=x_t,
                tau=tau,
            )
            # OT flow ODE: x_{t+dt} = x_t + dt * v_θ
            next_x_t = x_t + dt * velocity_dict["velocity"]
            if step_idx < self.rollout_steps - 1:
                # 중간 step: state는 detach해 그래프가 누적되지 않게 합니다.
                x_t = next_x_t.detach()
            else:
                # 마지막 step: gradient graph 유지
                x_t = next_x_t

        self._assert_finite_tensor("ode_final_state", x_t)
        return x_t, velocity_dict["residual_velocity"]

    def forward_open_loop(
        self,
        *,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        anchor_hidden_valid: Tensor,
        agent_type: Tensor,
        current_control: Optional[Tensor] = None,
        current_control_valid: Optional[Tensor] = None,
    ) -> TerminalCostFinalStepResult:
        """
        GT context에서 anchors를 독립적으로 샘플링하는 open-loop penalty입니다.
        """
        if anchor_hidden_valid.numel() == 0:
            # DDP: residual_velocity_head 파라미터를 gradient graph에 연결해야 함
            zero = self._zero_loss_with_trainable_dependency(
                flow_decoder=flow_decoder,
                device=anchor_hidden_valid.device,
                dtype=torch.float32,
            )
            return TerminalCostFinalStepResult(
                loss=zero,
                terminal_cost=zero.detach(),
                projection_gap=zero.detach(),
                final_count=zero.detach(),
            )

        anchor_hidden_valid = anchor_hidden_valid.to(dtype=torch.float32)
        if current_control is not None:
            current_control = current_control.to(dtype=torch.float32, device=anchor_hidden_valid.device)

        final_state, residual_velocity = self._rollout_ode_last_step_grad(
            flow_decoder=flow_decoder,
            flow_ode=flow_ode,
            anchor_hidden_valid=anchor_hidden_valid,
        )
        terminal_cost, metrics = self.projector.compute_terminal_cost(
            pred_clean_norm=final_state,
            agent_type=agent_type,
            current_control=current_control,
            current_control_valid=current_control_valid,
        )
        self._assert_finite_tensor("open_loop/terminal_cost", terminal_cost)
        self._assert_finite_tensor("open_loop/projection_gap", metrics["projection_gap"])

        # flow regularization: residual_velocity → 0 으로 당겨 pretrained 분포 유지
        flow_reg_loss = self.flow_reg_lambda * residual_velocity.pow(2).mean()
        loss = terminal_cost + flow_reg_loss

        return TerminalCostFinalStepResult(
            loss=loss,
            terminal_cost=terminal_cost.detach(),
            projection_gap=metrics["projection_gap"].detach(),
            final_count=torch.tensor(anchor_hidden_valid.shape[0], device=terminal_cost.device),
            flow_reg_loss=flow_reg_loss.detach(),
        )

    @staticmethod
    def _assert_finite_tensor(name: str, value: Tensor) -> None:
        """NaN/Inf가 있으면 즉시 실패시킵니다."""
        if value.numel() == 0:
            return
        finite_mask = torch.isfinite(value)
        if bool(finite_mask.all()):
            return
        bad_values = value.detach()[~finite_mask].flatten()[:8].cpu().tolist()
        raise RuntimeError(f"{name} contains non-finite values: {bad_values}")

    def _zero_loss_with_trainable_dependency(
        self,
        flow_decoder: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        빈 anchor에서도 graph 연결을 유지한 0-loss를 만듭니다.
        """
        # DDP에서 "unused parameter"를 피하려면 trainable 파라미터 전부가
        # 그래프에 연결되어야 합니다. 첫 파라미터 하나만 연결하면 나머지가
        # unused로 판정될 수 있습니다.
        zero = torch.zeros((), device=device, dtype=dtype)
        has_trainable = False
        for p in flow_decoder.parameters():
            if p.requires_grad:
                zero = zero + p.sum() * 0.0
                has_trainable = True
        if has_trainable:
            return zero
        return torch.zeros((), device=device, dtype=dtype)

    def forward_closed_loop(
        self,
        *,
        agent_decoder: nn.Module,
        flow_decoder: nn.Module,
        flow_ode: nn.Module,
        tokenized_agent: Dict[str, Tensor],
        map_feature: Dict[str, Tensor],
        sampling_seed: Optional[int] = None,
    ) -> TerminalCostFinalStepResult:
        """
        closed-loop receding horizon에서 각 step의 2초 terminal penalty를 누적합니다.

        중요한 점:
        - context/commit/retokenize는 autograd 그래프를 만들 필요가 없으므로 no_grad로 처리합니다.
        - diffusion 내부에서 memoryless SDE rollback은 마지막 step에서만 graph를 유지합니다.
        - 다음 horizon update는 생성된 상태를 detach해서 그래프 폭증을 방지합니다.
        """
        # rollout_cache/노이즈 테이프는 그래프가 필요 없습니다.
        with torch.no_grad():
            rollout_cache = agent_decoder.prepare_inference_cache(
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
            )

        state = agent_decoder._clone_rollout_cache(rollout_cache)
        n_agent = int(state["n_agent"])
        n_step_future_10hz = int(state["n_step_future_10hz"])
        n_step_future_2hz = int(state["n_step_future_2hz"])
        max_context_steps = int(state["max_context_steps"])

        pos_window = state["pos_window"]
        head_window = state["head_window"]
        head_vector_window = state["head_vector_window"]
        valid_window = state["valid_window"]
        pred_idx_window = state["pred_idx_window"]
        feat_a = state["feat_a"]
        agent_token_emb = state["agent_token_emb"]
        agent_token_emb_veh = state["agent_token_emb_veh"]
        agent_token_emb_ped = state["agent_token_emb_ped"]
        agent_token_emb_cyc = state["agent_token_emb_cyc"]
        veh_mask = state["veh_mask"]
        ped_mask = state["ped_mask"]
        cyc_mask = state["cyc_mask"]
        categorical_embs = state["categorical_embs"]
        feat_a_now = state["feat_a_now"]
        feat_a_t_dict: Dict[int, Tensor] = state["feat_a_t_dict"]

        if n_agent == 0:
            zero = self._zero_loss_with_trainable_dependency(
                flow_decoder=flow_decoder,
                device=pos_window.device,
                dtype=torch.float32,
            )
            return TerminalCostFinalStepResult(
                loss=zero,
                terminal_cost=zero.detach(),
                projection_gap=zero.detach(),
                final_count=zero.detach(),
            )

        # finetune_config.rollout_noise_scale을 rollout_from_cache의 x_init_norm 테이프로 재사용합니다.
        sampling_noise = SimpleNamespace(noise_scale=self.rollout_noise_scale)
        with torch.no_grad():
            rollout_noise_tape = agent_decoder._build_rollout_noise_tape(
                num_agent=n_agent,
                tape_steps=n_step_future_10hz + 20 - agent_decoder.shift,
                device=feat_a_now.device,
                dtype=feat_a_now.dtype,
                sampling_noise=sampling_noise,
                sampling_seed=sampling_seed,
                scenario_sampling_seeds=None,
                agent_batch=tokenized_agent.get("batch", None),
            )

        total_loss_sum = torch.zeros(
            (), device=feat_a_now.device, dtype=torch.float32
        )
        total_count = torch.zeros((), device=feat_a_now.device, dtype=torch.float32)
        projection_gap_sum = torch.zeros((), device=feat_a_now.device, dtype=torch.float32)

        for t in range(n_step_future_2hz):
            n_step = pos_window.shape[1]

            # context computation
            if t == 0:
                current_hidden = feat_a_now
            else:
                with torch.no_grad():
                    inference_mask = valid_window.clone()
                    inference_mask[:, :-1] = False

                    edge_index_t, r_t = agent_decoder.build_temporal_edge(
                        pos_a=pos_window,
                        head_a=head_window,
                        head_vector_a=head_vector_window,
                        mask=valid_window,
                        inference_mask=inference_mask,
                    )
                    edge_index_t[1] = (edge_index_t[1] + 1) // n_step - 1

                    edge_index_pl2a, r_pl2a = agent_decoder.build_map2agent_edge(
                        pos_pl=map_feature["position"],
                        orient_pl=map_feature["orientation"],
                        pos_a=pos_window[:, -1:],
                        head_a=head_window[:, -1:],
                        head_vector_a=head_vector_window[:, -1:],
                        mask=inference_mask[:, -1:],
                        batch_s=tokenized_agent["batch"],
                        batch_pl=map_feature["batch"],
                    )

                    recent_motion = agent_decoder._build_recent_coarse_motion(
                        pos_window=pos_window,
                        valid_window=valid_window,
                    )
                    edge_index_a2a, r_a2a = agent_decoder.build_interaction_edge(
                        pos_a=pos_window[:, -1:],
                        head_a=head_window[:, -1:],
                        head_vector_a=head_vector_window[:, -1:],
                        batch_s=tokenized_agent["batch"],
                        mask=inference_mask[:, -1:],
                        motion_a=recent_motion.unsqueeze(1),
                    )

                    for i in range(agent_decoder.num_layers):
                        temporal_feat = feat_a if i == 0 else feat_a_t_dict[i]
                        current_hidden = agent_decoder.t_attn_layers[i](
                            (temporal_feat.flatten(0, 1), temporal_feat[:, -1]),
                            r_t,
                            edge_index_t,
                        )
                        current_hidden = agent_decoder.pt2a_attn_layers[i](
                            (map_feature["pt_token"], current_hidden),
                            r_pl2a,
                            edge_index_pl2a,
                        )
                        current_hidden = agent_decoder.a2a_attn_layers[i](
                            current_hidden, r_a2a, edge_index_a2a
                        )
                        if i + 1 < agent_decoder.num_layers:
                            feat_a_t_dict[i + 1] = torch.cat(
                                [feat_a_t_dict[i + 1], current_hidden.unsqueeze(1)],
                                dim=1,
                            )

            active_mask = valid_window[:, -1]
            if not bool(active_mask.any()):
                # next state update should still advance context windows
                with torch.no_grad():
                    next_pos = pos_window[:, -1].clone()
                    next_head = head_window[:, -1].clone()
                    next_token_idx = pred_idx_window[:, -1].clone()

                    next_valid = active_mask.clone()
                    pos_window = torch.cat([pos_window, next_pos.unsqueeze(1)], dim=1)
                    head_window = torch.cat([head_window, next_head.unsqueeze(1)], dim=1)
                    valid_window = torch.cat([valid_window, next_valid.unsqueeze(1)], dim=1)
                    pred_idx_window = torch.cat([pred_idx_window, next_token_idx.unsqueeze(1)], dim=1)

                    head_vector_next = torch.stack(
                        [next_head.cos(), next_head.sin()], dim=-1
                    )
                    head_vector_window = torch.cat(
                        [head_vector_window, head_vector_next.unsqueeze(1)], dim=1
                    )

                    agent_token_emb_next = torch.zeros_like(agent_token_emb[:, 0])
                    agent_token_emb_next[veh_mask] = agent_token_emb_veh[next_token_idx[veh_mask]]
                    agent_token_emb_next[ped_mask] = agent_token_emb_ped[next_token_idx[ped_mask]]
                    agent_token_emb_next[cyc_mask] = agent_token_emb_cyc[next_token_idx[cyc_mask]]
                    agent_token_emb = torch.cat(
                        [agent_token_emb, agent_token_emb_next.unsqueeze(1)], dim=1
                    )

                    motion_vector_a = pos_window[:, -1] - pos_window[:, -2]
                    x_a = torch.stack(
                        [
                            torch.norm(motion_vector_a, p=2, dim=-1),
                        angle_between_2d_vectors(
                            ctr_vector=head_vector_window[:, -1],
                            nbr_vector=motion_vector_a,
                        ),
                        ],
                        dim=-1,
                    )
                    x_a = agent_decoder.x_a_emb(
                        continuous_inputs=x_a, categorical_embs=categorical_embs
                    )
                    feat_a_next = agent_decoder.fusion_emb(
                        torch.cat([agent_token_emb_next, x_a], dim=-1).unsqueeze(1)
                    )
                    feat_a = torch.cat([feat_a, feat_a_next], dim=1)

                    if pos_window.shape[1] > max_context_steps:
                        pos_window = pos_window[:, -max_context_steps:]
                        head_window = head_window[:, -max_context_steps:]
                        head_vector_window = head_vector_window[:, -max_context_steps:]
                        valid_window = valid_window[:, -max_context_steps:]
                        pred_idx_window = pred_idx_window[:, -max_context_steps:]
                        agent_token_emb = agent_token_emb[:, -max_context_steps:]
                        feat_a = feat_a[:, -max_context_steps:]
                        for key in feat_a_t_dict:
                            feat_a_t_dict[key] = feat_a_t_dict[key][
                                :, -max_context_steps:
                            ]
                continue

            # diffusion + terminal cost (여기만 gradient 유지)
            active_hidden = current_hidden[active_mask]
            noise_start = t * agent_decoder.shift
            x_init_norm = rollout_noise_tape[
                active_mask, noise_start : noise_start + 20
            ].contiguous()

            with torch.autocast(
                device_type=active_hidden.device.type if active_hidden.device.type else "cpu",
                enabled=False,
            ):
                final_state = self._rollout_memoryless_sde_last_step_grad(
                    flow_decoder=flow_decoder,
                    flow_ode=flow_ode,
                    anchor_hidden_valid=active_hidden,
                    x_init_norm=x_init_norm,
                )[0]

                self._assert_finite_tensor(f"closed_loop/t{t}/final_state", final_state)
                # closed-loop에서는 이전 continuity control을 쉽게 만들기 어려우므로 None 처리합니다.
                agent_type = tokenized_agent["type"][active_mask]
                terminal_cost, metrics = self.projector.compute_terminal_cost(
                    pred_clean_norm=final_state,
                    agent_type=agent_type,
                    current_control=None,
                    current_control_valid=None,
                )

            active_count = active_mask.sum().to(dtype=torch.float32, device=terminal_cost.device)
            total_loss_sum = total_loss_sum + terminal_cost.to(dtype=torch.float32) * active_count
            total_count = total_count + active_count
            projection_gap_sum = projection_gap_sum + metrics["projection_gap"].to(dtype=torch.float32) * active_count

            # commit + retokenize는 다음 horizon update만 위한 값이므로 detach합니다.
            with torch.no_grad():
                commit_pos_act, commit_head_act, next_pos_act, next_head_act = agent_decoder.commit_bridge.commit(
                    y_hat_norm=final_state.detach(),
                    current_pos=pos_window[active_mask, -1],
                    current_head=head_window[active_mask, -1],
                )
                next_token_idx_act = agent_decoder.commit_bridge.retokenize(
                    current_pos=pos_window[active_mask, -1],
                    current_head=head_window[active_mask, -1],
                    commit_pos=commit_pos_act,
                    commit_head=commit_head_act,
                    agent_type=tokenized_agent["type"][active_mask],
                    token_agent_shape=tokenized_agent["token_agent_shape"][active_mask],
                    token_bank_all_veh=tokenized_agent["token_bank_all_veh"],
                    token_bank_all_ped=tokenized_agent["token_bank_all_ped"],
                    token_bank_all_cyc=tokenized_agent["token_bank_all_cyc"],
                )

                next_pos = pos_window[:, -1].clone()
                next_head = head_window[:, -1].clone()
                next_token_idx = pred_idx_window[:, -1].clone()
                next_pos[active_mask] = next_pos_act
                next_head[active_mask] = next_head_act
                next_token_idx[active_mask] = next_token_idx_act

                next_valid = active_mask.clone()
                pos_window = torch.cat([pos_window, next_pos.unsqueeze(1)], dim=1)
                head_window = torch.cat([head_window, next_head.unsqueeze(1)], dim=1)
                valid_window = torch.cat([valid_window, next_valid.unsqueeze(1)], dim=1)
                pred_idx_window = torch.cat([pred_idx_window, next_token_idx.unsqueeze(1)], dim=1)

                head_vector_next = torch.stack([next_head.cos(), next_head.sin()], dim=-1)
                head_vector_window = torch.cat(
                    [head_vector_window, head_vector_next.unsqueeze(1)], dim=1
                )

                agent_token_emb_next = torch.zeros_like(agent_token_emb[:, 0])
                agent_token_emb_next[veh_mask] = agent_token_emb_veh[next_token_idx[veh_mask]]
                agent_token_emb_next[ped_mask] = agent_token_emb_ped[next_token_idx[ped_mask]]
                agent_token_emb_next[cyc_mask] = agent_token_emb_cyc[next_token_idx[cyc_mask]]
                agent_token_emb = torch.cat(
                    [agent_token_emb, agent_token_emb_next.unsqueeze(1)], dim=1
                )

                motion_vector_a = pos_window[:, -1] - pos_window[:, -2]
                x_a = torch.stack(
                    [
                        torch.norm(motion_vector_a, p=2, dim=-1),
                        angle_between_2d_vectors(
                            ctr_vector=head_vector_window[:, -1],
                            nbr_vector=motion_vector_a,
                        ),
                    ],
                    dim=-1,
                )

                x_a = agent_decoder.x_a_emb(
                    continuous_inputs=x_a, categorical_embs=categorical_embs
                )
                feat_a_next = agent_decoder.fusion_emb(
                    torch.cat([agent_token_emb_next, x_a], dim=-1).unsqueeze(1)
                )
                feat_a = torch.cat([feat_a, feat_a_next], dim=1)

                if pos_window.shape[1] > max_context_steps:
                    pos_window = pos_window[:, -max_context_steps:]
                    head_window = head_window[:, -max_context_steps:]
                    head_vector_window = head_vector_window[:, -max_context_steps:]
                    valid_window = valid_window[:, -max_context_steps:]
                    pred_idx_window = pred_idx_window[:, -max_context_steps:]
                    agent_token_emb = agent_token_emb[:, -max_context_steps:]
                    feat_a = feat_a[:, -max_context_steps:]
                    for key in feat_a_t_dict:
                        feat_a_t_dict[key] = feat_a_t_dict[key][
                            :, -max_context_steps:
                        ]

        if total_count.item() <= 0:
            zero = self._zero_loss_with_trainable_dependency(
                flow_decoder=flow_decoder,
                device=total_loss_sum.device,
                dtype=torch.float32,
            )
            return TerminalCostFinalStepResult(
                loss=zero,
                terminal_cost=zero.detach(),
                projection_gap=zero.detach(),
                final_count=zero.detach(),
            )

        avg_loss = total_loss_sum / total_count
        avg_projection_gap = projection_gap_sum / total_count.clamp_min(1.0)
        return TerminalCostFinalStepResult(
            loss=avg_loss,
            terminal_cost=avg_loss.detach(),
            projection_gap=avg_projection_gap.detach(),
            final_count=total_count.detach(),
        )

