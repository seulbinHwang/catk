from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData

from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import transform_to_global, wrap_angle

from .anchors import (
    AnchorSpec,
    NUM_AGENT_TYPES,
    execute_local_anchor,
    gather_anchors_by_type,
    make_local_future,
    match_anchors_by_type,
)


@dataclass(frozen=True)
class UniMMTrainingBatch:
    tokenized_map: Dict[str, Tensor]
    tokenized_agent: Dict[str, Tensor]
    context_indices: Tensor
    target_local: Tensor
    target_valid: Tensor
    z_star: Tensor
    z_star_error: Tensor
    posterior_stats: Dict[str, Tensor]


class UniMMProcessor(nn.Module):
    """Build UniMM continuous inputs from the existing WOMD cache schema."""

    def __init__(
        self,
        prediction_horizon_steps: int = 40,
        commit_steps: int = 5,
        match_steps: int = 5,
        first_context_step: int = 10,
        last_train_context_step: int = 85,
        anchor_heading_weight: float = 1.0,
        anchor_match_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        self.spec = AnchorSpec(
            num_prediction_steps=int(prediction_horizon_steps),
            num_commit_steps=int(commit_steps),
            num_match_steps=int(match_steps),
        )
        self.first_context_step = int(first_context_step)
        self.last_train_context_step = int(last_train_context_step)
        self.anchor_heading_weight = float(anchor_heading_weight)
        self.anchor_match_chunk_size = int(anchor_match_chunk_size)
        if self.first_context_step % self.spec.num_commit_steps != 0:
            raise ValueError("first_context_step must align to commit_steps")
        if self.last_train_context_step % self.spec.num_commit_steps != 0:
            raise ValueError("last_train_context_step must align to commit_steps")

    @staticmethod
    def tokenize_map(data: HeteroData) -> Dict[str, Tensor]:
        traj_pos = data["map_save"]["traj_pos"]
        traj_theta = data["map_save"]["traj_theta"]
        return {
            "traj_pos": traj_pos.contiguous(),
            "position": traj_pos[:, 0].contiguous(),
            "orientation": traj_theta.contiguous(),
            "type": data["pt_token"]["type"].long(),
            "pl_type": data["pt_token"]["pl_type"].long(),
            "light_type": data["pt_token"]["light_type"].long(),
            "batch": data["pt_token"]["batch"],
        }

    @staticmethod
    def _raw_agent_state(data: HeteroData) -> tuple[Tensor, Tensor, Tensor]:
        valid = data["agent"]["valid_mask"].clone()
        pos = data["agent"]["position"][..., :2].clone().contiguous()
        head = data["agent"]["heading"].clone()
        head = TokenProcessor._clean_heading(valid, head)
        return pos, head, valid

    def _context_raw_steps(self, training: bool) -> list[int]:
        if training:
            return list(
                range(
                    self.first_context_step,
                    self.last_train_context_step + 1,
                    self.spec.num_commit_steps,
                )
            )
        return [self.first_context_step]

    def _state_token_steps(self, max_context_step: int) -> list[int]:
        return list(range(self.spec.num_commit_steps, max_context_step + 1, self.spec.num_commit_steps))

    def _make_gt_state_sequence(
        self,
        pos: Tensor,
        head: Tensor,
        valid: Tensor,
        token_steps: Sequence[int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        idx = torch.tensor(token_steps, dtype=torch.long, device=pos.device)
        return pos[:, idx], head[:, idx], valid[:, idx]

    def _make_gt_tracklet_sequence(
        self,
        pos: Tensor,
        head: Tensor,
        valid: Tensor,
        token_steps: Sequence[int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        pos_rows = []
        head_rows = []
        valid_rows = []
        for raw_step in token_steps:
            start = int(raw_step) - self.spec.num_commit_steps + 1
            if start < 0:
                raise ValueError(f"tracklet ending at step {raw_step} starts before cache step 0")
            sl = slice(start, int(raw_step) + 1)
            pos_rows.append(pos[:, sl])
            head_rows.append(head[:, sl])
            valid_rows.append(valid[:, sl])
        return torch.stack(pos_rows, dim=1), torch.stack(head_rows, dim=1), torch.stack(valid_rows, dim=1)

    def _future_window(
        self,
        pos: Tensor,
        head: Tensor,
        valid: Tensor,
        start_step: int,
        horizon_steps: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be positive")
        if start_step < 0 or start_step >= pos.shape[1]:
            raise ValueError(
                f"start_step={start_step} is outside cache length {pos.shape[1]}"
            )

        n_agent = pos.shape[0]
        future_pos = pos.new_zeros((n_agent, horizon_steps, pos.shape[-1]))
        future_head = head.new_zeros((n_agent, horizon_steps))
        future_valid = valid.new_zeros((n_agent, horizon_steps))

        available_steps = min(horizon_steps, max(pos.shape[1] - start_step - 1, 0))
        if available_steps > 0:
            sl = slice(start_step + 1, start_step + 1 + available_steps)
            future_pos[:, :available_steps] = pos[:, sl]
            future_head[:, :available_steps] = head[:, sl]
            future_valid[:, :available_steps] = valid[:, sl]
        return future_pos, future_head, future_valid

    def _build_targets(
        self,
        pos: Tensor,
        head: Tensor,
        valid: Tensor,
        state_pos: Tensor,
        state_head: Tensor,
        state_valid: Tensor,
        context_indices: Tensor,
        context_raw_steps: Sequence[int],
    ) -> tuple[Tensor, Tensor]:
        target_local_rows = []
        target_valid_rows = []
        for ctx_idx, raw_step in zip(context_indices.tolist(), context_raw_steps):
            fut_pos, fut_head, fut_valid = self._future_window(
                pos,
                head,
                valid,
                start_step=int(raw_step),
                horizon_steps=self.spec.num_prediction_steps,
            )
            target_local_rows.append(
                make_local_future(
                    pos=fut_pos,
                    head=fut_head,
                    ref_pos=state_pos[:, ctx_idx],
                    ref_head=state_head[:, ctx_idx],
                )
            )
            target_valid_rows.append(fut_valid & state_valid[:, ctx_idx].unsqueeze(1))
        return torch.stack(target_local_rows, dim=1), torch.stack(target_valid_rows, dim=1)

    def _build_approx_posterior_states(
        self,
        pos: Tensor,
        head: Tensor,
        valid: Tensor,
        anchors_by_type: Tensor,
        posterior_threshold: Tensor | None,
        max_context_step: int,
        agent_type: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        token_steps = self._state_token_steps(max_context_step)
        state_pos, state_head, state_valid = self._make_gt_state_sequence(pos, head, valid, token_steps)
        tracklet_pos, tracklet_head, tracklet_valid = self._make_gt_tracklet_sequence(pos, head, valid, token_steps)
        max_context_idx = len(token_steps) - 1
        if max_context_idx <= 1:
            return (
                state_pos,
                state_head,
                state_valid,
                tracklet_pos,
                tracklet_head,
                tracklet_valid,
                self._empty_posterior_stats(pos.device, pos.dtype),
            )

        error_values = []
        error_over_threshold_values = []
        valid_count = pos.new_zeros(())
        accept_count = pos.new_zeros(())
        type_valid_count = pos.new_zeros((NUM_AGENT_TYPES,))
        type_accept_count = pos.new_zeros((NUM_AGENT_TYPES,))
        context_accept_rates = []
        context_raw_steps = []

        for ctx_idx in range(1, max_context_idx):
            raw_step = token_steps[ctx_idx]
            fut_pos, fut_head, fut_valid = self._future_window(
                pos,
                head,
                valid,
                start_step=raw_step,
                horizon_steps=self.spec.num_commit_steps,
            )
            target_local = make_local_future(
                pos=fut_pos,
                head=fut_head,
                ref_pos=state_pos[:, ctx_idx],
                ref_head=state_head[:, ctx_idx],
            )
            z_post, post_error = match_anchors_by_type(
                anchors_by_type=anchors_by_type,
                agent_type=agent_type,
                target_local=target_local,
                valid=fut_valid & state_valid[:, ctx_idx].unsqueeze(1),
                horizon_steps=self.spec.num_match_steps,
                heading_weight=self.anchor_heading_weight,
                row_chunk_size=self.anchor_match_chunk_size,
            )
            selected_anchor = gather_anchors_by_type(anchors_by_type, agent_type, z_post)
            commit_pos, commit_head = execute_local_anchor(
                anchor=selected_anchor,
                ref_pos=state_pos[:, ctx_idx],
                ref_head=state_head[:, ctx_idx],
                commit_steps=self.spec.num_commit_steps,
            )
            next_tracklet_pos = commit_pos
            next_tracklet_head = commit_head
            next_tracklet_valid = fut_valid & state_valid[:, ctx_idx].unsqueeze(1)
            next_pos = commit_pos[:, -1]
            next_head = commit_head[:, -1]
            next_valid = valid[:, raw_step + self.spec.num_commit_steps] & state_valid[:, ctx_idx]

            if posterior_threshold is not None:
                threshold = posterior_threshold.to(device=pos.device, dtype=post_error.dtype)[agent_type.long()]
                use_posterior = post_error <= threshold
                stats_valid = (fut_valid & state_valid[:, ctx_idx].unsqueeze(1))[
                    :, : self.spec.num_match_steps
                ].any(dim=-1)
                if bool(stats_valid.any()):
                    valid_error = post_error[stats_valid]
                    valid_threshold = threshold[stats_valid].clamp_min(1e-12)
                    valid_use = use_posterior[stats_valid]
                    error_values.append(valid_error.detach())
                    error_over_threshold_values.append((valid_error / valid_threshold).detach())
                    valid_count = valid_count + stats_valid.to(dtype=pos.dtype).sum()
                    accept_count = accept_count + (use_posterior & stats_valid).to(dtype=pos.dtype).sum()
                    for type_idx in range(NUM_AGENT_TYPES):
                        type_mask = (agent_type.long() == type_idx) & stats_valid
                        if bool(type_mask.any()):
                            type_valid_count[type_idx] = (
                                type_valid_count[type_idx] + type_mask.to(dtype=pos.dtype).sum()
                            )
                            type_accept_count[type_idx] = (
                                type_accept_count[type_idx]
                                + (use_posterior & type_mask).to(dtype=pos.dtype).sum()
                            )
                    context_accept_rates.append(valid_use.to(dtype=pos.dtype).mean())
                else:
                    context_accept_rates.append(pos.new_zeros(()))
                context_raw_steps.append(int(raw_step))

                next_pos = torch.where(use_posterior.unsqueeze(1), next_pos, pos[:, raw_step + self.spec.num_commit_steps])
                next_head = torch.where(use_posterior, next_head, head[:, raw_step + self.spec.num_commit_steps])
                next_tracklet_pos = torch.where(
                    use_posterior[:, None, None],
                    next_tracklet_pos,
                    fut_pos,
                )
                next_tracklet_head = torch.where(
                    use_posterior[:, None],
                    next_tracklet_head,
                    fut_head,
                )

            state_pos[:, ctx_idx + 1] = next_pos
            state_head[:, ctx_idx + 1] = wrap_angle(next_head)
            state_valid[:, ctx_idx + 1] = next_valid
            tracklet_pos[:, ctx_idx + 1] = next_tracklet_pos
            tracklet_head[:, ctx_idx + 1] = wrap_angle(next_tracklet_head)
            tracklet_valid[:, ctx_idx + 1] = next_tracklet_valid

        posterior_stats = self._posterior_stats_from_values(
            device=pos.device,
            dtype=pos.dtype,
            error_values=error_values,
            error_over_threshold_values=error_over_threshold_values,
            valid_count=valid_count,
            accept_count=accept_count,
            type_valid_count=type_valid_count,
            type_accept_count=type_accept_count,
            context_accept_rates=context_accept_rates,
            context_raw_steps=context_raw_steps,
        )
        return (
            state_pos,
            state_head,
            state_valid,
            tracklet_pos,
            tracklet_head,
            tracklet_valid,
            posterior_stats,
        )

    @staticmethod
    def _empty_posterior_stats(device: torch.device, dtype: torch.dtype) -> Dict[str, Tensor]:
        return {
            "accept_rate": torch.zeros((), device=device, dtype=dtype),
            "error_mean": torch.zeros((), device=device, dtype=dtype),
            "error_p50": torch.zeros((), device=device, dtype=dtype),
            "error_p90": torch.zeros((), device=device, dtype=dtype),
            "error_p95": torch.zeros((), device=device, dtype=dtype),
            "error_over_threshold": torch.zeros((), device=device, dtype=dtype),
            "accept_rate_by_type": torch.zeros((NUM_AGENT_TYPES,), device=device, dtype=dtype),
            "accept_rate_by_context": torch.zeros((0,), device=device, dtype=dtype),
            "context_raw_steps": torch.zeros((0,), dtype=torch.long),
        }

    @staticmethod
    def _posterior_stats_from_values(
        *,
        device: torch.device,
        dtype: torch.dtype,
        error_values: Sequence[Tensor],
        error_over_threshold_values: Sequence[Tensor],
        valid_count: Tensor,
        accept_count: Tensor,
        type_valid_count: Tensor,
        type_accept_count: Tensor,
        context_accept_rates: Sequence[Tensor],
        context_raw_steps: Sequence[int],
    ) -> Dict[str, Tensor]:
        if error_values:
            errors = torch.cat([value.to(device=device, dtype=dtype) for value in error_values])
            quantiles = torch.quantile(
                errors.float(),
                torch.tensor([0.5, 0.9, 0.95], device=device),
            ).to(dtype=dtype)
            error_mean = errors.mean()
            error_over_threshold = torch.cat(
                [value.to(device=device, dtype=dtype) for value in error_over_threshold_values]
            ).mean()
        else:
            quantiles = torch.zeros((3,), device=device, dtype=dtype)
            error_mean = torch.zeros((), device=device, dtype=dtype)
            error_over_threshold = torch.zeros((), device=device, dtype=dtype)

        accept_rate = accept_count / valid_count.clamp_min(1.0)
        type_accept_rate = type_accept_count / type_valid_count.clamp_min(1.0)
        context_accept_rate = (
            torch.stack(list(context_accept_rates)).to(device=device, dtype=dtype)
            if context_accept_rates
            else torch.zeros((0,), device=device, dtype=dtype)
        )
        context_raw_step_tensor = torch.tensor(
            list(context_raw_steps),
            dtype=torch.long,
        )
        return {
            "accept_rate": accept_rate.detach(),
            "error_mean": error_mean.detach(),
            "error_p50": quantiles[0].detach(),
            "error_p90": quantiles[1].detach(),
            "error_p95": quantiles[2].detach(),
            "error_over_threshold": error_over_threshold.detach(),
            "accept_rate_by_type": type_accept_rate.detach(),
            "accept_rate_by_context": context_accept_rate.detach(),
            "context_raw_steps": context_raw_step_tensor,
        }

    def _base_agent_dict(
        self,
        data: HeteroData,
        state_pos: Tensor,
        state_head: Tensor,
        state_valid: Tensor,
        tracklet_pos: Tensor,
        tracklet_head: Tensor,
        tracklet_valid: Tensor,
    ) -> Dict[str, Tensor]:
        num_graphs = getattr(data, "num_graphs", None)
        if num_graphs is None:
            num_graphs = int(data["agent"]["batch"].max().item()) + 1 if data["agent"]["batch"].numel() else 1
        return {
            "num_graphs": num_graphs,
            "type": data["agent"]["type"].long(),
            "shape": data["agent"]["shape"],
            "batch": data["agent"]["batch"],
            "ego_mask": data["agent"]["role"][:, 0],
            "state_pos": state_pos,
            "state_head": wrap_angle(state_head),
            "state_valid": state_valid,
            "tracklet_pos": tracklet_pos,
            "tracklet_head": wrap_angle(tracklet_head),
            "tracklet_valid": tracklet_valid,
        }

    def build_training_batch(
        self,
        data: HeteroData,
        anchors_by_type: Tensor,
        posterior_threshold: Tensor | None = None,
        use_closed_loop: bool = True,
    ) -> UniMMTrainingBatch:
        tokenized_map = self.tokenize_map(data)
        pos, head, valid = self._raw_agent_state(data)
        context_raw_steps = self._context_raw_steps(training=True)
        max_context_step = max(context_raw_steps)
        token_steps = self._state_token_steps(max_context_step)
        context_indices = torch.tensor(
            [token_steps.index(step) for step in context_raw_steps],
            dtype=torch.long,
            device=pos.device,
        )

        if use_closed_loop:
            (
                state_pos,
                state_head,
                state_valid,
                tracklet_pos,
                tracklet_head,
                tracklet_valid,
                posterior_stats,
            ) = self._build_approx_posterior_states(
                pos=pos,
                head=head,
                valid=valid,
                anchors_by_type=anchors_by_type,
                posterior_threshold=posterior_threshold,
                max_context_step=max_context_step,
                agent_type=data["agent"]["type"].long(),
            )
        else:
            state_pos, state_head, state_valid = self._make_gt_state_sequence(pos, head, valid, token_steps)
            tracklet_pos, tracklet_head, tracklet_valid = self._make_gt_tracklet_sequence(pos, head, valid, token_steps)
            posterior_stats = self._empty_posterior_stats(pos.device, pos.dtype)

        target_local, target_valid = self._build_targets(
            pos=pos,
            head=head,
            valid=valid,
            state_pos=state_pos,
            state_head=state_head,
            state_valid=state_valid,
            context_indices=context_indices,
            context_raw_steps=context_raw_steps,
        )
        n_agent, n_context = target_valid.shape[:2]
        row_agent_type = data["agent"]["type"].long().unsqueeze(1).expand(-1, n_context).reshape(-1)
        z_star, z_star_error = match_anchors_by_type(
            anchors_by_type=anchors_by_type,
            agent_type=row_agent_type,
            target_local=target_local.reshape(n_agent * n_context, self.spec.num_prediction_steps, 3),
            valid=target_valid.reshape(n_agent * n_context, self.spec.num_prediction_steps),
            horizon_steps=self.spec.num_match_steps,
            heading_weight=self.anchor_heading_weight,
            row_chunk_size=self.anchor_match_chunk_size,
        )
        return UniMMTrainingBatch(
            tokenized_map=tokenized_map,
            tokenized_agent=self._base_agent_dict(
                data,
                state_pos,
                state_head,
                state_valid,
                tracklet_pos,
                tracklet_head,
                tracklet_valid,
            ),
            context_indices=context_indices,
            target_local=target_local,
            target_valid=target_valid,
            z_star=z_star.view(n_agent, n_context),
            z_star_error=z_star_error.view(n_agent, n_context),
            posterior_stats=posterior_stats,
        )

    def build_rollout_seed(self, data: HeteroData) -> tuple[Dict[str, Tensor], Dict[str, Tensor], Tensor, Tensor, Tensor]:
        tokenized_map = self.tokenize_map(data)
        pos, head, valid = self._raw_agent_state(data)
        seed_steps = [self.first_context_step - self.spec.num_commit_steps, self.first_context_step]
        state_pos, state_head, state_valid = self._make_gt_state_sequence(pos, head, valid, seed_steps)
        tracklet_pos, tracklet_head, tracklet_valid = self._make_gt_tracklet_sequence(pos, head, valid, seed_steps)
        tokenized_agent = self._base_agent_dict(
            data,
            state_pos,
            state_head,
            state_valid,
            tracklet_pos,
            tracklet_head,
            tracklet_valid,
        )
        current_pos = pos[:, self.first_context_step].clone()
        current_head = head[:, self.first_context_step].clone()
        current_valid = valid[:, self.first_context_step].clone()
        return tokenized_map, tokenized_agent, current_pos, current_head, current_valid

    def append_rollout_state(
        self,
        tokenized_agent: Dict[str, Tensor],
        next_pos: Tensor,
        next_head: Tensor,
        next_valid: Tensor,
        next_tracklet_pos: Tensor | None = None,
        next_tracklet_head: Tensor | None = None,
        next_tracklet_valid: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        out = dict(tokenized_agent)
        out["state_pos"] = torch.cat([out["state_pos"], next_pos.unsqueeze(1)], dim=1)
        out["state_head"] = torch.cat([out["state_head"], wrap_angle(next_head).unsqueeze(1)], dim=1)
        out["state_valid"] = torch.cat([out["state_valid"], next_valid.unsqueeze(1)], dim=1)
        if next_tracklet_pos is not None and next_tracklet_head is not None:
            if next_tracklet_valid is None:
                next_tracklet_valid = next_valid[:, None].expand(
                    -1,
                    next_tracklet_pos.shape[1],
                )
            out["tracklet_pos"] = torch.cat(
                [out["tracklet_pos"], next_tracklet_pos.unsqueeze(1)],
                dim=1,
            )
            out["tracklet_head"] = torch.cat(
                [out["tracklet_head"], wrap_angle(next_tracklet_head).unsqueeze(1)],
                dim=1,
            )
            out["tracklet_valid"] = torch.cat(
                [out["tracklet_valid"], next_tracklet_valid.unsqueeze(1)],
                dim=1,
            )
        return out

    @staticmethod
    def local_prediction_to_global(
        mean_pos: Tensor,
        mean_head: Tensor,
        ref_pos: Tensor,
        ref_head: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pos_global, head_global = transform_to_global(
            pos_local=mean_pos,
            head_local=mean_head,
            pos_now=ref_pos,
            head_now=ref_head,
        )
        return pos_global, wrap_angle(head_global)
