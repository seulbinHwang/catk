from __future__ import annotations

import torch
from torch import Tensor

from src.smart.utils import wrap_angle


def clean_heading_dense(valid: Tensor, heading: Tensor) -> Tensor:
    """TokenProcessor heading cleanup shared by online and cache-time paths."""
    valid_pairs = valid[:, :-1] & valid[:, 1:]
    cleaned_steps = [heading[:, 0]]
    prev_heading = heading[:, 0]
    for i in range(heading.shape[1] - 1):
        raw_next_heading = heading[:, i + 1]
        heading_diff = torch.abs(wrap_angle(prev_heading - raw_next_heading))
        change_needed = (heading_diff > 1.5) & valid_pairs[:, i]
        next_heading = torch.where(change_needed, prev_heading, raw_next_heading)
        cleaned_steps.append(next_heading)
        prev_heading = next_heading
    return torch.stack(cleaned_steps, dim=1)


def extrapolate_agent_to_prev_token_step(
    valid: Tensor,
    pos: Tensor,
    heading: Tensor,
    vel: Tensor,
    *,
    shift: int = 5,
    current_step: int = 10,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fill missing steps just before the first valid token boundary.

    This mirrors ``TokenProcessor._extrapolate_agent_to_prev_token_step`` so
    cache-time control targets and online tokenization see the same raw state.
    """
    first_valid_step = torch.max(valid, dim=1).indices
    n_step_to_extrapolate = first_valid_step.remainder(int(shift))

    prev_token_step = int(current_step) - int(shift)
    if 0 <= prev_token_step < valid.shape[1]:
        needs_history_token = (first_valid_step == int(current_step)) & (
            ~valid[:, prev_token_step]
        )
        n_step_to_extrapolate = torch.where(
            needs_history_token,
            torch.full_like(n_step_to_extrapolate, int(shift)),
            n_step_to_extrapolate,
        )

    step_index = torch.arange(valid.shape[1], device=valid.device).unsqueeze(0)
    fill_start = first_valid_step - n_step_to_extrapolate
    fill_mask = (
        (n_step_to_extrapolate > 0).unsqueeze(1)
        & (step_index >= fill_start.unsqueeze(1))
        & (step_index < first_valid_step.unsqueeze(1))
    )
    if not bool(fill_mask.any().item()):
        return valid, pos, heading, vel

    agent_index, step_index_flat = fill_mask.nonzero(as_tuple=True)
    source_step = first_valid_step[agent_index]
    source_vel = vel[agent_index, source_step]

    valid[agent_index, step_index_flat] = True
    vel[agent_index, step_index_flat] = source_vel
    heading[agent_index, step_index_flat] = heading[agent_index, source_step]
    delta_step = (source_step - step_index_flat).to(dtype=pos.dtype).unsqueeze(-1)
    pos[agent_index, step_index_flat] = (
        pos[agent_index, source_step] - source_vel * (0.1 * delta_step)
    )

    return valid, pos, heading, vel
