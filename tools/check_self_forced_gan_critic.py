from __future__ import annotations

import torch

from src.smart.modules.self_forced_gan_critic import (
    SelfForcedGANDiscriminator,
    _safe_masked_max,
    _safe_masked_mean,
    _safe_masked_min,
    _safe_masked_std,
)


def main() -> None:
    torch.manual_seed(7)
    bsz, k, n_step, n_agent, n_map = 2, 16, 20, 12, 48
    critic = SelfForcedGANDiscriminator(hidden_dim=128, n_rollout=k, n_step=n_step)
    rollout = torch.randn(bsz, k, n_step, n_agent, 4, requires_grad=True)
    with torch.no_grad():
        rollout[..., 2:] = torch.nn.functional.normalize(rollout[..., 2:], dim=-1)
    current = torch.zeros(bsz, n_agent, 4)
    current[..., 2] = 1.0
    agent_type = torch.zeros(bsz, n_agent, dtype=torch.long)
    valid = torch.ones(bsz, n_agent, dtype=torch.bool)
    valid[1, -2:] = False
    agent_context = torch.randn(bsz, n_agent, 128)
    map_context = torch.randn(bsz, n_map, 128)
    map_position = torch.randn(bsz, n_map, 2) * 20.0
    map_orientation = torch.randn(bsz, n_map)
    map_valid = torch.ones(bsz, n_map, dtype=torch.bool)
    map_valid[1, -5:] = False

    logit = critic(
        rollout,
        current_pose=current,
        agent_type=agent_type,
        valid_mask=valid,
        agent_context=agent_context,
        map_context=map_context,
        map_position=map_position,
        map_orientation=map_orientation,
        map_valid_mask=map_valid,
    )
    loss = logit.square().mean()
    loss.backward()

    n_param = critic.count_trainable_parameters()
    has_grad = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in critic.parameters()
        if parameter.requires_grad
    )
    print(f"logit_shape={tuple(logit.shape)} trainable_params={n_param}")
    assert tuple(logit.shape) == (bsz, 1)
    assert torch.isfinite(logit).all()
    assert torch.isfinite(rollout.grad).all()
    assert has_grad
    assert 600_000 <= n_param <= 800_000

    pooled_input = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0], [-1000.0, 1000.0]]]],
        dtype=torch.float32,
    )
    pooled_mask = torch.tensor([[[[True], [True], [False]]]])
    assert torch.allclose(_safe_masked_mean(pooled_input, pooled_mask, dim=2), torch.tensor([[[2.0, 3.0]]]))
    assert torch.allclose(_safe_masked_std(pooled_input, pooled_mask, dim=2), torch.tensor([[[1.0, 1.0]]]))
    assert torch.allclose(_safe_masked_max(pooled_input, pooled_mask, dim=2), torch.tensor([[[3.0, 4.0]]]))
    assert torch.allclose(_safe_masked_min(pooled_input, pooled_mask, dim=2), torch.tensor([[[1.0, 2.0]]]))

    all_map_invalid = critic(
        rollout.detach(),
        current_pose=current,
        agent_type=agent_type,
        valid_mask=valid,
        agent_context=agent_context,
        map_context=map_context,
        map_position=map_position,
        map_orientation=map_orientation,
        map_valid_mask=torch.zeros_like(map_valid),
    )
    assert torch.isfinite(all_map_invalid).all()


if __name__ == "__main__":
    main()
