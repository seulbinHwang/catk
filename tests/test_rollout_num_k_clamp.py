import pytest
import torch
from omegaconf import OmegaConf

from src.smart.utils.rollout import sample_next_token_traj


def _sample_with_oversized_num_k(criterium: str):
    n_agent = 2
    n_token = 3
    generator = torch.Generator().manual_seed(7)
    token_traj = torch.randn(n_agent, n_token, 4, 2, generator=generator)
    token_traj_all = torch.randn(n_agent, n_token, 6, 4, 2, generator=generator)
    logits = torch.tensor(
        [
            [3.0, 1.0, -2.0],
            [-1.0, 2.0, 0.5],
        ],
        dtype=torch.float32,
    )
    zeros_xy = torch.zeros(n_agent, 2)
    zeros_head = torch.zeros(n_agent)
    valid_next_gt = torch.zeros(n_agent, dtype=torch.bool)
    token_agent_shape = torch.tensor([[2.0, 4.8], [1.0, 1.0]], dtype=torch.float32)
    sampling_scheme = OmegaConf.create(
        {
            "criterium": criterium,
            "num_k": 100,
            "temp": 1.0,
        }
    )

    return sample_next_token_traj(
        token_traj=token_traj,
        token_traj_all=token_traj_all,
        sampling_scheme=sampling_scheme,
        next_token_logits=logits,
        pos_now=zeros_xy,
        head_now=zeros_head,
        pos_next_gt=zeros_xy,
        head_next_gt=zeros_head,
        valid_next_gt=valid_next_gt,
        token_agent_shape=token_agent_shape,
    )


@pytest.mark.parametrize(
    "criterium",
    ["topk_prob", "topk_prob_sampled_with_dist", "topk_dist_sampled_with_prob", "road_samplek_dist"],
)
def test_sample_next_token_traj_clamps_num_k_to_token_count(criterium):
    next_token_idx, next_token_traj_all = _sample_with_oversized_num_k(criterium)

    assert next_token_idx.shape == (2,)
    assert next_token_traj_all.shape == (2, 6, 4, 2)
    assert int(next_token_idx.min()) >= 0
    assert int(next_token_idx.max()) < 3
