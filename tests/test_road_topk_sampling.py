from types import SimpleNamespace

import torch

from src.smart.utils.rollout import sample_next_token_traj


def _make_token_data():
    n_agent = 1
    n_token = 5
    token_traj = torch.zeros(n_agent, n_token, 4, 2)
    for token_idx in range(n_token):
        token_traj[:, token_idx, :, 0] = float(token_idx)
    token_traj_all = token_traj.unsqueeze(2).expand(n_agent, n_token, 6, 4, 2).clone()
    return token_traj, token_traj_all


def test_road_topk_dist_chooses_closest_gt_within_topk():
    token_traj, token_traj_all = _make_token_data()
    sampling_scheme = SimpleNamespace(criterium="road_topk_dist", num_k=3)
    next_token_logits = torch.tensor([[0.0, 8.0, 10.0, 1.0, 9.0]])

    selected_idx, _ = sample_next_token_traj(
        token_traj=token_traj,
        token_traj_all=token_traj_all,
        sampling_scheme=sampling_scheme,
        next_token_logits=next_token_logits,
        pos_now=torch.zeros(1, 2),
        head_now=torch.zeros(1),
        pos_next_gt=torch.tensor([[4.0, 0.0]]),
        head_next_gt=torch.zeros(1),
        valid_next_gt=torch.ones(1, dtype=torch.bool),
        token_agent_shape=torch.zeros(1, 2),
    )

    assert selected_idx.tolist() == [4]


def test_road_topk_dist_uses_highest_probability_when_gt_invalid():
    token_traj, token_traj_all = _make_token_data()
    sampling_scheme = SimpleNamespace(criterium="road_topk_dist", num_k=3)
    next_token_logits = torch.tensor([[0.0, 8.0, 10.0, 1.0, 9.0]])

    selected_idx, _ = sample_next_token_traj(
        token_traj=token_traj,
        token_traj_all=token_traj_all,
        sampling_scheme=sampling_scheme,
        next_token_logits=next_token_logits,
        pos_now=torch.zeros(1, 2),
        head_now=torch.zeros(1),
        pos_next_gt=torch.tensor([[4.0, 0.0]]),
        head_next_gt=torch.zeros(1),
        valid_next_gt=torch.zeros(1, dtype=torch.bool),
        token_agent_shape=torch.zeros(1, 2),
    )

    assert selected_idx.tolist() == [2]
