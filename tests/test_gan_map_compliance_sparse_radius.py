from __future__ import annotations

import torch

from src.smart.modules.self_forced_gan_critic import (
    MapComplianceEncoder,
    SelfForcedGANDiscriminator,
)


def _target_index(
    *,
    batch: int,
    rollout: int,
    endpoint: int,
    agent: int,
    n_rollout: int,
    n_endpoint: int,
    n_agent: int,
) -> int:
    return (((batch * n_rollout) + rollout) * n_endpoint + endpoint) * n_agent + agent


def test_map_compliance_builds_only_same_scene_radius_edges() -> None:
    encoder = MapComplianceEncoder(hidden_dim=8, radius_m=2.0, num_heads=2)
    bsz, n_rollout, n_endpoint, n_agent, n_map = 2, 2, 2, 3, 4

    endpoint_pose = torch.zeros(bsz, n_rollout, n_endpoint, n_agent, 4)
    endpoint_pose[..., 2] = 1.0
    endpoint_pose[0, :, :, 0, :2] = torch.tensor([0.0, 0.0])
    endpoint_pose[0, :, :, 1, :2] = torch.tensor([10.0, 0.0])
    endpoint_pose[0, :, :, 2, :2] = torch.tensor([0.0, 1.0])
    endpoint_pose[1, :, :, 0, :2] = torch.tensor([100.0, 0.0])
    endpoint_pose[1, :, :, 1, :2] = torch.tensor([200.0, 0.0])
    endpoint_pose[1, :, :, 2, :2] = torch.tensor([101.0, 0.0])

    map_position = torch.tensor(
        [
            [[1.0, 0.0], [11.5, 0.0], [50.0, 0.0], [0.0, 0.0]],
            [[101.0, 0.0], [201.0, 0.0], [0.0, 0.0], [100.0, 1.0]],
        ]
    )
    map_orientation = torch.zeros(bsz, n_map)
    valid_mask = torch.tensor([[True, True, False], [True, False, True]])
    map_valid_mask = torch.tensor([[True, True, True, False], [True, True, True, True]])

    edge_index, relation = encoder._build_sparse_map_edges(
        endpoint_pose=endpoint_pose,
        map_position=map_position,
        map_orientation=map_orientation,
        map_valid_mask=map_valid_mask,
        valid_mask=valid_mask,
    )
    actual = set(map(tuple, edge_index.t().tolist()))

    expected: set[tuple[int, int]] = set()
    for b in range(bsz):
        for k in range(n_rollout):
            for e in range(n_endpoint):
                for n in range(n_agent):
                    if not bool(valid_mask[b, n]):
                        continue
                    for m in range(n_map):
                        if not bool(map_valid_mask[b, m]):
                            continue
                        dist = torch.linalg.vector_norm(
                            map_position[b, m] - endpoint_pose[b, k, e, n, :2]
                        )
                        if float(dist) <= 2.0:
                            expected.add(
                                (
                                    b * n_map + m,
                                    _target_index(
                                        batch=b,
                                        rollout=k,
                                        endpoint=e,
                                        agent=n,
                                        n_rollout=n_rollout,
                                        n_endpoint=n_endpoint,
                                        n_agent=n_agent,
                                    ),
                                )
                            )

    assert actual == expected
    assert relation.shape == (len(expected), 3)
    source_scene = edge_index[0] // n_map
    target_scene = edge_index[1] // (n_rollout * n_endpoint * n_agent)
    assert torch.equal(source_scene, target_scene)
    assert torch.all(relation[:, 0] <= 1.0 + 1.0e-6)


def test_gan_discriminator_sparse_map_compliance_backward() -> None:
    torch.manual_seed(7)
    bsz, n_rollout, n_step, n_agent, n_map, hidden_dim = 2, 2, 20, 3, 5, 16
    discriminator = SelfForcedGANDiscriminator(
        hidden_dim=hidden_dim,
        n_rollout=n_rollout,
        n_step=n_step,
        num_attention_heads=4,
        map_radius_m=5.0,
        map_query_chunk_size=1,
        map_sender_chunk_size=2,
    )
    rollout_pose = torch.randn(bsz, n_rollout, n_step, n_agent, 4)
    yaw = torch.randn(bsz, n_rollout, n_step, n_agent)
    rollout_pose[..., 2] = torch.cos(yaw)
    rollout_pose[..., 3] = torch.sin(yaw)
    current_pose = rollout_pose[:, 0, 0].detach().clone()
    agent_type = torch.zeros(bsz, n_agent, dtype=torch.long)
    valid_mask = torch.tensor([[True, True, False], [True, True, True]])
    agent_context = torch.randn(bsz, n_agent, hidden_dim)
    map_context = torch.randn(bsz, n_map, hidden_dim)
    map_position = torch.randn(bsz, n_map, 2)
    map_orientation = torch.zeros(bsz, n_map)
    map_valid_mask = torch.tensor(
        [[True, True, False, True, True], [True, False, True, True, True]]
    )

    logit = discriminator(
        rollout_pose,
        current_pose=current_pose,
        agent_type=agent_type,
        valid_mask=valid_mask,
        agent_context=agent_context,
        map_context=map_context,
        map_position=map_position,
        map_orientation=map_orientation,
        map_valid_mask=map_valid_mask,
    )
    assert logit.shape == (bsz, 1)
    assert torch.isfinite(logit).all()
    loss = logit.square().mean()
    loss.backward()
    grad = discriminator.map_compliance_encoder.attention.query.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
