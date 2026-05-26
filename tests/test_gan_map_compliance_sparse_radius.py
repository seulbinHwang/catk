from __future__ import annotations

import copy

import torch

from src.smart.modules.self_forced_gan_critic import (
    InteractionEncoder,
    MapComplianceEncoder,
    RadiusAttentionLayer,
    SelfForcedGANDiscriminator,
    SparseRadiusAttentionLayer,
    add_rollout_pose_perturbation,
    relativistic_discriminator_loss,
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


def test_map_compliance_caps_map_tokens_per_endpoint_not_endpoints_per_map() -> None:
    encoder = MapComplianceEncoder(hidden_dim=8, radius_m=2.0, num_heads=2)

    endpoint_pose = torch.zeros(1, 1, 1, 301, 4)
    endpoint_pose[..., 2] = 1.0
    map_position = torch.zeros(1, 1, 2)
    map_orientation = torch.zeros(1, 1)
    valid_mask = torch.ones(1, 301, dtype=torch.bool)
    map_valid_mask = torch.ones(1, 1, dtype=torch.bool)

    edge_index, _ = encoder._build_sparse_map_edges(
        endpoint_pose=endpoint_pose,
        map_position=map_position,
        map_orientation=map_orientation,
        map_valid_mask=map_valid_mask,
        valid_mask=valid_mask,
    )

    assert edge_index.shape[1] == 301
    assert torch.equal(edge_index[1].sort().values, torch.arange(301))

    endpoint_pose = torch.zeros(1, 1, 1, 1, 4)
    endpoint_pose[..., 2] = 1.0
    map_position = torch.zeros(1, 305, 2)
    map_orientation = torch.zeros(1, 305)
    valid_mask = torch.ones(1, 1, dtype=torch.bool)
    map_valid_mask = torch.ones(1, 305, dtype=torch.bool)

    edge_index, _ = encoder._build_sparse_map_edges(
        endpoint_pose=endpoint_pose,
        map_position=map_position,
        map_orientation=map_orientation,
        map_valid_mask=map_valid_mask,
        valid_mask=valid_mask,
    )

    assert edge_index.shape[1] == 100
    assert torch.equal(edge_index[1], torch.zeros(100, dtype=torch.long))


def test_interaction_builds_only_same_endpoint_radius_edges() -> None:
    encoder = InteractionEncoder(hidden_dim=8, radius_m=2.0, num_heads=2)
    bsz, n_rollout, n_endpoint, n_agent = 2, 2, 2, 4

    endpoint_pose = torch.zeros(bsz, n_rollout, n_endpoint, n_agent, 4)
    endpoint_pose[..., 2] = 1.0
    endpoint_pose[0, :, :, 0, :2] = torch.tensor([0.0, 0.0])
    endpoint_pose[0, :, :, 1, :2] = torch.tensor([1.0, 0.0])
    endpoint_pose[0, :, :, 2, :2] = torch.tensor([10.0, 0.0])
    endpoint_pose[0, :, :, 3, :2] = torch.tensor([0.0, 1.5])
    endpoint_pose[1, :, :, 0, :2] = torch.tensor([100.0, 0.0])
    endpoint_pose[1, :, :, 1, :2] = torch.tensor([101.0, 0.0])
    endpoint_pose[1, :, :, 2, :2] = torch.tensor([200.0, 0.0])
    endpoint_pose[1, :, :, 3, :2] = torch.tensor([100.0, 1.0])
    # Same absolute coordinates in a different rollout/endpoint must not connect.
    endpoint_pose[0, 1, 1, 2, :2] = torch.tensor([1.0, 0.0])

    valid_mask = torch.tensor([[True, True, False, True], [True, True, False, True]])

    edge_index, relation = encoder._build_sparse_interaction_edges(
        endpoint_pose=endpoint_pose,
        valid_mask=valid_mask,
    )
    actual = set(map(tuple, edge_index.t().tolist()))

    expected: set[tuple[int, int]] = set()
    for b in range(bsz):
        for k in range(n_rollout):
            for e in range(n_endpoint):
                for src_agent in range(n_agent):
                    if not bool(valid_mask[b, src_agent]):
                        continue
                    for dst_agent in range(n_agent):
                        if src_agent == dst_agent or not bool(valid_mask[b, dst_agent]):
                            continue
                        src = _target_index(
                            batch=b,
                            rollout=k,
                            endpoint=e,
                            agent=src_agent,
                            n_rollout=n_rollout,
                            n_endpoint=n_endpoint,
                            n_agent=n_agent,
                        )
                        dst = _target_index(
                            batch=b,
                            rollout=k,
                            endpoint=e,
                            agent=dst_agent,
                            n_rollout=n_rollout,
                            n_endpoint=n_endpoint,
                            n_agent=n_agent,
                        )
                        dist = torch.linalg.vector_norm(
                            endpoint_pose.reshape(-1, 4)[src, :2]
                            - endpoint_pose.reshape(-1, 4)[dst, :2]
                        )
                        if float(dist) <= 2.0:
                            expected.add((src, dst))

    assert actual == expected
    assert relation.shape == (len(expected), 3)
    assert not torch.any(edge_index[0] == edge_index[1])
    assert torch.equal(edge_index[0] // n_agent, edge_index[1] // n_agent)
    assert torch.all(relation[:, 0] <= 1.0 + 1.0e-6)


def test_sparse_attention_parameter_count_matches_dense_attention() -> None:
    hidden_dim = 16
    kwargs = {"num_heads": 4, "relation_dim": 3}
    dense = RadiusAttentionLayer(hidden_dim, **kwargs, sender_chunk_size=8)
    sparse = SparseRadiusAttentionLayer(hidden_dim, **kwargs)
    dense_params = sum(parameter.numel() for parameter in dense.parameters())
    sparse_params = sum(parameter.numel() for parameter in sparse.parameters())
    assert sparse_params == dense_params


def test_gan_discriminator_sparse_map_compliance_backward() -> None:
    torch.manual_seed(7)
    bsz, n_rollout, n_step, n_agent, n_map, hidden_dim = 2, 2, 20, 3, 5, 16
    discriminator = SelfForcedGANDiscriminator(
        hidden_dim=hidden_dim,
        n_rollout=n_rollout,
        n_step=n_step,
        num_attention_heads=4,
        map_radius_m=5.0,
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
    interaction_grad = discriminator.interaction_encoder.attention.query.weight.grad
    assert interaction_grad is not None
    assert torch.isfinite(interaction_grad).all()


def test_gan_discriminator_reused_base_logit_matches_recomputed_regularizer_gradient() -> None:
    torch.manual_seed(11)
    bsz, n_rollout, n_step, n_agent, n_map, hidden_dim = 1, 2, 20, 3, 5, 16
    discriminator = SelfForcedGANDiscriminator(
        hidden_dim=hidden_dim,
        n_rollout=n_rollout,
        n_step=n_step,
        num_attention_heads=4,
        map_radius_m=10.0,
    )
    recompute_model = copy.deepcopy(discriminator)
    reuse_model = copy.deepcopy(discriminator)

    real_pose = torch.randn(bsz, n_rollout, n_step, n_agent, 4)
    fake_pose = torch.randn(bsz, n_rollout, n_step, n_agent, 4)
    for pose in (real_pose, fake_pose):
        yaw = torch.randn(bsz, n_rollout, n_step, n_agent)
        pose[..., 2] = torch.cos(yaw)
        pose[..., 3] = torch.sin(yaw)
    current_pose = real_pose[:, 0, 0].detach().clone()
    context = {
        "current_pose": current_pose,
        "agent_type": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "valid_mask": torch.tensor([[True, True, True]]),
        "agent_context": torch.randn(bsz, n_agent, hidden_dim),
        "map_context": torch.randn(bsz, n_map, hidden_dim),
        "map_position": torch.randn(bsz, n_map, 2),
        "map_orientation": torch.zeros(bsz, n_map),
        "map_valid_mask": torch.ones(bsz, n_map, dtype=torch.bool),
    }

    def forward(model: SelfForcedGANDiscriminator, pose: torch.Tensor) -> torch.Tensor:
        return model(pose, **context)

    def finite_difference(
        model: SelfForcedGANDiscriminator,
        pose: torch.Tensor,
        *,
        base_logit: torch.Tensor | None,
        seed: int,
    ) -> torch.Tensor:
        torch.manual_seed(seed)
        pose = pose.detach()
        base = forward(model, pose) if base_logit is None else base_logit
        perturbed = add_rollout_pose_perturbation(
            pose,
            current_pose=context["current_pose"],
            agent_type=context["agent_type"],
            position_type_scale=model.position_type_scale,
            position_sigma=0.01,
            yaw_sigma=0.01,
        )
        perturbed_logit = forward(model, perturbed)
        return ((perturbed_logit - base) / 0.01).square().mean()

    recompute_real = forward(recompute_model, real_pose.detach())
    recompute_fake = forward(recompute_model, fake_pose.detach())
    recompute_loss = relativistic_discriminator_loss(recompute_real, recompute_fake)
    recompute_loss = recompute_loss + 0.1 * finite_difference(
        recompute_model,
        real_pose,
        base_logit=None,
        seed=101,
    )
    recompute_loss = recompute_loss + 0.1 * finite_difference(
        recompute_model,
        fake_pose,
        base_logit=None,
        seed=202,
    )
    recompute_loss.backward()

    reuse_real = forward(reuse_model, real_pose.detach())
    reuse_fake = forward(reuse_model, fake_pose.detach())
    reuse_loss = relativistic_discriminator_loss(reuse_real, reuse_fake)
    reuse_loss = reuse_loss + 0.1 * finite_difference(
        reuse_model,
        real_pose,
        base_logit=reuse_real,
        seed=101,
    )
    reuse_loss = reuse_loss + 0.1 * finite_difference(
        reuse_model,
        fake_pose,
        base_logit=reuse_fake,
        seed=202,
    )
    reuse_loss.backward()

    for recompute_param, reuse_param in zip(
        recompute_model.parameters(),
        reuse_model.parameters(),
    ):
        if recompute_param.grad is None:
            assert reuse_param.grad is None
            continue
        assert reuse_param.grad is not None
        assert torch.allclose(recompute_param.grad, reuse_param.grad, atol=1.0e-5, rtol=1.0e-4)
