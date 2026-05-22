"""build_map2agent_edge 의 batch 정렬 누락 silent 버그 회귀 테스트.

caller 는 ``batch_s = tokenized_agent["batch"].repeat(n_step)`` 으로 step 마다
같은 scene 번호 묶음을 반복합니다. step 사이마다 큰 scene 번호에서 작은
값으로 떨어지기 때문에 ``torch_cluster.radius`` 의 sorted-batch silent 가정이
깨져 같은 scene 안의 map-agent edge 가 일부 누락됩니다. 이 테스트는 production
과 같은 패킹·위치 분포에서 expected edge 가 모두 생성되는지를 CPU/GPU
양쪽으로 확인합니다.
"""

from __future__ import annotations

import torch

from src.smart.modules.agent_encoder import SMARTAgentEncoder


class _IdentityRelation:
    """``r_pt2a_emb`` 자리만 메우는 더미입니다."""

    def __call__(self, continuous_inputs, categorical_embs=None):
        return continuous_inputs


class _ZeroLightEmbedding:
    def __call__(self, light_type):
        return torch.zeros(light_type.numel(), 3, device=light_type.device)


class _LightTimeIdentity:
    def __call__(self, continuous_inputs, categorical_embs=None):
        out = continuous_inputs.expand(-1, 3)
        if categorical_embs is not None:
            for emb in categorical_embs:
                out = out + emb
        return out


def _make_encoder() -> SMARTAgentEncoder:
    encoder = SMARTAgentEncoder.__new__(SMARTAgentEncoder)
    encoder.pl2a_radius = 100.0
    encoder.shift = 5
    encoder.r_pt2a_emb = _IdentityRelation()
    encoder.light_pl2a_emb = _ZeroLightEmbedding()
    encoder.light_time_pl2a_emb = _LightTimeIdentity()
    return encoder


def _build_map2agent_inputs(
    *,
    agents_per_scene: list[int],
    maps_per_scene: list[int],
    num_steps: int,
    scene_spacing_m: float,
    within_scene_std_m: float,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    rng = torch.Generator()
    rng.manual_seed(int(seed))

    agent_batch = torch.cat(
        [
            torch.full((count,), idx, dtype=torch.long)
            for idx, count in enumerate(agents_per_scene)
        ]
    )
    map_batch = torch.cat(
        [
            torch.full((count,), idx, dtype=torch.long)
            for idx, count in enumerate(maps_per_scene)
        ]
    )

    num_agent = agent_batch.numel()
    num_map = map_batch.numel()
    scene_centers = torch.stack(
        [
            torch.arange(len(agents_per_scene), dtype=torch.float32) * scene_spacing_m,
            torch.zeros(len(agents_per_scene), dtype=torch.float32),
        ],
        dim=-1,
    )

    agent_pos_per_step = (
        scene_centers[agent_batch].unsqueeze(0).expand(num_steps, num_agent, 2).clone()
        + torch.randn(num_steps, num_agent, 2, generator=rng) * within_scene_std_m
    )
    pos_a = agent_pos_per_step.transpose(0, 1).contiguous()
    head_a = torch.zeros(num_agent, num_steps)
    head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
    mask = torch.ones(num_agent, num_steps, dtype=torch.bool)
    pos_pl = scene_centers[map_batch] + torch.randn(num_map, 2, generator=rng) * 5.0
    orient_pl = torch.zeros(num_map)
    batch_s = agent_batch.repeat(num_steps)

    return {
        "pos_pl": pos_pl.to(device=device),
        "orient_pl": orient_pl.to(device=device),
        "pos_a": pos_a.to(device=device),
        "head_a": head_a.to(device=device),
        "head_vector_a": head_vector_a.to(device=device),
        "mask": mask.to(device=device),
        "batch_s": batch_s.to(device=device),
        "batch_pl": map_batch.to(device=device),
        "_num_agent": num_agent,
        "_num_steps": num_steps,
    }


def _expected_edge_count(inputs: dict[str, torch.Tensor]) -> int:
    num_agent = inputs["_num_agent"]
    num_steps = inputs["_num_steps"]
    pos_a = inputs["pos_a"]
    pos_pl = inputs["pos_pl"]
    batch_pl = inputs["batch_pl"]
    batch_s = inputs["batch_s"]
    pos_s = pos_a.transpose(0, 1).reshape(num_steps * num_agent, 2)

    count = 0
    for map_idx in range(pos_pl.shape[0]):
        for agent_idx in range(pos_s.shape[0]):
            if batch_pl[map_idx].item() != batch_s[agent_idx].item():
                continue
            dist = (pos_pl[map_idx] - pos_s[agent_idx]).norm().item()
            if dist <= 100.0:
                count += 1
    return count


def _run_and_count(device: torch.device) -> tuple[int, int, int]:
    encoder = _make_encoder()
    inputs = _build_map2agent_inputs(
        agents_per_scene=[6, 4, 8, 3],
        maps_per_scene=[3, 2, 5, 2],
        num_steps=4,
        scene_spacing_m=1000.0,
        within_scene_std_m=20.0,
        device=device,
        seed=11,
    )

    edge_index, _ = encoder.build_map2agent_edge(
        pos_pl=inputs["pos_pl"],
        orient_pl=inputs["orient_pl"],
        pos_a=inputs["pos_a"],
        head_a=inputs["head_a"],
        head_vector_a=inputs["head_vector_a"],
        mask=inputs["mask"],
        batch_s=inputs["batch_s"],
        batch_pl=inputs["batch_pl"],
    )
    src_batch = inputs["batch_pl"][edge_index[0]]
    dst_batch = inputs["batch_s"][edge_index[1]]
    cross_scene = int((src_batch != dst_batch).sum().item())
    return edge_index.shape[1], _expected_edge_count(inputs), cross_scene


def test_map2agent_edge_no_silent_drop_cpu() -> None:
    edges, expected, cross_scene = _run_and_count(torch.device("cpu"))
    assert cross_scene == 0, f"cross-scene map-agent edge 발생: {cross_scene}"
    assert edges == expected, (
        f"CPU: 생성된 edge 수 {edges} 가 기대값 {expected} 과 다름 — silent drop"
    )


def test_map2agent_edge_no_silent_drop_gpu() -> None:
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA unavailable")
    edges, expected, cross_scene = _run_and_count(torch.device("cuda"))
    assert cross_scene == 0, f"cross-scene map-agent edge 발생: {cross_scene}"
    assert edges == expected, (
        f"GPU: 생성된 edge 수 {edges} 가 기대값 {expected} 과 다름 — silent drop"
    )


def test_map2agent_edge_masks_stale_time_for_no_signal_lanes() -> None:
    encoder = _make_encoder()
    pos_pl = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    orient_pl = torch.zeros(2)
    pos_a = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    head_a = torch.zeros(1, 2)
    head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
    mask = torch.ones(1, 2, dtype=torch.bool)
    batch_s = torch.zeros(2, dtype=torch.long)
    batch_pl = torch.zeros(2, dtype=torch.long)
    light_type = torch.tensor([0, 3], dtype=torch.long)
    light_time_delta_norm = torch.tensor([[0.25, 0.5]])

    edge_index, relation = encoder.build_map2agent_edge(
        pos_pl=pos_pl,
        orient_pl=orient_pl,
        pos_a=pos_a,
        head_a=head_a,
        head_vector_a=head_vector_a,
        mask=mask,
        batch_s=batch_s,
        batch_pl=batch_pl,
        light_type=light_type,
        light_time_delta_norm=light_time_delta_norm,
    )

    base_relation = relation.clone()
    base_relation[:, 0] -= torch.linalg.vector_norm(
        pos_pl[edge_index[0]] - pos_a.reshape(-1, 2)[edge_index[1]],
        dim=-1,
    )
    for edge_idx in range(edge_index.shape[1]):
        map_idx = int(edge_index[0, edge_idx].item())
        dst_idx = int(edge_index[1, edge_idx].item())
        expected_stale = 0.0 if map_idx == 0 else float(light_time_delta_norm.reshape(-1)[dst_idx])
        torch.testing.assert_close(base_relation[edge_idx], torch.full((3,), expected_stale))
