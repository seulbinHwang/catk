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
    """``r_pt2a_emb`` 자리만 메우는 더미 — nn.Module 이 아닌 plain class 로 둬야
    ``SMARTAgentEncoder.__new__`` 이후 ``Module.__init__`` 가 호출되지 않은
    상태에서 attribute 로 할당될 수 있습니다.
    """

    def __call__(self, continuous_inputs, categorical_embs=None):
        return continuous_inputs


def _make_encoder() -> SMARTAgentEncoder:
    encoder = SMARTAgentEncoder.__new__(SMARTAgentEncoder)
    encoder.pl2a_radius = 100.0
    encoder.shift = 5
    encoder.r_pt2a_emb = _IdentityRelation()
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
    """flow_agent_decoder 의 build_map2agent_edge 호출 입력을 그대로 흉내냅니다.

    Args:
        agents_per_scene: 장면별 agent 수.
        maps_per_scene: 장면별 map polygon 수.
        num_steps: coarse step 개수 (caller 가 ``batch.repeat(n_step)`` 으로 만듭니다).
        scene_spacing_m: 장면 중심 사이 간격(m).
        within_scene_std_m: 장면 중심 주변 분산(m). pl2a_radius (=100m) 안에 들도록 작게 둡니다.
        device: 텐서 장치.
        seed: 결정적 재현용 시드.
    """
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
    # caller 와 동일하게 step 축이 1번 차원이 되도록 정렬합니다.
    pos_a = agent_pos_per_step.transpose(0, 1).contiguous()  # [n_agent, n_step, 2]
    head_a = torch.zeros(num_agent, num_steps)
    head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
    mask = torch.ones(num_agent, num_steps, dtype=torch.bool)

    pos_pl = scene_centers[map_batch] + torch.randn(num_map, 2, generator=rng) * 5.0
    orient_pl = torch.zeros(num_map)

    batch_s = agent_batch.repeat(num_steps)  # production 패턴

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
        "_num_map": num_map,
        "_num_steps": num_steps,
    }


def _expected_edge_count(inputs: dict[str, torch.Tensor]) -> int:
    """brute-force: 같은 scene + radius 안 map-agent 쌍 모두 세기."""
    num_agent = inputs["_num_agent"]
    num_steps = inputs["_num_steps"]
    pos_a = inputs["pos_a"]
    pos_pl = inputs["pos_pl"]
    batch_pl = inputs["batch_pl"]
    batch_s = inputs["batch_s"]
    pos_s = pos_a.transpose(0, 1).reshape(num_steps * num_agent, 2)

    count = 0
    for y_idx in range(pos_pl.shape[0]):
        for x_idx in range(pos_s.shape[0]):
            if batch_pl[y_idx].item() != batch_s[x_idx].item():
                continue
            dist = (pos_pl[y_idx] - pos_s[x_idx]).norm().item()
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
    # cross-scene 엣지가 섞이지 않았는지 같이 검증합니다.
    src_batch = inputs["batch_pl"][edge_index[0]]
    dst_batch = inputs["batch_s"][edge_index[1]]
    cross_scene = int((src_batch != dst_batch).sum().item())
    return edge_index.shape[1], _expected_edge_count(inputs), cross_scene


def test_map2agent_edge_no_silent_drop_cpu() -> None:
    edges, expected, cross_scene = _run_and_count(torch.device("cpu"))
    assert cross_scene == 0, (
        f"cross-scene map-agent edge 발생: {cross_scene}"
    )
    assert edges == expected, (
        f"CPU: 생성된 edge 수 {edges} 가 기대값 {expected} 과 다름 — silent drop"
    )


def test_map2agent_edge_no_silent_drop_gpu() -> None:
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA unavailable")
    edges, expected, cross_scene = _run_and_count(torch.device("cuda"))
    assert cross_scene == 0, (
        f"cross-scene map-agent edge 발생: {cross_scene}"
    )
    assert edges == expected, (
        f"GPU: 생성된 edge 수 {edges} 가 기대값 {expected} 과 다름 — silent drop"
    )
