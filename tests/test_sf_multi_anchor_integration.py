from __future__ import annotations

import torch

from src.smart.modules.self_forced_multi_anchor import (
    replicate_tokenized_agent_for_anchors,
)
from src.smart.tokens.flow_token_processor import FlowTokenProcessor


class _FakeSceneData:
    """token processor가 읽는 최소 필드만 가진 가짜 장면 배치입니다."""

    def __init__(self, agent: dict, num_graphs: int) -> None:
        self._stores = {"agent": agent}
        self.num_graphs = num_graphs

    def __getitem__(self, key: str) -> dict:
        return self._stores[key]


def _build_processor() -> FlowTokenProcessor:
    processor = FlowTokenProcessor(
        map_token_file="map_traj_token5.pkl",
        agent_token_file="agent_vocab_555_s2.pkl",
        flow_window_steps=20,
    )
    processor.eval()
    return processor


def _build_scene(num_agent: int = 5, num_step: int = 91, num_graphs: int = 2) -> _FakeSceneData:
    torch.manual_seed(0)
    # 직선 주행 궤적(에이전트마다 y 오프셋)으로 전체 step 유효한 장면을 만듭니다.
    pos = torch.zeros(num_agent, num_step, 3)
    pos[:, :, 0] = torch.arange(num_step, dtype=torch.float32).unsqueeze(0) * 0.5
    pos[:, :, 1] = torch.arange(num_agent, dtype=torch.float32).unsqueeze(1) * 4.0
    pos[:, :, 2] = 1.0
    velocity = torch.zeros(num_agent, num_step, 2)
    velocity[:, :, 0] = 5.0
    agent = {
        "valid_mask": torch.ones(num_agent, num_step, dtype=torch.bool),
        "heading": torch.zeros(num_agent, num_step),
        "position": pos,
        "velocity": velocity,
        "type": torch.tensor([0, 1, 2, 0, 0]),
        "shape": torch.tensor([[4.8, 2.0, 1.5]] * num_agent),
        "role": torch.zeros(num_agent, 3, dtype=torch.bool),
        "batch": torch.tensor([0, 0, 0, 1, 1]),
        "train_mask": torch.ones(num_agent, dtype=torch.bool),
    }
    return _FakeSceneData(agent=agent, num_graphs=num_graphs)


# closed-loop rollout 경로(_prepare_rollout_cache_impl + _rollout_from_cache_impl)가
# 복제 dict에서 읽는 필드 목록입니다. 누락되면 rollout에서 KeyError가 납니다.
_ROLLOUT_REQUIRED_KEYS = (
    "num_graphs",
    "batch",
    "type",
    "shape",
    "token_agent_shape",
    "trajectory_token_veh",
    "trajectory_token_ped",
    "trajectory_token_cyc",
    "token_bank_all_veh",
    "token_bank_all_ped",
    "token_bank_all_cyc",
    "gt_idx",
    "gt_pos",
    "gt_heading",
    "valid_mask",
    "gt_pos_raw",
    "gt_head_raw",
    "gt_valid_raw",
    "gt_z_raw",
    "rollout_init_fine_pos_history",
    "rollout_init_fine_head_history",
    "rollout_init_fine_valid_history",
    "rollout_init_fine_pos_pair",
    "rollout_init_fine_head_pair",
    "rollout_init_fine_valid_pair",
)


def test_eval_forward_produces_sf_anchor_fields_and_replication_is_complete() -> None:
    """실제 eval tokenize 경로의 출력으로 복제 dict를 만들었을 때, rollout이
    읽는 모든 필드가 올바른 shape으로 채워져야 합니다."""
    processor = _build_processor()
    data = _build_scene()
    tokenized_agent, processed_agent = processor.tokenize_agent(data, return_preprocessed=True)
    tokenized_agent["num_graphs"] = data.num_graphs
    tokenized_agent = processor._build_flow_targets(
        data=data,
        tokenized_agent=tokenized_agent,
        processed_agent=processed_agent,
    )

    num_agent = 5
    assert tokenized_agent["sf_anchor_fine_pos_history"].shape == (num_agent, 16, 6, 2)
    assert tokenized_agent["sf_anchor_z"].shape == (num_agent, 16)

    # anchor 0 fine history는 기존 anchor0 전용 필드와 정확히 같아야 합니다.
    torch.testing.assert_close(
        tokenized_agent["sf_anchor_fine_pos_history"][:, 0],
        tokenized_agent["rollout_init_fine_pos_history"],
    )
    torch.testing.assert_close(
        tokenized_agent["sf_anchor_z"][:, 0],
        tokenized_agent["gt_z_raw"],
    )

    offsets = [0, 4, 8, 12]
    replicated = replicate_tokenized_agent_for_anchors(
        tokenized_agent=tokenized_agent,
        anchor_offsets=offsets,
    )
    for key in _ROLLOUT_REQUIRED_KEYS:
        assert key in replicated, f"replicated dict is missing rollout key: {key}"

    n_replica_agent = num_agent * len(offsets)
    assert replicated["batch"].shape == (n_replica_agent,)
    assert replicated["num_graphs"] == data.num_graphs * len(offsets)
    assert int(replicated["batch"].max()) == data.num_graphs * len(offsets) - 1
    assert replicated["gt_pos"].shape[0] == n_replica_agent
    assert replicated["rollout_init_fine_pos_history"].shape == (n_replica_agent, 6, 2)

    # 복제 블록 0(anchor 0)은 원본과 완전히 같아야 합니다(기존 경로 보존).
    torch.testing.assert_close(replicated["gt_pos"][:num_agent], tokenized_agent["gt_pos"])
    assert torch.equal(replicated["valid_mask"][:num_agent], tokenized_agent["valid_mask"])
    torch.testing.assert_close(
        replicated["rollout_init_fine_pos_history"][:num_agent],
        tokenized_agent["rollout_init_fine_pos_history"],
    )

    # anchor k 복제 블록의 첫 토큰 window는 원본 토큰 k..k+1과 같아야 합니다
    # (rollout cache는 [:, :2]만 사용합니다).
    for a, offset in enumerate(offsets):
        block = slice(a * num_agent, (a + 1) * num_agent)
        torch.testing.assert_close(
            replicated["gt_pos"][block][:, :2],
            tokenized_agent["gt_pos"][:, offset : offset + 2],
        )

    # flow_eval_mask 자체는 복제 dict에 필요 없고(원본에서 packing), 원본은
    # full-future 규칙으로 anchor 12(7초)까지 유효해야 합니다.
    assert bool(tokenized_agent["flow_eval_mask"][:, [0, 4, 8, 12]].all())
    assert not bool(tokenized_agent["flow_eval_mask"][:, 13:].any())
