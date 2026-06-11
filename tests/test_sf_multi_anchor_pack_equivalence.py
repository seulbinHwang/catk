from __future__ import annotations

from types import SimpleNamespace

import torch

from src.smart.model.smart_flow import SMARTFlow
from src.smart.modules.self_forced_path_flow import build_anchor0_normalized_committed_path


def _build_stub_model(flow_window_steps: int = 20) -> SimpleNamespace:
    return SimpleNamespace(
        flow_window_steps=flow_window_steps,
        use_kinematic_control_flow=False,
    )


def _build_rollout_and_tokens(
    num_agent: int = 4,
    num_anchor: int = 16,
    t_rollout: int = 20,
    num_replicas: int = 1,
) -> tuple[dict, dict]:
    torch.manual_seed(0)
    rollout = {
        "pred_traj_10hz": torch.randn(num_replicas * num_agent, t_rollout, 2),
        "pred_head_10hz": torch.randn(num_replicas * num_agent, t_rollout),
    }
    tokenized_agent = {
        "ctx_sampled_pos": torch.randn(num_agent, 18, 2),
        "ctx_sampled_heading": torch.randn(num_agent, 18),
        "flow_eval_mask": torch.rand(num_agent, num_anchor) > 0.3,
        "type": torch.zeros(num_agent, dtype=torch.long),
    }
    return rollout, tokenized_agent


def test_pack_committed_rollout_anchor0_matches_legacy_path() -> None:
    """stride가 꺼진 기본 경로([0])는 기존 anchor0 구현과 수치 동일해야 합니다."""
    stub = _build_stub_model()
    rollout, tokenized_agent = _build_rollout_and_tokens()

    packed, anchor_mask = SMARTFlow._pack_self_forced_committed_rollout(
        stub,
        rollout=rollout,
        tokenized_agent=tokenized_agent,
        anchor_offsets=[0],
    )

    legacy_mask = tokenized_agent["flow_eval_mask"][:, 0].bool()
    legacy_norm = build_anchor0_normalized_committed_path(
        pred_traj_10hz=rollout["pred_traj_10hz"],
        pred_head_10hz=rollout["pred_head_10hz"],
        tokenized_agent=tokenized_agent,
        flow_window_steps=stub.flow_window_steps,
    )
    legacy_packed = legacy_norm[legacy_mask]

    assert anchor_mask.shape == (4, 1)
    assert torch.equal(anchor_mask[:, 0], legacy_mask)
    torch.testing.assert_close(packed, legacy_packed)


def test_pack_committed_rollout_multi_anchor_rows_use_own_origin() -> None:
    """multi-anchor 복제 rollout의 각 (anchor, agent) 행은 자기 anchor 원점으로
    정규화되어 anchor-major 순서로 packing되어야 합니다."""
    stub = _build_stub_model()
    offsets = [0, 4, 8]
    num_agent = 4
    rollout, tokenized_agent = _build_rollout_and_tokens(
        num_agent=num_agent,
        num_replicas=len(offsets),
    )

    packed, anchor_mask = SMARTFlow._pack_self_forced_committed_rollout(
        stub,
        rollout=rollout,
        tokenized_agent=tokenized_agent,
        anchor_offsets=offsets,
    )
    assert anchor_mask.shape == (num_agent, len(offsets))
    assert torch.equal(anchor_mask, tokenized_agent["flow_eval_mask"][:, offsets])

    # 기대값: 복제 블록 a의 agent i 행을 ctx slot 1+offset_a 원점으로 정규화.
    expected_rows = []
    for a, offset in enumerate(offsets):
        block = slice(a * num_agent, (a + 1) * num_agent)
        ctx_slot_agent = {
            "ctx_sampled_pos": tokenized_agent["ctx_sampled_pos"][:, [0, 1 + offset]],
            "ctx_sampled_heading": tokenized_agent["ctx_sampled_heading"][:, [0, 1 + offset]],
        }
        block_norm = build_anchor0_normalized_committed_path(
            pred_traj_10hz=rollout["pred_traj_10hz"][block],
            pred_head_10hz=rollout["pred_head_10hz"][block],
            tokenized_agent=ctx_slot_agent,
            flow_window_steps=stub.flow_window_steps,
        )
        expected_rows.append(block_norm[anchor_mask[:, a]])
    expected_packed = torch.cat(expected_rows, dim=0)
    torch.testing.assert_close(packed, expected_packed)


def test_get_self_forced_anchor_offsets_rules() -> None:
    """stride 설정과 GT 길이에 따라 offsets가 결정되어야 합니다."""
    tokenized_agent = {
        "flow_eval_mask": torch.ones(2, 16, dtype=torch.bool),
        "ctx_sampled_pos": torch.zeros(2, 18, 2),
    }

    def _stub(stride: int) -> SimpleNamespace:
        return SimpleNamespace(
            self_forced_start_anchor_stride=stride,
            token_processor=SimpleNamespace(shift=5),
            flow_window_steps=20,
        )

    assert SMARTFlow._get_self_forced_anchor_offsets(_stub(0), tokenized_agent) == [0]
    assert SMARTFlow._get_self_forced_anchor_offsets(_stub(4), tokenized_agent) == [0, 4, 8, 12]
    assert SMARTFlow._get_self_forced_anchor_offsets(_stub(6), tokenized_agent) == [0, 6, 12]

    # flow_eval_mask가 없으면(비-eval 토큰) stride와 무관하게 기존 동작입니다.
    assert (
        SMARTFlow._get_self_forced_anchor_offsets(
            _stub(4),
            {"ctx_sampled_pos": torch.zeros(2, 18, 2)},
        )
        == [0]
    )
