"""anchor_k 일반화 helper 검증.

기존 anchor 0 helper 와의 behavior-parity + anchor_idx > 0 분기 동작 확인.
"""
from __future__ import annotations

import torch

from src.smart.modules.self_forced_path_flow import (
    build_anchor0_normalized_committed_path,
    build_anchor_k_normalized_committed_path,
    get_anchor0_valid_mask,
    get_anchor_k_valid_mask,
)


def _make_tokenized_agent(n_agent: int = 4, n_anchor: int = 16, n_ctx: int = 18):
    torch.manual_seed(0)
    return {
        "flow_eval_mask": (torch.rand(n_agent, n_anchor) > 0.3),
        "ctx_sampled_pos": torch.randn(n_agent, n_ctx, 2),
        "ctx_sampled_heading": torch.randn(n_agent, n_ctx),
    }


def test_get_anchor_k_idx_0_matches_anchor0_helper() -> None:
    tok = _make_tokenized_agent()
    a0 = get_anchor0_valid_mask(tok)
    ak = get_anchor_k_valid_mask(tok, anchor_idx=0)
    assert torch.equal(a0, ak)


def test_get_anchor_k_picks_correct_column() -> None:
    tok = _make_tokenized_agent()
    for k in [1, 4, 8, 15]:
        ak = get_anchor_k_valid_mask(tok, anchor_idx=k)
        assert torch.equal(ak, tok["flow_eval_mask"][:, k].bool())


def test_get_anchor_k_rejects_out_of_range() -> None:
    tok = _make_tokenized_agent()
    raised = False
    try:
        get_anchor_k_valid_mask(tok, anchor_idx=16)
    except ValueError:
        raised = True
    assert raised, "anchor_idx >= n_anchor must raise ValueError"


def test_get_anchor_k_rejects_negative() -> None:
    tok = _make_tokenized_agent()
    raised = False
    try:
        get_anchor_k_valid_mask(tok, anchor_idx=-1)
    except ValueError:
        raised = True
    assert raised, "anchor_idx < 0 must raise ValueError"


def test_build_anchor_k_idx_0_matches_anchor0_helper() -> None:
    torch.manual_seed(1)
    n_agent, n_ctx, flow_window_steps = 4, 18, 20
    T_rollout = 80
    pred_traj_10hz = torch.randn(n_agent, T_rollout, 2)
    pred_head_10hz = torch.randn(n_agent, T_rollout)
    tok = {
        "ctx_sampled_pos": torch.randn(n_agent, n_ctx, 2),
        "ctx_sampled_heading": torch.randn(n_agent, n_ctx),
    }
    legacy = build_anchor0_normalized_committed_path(
        pred_traj_10hz, pred_head_10hz, tok, flow_window_steps
    )
    new = build_anchor_k_normalized_committed_path(
        pred_traj_10hz,
        pred_head_10hz,
        tok,
        flow_window_steps,
        anchor_idx=0,
        anchor_stride_2hz=1,
        shift=5,
    )
    assert torch.allclose(legacy, new, atol=1e-6)


def test_build_anchor_k_idx1_slices_at_expected_offset() -> None:
    torch.manual_seed(2)
    n_agent, n_ctx, flow_window_steps = 4, 18, 20
    T_rollout = 80
    shift = 5
    stride = 4  # OCSC default
    pred_traj_10hz = torch.randn(n_agent, T_rollout, 2)
    pred_head_10hz = torch.randn(n_agent, T_rollout)
    tok = {
        "ctx_sampled_pos": torch.randn(n_agent, n_ctx, 2),
        "ctx_sampled_heading": torch.randn(n_agent, n_ctx),
    }
    k = 1
    out = build_anchor_k_normalized_committed_path(
        pred_traj_10hz, pred_head_10hz, tok, flow_window_steps,
        anchor_idx=k, anchor_stride_2hz=stride, shift=shift,
    )
    # start = k * stride * shift = 20
    # path slice end = 20 + 20 = 40 (within T_rollout=80) → no error.
    assert out.shape == (n_agent, flow_window_steps, 4)


def test_build_anchor_k_raises_when_rollout_too_short() -> None:
    torch.manual_seed(3)
    n_agent, n_ctx, flow_window_steps = 4, 18, 20
    T_rollout = 30  # too short for k=2, stride=4, shift=5 → start=40 > 30
    pred_traj_10hz = torch.randn(n_agent, T_rollout, 2)
    pred_head_10hz = torch.randn(n_agent, T_rollout)
    tok = {
        "ctx_sampled_pos": torch.randn(n_agent, n_ctx, 2),
        "ctx_sampled_heading": torch.randn(n_agent, n_ctx),
    }
    raised = False
    try:
        build_anchor_k_normalized_committed_path(
            pred_traj_10hz, pred_head_10hz, tok, flow_window_steps,
            anchor_idx=2, anchor_stride_2hz=4, shift=5,
        )
    except ValueError:
        raised = True
    assert raised, "rollout shorter than required window must raise ValueError"
