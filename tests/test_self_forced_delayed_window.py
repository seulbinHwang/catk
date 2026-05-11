from __future__ import annotations

import math

import torch

from src.smart.modules.self_forced_delayed_window import (
    build_delayed_anchor0_tokenized_agent,
    build_delayed_normalized_committed_path,
    resolve_self_forced_delayed_window,
)


def test_resolve_self_forced_delayed_window_uses_fixed_four_epoch_stages() -> None:
    expected = {
        0: (0, 0, 4, 0.0, 2.0),
        3: (0, 0, 4, 0.0, 2.0),
        4: (1, 4, 8, 2.0, 4.0),
        7: (1, 4, 8, 2.0, 4.0),
        8: (2, 8, 12, 4.0, 6.0),
        12: (3, 12, 16, 6.0, 8.0),
        100: (3, 12, 16, 6.0, 8.0),
    }
    for epoch, values in expected.items():
        window = resolve_self_forced_delayed_window(
            current_epoch=epoch,
            start_epoch=0,
            flow_window_steps=20,
            commit_steps=5,
            stage_epochs=4,
            enabled=True,
        )
        assert (
            window.stage_index,
            window.skipped_blocks_2hz,
            window.rollout_steps_2hz,
            window.start_seconds,
            window.end_seconds,
        ) == values


def test_build_delayed_normalized_committed_path_uses_delayed_origin() -> None:
    # pred_traj_10hz: [N=1, T=40, 2]
    x = torch.arange(1, 41, dtype=torch.float32).view(1, 40, 1)
    pred_traj = torch.cat([x, torch.zeros_like(x)], dim=-1)
    pred_head = torch.zeros(1, 40)
    tokenized_agent = {
        "ctx_sampled_pos": torch.zeros(1, 14, 2),
        "ctx_sampled_heading": torch.zeros(1, 14),
        "_self_forced_delayed_start_step_10hz": 20,
    }

    path = build_delayed_normalized_committed_path(
        pred_traj_10hz=pred_traj,
        pred_head_10hz=pred_head,
        tokenized_agent=tokenized_agent,
        flow_window_steps=20,
        pos_scale_m=20.0,
    )

    # 2초 지점은 pred index 19의 x=20이고, 첫 학습 미래는 index 20의 x=21입니다.
    assert tuple(path.shape) == (1, 20, 4)
    assert math.isclose(float(path[0, 0, 0]), 1.0 / 20.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(path[0, -1, 0]), 20.0 / 20.0, rel_tol=0.0, abs_tol=1e-6)


def test_build_delayed_anchor0_tokenized_agent_replaces_current_context() -> None:
    pred_traj = torch.zeros(2, 40, 2)
    pred_traj[:, 19, 0] = torch.tensor([10.0, 20.0])
    pred_traj[:, 14, 0] = torch.tensor([7.0, 17.0])
    pred_head = torch.zeros(2, 40)
    pred_head[:, 19] = torch.tensor([0.5, 1.0])
    pred_head[:, 14] = torch.tensor([0.25, 0.75])
    tokenized_agent = {
        "ctx_sampled_pos": torch.zeros(2, 14, 2),
        "ctx_sampled_heading": torch.zeros(2, 14),
        "ctx_sampled_idx": torch.arange(28).view(2, 14),
        "ctx_valid": torch.ones(2, 14, dtype=torch.bool),
        "flow_eval_mask": torch.zeros(2, 13, dtype=torch.bool),
    }
    tokenized_agent["flow_eval_mask"][:, 4] = True
    window = resolve_self_forced_delayed_window(
        current_epoch=4,
        start_epoch=0,
        flow_window_steps=20,
        commit_steps=5,
        enabled=True,
    )

    delayed = build_delayed_anchor0_tokenized_agent(
        tokenized_agent=tokenized_agent,
        pred_traj_10hz=pred_traj,
        pred_head_10hz=pred_head,
        window=window,
        commit_steps=5,
    )

    assert delayed["flow_eval_mask"][:, 0].all()
    assert torch.allclose(delayed["ctx_sampled_pos"][:, 1], pred_traj[:, 19])
    assert torch.allclose(delayed["ctx_sampled_pos"][:, 0], pred_traj[:, 14])
    assert torch.allclose(delayed["ctx_sampled_heading"][:, 1], pred_head[:, 19])
    assert int(delayed["_self_forced_delayed_start_step_10hz"]) == 20


def test_self_forced_pack_returns_delayed_conditioning_agent() -> None:
    from types import MethodType, SimpleNamespace

    from src.smart.model.smart_flow import SMARTFlow

    model = object.__new__(SMARTFlow)
    object.__setattr__(model, "self_forced_delayed_window_enabled", True)
    object.__setattr__(model, "flow_window_steps", 20)
    object.__setattr__(model, "encoder", SimpleNamespace(agent_encoder=SimpleNamespace(shift=5)))

    window = resolve_self_forced_delayed_window(
        current_epoch=4,
        start_epoch=0,
        flow_window_steps=20,
        commit_steps=5,
        enabled=True,
    )
    object.__setattr__(
        model,
        "_resolve_self_forced_delayed_window",
        MethodType(lambda self: window, model),
    )

    pred_traj = torch.zeros(2, 40, 2)
    pred_traj[:, 19, 0] = torch.tensor([10.0, 20.0])
    pred_head = torch.zeros(2, 40)
    pred_head[:, 19] = torch.tensor([0.5, 1.0])
    rollout = {
        "pred_traj_10hz": pred_traj,
        "pred_head_10hz": pred_head,
    }
    tokenized_agent = {
        "ctx_sampled_pos": torch.zeros(2, 14, 2),
        "ctx_sampled_heading": torch.zeros(2, 14),
        "ctx_sampled_idx": torch.arange(28).view(2, 14),
        "ctx_valid": torch.ones(2, 14, dtype=torch.bool),
        "flow_eval_mask": torch.zeros(2, 13, dtype=torch.bool),
    }
    tokenized_agent["flow_eval_mask"][:, 4] = torch.tensor([True, False])

    path, anchor_mask, path_tokenized_agent = SMARTFlow._pack_self_forced_committed_rollout(
        model,
        rollout=rollout,
        tokenized_agent=tokenized_agent,
    )

    assert tuple(path.shape) == (1, 20, 4)
    assert anchor_mask.tolist() == [True, False]
    assert path_tokenized_agent is not tokenized_agent
    assert torch.allclose(path_tokenized_agent["ctx_sampled_pos"][:, 1], pred_traj[:, 19])
    assert torch.allclose(path_tokenized_agent["ctx_sampled_heading"][:, 1], pred_head[:, 19])
