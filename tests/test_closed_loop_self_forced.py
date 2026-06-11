from __future__ import annotations

import torch
import torch.nn as nn

from src.smart.model.smart_flow import SMARTFlow


class _DummyTrainer:
    def __init__(
        self,
        current_epoch: int = 0,
        check_val_every_n_epoch: int = 1,
        max_epochs: int = 18,
    ) -> None:
        self.current_epoch = int(current_epoch)
        self.check_val_every_n_epoch = int(check_val_every_n_epoch)
        self.max_epochs = int(max_epochs)
        self.fit_loop = type("_DummyFitLoop", (), {"max_epochs": int(max_epochs)})()


def _set_current_epoch(model: SMARTFlow, epoch: int) -> None:
    model.trainer.current_epoch = int(epoch)


def _make_closed_loop_model() -> SMARTFlow:
    model = SMARTFlow.__new__(SMARTFlow)
    nn.Module.__init__(model)
    model._fabric = None
    model._jit_is_scripting = False
    model.trainer = _DummyTrainer()
    model.self_forced_start_epoch = 0
    model.self_forced_estimator_warmup_epochs = 2
    model._self_forced_requested_estimator_warmup_epochs = 2
    model.closed_loop_sf_global_max_step = 2
    model.closed_loop_sf_local_max_step = 4
    model.closed_loop_see_all = False
    model.self_forced_rollout_anchor_stride = 2
    model._closed_loop_sf_base_generator_epochs = 4
    model._closed_loop_sf_stage_warmup_epochs = 2
    model.self_forced_enabled = True
    model.self_forced_use_distribution_matching_loss = True
    model.self_forced_skip_initial_stage_from_checkpoint = True
    model._self_forced_aux_loaded_from_checkpoint = False
    model._self_forced_resume_checkpoint_next_epoch = None
    model._closed_loop_sf_initial_stage_end_epoch_override = None
    model._self_forced_original_check_val_every_n_epoch = None
    model._self_forced_validation_schedule_captured = False
    return model


def _make_minimal_anchor_tokenized_agent(
    *,
    num_agents: int = 2,
    num_anchors: int = 16,
) -> dict[str, torch.Tensor | int]:
    return {
        "batch": torch.zeros(num_agents, dtype=torch.long),
        "num_graphs": 1,
        "self_forced_rollout_anchor_mask": torch.ones(
            num_agents,
            num_anchors,
            dtype=torch.bool,
        ),
        "trajectory_token_veh": torch.zeros(1),
        "trajectory_token_ped": torch.zeros(1),
        "trajectory_token_cyc": torch.zeros(1),
        "token_bank_all_veh": torch.zeros(1),
        "token_bank_all_ped": torch.zeros(1),
        "token_bank_all_cyc": torch.zeros(1),
    }


def test_closed_loop_self_forced_stage_uses_warmup_plus_generator_blocks() -> None:
    model = _make_closed_loop_model()

    expected = {
        0: 0,
        1: 0,
        2: 0,
        5: 0,
        6: 1,
        7: 1,
        11: 1,
        12: 2,
        17: 2,
        99: 2,
    }
    for epoch, stage in expected.items():
        _set_current_epoch(model, epoch)
        assert model._get_closed_loop_self_forced_stage() == stage


def test_initial_self_forcing_stage_always_uses_all_rollout_anchors() -> None:
    model = _make_closed_loop_model()
    model.self_forced_rollout_anchor_stride = 4
    _set_current_epoch(model, 2)

    assert model._get_closed_loop_self_forced_stage() == 0
    assert model._get_self_forced_rollout_anchor_stride_for_current_stage() == 1

    tokenized_agent = _make_minimal_anchor_tokenized_agent()
    rollout_tokenized_agent, anchor_mask = model._build_multi_anchor_self_forced_tokenized_agent(
        tokenized_agent,  # type: ignore[arg-type]
        anchor_stride=model._get_self_forced_rollout_anchor_stride_for_current_stage(),
    )

    assert rollout_tokenized_agent["_self_forced_anchor_repeat_count"] == 16
    assert rollout_tokenized_agent["num_graphs"] == 16
    assert anchor_mask.shape == (32,)
    assert anchor_mask.all()


def test_closed_loop_self_forcing_stage_uses_configured_rollout_anchor_stride() -> None:
    model = _make_closed_loop_model()
    model.self_forced_rollout_anchor_stride = 4
    _set_current_epoch(model, 6)

    assert model._get_closed_loop_self_forced_stage() == 1
    assert model._get_self_forced_rollout_anchor_stride_for_current_stage() == 4

    tokenized_agent = _make_minimal_anchor_tokenized_agent()
    rollout_tokenized_agent, anchor_mask = model._build_multi_anchor_self_forced_tokenized_agent(
        tokenized_agent,  # type: ignore[arg-type]
        anchor_stride=model._get_self_forced_rollout_anchor_stride_for_current_stage(),
    )

    assert rollout_tokenized_agent["_self_forced_anchor_repeat_count"] == 4
    assert rollout_tokenized_agent["num_graphs"] == 4
    assert anchor_mask.shape == (8,)
    assert anchor_mask.all()


def test_skip_initial_stage_resume_starts_at_closed_loop_stage_one() -> None:
    model = _make_closed_loop_model()
    model.self_forced_estimator_warmup_epochs = 0
    model._self_forced_requested_estimator_warmup_epochs = 0
    model.closed_loop_sf_global_max_step = 2
    model.self_forced_rollout_anchor_stride = 4
    model._self_forced_aux_loaded_from_checkpoint = True
    model._self_forced_resume_checkpoint_next_epoch = 6
    model._closed_loop_sf_schedule_configured = False
    model.trainer = _DummyTrainer(current_epoch=6, max_epochs=2)

    model._configure_closed_loop_self_forced_schedule()

    assert model._closed_loop_sf_base_generator_epochs == 2
    assert model._closed_loop_sf_initial_stage_end_epoch_override == 6
    assert model.trainer.max_epochs == 10

    _set_current_epoch(model, 6)
    assert model._get_closed_loop_self_forced_stage() == 1
    assert model._get_self_forced_rollout_anchor_stride_for_current_stage() == 4

    tokenized_agent = _make_minimal_anchor_tokenized_agent()
    rollout_tokenized_agent, anchor_mask = model._build_multi_anchor_self_forced_tokenized_agent(
        tokenized_agent,  # type: ignore[arg-type]
        anchor_stride=model._get_self_forced_rollout_anchor_stride_for_current_stage(),
    )

    assert rollout_tokenized_agent["_self_forced_anchor_repeat_count"] == 4
    assert anchor_mask.shape == (8,)


def test_closed_loop_stage_offsets_advance_by_local_max_step() -> None:
    model = _make_closed_loop_model()
    model.closed_loop_sf_global_max_step = 3
    model._sample_closed_loop_sf_prefix_steps = (  # type: ignore[method-assign]
        lambda device: 3 if model._get_closed_loop_self_forced_stage() > 0 else 0
    )
    device = torch.device("cpu")

    expected = {
        5: (0, 0, 0),
        6: (0, 3, 3),
        8: (0, 3, 3),
        12: (4, 3, 7),
        14: (4, 3, 7),
        18: (8, 3, 11),
    }
    for epoch, counts in expected.items():
        _set_current_epoch(model, epoch)
        assert model._sample_closed_loop_sf_prefix_step_counts(device=device) == counts


def test_closed_loop_see_all_samples_from_zero_to_current_stage_window(monkeypatch) -> None:
    model = _make_closed_loop_model()
    model.closed_loop_see_all = True
    model.closed_loop_sf_global_max_step = 3
    device = torch.device("cpu")
    requested_ranges: list[tuple[int, int]] = []

    def fake_randint(*, low, high, size, device, dtype):
        requested_ranges.append((low, high))
        return torch.tensor([high - 1], device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randint", fake_randint)

    expected = {
        6: (0, 4, 4),
        12: (0, 8, 8),
        18: (0, 12, 12),
    }
    for epoch, counts in expected.items():
        _set_current_epoch(model, epoch)
        assert model._sample_closed_loop_sf_prefix_step_counts(device=device) == counts

    assert requested_ranges == [(0, 5), (0, 9), (0, 13)]


def test_closed_loop_see_all_allows_zero_second_prefix(monkeypatch) -> None:
    model = _make_closed_loop_model()
    model.closed_loop_see_all = True
    device = torch.device("cpu")

    def fake_randint(*, low, high, size, device, dtype):
        assert low == 0
        return torch.tensor([0], device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randint", fake_randint)
    _set_current_epoch(model, 6)

    assert model._sample_closed_loop_sf_prefix_step_counts(device=device) == (0, 0, 0)


def test_closed_loop_self_forced_stage_warmup_repeats_after_bank_skipped_initial_warmup() -> None:
    model = _make_closed_loop_model()
    model.self_forced_estimator_warmup_epochs = 0
    model._self_forced_requested_estimator_warmup_epochs = 2
    model._closed_loop_sf_stage_warmup_epochs = 2

    expected_warmup = {
        0: False,
        3: False,
        4: True,
        5: True,
        6: False,
        9: False,
        10: True,
        11: True,
    }
    for epoch, is_warmup in expected_warmup.items():
        _set_current_epoch(model, epoch)
        assert model._is_self_forced_estimator_warmup_active() is is_warmup


def test_closed_loop_self_forced_completed_generator_count_skips_stage_warmup() -> None:
    model = _make_closed_loop_model()

    expected = {
        0: 0,
        1: 0,
        2: 1,
        5: 4,
        6: 4,
        7: 4,
        8: 5,
        11: 8,
        12: 8,
        13: 8,
        14: 9,
        17: 12,
    }
    for epoch, count in expected.items():
        _set_current_epoch(model, epoch)
        assert model._get_self_forced_completed_generator_epoch_count_for_current_epoch() == count


def test_closed_loop_stage_start_prepares_online_generator_even_during_stage_warmup() -> None:
    model = _make_closed_loop_model()
    _set_current_epoch(model, 6)
    model._closed_loop_sf_last_prepared_stage = 0
    model.update_open_loop_teacher_when_roll = False
    assert model._is_self_forced_estimator_warmup_active()

    calls = []
    model._is_self_forced_active = lambda: True  # type: ignore[method-assign]
    model._ensure_self_forced_generator_ema_ready = lambda: calls.append("ensure")  # type: ignore[method-assign]
    model._copy_self_forced_ema_to_online_generator = lambda: calls.append("copy")  # type: ignore[method-assign]
    model._reset_self_forced_generator_optimizer_state = lambda: calls.append("reset")  # type: ignore[method-assign]

    model._prepare_closed_loop_self_forced_stage_for_epoch()

    assert calls == ["ensure", "copy", "reset"]
    assert model._closed_loop_sf_last_prepared_stage == 1


def test_shifted_multi_anchor_self_forced_tokenized_agent_uses_prefix_final_state() -> None:
    model = _make_closed_loop_model()
    n_anchor_agent = 6
    tokenized_agent = {
        "ctx_sampled_pos": torch.zeros(n_anchor_agent, 5, 2),
        "ctx_sampled_heading": torch.zeros(n_anchor_agent, 5),
        "ctx_sampled_idx": torch.zeros(n_anchor_agent, 5, dtype=torch.long),
        "ctx_valid": torch.ones(n_anchor_agent, 5, dtype=torch.bool),
        "gt_pos": torch.zeros(n_anchor_agent, 2, 2),
        "gt_heading": torch.zeros(n_anchor_agent, 2),
        "gt_idx": torch.zeros(n_anchor_agent, 2, dtype=torch.long),
        "valid_mask": torch.ones(n_anchor_agent, 2, dtype=torch.bool),
        "kept": torch.arange(n_anchor_agent),
    }
    rollout_state = {
        "pos_window": torch.arange(n_anchor_agent * 4 * 2, dtype=torch.float32).view(n_anchor_agent, 4, 2),
        "head_window": torch.arange(n_anchor_agent * 4, dtype=torch.float32).view(n_anchor_agent, 4),
        "valid_window": torch.tensor(
            [
                [True, True, True, True],
                [True, True, True, False],
                [True, True, False, False],
                [True, True, True, True],
                [True, False, False, False],
                [True, True, True, True],
            ],
            dtype=torch.bool,
        ),
        "pred_idx_window": torch.arange(n_anchor_agent * 4, dtype=torch.long).view(n_anchor_agent, 4),
    }

    shifted = model._build_shifted_self_forced_tokenized_agent(tokenized_agent, rollout_state)

    torch.testing.assert_close(shifted["ctx_sampled_pos"][:, 0], rollout_state["pos_window"][:, -2])
    torch.testing.assert_close(shifted["ctx_sampled_pos"][:, 1], rollout_state["pos_window"][:, -1])
    torch.testing.assert_close(shifted["gt_pos"][:, 0], rollout_state["pos_window"][:, -2])
    torch.testing.assert_close(shifted["gt_pos"][:, 1], rollout_state["pos_window"][:, -1])
    torch.testing.assert_close(shifted["ctx_sampled_heading"][:, 0], rollout_state["head_window"][:, -2])
    torch.testing.assert_close(shifted["ctx_sampled_heading"][:, 1], rollout_state["head_window"][:, -1])
    torch.testing.assert_close(shifted["gt_heading"][:, 0], rollout_state["head_window"][:, -2])
    torch.testing.assert_close(shifted["gt_heading"][:, 1], rollout_state["head_window"][:, -1])
    assert torch.equal(shifted["ctx_sampled_idx"][:, 0], rollout_state["pred_idx_window"][:, -2])
    assert torch.equal(shifted["ctx_sampled_idx"][:, 1], rollout_state["pred_idx_window"][:, -1])
    assert torch.equal(shifted["valid_mask"][:, 0], rollout_state["valid_window"][:, -2])
    assert torch.equal(shifted["valid_mask"][:, 1], rollout_state["valid_window"][:, -1])
    assert torch.equal(shifted["flow_eval_mask"][:, 0], rollout_state["valid_window"][:, -1])
    assert shifted["kept"] is tokenized_agent["kept"]


def test_closed_loop_optimizer_state_reset_after_ema_copy_boundary() -> None:
    model = _make_closed_loop_model()
    online = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(online.parameters(), lr=0.1)
    loss = online(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    assert len(optimizer.state) > 0

    model.optimizers = lambda: [optimizer]  # type: ignore[method-assign]
    model._reset_self_forced_generator_optimizer_state()

    assert len(optimizer.state) == 0
    assert all(parameter.grad is None for parameter in online.parameters())


def test_prefix_state_clean_prediction_uses_rollout_context() -> None:
    model = _make_closed_loop_model()
    tokenized_map = {"map": torch.ones(1)}
    tokenized_agent = {"agent": torch.ones(1)}
    map_feature = {"feature": torch.ones(1)}
    noisy_path = torch.ones(2, 20, 4)
    tau = torch.ones(2)
    anchor_mask = torch.tensor([True, False, True])
    initial_state = {"pos_window": torch.ones(3, 4, 2)}
    calls: list[tuple[str, object | None]] = []

    class _Decoder:
        def encode_map(self, tokenized_map_arg):
            calls.append(("encode_map", tokenized_map_arg))
            return map_feature

        def path_flow_velocity_for_anchor0(self, **kwargs):
            raise AssertionError("prefix clean prediction must not use two-token anchor context")

        def path_flow_velocity_from_rollout_state(self, **kwargs):
            calls.append(("rollout_state", kwargs["initial_state"]))
            assert kwargs["tokenized_agent"] is tokenized_agent
            assert kwargs["map_feature"] is map_feature
            assert kwargs["path_noisy_norm"] is noisy_path
            assert kwargs["tau"] is tau
            assert kwargs["anchor_mask"] is anchor_mask
            return {
                "velocity": torch.zeros_like(noisy_path),
                "clean": torch.zeros_like(noisy_path),
            }

    pred = model._predict_path_flow_clean_estimate(
        decoder=_Decoder(),  # type: ignore[arg-type]
        tokenized_map=tokenized_map,
        tokenized_agent=tokenized_agent,
        noisy_path_norm=noisy_path,
        tau=tau,
        anchor_mask=anchor_mask,
        map_feature=map_feature,
        initial_rollout_state=initial_state,
    )

    assert calls == [("rollout_state", initial_state)]
    assert torch.equal(pred["velocity"], torch.zeros_like(noisy_path))
    assert torch.equal(pred["clean"], torch.zeros_like(noisy_path))


def test_base_clean_prediction_keeps_anchor0_context_without_prefix_state() -> None:
    model = _make_closed_loop_model()
    tokenized_map = {"map": torch.ones(1)}
    tokenized_agent = {"agent": torch.ones(1)}
    map_feature = {"feature": torch.ones(1)}
    noisy_path = torch.ones(2, 20, 4)
    tau = torch.ones(2)
    anchor_mask = torch.tensor([True, False, True])
    calls: list[str] = []

    class _Decoder:
        def encode_map(self, tokenized_map_arg):
            calls.append("encode_map")
            return map_feature

        def path_flow_velocity_for_anchor0(self, **kwargs):
            calls.append("anchor0")
            assert kwargs["tokenized_agent"] is tokenized_agent
            assert kwargs["map_feature"] is map_feature
            assert kwargs["path_noisy_norm"] is noisy_path
            assert kwargs["tau"] is tau
            assert kwargs["anchor_mask"] is anchor_mask
            return {
                "velocity": torch.zeros_like(noisy_path),
                "clean": torch.zeros_like(noisy_path),
            }

        def path_flow_velocity_from_rollout_state(self, **kwargs):
            raise AssertionError("base self-forcing must keep existing anchor0 context")

    pred = model._predict_path_flow_clean_estimate(
        decoder=_Decoder(),  # type: ignore[arg-type]
        tokenized_map=tokenized_map,
        tokenized_agent=tokenized_agent,
        noisy_path_norm=noisy_path,
        tau=tau,
        anchor_mask=anchor_mask,
        map_feature=map_feature,
    )

    assert calls == ["anchor0"]
    assert torch.equal(pred["velocity"], torch.zeros_like(noisy_path))
    assert torch.equal(pred["clean"], torch.zeros_like(noisy_path))


def test_self_forced_cosine_lr_default_ratio_keeps_constant_lr() -> None:
    model = _make_closed_loop_model()
    model.self_forced_enabled = True
    model.self_forced_start_epoch = 0
    model.self_forced_estimator_warmup_epochs = 0
    model._self_forced_requested_estimator_warmup_epochs = 0
    model.closed_loop_sf_global_max_step = 4
    model._closed_loop_sf_schedule_configured = False
    model.self_forced_lr_cosine_final_ratio = 1.0
    model.trainer = _DummyTrainer(max_epochs=5)

    model._configure_closed_loop_self_forced_schedule()

    assert model.trainer.max_epochs == 25
    for epoch in (0, 12, 24):
        _set_current_epoch(model, epoch)
        assert abs(model._get_self_forced_lr_cosine_ratio_for_epoch() - 1.0) < 1e-12


def test_self_forced_cosine_lr_updates_both_optimizers_on_resume_epoch() -> None:
    model = _make_closed_loop_model()
    model.self_forced_enabled = True
    model.self_forced_start_epoch = 0
    model.self_forced_lr_cosine_final_ratio = 0.01
    model.lr = 7e-5
    model.self_forced_generated_estimator_lr = 5e-5
    model.trainer = _DummyTrainer(current_epoch=24, max_epochs=25)
    model.log = lambda *args, **kwargs: None  # type: ignore[method-assign]

    generator = nn.Linear(2, 1)
    estimator = nn.Linear(2, 1)
    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=0.123)
    estimator_optimizer = torch.optim.AdamW(estimator.parameters(), lr=0.456)
    model.optimizers = lambda: [generator_optimizer, estimator_optimizer]  # type: ignore[method-assign]

    model._apply_self_forced_lr_schedule_for_current_epoch()

    assert abs(generator_optimizer.param_groups[0]["lr"] - 7e-7) < 1e-12
    assert abs(estimator_optimizer.param_groups[0]["lr"] - 5e-7) < 1e-12
