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
    model._closed_loop_sf_base_generator_epochs = 4
    model._closed_loop_sf_stage_warmup_epochs = 2
    model.self_forced_use_distribution_matching_loss = True
    model._self_forced_original_check_val_every_n_epoch = None
    model._self_forced_validation_schedule_captured = False
    return model


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


def test_closed_loop_stage_offsets_advance_by_local_max_step() -> None:
    model = _make_closed_loop_model()
    model.closed_loop_sf_global_max_step = 3
    model._sample_closed_loop_sf_prefix_steps = lambda device: 3  # type: ignore[method-assign]
    device = torch.device("cpu")

    expected = {
        6: (0, 3, 3),
        12: (4, 3, 7),
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


def test_closed_loop_stage_start_updates_teacher_when_enabled() -> None:
    model = _make_closed_loop_model()
    _set_current_epoch(model, 6)
    model._closed_loop_sf_last_prepared_stage = 0
    model.update_open_loop_teacher_when_roll = True

    calls = []
    model._is_self_forced_active = lambda: True  # type: ignore[method-assign]
    model._ensure_self_forced_generator_ema_ready = lambda: calls.append("ensure")  # type: ignore[method-assign]
    model._copy_self_forced_ema_to_online_generator = lambda: calls.append("copy")  # type: ignore[method-assign]
    model._reset_self_forced_generator_optimizer_state = lambda: calls.append("reset")  # type: ignore[method-assign]
    model._copy_online_generator_to_self_forced_teacher = lambda: calls.append("teacher")  # type: ignore[method-assign]

    model._prepare_closed_loop_self_forced_stage_for_epoch()

    assert calls == ["ensure", "copy", "reset", "teacher"]
    assert model._closed_loop_sf_last_prepared_stage == 1


def test_closed_loop_stage_warmup_skips_validation_and_resumes_on_generator_epoch() -> None:
    model = _make_closed_loop_model()
    model.self_forced_enabled = True
    model.trainer = _DummyTrainer(check_val_every_n_epoch=1)

    _set_current_epoch(model, 6)
    model._apply_self_forced_validation_schedule_for_current_epoch()
    assert model.trainer.check_val_every_n_epoch == 8

    _set_current_epoch(model, 8)
    model._apply_self_forced_validation_schedule_for_current_epoch()
    assert model.trainer.check_val_every_n_epoch == 1


def test_shifted_self_forced_tokenized_agent_uses_prefix_final_state() -> None:
    model = _make_closed_loop_model()
    tokenized_agent = {
        "ctx_sampled_pos": torch.zeros(3, 5, 2),
        "ctx_sampled_heading": torch.zeros(3, 5),
        "ctx_sampled_idx": torch.zeros(3, 5, dtype=torch.long),
        "ctx_valid": torch.ones(3, 5, dtype=torch.bool),
        "flow_eval_mask": torch.ones(3, 4, dtype=torch.bool),
        "kept": torch.tensor([1, 2, 3]),
    }
    rollout_state = {
        "pos_window": torch.arange(3 * 4 * 2, dtype=torch.float32).view(3, 4, 2),
        "head_window": torch.arange(3 * 4, dtype=torch.float32).view(3, 4),
        "valid_window": torch.tensor(
            [
                [True, True, True, True],
                [True, True, True, False],
                [False, False, False, False],
            ]
        ),
        "pred_idx_window": torch.arange(3 * 4, dtype=torch.long).view(3, 4),
    }

    shifted = model._build_shifted_self_forced_tokenized_agent(tokenized_agent, rollout_state)

    torch.testing.assert_close(shifted["ctx_sampled_pos"][:, 0], rollout_state["pos_window"][:, -2])
    torch.testing.assert_close(shifted["ctx_sampled_pos"][:, 1], rollout_state["pos_window"][:, -1])
    torch.testing.assert_close(shifted["ctx_sampled_heading"][:, 0], rollout_state["head_window"][:, -2])
    torch.testing.assert_close(shifted["ctx_sampled_heading"][:, 1], rollout_state["head_window"][:, -1])
    torch.testing.assert_close(shifted["ctx_sampled_idx"][:, 0], rollout_state["pred_idx_window"][:, -2])
    torch.testing.assert_close(shifted["ctx_sampled_idx"][:, 1], rollout_state["pred_idx_window"][:, -1])
    assert torch.equal(shifted["ctx_valid"][:, 0], rollout_state["valid_window"][:, -2])
    assert torch.equal(shifted["ctx_valid"][:, 1], rollout_state["valid_window"][:, -1])
    assert not shifted["ctx_valid"][:, 2:].any()
    assert torch.equal(shifted["flow_eval_mask"][:, 0], rollout_state["valid_window"][:, -1])
    assert not shifted["flow_eval_mask"][:, 1:].any()
    assert shifted["kept"] is tokenized_agent["kept"]


def test_self_forced_cosine_lr_uses_expanded_curriculum_length() -> None:
    model = _make_closed_loop_model()
    model.self_forced_enabled = True
    model.self_forced_start_epoch = 0
    model.self_forced_estimator_warmup_epochs = 0
    model._self_forced_requested_estimator_warmup_epochs = 0
    model.closed_loop_sf_global_max_step = 4
    model._closed_loop_sf_schedule_configured = False
    model.self_forced_lr_cosine_final_ratio = 0.01
    model.trainer = _DummyTrainer(max_epochs=5)

    model._configure_closed_loop_self_forced_schedule()

    assert model.trainer.max_epochs == 25
    _set_current_epoch(model, 0)
    assert abs(model._get_self_forced_lr_cosine_ratio_for_epoch() - 1.0) < 1e-12
    _set_current_epoch(model, 12)
    assert abs(model._get_self_forced_lr_cosine_ratio_for_epoch() - 0.505) < 1e-12
    _set_current_epoch(model, 24)
    assert abs(model._get_self_forced_lr_cosine_ratio_for_epoch() - 0.01) < 1e-12


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
