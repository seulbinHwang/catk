from __future__ import annotations

import torch

from src.smart.model.smart_flow_gan import SMARTFlowGAN


class _CountingRolloutEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.encode_calls = 0
        self.prepare_calls = 0
        self.rollout_calls = 0
        self.detach_block_transition_flags = []

    def encode_map(self, tokenized_map):
        self.encode_calls += 1
        return {"scale": self.scale}

    def prepare_training_rollout_cache(self, tokenized_agent, map_feature):
        self.prepare_calls += 1
        return {"scale": map_feature["scale"]}

    def training_rollout_from_cache(
        self,
        *,
        rollout_cache,
        tokenized_agent,
        map_feature,
        sampling_scheme,
        rollout_steps_2hz,
        self_forced_epoch,
        detach_block_transition,
        use_stop_motion,
        scenario_sampling_seeds=None,
    ):
        self.rollout_calls += 1
        self.detach_block_transition_flags.append(bool(detach_block_transition))
        self.last_scenario_sampling_seeds = scenario_sampling_seeds
        n_agent = int(tokenized_agent["n_agent"])
        n_step = int(tokenized_agent["n_step"])
        value = rollout_cache["scale"] * float(self.rollout_calls)
        pred_traj = value.expand(n_agent, n_step, 2)
        pred_head = value.new_zeros((n_agent, n_step))
        return {"pred_traj_10hz": pred_traj, "pred_head_10hz": pred_head}


class _FakeGAN:
    def __init__(self, *, k: int = 4, n_step: int = 3) -> None:
        self.encoder = _CountingRolloutEncoder()
        self.gan_rollout_set_size = k
        self.flow_window_steps = n_step
        self.self_forced_sampling = {}
        self.validation_closed_seed = 0
        self.current_epoch = 0
        self.self_forced_detach_block_transition = False
        self.self_forced_use_stop_motion = False
        self.switch_calls = 0
        self.restore_calls = 0

    def _switch_module_to_eval_preserving_modes(self, module):
        self.switch_calls += 1
        was_training = module.training
        module.eval()
        return was_training

    def _restore_module_training_modes(self, was_training):
        self.restore_calls += 1
        self.encoder.train(was_training)

    def _get_self_forced_rollout_steps_2hz(self) -> int:
        return 1

    def _gan_prepared_rollout_scene(self, tokenized_map, tokenized_agent):
        return SMARTFlowGAN._gan_prepared_rollout_scene(self, tokenized_map, tokenized_agent)

    def _make_gan_rollout_seed(self, scenario_id, *, stream, rollout_idx):
        return SMARTFlowGAN._make_gan_rollout_seed(self, scenario_id, stream=stream, rollout_idx=rollout_idx)

    def _get_gan_rollout_scenario_seeds(self, scenario_ids, *, stream, rollout_idx, device):
        return SMARTFlowGAN._get_gan_rollout_scenario_seeds(
            self,
            scenario_ids,
            stream=stream,
            rollout_idx=rollout_idx,
            device=device,
        )


def test_gan_fake_set_reuses_scene_preparation_once_and_keeps_gradients() -> None:
    fake = _FakeGAN(k=4, n_step=3)
    context = {
        "agent_batch": torch.tensor([0, 0]),
        "batch_size": 1,
        "scenario_ids": ("scene-a",),
        "n_max_agent": 2,
    }

    output = SMARTFlowGAN._sample_gan_fake_set(
        fake,
        tokenized_map={},
        tokenized_agent={"n_agent": 2, "n_step": 3, "batch": torch.tensor([0, 0])},
        context=context,
    )

    assert fake.switch_calls == 1
    assert fake.restore_calls == 1
    assert fake.encoder.encode_calls == 1
    assert fake.encoder.prepare_calls == 1
    assert fake.encoder.rollout_calls == 4
    assert output.shape == (1, 4, 3, 2, 4)
    assert torch.equal(output[0, :, 0, 0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))

    output[..., 0].sum().backward()
    assert fake.encoder.scale.grad is not None
    assert bool(torch.isfinite(fake.encoder.scale.grad).item())


def test_gan_fake_set_uses_provided_scene_preparation_without_recomputing() -> None:
    fake = _FakeGAN(k=4, n_step=3)
    context = {
        "agent_batch": torch.tensor([0, 0]),
        "batch_size": 1,
        "scenario_ids": ("scene-a",),
        "n_max_agent": 2,
    }
    map_feature = {"scale": fake.encoder.scale}
    rollout_cache = {"scale": fake.encoder.scale}

    output = SMARTFlowGAN._sample_gan_fake_set(
        fake,
        tokenized_map={},
        tokenized_agent={"n_agent": 2, "n_step": 3, "batch": torch.tensor([0, 0])},
        context=context,
        map_feature=map_feature,
        rollout_cache=rollout_cache,
    )

    assert fake.switch_calls == 0
    assert fake.restore_calls == 0
    assert fake.encoder.encode_calls == 0
    assert fake.encoder.prepare_calls == 0
    assert fake.encoder.rollout_calls == 4
    assert output.shape == (1, 4, 3, 2, 4)
    assert torch.equal(output[0, :, 0, 0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))

    output[..., 0].sum().backward()
    assert fake.encoder.scale.grad is not None
    assert bool(torch.isfinite(fake.encoder.scale.grad).item())


def test_gan_fake_set_requires_prepared_scene_inputs_together() -> None:
    fake = _FakeGAN(k=1, n_step=3)
    context = {
        "agent_batch": torch.tensor([0]),
        "batch_size": 1,
        "scenario_ids": ("scene-a",),
        "n_max_agent": 1,
    }

    try:
        SMARTFlowGAN._sample_gan_fake_set(
            fake,
            tokenized_map={},
            tokenized_agent={"n_agent": 1, "n_step": 3, "batch": torch.tensor([0])},
            context=context,
            map_feature={"scale": fake.encoder.scale},
        )
    except ValueError as exc:
        assert "map_feature and rollout_cache" in str(exc)
    else:
        raise AssertionError("Expected ValueError when only one prepared scene input is provided.")


def test_gan_fake_set_forwards_detach_block_transition() -> None:
    fake = _FakeGAN(k=3, n_step=3)
    fake.self_forced_detach_block_transition = True
    context = {
        "agent_batch": torch.tensor([0, 0]),
        "batch_size": 1,
        "scenario_ids": ("scene-a",),
        "n_max_agent": 2,
    }

    SMARTFlowGAN._sample_gan_fake_set(
        fake,
        tokenized_map={},
        tokenized_agent={"n_agent": 2, "n_step": 3, "batch": torch.tensor([0, 0])},
        context=context,
        map_feature={"scale": fake.encoder.scale},
        rollout_cache={"scale": fake.encoder.scale},
    )

    assert fake.encoder.detach_block_transition_flags == [True, True, True]


def test_gan_rollout_seed_is_scene_and_rollout_based() -> None:
    fake = _FakeGAN(k=1, n_step=3)

    seed_a_first = SMARTFlowGAN._make_gan_rollout_seed(
        fake,
        "scene-a",
        stream="g",
        rollout_idx=3,
    )
    seed_a_second = SMARTFlowGAN._make_gan_rollout_seed(
        fake,
        "scene-a",
        stream="g",
        rollout_idx=3,
    )
    seed_other_rollout = SMARTFlowGAN._make_gan_rollout_seed(
        fake,
        "scene-a",
        stream="g",
        rollout_idx=4,
    )
    seed_other_stream = SMARTFlowGAN._make_gan_rollout_seed(
        fake,
        "scene-a",
        stream="d",
        rollout_idx=3,
    )

    assert seed_a_first == seed_a_second
    assert seed_a_first != seed_other_rollout
    assert seed_a_first != seed_other_stream
