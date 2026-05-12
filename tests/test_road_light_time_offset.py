from __future__ import annotations

import torch

import src.smart.road.generator as road_generator
from src.smart.modules.flow_agent_decoder import SMARTFlowAgentDecoder
from src.smart.road.generator import RoadGenerationConfig, road_light_time_start_seconds


def test_rollout_light_time_delta_uses_road_start_offset() -> None:
    decoder = SMARTFlowAgentDecoder.__new__(SMARTFlowAgentDecoder)
    decoder.shift = 5

    first = decoder._build_rollout_light_time_delta_norm(
        num_agent=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        rollout_step_index=0,
        rollout_start_seconds=road_light_time_start_seconds(block_idx=3, commit_steps=5),
    )
    second = decoder._build_rollout_light_time_delta_norm(
        num_agent=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        rollout_step_index=1,
        rollout_start_seconds=road_light_time_start_seconds(block_idx=3, commit_steps=5),
    )

    torch.testing.assert_close(first, torch.tensor([[1.5 / 6.0]], dtype=torch.float32))
    torch.testing.assert_close(second, torch.tensor([[2.0 / 6.0]], dtype=torch.float32))


def test_initial_context_light_time_delta_uses_road_start_offset() -> None:
    decoder = SMARTFlowAgentDecoder.__new__(SMARTFlowAgentDecoder)
    decoder.shift = 5
    decoder.num_historical_steps = 11

    context = decoder._build_rollout_context_light_time_delta_norm(
        num_agent=1,
        num_steps=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        rollout_start_seconds=road_light_time_start_seconds(block_idx=3, commit_steps=5),
    )

    torch.testing.assert_close(
        context,
        torch.tensor([[1.0 / 6.0, 1.5 / 6.0]], dtype=torch.float32),
    )


def test_road_block_offset_is_forwarded_to_candidate_generation() -> None:
    calls: list[float] = []
    original = road_generator.sample_candidate_micro_batch

    def fake_sample_candidate_micro_batch(
        model,
        current_sample,
        transform,
        config,
        device,
        repeat_count,
        seed,
        light_time_start_seconds=0.0,
    ):
        calls.append(float(light_time_start_seconds))
        agent_count = int(current_sample["agent"]["position"].shape[0])
        horizon = int(config.selection_horizon_steps)
        return (
            torch.zeros((int(repeat_count), agent_count, horizon, 2)),
            torch.zeros((int(repeat_count), agent_count, horizon)),
            torch.ones((int(repeat_count), agent_count, horizon), dtype=torch.bool),
        )

    road_generator.sample_candidate_micro_batch = fake_sample_candidate_micro_batch
    try:
        config = RoadGenerationConfig(
            candidates_per_agent=3,
            candidate_micro_batch_size=2,
            commit_steps=5,
            selection_horizon_steps=20,
        )
        current_sample = {
            "agent": {
                "position": torch.zeros((2, 91, 3)),
            }
        }

        road_generator.sample_candidate_rollouts_for_block(
            model=object(),
            current_sample=current_sample,
            transform=lambda sample: sample,
            config=config,
            device=torch.device("cpu"),
            seed_base=17,
            block_idx=3,
        )
    finally:
        road_generator.sample_candidate_micro_batch = original

    assert calls == [1.5, 1.5]


def test_road_block_offset_is_forwarded_to_initial_cache() -> None:
    calls: list[float] = []
    original_to_batch = road_generator._to_repeated_batch

    class DummyTokenProcessor:
        def eval(self):
            return None

        def __call__(self, batch):
            return {}, {}

    class DummyEncoder:
        def encode_map(self, tokenized_map):
            return {}

        def prepare_inference_cache(
            self,
            tokenized_agent,
            map_feature,
            light_time_start_seconds=0.0,
        ):
            calls.append(float(light_time_start_seconds))
            return {}

        def rollout_from_cache(self, **kwargs):
            return {
                "pred_traj_10hz": torch.zeros((6, 20, 2), dtype=torch.float32),
                "pred_head_10hz": torch.zeros((6, 20), dtype=torch.float32),
                "pred_valid_10hz": torch.ones((6, 20), dtype=torch.bool),
            }

    class DummyModel:
        training = True
        token_processor = DummyTokenProcessor()
        encoder = DummyEncoder()

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

    road_generator._to_repeated_batch = lambda *args, **kwargs: object()
    try:
        config = RoadGenerationConfig(
            candidates_per_agent=3,
            candidate_micro_batch_size=3,
            commit_steps=5,
            selection_horizon_steps=20,
        )
        current_sample = {
            "agent": {
                "position": torch.zeros((2, 91, 3)),
            }
        }

        road_generator.sample_candidate_micro_batch(
            model=DummyModel(),
            current_sample=current_sample,
            transform=lambda sample: sample,
            config=config,
            device=torch.device("cpu"),
            repeat_count=3,
            seed=17,
            light_time_start_seconds=1.5,
        )
    finally:
        road_generator._to_repeated_batch = original_to_batch

    assert calls == [1.5]
