from types import SimpleNamespace

import torch

from src.smart.model.smart_flow import SMARTFlow


class _FakeAgentEncoder:
    flow_state_dim = 2

    def _build_rollout_noise_tape(
        self,
        *,
        num_agent,
        tape_steps,
        device,
        dtype,
        sampling_scheme,
        scenario_sampling_seeds,
        agent_batch,
    ):
        seed_by_agent = torch.remainder(
            scenario_sampling_seeds[agent_batch].to(torch.long),
            997,
        ).to(device=device, dtype=dtype)
        noise_scale = float(getattr(sampling_scheme, "noise_scale", 1.0))
        agent_offset = torch.arange(num_agent, device=device, dtype=dtype).view(num_agent, 1, 1)
        time_offset = torch.arange(tape_steps, device=device, dtype=dtype).view(1, tape_steps, 1)
        dim_offset = torch.arange(self.flow_state_dim, device=device, dtype=dtype).view(1, 1, -1)
        tape = seed_by_agent.view(num_agent, 1, 1) + agent_offset * 0.01
        tape = tape + time_offset * 0.001 + dim_offset * 0.0001
        return tape * noise_scale


def test_ocsc_open_loop_x_init_uses_closed_loop_scenario_seeds() -> None:
    model = object.__new__(SMARTFlow)
    model.flow_window_steps = 4
    model.validation_closed_seed = 17

    agent_enc = _FakeAgentEncoder()
    tokenized_agent = {"batch": torch.tensor([0, 0, 1, 1], dtype=torch.long)}
    active_mask = torch.tensor([True, False, True, True])
    scenario_ids = ["scenario-a", "scenario-b"]
    sampling_scheme = SimpleNamespace(noise_scale=2.0)

    stack = SMARTFlow._ocsc_build_open_loop_x_init_stack(
        model,
        agent_enc=agent_enc,
        tokenized_agent=tokenized_agent,
        active_mask=active_mask,
        scenario_ids=scenario_ids,
        sample_count=3,
        sampling_scheme=sampling_scheme,
        dtype=torch.float32,
    )

    assert stack.shape == (3, 3, 4, 2)
    for rollout_idx in range(3):
        scenario_seeds = model._get_closed_loop_scenario_seeds(
            scenario_ids=scenario_ids,
            rollout_idx=rollout_idx,
            device=tokenized_agent["batch"].device,
        )
        expected = agent_enc._build_rollout_noise_tape(
            num_agent=4,
            tape_steps=4,
            device=active_mask.device,
            dtype=torch.float32,
            sampling_scheme=sampling_scheme,
            scenario_sampling_seeds=scenario_seeds,
            agent_batch=tokenized_agent["batch"],
        )[active_mask]
        torch.testing.assert_close(stack[rollout_idx], expected)

    assert not torch.equal(stack[0], stack[1])
