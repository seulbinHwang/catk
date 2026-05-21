import torch

from src.smart.tokens.token_processor import TokenProcessor


def _reference_extrapolate(
    *,
    shift: int,
    valid: torch.Tensor,
    pos: torch.Tensor,
    heading: torch.Tensor,
    vel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    first_valid_step = torch.max(valid, dim=1).indices
    for i, t_tensor in enumerate(first_valid_step):
        t = int(t_tensor.item())
        n_step_to_extrapolate = t % shift
        if (t == 10) and (not bool(valid[i, 10 - shift].item())):
            n_step_to_extrapolate = shift

        if n_step_to_extrapolate > 0:
            vel[i, t - n_step_to_extrapolate : t] = vel[i, t]
            valid[i, t - n_step_to_extrapolate : t] = True
            heading[i, t - n_step_to_extrapolate : t] = heading[i, t]

            for j in range(n_step_to_extrapolate):
                pos[i, t - j - 1] = pos[i, t - j] - vel[i, t] * 0.1

    return valid, pos, heading, vel


def test_extrapolate_agent_to_prev_token_step_matches_reference_loop() -> None:
    generator = torch.Generator().manual_seed(20260521)
    n_agent = 9
    n_step = 18
    valid = torch.zeros((n_agent, n_step), dtype=torch.bool)
    first_valid_steps = [0, 1, 4, 5, 7, 10, 10, 13, 0]
    for agent_idx, first_step in enumerate(first_valid_steps[:-1]):
        valid[agent_idx, first_step:] = True
    valid[6, 5] = True

    pos = torch.randn((n_agent, n_step, 2), generator=generator)
    heading = torch.randn((n_agent, n_step), generator=generator)
    vel = torch.randn((n_agent, n_step, 2), generator=generator)

    processor = TokenProcessor.__new__(TokenProcessor)
    processor.shift = 5
    actual = processor._extrapolate_agent_to_prev_token_step(
        valid=valid.clone(),
        pos=pos.clone(),
        heading=heading.clone(),
        vel=vel.clone(),
    )
    expected = _reference_extrapolate(
        shift=processor.shift,
        valid=valid.clone(),
        pos=pos.clone(),
        heading=heading.clone(),
        vel=vel.clone(),
    )

    torch.testing.assert_close(actual[0], expected[0], atol=0.0, rtol=0.0)
    for actual_tensor, expected_tensor in zip(actual[1:], expected[1:]):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=1.0e-6, rtol=1.0e-6)


def test_extrapolate_agent_to_prev_token_step_noop_when_already_aligned() -> None:
    generator = torch.Generator().manual_seed(20260522)
    valid = torch.zeros((4, 16), dtype=torch.bool)
    valid[0, 0:] = True
    valid[1, 5:] = True
    valid[2, 10:] = True
    valid[2, 5] = True

    pos = torch.randn((4, 16, 2), generator=generator)
    heading = torch.randn((4, 16), generator=generator)
    vel = torch.randn((4, 16, 2), generator=generator)

    processor = TokenProcessor.__new__(TokenProcessor)
    processor.shift = 5
    actual = processor._extrapolate_agent_to_prev_token_step(
        valid=valid.clone(),
        pos=pos.clone(),
        heading=heading.clone(),
        vel=vel.clone(),
    )

    for actual_tensor, input_tensor in zip(actual, (valid, pos, heading, vel)):
        torch.testing.assert_close(actual_tensor, input_tensor, atol=0.0, rtol=0.0)
