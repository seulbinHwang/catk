import torch

from src.smart.tokens.agent_token_matching import match_token_idx_from_local_contour
from src.smart.tokens.token_processor import TokenProcessor


def test_match_token_idx_from_local_contour_uses_nearest_token() -> None:
    token_bank = torch.zeros((3, 6, 4, 2), dtype=torch.float32)
    token_bank[1] = 1.0
    token_bank[2] = -1.0
    contour_local = torch.stack(
        [
            token_bank[2] + 0.01,
            token_bank[0] + 0.02,
            token_bank[1] - 0.01,
        ],
        dim=0,
    )
    agent_type = torch.tensor([0, 1, 2], dtype=torch.long)

    token_idx = match_token_idx_from_local_contour(
        agent_type=agent_type,
        contour_local=contour_local,
        token_bank_all_veh=token_bank,
        token_bank_all_ped=token_bank,
        token_bank_all_cyc=token_bank,
        reduction="sum",
    )

    torch.testing.assert_close(token_idx, torch.tensor([2, 0, 1]))


def test_agent_token_sampling_fields_match_gt_fields() -> None:
    processor = TokenProcessor.__new__(TokenProcessor)
    processor.shift = 5
    processor.agent_token_all_veh = torch.zeros((2, 6, 4, 2), dtype=torch.float32)
    processor.agent_token_all_ped = torch.zeros((2, 6, 4, 2), dtype=torch.float32)
    processor.agent_token_all_cyc = torch.zeros((2, 6, 4, 2), dtype=torch.float32)

    valid = torch.ones((3, 11), dtype=torch.bool)
    pos = torch.zeros((3, 11, 2), dtype=torch.float32)
    pos[:, :, 0] = torch.arange(11, dtype=torch.float32)
    heading = torch.zeros((3, 11), dtype=torch.float32)
    agent_type = torch.tensor([0, 1, 2], dtype=torch.long)
    agent_shape = torch.tensor(
        [
            [2.0, 4.8],
            [1.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=torch.float32,
    )

    tokenized = processor._match_agent_token(
        valid=valid,
        pos=pos,
        heading=heading,
        agent_type=agent_type,
        agent_shape=agent_shape,
    )

    torch.testing.assert_close(tokenized["sampled_idx"], tokenized["gt_idx"])
    torch.testing.assert_close(tokenized["sampled_pos"], tokenized["gt_pos"])
    torch.testing.assert_close(tokenized["sampled_heading"], tokenized["gt_heading"])
