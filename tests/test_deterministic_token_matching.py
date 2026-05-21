import torch

from src.smart.tokens.agent_token_matching import match_token_idx_from_local_contour
from src.smart.tokens.flow_token_processor import FlowTokenProcessor
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


def test_train_sampled_agent_token_fields_match_gt_fields() -> None:
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


def test_match_token_idx_chunking_preserves_nearest_token() -> None:
    token_bank = torch.zeros((5, 6, 4, 2), dtype=torch.float32)
    for token_idx in range(token_bank.shape[0]):
        token_bank[token_idx] = float(token_idx)
    contour_local = torch.stack(
        [
            token_bank[4] + 0.01,
            token_bank[2] - 0.01,
            token_bank[0] + 0.02,
            token_bank[3] + 0.01,
            token_bank[1] - 0.02,
            token_bank[4] - 0.01,
        ],
        dim=0,
    )
    agent_type = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)

    unchunked_idx = match_token_idx_from_local_contour(
        agent_type=agent_type,
        contour_local=contour_local,
        token_bank_all_veh=token_bank,
        token_bank_all_ped=token_bank,
        token_bank_all_cyc=token_bank,
        reduction="sum",
        query_chunk_size=1024,
    )
    chunked_idx = match_token_idx_from_local_contour(
        agent_type=agent_type,
        contour_local=contour_local,
        token_bank_all_veh=token_bank,
        token_bank_all_ped=token_bank,
        token_bank_all_cyc=token_bank,
        reduction="sum",
        query_chunk_size=1,
    )

    torch.testing.assert_close(chunked_idx, unchunked_idx, atol=0, rtol=0)


def _reference_match_agent_token_loop(
    processor: TokenProcessor,
    *,
    valid: torch.Tensor,
    pos: torch.Tensor,
    heading: torch.Tensor,
    agent_type: torch.Tensor,
    agent_shape: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _, n_step = valid.shape

    prev_pos = pos[:, 0].clone()
    prev_head = heading[:, 0].clone()
    out_dict: dict[str, list[torch.Tensor]] = {
        "valid_mask": [],
        "gt_idx": [],
        "gt_pos": [],
        "gt_heading": [],
        "sampled_idx": [],
        "sampled_pos": [],
        "sampled_heading": [],
    }

    for i in range(processor.shift, n_step, processor.shift):
        segment_valid_mask = valid[:, i - processor.shift : i + 1].all(dim=1)
        invalid_mask = ~segment_valid_mask
        out_dict["valid_mask"].append(segment_valid_mask)

        gt_contour_local = processor._build_local_contour_sequence(
            pos_seq=pos[:, i - processor.shift : i + 1],
            heading_seq=heading[:, i - processor.shift : i + 1],
            ref_pos=prev_pos,
            ref_head=prev_head,
            agent_shape=agent_shape,
        )
        token_idx_gt = processor._match_token_idx_from_local_contour(
            agent_type=agent_type,
            contour_local=gt_contour_local,
            reduction="sum",
        ).masked_fill(invalid_mask, 0)

        prev_head = heading[:, i].clone()
        prev_pos = pos[:, i].clone()

        out_dict["gt_idx"].append(token_idx_gt)
        out_dict["gt_pos"].append(prev_pos.masked_fill(invalid_mask.unsqueeze(1), 0.0))
        out_dict["gt_heading"].append(prev_head.masked_fill(invalid_mask, 0.0))
        out_dict["sampled_idx"].append(out_dict["gt_idx"][-1])
        out_dict["sampled_pos"].append(out_dict["gt_pos"][-1])
        out_dict["sampled_heading"].append(out_dict["gt_heading"][-1])

    return {key: torch.stack(value, dim=1) for key, value in out_dict.items()}


def test_batched_agent_token_matching_matches_reference_loop() -> None:
    generator = torch.Generator().manual_seed(20260521)
    processor = FlowTokenProcessor(
        map_token_file="map_traj_token5.pkl",
        agent_token_file="agent_vocab_555_s2.pkl",
        flow_window_steps=20,
        use_kinematic_control_flow=True,
        control_vehicle_yaw_scale_rad=0.025,
        control_pedestrian_yaw_scale_rad=0.20,
        control_cyclist_yaw_scale_rad=0.06,
    )

    n_agent = 11
    n_step = 31
    valid = torch.ones((n_agent, n_step), dtype=torch.bool)
    valid[0, 8] = False
    valid[1, 5:11] = False
    valid[2, 15] = False
    valid[3, :6] = False
    valid[4, 21:26] = False
    valid[8, :] = False

    pos = torch.randn((n_agent, n_step, 2), generator=generator).cumsum(dim=1)
    heading = torch.randn((n_agent, n_step), generator=generator)
    agent_type = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2])
    agent_shape = processor._get_agent_shape(agent_type)

    actual = processor._match_agent_token(
        valid=valid,
        pos=pos,
        heading=heading,
        agent_type=agent_type,
        agent_shape=agent_shape,
    )
    expected = _reference_match_agent_token_loop(
        processor,
        valid=valid,
        pos=pos,
        heading=heading,
        agent_type=agent_type,
        agent_shape=agent_shape,
    )

    for key in ["valid_mask", "gt_idx", "sampled_idx"]:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    for key in ["gt_pos", "gt_heading", "sampled_pos", "sampled_heading"]:
        torch.testing.assert_close(actual[key], expected[key], atol=1.0e-6, rtol=1.0e-6)
