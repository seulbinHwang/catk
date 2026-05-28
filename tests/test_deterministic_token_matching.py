from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from src.smart.tokens.agent_token_matching import (
    build_agent_type_masks,
    match_token_idx_from_local_contour,
)
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import cal_polygon_contour, merge_by_type, transform_to_global


def _make_processor() -> TokenProcessor:
    processor = TokenProcessor(
        map_token_file="map_traj_token5.pkl",
        agent_token_file="trajtok_vocab.pkl",
    )
    processor.train()
    return processor


def _make_agent_data() -> HeteroData:
    generator = torch.Generator().manual_seed(20260521)
    n_agent = 11
    n_step = 31
    data = HeteroData()
    data.num_graphs = 2

    valid_mask = torch.ones((n_agent, n_step), dtype=torch.bool)
    valid_mask[1, 5] = False
    valid_mask[2, 10:15] = False
    valid_mask[3, :10] = False
    valid_mask[5, 15] = False
    valid_mask[7, :5] = False
    valid_mask[9, 20:25] = False

    position_xy = torch.randn((n_agent, n_step, 2), generator=generator).cumsum(dim=1)
    position_xy += torch.arange(n_agent, dtype=torch.float32).view(-1, 1, 1) * 3.0
    position = torch.zeros((n_agent, n_step, 3), dtype=torch.float32)
    position[..., :2] = position_xy
    position[..., 2] = torch.randn((n_agent, n_step), generator=generator) * 0.1

    heading = torch.randn((n_agent, n_step), generator=generator) * 0.2
    heading += torch.linspace(0.0, 1.2, n_step).view(1, -1)
    velocity = torch.randn((n_agent, n_step, 2), generator=generator) * 0.5

    role = torch.zeros((n_agent, 3), dtype=torch.bool)
    role[0, 0] = True
    agent_type = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2], dtype=torch.uint8)

    data["agent"]["num_nodes"] = n_agent
    data["agent"]["valid_mask"] = valid_mask
    data["agent"]["role"] = role
    data["agent"]["id"] = torch.arange(n_agent, dtype=torch.long)
    data["agent"]["type"] = agent_type
    data["agent"]["position"] = position
    data["agent"]["heading"] = heading
    data["agent"]["velocity"] = velocity
    data["agent"]["shape"] = torch.tensor(
        [[4.8, 2.0, 1.5]] * n_agent,
        dtype=torch.float32,
    )
    data["agent"]["batch"] = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    return data


def _reference_match_agent_token_loop(
    processor: TokenProcessor,
    *,
    valid: torch.Tensor,
    pos: torch.Tensor,
    heading: torch.Tensor,
    agent_shape: torch.Tensor,
    token_traj: torch.Tensor,
) -> dict[str, torch.Tensor]:
    n_agent, n_step = valid.shape
    range_a = torch.arange(n_agent, device=valid.device)
    prev_pos, prev_head = pos[:, 0], heading[:, 0]

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
        valid_mask = valid[:, i - processor.shift] & valid[:, i]
        invalid_mask = ~valid_mask
        out_dict["valid_mask"].append(valid_mask)

        gt_contour = cal_polygon_contour(pos[:, i], heading[:, i], agent_shape)
        gt_contour = gt_contour.unsqueeze(1)
        token_world_gt = transform_to_global(
            pos_local=token_traj.flatten(1, 2),
            head_local=None,
            pos_now=prev_pos,
            head_now=prev_head,
        )[0].view(*token_traj.shape)
        token_idx_gt = torch.argmin(
            torch.norm(token_world_gt - gt_contour, dim=-1).sum(-1),
            dim=-1,
        )
        token_contour_gt = token_world_gt[range_a, token_idx_gt]

        prev_head = heading[:, i].clone()
        dxy = token_contour_gt[:, 0] - token_contour_gt[:, 3]
        prev_head[valid_mask] = torch.arctan2(dxy[:, 1], dxy[:, 0])[valid_mask]
        prev_pos = pos[:, i].clone()
        prev_pos[valid_mask] = token_contour_gt.mean(1)[valid_mask]

        out_dict["gt_idx"].append(token_idx_gt)
        out_dict["gt_pos"].append(prev_pos.masked_fill(invalid_mask.unsqueeze(1), 0))
        out_dict["gt_heading"].append(prev_head.masked_fill(invalid_mask, 0))
        out_dict["sampled_idx"].append(out_dict["gt_idx"][-1])
        out_dict["sampled_pos"].append(out_dict["gt_pos"][-1])
        out_dict["sampled_heading"].append(out_dict["gt_heading"][-1])

    return {key: torch.stack(value, dim=1) for key, value in out_dict.items()}


def _reference_tokenize_agent(processor: TokenProcessor, data: HeteroData) -> dict[str, torch.Tensor]:
    agent_shape, _, token_traj = processor._get_agent_shape_and_token_traj(
        data["agent"]["type"]
    )
    valid = data["agent"]["valid_mask"].clone()
    heading = data["agent"]["heading"].clone()
    pos = data["agent"]["position"][..., :2].clone().contiguous()
    vel = data["agent"]["velocity"].clone()

    heading = processor._clean_heading(valid, heading)
    valid, pos, heading, _ = processor._extrapolate_agent_to_prev_token_step(
        valid,
        pos,
        heading,
        vel,
    )
    out_by_type = {}
    type_mask = build_agent_type_masks(data["agent"]["type"])
    for agent_type, mask in type_mask.items():
        out_by_type[agent_type] = _reference_match_agent_token_loop(
            processor,
            valid=valid[mask],
            pos=pos[mask],
            heading=heading[mask],
            agent_shape=agent_shape[mask],
            token_traj=token_traj[agent_type],
        )
    return {
        key: merge_by_type(
            {agent_type: value[key] for agent_type, value in out_by_type.items()},
            type_mask,
        )
        for key in ("valid_mask", "gt_idx", "gt_pos", "gt_heading", "sampled_idx", "sampled_pos", "sampled_heading")
    }


def test_agent_token_matching_is_type_aware_argmin() -> None:
    token_bank_veh = torch.tensor(
        [
            [[0.0, 0.0]],
            [[2.0, 0.0]],
            [[4.0, 0.0]],
        ]
    )
    token_bank_ped = torch.tensor(
        [
            [[10.0, 0.0]],
            [[12.0, 0.0]],
            [[14.0, 0.0]],
        ]
    )
    token_bank_cyc = torch.tensor(
        [
            [[20.0, 0.0]],
            [[22.0, 0.0]],
            [[24.0, 0.0]],
        ]
    )
    agent_type = torch.tensor([0, 1, 2, 0, 1, 2])
    contour_local = torch.tensor(
        [
            [[3.7, 0.0]],
            [[10.2, 0.0]],
            [[23.8, 0.0]],
            [[-0.1, 0.0]],
            [[13.7, 0.0]],
            [[21.8, 0.0]],
        ]
    )

    token_idx = match_token_idx_from_local_contour(
        agent_type=agent_type,
        contour_local=contour_local,
        token_bank_all_veh=token_bank_veh,
        token_bank_all_ped=token_bank_ped,
        token_bank_all_cyc=token_bank_cyc,
        reduction="sum",
    )

    torch.testing.assert_close(token_idx, torch.tensor([2, 0, 2, 0, 2, 1]))


def test_local_agent_token_matching_matches_previous_global_expansion_loop() -> None:
    processor = _make_processor()
    data = _make_agent_data()

    actual = processor.tokenize_agent(data)
    expected = _reference_tokenize_agent(processor, data)

    for key in ("valid_mask", "gt_idx", "sampled_idx"):
        torch.testing.assert_close(actual[key], expected[key], atol=0.0, rtol=0.0)
    for key in ("gt_pos", "gt_heading", "sampled_pos", "sampled_heading"):
        torch.testing.assert_close(actual[key], expected[key], atol=1.0e-5, rtol=1.0e-5)
