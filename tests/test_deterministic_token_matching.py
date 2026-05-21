import torch

from src.smart.tokens.agent_token_matching import match_token_idx_from_local_contour
from src.smart.tokens.flow_token_processor import FlowTokenProcessor


def test_agent_token_matching_is_type_aware_argmin() -> None:
    token_bank_veh = torch.tensor(
        [
            [[[0.0, 0.0]]],
            [[[2.0, 0.0]]],
            [[[4.0, 0.0]]],
        ]
    )
    token_bank_ped = torch.tensor(
        [
            [[[10.0, 0.0]]],
            [[[12.0, 0.0]]],
            [[[14.0, 0.0]]],
        ]
    )
    token_bank_cyc = torch.tensor(
        [
            [[[20.0, 0.0]]],
            [[[22.0, 0.0]]],
            [[[24.0, 0.0]]],
        ]
    )
    agent_type = torch.tensor([0, 1, 2, 0, 1, 2])
    contour_local = torch.tensor(
        [
            [[[3.7, 0.0]]],
            [[[10.2, 0.0]]],
            [[[23.8, 0.0]]],
            [[[-0.1, 0.0]]],
            [[[13.7, 0.0]]],
            [[[21.8, 0.0]]],
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


def test_flow_token_processor_no_longer_exposes_topk_sampling_config() -> None:
    processor = FlowTokenProcessor(
        map_token_file="map_traj_token5.pkl",
        agent_token_file="agent_vocab_555_s2.pkl",
        flow_window_steps=20,
        use_kinematic_control_flow=True,
        control_vehicle_yaw_scale_rad=0.025,
        control_pedestrian_yaw_scale_rad=0.20,
        control_cyclist_yaw_scale_rad=0.06,
    )

    assert not hasattr(processor, "map_token_sampling")
    assert not hasattr(processor, "agent_token_sampling")


def test_train_sampled_agent_token_matches_gt_token() -> None:
    processor = FlowTokenProcessor(
        map_token_file="map_traj_token5.pkl",
        agent_token_file="agent_vocab_555_s2.pkl",
        flow_window_steps=20,
        use_kinematic_control_flow=True,
        control_vehicle_yaw_scale_rad=0.025,
        control_pedestrian_yaw_scale_rad=0.20,
        control_cyclist_yaw_scale_rad=0.06,
    )
    processor.train()

    n_agent = 6
    n_step = 16
    valid = torch.ones((n_agent, n_step), dtype=torch.bool)
    pos = torch.zeros((n_agent, n_step, 2), dtype=torch.float32)
    pos[:, :, 0] = torch.arange(n_step, dtype=torch.float32).unsqueeze(0) * 0.2
    pos[:, :, 1] = torch.arange(n_agent, dtype=torch.float32).unsqueeze(1) * 0.1
    heading = torch.zeros((n_agent, n_step), dtype=torch.float32)
    agent_type = torch.tensor([0, 1, 2, 0, 1, 2])
    agent_shape = processor._get_agent_shape(agent_type)

    token_dict = processor._match_agent_token(
        valid=valid,
        pos=pos,
        heading=heading,
        agent_type=agent_type,
        agent_shape=agent_shape,
    )

    torch.testing.assert_close(token_dict["sampled_idx"], token_dict["gt_idx"])
    torch.testing.assert_close(token_dict["sampled_pos"], token_dict["gt_pos"])
    torch.testing.assert_close(token_dict["sampled_heading"], token_dict["gt_heading"])
