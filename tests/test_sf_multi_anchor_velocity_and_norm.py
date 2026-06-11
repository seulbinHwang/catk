from __future__ import annotations

import torch

from src.smart.modules.flow_agent_decoder import SMARTFlowAgentDecoder
from src.smart.modules.kinematic_control import VEHICLE_TYPE_ID
from src.smart.modules.self_forced_multi_anchor import (
    build_anchor_current_pose,
    pack_anchor_invariant,
)
from src.smart.modules.self_forced_path_flow import (
    build_anchor0_normalized_committed_control,
    build_anchor0_normalized_committed_path,
    build_anchor_normalized_committed_path,
    build_packed_normalized_committed_control,
)


CONTROL_YAW_SCALE_KWARGS = {
    "vehicle_yaw_scale_rad": 0.025,
    "pedestrian_yaw_scale_rad": 0.20,
    "cyclist_yaw_scale_rad": 0.06,
}


class _IdentityFlowDecoder:
    """anchor hidden과 noisy path를 그대로 합쳐 routing만 검증하는 stub입니다."""

    def __call__(
        self,
        anchor_hidden: torch.Tensor,
        path_noisy_norm: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        # 각 행에 anchor hidden의 첫 채널을 더해 어떤 anchor hidden이
        # 어떤 packed 행에 연결됐는지 추적할 수 있게 합니다.
        return path_noisy_norm + anchor_hidden[:, :1].unsqueeze(-1).expand_as(path_noisy_norm)


class _IdentityFlowOde:
    def predict_clean_from_velocity(
        self,
        x_t: torch.Tensor,
        velocity: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        return velocity


def _build_stub_decoder(ctx_hidden_pack: torch.Tensor) -> SMARTFlowAgentDecoder:
    decoder = SMARTFlowAgentDecoder.__new__(SMARTFlowAgentDecoder)
    decoder.flow_window_steps = 4
    decoder.flow_state_dim = 3
    decoder.flow_decoder = _IdentityFlowDecoder()
    decoder.flow_ode = _IdentityFlowOde()
    decoder._encode_context = lambda **kwargs: ctx_hidden_pack
    return decoder


def _build_ctx_tokenized_agent(num_agent: int, n_token: int = 18) -> dict:
    return {
        "ctx_sampled_idx": torch.zeros((num_agent, n_token), dtype=torch.long),
        "ctx_sampled_pos": torch.zeros((num_agent, n_token, 2)),
        "ctx_sampled_heading": torch.zeros((num_agent, n_token)),
        "ctx_valid": torch.ones((num_agent, n_token), dtype=torch.bool),
    }


def test_path_flow_velocity_for_anchors_routes_anchor_slots() -> None:
    """anchor offset k의 packed 행은 ctx slot ``1+k`` hidden과 연결되어야 합니다."""
    torch.manual_seed(0)
    num_agent, n_token, hidden = 3, 18, 2
    # ctx slot t의 첫 채널 값을 (agent_idx * 100 + t)로 둬서 추적 가능하게 합니다.
    ctx_hidden_pack = torch.zeros(num_agent, n_token, hidden)
    for agent_idx in range(num_agent):
        for token_idx in range(n_token):
            ctx_hidden_pack[agent_idx, token_idx, 0] = agent_idx * 100 + token_idx

    decoder = _build_stub_decoder(ctx_hidden_pack)
    offsets = [0, 4, 8]
    anchor_mask = torch.tensor(
        [
            [True, False, True],
            [False, True, True],
            [True, True, False],
        ]
    )
    n_valid = int(anchor_mask.sum())
    path_noisy_norm = torch.zeros(n_valid, decoder.flow_window_steps, decoder.flow_state_dim)
    tau = torch.rand(n_valid)

    out = decoder.path_flow_velocity_for_anchors(
        tokenized_agent=_build_ctx_tokenized_agent(num_agent),
        map_feature={},
        path_noisy_norm=path_noisy_norm,
        tau=tau,
        anchor_mask=anchor_mask,
        anchor_offsets=offsets,
    )
    # anchor-major packing: anchor0(agent0,2), anchor4(agent1,2), anchor8(agent0,1)
    expected_first_channel = torch.tensor(
        [
            0 * 100 + 1,
            2 * 100 + 1,
            1 * 100 + 5,
            2 * 100 + 5,
            0 * 100 + 9,
            1 * 100 + 9,
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(out["velocity"][:, 0, 0], expected_first_channel)


def test_path_flow_velocity_anchor0_wrapper_matches_general() -> None:
    torch.manual_seed(1)
    num_agent = 4
    ctx_hidden_pack = torch.randn(num_agent, 18, 2)
    decoder = _build_stub_decoder(ctx_hidden_pack)
    anchor_mask_1d = torch.tensor([True, False, True, True])
    n_valid = int(anchor_mask_1d.sum())
    path_noisy_norm = torch.randn(n_valid, decoder.flow_window_steps, decoder.flow_state_dim)
    tau = torch.rand(n_valid)
    tokenized_agent = _build_ctx_tokenized_agent(num_agent)

    out_wrapper = decoder.path_flow_velocity_for_anchor0(
        tokenized_agent=tokenized_agent,
        map_feature={},
        path_noisy_norm=path_noisy_norm,
        tau=tau,
        anchor_mask=anchor_mask_1d,
    )
    out_general = decoder.path_flow_velocity_for_anchors(
        tokenized_agent=tokenized_agent,
        map_feature={},
        path_noisy_norm=path_noisy_norm,
        tau=tau,
        anchor_mask=anchor_mask_1d.view(-1, 1),
        anchor_offsets=[0],
    )
    torch.testing.assert_close(out_wrapper["velocity"], out_general["velocity"])
    torch.testing.assert_close(out_wrapper["clean"], out_general["clean"])


def test_anchor0_normalized_path_matches_general_with_ctx_slot1() -> None:
    torch.manual_seed(2)
    num_agent, t_rollout, flow_window = 3, 25, 20
    pred_traj = torch.randn(num_agent, t_rollout, 2)
    pred_head = torch.randn(num_agent, t_rollout)
    tokenized_agent = {
        "ctx_sampled_pos": torch.randn(num_agent, 18, 2),
        "ctx_sampled_heading": torch.randn(num_agent, 18),
    }
    legacy = build_anchor0_normalized_committed_path(
        pred_traj_10hz=pred_traj,
        pred_head_10hz=pred_head,
        tokenized_agent=tokenized_agent,
        flow_window_steps=flow_window,
    )
    general = build_anchor_normalized_committed_path(
        pred_traj_10hz=pred_traj,
        pred_head_10hz=pred_head,
        current_pos=tokenized_agent["ctx_sampled_pos"][:, 1],
        current_head=tokenized_agent["ctx_sampled_heading"][:, 1],
        flow_window_steps=flow_window,
    )
    torch.testing.assert_close(legacy, general)


def test_general_normalized_path_uses_per_row_origin() -> None:
    """multi-anchor 복제 행마다 다른 원점이 정확히 적용되어야 합니다."""
    flow_window = 2
    pred_traj = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0]],
            [[1.0, 0.0], [2.0, 0.0]],
        ]
    )
    pred_head = torch.zeros(2, 2)
    current_pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    current_head = torch.zeros(2)
    normalized = build_anchor_normalized_committed_path(
        pred_traj_10hz=pred_traj,
        pred_head_10hz=pred_head,
        current_pos=current_pos,
        current_head=current_head,
        flow_window_steps=flow_window,
        pos_scale_m=1.0,
    )
    torch.testing.assert_close(normalized[0, :, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(normalized[1, :, 0], torch.tensor([0.0, 1.0]))


def test_packed_control_matches_anchor0_control() -> None:
    torch.manual_seed(3)
    committed_path_norm = torch.tensor(
        [
            [
                [1.0 / 20.0, 0.2 / 20.0, 1.0, 0.0],
                [2.0 / 20.0, 0.5 / 20.0, 0.9800666, 0.1986693],
            ]
        ],
        dtype=torch.float32,
    )
    tokenized_agent = {"type": torch.tensor([VEHICLE_TYPE_ID])}
    anchor_mask = torch.tensor([True])

    legacy = build_anchor0_normalized_committed_control(
        committed_path_norm=committed_path_norm,
        tokenized_agent=tokenized_agent,
        anchor_mask=anchor_mask,
        pos_scale_m=1.0,
        **CONTROL_YAW_SCALE_KWARGS,
    )
    packed = build_packed_normalized_committed_control(
        committed_path_norm=committed_path_norm,
        agent_type=pack_anchor_invariant(tokenized_agent["type"], anchor_mask.view(-1, 1)),
        agent_length=None,
        pos_scale_m=1.0,
        **CONTROL_YAW_SCALE_KWARGS,
    )
    torch.testing.assert_close(legacy, packed)


def test_build_anchor_current_pose_matches_anchor0_rule() -> None:
    torch.manual_seed(4)
    tokenized_agent = {
        "ctx_sampled_pos": torch.randn(3, 18, 2),
        "ctx_sampled_heading": torch.randn(3, 18),
    }
    current_pos, current_head = build_anchor_current_pose(
        tokenized_agent=tokenized_agent,
        anchor_offsets=[0],
    )
    torch.testing.assert_close(current_pos[:, 0], tokenized_agent["ctx_sampled_pos"][:, 1])
    torch.testing.assert_close(current_head[:, 0], tokenized_agent["ctx_sampled_heading"][:, 1])
