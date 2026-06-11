from __future__ import annotations

import torch

from src.smart.modules.flow_agent_decoder import SMARTFlowAgentDecoder
from src.smart.modules.self_forced_multi_anchor import (
    build_anchor_current_pose,
    build_multi_anchor_mask,
    pack_anchor_invariant,
    pack_anchor_variant,
    pack_replicated_rows,
    replicate_map_feature_for_anchors,
    replicate_tokenized_agent_for_anchors,
    select_self_forced_anchor_offsets,
)


def _select_kwargs() -> dict:
    return {
        "num_anchor": 16,
        "num_raw_steps": 91,
        "flow_window_steps": 20,
        "shift": 5,
    }


def test_select_anchor_offsets_disabled_returns_anchor0() -> None:
    assert select_self_forced_anchor_offsets(anchor_stride=0, **_select_kwargs()) == [0]
    assert select_self_forced_anchor_offsets(anchor_stride=-1, **_select_kwargs()) == [0]


def test_select_anchor_offsets_stride_rules() -> None:
    # 91 step GT에서 anchor k의 raw step은 5(k+2)이고, 2초(20 step) 미래가
    # 전부 GT 안에 있어야 하므로 k <= 12 까지만 시작점이 될 수 있습니다.
    assert select_self_forced_anchor_offsets(anchor_stride=4, **_select_kwargs()) == [0, 4, 8, 12]
    assert select_self_forced_anchor_offsets(anchor_stride=5, **_select_kwargs()) == [0, 5, 10]
    assert select_self_forced_anchor_offsets(anchor_stride=1, **_select_kwargs()) == list(range(13))
    assert select_self_forced_anchor_offsets(anchor_stride=100, **_select_kwargs()) == [0]


def _build_eval_tokenized_agent(num_agent: int = 4, num_graphs: int = 2) -> dict:
    torch.manual_seed(1)
    n_anchor = 16
    n_token = 18
    history_len = 6
    return {
        "num_graphs": num_graphs,
        "batch": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "type": torch.tensor([0, 1, 2, 0], dtype=torch.long),
        "shape": torch.randn(num_agent, 3),
        "ego_mask": torch.tensor([True, False, False, True]),
        "token_agent_shape": torch.randn(num_agent, 2),
        "trajectory_token_veh": torch.randn(7, 12),
        "trajectory_token_ped": torch.randn(7, 12),
        "trajectory_token_cyc": torch.randn(7, 12),
        "token_bank_all_veh": torch.randn(7, 6, 4, 2),
        "token_bank_all_ped": torch.randn(7, 6, 4, 2),
        "token_bank_all_cyc": torch.randn(7, 6, 4, 2),
        "gt_idx": torch.randint(0, 7, (num_agent, n_token)),
        "gt_pos": torch.randn(num_agent, n_token, 2),
        "gt_heading": torch.randn(num_agent, n_token),
        "valid_mask": torch.rand(num_agent, n_token) > 0.2,
        "gt_pos_raw": torch.randn(num_agent, n_token, 2),
        "gt_head_raw": torch.randn(num_agent, n_token),
        "gt_valid_raw": torch.rand(num_agent, n_token) > 0.2,
        "gt_z_raw": torch.randn(num_agent),
        "ctx_sampled_pos": torch.randn(num_agent, n_token, 2),
        "ctx_sampled_heading": torch.randn(num_agent, n_token),
        "sf_anchor_fine_pos_history": torch.randn(num_agent, n_anchor, history_len, 2),
        "sf_anchor_fine_head_history": torch.randn(num_agent, n_anchor, history_len),
        "sf_anchor_fine_valid_history": torch.rand(num_agent, n_anchor, history_len) > 0.2,
        "sf_anchor_z": torch.randn(num_agent, n_anchor),
        "rollout_init_fine_pos_history": torch.randn(num_agent, history_len, 2),
        "rollout_init_fine_head_history": torch.randn(num_agent, history_len),
        "rollout_init_fine_valid_history": torch.rand(num_agent, history_len) > 0.2,
        "rollout_init_fine_pos_pair": torch.randn(num_agent, 2, 2),
        "rollout_init_fine_head_pair": torch.randn(num_agent, 2),
        "rollout_init_fine_valid_pair": torch.rand(num_agent, 2) > 0.2,
        "flow_eval_mask": torch.rand(num_agent, n_anchor) > 0.3,
    }


def test_replicate_tokenized_agent_anchor0_only_is_passthrough() -> None:
    tokenized_agent = _build_eval_tokenized_agent()
    replicated = replicate_tokenized_agent_for_anchors(
        tokenized_agent=tokenized_agent,
        anchor_offsets=[0],
    )
    assert replicated is tokenized_agent


def test_replicate_tokenized_agent_multi_anchor_layout() -> None:
    tokenized_agent = _build_eval_tokenized_agent()
    offsets = [0, 4, 8, 12]
    n_sel = len(offsets)
    num_agent = 4
    replicated = replicate_tokenized_agent_for_anchors(
        tokenized_agent=tokenized_agent,
        anchor_offsets=offsets,
    )

    assert replicated["num_graphs"] == tokenized_agent["num_graphs"] * n_sel
    # batch는 anchor 복제마다 num_graphs씩 offset되어야 합니다(anchor-major).
    expected_batch = torch.cat(
        [tokenized_agent["batch"] + a * tokenized_agent["num_graphs"] for a in range(n_sel)]
    )
    assert torch.equal(replicated["batch"], expected_batch)

    # anchor-invariant 필드는 anchor-major로 단순 반복되어야 합니다.
    for key in ["type", "shape", "ego_mask", "token_agent_shape", "gt_pos_raw"]:
        for a in range(n_sel):
            block = replicated[key][a * num_agent : (a + 1) * num_agent]
            assert torch.equal(block, tokenized_agent[key])

    # 토큰 bank는 agent 축이 없으므로 그대로 공유되어야 합니다.
    assert replicated["token_bank_all_veh"] is tokenized_agent["token_bank_all_veh"]
    assert replicated["trajectory_token_veh"] is tokenized_agent["trajectory_token_veh"]

    # 토큰 window 필드는 anchor offset만큼 shift되어야 합니다.
    n_token = tokenized_agent["gt_pos"].shape[1]
    for a, offset in enumerate(offsets):
        block = slice(a * num_agent, (a + 1) * num_agent)
        valid_len = n_token - offset
        torch.testing.assert_close(
            replicated["gt_pos"][block][:, :valid_len],
            tokenized_agent["gt_pos"][:, offset:],
        )
        torch.testing.assert_close(
            replicated["gt_heading"][block][:, :valid_len],
            tokenized_agent["gt_heading"][:, offset:],
        )
        assert torch.equal(
            replicated["gt_idx"][block][:, :valid_len],
            tokenized_agent["gt_idx"][:, offset:],
        )
        # shift로 GT 범위를 벗어난 토큰 자리는 valid가 False여야 합니다.
        assert torch.equal(
            replicated["valid_mask"][block][:, :valid_len],
            tokenized_agent["valid_mask"][:, offset:],
        )
        assert not replicated["valid_mask"][block][:, valid_len:].any()

    # fine history/pair와 z는 anchor별 packed 텐서에서 골라져야 합니다.
    for a, offset in enumerate(offsets):
        block = slice(a * num_agent, (a + 1) * num_agent)
        torch.testing.assert_close(
            replicated["rollout_init_fine_pos_history"][block],
            tokenized_agent["sf_anchor_fine_pos_history"][:, offset],
        )
        torch.testing.assert_close(
            replicated["rollout_init_fine_pos_pair"][block],
            tokenized_agent["sf_anchor_fine_pos_history"][:, offset, -2:],
        )
        assert torch.equal(
            replicated["rollout_init_fine_valid_history"][block],
            tokenized_agent["sf_anchor_fine_valid_history"][:, offset],
        )
        torch.testing.assert_close(
            replicated["gt_z_raw"][block],
            tokenized_agent["sf_anchor_z"][:, offset],
        )


def test_replicate_map_feature_layout() -> None:
    torch.manual_seed(2)
    n_pl = 6
    map_feature = {
        "pt_token": torch.randn(n_pl, 8),
        "position": torch.randn(n_pl, 2),
        "orientation": torch.randn(n_pl),
        "light_type": torch.randint(0, 4, (n_pl,)),
        "batch": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
    }
    replicated = replicate_map_feature_for_anchors(
        map_feature=map_feature,
        num_graphs=2,
        num_selected_anchors=3,
    )
    assert replicated["pt_token"].shape == (n_pl * 3, 8)
    for a in range(3):
        block = slice(a * n_pl, (a + 1) * n_pl)
        torch.testing.assert_close(replicated["pt_token"][block], map_feature["pt_token"])
        assert torch.equal(replicated["batch"][block], map_feature["batch"] + a * 2)

    passthrough = replicate_map_feature_for_anchors(
        map_feature=map_feature,
        num_graphs=2,
        num_selected_anchors=1,
    )
    assert passthrough is map_feature


def test_pack_helpers_match_pack_anchor_hidden_order() -> None:
    """pack 헬퍼는 decoder의 ``_pack_anchor_hidden`` packing 순서(anchor-major)와
    정확히 같아야 DMD/critic packed 텐서와 행이 맞습니다."""
    torch.manual_seed(3)
    num_agent, n_sel, hidden = 5, 3, 4
    anchor_hidden = torch.randn(num_agent, n_sel, hidden)
    anchor_mask = torch.rand(num_agent, n_sel) > 0.4
    expected = SMARTFlowAgentDecoder._pack_anchor_hidden(
        None,
        anchor_hidden,
        anchor_mask,
    )
    torch.testing.assert_close(pack_anchor_variant(anchor_hidden, anchor_mask), expected)

    values = torch.randn(num_agent, 7)
    packed = pack_anchor_invariant(values, anchor_mask)
    expected_invariant = torch.cat(
        [values[anchor_mask[:, a]] for a in range(n_sel)],
        dim=0,
    )
    torch.testing.assert_close(packed, expected_invariant)

    replicated_rows = torch.randn(n_sel * num_agent, 7)
    packed_rows = pack_replicated_rows(replicated_rows, anchor_mask)
    expected_rows = torch.cat(
        [replicated_rows[a * num_agent : (a + 1) * num_agent][anchor_mask[:, a]] for a in range(n_sel)],
        dim=0,
    )
    torch.testing.assert_close(packed_rows, expected_rows)


def test_build_multi_anchor_mask_and_current_pose() -> None:
    tokenized_agent = _build_eval_tokenized_agent()
    offsets = [0, 4, 8]
    anchor_mask = build_multi_anchor_mask(
        flow_eval_mask=tokenized_agent["flow_eval_mask"],
        anchor_offsets=offsets,
    )
    assert torch.equal(anchor_mask, tokenized_agent["flow_eval_mask"][:, offsets])

    current_pos, current_head = build_anchor_current_pose(
        tokenized_agent=tokenized_agent,
        anchor_offsets=offsets,
    )
    # anchor k의 원점은 ctx slot 1+k 입니다(anchor0=slot1 규칙의 일반화).
    for a, offset in enumerate(offsets):
        torch.testing.assert_close(
            current_pos[:, a],
            tokenized_agent["ctx_sampled_pos"][:, 1 + offset],
        )
        torch.testing.assert_close(
            current_head[:, a],
            tokenized_agent["ctx_sampled_heading"][:, 1 + offset],
        )
