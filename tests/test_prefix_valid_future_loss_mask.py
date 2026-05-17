from __future__ import annotations

import torch

from src.smart.modules.flow_agent_decoder import SMARTFlowAgentDecoder
from src.smart.modules.kinematic_control import CYCLIST_TYPE_ID, VEHICLE_TYPE_ID
from src.smart.tokens.flow_token_processor import FlowTokenProcessor


def _build_processor(use_prefix_valid_future_loss_mask: bool) -> FlowTokenProcessor:
    """토큰 파일을 읽지 않고 loss mask helper만 테스트할 processor를 만듭니다.

    Args:
        use_prefix_valid_future_loss_mask: prefix-valid 방식을 사용할지 여부입니다.

    Returns:
        FlowTokenProcessor: ``flow_window_steps``와 옵션만 채운 테스트용 객체입니다.
    """
    processor = FlowTokenProcessor.__new__(FlowTokenProcessor)
    processor.flow_window_steps = 5
    processor.use_prefix_valid_future_loss_mask = use_prefix_valid_future_loss_mask
    return processor


def _build_target_processor(use_prefix_valid_future_loss_mask: bool) -> FlowTokenProcessor:
    processor = FlowTokenProcessor.__new__(FlowTokenProcessor)
    processor.training = True
    processor.shift = 5
    processor.flow_window_steps = 20
    processor.flow_target_dim = 4
    processor.use_prefix_valid_future_loss_mask = use_prefix_valid_future_loss_mask
    processor.use_kinematic_control_flow = False
    return processor


def _build_control_target_processor() -> FlowTokenProcessor:
    processor = _build_target_processor(use_prefix_valid_future_loss_mask=True)
    processor.use_kinematic_control_flow = True
    processor.flow_target_dim = 3
    processor.use_holonomic_model_only = False
    processor.control_pos_scale_m = 1.0
    processor.control_vehicle_yaw_scale_rad = 0.025
    processor.control_pedestrian_yaw_scale_rad = 0.20
    processor.control_cyclist_yaw_scale_rad = 0.06
    processor.control_vehicle_no_slip_point_ratio = 0.0
    processor.control_cyclist_no_slip_point_ratio = 0.0
    # 합성 testbed의 큰 teleport가 0.5초 substep alignment에서 6m+ distortion을 만들 수
    # 있으므로 기본 fixture는 filter를 꺼둡니다. 개별 filter 테스트에서만 켭니다.
    processor.control_alignment_filter_enabled = False
    processor.control_alignment_filter_vehicle_max_error_m = 6.0
    processor.control_alignment_filter_cyclist_max_error_m = 2.2

    def match_from_passed_trajectory(valid, pos, heading, agent_type, agent_shape):
        coarse_steps = list(range(processor.shift, valid.shape[1], processor.shift))
        shape = (pos.shape[0], len(coarse_steps))
        token_idx = torch.zeros(shape, dtype=torch.long, device=pos.device)
        return {
            "valid_mask": valid[:, coarse_steps],
            "gt_idx": token_idx,
            "gt_pos": pos[:, coarse_steps],
            "gt_heading": heading[:, coarse_steps],
            "sampled_idx": token_idx,
            "sampled_pos": pos[:, coarse_steps],
            "sampled_heading": heading[:, coarse_steps],
        }

    processor._match_agent_token = match_from_passed_trajectory
    return processor


def _build_tokenized_agent_for_18_context() -> dict[str, torch.Tensor]:
    return {
        "sampled_idx": torch.zeros((1, 18), dtype=torch.long),
        "sampled_pos": torch.zeros((1, 18, 2), dtype=torch.float32),
        "sampled_heading": torch.zeros((1, 18), dtype=torch.float32),
        "valid_mask": torch.ones((1, 18), dtype=torch.bool),
        "type": torch.zeros((1,), dtype=torch.long),
        "shape": torch.tensor([[2.0, 4.8, 1.5]], dtype=torch.float32),
        "token_agent_shape": torch.tensor([[2.0, 4.8]], dtype=torch.float32),
    }


def _build_processed_agent_for_full_womd_horizon() -> dict[str, torch.Tensor]:
    raw_step = torch.arange(91, dtype=torch.float32)
    pos = torch.zeros((1, 91, 2), dtype=torch.float32)
    pos[0, :, 0] = raw_step
    return {
        "valid": torch.ones((1, 91), dtype=torch.bool),
        "pos": pos,
        "heading": torch.zeros((1, 91), dtype=torch.float32),
    }


def test_prefix_valid_future_loss_mask_keeps_only_continuous_prefix() -> None:
    """가까운 미래부터 처음 끊기기 전까지만 True로 남는지 확인합니다."""
    processor = _build_processor(use_prefix_valid_future_loss_mask=True)
    # valid: [n_agent, n_step]
    valid = torch.tensor(
        [
            [True, True, True, True, False, True, True],
            [True, True, False, True, True, True, True],
            [True, True, True, True, True, True, True],
        ],
        dtype=torch.bool,
    )

    # raw_step=1이면 future는 step 2부터 최대 5개입니다.
    loss_mask = processor._build_anchor_future_loss_mask(valid=valid, raw_step=1)

    expected = torch.tensor(
        [
            [True, True, False, False, False],
            [False, False, False, False, False],
            [True, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(loss_mask, expected)


def test_full_window_future_loss_mask_keeps_original_behavior() -> None:
    """옵션이 꺼져 있으면 전체 미래가 유효한 경우에만 모두 True인지 확인합니다."""
    processor = _build_processor(use_prefix_valid_future_loss_mask=False)
    # valid: [n_agent, n_step]
    valid = torch.tensor(
        [
            [True, True, True, True, False, True, True],
            [True, True, True, True, True, True, True],
        ],
        dtype=torch.bool,
    )

    loss_mask = processor._build_anchor_future_loss_mask(valid=valid, raw_step=1)

    expected = torch.tensor(
        [
            [False, False, False, False, False],
            [True, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(loss_mask, expected)


def test_flow_targets_use_18_context_and_16_prefix_valid_anchors() -> None:
    processor = _build_target_processor(use_prefix_valid_future_loss_mask=True)

    out = processor._build_flow_targets(
        data={"agent": {}},
        tokenized_agent=_build_tokenized_agent_for_18_context(),
        processed_agent=_build_processed_agent_for_full_womd_horizon(),
    )

    assert tuple(out["ctx_sampled_idx"].shape) == (1, 18)
    assert tuple(out["flow_train_mask"].shape) == (1, 16)
    assert int(out["flow_train_mask"].sum().item()) == 16
    assert tuple(out["flow_train_clean_norm"].shape) == (16, 20, 4)
    torch.testing.assert_close(
        out["flow_train_loss_mask"].sum(dim=1).cpu(),
        torch.tensor([20] * 13 + [15, 10, 5]),
    )


def test_flow_targets_keep_16_anchor_slots_for_full_window_mode() -> None:
    processor = _build_target_processor(use_prefix_valid_future_loss_mask=False)

    out = processor._build_flow_targets(
        data={"agent": {}},
        tokenized_agent=_build_tokenized_agent_for_18_context(),
        processed_agent=_build_processed_agent_for_full_womd_horizon(),
    )

    assert tuple(out["ctx_sampled_idx"].shape) == (1, 18)
    assert tuple(out["flow_train_mask"].shape) == (1, 16)
    assert out["flow_train_mask"].tolist() == [[True] * 13 + [False] * 3]
    torch.testing.assert_close(
        out["flow_train_loss_mask"].sum(dim=1).cpu(),
        torch.tensor([20] * 13),
    )


def test_control_flow_targets_retokenize_context_from_transition_aligned_future() -> None:
    processor = _build_control_target_processor()
    tokenized_agent = {
        "type": torch.zeros((1,), dtype=torch.long),
        "shape": torch.tensor([[2.0, 4.8, 1.5]], dtype=torch.float32),
        "token_agent_shape": torch.tensor([[2.0, 4.8]], dtype=torch.float32),
    }
    processed_agent = _build_processed_agent_for_full_womd_horizon()
    processed_agent["pos"][0, 11:, 0] = torch.arange(1, 81, dtype=torch.float32)
    processed_agent["pos"][0, 11:, 1] = 1.0

    out = processor._build_flow_targets(
        data={"agent": {}},
        tokenized_agent=tokenized_agent,
        processed_agent=processed_agent,
    )

    assert tuple(out["flow_train_clean_norm"].shape) == (16, 20, 3)
    torch.testing.assert_close(out["flow_train_clean_norm"][0, :, 1], torch.zeros(20))
    torch.testing.assert_close(out["flow_train_clean_metric_norm"][0, :, 1], torch.zeros(20))
    # token 0/1 are observed raw history ending at raw step 5/10; token 2 is the
    # first current-after-observation context token and must come from the
    # transition-aligned future rather than the raw lateral-offset GT.
    torch.testing.assert_close(out["ctx_sampled_pos"][0, 1], processed_agent["pos"][0, 10])
    torch.testing.assert_close(out["ctx_sampled_pos"][0, 2, 1], torch.tensor(0.0))


def test_control_flow_loss_mask_excludes_substeps_with_invalid_block_endpoint() -> None:
    """Block endpoint이 invalid면 prefix-valid 안의 substep도 loss에서 제외해야 합니다."""
    processor = _build_control_target_processor()
    tokenized_agent = {
        "type": torch.zeros((1,), dtype=torch.long),  # vehicle
        "shape": torch.tensor([[2.0, 4.8, 1.5]], dtype=torch.float32),
        "token_agent_shape": torch.tensor([[2.0, 4.8]], dtype=torch.float32),
    }
    processed_agent = _build_processed_agent_for_full_womd_horizon()
    # Make raw step 25 invalid → block (20, 25] endpoint invalid.
    # Place a non-trivial offset so the invalid placeholder (0, 0) would otherwise
    # pull substeps 21..24 strongly toward origin.
    processed_agent["pos"][0, :, 0] = 1000.0
    processed_agent["pos"][0, 25] = torch.tensor([0.0, 0.0])  # placeholder
    processed_agent["valid"][0, 25] = False
    # All other future steps remain valid.

    out = processor._build_flow_targets(
        data={"agent": {}},
        tokenized_agent=tokenized_agent,
        processed_agent=processed_agent,
    )

    # Recover which anchor slots ended up active.
    flow_train_mask = out["flow_train_mask"]  # [n_agent, 16]
    flow_train_loss_mask = out["flow_train_loss_mask"]  # [n_active, 20]

    # 16 anchors at raw_step in [10, 15, ..., 85]. With raw step 25 invalid:
    #   - Anchor at raw_step=25 has current_valid=False -> excluded.
    #   - Anchor at raw_step=20: prefix-valid in window 21..40 would normally cover
    #     21..24 (4 valid steps before invalid 25). aligned_substep_valid masks all
    #     four (block (20, 25] mid-steps with invalid endpoint) -> loss-mask sum
    #     for this anchor becomes 0, so the anchor is dropped.
    active_anchor_offsets = [i for i in range(16) if bool(flow_train_mask[0, i])]
    # Anchor offsets correspond to raw_steps 10, 15, 20, 25, 30, ..., 85.
    # raw_step=25 (offset 3) excluded; raw_step=20 (offset 2) excluded because its
    # loss window only covered 21..24 and all four are masked.
    assert 3 not in active_anchor_offsets  # raw_step=25 anchor dropped (invalid current)
    assert 2 not in active_anchor_offsets  # raw_step=20 anchor dropped (loss all masked)
    # Anchors before raw_step=20 (i.e. raw_step=10, 15) should still be active.
    assert 0 in active_anchor_offsets
    assert 1 in active_anchor_offsets
    # Anchors at raw_step=30+ should still be active.
    for i in (4, 5):
        assert i in active_anchor_offsets

    # For the surviving anchor at raw_step=10 (offset 0), the loss must end at
    # step 20 — substeps 21..24 are masked by aligned_substep_valid.
    raw_steps = [5 * (i + 2) for i in range(16)]
    anchor0_idx = active_anchor_offsets.index(0)
    # raw_step=10 → loss window covers raw 11..30 mapped to indices 0..19. We
    # expect indices 0..9 (steps 11..20) True, indices 10..19 False (steps 21..30
    # where 21..24 are aligned-invalid and 25..30 are raw-invalid by prefix mask).
    expected_mask = torch.tensor([True] * 10 + [False] * 10)
    assert torch.equal(flow_train_loss_mask[anchor0_idx].cpu(), expected_mask)
    del raw_steps


def test_pedestrian_control_flow_keeps_valid_midsteps_before_invalid_endpoint() -> None:
    """Pedestrian은 raw 0.1s holonomic target이므로 invalid endpoint 오염 전파를 받지 않습니다."""
    processor = _build_control_target_processor()
    tokenized_agent = {
        "type": torch.ones((1,), dtype=torch.long),  # pedestrian
        "shape": torch.tensor([[0.8, 0.8, 1.7]], dtype=torch.float32),
        "token_agent_shape": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
    }
    processed_agent = _build_processed_agent_for_full_womd_horizon()
    processed_agent["pos"][0, :, 0] = 1000.0
    processed_agent["pos"][0, 21:25, 1] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    processed_agent["pos"][0, 25] = torch.tensor([0.0, 0.0])  # placeholder
    processed_agent["valid"][0, 25] = False

    out = processor._build_flow_targets(
        data={"agent": {}},
        tokenized_agent=tokenized_agent,
        processed_agent=processed_agent,
    )

    flow_train_mask = out["flow_train_mask"]
    flow_train_loss_mask = out["flow_train_loss_mask"]
    active_anchor_offsets = [i for i in range(16) if bool(flow_train_mask[0, i])]

    assert 3 not in active_anchor_offsets  # raw_step=25 current itself is invalid.
    assert 2 in active_anchor_offsets  # raw_step=20 keeps valid raw steps 21..24.
    anchor2_idx = active_anchor_offsets.index(2)
    expected_mask = torch.tensor([True] * 4 + [False] * 16)
    assert torch.equal(flow_train_loss_mask[anchor2_idx].cpu(), expected_mask)


def test_control_alignment_filter_thresholds_per_agent_type() -> None:
    """vehicle/cyclist max-error threshold가 type별로 다르게 적용되는지 직접 확인합니다."""
    processor = _build_control_target_processor()
    processor.control_alignment_filter_enabled = True
    processor.control_alignment_filter_vehicle_max_error_m = 6.0
    processor.control_alignment_filter_cyclist_max_error_m = 2.2

    # 3개 agent: cyclist(>2.2m), vehicle(<6m), cyclist(<2.2m).
    raw_pos = torch.zeros((3, 30, 2), dtype=torch.float32)
    aligned_pos = raw_pos.clone()
    # 모든 agent에 raw_step=10 이후 distortion을 직접 주입합니다.
    aligned_pos[0, 11:31, 0] = 3.0  # 3m: cyclist 2.2m 기준 초과
    aligned_pos[1, 11:31, 0] = 3.0  # 3m: vehicle 6.0m 기준 이내
    aligned_pos[2, 11:31, 0] = 1.5  # 1.5m: cyclist 2.2m 기준 이내
    agent_type = torch.tensor(
        [CYCLIST_TYPE_ID, VEHICLE_TYPE_ID, CYCLIST_TYPE_ID], dtype=torch.long
    )
    future_loss_mask = torch.ones((3, 20), dtype=torch.bool)

    mask = processor._build_control_alignment_filter_mask(
        raw_pos=raw_pos,
        aligned_pos=aligned_pos,
        agent_type=agent_type,
        raw_step=10,
        future_loss_mask=future_loss_mask,
    )

    assert mask.tolist() == [False, True, True]


def test_control_alignment_filter_disabled_keeps_all_anchors() -> None:
    """enabled=false면 distortion이 커도 anchor를 그대로 둡니다."""
    processor = _build_control_target_processor()
    processor.control_alignment_filter_enabled = False

    raw_pos = torch.zeros((1, 30, 2), dtype=torch.float32)
    aligned_pos = raw_pos.clone()
    aligned_pos[0, 11:31, 0] = 100.0  # 100m: 모든 기본 기준 초과
    agent_type = torch.tensor([VEHICLE_TYPE_ID], dtype=torch.long)
    future_loss_mask = torch.ones((1, 20), dtype=torch.bool)

    mask = processor._build_control_alignment_filter_mask(
        raw_pos=raw_pos,
        aligned_pos=aligned_pos,
        agent_type=agent_type,
        raw_step=10,
        future_loss_mask=future_loss_mask,
    )

    assert mask.tolist() == [True]


def test_control_alignment_filter_ignores_pedestrian() -> None:
    """pedestrian/holonomic target은 raw GT를 그대로 쓰므로 filter에서 무한대 기준을 받습니다."""
    processor = _build_control_target_processor()
    processor.control_alignment_filter_enabled = True

    raw_pos = torch.zeros((1, 30, 2), dtype=torch.float32)
    aligned_pos = raw_pos.clone()
    aligned_pos[0, 11:31, 0] = 1000.0  # 비현실적인 distortion이라도 통과시켜야 합니다.
    agent_type = torch.ones((1,), dtype=torch.long)  # PEDESTRIAN_TYPE_ID
    future_loss_mask = torch.ones((1, 20), dtype=torch.bool)

    mask = processor._build_control_alignment_filter_mask(
        raw_pos=raw_pos,
        aligned_pos=aligned_pos,
        agent_type=agent_type,
        raw_step=10,
        future_loss_mask=future_loss_mask,
    )

    assert mask.tolist() == [True]


def test_anchor_context_uses_mask_width_and_ignores_extra_tail_context() -> None:
    decoder = SMARTFlowAgentDecoder.__new__(SMARTFlowAgentDecoder)
    encoded = torch.arange(2 * 18 * 3, dtype=torch.float32).view(2, 18, 3)
    decoder._encode_context = lambda **kwargs: encoded
    tokenized_agent = {
        "ctx_sampled_idx": torch.zeros((2, 18), dtype=torch.long),
        "ctx_sampled_pos": torch.zeros((2, 18, 2), dtype=torch.float32),
        "ctx_sampled_heading": torch.zeros((2, 18), dtype=torch.float32),
        "ctx_valid": torch.ones((2, 18), dtype=torch.bool),
    }

    out = decoder.build_anchor_context(
        tokenized_agent=tokenized_agent,
        map_feature={},
        anchor_mask=torch.ones((2, 16), dtype=torch.bool),
        flow_clean_norm=torch.zeros((32, 20, 4), dtype=torch.float32),
    )

    assert tuple(out["ctx_hidden_pack"].shape) == (2, 18, 3)
    assert tuple(out["anchor_hidden"].shape) == (2, 16, 3)
    torch.testing.assert_close(out["anchor_hidden"], encoded[:, 1:17])
