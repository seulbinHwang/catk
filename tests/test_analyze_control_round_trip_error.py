from __future__ import annotations

import pickle

import numpy as np
import torch

from src.smart.modules.kinematic_control import VEHICLE_TYPE_ID, decode_control_sequence
from tools.analyze_control_round_trip_error import (
    AnalysisConfig,
    _build_future_loss_mask,
    analyze_cache_file,
)


def _write_cache(path, *, position: torch.Tensor, heading: torch.Tensor, valid: torch.Tensor) -> None:
    num_agent = position.shape[0]
    agent = {
        "position": position,
        "heading": heading,
        "valid_mask": valid,
        "type": torch.full((num_agent,), VEHICLE_TYPE_ID, dtype=torch.uint8),
        "shape": torch.tensor([[4.5, 2.0, 1.5]] * num_agent, dtype=torch.float32),
    }
    with path.open("wb") as handle:
        pickle.dump({"agent": agent}, handle)


def test_future_loss_mask_prefix_uses_complete_chunks_only() -> None:
    valid = torch.ones((4, 35), dtype=torch.bool)
    raw_step = 10
    valid[0, raw_step + 1 + 3] = False
    valid[1, raw_step + 1 + 7] = False
    valid[2, raw_step + 1 + 12] = False

    mask = _build_future_loss_mask(
        valid=valid,
        raw_step=raw_step,
        cfg=AnalysisConfig(flow_window_steps=20, use_prefix_valid_future_loss_mask=True),
    )

    expected_counts = torch.tensor([0, 5, 10, 20])
    torch.testing.assert_close(mask.long().sum(dim=1), expected_counts)


def test_analyze_cache_file_reports_known_vehicle_round_trip_error(tmp_path) -> None:
    position = torch.zeros((1, 91, 3), dtype=torch.float32)
    heading = torch.zeros((1, 91), dtype=torch.float32)
    valid = torch.zeros((1, 91), dtype=torch.bool)
    valid[:, :31] = True
    position[0, 10, :2] = torch.tensor([0.0, 0.0])
    for offset in range(1, 21):
        position[0, 10 + offset, :2] = torch.tensor([float(offset), 6.0])

    path = tmp_path / "lateral_jump.pkl"
    _write_cache(path, position=position, heading=heading, valid=valid)

    cfg = AnalysisConfig(flow_window_steps=20, raw_start=10, raw_end=10, hist_max_error_m=10.0, hist_bins=10_000)
    stats = analyze_cache_file(path, cfg, thresholds=np.array([2.0, 5.0, 10.0]))

    assert stats["all"]["anchor_count"] == 1
    assert stats["vehicle"]["anchor_count"] == 1
    assert stats["all"]["threshold_keep_counts"].tolist() == [0, 0, 1]
    assert stats["vehicle"]["max_anchor_error_m"] == 6.0


def test_analyze_cache_file_zero_for_decoder_consistent_trajectory(tmp_path) -> None:
    agent_type = torch.tensor([VEHICLE_TYPE_ID], dtype=torch.long)
    agent_length = torch.tensor([4.5], dtype=torch.float32)
    current_pos = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    current_head = torch.tensor([0.25], dtype=torch.float32)
    control = torch.zeros((1, 20, 3), dtype=torch.float32)
    control[..., 0] = 0.7
    control[..., 2] = 0.04
    cfg = AnalysisConfig(flow_window_steps=20, raw_start=10, raw_end=10)
    future_pos, future_head = decode_control_sequence(
        control=control,
        agent_type=agent_type,
        agent_length=agent_length,
        current_pos=current_pos,
        current_head=current_head,
        vehicle_no_slip_point_ratio=cfg.control_vehicle_no_slip_point_ratio,
        cyclist_no_slip_point_ratio=cfg.control_cyclist_no_slip_point_ratio,
    )

    position = torch.zeros((1, 91, 3), dtype=torch.float32)
    heading = torch.zeros((1, 91), dtype=torch.float32)
    valid = torch.zeros((1, 91), dtype=torch.bool)
    valid[:, :31] = True
    position[0, 10, :2] = current_pos[0]
    heading[0, 10] = current_head[0]
    position[0, 11:31, :2] = future_pos[0]
    heading[0, 11:31] = future_head[0]

    path = tmp_path / "consistent.pkl"
    _write_cache(path, position=position, heading=heading, valid=valid)

    stats = analyze_cache_file(path, cfg, thresholds=np.array([0.001, 0.01]))

    assert stats["all"]["anchor_count"] == 1
    assert stats["all"]["threshold_keep_counts"].tolist() == [1, 1]
    assert stats["all"]["max_anchor_error_m"] < 1.0e-5
