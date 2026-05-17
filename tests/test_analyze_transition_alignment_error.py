from __future__ import annotations

import pickle

import numpy as np
import torch

from src.smart.modules.kinematic_control import (
    build_transition_aligned_control_trajectory,
    compute_aligned_substep_validity,
)
from tools.analyze_transition_alignment_error import (
    AlignmentStatsConfig,
    analyze_file,
    analyze_record,
    compute_aligned_substep_validity_np,
    summarize_stats,
    transition_aligned_position_error,
)


CONTROL_YAW_SCALE_KWARGS = {
    "vehicle_yaw_scale_rad": 0.025,
    "pedestrian_yaw_scale_rad": 0.20,
    "cyclist_yaw_scale_rad": 0.06,
}


def test_numpy_transition_alignment_error_matches_torch_helper() -> None:
    torch.manual_seed(3)
    pos = torch.randn(3, 12, 2, dtype=torch.float32).cumsum(dim=1)
    heading = torch.randn(3, 12, dtype=torch.float32) * 0.2
    agent_type = torch.tensor([0, 1, 2], dtype=torch.long)
    agent_length = torch.tensor([4.8, 0.8, 1.9], dtype=torch.float32)
    cfg = AlignmentStatsConfig(
        current_step=2,
        vehicle_no_slip_point_ratio=0.2289518863,
        cyclist_no_slip_point_ratio=0.0495847873,
    )

    error_np = transition_aligned_position_error(
        pos=pos.numpy(),
        heading=heading.numpy(),
        agent_type=agent_type.numpy(),
        agent_length=agent_length.numpy(),
        cfg=cfg,
    )
    aligned_pos, _, _ = build_transition_aligned_control_trajectory(
        pos=pos,
        heading=heading,
        agent_type=agent_type,
        agent_length=agent_length,
        current_step=cfg.current_step,
        vehicle_no_slip_point_ratio=cfg.vehicle_no_slip_point_ratio,
        cyclist_no_slip_point_ratio=cfg.cyclist_no_slip_point_ratio,
        **CONTROL_YAW_SCALE_KWARGS,
    )
    expected = torch.linalg.vector_norm(
        aligned_pos[:, cfg.current_step + 1 :] - pos[:, cfg.current_step + 1 :],
        dim=-1,
    )

    np.testing.assert_allclose(error_np, expected.numpy(), atol=2.0e-5, rtol=2.0e-5)


def _simple_record() -> dict[str, np.ndarray]:
    position = np.zeros((2, 4, 3), dtype=np.float32)
    position[:, :, 0] = np.arange(4, dtype=np.float32)
    position[:, 1:, 1] = 1.0
    heading = np.zeros((2, 4), dtype=np.float32)
    valid = np.ones((2, 4), dtype=bool)
    velocity = np.zeros((2, 4, 2), dtype=np.float32)
    agent_type = np.array([0, 1], dtype=np.int16)
    shape = np.array([[4.8, 2.0, 1.5], [0.8, 0.8, 1.7]], dtype=np.float32)
    return {
        "pos": position[..., :2],
        "heading": heading,
        "valid": valid,
        "vel": velocity,
        "type": agent_type,
        "length": shape[:, 0],
        "shape": shape,
    }


def test_analyze_record_reports_vehicle_distortion_and_pedestrian_zero() -> None:
    cfg = AlignmentStatsConfig(
        current_step=0,
        commit_steps=1,
        flow_window_steps=2,
        num_anchors=2,
        max_future_steps=3,
        vehicle_no_slip_point_ratio=0.0,
        cyclist_no_slip_point_ratio=0.0,
        hist_bins=1000,
    )

    summary = summarize_stats(analyze_record(_simple_record(), cfg), cfg)

    assert summary["vehicle"]["step_error"]["count"] == 3
    assert summary["vehicle"]["step_error"]["mean_m"] == 1.0
    assert summary["pedestrian"]["step_error"]["max_m"] == 0.0
    assert summary["all"]["anchor_window_max_error"]["count"] > 0


def test_prefix_anchor_valid_mode_keeps_short_tail_window() -> None:
    cfg = AlignmentStatsConfig(
        current_step=0,
        commit_steps=1,
        flow_window_steps=2,
        num_anchors=2,
        max_future_steps=3,
        anchor_valid_mode="prefix",
        vehicle_no_slip_point_ratio=0.0,
        cyclist_no_slip_point_ratio=0.0,
        hist_bins=1000,
    )

    summary = summarize_stats(analyze_record(_simple_record(), cfg), cfg)

    assert summary["all"]["anchor_window_max_error"]["count"] == 2


def test_full_anchor_valid_mode_excludes_short_tail_window() -> None:
    cfg = AlignmentStatsConfig(
        current_step=0,
        commit_steps=1,
        flow_window_steps=2,
        num_anchors=2,
        max_future_steps=3,
        anchor_valid_mode="full",
        vehicle_no_slip_point_ratio=0.0,
        cyclist_no_slip_point_ratio=0.0,
        hist_bins=1000,
    )

    summary = summarize_stats(analyze_record(_simple_record(), cfg), cfg)

    assert summary["all"]["anchor_window_max_error"]["count"] == 0


def test_analyze_file_reads_cache_pickle(tmp_path) -> None:
    record = _simple_record()
    agent = {
        "position": torch.from_numpy(
            np.pad(record["pos"], ((0, 0), (0, 0), (0, 1)), mode="constant")
        ),
        "heading": torch.from_numpy(record["heading"]),
        "valid_mask": torch.from_numpy(record["valid"]),
        "velocity": torch.from_numpy(record["vel"]),
        "type": torch.from_numpy(record["type"]),
        "shape": torch.from_numpy(record["shape"]),
    }
    path = tmp_path / "sample.pkl"
    with path.open("wb") as handle:
        pickle.dump({"agent": agent}, handle)
    cfg = AlignmentStatsConfig(
        current_step=0,
        commit_steps=1,
        flow_window_steps=2,
        num_anchors=2,
        max_future_steps=3,
        vehicle_no_slip_point_ratio=0.0,
        cyclist_no_slip_point_ratio=0.0,
    )

    stats = analyze_file(path, cfg)
    summary = summarize_stats(stats, cfg)

    assert summary["all"]["step_error"]["count"] == 6
    assert summary["vehicle"]["agent_max_error"]["max_m"] == 1.0


def test_aligned_substep_validity_numpy_matches_torch() -> None:
    rng = np.random.default_rng(0)
    valid_np = (rng.random((4, 91)) > 0.05)
    valid_np[:, 10] = True  # current_step always valid for this comparison
    torch_result = compute_aligned_substep_validity(
        torch.from_numpy(valid_np), current_step=10, commit_steps=5
    ).numpy()
    np_result = compute_aligned_substep_validity_np(valid_np, current_step=10, commit_steps=5)
    np.testing.assert_array_equal(np_result, torch_result)


def test_analyze_record_masks_invalid_block_endpoint_pollution() -> None:
    # 1 vehicle, 11 raw steps. valid at all but step 5 (invalid endpoint of
    # block (0, 5]). Place the agent far from origin so the (0, 0) placeholder
    # would otherwise produce massive distortion at mid-steps 1..4 and 6..9.
    pos = np.full((1, 11, 2), 1000.0, dtype=np.float32)
    pos[0, 5] = 0.0
    heading = np.zeros((1, 11), dtype=np.float32)
    valid = np.ones((1, 11), dtype=bool)
    valid[0, 5] = False
    vel = np.zeros((1, 11, 2), dtype=np.float32)
    record = {
        "pos": pos,
        "heading": heading,
        "valid": valid,
        "vel": vel,
        "type": np.array([0], dtype=np.int16),
        "length": np.array([4.8], dtype=np.float32),
        "shape": np.array([[4.8, 2.0, 1.5]], dtype=np.float32),
    }
    cfg = AlignmentStatsConfig(
        current_step=0,
        commit_steps=5,
        flow_window_steps=5,
        num_anchors=2,
        max_future_steps=10,
        vehicle_no_slip_point_ratio=0.0,
        cyclist_no_slip_point_ratio=0.0,
        hist_bins=2000,
        hist_max_error_m=2000.0,
        mask_by_aligned_substep_validity=True,
    )

    summary = summarize_stats(analyze_record(record, cfg), cfg)
    # With masking ON, the polluted steps 1..4 and 6..9 are excluded. The block
    # endpoint at step 10 re-anchors (clean) and step 5 is masked by raw valid,
    # so only step 10 survives, with negligible error.
    assert summary["vehicle"]["step_error"]["count"] == 1
    assert summary["vehicle"]["step_error"]["max_m"] < 1.0

    # With masking OFF, the polluted mid-steps slip back in.
    cfg_off = AlignmentStatsConfig(
        current_step=0,
        commit_steps=5,
        flow_window_steps=5,
        num_anchors=2,
        max_future_steps=10,
        vehicle_no_slip_point_ratio=0.0,
        cyclist_no_slip_point_ratio=0.0,
        hist_bins=2000,
        hist_max_error_m=2000.0,
        mask_by_aligned_substep_validity=False,
    )
    summary_off = summarize_stats(analyze_record(record, cfg_off), cfg_off)
    assert summary_off["vehicle"]["step_error"]["count"] > summary["vehicle"]["step_error"]["count"]
    assert summary_off["vehicle"]["step_error"]["max_m"] > 100.0


def test_analyze_record_does_not_mask_pedestrian_valid_midsteps_before_invalid_endpoint() -> None:
    pos = np.full((1, 11, 2), 1000.0, dtype=np.float32)
    pos[0, 1:5, 1] = np.arange(1, 5, dtype=np.float32)
    pos[0, 5] = 0.0
    heading = np.zeros((1, 11), dtype=np.float32)
    valid = np.ones((1, 11), dtype=bool)
    valid[0, 5] = False
    record = {
        "pos": pos,
        "heading": heading,
        "valid": valid,
        "vel": np.zeros((1, 11, 2), dtype=np.float32),
        "type": np.array([1], dtype=np.int16),
        "length": np.array([0.8], dtype=np.float32),
        "shape": np.array([[0.8, 0.8, 1.7]], dtype=np.float32),
    }
    cfg = AlignmentStatsConfig(
        current_step=0,
        commit_steps=5,
        flow_window_steps=5,
        num_anchors=2,
        max_future_steps=10,
        vehicle_no_slip_point_ratio=0.0,
        cyclist_no_slip_point_ratio=0.0,
        hist_bins=2000,
        hist_max_error_m=2000.0,
        mask_by_aligned_substep_validity=True,
    )

    summary = summarize_stats(analyze_record(record, cfg), cfg)

    assert summary["pedestrian"]["step_error"]["count"] == 9
    assert summary["pedestrian"]["step_error"]["max_m"] == 0.0
