from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from src.smart.tokens.trajtok import TrajTok
from src.smart.utils import wrap_angle


def test_trajtok_grid_representative_uses_circular_heading_mean() -> None:
    trajs = np.zeros((2, 6, 3), dtype=np.float64)
    trajs[:, :, 0] = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
    trajs[:, :, 1] = np.array([[0, 0, 0, 0, 0, 0], [0, 2, 2, 2, 2, 2]])
    trajs[0, :, 2] = np.deg2rad(179.0)
    trajs[1, :, 2] = np.deg2rad(-179.0)

    mean_traj, concentration = TrajTok._mean_traj_with_circular_heading(trajs)

    assert mean_traj.shape == (6, 3)
    np.testing.assert_allclose(mean_traj[:, :2], trajs[:, :, :2].mean(axis=0))
    assert np.all(np.abs(wrap_angle(mean_traj[:, 2] - np.pi)) < np.deg2rad(2.0))
    assert concentration.shape == (6,)
    assert np.all(concentration > 0.99)


def test_trajtok_interpolation_preserves_endpoint_heading() -> None:
    generator = object.__new__(TrajTok)

    for theta in (0.0, np.deg2rad(45.0), np.deg2rad(-135.0)):
        curve = generator.interpolate_curve(4.0, 1.5, theta)
        assert curve.shape == (6, 3)
        np.testing.assert_allclose(curve[0], np.array([0.0, 0.0, 0.0]), atol=1e-12)
        np.testing.assert_allclose(curve[-1, :2], np.array([4.0, 1.5]), atol=1e-12)
        assert abs(wrap_angle(curve[-1, 2] - theta)) < 1e-12
        assert np.isfinite(curve).all()


def test_endpoint_assignment_uses_containing_cell_and_centers_are_midpoints() -> None:
    generator = object.__new__(TrajTok)
    generator.x_min = {"veh": 0.0}
    generator.x_max = {"veh": 2.0}
    generator.y_min = {"veh": -1.0}
    generator.y_max = {"veh": 1.0}
    generator.x_binnum = {"veh": 4}
    generator.y_binnum = {"veh": 4}

    endpoints = np.array(
        [
            [0.01, -0.99],
            [0.49, -0.51],
            [0.50, -0.50],
            [1.99, 0.99],
            [2.00, 0.00],
        ],
        dtype=np.float64,
    )

    grid_x, grid_y, valid = generator._grid_indices_from_endpoints(endpoints, "veh")

    np.testing.assert_array_equal(grid_x[:4], np.array([0, 0, 1, 3]))
    np.testing.assert_array_equal(grid_y[:4], np.array([0, 0, 1, 3]))
    np.testing.assert_array_equal(valid, np.array([True, True, True, True, False]))
    assert generator._grid_center("veh", 0, 0) == (0.25, -0.75)
    assert generator._grid_center("veh", 3, 3) == (1.75, 0.75)


def test_non_empty_grid_keeps_mean_trajectory_endpoint_not_cell_center() -> None:
    generator = object.__new__(TrajTok)
    generator.shift = 5
    generator.agent_classes = ["veh"]
    generator.flip_trajs = False
    generator.x_max = {"veh": 2.0}
    generator.x_min = {"veh": 0.0}
    generator.y_max = {"veh": 2.0}
    generator.y_min = {"veh": 0.0}
    generator.x_binnum = {"veh": 2}
    generator.y_binnum = {"veh": 2}
    generator.valid_count_threshold = {"veh": 1}
    generator.filter_range = {"veh": 1}
    generator.filter_threshold_add = {"veh": 99}
    generator.filter_threshold_remove = {"veh": 0}
    generator.max_traj_nums = None
    generator.sample_seed = 2025
    generator.enforce_paper_vocab_size = False
    generator.target_vocab_size = {}

    trajs = np.zeros((2, 5, 3), dtype=np.float64)
    trajs[0, :, 0] = np.linspace(0.02, 0.20, 5)
    trajs[1, :, 0] = np.linspace(0.04, 0.30, 5)
    trajs[:, :, 1] = 0.2
    trajs[:, :, 2] = 0.1
    generator.traj_data = {"veh": trajs}

    with tempfile.TemporaryDirectory() as tmp_dir:
        generator.output_path = Path(tmp_dir) / "vocab.pkl"
        generator.get_trajtok_vocab()

    mean_endpoint = np.array([0.25, 0.2])
    cell_center = np.array(generator._grid_center("veh", 0, 0))
    np.testing.assert_allclose(generator.vocab["traj"]["veh"][0, -1, :2], mean_endpoint)
    assert not np.allclose(mean_endpoint, cell_center)


def test_target_size_calibration_exact_on_synthetic_grid() -> None:
    generator = object.__new__(TrajTok)
    generator.target_vocab_size = {"veh": 3}
    generator.enforce_paper_vocab_size = True
    generator.filter_threshold_add = {"veh": 1}
    generator.filter_threshold_remove = {"veh": 1}
    generator.filter_threshold_search_radius = 0

    grid_mask = np.array(
        [
            [True, False, False],
            [False, False, False],
            [False, False, True],
        ],
        dtype=bool,
    )
    grid_mask_filtered = grid_mask.copy()
    neighbor_counts = np.array(
        [
            [2, 2, 1],
            [2, 2, 1],
            [1, 1, 2],
        ],
        dtype=np.int32,
    )
    grid_mask_count = grid_mask.astype(np.int64)

    calibrated = generator._calibrate_grid_mask_to_target(
        "veh",
        grid_mask,
        grid_mask_filtered,
        neighbor_counts,
        grid_mask_count,
    )

    assert int(calibrated.sum()) == 3
    assert calibrated[0, 0]
    assert calibrated[2, 2]


def test_grid_stats_path_builds_circular_mean_for_non_empty_cell() -> None:
    generator = object.__new__(TrajTok)
    generator.shift = 5
    generator.agent_classes = ["veh"]
    generator.x_max = {"veh": 2.0}
    generator.x_min = {"veh": 0.0}
    generator.y_max = {"veh": 2.0}
    generator.y_min = {"veh": 0.0}
    generator.x_binnum = {"veh": 2}
    generator.y_binnum = {"veh": 2}

    stats = generator._new_grid_stats()["veh"]
    trajs = torch.zeros((2, 6, 3), dtype=torch.float32)
    trajs[:, :, 0] = torch.tensor(
        [
            [0.0, 0.02, 0.04, 0.06, 0.08, 0.20],
            [0.0, 0.04, 0.08, 0.12, 0.16, 0.30],
        ]
    )
    trajs[:, :, 1] = 0.2
    trajs[0, :, 2] = np.deg2rad(179.0)
    trajs[1, :, 2] = np.deg2rad(-179.0)

    generator._accumulate_class_trajs_to_grid_stats(trajs, "veh", stats)
    counts, mean_traj_in_bin, heading_concentration_in_bin = generator._mean_trajs_from_grid_stats("veh", stats)

    assert int(counts.sum()) == 2
    np.testing.assert_allclose(mean_traj_in_bin[0][0][-1, :2], np.array([0.25, 0.2]), atol=1e-6)
    assert abs(wrap_angle(mean_traj_in_bin[0][0][-1, 2] - np.pi)) < np.deg2rad(2.0)
    assert heading_concentration_in_bin[0][0][-1] > 0.99


def test_trajtok_sparse_filtered_source_uses_raw_non_empty_grid() -> None:
    generator = object.__new__(TrajTok)
    generator.shift = 5
    generator.agent_classes = ["veh"]
    generator.flip_trajs = False
    generator.x_max = {"veh": 3.0}
    generator.x_min = {"veh": 0.0}
    generator.y_max = {"veh": 1.0}
    generator.y_min = {"veh": 0.0}
    generator.x_binnum = {"veh": 3}
    generator.y_binnum = {"veh": 1}
    generator.valid_count_threshold = {"veh": 1}
    generator.filter_range = {"veh": 1}
    generator.filter_threshold_add = {"veh": 0}
    generator.filter_threshold_remove = {"veh": 2}
    generator.max_traj_nums = None
    generator.sample_seed = 2025

    traj = np.zeros((1, 5, 3), dtype=np.float64)
    traj[0, :, 0] = np.linspace(0.02, 0.10, 5)
    traj[0, :, 1] = 0.1
    traj[0, :, 2] = 0.25
    generator.traj_data = {"veh": traj}

    with tempfile.TemporaryDirectory() as tmp_dir:
        generator.output_path = Path(tmp_dir) / "vocab.pkl"
        generator.get_trajtok_vocab()

    out_traj = generator.vocab["traj"]["veh"]
    out_token_all = generator.vocab["token_all"]["veh"]
    assert out_traj.shape == (1, 6, 3)
    assert out_token_all.shape == (1, 6, 4, 2)
    assert np.isfinite(out_traj).all()
    assert abs(wrap_angle(out_traj[0, -1, 2] - 0.25)) < 1e-12
