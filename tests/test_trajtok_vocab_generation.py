from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

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
