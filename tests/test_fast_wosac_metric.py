from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from src.smart.metrics.sim_agents_metrics import (
    _load_waymo_sim_agents_2025_config,
    _prediction_arrays_to_fast_bundle,
)


def test_prediction_tensors_are_packed_for_trajtok_fast_wosac() -> None:
    agent_id = torch.tensor([101, 202], dtype=torch.int64)
    pred_traj = torch.zeros((2, 3, 4, 2), dtype=torch.float32)
    pred_z = torch.ones((2, 3, 4), dtype=torch.float32)
    pred_head = torch.full((2, 3, 4), 0.5, dtype=torch.float32)

    bundle = _prediction_arrays_to_fast_bundle(
        agent_id=agent_id,
        pred_traj=pred_traj,
        pred_z=pred_z,
        pred_head=pred_head,
        device=torch.device("cpu"),
    )

    assert bundle["agent_id"].dtype == torch.int32
    assert tuple(bundle["simulated_states"].shape) == (3, 2, 4, 4)
    torch.testing.assert_close(bundle["simulated_states"][0, :, :, :2], pred_traj[:, 0])
    torch.testing.assert_close(bundle["simulated_states"][0, :, :, 2], pred_z[:, 0])
    torch.testing.assert_close(bundle["simulated_states"][0, :, :, 3], pred_head[:, 0])


@pytest.mark.skipif(
    os.environ.get("CATK_RUN_SLOW_WOSAC") != "1",
    reason="set CATK_RUN_SLOW_WOSAC=1 to compare against Waymo official scorer",
)
def test_fast_wosac_matches_official_scorer_on_local_waymo_scenario() -> None:
    from tools.compare_fast_wosac_metric import _compare_scenario

    scenario_dir = Path("womd_v1_3/cache/SMART/validation_tfrecords_splitted")
    scenario_paths = sorted(scenario_dir.glob("*.tfrecords"))
    if not scenario_paths:
        pytest.skip(f"no split validation TFRecords under {scenario_dir}")

    result = _compare_scenario(
        scenario_paths[0],
        config=_load_waymo_sim_agents_2025_config(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        num_rollouts=32,
        perturb=True,
    )

    assert result["max_abs_error"] <= 1.0e-6
