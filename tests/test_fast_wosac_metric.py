from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from waymo_open_dataset.protos import sim_agents_metrics_pb2
from src.smart.metrics import sim_agents_metrics as sim_agents_metrics_module
from src.smart.metrics.sim_agents_metrics import (
    SimAgentsMetrics,
    _load_waymo_sim_agents_2025_config,
    _get_scalar_field_names,
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


def test_sim_agents_metric_workers_match_sequential_payload_updates(monkeypatch) -> None:
    field_names = _get_scalar_field_names(
        sim_agents_metrics_pb2.SimAgentMetrics.DESCRIPTOR,
        skip_names=("scenario_id",),
    )

    def fake_compute_scenario_metrics_from_arrays(
        *,
        config,
        scenario_file,
        agent_id,
        pred_traj,
        pred_z,
        pred_head,
        ego_only,
        device=None,
    ):
        del config, pred_traj, pred_z, pred_head, ego_only, device
        scenario_value = float(agent_id[0])
        scenario_metrics = sim_agents_metrics_pb2.SimAgentMetrics(
            scenario_id=str(scenario_file)
        )
        for field_idx, field_name in enumerate(field_names, start=1):
            setattr(scenario_metrics, field_name, scenario_value / field_idx)
        return scenario_metrics

    monkeypatch.setattr(
        sim_agents_metrics_module,
        "_compute_scenario_metrics_from_arrays",
        fake_compute_scenario_metrics_from_arrays,
    )
    payloads = [
        (f"scenario_{idx}", [idx], [], [], [])
        for idx in range(1, 5)
    ]
    sequential_metric = SimAgentsMetrics("bench", max_workers=1)
    worker_metric = SimAgentsMetrics("bench", max_workers=4)

    sequential_metric.update_from_prediction_payloads(payloads)
    worker_metric.update_from_prediction_payloads(payloads)

    sequential_output = sequential_metric.compute()
    worker_output = worker_metric.compute()
    assert set(worker_output) == set(sequential_output)
    for key, sequential_value in sequential_output.items():
        torch.testing.assert_close(worker_output[key], sequential_value)


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
