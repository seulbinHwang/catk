from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from waymo_open_dataset.protos import sim_agents_metrics_pb2
from src.smart.metrics import sim_agents_metrics as sim_agents_metrics_module
from src.smart.metrics.wosac_fast_eval_tool.fast_sim_agents_metrics import (
    estimators as fast_estimators,
    metric_features as fast_metric_features,
)
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


def test_fast_wosac_features_follow_short_prediction_horizon(monkeypatch) -> None:
    horizon_steps = 15
    hist_steps = 11
    full_steps = 91
    n_rollout = 3
    n_agent = 2
    object_ids = torch.tensor([101, 202], dtype=torch.int32)
    tracks = torch.zeros((n_agent, full_steps, 7), dtype=torch.float32)
    tracks[:, :, 3] = 4.0
    tracks[:, :, 4] = 2.0
    tracks[:, :, 5] = 1.5
    tracks[:, :, 6] = 0.0
    tracks[:, :, 0] = torch.arange(full_steps, dtype=torch.float32).unsqueeze(0)
    track_masks = torch.ones((n_agent, full_steps), dtype=torch.bool)
    gt_scenario = {
        "scenario_id": "short-horizon",
        "object_ids": object_ids,
        "object_types": torch.full((n_agent,), 1, dtype=torch.int32),
        "sim_agent_ids": object_ids,
        "predict_agent_ids": object_ids,
        "tracks": tracks,
        "track_masks": track_masks,
        "traffic_signals": [[] for _ in range(full_steps)],
        "lane_ids": [1],
        "lane_polylines": [torch.zeros((2, 3), dtype=torch.float32)],
        "road_edges": [],
    }
    scenario_rollouts = {
        "agent_id": object_ids,
        "simulated_states": torch.zeros((n_rollout, n_agent, horizon_steps, 4), dtype=torch.float32),
    }
    scenario_rollouts["simulated_states"][..., 0] = torch.arange(horizon_steps, dtype=torch.float32)

    def fake_distance_to_nearest_object(*, boxes, valid, evaluated_object_mask):
        assert boxes.shape[2] == horizon_steps
        assert valid.shape[1] == horizon_steps
        return boxes.new_zeros((boxes.shape[0], int(evaluated_object_mask.sum()), horizon_steps))

    def fake_time_to_collision_with_object_in_front(
        *,
        center_x,
        center_y,
        length,
        width,
        heading,
        valid,
        evaluated_object_mask,
        seconds_per_step,
    ):
        del center_y, width, heading, seconds_per_step
        assert center_x.shape[2] == hist_steps + horizon_steps
        assert length.shape[2] == horizon_steps
        assert valid.shape[1] == horizon_steps
        return center_x.new_zeros((center_x.shape[0], int(evaluated_object_mask.sum()), horizon_steps))

    def fake_distance_to_road_edge(*, boxes, valid, evaluated_object_mask, road_edge_polylines, road_edge_tensors=None):
        del road_edge_polylines, road_edge_tensors
        assert boxes.shape[2] == horizon_steps
        assert valid.shape[1] == horizon_steps
        return boxes.new_zeros((boxes.shape[0], int(evaluated_object_mask.sum()), horizon_steps))

    def fake_red_light_violation(
        *,
        center_x,
        center_y,
        valid,
        evaluated_object_mask,
        lane_polylines,
        lane_ids,
        traffic_signals,
        lane_tensor_cache=None,
        traffic_signal_tensor_cache=None,
    ):
        del center_y, lane_polylines, lane_ids, lane_tensor_cache, traffic_signal_tensor_cache
        assert center_x.shape[2] == hist_steps + horizon_steps
        assert valid.shape[1] == hist_steps + horizon_steps
        assert len(traffic_signals) == hist_steps + horizon_steps
        return torch.zeros(
            (center_x.shape[0], int(evaluated_object_mask.sum()), hist_steps + horizon_steps),
            dtype=torch.bool,
            device=center_x.device,
        )

    monkeypatch.setattr(
        fast_metric_features.interaction_features,
        "compute_distance_to_nearest_object",
        fake_distance_to_nearest_object,
    )
    monkeypatch.setattr(
        fast_metric_features.interaction_features,
        "compute_time_to_collision_with_object_in_front",
        fake_time_to_collision_with_object_in_front,
    )
    monkeypatch.setattr(
        fast_metric_features.map_metric_features,
        "compute_distance_to_road_edge",
        fake_distance_to_road_edge,
    )
    monkeypatch.setattr(
        fast_metric_features.traffic_light_features,
        "compute_red_light_violation",
        fake_red_light_violation,
    )

    fast_metric_features.clear_log_feature_cache()
    log_features, sim_features, valid_masks = fast_metric_features.compute_scenario_rollouts_features(
        gt_scenario,
        scenario_rollouts,
        version="2025",
    )

    assert log_features["linear_speed"].shape == (1, n_agent, horizon_steps)
    assert sim_features["linear_speed"].shape == (n_rollout, n_agent, horizon_steps)
    assert sim_features["traffic_light_violation_per_step"].shape == (n_rollout, n_agent, horizon_steps)
    assert valid_masks.shape == (n_agent, horizon_steps)


def test_soft_histogram_estimator_backpropagates_to_sim_values() -> None:
    feature_config = sim_agents_metrics_pb2.SimAgentMetricsConfig.FeatureConfig()
    feature_config.independent_timesteps = False
    feature_config.histogram.min_val = -5.0
    feature_config.histogram.max_val = 5.0
    feature_config.histogram.num_bins = 16
    feature_config.histogram.additive_smoothing_pseudocount = 0.1
    log_values = torch.zeros((2, 3), dtype=torch.float32)
    sim_values = torch.tensor(
        [
            [[-0.2, 0.1, 0.4], [0.3, -0.1, 0.2]],
            [[0.5, 0.4, -0.3], [-0.4, 0.2, 0.1]],
            [[1.0, -0.5, 0.2], [0.7, -0.2, 0.5]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    log_likelihood = fast_estimators.soft_log_likelihood_estimate_timeseries(
        feature_config,
        log_values,
        sim_values,
        histogram_temperature=1.0,
    )
    loss = -log_likelihood.mean()
    loss.backward()

    assert sim_values.grad is not None
    assert torch.isfinite(sim_values.grad).all()
    assert sim_values.grad.abs().sum() > 0.0


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
