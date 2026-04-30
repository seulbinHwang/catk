"""Standalone module for WOSAC_VERIFY worker functions.

Kept separate from __init__.py so that linter doesn't strip them as 'unused'.
"""
import numpy as np


def real_full_metrics_worker(
    config_bytes,
    scenario_file,
    agent_ids,
    pred_traj_np,
    pred_z_np,
    pred_head_np,
):
    """Run official WOSAC metrics on one scenario, return all sub-metric likelihoods."""
    import tensorflow as tf
    import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as wm
    from waymo_open_dataset.protos import (
        scenario_pb2,
        sim_agents_metrics_pb2,
        sim_agents_submission_pb2,
    )

    tf.config.set_visible_devices([], "GPU")

    config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
    config.ParseFromString(config_bytes)

    scenario = scenario_pb2.Scenario()
    for tfdata in tf.data.TFRecordDataset([scenario_file], compression_type=""):
        scenario.ParseFromString(bytes(tfdata.numpy()))
        break

    n_agents, n_rollout = pred_traj_np.shape[:2]
    joint_scenes = []
    for ir in range(n_rollout):
        sims = []
        for ia in range(n_agents):
            sims.append(
                sim_agents_submission_pb2.SimulatedTrajectory(
                    center_x=pred_traj_np[ia, ir, :, 0],
                    center_y=pred_traj_np[ia, ir, :, 1],
                    center_z=pred_z_np[ia, ir],
                    heading=pred_head_np[ia, ir],
                    object_id=int(agent_ids[ia]),
                )
            )
        joint_scenes.append(
            sim_agents_submission_pb2.JointScene(simulated_trajectories=sims)
        )

    sr = sim_agents_submission_pb2.ScenarioRollouts(
        joint_scenes=joint_scenes, scenario_id=scenario.scenario_id
    )
    result = wm.compute_scenario_metrics_for_bundle(config, scenario, sr)
    return {
        "scenario_id": scenario.scenario_id,
        "metametric": float(result.metametric),
        "linear_speed": float(result.linear_speed_likelihood),
        "linear_acceleration": float(result.linear_acceleration_likelihood),
        "angular_speed": float(result.angular_speed_likelihood),
        "angular_acceleration": float(result.angular_acceleration_likelihood),
        "distance_to_nearest_object": float(
            result.distance_to_nearest_object_likelihood
        ),
        "collision_indication": float(result.collision_indication_likelihood),
        "time_to_collision": float(result.time_to_collision_likelihood),
        "distance_to_road_edge": float(result.distance_to_road_edge_likelihood),
        "offroad_indication": float(result.offroad_indication_likelihood),
        "traffic_light_violation": float(result.traffic_light_violation_likelihood),
    }
