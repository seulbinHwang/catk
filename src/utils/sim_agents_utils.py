# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import List

import numpy as np
import torch
from torch import Tensor
from waymo_open_dataset.protos import sim_agents_submission_pb2


def _batch_slices(batch: Tensor) -> List[slice]:
    batch_cpu = batch.detach().to(device="cpu", dtype=torch.long)
    sizes = torch.bincount(batch_cpu).tolist()
    start = 0
    slices: List[slice] = []
    for size in sizes:
        end = start + int(size)
        slices.append(slice(start, end))
        start = end
    return slices


def get_scenario_rollouts(
    scenario_id: Tensor,  # [n_scenario, n_str_length]
    agent_id: Tensor,  # [n_agent]
    agent_batch: Tensor,  # [n_agent]
    pred_traj: Tensor,  # [n_agent, n_rollout, n_step, 2]
    pred_z: Tensor,  # [n_agent, n_rollout, n_step]
    pred_head: Tensor,  # [n_agent, n_rollout, n_step]
) -> List[sim_agents_submission_pb2.ScenarioRollouts]:
    scenario_id_np = scenario_id.detach().cpu().numpy()
    agent_slices = _batch_slices(agent_batch)
    agent_id_np = agent_id.detach().cpu().numpy()
    pred_traj_np = pred_traj.detach().cpu().numpy()
    pred_z_np = pred_z.detach().cpu().numpy()
    pred_head_np = pred_head.detach().cpu().numpy()

    n_scenario = scenario_id_np.shape[0]
    n_rollout = int(pred_traj_np.shape[1])
    scenario_rollouts = []
    for i_scenario in range(n_scenario):
        scenario_slice = agent_slices[i_scenario]
        scenario_agent_ids = agent_id_np[scenario_slice]
        scenario_pred_traj = pred_traj_np[scenario_slice]
        scenario_pred_z = pred_z_np[scenario_slice]
        scenario_pred_head = pred_head_np[scenario_slice]
        joint_scenes = []
        for i_rollout in range(n_rollout):
            simulated_trajectories = []
            for i_agent, object_id in enumerate(scenario_agent_ids):
                simulated_trajectories.append(
                    sim_agents_submission_pb2.SimulatedTrajectory(
                        center_x=scenario_pred_traj[i_agent, i_rollout, :, 0],
                        center_y=scenario_pred_traj[i_agent, i_rollout, :, 1],
                        center_z=scenario_pred_z[i_agent, i_rollout],
                        heading=scenario_pred_head[i_agent, i_rollout],
                        object_id=int(object_id),
                    )
                )
            joint_scenes.append(
                sim_agents_submission_pb2.JointScene(
                    simulated_trajectories=simulated_trajectories
                )
            )

        scenario_id_bytes = bytes(
            int(value) for value in scenario_id_np[i_scenario] if int(value) > 0
        )
        _str_scenario_id = scenario_id_bytes.decode("ascii")
        scenario_rollouts.append(
            sim_agents_submission_pb2.ScenarioRollouts(
                joint_scenes=joint_scenes, scenario_id=_str_scenario_id
            )
        )

    return scenario_rollouts


def get_scenario_id_int_tensor(scenario_id: List[str], device: torch.device) -> Tensor:
    scenario_id_int_tensor = []
    for str_id in scenario_id:
        encoded = str_id.encode("ascii")
        int_id = [0] * 16  # max_len of scenario_id string is 16
        if len(encoded) > len(int_id):
            raise ValueError(
                f"Scenario id {str_id!r} is longer than the supported {len(int_id)} bytes."
            )
        for i, value in enumerate(encoded):
            int_id[i] = value
        scenario_id_int_tensor.append(
            torch.tensor(int_id, dtype=torch.uint8, device=device)
        )
    return torch.stack(scenario_id_int_tensor, dim=0)  # [n_scenario, 16]
