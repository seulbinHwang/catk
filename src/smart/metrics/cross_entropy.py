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

from typing import Optional

import torch
from torch import Tensor, tensor
from torch.nn.functional import cross_entropy
from torchmetrics.metric import Metric

from .utils import (
    STATE_CONDITIONED_TARGET_CHUNK_SIZE,
    get_euclidean_targets,
    get_prob_targets,
    get_prob_targets_from_index,
    match_state_conditioned_trajectory_token_rows,
)
from src.smart.utils import merge_by_type, split_by_type


class CrossEntropy(Metric):

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        use_gt_raw: bool,
        gt_thresh_scale_length: float,  # {"veh": 4.8, "cyc": 2.0, "ped": 1.0}
        label_smoothing: float,
        rollout_as_gt: bool,
        spatial_aware_smoothing: bool = False,
        state_conditioned_target: bool = False,
        state_conditioned_target_chunk_size: int = STATE_CONDITIONED_TARGET_CHUNK_SIZE,
    ) -> None:
        super().__init__()
        self.use_gt_raw = use_gt_raw
        self.gt_thresh_scale_length = gt_thresh_scale_length
        self.label_smoothing = label_smoothing
        self.rollout_as_gt = rollout_as_gt
        self.spatial_aware_smoothing = spatial_aware_smoothing
        self.state_conditioned_target = state_conditioned_target
        self.state_conditioned_target_chunk_size = int(state_conditioned_target_chunk_size)
        self.add_state("loss_sum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        # ! action that goes from [(10->15), ..., (85->90)]
        next_token_logits: Tensor | dict[str, Tensor],  # [n_agent, 16, n_token] or type -> logits
        next_token_valid: Tensor,  # [n_agent, 16]
        # ! for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
        pred_pos: Tensor,  # [n_agent, 18, 2]
        pred_head: Tensor,  # [n_agent, 18]
        pred_valid: Tensor,  # [n_agent, 18]
        # ! for step {5, 10, ..., 90}
        gt_pos_raw: Tensor,  # [n_agent, 18, 2]
        gt_head_raw: Tensor,  # [n_agent, 18]
        gt_valid_raw: Tensor,  # [n_agent, 18]
        # or use the tokenized gt
        gt_pos: Tensor,  # [n_agent, 18, 2]
        gt_head: Tensor,  # [n_agent, 18]
        gt_valid: Tensor,  # [n_agent, 18]
        # ! for tokenization
        token_agent_shape: Tensor,  # [n_agent, 2]
        token_traj: Tensor,  # [n_agent, n_token, 4, 2]
        token_trajectory: Optional[Tensor | dict[str, Tensor]] = None,
        token_contour_trajectory: Optional[Tensor | dict[str, Tensor]] = None,
        gt_pos_segment_raw: Optional[Tensor] = None,  # [n_agent, 16, 5, 2]
        gt_head_segment_raw: Optional[Tensor] = None,  # [n_agent, 16, 5]
        gt_valid_segment_raw: Optional[Tensor] = None,  # [n_agent, 16, 5]
        # ! for filtering intersting agent for training
        train_mask: Optional[Tensor] = None,  # [n_agent]
        # ! for rollout_as_gt
        next_token_action: Optional[Tensor] = None,  # [n_agent, 16, 3]
        gt_idx: Optional[Tensor] = None,  # [n_agent, 16]
        gt_valid_mask: Optional[Tensor] = None,  # [n_agent, 16]
        type_mask: Optional[dict[str, Tensor]] = None,
        pred_idx: Optional[Tensor] = None,
        **kwargs,
    ) -> None:
        use_direct_gt_idx = (
            gt_idx is not None
            and gt_valid_mask is not None
            and pred_idx is None
            and not self.rollout_as_gt
        )
        use_state_conditioned_target = (
            self.state_conditioned_target
            and pred_idx is not None
            and not self.rollout_as_gt
            and token_trajectory is not None
            and (not self.spatial_aware_smoothing or token_contour_trajectory is not None)
            and gt_pos_segment_raw is not None
            and gt_head_segment_raw is not None
            and gt_valid_segment_raw is not None
        )

        if use_state_conditioned_target:
            self._update_state_conditioned_target_loss(
                next_token_logits=next_token_logits,
                next_token_valid=next_token_valid,
                pred_pos=pred_pos,
                pred_head=pred_head,
                gt_pos_segment_raw=gt_pos_segment_raw,
                gt_head_segment_raw=gt_head_segment_raw,
                gt_valid_segment_raw=gt_valid_segment_raw,
                token_traj=token_traj,
                token_trajectory=token_trajectory,
                token_contour_trajectory=token_contour_trajectory,
                train_mask=train_mask,
                type_mask=type_mask,
            )
            return

        if use_direct_gt_idx:
            self._update_direct_gt_idx_loss(
                next_token_logits=next_token_logits,
                next_token_valid=next_token_valid,
                gt_idx=gt_idx,
                gt_valid_mask=gt_valid_mask,
                token_traj=token_traj,
                token_trajectory=token_trajectory,
                token_contour_trajectory=token_contour_trajectory,
                train_mask=train_mask,
                type_mask=type_mask,
            )
            return
        else:
            # ! use raw or tokenized GT
            if self.use_gt_raw:
                gt_pos = gt_pos_raw
                gt_head = gt_head_raw
                gt_valid = gt_valid_raw

            # ! GT is valid if it's close to the rollout.
            if self.gt_thresh_scale_length > 0:
                dist = torch.norm(pred_pos - gt_pos, dim=-1)  # [n_agent, n_step]
                _thresh = token_agent_shape[:, 1] * self.gt_thresh_scale_length
                gt_valid = gt_valid & (dist < _thresh.unsqueeze(1))

            euclidean_target, target_valid = get_euclidean_targets(
                pred_pos=pred_pos,
                pred_head=pred_head,
                pred_valid=pred_valid,
                gt_pos=gt_pos,
                gt_head=gt_head,
                gt_valid=gt_valid,
            )
            if self.rollout_as_gt and (next_token_action is not None):
                euclidean_target = next_token_action

        if isinstance(next_token_logits, dict):
            if type_mask is None:
                raise ValueError("type_mask is required for type-specific token logits.")
            loss_by_type = {}
            gt_idx_by_type = split_by_type(gt_idx, type_mask) if use_direct_gt_idx else None
            target_by_type = (
                split_by_type(euclidean_target, type_mask)
                if not use_direct_gt_idx
                else None
            )
            token_agent_shape_by_type = split_by_type(token_agent_shape, type_mask)
            for agent_type, mask in type_mask.items():
                if not bool(mask.any()) or agent_type not in next_token_logits:
                    continue
                token_trajectory_type = (
                    token_trajectory.get(agent_type)
                    if isinstance(token_trajectory, dict)
                    else None
                )
                token_contour_trajectory_type = (
                    token_contour_trajectory.get(agent_type)
                    if isinstance(token_contour_trajectory, dict)
                    else None
                )
                if use_direct_gt_idx:
                    prob_target = get_prob_targets_from_index(
                        gt_idx=gt_idx_by_type[agent_type],
                        token_traj=token_traj[agent_type],
                        token_contour_trajectory=token_contour_trajectory_type,
                        label_smoothing=self.label_smoothing,
                        spatial_aware_smoothing=self.spatial_aware_smoothing,
                    )
                else:
                    prob_target = get_prob_targets(
                        target=target_by_type[agent_type],
                        token_agent_shape=token_agent_shape_by_type[agent_type],
                        token_traj=token_traj[agent_type],
                        label_smoothing=self.label_smoothing,
                        spatial_aware_smoothing=self.spatial_aware_smoothing,
                    )
                loss_by_type[agent_type] = cross_entropy(
                    next_token_logits[agent_type].transpose(1, 2),
                    prob_target.transpose(1, 2),
                    reduction="none",
                    label_smoothing=0.0 if self.spatial_aware_smoothing else self.label_smoothing,
                )
            loss = merge_by_type(loss_by_type, type_mask)
        else:
            if use_direct_gt_idx:
                prob_target = get_prob_targets_from_index(
                    gt_idx=gt_idx,
                    token_traj=token_traj,
                    token_contour_trajectory=(
                        token_contour_trajectory
                        if isinstance(token_contour_trajectory, Tensor)
                        else None
                    ),
                    label_smoothing=self.label_smoothing,
                    spatial_aware_smoothing=self.spatial_aware_smoothing,
                )
            else:
                prob_target = get_prob_targets(
                    target=euclidean_target,
                    token_agent_shape=token_agent_shape,
                    token_traj=token_traj,
                    label_smoothing=self.label_smoothing,
                    spatial_aware_smoothing=self.spatial_aware_smoothing,
                )
            loss = cross_entropy(
                next_token_logits.transpose(1, 2),
                prob_target.transpose(1, 2),
                reduction="none",
                label_smoothing=0.0 if self.spatial_aware_smoothing else self.label_smoothing,
            )

        # ! weighting final loss [n_agent, n_step]
        loss_weighting_mask = next_token_valid & target_valid
        if self.training and train_mask is not None:
            loss_weighting_mask &= train_mask.unsqueeze(1)  # [n_agent, n_step]

        self.loss_sum += (loss * loss_weighting_mask).sum()
        self.count += (loss_weighting_mask > 0).sum()

    def _update_state_conditioned_target_loss(
        self,
        next_token_logits: Tensor | dict[str, Tensor],
        next_token_valid: Tensor,
        pred_pos: Tensor,
        pred_head: Tensor,
        gt_pos_segment_raw: Tensor,
        gt_head_segment_raw: Tensor,
        gt_valid_segment_raw: Tensor,
        token_traj: Tensor | dict[str, Tensor],
        token_trajectory: Tensor | dict[str, Tensor],
        token_contour_trajectory: Optional[Tensor | dict[str, Tensor]],
        train_mask: Optional[Tensor],
        type_mask: Optional[dict[str, Tensor]],
    ) -> None:
        target_valid = gt_valid_segment_raw.all(dim=-1)
        loss_weighting_mask = next_token_valid & target_valid
        if self.training and train_mask is not None:
            loss_weighting_mask &= train_mask.unsqueeze(1)

        valid_count = loss_weighting_mask.sum()
        if not bool(valid_count):
            self.count += valid_count.to(dtype=self.count.dtype)
            return

        pred_pos_now = pred_pos[:, 1:-1]
        pred_head_now = pred_head[:, 1:-1]

        if isinstance(next_token_logits, dict):
            if type_mask is None:
                raise ValueError("type_mask is required for type-specific token logits.")
            if not isinstance(token_traj, dict) or not isinstance(token_trajectory, dict):
                raise ValueError(
                    "state-conditioned type-specific logits require type-specific "
                    "token_traj and token_trajectory."
                )
            if self.spatial_aware_smoothing and not isinstance(token_contour_trajectory, dict):
                raise ValueError(
                    "spatial-aware state-conditioned loss requires type-specific "
                    "token_contour_trajectory."
                )

            loss_sum = None
            for agent_type, mask in type_mask.items():
                if not bool(mask.any()) or agent_type not in next_token_logits:
                    continue
                type_valid_mask = loss_weighting_mask[mask]
                logits_type = next_token_logits[agent_type]
                if not bool(type_valid_mask.any()):
                    zero_touch = logits_type.sum() * 0.0
                    loss_sum = zero_touch if loss_sum is None else loss_sum + zero_touch
                    continue

                flat_valid = type_valid_mask.reshape(-1)
                logits_valid = logits_type.reshape(-1, logits_type.shape[-1])[flat_valid]
                pred_pos_valid = pred_pos_now[mask].reshape(-1, 2)[flat_valid]
                pred_head_valid = pred_head_now[mask].reshape(-1)[flat_valid]
                gt_pos_segment_valid = gt_pos_segment_raw[mask].reshape(
                    -1, *gt_pos_segment_raw.shape[2:]
                )[flat_valid]
                gt_head_segment_valid = gt_head_segment_raw[mask].reshape(
                    -1, gt_head_segment_raw.shape[-1]
                )[flat_valid]
                token_trajectory_type = token_trajectory[agent_type]
                token_contour_trajectory_type = (
                    token_contour_trajectory[agent_type]
                    if isinstance(token_contour_trajectory, dict)
                    else None
                )
                gt_idx_valid = match_state_conditioned_trajectory_token_rows(
                    pred_pos=pred_pos_valid,
                    pred_head=pred_head_valid,
                    gt_pos_segment=gt_pos_segment_valid,
                    gt_head_segment=gt_head_segment_valid,
                    token_trajectory=token_trajectory_type,
                    chunk_size=self.state_conditioned_target_chunk_size,
                )
                token_traj_rows = self._select_token_traj_rows(
                    token_traj=token_traj[agent_type],
                    valid_mask=type_valid_mask,
                    flat_valid=flat_valid,
                    token_trajectory=token_trajectory_type,
                )
                loss_type = self._direct_gt_idx_loss_for_rows(
                    logits=logits_valid,
                    gt_idx=gt_idx_valid,
                    token_traj=token_traj_rows,
                    token_trajectory=token_trajectory_type,
                    token_contour_trajectory=token_contour_trajectory_type,
                )
                loss_sum = loss_type if loss_sum is None else loss_sum + loss_type

            if loss_sum is None:
                self.count += valid_count.to(dtype=self.count.dtype)
                return
        else:
            if isinstance(token_trajectory, dict):
                raise ValueError("single token logits require tensor token_trajectory.")
            if self.spatial_aware_smoothing and not isinstance(token_contour_trajectory, Tensor):
                raise ValueError(
                    "spatial-aware state-conditioned loss requires tensor "
                    "token_contour_trajectory."
                )
            flat_valid = loss_weighting_mask.reshape(-1)
            logits_valid = next_token_logits.reshape(-1, next_token_logits.shape[-1])[
                flat_valid
            ]
            gt_idx_valid = match_state_conditioned_trajectory_token_rows(
                pred_pos=pred_pos_now.reshape(-1, 2)[flat_valid],
                pred_head=pred_head_now.reshape(-1)[flat_valid],
                gt_pos_segment=gt_pos_segment_raw.reshape(
                    -1, *gt_pos_segment_raw.shape[2:]
                )[flat_valid],
                gt_head_segment=gt_head_segment_raw.reshape(
                    -1, gt_head_segment_raw.shape[-1]
                )[flat_valid],
                token_trajectory=token_trajectory,
                chunk_size=self.state_conditioned_target_chunk_size,
            )
            token_traj_rows = self._select_token_traj_rows(
                token_traj=token_traj,
                valid_mask=loss_weighting_mask,
                flat_valid=flat_valid,
                token_trajectory=token_trajectory,
            )
            loss_sum = self._direct_gt_idx_loss_for_rows(
                logits=logits_valid,
                gt_idx=gt_idx_valid,
                token_traj=token_traj_rows,
                token_trajectory=token_trajectory,
                token_contour_trajectory=(
                    token_contour_trajectory
                    if isinstance(token_contour_trajectory, Tensor)
                    else None
                ),
            )

        self.loss_sum += loss_sum
        self.count += valid_count.to(dtype=self.count.dtype)

    def _update_direct_gt_idx_loss(
        self,
        next_token_logits: Tensor | dict[str, Tensor],
        next_token_valid: Tensor,
        gt_idx: Tensor,
        gt_valid_mask: Tensor,
        token_traj: Tensor | dict[str, Tensor],
        token_trajectory: Optional[Tensor | dict[str, Tensor]],
        token_contour_trajectory: Optional[Tensor | dict[str, Tensor]],
        train_mask: Optional[Tensor],
        type_mask: Optional[dict[str, Tensor]],
    ) -> None:
        loss_weighting_mask = next_token_valid & gt_valid_mask
        if self.training and train_mask is not None:
            loss_weighting_mask &= train_mask.unsqueeze(1)

        valid_count = loss_weighting_mask.sum()
        if not bool(valid_count):
            self.count += valid_count.to(dtype=self.count.dtype)
            return

        if isinstance(next_token_logits, dict):
            if type_mask is None:
                raise ValueError("type_mask is required for type-specific token logits.")
            if not isinstance(token_traj, dict):
                raise ValueError("type-specific token logits require type-specific token_traj.")

            loss_sum = None
            for agent_type, mask in type_mask.items():
                if not bool(mask.any()) or agent_type not in next_token_logits:
                    continue
                type_valid_mask = loss_weighting_mask[mask]
                logits_type = next_token_logits[agent_type]
                if not bool(type_valid_mask.any()):
                    zero_touch = logits_type.sum() * 0.0
                    loss_sum = zero_touch if loss_sum is None else loss_sum + zero_touch
                    continue

                gt_idx_type = gt_idx[mask]
                flat_valid = type_valid_mask.reshape(-1)
                logits_valid = logits_type.reshape(-1, logits_type.shape[-1])[flat_valid]
                gt_idx_valid = gt_idx_type.reshape(-1)[flat_valid]
                token_trajectory_type = (
                    token_trajectory.get(agent_type)
                    if isinstance(token_trajectory, dict)
                    else None
                )
                token_contour_trajectory_type = (
                    token_contour_trajectory.get(agent_type)
                    if isinstance(token_contour_trajectory, dict)
                    else None
                )
                token_traj_type = token_traj[agent_type]
                token_traj_rows = self._select_token_traj_rows(
                    token_traj=token_traj_type,
                    valid_mask=type_valid_mask,
                    flat_valid=flat_valid,
                    token_trajectory=token_trajectory_type,
                )

                loss_type = self._direct_gt_idx_loss_for_rows(
                    logits=logits_valid,
                    gt_idx=gt_idx_valid,
                    token_traj=token_traj_rows,
                    token_trajectory=token_trajectory_type,
                    token_contour_trajectory=token_contour_trajectory_type,
                )
                loss_sum = loss_type if loss_sum is None else loss_sum + loss_type

            if loss_sum is None:
                self.count += valid_count.to(dtype=self.count.dtype)
                return
        else:
            flat_valid = loss_weighting_mask.reshape(-1)
            logits_valid = next_token_logits.reshape(-1, next_token_logits.shape[-1])[
                flat_valid
            ]
            gt_idx_valid = gt_idx.reshape(-1)[flat_valid]
            token_trajectory_tensor = (
                token_trajectory if isinstance(token_trajectory, Tensor) else None
            )
            token_traj_rows = self._select_token_traj_rows(
                token_traj=token_traj,
                valid_mask=loss_weighting_mask,
                flat_valid=flat_valid,
                token_trajectory=token_trajectory_tensor,
            )
            loss_sum = self._direct_gt_idx_loss_for_rows(
                logits=logits_valid,
                gt_idx=gt_idx_valid,
                token_traj=token_traj_rows,
                token_trajectory=token_trajectory_tensor,
                token_contour_trajectory=(
                    token_contour_trajectory
                    if isinstance(token_contour_trajectory, Tensor)
                    else None
                ),
            )

        self.loss_sum += loss_sum
        self.count += valid_count.to(dtype=self.count.dtype)

    @staticmethod
    def _select_token_traj_rows(
        token_traj: Tensor,
        valid_mask: Tensor,
        flat_valid: Tensor,
        token_trajectory: Optional[Tensor],
    ) -> Tensor:
        if token_trajectory is not None:
            return token_traj

        n_step = valid_mask.shape[1]
        return (
            token_traj[:, None, :, :, :]
            .expand(-1, n_step, -1, -1, -1)
            .reshape(-1, *token_traj.shape[1:])[flat_valid]
        )

    def _direct_gt_idx_loss_for_rows(
        self,
        logits: Tensor,
        gt_idx: Tensor,
        token_traj: Tensor,
        token_trajectory: Optional[Tensor],
        token_contour_trajectory: Optional[Tensor],
    ) -> Tensor:
        if self.spatial_aware_smoothing:
            prob_target = get_prob_targets_from_index(
                gt_idx=gt_idx.unsqueeze(1),
                token_traj=token_traj,
                token_contour_trajectory=token_contour_trajectory,
                label_smoothing=self.label_smoothing,
                spatial_aware_smoothing=self.spatial_aware_smoothing,
            ).squeeze(1)
            return cross_entropy(
                logits,
                prob_target,
                reduction="sum",
                label_smoothing=0.0,
            )

        return cross_entropy(
            logits,
            gt_idx,
            reduction="sum",
            label_smoothing=self.label_smoothing,
        )

    def compute(self) -> Tensor:
        return self.loss_sum / self.count
