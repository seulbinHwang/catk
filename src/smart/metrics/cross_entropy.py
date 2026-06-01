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
    SPATIAL_SMOOTHING_MODE_THINKLAB,
    get_euclidean_targets,
    get_prob_targets,
    get_prob_targets_from_index,
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
        spatial_aware_smoothing_mode: str = SPATIAL_SMOOTHING_MODE_THINKLAB,
    ) -> None:
        super().__init__()
        self.use_gt_raw = use_gt_raw
        self.gt_thresh_scale_length = gt_thresh_scale_length
        self.label_smoothing = label_smoothing
        self.rollout_as_gt = rollout_as_gt
        self.spatial_aware_smoothing = spatial_aware_smoothing
        self.spatial_aware_smoothing_mode = spatial_aware_smoothing_mode
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

        if use_direct_gt_idx:
            target_valid = gt_valid_mask
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
                if use_direct_gt_idx:
                    prob_target = get_prob_targets_from_index(
                        gt_idx=gt_idx_by_type[agent_type],
                        token_traj=token_traj[agent_type],
                        label_smoothing=self.label_smoothing,
                        spatial_aware_smoothing=self.spatial_aware_smoothing,
                        spatial_aware_smoothing_mode=self.spatial_aware_smoothing_mode,
                    )
                else:
                    prob_target = get_prob_targets(
                        target=target_by_type[agent_type],
                        token_agent_shape=token_agent_shape_by_type[agent_type],
                        token_traj=token_traj[agent_type],
                        label_smoothing=self.label_smoothing,
                        spatial_aware_smoothing=self.spatial_aware_smoothing,
                        spatial_aware_smoothing_mode=self.spatial_aware_smoothing_mode,
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
                    label_smoothing=self.label_smoothing,
                    spatial_aware_smoothing=self.spatial_aware_smoothing,
                    spatial_aware_smoothing_mode=self.spatial_aware_smoothing_mode,
                )
            else:
                prob_target = get_prob_targets(
                    target=euclidean_target,
                    token_agent_shape=token_agent_shape,
                    token_traj=token_traj,
                    label_smoothing=self.label_smoothing,
                    spatial_aware_smoothing=self.spatial_aware_smoothing,
                    spatial_aware_smoothing_mode=self.spatial_aware_smoothing_mode,
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

    def compute(self) -> Tensor:
        return self.loss_sum / self.count
