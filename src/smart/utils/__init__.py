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

from src.smart.utils.geometry import angle_between_2d_vectors, wrap_angle
from src.smart.utils.rollout import (
    cal_polygon_contour,
    sample_next_gmm_traj,
    sample_next_token_traj,
    transform_to_global,
    transform_to_local,
)
from src.smart.utils.geometry import angle_between_2d_vectors, wrap_angle
from src.smart.utils.rollout import (
    cal_polygon_contour,
    sample_next_gmm_traj,
    sample_next_token_traj,
    transform_to_global,
    transform_to_local,
)
from src.smart.utils.weight_init import weight_init

from src.smart.utils.flow_traj import (
    assemble_4x6_to_21,
    build_anchor_10hz_indices,
    build_current_anchor_feature,
    build_local_future_target,
    build_ot_flow_path,
    chunk_future_21_to_4x6,
    chunk_valid_21_to_4x6,
    match_first_segment_token,
    normalize_sincos,
    overlap_consistency_residual,
    sample_anchor_10hz_indices,
    segment_endpoint_pose_global,
    segment_local_to_global,
)