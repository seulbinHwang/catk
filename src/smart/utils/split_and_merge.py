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

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def split_by_type(data: Tensor | Dict[str, Tensor], type_mask: Dict[str, Tensor]):
    if not isinstance(data, dict):
        return {agent_type: data[mask] for agent_type, mask in type_mask.items()}

    return {
        agent_type: {
            key: value[mask]
            for key, value in data.items()
        }
        for agent_type, mask in type_mask.items()
    }


def merge_by_type(data: Dict[str, Tensor], type_mask: Dict[str, Tensor]) -> Tensor:
    first_value = next((value for value in data.values() if value.numel() > 0), None)
    if first_value is None:
        first_value = next(iter(data.values()))

    total = sum(int(mask.sum().item()) for mask in type_mask.values())
    output_shape = (total,) + tuple(first_value.shape[1:])
    output = first_value.new_zeros(output_shape)
    for agent_type, value in data.items():
        if agent_type not in type_mask or value.numel() == 0:
            continue
        output[type_mask[agent_type].to(output.device)] = value
    return output
