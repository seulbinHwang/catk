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

import torch

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def set_model_for_finetuning(model: torch.nn.Module, finetune: bool) -> None:
    def _unfreeze(module: torch.nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad = True

    if finetune:
        for p in model.parameters():
            p.requires_grad = False

        try:
            _unfreeze(model.agent_encoder.flow_decoder.step_refiner)
            log.info("Unfreezing flow_decoder.step_refiner")
        except Exception:
            log.info("No flow_decoder.step_refiner in model.agent_encoder")

        try:
            _unfreeze(model.agent_encoder.flow_decoder.velocity_head)
            log.info("Unfreezing flow_decoder.velocity_head")
        except Exception:
            log.info("No flow_decoder.velocity_head in model.agent_encoder")
