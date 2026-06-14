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


def _get_finetune_option(finetune_cfg, key: str, default: bool) -> bool:
    if finetune_cfg is None or isinstance(finetune_cfg, bool):
        return default

    getter = getattr(finetune_cfg, "get", None)
    if callable(getter):
        value = getter(key, default)
    else:
        value = getattr(finetune_cfg, key, default)
    return default if value is None else bool(value)


def _is_finetune_enabled(finetune_cfg) -> bool:
    if isinstance(finetune_cfg, bool):
        return finetune_cfg
    return _get_finetune_option(finetune_cfg, "enabled", False)


def set_model_for_finetuning(
    model: torch.nn.Module,
    finetune=None,
    *,
    finetune_cfg=None,
) -> None:
    def _unfreeze(module: torch.nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad = True

    def _freeze(module: torch.nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad = False

    resolved_finetune_cfg = finetune if finetune_cfg is None else finetune_cfg

    if _is_finetune_enabled(resolved_finetune_cfg):
        if _get_finetune_option(resolved_finetune_cfg, "freeze_smart_map_decoder_only", False):
            for p in model.parameters():
                p.requires_grad = True
            if not hasattr(model, "map_encoder"):
                raise ValueError("freeze_smart_map_decoder_only requires model.map_encoder")
            _freeze(model.map_encoder)
            log.info("Freezing SMARTMapDecoder/map_encoder only for finetuning")
            return

        for p in model.parameters():
            p.requires_grad = False

        if _get_finetune_option(resolved_finetune_cfg, "train_full_flow_decoder_only", False):
            try:
                _unfreeze(model.agent_encoder.flow_decoder)
                log.info("Unfreezing full agent_encoder.flow_decoder")
            except Exception:
                log.info("No flow_decoder in model.agent_encoder")
            return

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
