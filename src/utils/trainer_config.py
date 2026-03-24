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

from omegaconf import DictConfig, open_dict

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def maybe_enable_ddp_find_unused_parameters(cfg: DictConfig) -> bool:
    trainer_cfg = cfg.get("trainer")
    if trainer_cfg is None:
        return False

    strategy_cfg = trainer_cfg.get("strategy")
    if not isinstance(strategy_cfg, DictConfig):
        return False

    if strategy_cfg.get("_target_") != "lightning.pytorch.strategies.DDPStrategy":
        return False
    if bool(strategy_cfg.get("find_unused_parameters", False)):
        return False

    model_cfg = cfg.get("model")
    if model_cfg is None:
        return False
    model_config = model_cfg.get("model_config")
    if model_config is None:
        return False

    finetune_cfg = model_config.get("finetune")
    am_finetune_cfg = model_config.get("am_finetune")
    is_residual_only_finetune = bool(finetune_cfg and finetune_cfg.get("enabled", False)) and (
        str(finetune_cfg.get("mode", "legacy")) == "flow_residual_only"
    )
    is_am_finetune = bool(am_finetune_cfg and am_finetune_cfg.get("enabled", False))
    if cfg.get("action") != "finetune" or not is_residual_only_finetune or not is_am_finetune:
        return False

    with open_dict(strategy_cfg):
        strategy_cfg.find_unused_parameters = True

    log.warning(
        "Enabled DDP find_unused_parameters for AM flow_residual_only finetuning."
    )
    return True
