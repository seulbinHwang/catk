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

from typing import Any, Dict

import torch

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _resolve_finetune_config(finetune: Any) -> Dict[str, Any]:
    """여러 형태의 finetune 설정을 한 가지 사전으로 정리합니다.

    Args:
        finetune: bool, dict, DictConfig 중 하나입니다.

    Returns:
        Dict[str, Any]:
            최소한 ``enabled`` 와 ``mode`` 키를 가진 사전입니다.
    """
    if isinstance(finetune, bool):
        return {"enabled": finetune, "mode": "legacy"}
    if finetune is None:
        return {"enabled": False, "mode": "legacy"}
    if hasattr(finetune, "items"):
        cfg = dict(finetune.items())
    elif isinstance(finetune, dict):
        cfg = dict(finetune)
    else:
        return {"enabled": bool(finetune), "mode": "legacy"}
    cfg.setdefault("enabled", True)
    cfg.setdefault("mode", "legacy")
    return cfg


def set_model_for_finetuning(model: torch.nn.Module, finetune: Any) -> None:
    """미세조정 모드에 맞게 학습할 파라미터만 켭니다.

    Args:
        model: 실제 학습할 모델입니다.
        finetune: bool 또는 설정 사전입니다.
            ``mode=flow_residual_only`` 이면 residual head만 학습합니다.

    Returns:
        None
    """

    def _unfreeze(module: torch.nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad = True

    finetune_cfg = _resolve_finetune_config(finetune)
    if not bool(finetune_cfg["enabled"]):
        return

    for p in model.parameters():
        p.requires_grad = False

    mode = str(finetune_cfg.get("mode", "legacy"))
    if mode == "flow_residual_only":
        residual_head = getattr(model.agent_encoder.flow_decoder, "residual_velocity_head", None)
        if residual_head is None:
            raise ValueError(
                "flow_residual_only 모드를 쓰려면 residual_velocity_head가 있어야 합니다."
            )
        _unfreeze(residual_head)
        log.info("Unfreezing flow residual_velocity_head only")
        return

    try:
        _unfreeze(model.agent_encoder.token_predict_head)
        log.info("Unfreezing token_predict_head")
    except Exception:
        log.info("No token_predict_head in model.agent_encoder")

    try:
        _unfreeze(model.agent_encoder.gmm_logits_head)
        _unfreeze(model.agent_encoder.gmm_pose_head)
        log.info("Unfreezing gmm heads")
    except Exception:
        log.info("No gmm_logits_head in model.agent_encoder")

    _unfreeze(model.agent_encoder.t_attn_layers)
    _unfreeze(model.agent_encoder.pt2a_attn_layers)
    _unfreeze(model.agent_encoder.a2a_attn_layers)
