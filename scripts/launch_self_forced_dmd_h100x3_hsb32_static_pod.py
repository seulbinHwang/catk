#!/usr/bin/env python3
"""Launch DMD-style self-forced fine-tuning on hsb-npc-training-3-2 H100x3.

This keeps the hsb-npc-training-3-1 launcher behavior unchanged and only swaps
the pod-specific defaults: pod, clean checkout path, task/session names, and
pretrained checkpoint cache path.
"""

from __future__ import annotations

import launch_self_forced_dmd_h100x3_hsb31_static_pod as base


base.DEFAULT_POD = "hsb-npc-training-3-2"
base.DEFAULT_PROJECT_ROOT = "/tmp/catk_self_forced_dmd_h100x3_hsb32"
base.DEFAULT_EXPERIMENT = "self_forced_npfm_h100_3_hsb32"
base.DEFAULT_PRETRAIN_CKPT = (
    "/workspace/flow_self_forced_dmd_h100x3_hsb32_pretrain_epoch061_x5f9g0ce/"
    "v57/epoch_061.ckpt"
)
base.DEFAULT_PRETRAIN_DOWNLOAD_DIR = (
    "/workspace/flow_self_forced_dmd_h100x3_hsb32_pretrain_epoch061_x5f9g0ce/"
    "v57/artifact"
)
base.DEFAULT_TASK_NAME = (
    "flow_self_forced_dmd_h100x3_hsb32_epoch061_x5f9g0ce_activecontrol_"
    "sample16_backprop8_lr1e-6_bs144_frac025_ep16_middle"
)
base.DEFAULT_SESSION = "catk-self-forced-dmd-h100x3-hsb32"


if __name__ == "__main__":
    base.main()
