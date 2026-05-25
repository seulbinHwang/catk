#!/usr/bin/env python3
"""Launch Set-level Self-Forced GAN fine-tuning on hsb-npc-training-1.

This is the single-pod H100x6 entrypoint for the same K=16 set-level
self-forced GAN fine-tuning used by the svv/svvv V100 launcher. The H100 preset
keeps bf16 mixed precision and uses all 6 local GPUs in one torchrun group.
"""

from __future__ import annotations

import launch_self_forced_gan_v100x4x2_svv_svvv_static_pods as base


base.DEFAULT_PODS = ("hsb-npc-training-1",)
base.DEFAULT_TEACHER_CACHE_ROOT = (
    "/workspace/womd_v1_3/SMART_teacher_gan_cache_h100x6_hsb_npc_training_1"
)
base.DEFAULT_EXPERIMENT = "self_forced_gan_h100_6"
base.DEFAULT_TASK_NAME = "sf_gan_k16_h100x6_hsb_npc_training_1"
base.DEFAULT_SESSION = "catk-sf-gan-h100x6-hsb-npc-training-1"
base.DEFAULT_DESCRIPTION = (
    "Launch Set-level Self-Forced GAN fine-tuning on hsb-npc-training-1 H100x6."
)
base.DEFAULT_EXPECTED_POD_COUNT = 1
base.DEFAULT_MASTER_PORT = "29680"
base.DEFAULT_CHECKPOINT_SYNC_PORT = "29681"
base.DEFAULT_NPROC_PER_NODE = 6
base.DEFAULT_TRAIN_BATCH_SIZE = 2
base.DEFAULT_VAL_BATCH_SIZE = 16
base.DEFAULT_PRECISION = "bf16-mixed"
base.DEFAULT_TEACHER_CACHE_GPUS_PER_POD = 6


if __name__ == "__main__":
    raise SystemExit(base.main())
