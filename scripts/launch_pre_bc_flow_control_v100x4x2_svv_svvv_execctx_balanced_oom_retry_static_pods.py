#!/usr/bin/env python3
"""Launch V100x4x2 execution-context pretrain on svv + svvv.

This is the pod-specific entrypoint for the semi_control_rolling_gan V100x4x2
experiment that should run on the already-running ``svv`` and ``svvv`` pods.
The implementation is shared with the older sv/svv launcher to keep metadata
prebuild, dry-run, stop, and OOM retry behavior identical.
"""

from __future__ import annotations

import launch_pre_bc_flow_control_v100x4x2_sv_svv_execctx_balanced_oom_retry_static_pods as base


base.DEFAULT_PODS = ("svv", "svvv")
base.DEFAULT_TASK_NAME = (
    "flow_control_space_pretrain_v100x4x2_svv_svvv_"
    "execctx_prefix_balanced_lr2e-4_bs2_accum7_oomretry"
)
base.DEFAULT_SESSION = "catk-control-pretrain-v100x4x2-svv-svvv-execctx-balanced"
base.DEFAULT_INITIAL_BS = 2
base.DEFAULT_ACCUMULATE_GRAD_BATCHES = "7"


if __name__ == "__main__":
    raise SystemExit(base.main())
