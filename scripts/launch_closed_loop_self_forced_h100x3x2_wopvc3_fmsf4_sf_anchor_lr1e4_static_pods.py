#!/usr/bin/env python3
"""Launch fmsf4-style closed-loop SF-anchor fine-tuning on wo-pvc-3-1/3-2."""

from __future__ import annotations

from launch_closed_loop_self_forced_fmsf4_sf_anchor_variants import launch_variant


if __name__ == "__main__":
    raise SystemExit(
        launch_variant(
            pod_label="h100x3x2_wopvc3",
            pods=["wo-pvc-3-1", "wo-pvc-3-2"],
            nproc_per_node=3,
            learning_rate="1.0e-4",
            lr_tag="lr1e-4",
            session="catk-closed-loop-sf-h100x3x2-wopvc3-fmsf4-lr1e4",
        )
    )
