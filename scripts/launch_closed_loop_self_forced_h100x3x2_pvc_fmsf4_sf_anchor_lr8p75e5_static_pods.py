#!/usr/bin/env python3
"""Launch fmsf4-style closed-loop SF-anchor fine-tuning on pvc-1/pvc-2 H100x3x2."""

from __future__ import annotations

from launch_closed_loop_self_forced_fmsf4_sf_anchor_variants import launch_variant


if __name__ == "__main__":
    raise SystemExit(
        launch_variant(
            pod_label="h100x3x2_pvc",
            pods=["pvc-1", "pvc-2"],
            nproc_per_node=3,
            learning_rate="8.75e-5",
            lr_tag="lr8p75e-5",
            session="catk-closed-loop-sf-h100x3x2-pvc-fmsf4-lr8p75e5",
        )
    )
