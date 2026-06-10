#!/usr/bin/env python3
"""Launch fmsf4-style closed-loop SF-anchor fine-tuning on testas A100x7."""

from __future__ import annotations

from launch_closed_loop_self_forced_fmsf4_sf_anchor_variants import launch_variant


if __name__ == "__main__":
    raise SystemExit(
        launch_variant(
            pod_label="a100x7_testas",
            pods=["testas"],
            nproc_per_node=7,
            learning_rate="1.5e-4",
            lr_tag="lr1p5e-4",
            session="catk-closed-loop-sf-a100x7-testas-fmsf4-lr1p5e4",
        )
    )
