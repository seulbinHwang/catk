#!/usr/bin/env python3
"""Launch fmsf4-style closed-loop SF-anchor fine-tuning on hsb-npc-training-3-1/3-2."""

from __future__ import annotations

from launch_closed_loop_self_forced_fmsf4_sf_anchor_variants import launch_variant


if __name__ == "__main__":
    raise SystemExit(
        launch_variant(
            pod_label="h100x3x2_hsb3",
            pods=["hsb-npc-training-3-1", "hsb-npc-training-3-2"],
            nproc_per_node=3,
            learning_rate="2.5e-5",
            lr_tag="lr2p5e-5",
            session="catk-closed-loop-sf-h100x3x2-hsb3-fmsf4-lr2p5e5",
        )
    )
