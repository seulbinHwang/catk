#!/usr/bin/env python3
"""Launch fmsf4-style closed-loop SF-anchor fine-tuning on testa/testaa A100x4x2."""

from __future__ import annotations

from launch_closed_loop_self_forced_fmsf4_sf_anchor_variants import launch_variant


if __name__ == "__main__":
    raise SystemExit(
        launch_variant(
            pod_label="a100x4x2_testa_testaa",
            pods=["testa", "testaa"],
            nproc_per_node=4,
            learning_rate="1.0e-5",
            lr_tag="lr1e-5",
            session="catk-closed-loop-sf-a100x4x2-testa-testaa-fmsf4-lr1e5",
        )
    )
