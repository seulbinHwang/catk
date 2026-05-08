"""PyTorch reimplementation of Waymo WOSAC metric feature extraction (stage ①).

This package aims to mirror `waymo_open_dataset.wdl_limited.sim_agents_metrics.metric_features`
and friends, while keeping existing project code untouched.
"""

from .types import MetricFeaturesTorch

__all__ = ["MetricFeaturesTorch"]

