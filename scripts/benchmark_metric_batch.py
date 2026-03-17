from __future__ import annotations

import json
import os
import time

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from src.smart.metrics.sim_agents_metrics import SimAgentsMetrics


@hydra.main(version_base=None, config_path="../configs", config_name="run")
def main(cfg: DictConfig) -> None:
    cfg.paths.cache_root = "/workspace/womd_v1_3/SMART_cache"
    datamodule = instantiate(cfg.data)
    datamodule.setup("validate")
    batch_size = int(os.environ.get("CATK_BENCH_BATCH_SIZE", "12"))
    batch = next(iter(DataLoader(datamodule.val_dataset, batch_size=batch_size, shuffle=False, num_workers=0))).cuda()

    n_rollout = int(cfg.model.model_config.n_rollout_closed_val)
    future_pos = batch["agent"]["position"][:, 11:, :2]
    future_z = batch["agent"]["position"][:, 11:, 2]
    future_head = batch["agent"]["heading"][:, 11:]
    pred_traj = future_pos[:, None].repeat(1, n_rollout, 1, 1)
    pred_z = future_z[:, None].repeat(1, n_rollout, 1)
    pred_head = future_head[:, None].repeat(1, n_rollout, 1)

    metrics = SimAgentsMetrics("val_closed")
    t0 = time.perf_counter()
    metrics.update_from_prediction_tensors(
        scenario_files=batch["tfrecord_path"],
        agent_id=batch["agent"]["id"],
        agent_batch=batch["agent"]["batch"],
        pred_traj=pred_traj,
        pred_z=pred_z,
        pred_head=pred_head,
    )
    t1 = time.perf_counter()
    computed = metrics.compute()
    t2 = time.perf_counter()
    metrics._shutdown_executor()

    print(
        json.dumps(
            {
                "workers_env": os.environ.get("CATK_SIM_AGENTS_METRIC_WORKERS", ""),
                "batch_size": batch_size,
                "resolved_workers": metrics._max_workers,
                "update_seconds": round(t1 - t0, 4),
                "compute_seconds": round(t2 - t1, 4),
                "total_seconds": round(t2 - t0, 4),
                "metric_preview": {
                    key: float(value)
                    for key, value in list(computed.items())[:5]
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
