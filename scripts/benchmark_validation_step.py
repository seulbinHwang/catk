from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader


@hydra.main(version_base=None, config_path="../configs", config_name="run")
def main(cfg: DictConfig) -> None:
    cfg.paths.cache_root = "/workspace/womd_v1_3/SMART_cache"
    ckpt_path = Path(
        "logs/flow_pretrain_h1006/runs/2026-03-16_05-17-27/checkpoints/epoch_002.ckpt"
    )

    from src.smart.model.smart_flow import SMARTFlow
    from src.utils.sim_agents_utils import (
        get_scenario_id_int_tensor,
        get_scenario_rollouts,
    )

    model = SMARTFlow.load_from_checkpoint(
        ckpt_path.as_posix(),
        model_config=cfg.model.model_config,
        map_location="cuda",
    )
    model = model.eval().cuda()

    datamodule = instantiate(cfg.data)
    datamodule.setup("validate")
    loader = DataLoader(
        datamodule.val_dataset,
        batch_size=24,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader)).cuda()

    with torch.inference_mode():
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        tokenized_map, tokenized_agent = model.token_processor(batch)
        torch.cuda.synchronize()
        timings["tokenize"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        map_feature = model.encoder.encode_map(tokenized_map)
        torch.cuda.synchronize()
        timings["encode_map"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        denoise_pred = model.encoder.forward_from_map_feature(
            map_feature=map_feature,
            tokenized_agent=tokenized_agent,
            anchor_mask_key="flow_eval_mask",
        )
        torch.cuda.synchronize()
        timings["open_context"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = model.encoder.sample_open_loop_future(
            anchor_hidden=denoise_pred["anchor_hidden"],
            anchor_mask=denoise_pred["anchor_mask"],
            sampling_scheme=model.validation_rollout_sampling,
            sampling_seed=model._get_validation_open_seed(0),
        )
        torch.cuda.synchronize()
        timings["open_sample"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_traj, pred_z, pred_head = model._run_closed_loop_rollouts(
            batch,
            tokenized_agent,
            map_feature,
        )
        torch.cuda.synchronize()
        timings["closed_rollouts"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        scenario_rollouts = get_scenario_rollouts(
            scenario_id=get_scenario_id_int_tensor(batch["scenario_id"], pred_traj.device),
            agent_id=batch["agent"]["id"],
            agent_batch=batch["agent"]["batch"],
            pred_traj=pred_traj,
            pred_z=pred_z,
            pred_head=pred_head,
        )
        timings["scenario_rollouts"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        model.sim_agents_metrics.update(batch["tfrecord_path"], scenario_rollouts)
        timings["metric_update"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        metrics = model.sim_agents_metrics.compute()
        timings["metric_compute"] = time.perf_counter() - t0

    output = {
        "timings_seconds": {key: round(value, 4) for key, value in timings.items()},
        "metric_preview": {
            key: float(value)
            for key, value in list(metrics.items())[:5]
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
