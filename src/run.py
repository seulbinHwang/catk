# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import List

import hydra
import lightning as L
import torch
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig

from src.smart.road import RoadCacheRefreshCallback
from src.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    print_config_tree,
)
from src.utils.waymo_submission import (
    cleanup_prepared_waymo_storage_state,
    maybe_prepare_waymo_storage_state,
    maybe_submit_waymo_submission,
)

log = RankedLogger(__name__, rank_zero_only=True)

torch.set_float32_matmul_precision("high")


def build_road_cache_callback(cfg: DictConfig) -> RoadCacheRefreshCallback:
    """RoaD fine-tuning용 cache refresh callback을 만든다.

    Args:
        cfg: Hydra 전체 설정이다. ``data``에는 원본 WOMD cache 위치가 있고,
            ``road``에는 RoaD rollout 생성 설정이 있다.

    Returns:
        epoch마다 RoaD cache를 만들고 dataloader가 그 cache를 읽게 하는 callback이다.
    """
    return RoadCacheRefreshCallback(
        original_train_raw_dir=cfg.data.train_raw_dir,
        cache_root_dir=cfg.road.cache_root_dir,
        sampling_scheme=cfg.road.rollout_sampling,
        num_rollouts_per_scenario=cfg.road.num_rollouts_per_scenario,
        generation_batch_size=cfg.road.generation_batch_size,
        num_workers=cfg.road.num_workers,
        pin_memory=cfg.road.pin_memory,
        delete_after_use=cfg.road.delete_after_use,
    )


def run(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    if cfg.action == "road_finetune":
        callbacks.append(build_road_cache_callback(cfg))

    log.info(f"Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    for _logger in logger:
        if isinstance(_logger, WandbLogger):
            _logger.watch(model, log="all")

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    log.info("Logging hyperparameters!")
    log_hyperparameters(
        {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "callbacks": callbacks,
            "logger": logger,
            "trainer": trainer,
        }
    )

    log.info(f"Resuming from ckpt: cfg.ckpt_path={cfg.ckpt_path}")
    if cfg.action == "fit":
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    elif cfg.action == "finetune":
        log.info("Starting finetuning!")
        model.load_state_dict(
            torch.load(cfg.ckpt_path, weights_only=False)["state_dict"],
            strict=False,
        )
        trainer.fit(model=model, datamodule=datamodule)
    elif cfg.action == "road_finetune":
        log.info("Starting RoaD finetuning!")
        model.load_state_dict(
            torch.load(cfg.ckpt_path, weights_only=False)["state_dict"],
            strict=False,
        )
        trainer.fit(model=model, datamodule=datamodule)
    elif cfg.action == "validate":
        log.info("Starting validating!")
        trainer.validate(
            model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
        )
    elif cfg.action == "test":
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


@hydra.main(config_path="../configs/", config_name="run.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.set_printoptions(precision=3)
    prepared_waymo_storage_state = None

    try:
        prepared_waymo_storage_state = maybe_prepare_waymo_storage_state(cfg)

        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)

        run(cfg)
        maybe_submit_waymo_submission(cfg)
    finally:
        cleanup_prepared_waymo_storage_state(prepared_waymo_storage_state)
        log.info("Closing wandb!")
        wandb.finish()
    log.info(f"Output dir: {cfg.paths.output_dir}")


if __name__ == "__main__":
    main()
    log.info("run.py DONE!!!")
