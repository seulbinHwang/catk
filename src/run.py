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

import os
from typing import List

import hydra
import lightning as L
import torch
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, open_dict

from src.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    print_config_tree,
)

log = RankedLogger(__name__, rank_zero_only=True)

torch.set_float32_matmul_precision("high")


def _configure_wandb_checkpoint_upload(cfg: DictConfig) -> None:
    logger_cfg = cfg.get("logger")
    if not logger_cfg:
        return

    wandb_cfg = logger_cfg.get("wandb")
    if not wandb_cfg or wandb_cfg.get("log_model") in (False, None):
        return

    wandb_mode = os.getenv("WANDB_MODE", "").strip().lower()
    wandb_disabled = os.getenv("WANDB_DISABLED", "").strip().lower()
    is_offline = bool(wandb_cfg.get("offline")) or wandb_mode in {
        "offline",
        "dryrun",
        "disabled",
    }
    is_disabled = wandb_disabled in {"true", "1", "yes"}
    if not is_offline and not is_disabled:
        return

    with open_dict(wandb_cfg):
        wandb_cfg.log_model = False

    log.warning(
        "Disabled W&B checkpoint artifact upload because W&B is configured for offline/disabled mode."
    )


def run(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    if cfg.trainer.get("accelerator") == "gpu":
        requested_devices = cfg.trainer.get("devices")
        if isinstance(requested_devices, int) and requested_devices > 0:
            visible_cuda_devices = torch.cuda.device_count()
            if visible_cuda_devices < requested_devices:
                raise ValueError(
                    f"Requested {requested_devices} GPU(s), but only {visible_cuda_devices} "
                    "CUDA device(s) are visible. Check CUDA_VISIBLE_DEVICES and trainer.devices."
                )

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    _configure_wandb_checkpoint_upload(cfg)

    log.info(f"Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

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
        model.load_state_dict(torch.load(cfg.ckpt_path)["state_dict"], strict=False)
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

    log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
    print_config_tree(cfg, resolve=True, save_to_file=True)

    run(cfg)  # train/val/test the model

    log.info("Closing wandb!")
    wandb.finish()
    log.info(f"Output dir: {cfg.paths.output_dir}")


if __name__ == "__main__":
    main()
    log.info("run.py DONE!!!")
