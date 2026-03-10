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
import signal
import sys
import traceback
from pathlib import Path
from typing import List

os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_CONSOLE", "off")

import hydra
import lightning as L
import torch
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig

from src.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    print_config_tree,
)

log = RankedLogger(__name__, rank_zero_only=True)

torch.set_float32_matmul_precision("high")


def get_wandb_logger(loggers: List[Logger]) -> WandbLogger | None:
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            return logger
    return None


def log_wandb_checkpoint_refs(cfg: DictConfig, loggers: List[Logger]) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return

    wandb_logger = get_wandb_logger(loggers)
    if wandb_logger is None or wandb_logger.offline:
        return
    if not wandb_logger.log_model:
        return

    run_id = wandb_logger.experiment.id
    entity = wandb_logger.experiment.entity
    project = wandb_logger.experiment.project
    artifact_name = f"model-{run_id}"
    best_ref = f"{entity}/{project}/{artifact_name}:best"
    latest_ref = f"{entity}/{project}/{artifact_name}:latest"

    artifact_refs = {
        "artifact/run_path": f"{entity}/{project}/{run_id}",
        "artifact/best_ckpt_ref": best_ref,
        "artifact/latest_ckpt_ref": latest_ref,
    }
    wandb_logger.experiment.summary.update(artifact_refs)

    refs_path = Path(cfg.paths.output_dir) / "artifact_refs.txt"
    refs_path.write_text(
        "\n".join(
            [
                f"run_path={artifact_refs['artifact/run_path']}",
                f"best={best_ref}",
                f"latest={latest_ref}",
            ]
        )
        + "\n"
    )

    log.info("W&B artifact refs")
    log.info(f"  run_path: {artifact_refs['artifact/run_path']}")
    log.info(f"  best:     {best_ref}")
    log.info(f"  latest:   {latest_ref}")
    log.info(f"  saved:    {refs_path}")


def resolve_ckpt_path(cfg: DictConfig) -> str | None:
    ckpt_artifact = cfg.get("ckpt_artifact")
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_artifact:
        artifact_root = Path(cfg.paths.output_dir) / "wandb_artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        log.info(f"Downloading checkpoint artifact <{ckpt_artifact}>")
        artifact = wandb.Api().artifact(ckpt_artifact)
        download_dir = Path(
            artifact.download(root=(artifact_root / artifact.name.replace(":", "-")).as_posix())
        )
        ckpt_files = sorted(download_dir.rglob("*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(
                f"No .ckpt file found after downloading artifact {ckpt_artifact} into {download_dir}"
            )
        if len(ckpt_files) > 1:
            raise RuntimeError(
                f"Expected exactly one .ckpt file in artifact {ckpt_artifact}, found {len(ckpt_files)}"
            )
        resolved_path = ckpt_files[0].as_posix()
        log.info(f"Resolved artifact checkpoint to local path <{resolved_path}>")
        return resolved_path
    return ckpt_path


def terminate_torchrun_worker_group() -> None:
    if "LOCAL_RANK" not in os.environ:
        return

    try:
        parent_cmdline = (
            Path(f"/proc/{os.getppid()}/cmdline")
            .read_bytes()
            .replace(b"\x00", b" ")
            .decode("utf-8", errors="ignore")
        )
    except OSError:
        parent_cmdline = ""

    if "torchrun" not in parent_cmdline and "torch.distributed.run" not in parent_cmdline:
        return

    try:
        os.killpg(os.getpgrp(), signal.SIGTERM)
    except ProcessLookupError:
        pass


def report_unhandled_exception(cfg: DictConfig) -> None:
    rank = os.environ.get("RANK", "0")
    local_rank = os.environ.get("LOCAL_RANK", "0")
    header = f"[rank {rank} local_rank {local_rank}] Unhandled exception"
    traceback_text = "".join(traceback.format_exc())

    print(header, file=sys.stderr, flush=True)
    print(traceback_text, file=sys.stderr, flush=True)

    try:
        failure_path = Path(cfg.paths.output_dir) / f"failure_rank{rank}.log"
        failure_path.write_text(header + "\n" + traceback_text)
    except OSError:
        pass


def install_wandb_signal_handlers() -> None:
    def _finish_wandb(signum, _frame) -> None:
        try:
            wandb.finish(exit_code=128 + signum)
        except Exception:
            pass
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, _finish_wandb)
    signal.signal(signal.SIGTERM, _finish_wandb)


def run(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    # setup model watching
    is_global_zero = int(os.environ.get("RANK", "0")) == 0
    # for _logger in logger:
    #     if isinstance(_logger, WandbLogger) and is_global_zero:
    #         _logger.watch(model, log="all")

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

    resolved_ckpt_path = resolve_ckpt_path(cfg)
    log.info(
        f"Resolved checkpoint source: ckpt_path={cfg.get('ckpt_path')} "
        f"ckpt_artifact={cfg.get('ckpt_artifact')} resolved={resolved_ckpt_path}"
    )
    if cfg.action == "fit":
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=resolved_ckpt_path)
    elif cfg.action == "finetune":
        log.info("Starting finetuning!")
        if resolved_ckpt_path is None:
            raise ValueError("finetune action requires ckpt_path or ckpt_artifact")
        model.load_state_dict(torch.load(resolved_ckpt_path, map_location="cpu")["state_dict"], strict=False)
        trainer.fit(model=model, datamodule=datamodule)
    elif cfg.action == "validate":
        log.info("Starting validating!")
        trainer.validate(
            model=model, datamodule=datamodule, ckpt_path=resolved_ckpt_path
        )
    elif cfg.action == "test":
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=resolved_ckpt_path)

    if cfg.action in {"fit", "finetune"}:
        log_wandb_checkpoint_refs(cfg, logger)


@hydra.main(config_path="../configs/", config_name="run.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.set_printoptions(precision=3)
    install_wandb_signal_handlers()

    log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
    print_config_tree(cfg, resolve=True, save_to_file=True)

    try:
        run(cfg)  # train/val/test the model
    except Exception:
        report_unhandled_exception(cfg)
        terminate_torchrun_worker_group()
        raise

    log.info("Closing wandb!")
    wandb.finish()
    log.info(f"Output dir: {cfg.paths.output_dir}")


if __name__ == "__main__":
    main()
    log.info("run.py DONE!!!")
