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
from typing import Any, List, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
from src.utils.waymo_submission import (
    cleanup_prepared_waymo_storage_state,
    maybe_prepare_waymo_storage_state,
    maybe_submit_waymo_submission,
)

log = RankedLogger(__name__, rank_zero_only=True)

torch.set_float32_matmul_precision("high")


def _format_key_list(keys: Sequence[str], max_items: int = 20) -> str:
    shown = list(keys[:max_items])
    suffix = "" if len(keys) <= max_items else f", ... (+{len(keys) - max_items} more)"
    return ", ".join(shown) + suffix


def _load_lightning_checkpoint(ckpt_path: str) -> dict[str, Any]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            "ckpt_path must point to a Lightning checkpoint dictionary containing "
            f"a 'state_dict' entry, got {ckpt_path!r}."
        )
    if not isinstance(checkpoint["state_dict"], dict):
        raise ValueError(
            "ckpt_path must point to a Lightning checkpoint whose 'state_dict' is a mapping, "
            f"got {type(checkpoint['state_dict']).__name__} from {ckpt_path!r}."
        )
    return checkpoint


def _validate_finetune_loaded_trainable_params(
    model: LightningModule,
    missing_keys: Sequence[str],
    unexpected_keys: Sequence[str],
) -> None:
    trainable_param_names = {
        name for name, param in model.named_parameters() if param.requires_grad
    }
    missing_trainable = sorted(set(missing_keys) & trainable_param_names)
    if missing_trainable:
        raise RuntimeError(
            "action=finetune loaded the checkpoint with strict=False, but the checkpoint "
            "is missing parameter(s) that are trainable in this fine-tuning run. "
            "Starting would leave those trainable weights randomly initialized. "
            f"Missing trainable key(s): {_format_key_list(missing_trainable)}"
        )

    if missing_keys:
        log.warning(
            "Ignoring non-trainable missing checkpoint key(s) during finetune load: "
            f"{_format_key_list(list(missing_keys))}"
        )
    if unexpected_keys:
        log.warning(
            "Ignoring unexpected checkpoint key(s) during finetune load: "
            f"{_format_key_list(list(unexpected_keys))}"
        )


def _apply_submission_overrides(cfg: DictConfig) -> None:
    submission_override_cfg = cfg.get("submission")
    if not submission_override_cfg:
        return

    description = submission_override_cfg.get("description")
    if description in (None, ""):
        return

    model_cfg = cfg.get("model")
    model_config = model_cfg.get("model_config") if model_cfg else None
    sim_agents_submission = model_config.get("sim_agents_submission") if model_config else None
    if not sim_agents_submission:
        raise ValueError(
            "submission.description was provided, but model.model_config.sim_agents_submission "
            "is not configured."
        )

    with open_dict(sim_agents_submission):
        sim_agents_submission.description = str(description)


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


def _configure_checkpoint_monitor(cfg: DictConfig, model: LightningModule) -> None:
    callbacks_cfg = cfg.get("callbacks")
    if not callbacks_cfg:
        return

    checkpoint_cfg = callbacks_cfg.get("model_checkpoint")
    if not checkpoint_cfg:
        return

    closed_loop_metric = getattr(model, "closed_loop_metric_name", None)
    open_metric_names = getattr(model, "open_metric_names", {})
    open_ade_metric = open_metric_names.get("ade")

    if getattr(model, "val_closed_loop", False) and closed_loop_metric:
        desired_monitor = closed_loop_metric
        desired_mode = "max"
    elif getattr(model, "val_open_loop", False) and open_ade_metric:
        desired_monitor = f"val_open/{open_ade_metric}"
        desired_mode = "min"
    else:
        desired_monitor = "train/loss"
        desired_mode = "min"

    configured_monitor = checkpoint_cfg.get("monitor")
    should_override_monitor = (
        configured_monitor is None
        or configured_monitor == closed_loop_metric
        or str(configured_monitor).startswith("val_open/ADE")
        or configured_monitor == "train/loss"
    )

    with open_dict(checkpoint_cfg):
        if should_override_monitor:
            checkpoint_cfg.monitor = desired_monitor
        checkpoint_cfg.mode = desired_mode

    log.info(
        "Configured checkpoint monitor: "
        f"monitor={checkpoint_cfg.monitor}, mode={checkpoint_cfg.mode}"
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
    if hasattr(datamodule, "num_workers"):
        os.environ["CATK_DATA_WORKERS"] = str(int(getattr(datamodule, "num_workers")))

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)

    _configure_checkpoint_monitor(cfg, model)

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
        if not cfg.get("ckpt_path"):
            raise ValueError("action=finetune requires ckpt_path for weight-only initialization.")
        checkpoint = _load_lightning_checkpoint(str(cfg.ckpt_path))
        log.info("Starting finetuning!")
        load_result = model.load_state_dict(checkpoint["state_dict"], strict=False)
        _validate_finetune_loaded_trainable_params(
            model=model,
            missing_keys=load_result.missing_keys,
            unexpected_keys=load_result.unexpected_keys,
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
        _apply_submission_overrides(cfg)
        prepared_waymo_storage_state = maybe_prepare_waymo_storage_state(cfg)

        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)

        run(cfg)  # train/val/test the model
        maybe_submit_waymo_submission(cfg)
    finally:
        cleanup_prepared_waymo_storage_state(prepared_waymo_storage_state)
        log.info("Closing wandb!")
        wandb.finish()
    log.info(f"Output dir: {cfg.paths.output_dir}")


if __name__ == "__main__":
    main()
    log.info("run.py DONE!!!")
