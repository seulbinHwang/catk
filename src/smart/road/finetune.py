from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

from src.smart.datamodules.target_builder import WaymoTargetBuilderVal
from src.smart.road.cache import build_selected_epoch_cache
from src.smart.road.generator import RoadGenerationConfig, generate_road_epoch_cache
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass(frozen=True)
class RoadRuntimeConfig:
    """RoaD fine-tuning 실행 중 필요한 경로와 생성 설정입니다."""

    source_train_raw_dir: Path
    work_dir: Path
    selected_variant_seed: int
    cleanup_used_cache: bool
    generation: RoadGenerationConfig


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """DictConfig와 일반 객체에서 같은 방식으로 값을 읽습니다.

    Args:
        cfg: 값을 읽을 설정 객체입니다.
        key: 읽을 이름입니다.
        default: 값이 없을 때 사용할 기본값입니다.

    Returns:
        Any: 설정값 또는 기본값입니다.
    """
    if cfg is None:
        return default
    if isinstance(cfg, DictConfig):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_road_runtime_config(cfg: DictConfig) -> RoadRuntimeConfig:
    """Hydra config에서 RoaD runtime 설정만 분리합니다.

    Args:
        cfg: 전체 Hydra 설정입니다.

    Returns:
        RoadRuntimeConfig: RoaD cache 생성과 삭제에 필요한 설정입니다.
    """
    road_cfg = cfg.get("road")
    if not road_cfg or not bool(road_cfg.get("enabled", False)):
        raise ValueError("RoaD fine-tuning requires road.enabled=true.")

    generation = RoadGenerationConfig(
        candidates_per_agent=int(road_cfg.get("candidates_per_agent", 64)),
        rollouts_per_scenario=int(road_cfg.get("rollouts_per_scenario", 3)),
        rollout_steps=int(road_cfg.get("rollout_steps", 80)),
        commit_steps=int(road_cfg.get("commit_steps", 5)),
        selection_horizon_steps=int(road_cfg.get("selection_horizon_steps", 20)),
        temperature=float(road_cfg.get("temperature", 0.8)),
        sample_steps=int(road_cfg.get("sample_steps", 16)),
        sample_method=str(road_cfg.get("sample_method", "euler")),
        generation_batch_size=int(road_cfg.get("generation_batch_size", 1)),
        candidate_micro_batch_size=int(road_cfg.get("candidate_micro_batch_size", 4)),
        seed=int(cfg.get("seed", 817)),
        source_count_hint=int(road_cfg.get("num_source_scenarios", 486_995)),
        road_data_use_ratio=float(road_cfg.get("road_data_use_ratio", 0.1)),
        overwrite_cache=bool(road_cfg.get("overwrite_cache", False)),
    )
    if generation.rollouts_per_scenario != 3:
        raise ValueError(
            "This RoaD implementation is configured for the requested default of "
            f"3 rollouts per scenario, got {generation.rollouts_per_scenario}."
        )
    if generation.candidates_per_agent != 64:
        raise ValueError(
            "This RoaD implementation follows the requested K=64 candidate setting, "
            f"got {generation.candidates_per_agent}."
        )
    if generation.sample_steps != 16:
        raise ValueError(
            "RoaD candidate generation diffusion step must be 16, "
            f"got {generation.sample_steps}."
        )
    if generation.road_data_use_ratio <= 0.0 or generation.road_data_use_ratio > 1.0:
        raise ValueError(
            "road_data_use_ratio must be in (0, 1], "
            f"got {generation.road_data_use_ratio}."
        )
    if generation.rollout_steps != 80 or generation.commit_steps != 5:
        raise ValueError(
            "RoaD WOSAC setting must use 80 future steps and 5-step commits, "
            f"got rollout_steps={generation.rollout_steps}, commit_steps={generation.commit_steps}."
        )

    if generation.selection_horizon_steps != 20:
        raise ValueError(
            "RoaD WOSAC candidate selection must use 20 steps, "
            f"got {generation.selection_horizon_steps}."
        )
    if abs(float(generation.temperature) - 0.8) > 1.0e-6:
        raise ValueError(
            "RoaD sampling temperature must be 0.8, "
            f"got {generation.temperature}."
        )

    return RoadRuntimeConfig(
        source_train_raw_dir=Path(str(road_cfg.get("source_train_raw_dir"))),
        work_dir=Path(str(road_cfg.get("work_dir"))),
        selected_variant_seed=int(road_cfg.get("selected_variant_seed", int(cfg.get("seed", 817)))),
        cleanup_used_cache=bool(road_cfg.get("cleanup_used_cache", True)),
        generation=generation,
    )


def _barrier(trainer: Trainer) -> None:
    """분산 학습 process 사이의 파일 생성 순서를 맞춥니다.

    Args:
        trainer: Lightning trainer입니다.

    Returns:
        None
    """
    strategy = getattr(trainer, "strategy", None)
    if strategy is not None and hasattr(strategy, "barrier"):
        strategy.barrier("road-cache-sync")


def _trainer_rank_world(trainer: Trainer) -> tuple[int, int]:
    """Lightning trainer에서 rank와 world size를 가져옵니다.

    Args:
        trainer: Lightning trainer입니다.

    Returns:
        tuple[int, int]: ``(rank, world_size)`` 입니다.
    """
    rank = int(getattr(trainer, "global_rank", 0) or 0)
    world_size = int(getattr(trainer, "world_size", 1) or 1)
    return rank, max(1, world_size)


def _model_device(model: LightningModule) -> torch.device:
    """model parameter가 올라간 장치를 가져옵니다.

    Args:
        model: RoaD 데이터를 생성할 Lightning model입니다.

    Returns:
        torch.device: model 장치입니다.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RoadEpochCacheManager:
    """Always-refresh RoaD cache를 epoch 단위로 만들고 지웁니다."""

    def __init__(self, runtime_config: RoadRuntimeConfig) -> None:
        """RoaD cache manager를 초기화합니다.

        Args:
            runtime_config: RoaD 경로와 생성 설정입니다.
        """
        self.runtime_config = runtime_config
        self.transform = WaymoTargetBuilderVal()

    def epoch_dir(self, epoch_idx: int) -> Path:
        """특정 epoch의 RoaD cache 루트 경로를 만듭니다.

        Args:
            epoch_idx: epoch 번호입니다. 0부터 시작합니다.

        Returns:
            Path: ``epoch_xxx`` 형식의 경로입니다.
        """
        return self.runtime_config.work_dir / f"epoch_{int(epoch_idx):03d}"

    def selected_dir(self, epoch_idx: int) -> Path:
        """특정 epoch에서 실제 학습 loader가 읽을 selected cache 경로를 만듭니다.

        Args:
            epoch_idx: epoch 번호입니다. 0부터 시작합니다.

        Returns:
            Path: selected cache 폴더 경로입니다.
        """
        return self.epoch_dir(epoch_idx) / "selected"

    def variant_dirs(self, epoch_idx: int) -> list[Path]:
        """특정 epoch의 rollout variant 폴더 목록을 만듭니다.

        Args:
            epoch_idx: epoch 번호입니다. 0부터 시작합니다.

        Returns:
            list[Path]: ``variant_00``부터 ``variant_02``까지의 경로입니다.
        """
        return [
            self.epoch_dir(epoch_idx) / "all" / f"variant_{variant_idx:02d}"
            for variant_idx in range(self.runtime_config.generation.rollouts_per_scenario)
        ]

    def prepare_epoch(
        self,
        model: LightningModule,
        datamodule: LightningDataModule,
        trainer: Trainer,
        epoch_idx: int,
    ) -> Path:
        """현재 model로 한 epoch용 RoaD cache를 만들고 train dataset을 바꿉니다.

        Args:
            model: 현재 Flow Matching model입니다.
            datamodule: 학습 datamodule입니다.
            trainer: Lightning trainer입니다.
            epoch_idx: 준비할 epoch 번호입니다. 0부터 시작합니다.

        Returns:
            Path: datamodule이 읽게 될 selected cache 폴더입니다.
        """
        rank, world_size = _trainer_rank_world(trainer)
        epoch_dir = self.epoch_dir(epoch_idx)
        selected_dir = self.selected_dir(epoch_idx)
        epoch_dir.mkdir(parents=True, exist_ok=True)

        if getattr(trainer, "is_global_zero", rank == 0):
            selected_hint = max(
                1,
                int(torch.ceil(torch.tensor(
                    float(self.runtime_config.generation.source_count_hint)
                    * float(self.runtime_config.generation.road_data_use_ratio)
                )).item()),
            )
            log.info(
                "Preparing RoaD epoch cache: "
                f"epoch={epoch_idx}, source={self.runtime_config.source_train_raw_dir}, "
                f"work_dir={epoch_dir}, "
                f"road_data_use_ratio={self.runtime_config.generation.road_data_use_ratio}, "
                f"selected_N~={selected_hint}, "
                f"generated_per_epoch~={selected_hint * self.runtime_config.generation.rollouts_per_scenario}"
            )

        was_training = bool(model.training)
        model.eval()
        generated = generate_road_epoch_cache(
            model=model,
            source_train_raw_dir=self.runtime_config.source_train_raw_dir,
            epoch_dir=epoch_dir,
            transform=self.transform,
            config=self.runtime_config.generation,
            epoch_idx=epoch_idx,
            device=_model_device(model),
            rank=rank,
            world_size=world_size,
        )
        if was_training:
            model.train()
        _barrier(trainer)

        selected_dir.mkdir(parents=True, exist_ok=True)
        selected = build_selected_epoch_cache(
            variant_dirs=self.variant_dirs(epoch_idx),
            selected_dir=selected_dir,
            epoch_idx=epoch_idx,
            seed=self.runtime_config.selected_variant_seed,
            rank=rank,
            world_size=world_size,
        )
        _barrier(trainer)

        if not hasattr(datamodule, "refresh_train_dataset"):
            raise AttributeError(
                "RoaD fine-tuning requires MultiDataModule.refresh_train_dataset(). "
                "Apply the RoaD datamodule patch first."
            )
        datamodule.refresh_train_dataset(selected_dir.as_posix())
        if getattr(trainer, "is_global_zero", rank == 0):
            log.info(
                "RoaD epoch cache is ready: "
                f"epoch={epoch_idx}, generated_by_rank={generated}, selected_by_rank={selected}, "
                f"train_raw_dir={selected_dir}"
            )
        return selected_dir

    def cleanup_epoch(self, trainer: Trainer, epoch_idx: int) -> None:
        """이미 학습에 사용한 RoaD epoch cache를 삭제합니다.

        Args:
            trainer: Lightning trainer입니다.
            epoch_idx: 삭제할 epoch 번호입니다. 0부터 시작합니다.

        Returns:
            None
        """
        if not self.runtime_config.cleanup_used_cache:
            return
        _barrier(trainer)
        rank, _ = _trainer_rank_world(trainer)
        if getattr(trainer, "is_global_zero", rank == 0):
            shutil.rmtree(self.epoch_dir(epoch_idx), ignore_errors=True)
            log.info(f"Deleted used RoaD cache: epoch={epoch_idx}, dir={self.epoch_dir(epoch_idx)}")
        _barrier(trainer)


class RoadAlwaysRefreshCallback(Callback):
    """매 epoch 종료 후 다음 epoch RoaD cache를 새로 만드는 callback입니다."""

    def __init__(self, manager: RoadEpochCacheManager) -> None:
        """callback을 초기화합니다.

        Args:
            manager: epoch cache 생성/삭제 manager입니다.
        """
        super().__init__()
        self.manager = manager
        self._initial_epoch_prepared = False

    def on_fit_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """학습 시작 직전에 epoch 0용 RoaD cache를 만듭니다.

        Args:
            trainer: Lightning trainer입니다.
            pl_module: 현재 학습할 model입니다.

        Returns:
            None
        """
        if self._initial_epoch_prepared:
            return
        datamodule = trainer.datamodule
        if datamodule is None:
            raise RuntimeError("RoaD callback requires trainer.datamodule.")
        self.manager.prepare_epoch(
            model=pl_module,
            datamodule=datamodule,
            trainer=trainer,
            epoch_idx=0,
        )
        self._initial_epoch_prepared = True

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """한 epoch 학습이 끝나면 다음 epoch cache를 만들고 현재 cache를 지웁니다.

        Args:
            trainer: Lightning trainer입니다.
            pl_module: 현재 학습 중인 model입니다.

        Returns:
            None
        """
        datamodule = trainer.datamodule
        if datamodule is None:
            raise RuntimeError("RoaD callback requires trainer.datamodule.")
        current_epoch = int(trainer.current_epoch)
        next_epoch = current_epoch + 1
        max_epochs = int(trainer.max_epochs or 0)
        if next_epoch < max_epochs and not bool(getattr(trainer, "should_stop", False)):
            self.manager.prepare_epoch(
                model=pl_module,
                datamodule=datamodule,
                trainer=trainer,
                epoch_idx=next_epoch,
            )
        self.manager.cleanup_epoch(trainer=trainer, epoch_idx=current_epoch)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """학습이 완전히 끝나면 마지막 epoch cache를 정리합니다.

        Args:
            trainer: Lightning trainer입니다.
            pl_module: 현재 model입니다.

        Returns:
            None
        """
        if trainer.current_epoch is None:
            return
        self.manager.cleanup_epoch(trainer=trainer, epoch_idx=int(trainer.current_epoch))


def run_road_finetune(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    model: LightningModule,
    trainer: Trainer,
) -> None:
    """RoaD fine-tuning을 독립 action으로 실행합니다.

    Args:
        cfg: 전체 Hydra 설정입니다.
        datamodule: 기존 WOMD cache datamodule입니다.
        model: pretrained checkpoint weight가 로드된 Flow Matching model입니다.
        trainer: Lightning trainer입니다.

    Returns:
        None
    """
    runtime_config = build_road_runtime_config(cfg)
    manager = RoadEpochCacheManager(runtime_config)
    callback = RoadAlwaysRefreshCallback(manager)
    trainer.callbacks.append(callback)

    reload_frequency = int(getattr(trainer, "reload_dataloaders_every_n_epochs", 0) or 0)
    if reload_frequency != 1:
        raise ValueError(
            "RoaD fine-tuning requires trainer.reload_dataloaders_every_n_epochs=1 "
            "so each epoch reads the newly selected cache."
        )

    log.info("Starting independent RoaD fine-tuning!")
    trainer.fit(model=model, datamodule=datamodule)
