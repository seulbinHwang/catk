from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from lightning import Callback, Trainer
from torch import Tensor
from torch_geometric.loader import DataLoader


@dataclass
class _ProbeBatch:
    data: Any
    n_scenarios: int


class EPGRMMProbeCallback(Callback):
    """주기적으로 고정된 Val 시나리오 몇 개로 RMM을 다시 계산해 로깅합니다.

    목적:
    - train step마다 시나리오가 달라서 train RMM 추세가 노이즈가 큰 문제를 완화
    - 동일한(고정된) 시나리오 셋에서 RMM이 올라가는지 확인

    제약/가정:
    - DDP에서도 global_rank==0에서만 실행합니다(로그도 rank0만).
    - val_dataset[0:n_probe_scenarios]를 고정 프로브로 사용합니다.
    """

    def __init__(
        self,
        every_n_train_steps: int = 5,
        n_probe_scenarios: int = 4,
        rollout_idx: int = 0,
        prefix: str = "probe",
    ) -> None:
        super().__init__()
        self.every_n_train_steps = int(every_n_train_steps)
        self.n_probe_scenarios = int(n_probe_scenarios)
        self.rollout_idx = int(rollout_idx)
        self.prefix = str(prefix)
        self._probe: Optional[_ProbeBatch] = None

    def _build_probe_batch(self, trainer: Trainer) -> Optional[_ProbeBatch]:
        dm = trainer.datamodule
        if dm is None or not hasattr(dm, "val_dataset"):
            return None
        ds = getattr(dm, "val_dataset", None)
        if ds is None:
            return None

        n = min(self.n_probe_scenarios, len(ds))
        if n <= 0:
            return None

        # num_workers=0: 항상 동일 샘플을 즉시 가져오고, 프로브에 worker 오버헤드 최소화
        loader = DataLoader(ds, batch_size=n, shuffle=False, num_workers=0, drop_last=False)
        data = next(iter(loader))

        # 시나리오 수는 tokenized_agent["batch"] 기반으로 재계산할 수도 있으나,
        # 여기서는 loader 배치 크기를 기본값으로 씁니다.
        return _ProbeBatch(data=data, n_scenarios=n)

    def on_fit_start(self, trainer: Trainer, pl_module) -> None:
        if getattr(trainer, "global_rank", 0) != 0:
            return
        self._probe = self._build_probe_batch(trainer)

    def on_train_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        if getattr(trainer, "global_rank", 0) != 0:
            return
        if self.every_n_train_steps <= 0:
            return
        step = int(getattr(trainer, "global_step", 0))
        if step == 0 or (step % self.every_n_train_steps) != 0:
            return

        if self._probe is None:
            self._probe = self._build_probe_batch(trainer)
            if self._probe is None:
                return

        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.no_grad():
                data = self._probe.data
                # token_processor 내부 버퍼(토큰 샘플 등)가 pl_module.device에 있을 수 있어
                # probe batch도 동일 디바이스로 이동시켜 device mismatch를 방지합니다.
                if hasattr(data, "to"):
                    data = data.to(pl_module.device)
                tokenized_map, tokenized_agent = pl_module.token_processor(data)
                map_feature = pl_module.encoder.encode_map(tokenized_map)
                rollout_cache = pl_module.encoder.agent_encoder.prepare_inference_cache(
                    tokenized_agent=tokenized_agent,
                    map_feature=map_feature,
                )
                pred_traj, pred_z, pred_head = pl_module._run_parallel_rollout_chunk(
                    data=data,
                    tokenized_agent=tokenized_agent,
                    map_feature=map_feature,
                    rollout_cache=rollout_cache,
                    rollout_indices=[self.rollout_idx],
                    return_anchor_hidden=False,
                )

                agent_batch: Tensor = tokenized_agent["batch"]
                agent_ids: Tensor = data["agent"]["id"]
                rmm_scores = pl_module._compute_rmm_group(
                    data=data,
                    agent_ids=agent_ids,
                    agent_batch=agent_batch,
                    pred_traj=pred_traj,
                    pred_z=pred_z,
                    pred_head=pred_head,
                )  # [n_scenarios, 1]
                rmm_scores = rmm_scores.to(device=pl_module.device, dtype=torch.float32)

                rmm_mean = rmm_scores.mean()
                rmm_max = rmm_scores.max(dim=-1).values.mean()
                rmm_min = rmm_scores.min(dim=-1).values.mean()

                pl_module.log(f"{self.prefix}/rmm_mean", rmm_mean, on_step=True, on_epoch=False, sync_dist=False)
                pl_module.log(f"{self.prefix}/rmm_max", rmm_max, on_step=True, on_epoch=False, sync_dist=False)
                pl_module.log(f"{self.prefix}/rmm_min", rmm_min, on_step=True, on_epoch=False, sync_dist=False)
                pl_module.log(
                    f"{self.prefix}/n_scenarios",
                    torch.tensor(float(rmm_scores.shape[0]), device=pl_module.device),
                    on_step=True,
                    on_epoch=False,
                    sync_dist=False,
                )
        finally:
            if was_training:
                pl_module.train()

