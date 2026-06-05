"""RMM floor 조기 종료 콜백.

Validation 마다 RMM(또는 지정 monitor)을 stdout 으로 찍어 ``artifacts/*.log`` 에서 tail/grep
으로 추적할 수 있게 하고, 값이 ``floor`` 아래로 떨어지면 ``trainer.should_stop`` 을 세워 학습을
즉시(현재 validation 직후) 종료한다.

Lightning ``EarlyStopping(divergence_threshold=...)`` 과 같은 역할이지만, 매 validation 마다
RMM 값을 stdout 으로 명시 출력한다는 점이 다르다.  (no-ckpt overfit sweep 에서 degrade 하는
run 을 일찍 끊고 다음 큐로 넘어가기 위한 용도.)
"""

from __future__ import annotations

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


class RmmFloorStop(Callback):
    """RMM 이 ``floor`` 아래로 떨어지면 학습을 조기 종료하는 콜백.

    Args:
        monitor: 추적할 logged metric 키 (예: ``val_closed/sim_agents_2025/realism_meta_metric``).
        floor: 이 값 미만이면 종료한다 (mode=max 가정, 클수록 좋음).
    """

    def __init__(self, monitor: str, floor: float = 0.775) -> None:
        super().__init__()
        self.monitor = str(monitor)
        self.floor = float(floor)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        value = float(metric)
        print(
            f"[RMM-floor] epoch={trainer.current_epoch} step={trainer.global_step} "
            f"{self.monitor}={value:.5f} floor={self.floor}",
            flush=True,
        )
        if value < self.floor:
            print(
                f"[RMM-floor] RMM {value:.5f} < floor {self.floor} "
                f"→ trainer.should_stop=True (조기 종료)",
                flush=True,
            )
            trainer.should_stop = True
