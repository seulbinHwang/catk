"""OCSC fine-tuning 용 LoRA (Low-Rank Adaptation) 헬퍼.

외부 라이브러리 (peft, bitsandbytes 등) 의존 없이 ``nn.Linear`` 만 LoRA wrap 한다.

LoRA 식:
  y = base(x) + (alpha / r) * (dropout(x) @ A^T) @ B^T
  - A: ``[r, in_features]`` — Kaiming uniform 초기화.
  - B: ``[out_features, r]`` — zero 초기화 → 학습 시작 시 base 출력과 동일.

Args:
    LoraLinear: ``nn.Linear`` 주위에 LoRA branch 만 추가하는 thin wrapper.
    inject_lora_into_linear_modules: 지정된 dotted name 의 ``nn.Linear`` 를
        ``LoraLinear`` 로 in-place 교체한다.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class LoraLinear(nn.Module):
    """``nn.Linear`` 위에 low-rank adapter 만 학습 가능한 wrapper.

    base linear 는 ``requires_grad=False`` 로 freeze 되며, ``lora_A``/``lora_B``
    만 trainable 이다.  ``__getattr__`` fallback 으로 ``in_features``,
    ``out_features``, ``weight``, ``bias`` 등 base 의 속성에 투명하게 접근할 수
    있어, 이 wrapper 가 들어와도 호출자가 ``nn.Linear`` 처럼 사용할 수 있다.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int,
        alpha: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive (got r={r}).")
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        in_features = int(base_linear.in_features)
        out_features = int(base_linear.out_features)
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r)
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 는 0 으로 두어 학습 시작 시 base 출력과 일치.
        self.lora_dropout: nn.Module = (
            nn.Dropout(p=float(dropout)) if dropout and dropout > 0.0 else nn.Identity()
        )

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # x: [..., in_features].  lora_A.T: [in, r] -> [..., r] -> lora_B.T: [r, out] -> [..., out].
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + self.scaling * lora_out


def _resolve_parent_and_attr(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    """``"a.b.c"`` 처럼 dotted name 을 받아 ``(parent_module, "c")`` 를 반환."""
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def collect_lora_target_names(
    root: nn.Module,
    *,
    layer_filter: str = "t_attn_layers",
    projection_names: Sequence[str] = ("to_q", "to_v"),
) -> List[str]:
    """``root`` 아래 nn.Linear 중 dotted-name 이 ``layer_filter`` substring 을
    포함하고 leaf attribute 이 ``projection_names`` 에 속하는 것들을 모두 수집.

    Returns:
        List[str]: dotted names (예: ``"agent_encoder.t_attn_layers.0.to_q"``).
    """
    targets: List[str] = []
    for name, module in root.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if layer_filter and layer_filter not in name:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in tuple(projection_names):
            targets.append(name)
    return targets


def inject_lora_into_linear_modules(
    root: nn.Module,
    *,
    target_names: Iterable[str],
    r: int,
    alpha: int,
    dropout: float = 0.0,
) -> int:
    """``target_names`` 의 각 dotted name 에 해당하는 ``nn.Linear`` 를
    ``LoraLinear`` 로 in-place 교체.  base linear 는 freeze 되고 LoRA A/B 만 학습.

    Returns:
        int: 실제로 wrap 된 모듈 개수.
    """
    n_wrapped = 0
    for dotted in target_names:
        parent, leaf = _resolve_parent_and_attr(root, dotted)
        original = getattr(parent, leaf)
        if not isinstance(original, nn.Linear):
            log.warning("[lora] skip non-Linear at %s (type=%s)", dotted, type(original).__name__)
            continue
        wrapped = LoraLinear(original, r=r, alpha=alpha, dropout=dropout)
        # 같은 device / dtype 으로 옮긴다 (base linear 기준).
        wrapped = wrapped.to(device=original.weight.device, dtype=original.weight.dtype)
        setattr(parent, leaf, wrapped)
        n_wrapped += 1
    return n_wrapped


def freeze_all_then_unfreeze_lora(root: nn.Module) -> tuple[int, int]:
    """``root`` 의 모든 파라미터를 freeze 한 뒤 LoRA A/B 만 다시 trainable 로.

    Returns:
        tuple[int, int]: ``(n_trainable_params, n_total_params)``.
    """
    for p in root.parameters():
        p.requires_grad = False
    for module in root.modules():
        if isinstance(module, LoraLinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
    n_trainable = sum(p.numel() for p in root.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in root.parameters())
    return n_trainable, n_total
