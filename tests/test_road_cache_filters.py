"""RoaD cache의 autocast precision 정합 헬퍼를 검증한다.

RoaD는 모델 자기 자신의 rollout을 그대로 새 정답으로 학습시키는 방식이라,
거리 클립이나 도로 이탈 같은 후처리 방어 로직은 일부러 두지 않는다. 이 테스트는
precision→autocast dtype 매핑만 검증한다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.smart.road.cache import (
    resolve_autocast_dtype_from_precision,
    select_road_dataset_indices,
)


def test_resolve_autocast_dtype_mapping():
    cases = {
        "bf16-mixed": torch.bfloat16,
        "BF16-MIXED": torch.bfloat16,
        "16-mixed": torch.float16,
        "bf16": torch.bfloat16,
        "16": torch.float16,
        "32-true": None,
        "32": None,
        None: None,
        "": None,
    }
    for prec, expected in cases.items():
        got = resolve_autocast_dtype_from_precision(prec)
        assert got == expected, f"precision={prec!r}: expected {expected}, got {got}"
    print(f"resolve_autocast_dtype_from_precision: {len(cases)} cases OK")


def test_select_road_dataset_indices_ratio_count():
    assert select_road_dataset_indices(10, 1.0) is None
    assert len(select_road_dataset_indices(10, 0.25)) == 3
    assert len(select_road_dataset_indices(10, 0.01)) == 1


def main():
    test_resolve_autocast_dtype_mapping()
    test_select_road_dataset_indices_ratio_count()
    print("\nAll autocast precision mapping tests PASSED.")


if __name__ == "__main__":
    main()
