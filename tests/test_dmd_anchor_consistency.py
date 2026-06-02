"""DMD GT-grounded per-anchor rollout 정합성 단위 검증.

docs/dmd_anchor_consistency_rules.md 의 VR3/VR4/VR6 (GPU 불필요분)을 검증한다.
VR1/VR5/VR7/VR10 (frame/mask/finite) 은 통합 smoke 에서 별도 확인.
"""
from __future__ import annotations

import torch

from src.smart.model.smart_flow import SMARTFlow


SHIFT = 5


def _synthetic_tokens(n_agent: int = 3, n_coarse: int = 18, n_fine: int = 50):
    """coarse 키와 full-10Hz 키를 인덱스가 그대로 값이 되도록 구성."""
    coarse = torch.arange(n_coarse, dtype=torch.float32)
    fine = torch.arange(n_fine, dtype=torch.float32)
    tok = {
        # coarse: gt_pos[a, c] = [c, c]
        "gt_pos": coarse.view(1, n_coarse, 1).expand(n_agent, n_coarse, 2).clone(),
        "gt_heading": coarse.view(1, n_coarse).expand(n_agent, n_coarse).clone(),
        "valid_mask": torch.ones(n_agent, n_coarse, dtype=torch.bool),
        "gt_idx": coarse.view(1, n_coarse).expand(n_agent, n_coarse).long().clone(),
        # full 10Hz: pos[a, t] = [t, t]
        "rollout_full_pos_10hz": fine.view(1, n_fine, 1).expand(n_agent, n_fine, 2).clone(),
        "rollout_full_head_10hz": fine.view(1, n_fine).expand(n_agent, n_fine).clone(),
        "rollout_full_valid_10hz": torch.ones(n_agent, n_fine, dtype=torch.bool),
        # non-time key (보존 확인용)
        "type": torch.zeros(n_agent, dtype=torch.long),
    }
    return tok


def test_vr6_anchor0_identity():
    """VR6: anchor_idx=0 이면 입력 토큰 그대로 (base 동작 무회귀)."""
    tok = _synthetic_tokens()
    out = SMARTFlow._anchor_rollout_tokens_static(tok, anchor_idx=0, shift=SHIFT)
    assert out is tok


def test_vr3_coarse_slice():
    """VR3: anchor k 의 coarse 키가 [:, k:] 슬라이스 → cache window 가 gt_pos[:, k:k+2]."""
    tok = _synthetic_tokens()
    k = 2
    out = SMARTFlow._anchor_rollout_tokens_static(tok, anchor_idx=k, shift=SHIFT)
    # gt_pos[:, :2] (cache history window) == 원본 gt_pos[:, k:k+2]
    assert torch.equal(out["gt_pos"][:, :2], tok["gt_pos"][:, k : k + 2])
    # current (window 끝, index 1) == anchor k current = gt_pos[:, k+1]
    assert torch.allclose(out["gt_pos"][:, 1], tok["gt_pos"][:, k + 1])
    for key in ("gt_heading", "valid_mask", "gt_idx"):
        assert torch.equal(out[key][:, :2], tok[key][:, k : k + 2])
    # 비-시간 키 보존
    assert torch.equal(out["type"], tok["type"])
    # full-10Hz 키는 rollout 토큰에서 제거 (cache 미사용)
    assert "rollout_full_pos_10hz" not in out


def test_vr4_fine_history_ends_at_anchor_current():
    """VR4: anchor k 의 fine exec-history 가 current_raw_step=shift*(k+2) 에서 끝나고 길이 shift+1."""
    tok = _synthetic_tokens()
    for k in (1, 2, 4):
        out = SMARTFlow._anchor_rollout_tokens_static(tok, anchor_idx=k, shift=SHIFT)
        ph = out["rollout_init_fine_pos_history"]
        assert ph.shape[1] == SHIFT + 1  # 6 steps
        cur = SHIFT * (k + 2)
        # 값 = 인덱스 이므로 마지막 step == cur, 첫 step == cur-shift
        assert torch.allclose(ph[:, -1], torch.full_like(ph[:, -1], float(cur)))
        assert torch.allclose(ph[:, 0], torch.full_like(ph[:, 0], float(cur - SHIFT)))
        # pair = 마지막 2개
        pair = out["rollout_init_fine_pos_pair"]
        assert pair.shape[1] == 2
        assert torch.allclose(pair[:, -1], torch.full_like(pair[:, -1], float(cur)))


def test_build_anchor_fine_exec_history_direct():
    """헬퍼 직접: 임의 current_raw_step 에 대해 길이/마지막 step 검증 + 짧을 때 pad."""
    n_fine = 50
    fine = torch.arange(n_fine, dtype=torch.float32)
    pos10 = fine.view(1, n_fine, 1).expand(2, n_fine, 2).clone()
    head10 = fine.view(1, n_fine).expand(2, n_fine).clone()
    valid10 = torch.ones(2, n_fine, dtype=torch.bool)
    ph, hh, vh = SMARTFlow._build_anchor_fine_exec_history(pos10, head10, valid10, current_raw_step=20, shift=SHIFT)
    assert ph.shape == (2, SHIFT + 1, 2)
    assert torch.allclose(ph[:, -1], torch.full_like(ph[:, -1], 20.0))
    # current_raw_step=2 (shift 보다 작음) → 앞 pad 로 길이 6 유지
    ph2, _, _ = SMARTFlow._build_anchor_fine_exec_history(pos10, head10, valid10, current_raw_step=2, shift=SHIFT)
    assert ph2.shape == (2, SHIFT + 1, 2)
    assert torch.allclose(ph2[:, -1], torch.full_like(ph2[:, -1], 2.0))
