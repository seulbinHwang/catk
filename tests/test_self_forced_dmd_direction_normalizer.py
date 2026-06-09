"""build_clean_dmd_direction 의 full(시간+채널) normalizer + dead-channel masking 검증."""

from __future__ import annotations

import torch

from src.smart.modules.self_forced_dmd_guidance import build_clean_dmd_direction


def test_full_normalizer_uses_single_agentwise_scalar_over_time_and_channel() -> None:
    # committed/target 차이가 채널마다 다르면, full normalizer 는 시간+채널 전체 평균을
    # 쓰므로 모든 채널이 같은 agent 단일 스칼라로 나뉜다(원본 DMD mean(dim=[1..]) 정합).
    committed = torch.zeros(1, 2, 3)
    target = torch.tensor([[[1.0, 3.0, 5.0], [1.0, 3.0, 5.0]]])  # |gap| 채널평균 = (1+3+5)/3 = 3
    generated = torch.zeros(1, 2, 3)

    direction = build_clean_dmd_direction(
        committed_path_norm=committed,
        target_clean_norm=target,
        generated_clean_norm=generated,
        normalizer_eps=1.0e-6,
        channel_mask=None,
        per_channel_normalizer=False,
        normalize_direction=True,
    )

    # dir = (target - generated) / 3.0
    expected = target / 3.0
    assert torch.allclose(direction, expected, atol=1e-5)


def test_dead_channel_excluded_from_direction_and_normalizer() -> None:
    # ch1(dn)을 죽은 채널로 마스킹하면 (1) direction[...,1]==0 이고,
    # (2) normalizer 의 평균에서도 ch1 의 gap 이 빠져야 한다("아예 없는 tensor").
    committed = torch.zeros(1, 1, 3)
    # ch0,ch2 gap=2, ch1 gap=100 (죽은 채널이라 normalizer 를 오염시키면 안 됨)
    target = torch.tensor([[[2.0, 100.0, 2.0]]])
    generated = torch.zeros(1, 1, 3)
    channel_mask = torch.tensor([[[1.0, 0.0, 1.0]]])  # ch1 죽음

    direction = build_clean_dmd_direction(
        committed_path_norm=committed,
        target_clean_norm=target,
        generated_clean_norm=generated,
        normalizer_eps=1.0e-6,
        channel_mask=channel_mask,
        per_channel_normalizer=False,
        normalize_direction=True,
    )

    # 죽은 채널 direction 은 정확히 0
    assert direction[0, 0, 1].abs().item() == 0.0
    # normalizer 는 valid 채널만 평균 = (2 + 2) / 2 = 2.0 → dir = 2/2 = 1.0
    assert torch.allclose(direction[0, 0, 0], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(direction[0, 0, 2], torch.tensor(1.0), atol=1e-5)


def test_dead_channel_does_not_inflate_or_deflate_valid_channels() -> None:
    # 죽은 채널의 큰 gap 이 normalizer 에 끼면 valid 채널 dir 이 작아진다.
    # masking 이 제대로면 valid 채널 dir 은 mask 유무와 무관해야 한다.
    committed = torch.zeros(2, 3, 3)
    target = torch.randn(2, 3, 3)
    target[..., 1] = 50.0  # ch1 을 비정상적으로 크게
    generated = torch.zeros(2, 3, 3)

    mask = torch.ones(2, 1, 3)
    mask[..., 1] = 0.0

    masked = build_clean_dmd_direction(
        committed, target, generated,
        normalizer_eps=1.0e-6, channel_mask=mask,
        per_channel_normalizer=False, normalize_direction=True,
    )
    # ch1 제거 후 valid(ch0,ch2)만으로 직접 계산한 기준
    abs_gap = (committed - target).abs()
    denom = (abs_gap[..., 0] + abs_gap[..., 2]).sum(dim=1, keepdim=True) / (2 * 3)
    expected_ch0 = target[..., 0] / denom.clamp_min(1e-6)

    assert torch.allclose(masked[..., 0], expected_ch0, atol=1e-4)
    assert torch.all(masked[..., 1] == 0.0)
