"""OCSC `_ocsc_world_traj_to_anchor0_pose_norm` 의 frame=local/global 단위 테스트.

global 분기는 self 를 쓰지 않으므로 unbound method 를 self=None 으로 호출해 검증한다.
"""

import math

import torch

from src.smart.model.smart_flow import SMARTFlow

_convert = SMARTFlow._ocsc_world_traj_to_anchor0_pose_norm


def _inputs():
    # pred_pos_global: [N=2, T=3, 2] (raw world meter), pred_head_global: [N, T]
    pred_pos_global = torch.tensor(
        [
            [[10.0, 5.0], [12.0, 5.0], [14.0, 6.0]],
            [[-3.0, 2.0], [-3.0, 4.0], [-2.0, 7.0]],
        ]
    )
    pred_head_global = torch.tensor(
        [[0.0, 0.3, 0.6], [1.2, 1.0, 0.8]]
    )
    current_pos = torch.tensor([[10.0, 5.0], [-3.0, 2.0]])
    current_head = torch.tensor([0.0, 1.2])
    return pred_pos_global, pred_head_global, current_pos, current_head


def test_global_frame_is_raw_world_meter_pose() -> None:
    pos, head, cur_pos, cur_head = _inputs()
    out = _convert(
        None,
        pred_pos_global=pos,
        pred_head_global=head,
        current_pos=cur_pos,
        current_head=cur_head,
        frame="global",
    )
    assert out.shape == (2, 3, 4)
    # position = raw meter (정규화 없음)
    torch.testing.assert_close(out[..., 0], pos[..., 0])
    torch.testing.assert_close(out[..., 1], pos[..., 1])
    # heading = 절대 cos/sin
    torch.testing.assert_close(out[..., 2], head.cos())
    torch.testing.assert_close(out[..., 3], head.sin())


def test_global_frame_invariant_to_current_origin() -> None:
    pos, head, cur_pos, cur_head = _inputs()
    out_a = _convert(
        None, pred_pos_global=pos, pred_head_global=head,
        current_pos=cur_pos, current_head=cur_head, frame="global",
    )
    out_b = _convert(
        None, pred_pos_global=pos, pred_head_global=head,
        current_pos=cur_pos + 100.0, current_head=cur_head + 1.0, frame="global",
    )
    # global frame 은 current_pos/head 무시 → 동일
    torch.testing.assert_close(out_a, out_b)


def test_unsupported_frame_raises() -> None:
    pos, head, cur_pos, cur_head = _inputs()
    try:
        _convert(
            None, pred_pos_global=pos, pred_head_global=head,
            current_pos=cur_pos, current_head=cur_head, frame="bogus",
        )
    except ValueError as e:
        assert "match_frame" in str(e)
    else:
        raise AssertionError("expected ValueError for unsupported frame")
