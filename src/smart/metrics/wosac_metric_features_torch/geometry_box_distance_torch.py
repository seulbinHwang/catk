"""2D 박스·볼록 다각형 기하 연산 (PyTorch).

Waymo Open Sim Agents Challenge(WOSAC) 메트릭 쪽 기하 유틸의 Torch 이식본이다.
차량/장애물을 회전 직사각형(4꼭짓점, CCW)으로 두고, Minkowski 합으로 "팽창 박스"를 만든 뒤
점-다각형 부호 있는 거리(signed distance)를 쓰는 경로(예: soft RMM)에서 사용된다.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from . import geometry_utils_torch as geom

NUM_VERTICES_IN_BOX = 4
# 퇴화한 박스/폴리곤(모서리 길이 0, 중복 꼭짓점)에서 edge_vec/edge_len 정규화 시 backward가 NaN이 됨.
# soft RMM 경로의 Minkowski 합 → signed distance가 실데이터에서 간헐적으로 이 케이스를 밟음.
_EDGE_LEN_EPS = 1e-8


def _safe_norm_2d(x: Tensor, eps: float = _EDGE_LEN_EPS) -> Tensor:
    """Gradient-safe L2 norm along last dim for (..., 2) tensors.

    torch.linalg.norm backward computes x/||x||, which is NaN when ||x||=0.
    Using sqrt(sum(x²).clamp(min=ε²)) gives gradient 0 at x=0 instead.
    """
    return torch.sqrt((x * x).sum(dim=-1).clamp(min=eps * eps))


def rotate_2d_points(xys: Tensor, rotation_yaws: Tensor) -> Tensor:
    """2D 점들을 yaw만큼 평면 회전 (Waymo `geometry_utils.rotate_2d_points` 대응).

    각 점에 동일 배치 축의 회전각을 적용한다. CCW가 양의 각도인 표준 2D 회전 행렬과 동일하다.

    Args:
        xys: (..., 2) 평면 좌표 (x, y).
        rotation_yaws: xys와 브로드캐스트 가능한 shape의 라디안 yaw.

    Returns:
        회전된 좌표, 마지막 차원 2.
    """
    c = torch.cos(rotation_yaws)
    s = torch.sin(rotation_yaws)
    xs_out = c * xys[..., 0] - s * xys[..., 1]
    ys_out = s * xys[..., 0] + c * xys[..., 1]
    return torch.stack([xs_out, ys_out], dim=-1)


def _get_downmost_edge_in_box(box: Tensor) -> Tuple[Tensor, Tensor]:
    """박스에서 y가 가장 작은 꼭짓점을 고르고, 그 꼭짓점에서 출발하는 모서리의 단위 방향을 반환.

    Minkowski 합 시 두 박스의 꼭짓점 순서를 맞추기 위한 기준점/기준변이다.
    CCW 순서이므로 "가장 아래 꼭짓점"에서 다음 꼭짓점으로 가는 변이 한 변이 된다.

    Args:
        box: (B, 4, 2) CCW 꼭짓점.

    Returns:
        downmost_vertex_idx: (B, 1) int64, y 최소 꼭짓점 인덱스.
        edge_dir: (B, 1, 2) 해당 꼭짓점에서 CCW로 이어지는 변의 단위 방향 벡터.
    """
    downmost_vertex_idx = torch.argmin(box[..., 1], dim=-1, keepdim=True)  # (B,1)
    edge_start = torch.gather(box, 1, downmost_vertex_idx[..., None].expand(-1, -1, 2))
    edge_end_idx = torch.remainder(downmost_vertex_idx + 1, NUM_VERTICES_IN_BOX)
    edge_end = torch.gather(box, 1, edge_end_idx[..., None].expand(-1, -1, 2))
    edge = edge_end - edge_start  # (B, 1, 2)
    # torch.linalg.norm backward: x/||x|| → NaN at ||x||=0.
    # Safe: sqrt((x²).sum().clamp(ε²)) → gradient 0 at x=0, not NaN.
    edge_sq = (edge * edge).sum(dim=-1, keepdim=True)  # (B, 1, 1)
    edge_len = torch.sqrt(edge_sq.clamp(min=_EDGE_LEN_EPS ** 2))  # (B, 1, 1)
    edge_dir = edge / edge_len
    return downmost_vertex_idx, edge_dir


def minkowski_sum_of_box_and_box_points(box1_points: Tensor, box2_points: Tensor) -> Tensor:
    """두 축정렬 박스(4꼭짓점)의 Minkowski 합의 경계 — 8각형 꼭짓점 (CCW).

    이산적으로는 (box1의 각 꼭짓점 + box2의 각 꼭짓점) 조합 중 볼록 껍질이 되지만,
    여기서는 Waymo 구현과 같이 "아래쪽 기준 변" 정렬 후 8개 꼭짓점을 순서대로 더해 폐곡선을 만든다.
    두 박스가 회전 직사각형일 때 결과는 팽창/오차 영역 표현에 쓰인다.

    Args:
        box1_points: (B, 4, 2) CCW.
        box2_points: (B, 4, 2) CCW.

    Returns:
        (B, 8, 2) Minkowski 합 다각형의 CCW 꼭짓점 (대응 꼭짓점 합).
    """
    # 8단계에서 box1/box2 꼭짓점을 어떤 순서로 집을지 미리 정의한 패턴 (Waymo 원본과 동일)
    point_order_1 = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long, device=box1_points.device)
    point_order_2 = torch.tensor([0, 1, 1, 2, 2, 3, 3, 0], dtype=torch.long, device=box1_points.device)

    box1_start_idx, box1_dir = _get_downmost_edge_in_box(box1_points)
    box2_start_idx, box2_dir = _get_downmost_edge_in_box(box2_points)

    # 두 "아래 변" 방향의 외적 부호에 따라 8각형을 도는 순서(어느 박스 패턴을 쓸지)가 바뀜
    condition = (geom.cross_product_2d(box1_dir, box2_dir) >= 0.0)  # (B,1)
    condition = condition.expand(-1, 8)

    box1_order = torch.where(condition, point_order_2, point_order_1)
    box1_order = torch.remainder(box1_order + box1_start_idx, NUM_VERTICES_IN_BOX)
    ordered_box1 = torch.gather(box1_points, 1, box1_order[..., None].expand(-1, -1, 2))

    box2_order = torch.where(condition, point_order_1, point_order_2)
    box2_order = torch.remainder(box2_order + box2_start_idx, NUM_VERTICES_IN_BOX)
    ordered_box2 = torch.gather(box2_points, 1, box2_order[..., None].expand(-1, -1, 2))

    return ordered_box1 + ordered_box2


def _get_edge_info(polygon_points: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """볼록 다각형 각 변의 접선, 외향 법선, 변 길이(안전 클램프).

    꼭짓점이 CCW이면 법선은 다각형 바깥을 향하게 쌓는다 (-tangent_y, tangent_x).

    Args:
        polygon_points: (B, K, 2) CCW.

    Returns:
        tangent: (B, K, 2) 단위 접선.
        normal: (B, K, 2) 단위 외향 법선.
        edge_len_safe: (B, K) 변 길이, 최소 _EDGE_LEN_EPS.
    """
    first = polygon_points[:, 0:1, :]
    shifted = torch.cat([polygon_points[:, 1:, :], first], dim=-2)
    edge_vec = shifted - polygon_points  # (B, K, 2)
    # Safe norm: avoids NaN gradient when edge_vec=0 (degenerate polygon)
    edge_len_safe = torch.sqrt((edge_vec * edge_vec).sum(dim=-1).clamp(min=_EDGE_LEN_EPS ** 2))  # (B, K)
    tangent = edge_vec / edge_len_safe[..., None]
    # CCW 폴리곤: 접선 (tx, ty)에 대해 외향 법선은 (-ty, tx)
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
    return tangent, normal, edge_len_safe


def signed_distance_from_point_to_convex_polygon(query_points: Tensor, polygon_points: Tensor) -> Tensor:
    """쿼리 점에서 볼록 다각형까지의 부호 있는 거리 (Waymo `signed_distance_from_point_to_convex_polygon` 대응).

    모든 변에 대해 외향 법선 기준으로 점이 "안쪽"이면 모든 half-plane 부등식이 한쪽으로 맞고,
    거리는 경계(변 또는 꼭짓점)까지의 최소 기하 거리에 부호만 붙인다 (내부 음수, 외부 양수).

    Args:
        query_points: (B, 2).
        polygon_points: (B, K, 2) CCW.

    Returns:
        (B,) signed distance (dtype/shape은 입력과 맞춤).
    """
    tangent, normal, edge_len = _get_edge_info(polygon_points)
    qp = query_points[:, None, :]
    v_to_q = qp - polygon_points  # (B,K,2) 각 꼭짓점에서 쿼리로 가는 벡터
    # Safe norm: NaN gradient when query point coincides with polygon vertex (v_to_q=0)
    v_dist = torch.sqrt((v_to_q * v_to_q).sum(dim=-1).clamp(min=_EDGE_LEN_EPS ** 2))  # (B, K)

    # -n·(q-v): CCW 외향 법선이면 내부에서 <= 0 (모든 변에 대해 만족하면 내부)
    edge_signed_perp = torch.sum(-normal * v_to_q, dim=-1)  # (B,K)
    is_inside = torch.all(edge_signed_perp <= 0, dim=-1)

    # 쿼리를 각 변 위로 사영했을 때 [0,1] 구간 안이면 그 변 위의 가장 가까운 점은 선분
    proj = torch.sum(tangent * v_to_q, dim=-1)
    proj_prop = proj / edge_len
    is_on_edge = (proj_prop >= 0.0) & (proj_prop <= 1.0)

    edge_perp = torch.abs(edge_signed_perp)
    inf = torch.tensor(float("inf"), device=polygon_points.device, dtype=polygon_points.dtype)
    # 선분 밖이면 그 변은 후보에서 제외 (inf), 선분 위면 수직 거리만 사용
    edge_dist = torch.where(is_on_edge, edge_perp, inf)

    edge_and_vertex = torch.cat([edge_dist, v_dist], dim=-1)
    min_dist = torch.min(edge_and_vertex, dim=-1).values
    return torch.where(is_inside, -min_dist, min_dist)


__all__ = [
    "rotate_2d_points",
    "minkowski_sum_of_box_and_box_points",
    "signed_distance_from_point_to_convex_polygon",
]
