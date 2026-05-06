from __future__ import annotations

import torch
from torch import Tensor


def get_yaw_rotation(heading: Tensor) -> Tensor:
    """Yaw rotation matrix about z-axis. Matches Waymo `transform_utils.get_yaw_rotation`."""
    c = torch.cos(heading)
    s = torch.sin(heading)
    zeros = torch.zeros_like(c)
    ones = torch.ones_like(c)
    # [N,3,3]
    return torch.stack(
        [
            torch.stack([c, -s, zeros], dim=-1),
            torch.stack([s, c, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )


def get_upright_3d_box_corners(boxes: Tensor) -> Tensor:
    """Torch port of Waymo `box_utils.get_upright_3d_box_corners`.

    Args:
      boxes: [N,7] = [cx,cy,cz,length,width,height,heading]
    Returns:
      corners: [N,8,3] ordered (bottom then top, CCW within each).
    """
    if boxes.dim() != 2 or boxes.shape[1] != 7:
        raise ValueError(f"boxes must be [N,7], got {boxes.shape}")
    cx, cy, cz, length, width, height, heading = boxes.unbind(dim=-1)
    rot = get_yaw_rotation(heading)  # [N,3,3]
    trans = torch.stack([cx, cy, cz], dim=-1)  # [N,3]

    l2 = length * 0.5
    w2 = width * 0.5
    h2 = height * 0.5

    corners_local = torch.stack(
        [
            l2,
            w2,
            -h2,
            -l2,
            w2,
            -h2,
            -l2,
            -w2,
            -h2,
            l2,
            -w2,
            -h2,
            l2,
            w2,
            h2,
            -l2,
            w2,
            h2,
            -l2,
            -w2,
            h2,
            l2,
            -w2,
            h2,
        ],
        dim=-1,
    ).reshape(-1, 8, 3)  # [N,8,3]

    # Apply rotation and translation: einsum('nij,nkj->nki', rot, corners)
    corners = torch.einsum("nij,nkj->nki", rot, corners_local) + trans[:, None, :]
    return corners


__all__ = ["get_upright_3d_box_corners"]

