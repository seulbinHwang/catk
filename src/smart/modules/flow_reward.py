"""Modular reward functions for ODE reward-gradient fine-tuning.

Each reward is a plain callable (not nn.Module) so no parameters get accidentally
registered as trainable.  Signature::

    reward_fn(y_hat, **kwargs) -> tuple[Tensor, dict[str, Tensor]]

Returns:
    loss  - scalar, to be minimised (reward = -loss conceptually)
    metrics - dict of detached scalars for logging
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


class KinematicProjectionReward:
    """Soft kinematic-feasibility reward via Huber (smooth_l1) distance.

    Instead of hard-clamping the trajectory to the bicycle-model projection,
    we measure the Huber distance between the ODE output ``y_hat`` and the
    *detached* projected trajectory ``y_proj``.  This keeps gradients smooth
    everywhere — no zero-gradient plateau from a hard clamp.

    The KinematicProjection itself is also run with ``torch.no_grad()`` so only
    ``y_hat`` (which carries the ODE gradient) participates in the backward pass.

    Args:
        kinematic_projector: ``KinematicProjection`` module (not trained).
        huber_beta: Transition point between L1 and L2 regions of Huber loss.
            Smaller values keep the loss closer to L1 (more robust to large gaps).
    """

    def __init__(self, kinematic_projector, huber_beta: float = 0.05) -> None:
        self.kinematic_projector = kinematic_projector
        self.huber_beta = float(huber_beta)

    def __call__(
        self,
        y_hat: Tensor,
        agent_type: Optional[Tensor] = None,
        v_init: Optional[Tensor] = None,
        delta_init: Optional[Tensor] = None,
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        """Compute kinematic-projection reward loss.

        Args:
            y_hat: ODE-generated trajectory. shape ``[n, 20, 4]``.
                gradient w.r.t. flow_decoder parameters must flow through this.
            agent_type: shape ``[n]`` or None.
            v_init: initial speed per agent. shape ``[n]`` or None.
            delta_init: initial steering angle per agent. shape ``[n]`` or None.

        Returns:
            tuple:
                - loss (scalar): Huber distance between ``y_hat`` and ``y_proj``.
                - metrics (dict): ``{"projection_gap": ...}`` for logging.
        """
        dev = y_hat.device

        with torch.no_grad():
            y_proj = self.kinematic_projector(
                y_hat.detach().clone(),
                agent_type=agent_type.to(dev) if agent_type is not None else None,
                v_init=v_init.to(dev) if v_init is not None else None,
                delta_init=delta_init.to(dev) if delta_init is not None else None,
            )

        # Huber loss: smooth near 0, linear for large residuals → no zero-grad plateau
        loss = F.smooth_l1_loss(y_hat, y_proj.detach(), beta=self.huber_beta)

        projection_gap = (y_hat.detach() - y_proj).abs().mean()
        return loss, {"projection_gap": projection_gap}
