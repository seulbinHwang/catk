"""Goal-guided ODE sampler for GOD-FM teacher inpainting.

Teacher receives c_shift anchor_hidden (encoded from rollout-drifted context)
and a goal endpoint in the local frame. Produces a recovery trajectory
tau_target via pure ODE with endpoint gradient guidance.

No SDE. No adjoint. Only flow_ode.generate-style Euler steps + guidance.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GoalGuidedODESampler(nn.Module):
    """Euler ODE sampler with endpoint guidance.

    At each step, predicts the clean endpoint from the current noisy state
    via predict_clean_from_velocity, then pulls x_t toward trajectories
    whose endpoint matches the given goal.

    The flow_decoder is frozen (teacher); gradients are enabled only for the
    guidance gradient computation w.r.t. x_t and are discarded afterwards.
    """

    def __init__(self, inpaint_steps: int = 10, goal_weight: float = 5.0) -> None:
        super().__init__()
        self.inpaint_steps = int(inpaint_steps)
        self.goal_weight = float(goal_weight)

    @torch.no_grad()
    def sample(
        self,
        flow_decoder: nn.Module,
        flow_ode: object,
        anchor_hidden: Tensor,
        goal: Tensor,
    ) -> Tensor:
        """Generate recovery trajectory with endpoint guided toward goal.

        Args:
            flow_decoder: Frozen teacher HierarchicalFlowDecoder.
            flow_ode: FlowODE instance.
            anchor_hidden: c_shift encoded context. shape [n, hidden_dim].
            goal: GT endpoint in c_shift local frame. shape [n, 4].
                  Channels: [x/20, y/20, cos_dhead, sin_dhead].

        Returns:
            Tensor: Recovery trajectory in c_shift local frame. shape [n, 20, 4].
        """
        n = anchor_hidden.shape[0]
        device = anchor_hidden.device
        dtype = anchor_hidden.dtype

        eps = float(flow_ode.eps)
        dt = (1.0 - eps) / self.inpaint_steps
        x_t = torch.randn(n, 20, 4, device=device, dtype=dtype)

        for i in range(self.inpaint_steps):
            t = eps + i * dt
            tau = x_t.new_full((n,), t)

            # Compute velocity and endpoint gradient in a single forward pass.
            x_in = x_t.detach().requires_grad_(True)
            with torch.enable_grad():
                v = flow_decoder(anchor_hidden.detach(), x_in, tau)
                x_1_hat = flow_ode.predict_clean_from_velocity(x_in, v, tau)
                # guidance: pull predicted endpoint toward goal
                endpoint_err = (x_1_hat[:, -1, :] - goal.detach()).pow(2).sum(-1).mean()
                grad = torch.autograd.grad(endpoint_err, x_in)[0]

            x_t = x_t + dt * v.detach() - self.goal_weight * grad.detach()

        return x_t.detach()
