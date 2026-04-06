from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DICECritic(nn.Module):
    """Q-function for Chi^2 f-DICE / IQ-Learn imitation learning.

    Q(s, a) → scalar estimate of the occupancy measure ratio between expert
    and policy distributions.

    State s = (scene context, kinematic init)
        anchor_hidden : [n, d_state]  — frozen encoder output per agent
        v_init        : [n]           — speed at chunk start (m/s)
        delta_init    : [n]           — steer angle at chunk start (rad)

    Action a = trajectory chunk
        action        : [n, T, 4]    — (x_cum_norm, y_cum_norm, cos θ, sin θ)

    DICE Chi^2 training objective (minimise w.r.t. Q):
        L_Q = E_π[ Q²/4 + Q ] − E_E[ Q ]

    Actor update (minimise w.r.t. flow model, Q frozen):
        L_actor = −E_π[ Q(s, a_π) ]   (+optional external reward)
    """

    def __init__(
        self,
        d_state: int = 128,
        T: int = 20,
        action_hidden: int = 128,
        critic_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.T = T

        # ── Action encoder: [n, T*4] → action_hidden ──────────────────────
        self.action_encoder = nn.Sequential(
            nn.Linear(T * 4, action_hidden),
            nn.LayerNorm(action_hidden),
            nn.SiLU(),
            nn.Linear(action_hidden, action_hidden),
            nn.LayerNorm(action_hidden),
            nn.SiLU(),
        )

        # ── State encoder: [n, d_state + 2] → critic_hidden ───────────────
        self.state_encoder = nn.Sequential(
            nn.Linear(d_state + 2, critic_hidden),
            nn.LayerNorm(critic_hidden),
            nn.SiLU(),
        )

        # ── Q head: [n, critic_hidden + action_hidden] → 1 ────────────────
        self.q_head = nn.Sequential(
            nn.Linear(critic_hidden + action_hidden, critic_hidden),
            nn.SiLU(),
            nn.Linear(critic_hidden, 1),
        )

        self._init_weights()

    # ──────────────────────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        """Small final-layer init for stable early training."""
        nn.init.orthogonal_(self.q_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.q_head[-1].bias)

    # ──────────────────────────────────────────────────────────────────────
    def forward(
        self,
        anchor_hidden: Tensor,
        v_init: Tensor,
        delta_init: Tensor,
        action: Tensor,
    ) -> Tensor:
        """Compute Q(s, a).

        Args:
            anchor_hidden : [n, d_state]  frozen scene context per agent.
            v_init        : [n]           speed at trajectory start (m/s).
            delta_init    : [n]           steer angle at start (rad).
            action        : [n, T, 4]     trajectory chunk.

        Returns:
            Tensor: Q-values [n].
        """
        n = anchor_hidden.shape[0]

        state = torch.cat(
            [anchor_hidden, v_init.unsqueeze(-1), delta_init.unsqueeze(-1)],
            dim=-1,
        )  # [n, d_state + 2]

        s_enc = self.state_encoder(state)                     # [n, critic_hidden]
        a_enc = self.action_encoder(action.reshape(n, -1))    # [n, action_hidden]

        q = self.q_head(torch.cat([s_enc, a_enc], dim=-1))   # [n, 1]
        return q.squeeze(-1)                                   # [n]

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def critic_loss(q_policy: Tensor, q_expert: Tensor) -> Tensor:
        """Chi^2 f-DICE critic loss (minimise).

        L_Q = E_π[ Q²/4 + Q ] − E_E[ Q ]

        Args:
            q_policy : Q values on policy samples [n_π].
            q_expert : Q values on expert samples [n_E].

        Returns:
            Scalar loss.
        """
        return (q_policy.pow(2) / 4.0 + q_policy).mean() - q_expert.mean()

    @staticmethod
    def implied_reward(q: Tensor) -> Tensor:
        """Reward induced by Chi^2 DICE: r(s,a) = 1 − Q²/4 − Q.

        Useful for logging; not used in training.
        """
        return 1.0 - q.pow(2) / 4.0 - q
