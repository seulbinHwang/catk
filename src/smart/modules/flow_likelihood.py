from __future__ import annotations

import contextlib
import math
from typing import Callable

import torch
from torch import Tensor


def _double_backward_sdpa_ctx(enabled: bool):
    """double-backward 가 필요한 경우 SDPA 를 math 백엔드로 강제한다.

    PyTorch 의 ``_scaled_dot_product_efficient_attention`` / ``_flash_attention``
    백엔드는 1차 backward 만 구현돼 있어 ``create_graph=True`` 로 만든 2차 그래프를
    통한 추가 backward 시 NotImplementedError 가 난다. math backend 만 이중 미분
    지원.
    """
    if not enabled:
        return contextlib.nullcontext()
    try:  # PyTorch ≥ 2.3
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    except ImportError:  # PyTorch < 2.3
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        )


def hutchinson_divergence_estimate(
    v_fn: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    tau: Tensor,
    n_samples: int = 1,
    create_graph: bool = False,
) -> Tensor:
    """Hutchinson stochastic trace estimator: E_ε[εᵀ (∂v/∂x) ε] = div(v).

    Args:
        v_fn: velocity field (x_t, tau) -> velocity, same shape as x.
        x: state tensor [N, D1, D2].
        tau: time embedding [N].
        n_samples: number of Rademacher probe vectors.
        create_graph: keep second-order graph through VJP (expensive; for full grad).

    Returns:
        Tensor: per-sample divergence estimate [N].
    """
    N = x.shape[0]
    divs = x.new_zeros(N)

    # Always enable grad for VJP, regardless of the outer context.
    # When create_graph=False: isolate with a fresh leaf so no outer graph pollution.
    with torch.enable_grad():
        x_inner = x if create_graph else x.detach().requires_grad_(True)
        v = v_fn(x_inner, tau)

        for i in range(n_samples):
            eps = torch.bernoulli(torch.full_like(x_inner, 0.5)) * 2.0 - 1.0  # Rademacher ±1
            vjp = torch.autograd.grad(
                (v * eps).sum(),
                x_inner,
                create_graph=create_graph,
                retain_graph=(i < n_samples - 1) or create_graph,
            )[0]
            divs = divs + (eps * vjp).flatten(1).sum(1)

    return divs / n_samples


def backward_ode_log_prob_and_grad(
    x1: Tensor,
    v_fn: Callable[[Tensor, Tensor], Tensor],
    steps: int,
    eps_t: float,
    n_hutch: int = 1,
    use_full_div_grad: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute log p_ref(x1) and ∂ log p_ref / ∂ x1 via backward ODE + Hutchinson.

    Implements the CNF log-likelihood formula:
        log p(x₁) = log p₀(x₀) - ∫₀¹ div(v(xₜ, t)) dt

    Backward Euler integration from t=1 → t=eps_t:
        x_{t-h} = x_t - h · v_ref(x_t, t)

    The gradient ∂ log p / ∂ x₁ is computed by backpropagating through:
    - log p₀(x₀) via the ODE chain (always active).
    - The divergence accumulation (only when use_full_div_grad=True; requires
      create_graph inside Hutchinson → expensive).

    When use_full_div_grad=False (default/cheap), the divergence terms are
    accumulated as detached scalars so the gradient only flows through the
    prior term. This is a first-order approximation sufficient for
    straight-through BPTT.

    Args:
        x1: generated trajectory [N, 20, 4].
        v_fn: frozen reference velocity field (x_t, tau) -> velocity.
        steps: number of backward Euler steps (= flow_ode.solver_steps).
        eps_t: small t offset to avoid t=0 (= flow_ode.eps).
        n_hutch: number of Hutchinson probe vectors per step.
        use_full_div_grad: if True, maintain gradient through divergence accumulation.

    Returns:
        log_prob: [N] log-likelihood under reference model (detached).
        grad_x1:  [N, 20, 4] gradient ∂ log p / ∂ x₁ (straight-through signal).
    """
    dt = (1.0 - eps_t) / float(steps)

    with torch.enable_grad(), _double_backward_sdpa_ctx(use_full_div_grad):
        x1_leaf = x1.detach().requires_grad_(True)
        x_t = x1_leaf
        # Accumulated ∫₀¹ div(v) dt (≈ negative log-Jacobian contribution)
        log_det = x1.new_zeros(x1.shape[0])

        for i in range(steps - 1, -1, -1):
            t_curr = eps_t + (i + 1) * dt   # descends: 1, 1-dt, ..., eps_t+dt
            tau = x_t.new_full((x_t.shape[0],), t_curr)

            if use_full_div_grad:
                # Keep graph through div for complete ∂ log p / ∂ x₁
                div_est = hutchinson_divergence_estimate(
                    v_fn, x_t, tau, n_hutch, create_graph=True
                )
                log_det = log_det + div_est * dt
            else:
                # Cheap: evaluate div on a detached copy → result has no connection
                # to x1_leaf.  .detach() to be explicit; v_fn must not be no_grad-wrapped.
                div_est = hutchinson_divergence_estimate(
                    v_fn, x_t.detach(), tau, n_hutch, create_graph=False
                ).detach()
                log_det = log_det + div_est * dt

            # Backward Euler step — x_t stays in graph w.r.t. x1_leaf
            v = v_fn(x_t, tau)
            x_t = x_t - dt * v   # x_{t-h} = x_t - h · v_ref(x_t, t)

        # x_t ≈ x₀ after N backward steps; prior = N(0, I)
        D = int(x1.shape[1] * x1.shape[2])
        log_p0 = -0.5 * (
            x_t.flatten(1).pow(2).sum(1) + D * math.log(2.0 * math.pi)
        )
        log_p = log_p0 - log_det   # CNF: log p(x₁) = log p₀(x₀) - ∫ div dt

        log_p.sum().backward()

    grad_x1 = (
        x1_leaf.grad.clone()
        if x1_leaf.grad is not None
        else torch.zeros_like(x1)
    )
    return log_p.detach(), grad_x1
