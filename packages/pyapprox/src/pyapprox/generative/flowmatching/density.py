"""Density utilities for flow matching.

Provides density evaluation via backward ODE integration with divergence
tracking, and KL divergence computation using quadrature.
"""

from typing import Callable, Optional

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend


def _log_standard_normal(x: Array) -> Array:
    return -0.5 * np.log(2 * np.pi) - 0.5 * x**2


def compute_flow_density(
    model: object,
    x1_samples: Array,
    bkd: Backend[Array],
    n_steps: int = 100,
    scheme: str = "heun",
    log_source_density: Optional[Callable[[Array], Array]] = None,
    t_end: float = 1.0 - 1e-8,
) -> Array:
    """Evaluate density q(x1) by solving the backward ODE with divergence tracking.

    Computes log q(x1) = log p0(x0) - int_0^1 div(v(x_t, t)) dt
    where p0 is the source density.

    Parameters
    ----------
    model : object
        Trained vector field with ``__call__`` and ``jacobian_batch`` methods.
        Input shape ``(2, nsamples)`` for ``[t; x]``.
    x1_samples : Array
        Evaluation points at t=1, shape ``(1, nsamples)``.
    bkd : Backend[Array]
        Computational backend.
    n_steps : int
        Number of ODE integration steps.
    scheme : str
        ODE integration scheme: ``"euler"`` or ``"heun"``.
    log_source_density : callable, optional
        ``log p0(x)`` evaluated at traced-back source samples.  Takes and
        returns arrays of shape ``(1, nsamples)``.  Defaults to standard
        normal ``N(0, 1)``.
    t_end : float
        Upper integration limit, shifted slightly below 1.0 to avoid
        singularities in the velocity field at ``t = 1``.

    Returns
    -------
    Array
        Density values q(x1), shape ``(1, nsamples)``.
    """
    if scheme not in ("euler", "heun"):
        raise ValueError(f"Unknown ODE scheme: {scheme!r}. Use 'euler' or 'heun'.")

    nsamples = x1_samples.shape[1]
    x = bkd.copy(x1_samples)
    log_div_integral = bkd.zeros((1, nsamples))
    dt = t_end / n_steps

    for i in range(n_steps):
        t = t_end - i * dt
        t_row = bkd.full((1, nsamples), t)

        vf_in1 = bkd.vstack([t_row, x])
        v1 = model(vf_in1)  # type: ignore[operator]
        jac1 = model.jacobian_batch(vf_in1)  # type: ignore[attr-defined]
        div1 = bkd.reshape(jac1[:, 0, 1], (1, -1))

        if scheme == "euler":
            x = x - dt * v1
            log_div_integral = log_div_integral + dt * div1
        else:  # heun
            t2 = max(t - dt, 0.0)
            t_row2 = bkd.full((1, nsamples), t2)
            x2 = x - dt * v1
            vf_in2 = bkd.vstack([t_row2, x2])
            v2 = model(vf_in2)  # type: ignore[operator]
            jac2 = model.jacobian_batch(vf_in2)  # type: ignore[attr-defined]
            div2 = bkd.reshape(jac2[:, 0, 1], (1, -1))

            x = x - 0.5 * dt * (v1 + v2)
            log_div_integral = log_div_integral + 0.5 * dt * (div1 + div2)

    if log_source_density is None:
        log_source_density = _log_standard_normal
    log_p0 = log_source_density(x)
    log_q = log_p0 - log_div_integral
    return bkd.exp(log_q)


def compute_kl_divergence(
    model: object,
    target_pdf_fn: object,
    quad_pts: Array,
    quad_wts: Array,
    bkd: Backend[Array],
    n_steps: int = 100,
    scheme: str = "heun",
    log_source_density: Optional[Callable[[Array], Array]] = None,
    t_end: float = 1.0 - 1e-8,
) -> float:
    """Compute D_KL(p || q) using quadrature nodes for target p.

    Parameters
    ----------
    model : object
        Trained vector field.
    target_pdf_fn : callable
        Evaluates target density p(x), signature ``(Array) -> Array``.
    quad_pts : Array
        Quadrature points, shape ``(1, npts)``.
    quad_wts : Array
        Quadrature weights, shape ``(npts,)``.
    bkd : Backend[Array]
        Computational backend.
    n_steps : int
        Number of ODE steps for density evaluation.
    scheme : str
        ODE integration scheme.
    log_source_density : callable, optional
        Passed to ``compute_flow_density``.
    t_end : float
        Passed to ``compute_flow_density``.

    Returns
    -------
    float
        KL divergence D_KL(p || q).
    """
    p_vals = target_pdf_fn(quad_pts)  # type: ignore[operator]
    q_vals = compute_flow_density(
        model, quad_pts, bkd, n_steps=n_steps, scheme=scheme,
        log_source_density=log_source_density, t_end=t_end,
    )
    eps = 1e-15
    integrand = bkd.log((p_vals + eps) / (q_vals + eps))
    kl = bkd.sum(
        bkd.flatten(integrand) * quad_wts
    ) / bkd.sum(quad_wts)
    return bkd.to_float(kl)
