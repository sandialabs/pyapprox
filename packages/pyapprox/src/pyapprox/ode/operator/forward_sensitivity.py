"""Full-matrix forward-sensitivity (tangent linear) solver.

Propagates the sensitivity matrix :math:`W_n = dy_n/dp` forward through
a solved trajectory. The directional variant (:math:`W v` for a single
direction) lives inside ``TimeAdjointOperatorWithHVP``; this module
provides the full-matrix sweep needed for vector quantities of interest
(:math:`dQ/dp = dQ/dy(T) \\, W_T`) and for adjoint-vs-sensitivity
cross-checks. It is solver-agnostic: it consumes only the
``AdjointEnabledTimeSteppingResidualProtocol`` surface.
"""

from pyapprox.ode.protocols.time_stepping import (
    AdjointEnabledTimeSteppingResidualProtocol,
)
from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.sparse_dispatch import solve_maybe_sparse


def solve_final_forward_sensitivity(
    time_residual: AdjointEnabledTimeSteppingResidualProtocol[Array],
    fwd_sols: Array,
    times: Array,
    bkd: Backend[Array],
) -> Array:
    """Solve the tangent linear model for the final sensitivity matrix.

    Each step's residual :math:`R_n(y_n, y_{n-1}, p) = 0` implies

    .. math::

        W_n = -(\\partial R_n/\\partial y_n)^{-1}
        [\\partial R_n/\\partial y_{n-1} \\, W_{n-1}
         + \\partial R_n/\\partial p]

    starting from :math:`W_0 = dy_0/dp`
    (``time_residual.initial_param_jacobian()``). Only the final-time
    matrix is returned.

    Parameters
    ----------
    time_residual : AdjointEnabledTimeSteppingResidualProtocol
        Time-stepping residual (bound per step via ``bind``).
    fwd_sols : Array
        Forward trajectory. Shape: ``(nstates, ntimes)``.
    times : Array
        Time points (may be nonuniform). Shape: ``(ntimes,)``.
    bkd : Backend
        Computational backend.

    Returns
    -------
    Array
        Final sensitivity matrix :math:`W_T = dy(T)/dp`.
        Shape: ``(nstates, nparams)``.
    """
    ntimes = fwd_sols.shape[1]
    if ntimes < 2:
        raise ValueError(
            f"trajectory must contain at least two time points, got "
            f"{ntimes}"
        )
    ctx_0 = StepContext(
        t_prev=bkd.to_float(times[0]),
        deltat=bkd.to_float(times[1] - times[0]),
        y_prev=fwd_sols[:, 0],
    )
    time_residual.bind(ctx_0)
    w_prev = time_residual.initial_param_jacobian()

    for nn in range(1, ntimes):
        ctx_nn = StepContext(
            t_prev=bkd.to_float(times[nn - 1]),
            deltat=bkd.to_float(times[nn] - times[nn - 1]),
            y_prev=fwd_sols[:, nn - 1],
        )
        time_residual.bind(ctx_nn)

        drdy_n = time_residual.jacobian(fwd_sols[:, nn])
        drdy_nm1 = time_residual.sensitivity_off_diag_jacobian(
            ctx_nn, fwd_sols[:, nn]
        )
        drdp_n = time_residual.param_jacobian(ctx_nn, fwd_sols[:, nn])

        # Sparse-aware: galerkin wrappers return sparse Jacobians, and
        # solve_maybe_sparse handles the multi-column right-hand side.
        rhs = bkd.asarray(drdy_nm1 @ w_prev) + bkd.asarray(drdp_n)
        w_prev = -solve_maybe_sparse(bkd, drdy_n, rhs)

    return w_prev
