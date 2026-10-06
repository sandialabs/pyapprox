"""Full-matrix forward-sensitivity (tangent linear) solver.

Propagates the sensitivity matrix :math:`W_n = dy_n/dp` forward through
a solved trajectory and contracts it with a functional's state Jacobian
step by step, giving the parameter Jacobian of a vector quantity of
interest that may depend on the state at any number of times. The
directional variant (:math:`W v` for a single direction) lives inside
``TimeAdjointOperatorWithHVP``. It is solver-agnostic: it consumes only
the integrator's ``AdjointEnabledTimeSteppingResidualProtocol`` surface.
"""

from pyapprox.ode.functionals.protocols import (
    TransientFunctionalWithJacobianProtocol,
    TransientFunctionalWithStateJacobianActionProtocol,
)
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.sparse_dispatch import solve_maybe_sparse


def forward_sensitivity_jacobian(
    integrator: TimeIntegrator[Array],
    functional: TransientFunctionalWithJacobianProtocol[Array],
    fwd_sols: Array,
    times: Array,
    param: Array,
) -> Array:
    """Compute a functional's parameter Jacobian by the tangent linear model.

    Each step's residual :math:`R_n(y_n, y_{n-1}, p) = 0` implies

    .. math::

        W_n = -(\\partial R_n/\\partial y_n)^{-1}
        [\\partial R_n/\\partial y_{n-1} \\, W_{n-1}
         + \\partial R_n/\\partial p]

    starting from :math:`W_0 = dy_0/dp`
    (``time_residual.initial_param_jacobian()``), and

    .. math::

        \\frac{dQ}{dp} = \\sum_n \\frac{\\partial Q}{\\partial y_n} W_n
        + \\frac{\\partial Q}{\\partial p}.

    Each :math:`W_n` is applied through
    ``functional.apply_state_jacobian`` before the next step overwrites
    it, so memory does not grow with the number of times the functional
    reads. The cost is one linear solve with ``nparams`` right-hand
    sides per step, independent of ``nqoi``.

    Parameters
    ----------
    integrator : TimeIntegrator
        Integrator that produced ``fwd_sols``; its time-stepping
        residual (bound per step via ``bind``) defines the sweep.
    functional : TransientFunctionalWithJacobianProtocol
        Quantity of interest; must also satisfy
        ``TransientFunctionalWithStateJacobianActionProtocol``. Its
        parameters are its own ``nunique_params()`` followed by the
        residual's.
    fwd_sols : Array
        Forward trajectory. Shape: ``(nstates, ntimes)``.
    times : Array
        Time points (may be nonuniform). Shape: ``(ntimes,)``.
    param : Array
        Parameters, in the functional's ordering. Shape: ``(nparams, 1)``.

    Returns
    -------
    Array
        :math:`dQ/dp`. Shape: ``(nqoi, nparams)``.
    """
    if not isinstance(
        functional, TransientFunctionalWithStateJacobianActionProtocol
    ):
        raise TypeError(
            "the tangent-linear jacobian requires a functional satisfying "
            "TransientFunctionalWithStateJacobianActionProtocol, got "
            f"{type(functional).__name__}"
        )
    # A runtime check cannot see the type parameter, so isinstance
    # narrows to ``...[Any]``; rebinding restores ``Array``.
    action: TransientFunctionalWithStateJacobianActionProtocol[Array] = (
        functional
    )
    bkd = integrator.bkd()
    time_residual = integrator.time_residual()
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
    n_unique = action.nunique_params()
    if action.nparams() != n_unique + w_prev.shape[1]:
        raise ValueError(
            f"functional has {action.nparams()} parameters, expected "
            f"{n_unique} of its own plus the residual's {w_prev.shape[1]}"
        )
    state_part = action.apply_state_jacobian(fwd_sols, param, 0, w_prev)

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
        state_part = state_part + action.apply_state_jacobian(
            fwd_sols, param, nn, w_prev
        )

    # The residual does not see the functional's own parameters.
    if n_unique > 0:
        state_part = bkd.hstack(
            (bkd.zeros((action.nqoi(), n_unique)), state_part)
        )
    return state_part + action.param_jacobian(fwd_sols, param)
