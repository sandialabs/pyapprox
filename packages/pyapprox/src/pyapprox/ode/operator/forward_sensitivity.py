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
    TransientFunctionalProtocol,
)
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.state_constraint import sweep_tangent
from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Array


def forward_sensitivity_jacobian(
    integrator: TimeIntegrator[Array],
    functional: TransientFunctionalProtocol[Array],
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

    With the integrator's state constraint :math:`\\Pi_t`, the stored state
    is :math:`y_n = \\Pi_{t_n}(z_n)` for the step's Newton solution
    :math:`z_n`, so the step gives :math:`dz_n/dp` by the formula above,
    with :math:`\\partial R_n/\\partial y_n` taken at :math:`z_n` and
    :math:`\\partial R_n/\\partial y_{n-1}` at the stored :math:`y_{n-1}`,
    and :math:`W_n = \\Pi'(z_n)\\, dz_n/dp`; likewise
    :math:`W_0 = \\Pi'(y_0)\\, dy_0/dp` for the initial state as given.
    Under the identity constraint this is the unconstrained sweep.

    Each :math:`W_n` is applied through the functional's
    ``apply_state_jacobian`` before the next step overwrites
    it, so memory does not grow with the number of times the functional
    reads. The cost is one linear solve with ``nparams`` right-hand
    sides per step, independent of ``nqoi``.

    Parameters
    ----------
    integrator : TimeIntegrator
        Integrator that produced ``fwd_sols``; its time-stepping
        residual (bound per step via ``bind``) defines the sweep.
    functional : TransientFunctionalProtocol
        Quantity of interest; its ``derivatives()`` must provide
        ``apply_state_jacobian``. Its parameters are its own
        ``nunique_params()`` followed by the residual's.
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

    Raises
    ------
    TypeError
        If the constraint provides no tangent or parameter derivative.
    NotImplementedError
        If the constraint's parameter derivative is not a stated zero.
    ValueError
        With a constraint, if ``fwd_sols`` is not the integrator's last
        solve.
    """
    derivs = functional.derivatives()
    apply_state_jacobian = derivs.apply_state_jacobian
    param_jacobian = derivs.param_jacobian
    if apply_state_jacobian is None or param_jacobian is None:
        raise TypeError(
            "the tangent-linear jacobian requires the functional's "
            "derivatives() to provide apply_state_jacobian; "
            f"{type(functional).__name__} does not"
        )
    action = functional
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
    tangent = sweep_tangent(
        integrator.state_constraint(fwd_sols.shape[0], w_prev.shape[1]).derivatives(),
        fwd_sols[:, 0],
        bkd.to_float(times[0]),
    )
    # Each step is linearized at its own unknown z_n; the previous state
    # and the functional read the stored states.
    lin_sols = integrator.linearization_states(fwd_sols)
    w_prev = tangent(lin_sols[:, 0], bkd.to_float(times[0])).apply(w_prev)
    state_part = apply_state_jacobian(fwd_sols, param, 0, w_prev)

    for nn in range(1, ntimes):
        ctx_nn = StepContext(
            t_prev=bkd.to_float(times[nn - 1]),
            deltat=bkd.to_float(times[nn] - times[nn - 1]),
            y_prev=fwd_sols[:, nn - 1],
        )
        time_residual.bind(ctx_nn)

        # A_n^T as an operator; its transposed solve is a solve with A_n.
        diag_t = time_residual.adjoint_diag_jacobian(ctx_nn, lin_sols[:, nn])
        offdiag = time_residual.off_diag_operator(ctx_nn, lin_sols[:, nn])
        drdp_n = time_residual.param_jacobian(ctx_nn, lin_sols[:, nn])

        # B_n and the solve apply to the whole block of sensitivities.
        rhs = offdiag.apply(w_prev) + bkd.asarray(drdp_n)
        w_prev = tangent(lin_sols[:, nn], bkd.to_float(times[nn])).apply(
            -diag_t.solve_transpose(rhs)
        )
        state_part = state_part + apply_state_jacobian(
            fwd_sols, param, nn, w_prev
        )

    # The residual does not see the functional's own parameters.
    if n_unique > 0:
        state_part = bkd.hstack(
            (bkd.zeros((action.nqoi(), n_unique)), state_part)
        )
    return state_part + param_jacobian(fwd_sols, param)
