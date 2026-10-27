"""
Time-dependent adjoint operator with Hessian-vector product support.

This module provides HVP computation for time-dependent problems using
the second-order adjoint method (forward sensitivity + adjoint sensitivity).

Algorithm:
1. Forward ODE solve: y(t) trajectory
2. Backward adjoint solve: λ(t) trajectory
3. Forward sensitivity solve: w(t) = (dy/dp)·v via tangent linear model
4. Backward second-adjoint solve: s(t) for Hessian contribution
5. Accumulate HVP from Lagrangian terms at each time step
"""

from typing import Generic, Optional, Tuple

from pyapprox.ode.functionals.protocols import (
    TransientFunctionalProtocol,
    TransientJacobianFn,
    TransientParamHVPFn,
    TransientStepHVPFn,
)
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.operator.storage import TimeTrajectoryStorage
from pyapprox.ode.protocols.ode_residual import (
    ODEResidualWithParamJacobianProtocol,
)
from pyapprox.ode.protocols.time_stepping import (
    HVPEnabledTimeSteppingResidualProtocol,
)
from pyapprox.ode.protocols.type_guards import (
    is_hvp_enabled,
)
from pyapprox.ode.residual_derivatives import SecondDerivativeFn
from pyapprox.ode.state_constraint import TangentFn, sweep_tangent
from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Array, Backend


class TimeAdjointOperatorWithHVP(Generic[Array]):
    """
    Adjoint operator for time-dependent problems with HVP support.

    This class encapsulates the forward solve, adjoint solve, gradient
    computation, and Hessian-vector product computation for time-dependent
    problems.

    The Lagrangian is:
        L(y, λ, p) = Q(y, p) + Σ_n λ_n^T · R_n(y_n, y_{n-1}, p)

    where R_n is the time stepping residual and Q is the quantity of interest.

    Parameters
    ----------
    integrator : TimeIntegrator
        Time integrator with adjoint-enabled time stepping residual.
    functional : TransientFunctionalProtocol
        Functional whose ``derivatives()`` bundle has all four HVPs.

    Raises
    ------
    TypeError
        If the functional's bundle has no HVPs.
    """

    def __init__(
        self,
        integrator: TimeIntegrator[Array],
        functional: TransientFunctionalProtocol[Array],
    ):
        derivs = functional.derivatives()
        state_jacobian = derivs.state_jacobian
        state_state_hvp = derivs.state_state_hvp
        state_param_hvp = derivs.state_param_hvp
        param_state_hvp = derivs.param_state_hvp
        param_param_hvp = derivs.param_param_hvp
        # A bundle with HVPs always has its Jacobians; the check narrows.
        if (
            state_jacobian is None
            or state_state_hvp is None
            or state_param_hvp is None
            or param_state_hvp is None
            or param_param_hvp is None
        ):
            raise TypeError(
                f"{type(functional).__name__} provides no HVPs "
                "(derivatives().state_state_hvp is None); the HVP adjoint "
                "needs all four"
            )
        self._state_state_hvp: TransientStepHVPFn[Array] = state_state_hvp
        self._state_param_hvp: TransientStepHVPFn[Array] = state_param_hvp
        self._param_state_hvp: TransientStepHVPFn[Array] = param_state_hvp
        self._param_param_hvp: TransientParamHVPFn[Array] = param_param_hvp
        self._state_jacobian: TransientJacobianFn[Array] = state_jacobian
        self._integrator = integrator
        self._functional = functional
        self._bkd = integrator.bkd()
        self._time_residual = integrator.time_residual()

        # Set functional on integrator
        integrator.set_functional(functional)

        # Create storage
        nstates = functional.nstates()
        nparams = functional.nparams()
        self._storage = TimeTrajectoryStorage(nstates, nparams, self._bkd)

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def storage(self) -> TimeTrajectoryStorage[Array]:
        """Return the trajectory storage."""
        return self._storage

    def nstates(self) -> int:
        """Return the number of states."""
        return self._functional.nstates()

    def nparams(self) -> int:
        """Return the number of parameters."""
        return self._functional.nparams()

    def _make_ctx(
        self, times: Array, nn: int, fwd_sols: Array
    ) -> StepContext[Array]:
        """Build StepContext for step nn (from nn-1 to nn)."""
        return StepContext(
            t_prev=self._bkd.to_float(times[nn - 1]),
            deltat=self._bkd.to_float(times[nn] - times[nn - 1]),
            y_prev=fwd_sols[:, nn - 1],
        )

    # =========================================================================
    # Forward and Adjoint Solves
    # =========================================================================

    def _get_forward_trajectory(
        self, init_state: Array, param: Array
    ) -> Tuple[Array, Array]:
        """Get or compute forward trajectory."""
        if (
            not self._storage.has_parameter(param)
            or not self._storage.has_forward_trajectory()
        ):
            self._storage.set_parameter(param)
            native = self._time_residual.native_residual
            if not isinstance(native, ODEResidualWithParamJacobianProtocol):
                raise TypeError(
                    "adjoint/HVP operators require an ODE residual with "
                    f"parameter support; got {type(native).__name__} "
                    "(construct the residual with a parameterization)"
                )
            native.set_param(self._bkd.flatten(param))
            fwd_sols, times = self._integrator.solve(init_state)
            self._storage.set_forward_trajectory(fwd_sols, times)
        return self._storage.get_forward_trajectory()

    def _get_adjoint_trajectory(self, init_state: Array, param: Array) -> Array:
        """Get or compute adjoint trajectory."""
        if not self._storage.has_adjoint_trajectory():
            fwd_sols, times = self._get_forward_trajectory(init_state, param)
            adj_sols = self._integrator.solve_adjoint(fwd_sols, times, param)
            self._storage.set_adjoint_trajectory(adj_sols)
        return self._storage.get_adjoint_trajectory()

    # =========================================================================
    # Value, Jacobian, HVP
    # =========================================================================

    def __call__(self, init_state: Array, param: Array) -> Array:
        """Evaluate the functional Q(y(T; p), p)."""
        fwd_sols, times = self._get_forward_trajectory(init_state, param)
        return self._functional(fwd_sols, param)

    def jacobian(self, init_state: Array, param: Array) -> Array:
        """Compute the Jacobian dQ/dp via adjoint method."""
        fwd_sols, times = self._get_forward_trajectory(init_state, param)
        adj_sols = self._get_adjoint_trajectory(init_state, param)
        return self._integrator.gradient_from_adjoint_sols(
            adj_sols, fwd_sols, times, param
        )

    def hvp(self, init_state: Array, param: Array, vvec: Array) -> Array:
        """
        Compute Hessian-vector product H·v via second-order adjoints.

        Parameters
        ----------
        init_state : Array
            Initial state. Shape: (nstates,)
        param : Array
            Parameters. Shape: (nparams, 1)
        vvec : Array
            Direction vector v. Shape: (nparams, 1)

        Returns
        -------
        Array
            Hessian-vector product (d²Q/dp²)·v. Shape: (nparams, 1)
        """
        if not is_hvp_enabled(self._time_residual):
            raise RuntimeError(
                "HVP not available: time residual lacks HVP methods. "
                "Ensure the ODE residual implements state_state_hvp, etc."
            )
        hvp_residual = self._time_residual

        if vvec.shape != (self.nparams(), 1):
            raise ValueError(
                f"vvec has shape {vvec.shape} but must be ({self.nparams()}, 1)"
            )

        # Get forward and adjoint trajectories
        fwd_sols, times = self._get_forward_trajectory(init_state, param)
        adj_sols = self._get_adjoint_trajectory(init_state, param)
        # Steps are linearized at their own unknowns z_n; the functional
        # and the steps' previous states read the stored y_n = Pi(z_n).
        lin_sols = self._integrator.linearization_states(fwd_sols)
        tangent, constraint_curvature = self._constraint_terms(fwd_sols, times)

        # Solve forward sensitivity equations: stored tangents w, and the
        # unprojected dz the steps' own unknowns move by.
        w_sols, dz_sols = self._solve_forward_sensitivity(
            fwd_sols, lin_sols, times, vvec, tangent
        )

        # Solve second adjoint equations
        s_sols = self._solve_second_adjoint(
            fwd_sols, lin_sols, adj_sols, w_sols, dz_sols, times, param, vvec,
            hvp_residual, tangent, constraint_curvature,
        )

        # Accumulate HVP
        return self._accumulate_hvp(
            fwd_sols, lin_sols, adj_sols, w_sols, dz_sols, s_sols, times,
            param, vvec, hvp_residual, tangent, constraint_curvature,
        )

    def _constraint_terms(
        self, fwd_sols: Array, times: Array
    ) -> Tuple[TangentFn[Array], Optional[SecondDerivativeFn[Array]]]:
        r"""The integrator's constraint tangent :math:`\Pi'` and curvature
        :math:`\lambda^\top \Pi''`, the latter ``None`` when it is a stated
        zero (an affine constraint), so the sweep skips it.

        Raises
        ------
        TypeError
            If the constraint declares no second derivatives.
        NotImplementedError
            If its mixed or parameter second derivative is not a stated
            zero (constraint data depending on the parameters).
        """
        nres = self.nparams() - self._functional.nunique_params()
        derivs = self._integrator.state_constraint(
            fwd_sols.shape[0], nres
        ).derivatives()
        time_0 = self._bkd.to_float(times[0])
        tangent = sweep_tangent(derivs, fwd_sols[:, 0], time_0)
        seconds = (derivs.state_state, derivs.state_param, derivs.param_param)
        state_state, state_param, param_param = seconds
        if state_state is None or state_param is None or param_param is None:
            raise TypeError(
                "the Hessian-vector product needs the state constraint's "
                "second derivatives; its derivatives() lacks them"
            )
        adj = self._bkd.zeros((fwd_sols.shape[0],))
        for second in (state_param, param_param):
            if not second(fwd_sols[:, 0], time_0, adj).properties().zero:
                raise NotImplementedError(
                    "constraint data depending on the parameters is not "
                    "supported: the constraint's mixed and parameter second "
                    "derivatives must be stated zeros"
                )
        if state_state(fwd_sols[:, 0], time_0, adj).properties().zero:
            return tangent, None
        return tangent, state_state

    def _constraint_curvature_term(
        self,
        constraint_curvature: SecondDerivativeFn[Array],
        state: Array,
        time: Array,
        weight: Array,
        direction: Array,
    ) -> Array:
        r""":math:`(\mathrm{weight}^\top \Pi''(state))` applied to
        ``direction``; 1-D vectors in and out."""
        op = constraint_curvature(state, self._bkd.to_float(time), weight)
        block = self._bkd.reshape(direction, (direction.shape[0], 1))
        return op.apply(block)[:, 0]

    def _tangent_transpose(
        self, tangent: TangentFn[Array], state: Array, time: Array, vec: Array
    ) -> Array:
        r""":math:`\Pi'(state)^\top` applied to a 1-D vector."""
        block = self._bkd.reshape(vec, (vec.shape[0], 1))
        return tangent(state, self._bkd.to_float(time)).apply_transpose(block)[:, 0]

    # =========================================================================
    # Forward Sensitivity (Tangent Linear Model)
    # =========================================================================

    def _solve_forward_sensitivity(
        self,
        fwd_sols: Array,
        lin_sols: Array,
        times: Array,
        vvec: Array,
        tangent: TangentFn[Array],
    ) -> Tuple[Array, Array]:
        r"""
        Solve the tangent linear model along direction v.

        Each step moves its unknown :math:`z_n` by
        :math:`dz_n = -(\partial R_n/\partial z_n)^{-1}
        [\partial R_n/\partial y_{n-1}\, w_{n-1} + \partial R_n/\partial p\, v]`,
        linearized at :math:`z_n` and the stored :math:`y_{n-1}`, and the
        stored state by :math:`w_n = \Pi'(z_n)\, dz_n`;
        :math:`dz_0 = (dy_0/dp) v` for the initial state as given.

        Returns
        -------
        Tuple[Array, Array]
            The stored tangents :math:`w` and the unprojected :math:`dz`.
            Shape: (nstates, ntimes) each.
        """
        ntimes = fwd_sols.shape[1]
        w_sols = self._bkd.copy(self._bkd.zeros(fwd_sols.shape))
        dz_sols = self._bkd.copy(self._bkd.zeros(fwd_sols.shape))

        # Initial condition: w_0 = (dy_0/dp)·v
        ctx_0 = StepContext(
            t_prev=self._bkd.to_float(times[0]),
            deltat=self._bkd.to_float(times[1] - times[0]),
            y_prev=fwd_sols[:, 0],
        )
        self._time_residual.bind(ctx_0)
        dy0dp = self._time_residual.initial_param_jacobian()
        n_unique = self._functional.nunique_params()
        # Residual-facing direction is 1D (ODEResidual protocol); the
        # operator's public vvec stays a 2D column.
        v_res = self._bkd.flatten(
            vvec[n_unique:] if n_unique > 0 else vvec
        )
        dz_sols[:, 0:1] = (dy0dp @ v_res)[:, None]
        w_sols[:, 0:1] = tangent(
            lin_sols[:, 0], self._bkd.to_float(times[0])
        ).apply(dz_sols[:, 0:1])

        # Forward sweep: propagate sensitivity
        for nn in range(1, ntimes):
            ctx_nn = self._make_ctx(times, nn, fwd_sols)
            self._time_residual.bind(ctx_nn)

            # (dR_n/dz_n)^T (diagonal block) as an operator; its
            # transposed solve is a solve with dR_n/dz_n
            drdy_n_t = self._time_residual.adjoint_diag_jacobian(
                ctx_nn, lin_sols[:, nn]
            )

            # dR_n/dy_{n-1} (off-diagonal block), as an operator
            offdiag = self._time_residual.off_diag_operator(ctx_nn, lin_sols[:, nn])

            # dR_n/dp
            drdp_n = self._time_residual.param_jacobian(ctx_nn, lin_sols[:, nn])

            # dz_n = -(dR_n/dz_n)^{-1} · [dR_n/dy_{n-1} · w_{n-1} + dR_n/dp · v]
            rhs = (
                offdiag.apply(w_sols[:, nn - 1 : nn])
                + (drdp_n @ v_res)[:, None]
            )
            dz_sols[:, nn : nn + 1] = -drdy_n_t.solve_transpose(
                self._bkd.flatten(self._bkd.asarray(rhs))
            )[:, None]
            w_sols[:, nn : nn + 1] = tangent(
                lin_sols[:, nn], self._bkd.to_float(times[nn])
            ).apply(dz_sols[:, nn : nn + 1])

        return w_sols, dz_sols

    # =========================================================================
    # Second Adjoint (for Hessian)
    # =========================================================================

    def _solve_second_adjoint(
        self,
        fwd_sols: Array,
        lin_sols: Array,
        adj_sols: Array,
        w_sols: Array,
        dz_sols: Array,
        times: Array,
        param: Array,
        vvec: Array,
        hvp_residual: HVPEnabledTimeSteppingResidualProtocol[Array],
        tangent: TangentFn[Array],
        constraint_curvature: Optional[SecondDerivativeFn[Array]],
    ) -> Array:
        r"""
        Solve the second adjoint equations for HVP computation.

        For the unknown :math:`z_n` of step :math:`n`,
        :math:`(\partial R_n/\partial z_n)^\top s_n = -[\Pi'(z_n)^\top a_n
        + b_n + (g_n^\top \Pi''(z_n))\, dz_n]`, where :math:`a_n` collects
        what reaches :math:`z_n` through the stored :math:`y_n` (the
        coupling :math:`B_{n+1}^\top s_{n+1}`, the functional's terms and
        step :math:`n+1`'s previous-state terms), :math:`b_n` step
        :math:`n`'s own terms in its unknown, and
        :math:`g_n = dQ/dy_n + B_{n+1}^\top \lambda_{n+1}` is the bracket
        of the first adjoint, weighting the constraint's curvature (skipped
        when ``constraint_curvature`` is ``None``, an affine constraint).
        Directions follow the variable: the stored tangent :math:`w` for
        stored states, the
        unprojected :math:`dz` for a step's unknown. :math:`s_0` is
        unchanged: :math:`y_0` is projected data, not an unknown.

        Uses a unified loop: every stepper now has prev_* methods, so
        no is_explicit/is_crank_nicolson branching is needed.
        """
        ntimes = fwd_sols.shape[1]
        s_sols = self._bkd.zeros(fwd_sols.shape)
        s_sols = self._bkd.copy(s_sols)

        v_res = self._bkd.flatten(
            vvec[self._functional.nunique_params():]
            if self._functional.nunique_params() > 0
            else vvec
        )

        # Initial condition at final time
        ctx_N = self._make_ctx(times, ntimes - 1, fwd_sols)
        self._time_residual.bind(ctx_N)

        # Functional Hessian terms at final time, through y_N
        through_stored = self._bkd.flatten(
            self._state_state_hvp(fwd_sols, param, ntimes - 1, w_sols[:, -1:])
        ) + self._bkd.flatten(
            self._state_param_hvp(fwd_sols, param, ntimes - 1, vvec)
        )

        # Residual Hessian at final time: R_N same-step contribution
        own = hvp_residual.state_state_hvp(
            ctx_N, lin_sols[:, -1], adj_sols[:, -1], dz_sols[:, -1],
        ) + hvp_residual.state_param_hvp(
            ctx_N, lin_sols[:, -1], adj_sols[:, -1], v_res,
        )

        # Mixed derivative: d²R_N/(dz_N dy_{N-1}) · w_{N-1}
        # Nonzero for IM (both endpoints enter f through shared midpoint)
        if ntimes > 2:
            own = own + hvp_residual.state_prev_state_hvp(
                ctx_N, lin_sols[:, -1], adj_sols[:, -1], w_sols[:, -2],
            )

        # The constraint's curvature, weighted by g_N = dQ/dy_N.
        if constraint_curvature is not None:
            dqdu = self._state_jacobian(fwd_sols, param)
            own = own + self._constraint_curvature_term(
                constraint_curvature,
                lin_sols[:, -1],
                times[-1],
                self._time_residual.zero_adjoint_rhs(dqdu[:, -1]),
                dz_sols[:, -1],
            )

        rhs_N = -(
            self._tangent_transpose(
                tangent, lin_sols[:, -1], times[-1], through_stored
            )
            + own
        )

        # Solve (dR/dz_N)^T · s_N = RHS
        drduT_N = self._time_residual.adjoint_diag_jacobian(ctx_N, lin_sols[:, -1])
        s_sols[:, -1:] = drduT_N.solve(self._bkd.flatten(rhs_N))[:, None]

        # Backward sweep — unified loop
        for nn in range(ntimes - 2, 0, -1):
            ctx_n = self._make_ctx(times, nn, fwd_sols)
            next_ctx = StepContext(
                t_prev=self._bkd.to_float(times[nn]),
                deltat=self._bkd.to_float(times[nn + 1] - times[nn]),
                y_prev=fwd_sols[:, nn],
            )

            self._time_residual.bind(ctx_n)

            # Diagonal block (dR_n/dz_n)^T
            drduT_diag = self._time_residual.adjoint_diag_jacobian(
                ctx_n, lin_sols[:, nn]
            )

            # Off-diagonal block dR_{n+1}/dy_n, applied transposed below
            offdiag = self._time_residual.off_diag_operator(
                next_ctx, lin_sols[:, nn + 1]
            )

            # Through the stored y_n: the coupling, the functional's
            # Hessian terms, and R_{n+1}'s previous-state terms (the mixed
            # one in the direction step n+1's unknown moves, dz_{n+1}).
            through_stored = (
                offdiag.apply_transpose(s_sols[:, nn + 1 : nn + 2])[:, 0]
                + self._bkd.flatten(
                    self._state_state_hvp(
                        fwd_sols, param, nn, w_sols[:, nn : nn + 1]
                    )
                )
                + self._bkd.flatten(self._state_param_hvp(fwd_sols, param, nn, vvec))
                + hvp_residual.prev_state_state_hvp(
                    next_ctx, lin_sols[:, nn + 1], adj_sols[:, nn + 1], w_sols[:, nn],
                )
                + hvp_residual.prev_state_param_hvp(
                    next_ctx, lin_sols[:, nn + 1], adj_sols[:, nn + 1], v_res,
                )
                + hvp_residual.prev_state_curr_state_hvp(
                    next_ctx, lin_sols[:, nn + 1],
                    adj_sols[:, nn + 1], dz_sols[:, nn + 1],
                )
            )

            # R_n's own terms in its unknown z_n (mixed: nonzero for IM)
            own = (
                hvp_residual.state_state_hvp(
                    ctx_n, lin_sols[:, nn], adj_sols[:, nn], dz_sols[:, nn],
                )
                + hvp_residual.state_param_hvp(
                    ctx_n, lin_sols[:, nn], adj_sols[:, nn], v_res,
                )
                + hvp_residual.state_prev_state_hvp(
                    ctx_n, lin_sols[:, nn], adj_sols[:, nn], w_sols[:, nn - 1],
                )
            )
            # The constraint's curvature, weighted by the first adjoint's
            # bracket g_n = dQ/dy_n + B_{n+1}^T lambda_{n+1}.
            if constraint_curvature is not None:
                weight = (
                    self._time_residual.zero_adjoint_rhs(dqdu[:, nn])
                    + offdiag.apply_transpose(adj_sols[:, nn + 1 : nn + 2])[:, 0]
                )
                own = own + self._constraint_curvature_term(
                    constraint_curvature,
                    lin_sols[:, nn],
                    times[nn],
                    weight,
                    dz_sols[:, nn],
                )

            rhs = -(
                self._tangent_transpose(
                    tangent, lin_sols[:, nn], times[nn], through_stored
                )
                + own
            )

            s_sols[:, nn] = drduT_diag.solve(rhs)

        # Final step at t=0
        ctx_1 = StepContext(
            t_prev=self._bkd.to_float(times[0]),
            deltat=self._bkd.to_float(times[1] - times[0]),
            y_prev=fwd_sols[:, 0],
        )
        self._time_residual.bind(ctx_1)
        mass = self._time_residual.native_residual.mass_matrix()

        offdiag = self._time_residual.off_diag_operator(ctx_1, lin_sols[:, 1])

        qss_hvp = self._state_state_hvp(fwd_sols, param, 0, w_sols[:, 0:1])
        qsp_hvp = self._state_param_hvp(fwd_sols, param, 0, vvec)

        # Cross-step HVP: R_1 depends on y_0 as prev_state. y_0 is projected
        # data, not an unknown, so no tangent applies here.
        rss_hvp_1 = hvp_residual.prev_state_state_hvp(
            ctx_1, lin_sols[:, 1], adj_sols[:, 1], w_sols[:, 0],
        )
        rsp_hvp_1 = hvp_residual.prev_state_param_hvp(
            ctx_1, lin_sols[:, 1], adj_sols[:, 1], v_res,
        )

        # Mixed derivative: d²R_1/(dy_0 dz_1) · dz_1
        mixed_cross_1 = hvp_residual.prev_state_curr_state_hvp(
            ctx_1, lin_sols[:, 1], adj_sols[:, 1], dz_sols[:, 1],
        )
        rss_hvp_1 = rss_hvp_1 + mixed_cross_1

        rhs = (
            -offdiag.apply_transpose(s_sols[:, 1:2])[:, 0]
            - (self._bkd.flatten(qss_hvp) + rss_hvp_1)
            - (self._bkd.flatten(qsp_hvp) + rsp_hvp_1)
        )
        s_sols[:, 0] = mass.solve_transpose(rhs)

        return s_sols

    # =========================================================================
    # HVP Accumulation
    # =========================================================================

    def _accumulate_hvp(
        self,
        fwd_sols: Array,
        lin_sols: Array,
        adj_sols: Array,
        w_sols: Array,
        dz_sols: Array,
        s_sols: Array,
        times: Array,
        param: Array,
        vvec: Array,
        hvp_residual: HVPEnabledTimeSteppingResidualProtocol[Array],
        tangent: TangentFn[Array],
        constraint_curvature: Optional[SecondDerivativeFn[Array]],
    ) -> Array:
        """
        Accumulate the Hessian-vector product from all contributions.

        HVP = dQ²/dp² · v - (dy_0/dp)^T M^T s_0
              + sum_i g_i (d²y_0,i/dp²) v
              + Σ_n [(dR_n/dp)^T · s_n + L_py · w + L_pp · v]

        The middle two terms differentiate the gradient's initial-state
        term g^T dy_0/dp, with g = dQ/dy_0 + B_1^T lambda_1 = -M^T lambda_0:
        the first through s_0 (the direction derivative of lambda_0), the
        second through the curvature of y_0(p), which the residual must
        state via ``initial_param_hvp`` (zero for an affine y_0). With a
        constraint the stored y_0 is Pi(given y_0), so dy_0/dp carries
        Pi', the curvature's weight is Pi'^T g, and the constraint's own
        curvature adds (dy_0/dp)^T (g^T Pi'') dz_0, dy_0/dp and
        dz_0 = (dy_0/dp) v being the given state's. Each step's terms are
        taken at its unknown z_n, its own current-state direction being
        dz_n and its previous state's the stored w_{n-1}.
        """
        nparams = self.nparams()
        hvp = self._bkd.zeros((nparams, 1))
        hvp = self._bkd.copy(hvp)

        # Direct functional Hessian: d²Q/dp² · v
        qpp_hvp = self._param_param_hvp(fwd_sols, param, vvec)
        hvp += qpp_hvp

        n_unique = self._functional.nunique_params()
        v_res = self._bkd.flatten(
            vvec[n_unique:] if n_unique > 0 else vvec
        )

        # Initial-state contribution, using the mass the s_0 solve used.
        ctx_0 = self._make_ctx(times, 1, fwd_sols)
        self._time_residual.bind(ctx_0)
        mass = self._time_residual.native_residual.mass_matrix()
        given_dy0dp = self._time_residual.initial_param_jacobian()
        dy0dp = tangent(lin_sols[:, 0], self._bkd.to_float(times[0])).apply(
            given_dy0dp
        )
        init_hvp = -self._bkd.reshape(
            mass.apply_transpose(s_sols[:, 0]) @ dy0dp, (-1, 1)
        )
        # Curvature of y_0(p), weighted by the gradient's initial-state
        # sensitivity g = dQ/dy_0 + B_1^T lambda_1 (as in the gradient),
        # carried back through the constraint to the given state.
        dqdu_0 = self._time_residual.zero_adjoint_rhs(
            self._state_jacobian(fwd_sols, param)[:, 0]
        )
        offdiag = self._time_residual.off_diag_operator(ctx_0, lin_sols[:, 1])
        sensitivity_0 = offdiag.apply_transpose(adj_sols[:, 1:2])[:, 0] + dqdu_0
        init_weight = self._tangent_transpose(
            tangent, lin_sols[:, 0], times[0], sensitivity_0
        )
        init_hvp = init_hvp + self._bkd.reshape(
            hvp_residual.initial_param_hvp(init_weight, v_res), (-1, 1)
        )
        if constraint_curvature is not None:
            init_hvp = init_hvp + self._bkd.reshape(
                self._constraint_curvature_term(
                    constraint_curvature,
                    lin_sols[:, 0],
                    times[0],
                    sensitivity_0,
                    dz_sols[:, 0],
                )
                @ given_dy0dp,
                (-1, 1),
            )
        if n_unique > 0:
            init_full = self._bkd.zeros((nparams, 1))
            init_full = self._bkd.copy(init_full)
            init_full[n_unique:] = init_hvp
            init_hvp = init_full
        hvp += init_hvp

        # Cross-step contribution at y_0: R_1 depends on y_0 as prev_state
        ctx_1 = self._make_ctx(times, 1, fwd_sols)
        rps_hvp_0 = hvp_residual.prev_param_state_hvp(
            ctx_1, lin_sols[:, 1], adj_sols[:, 1], w_sols[:, 0],
        )
        if rps_hvp_0.ndim == 1:
            rps_hvp_0 = self._bkd.reshape(rps_hvp_0, (-1, 1))
        if n_unique > 0:
            rps_full = self._bkd.zeros((nparams, 1))
            rps_full = self._bkd.copy(rps_full)
            rps_full[n_unique:] = rps_hvp_0
            rps_hvp_0 = rps_full
        hvp += rps_hvp_0

        # Contributions from each time step — unified loop
        for nn in range(1, len(times)):
            ctx_nn = self._make_ctx(times, nn, fwd_sols)
            self._time_residual.bind(ctx_nn)

            # (dR_n/dp)^T · s_n
            drdp = self._time_residual.param_jacobian(ctx_nn, lin_sols[:, nn])
            if n_unique > 0:
                zeros = self._bkd.zeros((drdp.shape[0], n_unique))
                drdp = self._bkd.hstack((zeros, drdp))
            hvp += drdp.T @ s_sols[:, nn : nn + 1]

            # L_py · w: d²Q/dpdy + λ^T · d²R/dpdy
            qps_hvp = self._param_state_hvp(
                fwd_sols, param, nn, w_sols[:, nn : nn + 1],
            )
            # Step n's own current-state direction is dz_n.
            rps_hvp = hvp_residual.param_state_hvp(
                ctx_nn, lin_sols[:, nn],
                adj_sols[:, nn], dz_sols[:, nn],
            )

            # Cross-step contribution via prev_param_state_hvp
            if nn < len(times) - 1:
                next_ctx = StepContext(
                    t_prev=self._bkd.to_float(times[nn]),
                    deltat=self._bkd.to_float(times[nn + 1] - times[nn]),
                    y_prev=fwd_sols[:, nn],
                )
                rps_hvp_np1 = hvp_residual.prev_param_state_hvp(
                    next_ctx, lin_sols[:, nn + 1],
                    adj_sols[:, nn + 1], w_sols[:, nn],
                )
                rps_hvp = rps_hvp + rps_hvp_np1

            if rps_hvp.ndim == 1:
                rps_hvp = self._bkd.reshape(rps_hvp, (-1, 1))
            if n_unique > 0:
                rps_full = self._bkd.zeros((nparams, 1))
                rps_full = self._bkd.copy(rps_full)
                rps_full[n_unique:] = rps_hvp
                rps_hvp = rps_full

            hvp += qps_hvp + rps_hvp

            # L_pp = d²Q/dp² + λ^T · d²R/dp² (functional part already above)
            rpp_hvp = hvp_residual.param_param_hvp(
                ctx_nn, lin_sols[:, nn],
                adj_sols[:, nn], v_res,
            )
            if rpp_hvp.ndim == 1:
                rpp_hvp = self._bkd.reshape(rpp_hvp, (-1, 1))
            if n_unique > 0:
                rpp_full = self._bkd.zeros((nparams, 1))
                rpp_full = self._bkd.copy(rpp_full)
                rpp_full[n_unique:] = rpp_hvp
                rpp_hvp = rpp_full

            hvp += rpp_hvp

        return hvp

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nstates={self.nstates()}, "
            f"nparams={self.nparams()}, "
            f"has_hvp={is_hvp_enabled(self._time_residual)})"
        )
