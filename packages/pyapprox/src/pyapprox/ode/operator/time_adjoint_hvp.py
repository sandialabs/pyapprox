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

from typing import Generic, Tuple

from pyapprox.ode.functionals.protocols import (
    TransientFunctionalWithJacobianAndHVPProtocol,
)
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.operator.storage import TimeTrajectoryStorage
from pyapprox.ode.protocols.time_stepping import (
    HVPEnabledTimeSteppingResidualProtocol,
)
from pyapprox.ode.protocols.type_guards import (
    is_hvp_enabled,
)
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
    functional : TransientFunctionalWithJacobianAndHVPProtocol
        Functional with state/param Jacobians and HVP methods.
    """

    def __init__(
        self,
        integrator: TimeIntegrator[Array],
        functional: TransientFunctionalWithJacobianAndHVPProtocol[Array],
    ):
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
            self._time_residual.native_residual.set_param(self._bkd.flatten(param))
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

        # Solve forward sensitivity equations
        w_sols = self._solve_forward_sensitivity(fwd_sols, times, param, vvec)

        # Solve second adjoint equations
        s_sols = self._solve_second_adjoint(
            fwd_sols, adj_sols, w_sols, times, param, vvec, hvp_residual,
        )

        # Accumulate HVP
        return self._accumulate_hvp(
            fwd_sols, adj_sols, w_sols, s_sols, times, param, vvec,
            hvp_residual,
        )

    # =========================================================================
    # Forward Sensitivity (Tangent Linear Model)
    # =========================================================================

    def _solve_forward_sensitivity(
        self,
        fwd_sols: Array,
        times: Array,
        param: Array,
        vvec: Array,
    ) -> Array:
        """
        Solve the tangent linear model: w = (dy/dp)·v.

        For each time step, the sensitivity satisfies:
            dR_n/dy_n · w_n + dR_n/dy_{n-1} · w_{n-1} + dR_n/dp · v = 0

        So:
            w_n = -(dR_n/dy_n)^{-1} · [dR_n/dy_{n-1} · w_{n-1} + dR_n/dp · v]
        """
        ntimes = fwd_sols.shape[1]
        w_sols = self._bkd.zeros(fwd_sols.shape)
        w_sols = self._bkd.copy(w_sols)

        # Initial condition: w_0 = (dy_0/dp)·v
        ctx_0 = StepContext(
            t_prev=self._bkd.to_float(times[0]),
            deltat=self._bkd.to_float(times[1] - times[0]),
            y_prev=fwd_sols[:, 0],
        )
        self._time_residual.bind(ctx_0)
        dy0dp = self._time_residual.initial_param_jacobian()
        n_unique = self._functional.nunique_params()
        v_res = vvec[n_unique:] if n_unique > 0 else vvec
        w_sols[:, 0] = self._bkd.flatten(dy0dp @ v_res)

        # Forward sweep: propagate sensitivity
        for nn in range(1, ntimes):
            ctx_nn = self._make_ctx(times, nn, fwd_sols)
            self._time_residual.bind(ctx_nn)

            # dR_n/dy_n (diagonal block)
            drdy_n = self._time_residual.jacobian(fwd_sols[:, nn])

            # dR_n/dy_{n-1} (off-diagonal block)
            drdy_nm1 = self._time_residual.sensitivity_off_diag_jacobian(
                ctx_nn, fwd_sols[:, nn]
            )

            # dR_n/dp
            drdp_n = self._time_residual.param_jacobian(
                ctx_nn, fwd_sols[:, nn]
            )

            n_unique = self._functional.nunique_params()
            v_res = vvec[n_unique:] if n_unique > 0 else vvec

            # w_n = -(dR_n/dy_n)^{-1} · [dR_n/dy_{n-1} · w_{n-1} + dR_n/dp · v]
            rhs = drdy_nm1 @ w_sols[:, nn - 1 : nn] + drdp_n @ v_res
            w_sols[:, nn : nn + 1] = -self._bkd.solve(drdy_n, rhs)

        return w_sols

    # =========================================================================
    # Second Adjoint (for Hessian)
    # =========================================================================

    def _solve_second_adjoint(
        self,
        fwd_sols: Array,
        adj_sols: Array,
        w_sols: Array,
        times: Array,
        param: Array,
        vvec: Array,
        hvp_residual: HVPEnabledTimeSteppingResidualProtocol[Array],
    ) -> Array:
        """
        Solve the second adjoint equations for HVP computation.

        Uses a unified loop: every stepper now has prev_* methods, so
        no is_explicit/is_crank_nicolson branching is needed.
        """
        ntimes = fwd_sols.shape[1]
        s_sols = self._bkd.zeros(fwd_sols.shape)
        s_sols = self._bkd.copy(s_sols)

        v_res = (
            vvec[self._functional.nunique_params():]
            if self._functional.nunique_params() > 0
            else vvec
        )

        # Initial condition at final time
        ctx_N = self._make_ctx(times, ntimes - 1, fwd_sols)
        self._time_residual.bind(ctx_N)

        # Functional Hessian terms at final time
        qss_hvp = self._functional.state_state_hvp(
            fwd_sols, param, ntimes - 1, w_sols[:, -1:]
        )
        qsp_hvp = self._functional.state_param_hvp(fwd_sols, param, ntimes - 1, vvec)

        # Residual Hessian at final time: R_N same-step contribution
        rss_hvp = hvp_residual.state_state_hvp(
            ctx_N, fwd_sols[:, -1], adj_sols[:, -1], w_sols[:, -1],
        )
        rsp_hvp = hvp_residual.state_param_hvp(
            ctx_N, fwd_sols[:, -1], adj_sols[:, -1], v_res,
        )

        # Mixed derivative: d²R_N/(dy_N dy_{N-1}) · w_{N-1}
        # Nonzero for IM (both endpoints enter f through shared midpoint)
        if ntimes > 2:
            mixed_N = hvp_residual.state_prev_state_hvp(
                ctx_N, fwd_sols[:, -1], adj_sols[:, -1], w_sols[:, -2],
            )
            rss_hvp = rss_hvp + mixed_N

        rhs_N = (
            -(self._bkd.flatten(qss_hvp) + rss_hvp)
            - (self._bkd.flatten(qsp_hvp) + rsp_hvp)
        )

        # Solve (dR/dy_N)^T · s_N = RHS
        drdy_N = self._time_residual.jacobian(fwd_sols[:, -1])
        s_sols[:, -1:] = self._bkd.solve(drdy_N.T, self._bkd.reshape(rhs_N, (-1, 1)))

        # Backward sweep — unified loop
        for nn in range(ntimes - 2, 0, -1):
            ctx_n = self._make_ctx(times, nn, fwd_sols)
            next_ctx = StepContext(
                t_prev=self._bkd.to_float(times[nn]),
                deltat=self._bkd.to_float(times[nn + 1] - times[nn]),
                y_prev=fwd_sols[:, nn],
            )

            self._time_residual.bind(ctx_n)

            # Diagonal block (dR_n/dy_n)^T
            drduT_diag = self._time_residual.adjoint_diag_jacobian(
                ctx_n, fwd_sols[:, nn]
            )

            # Off-diagonal block (dR_{n+1}/dy_n)^T
            drduT_offdiag = self._time_residual.adjoint_off_diag_jacobian(
                next_ctx, fwd_sols[:, nn + 1]
            )

            # Functional Hessian terms at y_n
            qss_hvp = self._functional.state_state_hvp(
                fwd_sols, param, nn, w_sols[:, nn : nn + 1]
            )
            qsp_hvp = self._functional.state_param_hvp(fwd_sols, param, nn, vvec)

            # R_n same-step contribution
            rss_hvp_n = hvp_residual.state_state_hvp(
                ctx_n, fwd_sols[:, nn], adj_sols[:, nn], w_sols[:, nn],
            )
            rsp_hvp_n = hvp_residual.state_param_hvp(
                ctx_n, fwd_sols[:, nn], adj_sols[:, nn], v_res,
            )

            # R_{n+1} contribution (cross-step, via prev_* methods)
            rss_hvp_np1 = hvp_residual.prev_state_state_hvp(
                next_ctx, fwd_sols[:, nn + 1],
                adj_sols[:, nn + 1], w_sols[:, nn],
            )
            rsp_hvp_np1 = hvp_residual.prev_state_param_hvp(
                next_ctx, fwd_sols[:, nn + 1],
                adj_sols[:, nn + 1], v_res,
            )

            # Mixed derivatives (nonzero for IM)
            mixed_same = hvp_residual.state_prev_state_hvp(
                ctx_n, fwd_sols[:, nn],
                adj_sols[:, nn], w_sols[:, nn - 1],
            )
            mixed_cross = hvp_residual.prev_state_curr_state_hvp(
                next_ctx, fwd_sols[:, nn + 1],
                adj_sols[:, nn + 1], w_sols[:, nn + 1],
            )

            rss_hvp = rss_hvp_n + rss_hvp_np1 + mixed_same + mixed_cross
            rsp_hvp = rsp_hvp_n + rsp_hvp_np1

            rhs = (
                -drduT_offdiag @ s_sols[:, nn + 1]
                - (self._bkd.flatten(qss_hvp) + rss_hvp)
                - (self._bkd.flatten(qsp_hvp) + rsp_hvp)
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

        drduT_offdiag = self._time_residual.adjoint_off_diag_jacobian(
            ctx_1, fwd_sols[:, 1]
        )

        qss_hvp = self._functional.state_state_hvp(fwd_sols, param, 0, w_sols[:, 0:1])
        qsp_hvp = self._functional.state_param_hvp(fwd_sols, param, 0, vvec)

        # Cross-step HVP: R_1 depends on y_0 as prev_state
        rss_hvp_1 = hvp_residual.prev_state_state_hvp(
            ctx_1, fwd_sols[:, 1], adj_sols[:, 1], w_sols[:, 0],
        )
        rsp_hvp_1 = hvp_residual.prev_state_param_hvp(
            ctx_1, fwd_sols[:, 1], adj_sols[:, 1], v_res,
        )

        # Mixed derivative: d²R_1/(dy_0 dy_1) · w_1
        mixed_cross_1 = hvp_residual.prev_state_curr_state_hvp(
            ctx_1, fwd_sols[:, 1], adj_sols[:, 1], w_sols[:, 1],
        )
        rss_hvp_1 = rss_hvp_1 + mixed_cross_1

        rhs = (
            -drduT_offdiag @ s_sols[:, 1]
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
        adj_sols: Array,
        w_sols: Array,
        s_sols: Array,
        times: Array,
        param: Array,
        vvec: Array,
        hvp_residual: HVPEnabledTimeSteppingResidualProtocol[Array],
    ) -> Array:
        """
        Accumulate the Hessian-vector product from all contributions.

        HVP = dQ²/dp² · v + Σ_n [(dR_n/dp)^T · s_n + L_py · w + L_pp · v]
        """
        nparams = self.nparams()
        hvp = self._bkd.zeros((nparams, 1))
        hvp = self._bkd.copy(hvp)

        # Direct functional Hessian: d²Q/dp² · v
        qpp_hvp = self._functional.param_param_hvp(fwd_sols, param, vvec)
        hvp += qpp_hvp

        n_unique = self._functional.nunique_params()
        v_res = vvec[n_unique:] if n_unique > 0 else vvec

        # Cross-step contribution at y_0: R_1 depends on y_0 as prev_state
        ctx_1 = self._make_ctx(times, 1, fwd_sols)
        rps_hvp_0 = hvp_residual.prev_param_state_hvp(
            ctx_1, fwd_sols[:, 1], adj_sols[:, 1], w_sols[:, 0],
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
            drdp = self._time_residual.param_jacobian(
                ctx_nn, fwd_sols[:, nn]
            )
            if n_unique > 0:
                zeros = self._bkd.zeros((drdp.shape[0], n_unique))
                drdp = self._bkd.hstack((zeros, drdp))
            hvp += drdp.T @ s_sols[:, nn : nn + 1]

            # L_py · w: d²Q/dpdy + λ^T · d²R/dpdy
            qps_hvp = self._functional.param_state_hvp(
                fwd_sols, param, nn, w_sols[:, nn : nn + 1],
            )
            rps_hvp = hvp_residual.param_state_hvp(
                ctx_nn, fwd_sols[:, nn],
                adj_sols[:, nn], w_sols[:, nn],
            )

            # Cross-step contribution via prev_param_state_hvp
            if nn < len(times) - 1:
                next_ctx = StepContext(
                    t_prev=self._bkd.to_float(times[nn]),
                    deltat=self._bkd.to_float(times[nn + 1] - times[nn]),
                    y_prev=fwd_sols[:, nn],
                )
                rps_hvp_np1 = hvp_residual.prev_param_state_hvp(
                    next_ctx, fwd_sols[:, nn + 1],
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
                ctx_nn, fwd_sols[:, nn],
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
