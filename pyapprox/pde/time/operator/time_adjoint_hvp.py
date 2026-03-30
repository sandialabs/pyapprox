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

from pyapprox.pde.time.functionals.protocols import (
    TransientFunctionalWithJacobianAndHVPProtocol,
)
from pyapprox.pde.time.implicit_steppers.integrator import TimeIntegrator
from pyapprox.pde.time.operator.storage import TimeTrajectoryStorage
from pyapprox.pde.time.protocols.time_stepping import (
    HVPEnabledTimeSteppingResidualProtocol,
    PrevStepHVPEnabledTimeSteppingResidualProtocol,
)
from pyapprox.pde.time.protocols.type_guards import (
    is_hvp_enabled,
    is_prev_step_hvp_enabled,
)
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

    Notes
    -----
    The time stepping residual must have HVP methods for full HVP support.
    Check with ``hasattr(integrator.time_residual(), 'state_state_hvp')``.
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

    def _is_explicit_scheme(self) -> bool:
        """Check if the time stepping scheme is explicit."""
        return bool(self._time_residual.is_explicit())

    def _is_crank_nicolson(self) -> bool:
        """Check if scheme has prev-state Hessian contributions.

        Returns True for schemes where R_{n+1} depends on f(y_n), meaning
        the Hessian at y_n includes contributions from BOTH R_n and R_{n+1}.
        (e.g., Crank-Nicolson)
        """
        return bool(self._time_residual.has_prev_state_hessian())

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
        """
        Evaluate the functional Q(y(T; p), p).

        Parameters
        ----------
        init_state : Array
            Initial state. Shape: (nstates,)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Functional value. Shape: (1, 1)
        """
        fwd_sols, times = self._get_forward_trajectory(init_state, param)
        return self._functional(fwd_sols, param)

    def jacobian(self, init_state: Array, param: Array) -> Array:
        """
        Compute the Jacobian dQ/dp via adjoint method.

        Parameters
        ----------
        init_state : Array
            Initial state. Shape: (nstates,)
        param : Array
            Parameters. Shape: (nparams, 1)

        Returns
        -------
        Array
            Jacobian dQ/dp. Shape: (1, nparams)
        """
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

        Raises
        ------
        RuntimeError
            If HVP methods are not available on the time residual.
        """
        if not is_hvp_enabled(self._time_residual):
            raise RuntimeError(
                "HVP not available: time residual lacks HVP methods. "
                "Ensure the ODE residual implements state_state_hvp, etc."
            )
        hvp_residual = self._time_residual
        prev_hvp_residual: Optional[
            PrevStepHVPEnabledTimeSteppingResidualProtocol[Array]
        ] = (
            hvp_residual
            if is_prev_step_hvp_enabled(hvp_residual)
            else None
        )

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
            fwd_sols, adj_sols, w_sols, times, param, vvec,
            hvp_residual, prev_hvp_residual,
        )

        # Accumulate HVP
        return self._accumulate_hvp(
            fwd_sols, adj_sols, w_sols, s_sols, times, param, vvec,
            hvp_residual, prev_hvp_residual,
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

        Parameters
        ----------
        fwd_sols : Array
            Forward solutions. Shape: (nstates, ntimes)
        times : Array
            Time points. Shape: (ntimes,)
        param : Array
            Parameters. Shape: (nparams, 1)
        vvec : Array
            Direction vector. Shape: (nparams, 1)

        Returns
        -------
        Array
            Sensitivity trajectory w. Shape: (nstates, ntimes)
        """
        ntimes = fwd_sols.shape[1]
        w_sols = self._bkd.zeros(fwd_sols.shape)
        w_sols = self._bkd.copy(w_sols)

        # Initial condition: w_0 = (dy_0/dp)·v
        deltat_0 = self._bkd.to_float(times[1] - times[0])
        t0 = self._bkd.to_float(times[0])
        self._time_residual.set_time(t0, deltat_0, fwd_sols[:, 0])
        dy0dp = self._time_residual.initial_param_jacobian()
        # Only use residual params (exclude functional params)
        n_unique = self._functional.nunique_params()
        if n_unique > 0:
            v_res = vvec[n_unique:]
        else:
            v_res = vvec
        w_sols[:, 0] = self._bkd.flatten(dy0dp @ v_res)

        # Forward sweep: propagate sensitivity
        for nn in range(1, ntimes):
            deltat_n = self._bkd.to_float(times[nn] - times[nn - 1])
            self._time_residual.set_time(
                self._bkd.to_float(times[nn - 1]), deltat_n, fwd_sols[:, nn - 1]
            )

            # dR_n/dy_n (diagonal block)
            drdy_n = self._time_residual.jacobian(fwd_sols[:, nn])

            # dR_n/dy_{n-1} (off-diagonal block)
            # This is embedded in adjoint_off_diag_jacobian (transposed)
            # We need the non-transposed version
            drdy_nm1 = self._get_sensitivity_off_diag(
                fwd_sols[:, nn - 1], fwd_sols[:, nn], deltat_n
            )

            # dR_n/dp
            drdp_n = self._time_residual.param_jacobian(
                fwd_sols[:, nn - 1], fwd_sols[:, nn]
            )

            # Handle functional-only parameters
            n_unique = self._functional.nunique_params()
            if n_unique > 0:
                v_res = vvec[n_unique:]
            else:
                v_res = vvec

            # w_n = -(dR_n/dy_n)^{-1} · [dR_n/dy_{n-1} · w_{n-1} + dR_n/dp · v]
            rhs = drdy_nm1 @ w_sols[:, nn - 1 : nn] + drdp_n @ v_res
            w_sols[:, nn : nn + 1] = -self._bkd.solve(drdy_n, rhs)

        return w_sols

    def _get_sensitivity_off_diag(
        self, fsol_nm1: Array, fsol_n: Array, deltat_n: float
    ) -> Array:
        """
        Get dR_n/dy_{n-1} for sensitivity propagation.

        Delegates to the stepper's sensitivity_off_diag_jacobian method.
        Each stepper knows its own formula for dR_n/dy_{n-1}.
        """
        return self._time_residual.sensitivity_off_diag_jacobian(
            fsol_nm1, fsol_n, deltat_n
        )

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
        prev_hvp_residual: Optional[
            PrevStepHVPEnabledTimeSteppingResidualProtocol[Array]
        ],
    ) -> Array:
        """
        Solve the second adjoint equations for HVP computation.

        The second adjoint s satisfies equations similar to the first adjoint,
        but with Hessian terms on the RHS.

        Key insight from Heinkenschloss Algorithm 4.1:
        - Algorithm 4.1 defines w via: c_y·w = c_u·v (positive sign)
        - Our implementation uses w = dy/dp·v, which is w = -c_y^{-1}·c_u·v
        - So our w = -w_Heinkenschloss

        This affects the second adjoint RHS:
        - Algorithm 4.1: c_y^T·p = ∇_{yy}L·w_H - ∇_{yu}L·v
        - With our w: RHS = -∇_{yy}L·w - ∇_{yu}L·v

        Parameters
        ----------
        fwd_sols : Array
            Forward solutions. Shape: (nstates, ntimes)
        adj_sols : Array
            First adjoint solutions. Shape: (nstates, ntimes)
        w_sols : Array
            Sensitivity solutions w = dy/dp·v. Shape: (nstates, ntimes)
        times : Array
            Time points. Shape: (ntimes,)
        param : Array
            Parameters. Shape: (nparams, 1)
        vvec : Array
            Direction vector. Shape: (nparams, 1)

        Returns
        -------
        Array
            Second adjoint trajectory s. Shape: (nstates, ntimes)
        """
        ntimes = fwd_sols.shape[1]
        s_sols = self._bkd.zeros(fwd_sols.shape)
        s_sols = self._bkd.copy(s_sols)

        # Initial condition at final time
        deltat_N = self._bkd.to_float(times[-1] - times[-2])
        t_Nm1 = self._bkd.to_float(times[-2])
        self._time_residual.set_time(t_Nm1, deltat_N, fwd_sols[:, -2])

        # Functional Hessian terms at final time
        # ∇_{yy}Q · w and ∇_{yu}Q · v
        qss_hvp = self._functional.state_state_hvp(
            fwd_sols, param, ntimes - 1, w_sols[:, -1:]
        )
        qsp_hvp = self._functional.state_param_hvp(fwd_sols, param, ntimes - 1, vvec)

        # For explicit schemes (Forward Euler, Heun):
        #   R_n = y_n - y_{n-1} - Δt·f(y_{n-1})
        #   dR_n/dy_n = I (constant), so d²R_n/dy_n² = 0
        #   The Hessian contribution comes from the off-diagonal coupling
        #   via d/dp[c_Y^T], not from d²R_n/dy_n².
        #
        # For implicit schemes (Backward Euler, Crank-Nicolson):
        #   R_n depends on y_n through f(y_n), so we include d²R_n/dy_n².

        if self._is_explicit_scheme():
            # For explicit schemes, s_N = 0 at final time because:
            # - The final row of c_Y^T is [0, ..., 0, I] which doesn't depend on p
            # - Only functional Hessian terms contribute
            rhs_N = -(self._bkd.flatten(qss_hvp) + self._bkd.flatten(qsp_hvp))
        else:
            # For implicit schemes, include residual Hessian at final time
            rss_hvp = hvp_residual.state_state_hvp(
                fwd_sols[:, -2],
                fwd_sols[:, -1],
                adj_sols[:, -1],
                w_sols[:, -1],  # Use w_N for implicit
            )
            rsp_hvp = hvp_residual.state_param_hvp(
                fwd_sols[:, -2],
                fwd_sols[:, -1],
                adj_sols[:, -1],
                vvec[self._functional.nunique_params() :]
                if self._functional.nunique_params() > 0
                else vvec,
            )
            rhs_N = -(self._bkd.flatten(qss_hvp) + rss_hvp) - (self._bkd.flatten(qsp_hvp) + rsp_hvp)

        # Solve (dR/dy_N)^T · s_N = RHS
        drdy_N = self._time_residual.jacobian(fwd_sols[:, -1])
        s_sols[:, -1:] = self._bkd.solve(drdy_N.T, self._bkd.reshape(rhs_N, (-1, 1)))

        # Backward sweep
        for nn in range(ntimes - 2, 0, -1):
            deltat_np1 = self._bkd.to_float(times[nn + 1] - times[nn])
            deltat_n = self._bkd.to_float(times[nn] - times[nn - 1])

            # Diagonal block (dR_n/dy_n)^T for solving s_n
            self._time_residual.set_time(
                self._bkd.to_float(times[nn - 1]), deltat_n, fwd_sols[:, nn - 1]
            )
            drduT_diag = self._time_residual.adjoint_diag_jacobian(fwd_sols[:, nn])

            # Off-diagonal block (dR_{n+1}/dy_n)^T for coupling to s_{n+1}
            drduT_offdiag = self._time_residual.adjoint_off_diag_jacobian(
                fwd_sols[:, nn], deltat_np1
            )

            # Functional Hessian terms at y_n
            qss_hvp = self._functional.state_state_hvp(
                fwd_sols, param, nn, w_sols[:, nn : nn + 1]
            )
            qsp_hvp = self._functional.state_param_hvp(fwd_sols, param, nn, vvec)

            # Residual Hessian terms: ∇_{y_n y_n}L and ∇_{y_n p}L
            #
            # For explicit schemes (Forward Euler, Heun):
            #   R_n depends on y_{n-1}, so d²R_n/dy_n² = 0
            #   The Hessian at y_n comes from R_{n+1} (which depends on y_n)
            #   ∇_{y_n y_n}L = λ_{n+1} * d²R_{n+1}/dy_n²
            #   Use: time context for R_{n+1}, adj_sols[:, n+1], w_sols[:, n]
            #
            # For implicit schemes (Backward Euler, Crank-Nicolson):
            #   R_n depends on y_n, so d²R_n/dy_n² ≠ 0
            #   ∇_{y_n y_n}L = λ_n * d²R_n/dy_n² (+ contribution from R_{n+1})
            #   Use: time context for R_n, adj_sols[:, n], w_sols[:, n]

            v_res = (
                vvec[self._functional.nunique_params() :]
                if self._functional.nunique_params() > 0
                else vvec
            )

            if self._is_explicit_scheme():
                # For explicit: Hessian at y_n comes from R_{n+1}
                # Set time context for R_{n+1}
                self._time_residual.set_time(
                    self._bkd.to_float(times[nn]), deltat_np1, fwd_sols[:, nn]
                )
                rss_hvp = hvp_residual.state_state_hvp(
                    fwd_sols[:, nn],  # y_{n} = prev state for R_{n+1}
                    fwd_sols[:, nn + 1],  # y_{n+1} = current state
                    adj_sols[:, nn + 1],  # λ_{n+1}
                    w_sols[:, nn],  # w_n (sensitivity at y_n)
                )
                rsp_hvp = hvp_residual.state_param_hvp(
                    fwd_sols[:, nn],
                    fwd_sols[:, nn + 1],
                    adj_sols[:, nn + 1],
                    v_res,
                )
            elif self._is_crank_nicolson():
                # For Crank-Nicolson: Hessian at y_n comes from BOTH R_n AND R_{n+1}
                # - R_n contribution: λ_n^T · d²R_n/dy_n² (evaluated at y_n)
                # - R_{n+1} contribution: λ_{n+1}^T · d²R_{n+1}/dy_n² (y_n is y_{n-1}
                # for R_{n+1})

                # Contribution from R_n (time context already set for R_n above)
                rss_hvp_n = hvp_residual.state_state_hvp(
                    fwd_sols[:, nn - 1],
                    fwd_sols[:, nn],
                    adj_sols[:, nn],
                    w_sols[:, nn],
                )
                rsp_hvp_n = hvp_residual.state_param_hvp(
                    fwd_sols[:, nn - 1],
                    fwd_sols[:, nn],
                    adj_sols[:, nn],
                    v_res,
                )

                # Contribution from R_{n+1} (using prev_* methods)
                # Set time to t_n for evaluating f at y_n
                if prev_hvp_residual is None:
                    raise TypeError(
                        "Crank-Nicolson HVP requires prev_* methods but "
                        "time residual does not implement "
                        "PrevStepHVPEnabledTimeSteppingResidualProtocol"
                    )
                prev_hvp_residual.native_residual.set_time(
                    self._bkd.to_float(times[nn])
                )
                rss_hvp_np1 = prev_hvp_residual.prev_state_state_hvp(
                    fwd_sols[:, nn],  # y_n (acts as y_{n-1} for R_{n+1})
                    adj_sols[:, nn + 1],  # λ_{n+1}
                    w_sols[:, nn],  # w_n
                )
                rsp_hvp_np1 = prev_hvp_residual.prev_state_param_hvp(
                    fwd_sols[:, nn],
                    adj_sols[:, nn + 1],
                    v_res,
                )

                rss_hvp = rss_hvp_n + rss_hvp_np1
                rsp_hvp = rsp_hvp_n + rsp_hvp_np1
            else:
                # For other implicit schemes (Backward Euler): Hessian at y_n comes from
                # R_n only
                # Time context already set for R_n above
                rss_hvp = hvp_residual.state_state_hvp(
                    fwd_sols[:, nn - 1],
                    fwd_sols[:, nn],
                    adj_sols[:, nn],
                    w_sols[:, nn],
                )
                rsp_hvp = hvp_residual.state_param_hvp(
                    fwd_sols[:, nn - 1],
                    fwd_sols[:, nn],
                    adj_sols[:, nn],
                    v_res,
                )

            # RHS: -B^T·s_{n+1} - ∇_{yy}L·w - ∇_{yu}L·v
            rhs = (
                -drduT_offdiag @ s_sols[:, nn + 1]
                - (self._bkd.flatten(qss_hvp) + rss_hvp)
                - (self._bkd.flatten(qsp_hvp) + rsp_hvp)
            )

            s_sols[:, nn] = self._bkd.solve(drduT_diag, rhs)

        # Final step at t=0
        deltat_1 = self._bkd.to_float(times[1] - times[0])
        t0 = self._bkd.to_float(times[0])
        self._time_residual.set_time(t0, deltat_1, fwd_sols[:, 0])
        drduT_diag = self._time_residual.native_residual.mass_matrix(
            fwd_sols.shape[0]
        ).T
        drduT_offdiag = self._time_residual.adjoint_off_diag_jacobian(
            fwd_sols[:, 0], deltat_1
        )

        qss_hvp = self._functional.state_state_hvp(fwd_sols, param, 0, w_sols[:, 0:1])
        qsp_hvp = self._functional.state_param_hvp(fwd_sols, param, 0, vvec)

        # CORRECTED RHS
        rhs = -drduT_offdiag @ s_sols[:, 1] - self._bkd.flatten(qss_hvp) - self._bkd.flatten(qsp_hvp)
        s_sols[:, 0] = self._bkd.solve(drduT_diag, rhs)

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
        prev_hvp_residual: Optional[
            PrevStepHVPEnabledTimeSteppingResidualProtocol[Array]
        ],
    ) -> Array:
        """
        Accumulate the Hessian-vector product from all contributions.

        HVP = dQ²/dp² · v + Σ_n [(dR_n/dp)^T · s_n - L_py · w + L_pp · v]

        Parameters
        ----------
        fwd_sols : Array
            Forward solutions. Shape: (nstates, ntimes)
        adj_sols : Array
            First adjoint solutions. Shape: (nstates, ntimes)
        w_sols : Array
            Sensitivity solutions. Shape: (nstates, ntimes)
        s_sols : Array
            Second adjoint solutions. Shape: (nstates, ntimes)
        times : Array
            Time points. Shape: (ntimes,)
        param : Array
            Parameters. Shape: (nparams, 1)
        vvec : Array
            Direction vector. Shape: (nparams, 1)

        Returns
        -------
        Array
            Hessian-vector product. Shape: (nparams, 1)
        """
        nparams = self.nparams()
        hvp = self._bkd.zeros((nparams, 1))
        hvp = self._bkd.copy(hvp)

        # Direct functional Hessian: d²Q/dp² · v
        qpp_hvp = self._functional.param_param_hvp(fwd_sols, param, vvec)
        hvp += qpp_hvp

        # Contributions from each time step
        for nn in range(1, len(times)):
            deltat_n = self._bkd.to_float(times[nn] - times[nn - 1])
            self._time_residual.set_time(
                self._bkd.to_float(times[nn - 1]), deltat_n, fwd_sols[:, nn - 1]
            )

            # (dR_n/dp)^T · s_n
            drdp = self._time_residual.param_jacobian(
                fwd_sols[:, nn - 1], fwd_sols[:, nn]
            )
            n_unique = self._functional.nunique_params()
            if n_unique > 0:
                # Prepend zeros for functional-only parameters
                zeros = self._bkd.zeros((drdp.shape[0], n_unique))
                drdp = self._bkd.hstack((zeros, drdp))
            hvp += drdp.T @ s_sols[:, nn : nn + 1]

            # +L_py · w + L_pp · v (corrected sign: our w = -w_Heinkenschloss)
            # L_py = d²Q/dpdy + λ^T · d²R/dpdy
            # For explicit schemes, use w_{n-1} for residual Hessian
            w_idx = nn - 1 if self._is_explicit_scheme() else nn
            qps_hvp = self._functional.param_state_hvp(
                fwd_sols,
                param,
                nn,
                w_sols[:, nn : nn + 1],  # Functional uses w_n
            )
            rps_hvp = hvp_residual.param_state_hvp(
                fwd_sols[:, nn - 1],
                fwd_sols[:, nn],
                adj_sols[:, nn],
                w_sols[:, w_idx],  # Residual uses w_{n-1} for explicit schemes
            )

            # For Crank-Nicolson: also add contribution from R_{n+1}
            # (λ_{n+1}^T · ∂²R_{n+1}/∂p ∂y_n · w_n)
            if self._is_crank_nicolson() and nn < len(times) - 1:
                if prev_hvp_residual is None:
                    raise TypeError(
                        "Crank-Nicolson HVP requires prev_* methods but "
                        "time residual does not implement "
                        "PrevStepHVPEnabledTimeSteppingResidualProtocol"
                    )
                prev_hvp_residual.native_residual.set_time(
                    self._bkd.to_float(times[nn])
                )
                rps_hvp_np1 = prev_hvp_residual.prev_param_state_hvp(
                    fwd_sols[:, nn],  # y_n (acts as y_{n-1} for R_{n+1})
                    adj_sols[:, nn + 1],  # λ_{n+1}
                    w_sols[:, nn],  # w_n
                )
                rps_hvp = rps_hvp + rps_hvp_np1

            # Handle dimension - rps_hvp may be (nparams_res,)
            if rps_hvp.ndim == 1:
                rps_hvp = self._bkd.reshape(rps_hvp, (-1, 1))
            if n_unique > 0:
                rps_full = self._bkd.zeros((nparams, 1))
                rps_full = self._bkd.copy(rps_full)
                rps_full[n_unique:] = rps_hvp
                rps_hvp = rps_full

            # CORRECTED: Since our w = -w_Heinkenschloss, the -∇_{uy}L·w_H
            # term becomes +∇_{uy}L·w
            hvp += qps_hvp + rps_hvp

            # L_pp = d²Q/dp² + λ^T · d²R/dp²
            # (functional d²Q/dp² already added above)
            v_res = vvec[n_unique:] if n_unique > 0 else vvec
            rpp_hvp = hvp_residual.param_param_hvp(
                fwd_sols[:, nn - 1],
                fwd_sols[:, nn],
                adj_sols[:, nn],
                v_res,
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
