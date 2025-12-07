"""
Crank-Nicolson time stepping residual with adjoint support.

The Crank-Nicolson method is a second-order implicit time integrator:

    y_n - y_{n-1} - (Δt/2)·[f(y_{n-1}, t_{n-1}) + f(y_n, t_n)] = 0

This is also known as the trapezoidal rule or implicit midpoint method.

This module provides full adjoint support for gradient computation dQ/dp
via the adjoint method.
"""

from typing import Tuple

from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.pde.time.protocols import (
    TimeSteppingResidualBase,
    ODEResidualProtocol,
)


class CrankNicolsonResidual(TimeSteppingResidualBase[Array]):
    """
    Crank-Nicolson time stepping residual.

    Residual: R(y_n) = y_n - y_{n-1} - (Δt/2)·[f(y_{n-1}, t_{n-1}) + f(y_n, t_n)] = 0

    This is a second-order implicit method (A-stable).

    Optional Methods
    ----------------
    The following methods are conditionally available based on the
    underlying ODE residual capabilities:

    - ``param_jacobian(fsol_nm1, fsol_n)``: Available if ODE has ``param_jacobian``
    - ``state_state_hvp(...)``, etc.: Available if ODE has HVP methods

    Check availability with ``hasattr(residual, 'param_jacobian')``.
    """

    def __call__(self, state: Array) -> Array:
        """
        Evaluate the Crank-Nicolson residual.

        R(y_n) = y_n - y_{n-1} - (Δt/2)·[f(y_{n-1}, t_{n-1}) + f(y_n, t_n)]

        Parameters
        ----------
        state : Array
            State at current time step y_n. Shape: (nstates,)

        Returns
        -------
        Array
            Residual value. Shape: (nstates,)
        """
        # f(y_{n-1}, t_{n-1})
        self._residual.set_time(self._time)
        current_res = self._residual(self._prev_state)

        # f(y_n, t_n)
        self._residual.set_time(self._time + self._deltat)
        next_res = self._residual(state)

        return (
            state
            - self._prev_state
            - 0.5 * self._deltat * (current_res + next_res)
        )

    def jacobian(self, state: Array) -> Array:
        """
        Compute the Jacobian dR/dy_n.

        dR/dy_n = M - (Δt/2)·(df/dy)|_{y_n, t_n}

        Parameters
        ----------
        state : Array
            State at current time step y_n. Shape: (nstates,)

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nstates, nstates)
        """
        self._residual.set_time(self._time + self._deltat)
        return (
            self._residual.mass_matrix(state.shape[0])
            - 0.5 * self._deltat * self._residual.jacobian(state)
        )

    # =========================================================================
    # Adjoint Methods
    # =========================================================================

    def _param_jacobian(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        """
        Compute the parameter Jacobian dR/dp for one time step.

        dR/dp = -(Δt/2)·[(df/dp)|_{y_{n-1}, t_{n-1}} + (df/dp)|_{y_n, t_n}]

        Parameters
        ----------
        fsol_nm1 : Array
            Forward solution at previous time step. Shape: (nstates,)
        fsol_n : Array
            Forward solution at current time step. Shape: (nstates,)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nstates, nparams)
        """
        self._residual.set_time(self._time)
        current_param_jac = self._residual.param_jacobian(fsol_nm1)

        self._residual.set_time(self._time + self._deltat)
        next_param_jac = self._residual.param_jacobian(fsol_n)

        return -0.5 * self._deltat * (current_param_jac + next_param_jac)

    def adjoint_diag_jacobian(self, fsol_n: Array) -> Array:
        """
        Compute the diagonal Jacobian block for adjoint solve.

        (dR/dy_n)^T = (M - (Δt/2)·J)^T

        Note: set_time should have been called with time = t_n (current time),
        not t_{n-1}. The adjoint solve works backward from final time.

        Parameters
        ----------
        fsol_n : Array
            Forward solution at current time step. Shape: (nstates,)

        Returns
        -------
        Array
            Transpose of Jacobian. Shape: (nstates, nstates)
        """
        self._residual.set_time(self._time)
        return (
            self._residual.mass_matrix(fsol_n.shape[0])
            - 0.5 * self._deltat * self._residual.jacobian(fsol_n)
        ).T

    def adjoint_off_diag_jacobian(
        self, fsol_n: Array, deltat_np1: float
    ) -> Array:
        """
        Compute the off-diagonal Jacobian for adjoint coupling.

        For Crank-Nicolson:
        dR_{n+1}/dy_n = -(M + (Δt_{n+1}/2)·J_n)

        Parameters
        ----------
        fsol_n : Array
            Forward solution at current time step. Shape: (nstates,)
        deltat_np1 : float
            Time step size for the next interval.

        Returns
        -------
        Array
            Off-diagonal coupling. Shape: (nstates, nstates)
        """
        self._residual.set_time(self._time)
        return -(
            self._residual.mass_matrix(fsol_n.shape[0])
            + 0.5 * deltat_np1 * self._residual.jacobian(fsol_n)
        ).T

    def adjoint_initial_condition(
        self, final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        """
        Compute initial condition for backward adjoint solve.

        At final time T, solve: (dR/dy_N)^T·λ_N = -dQ/dy_N

        Parameters
        ----------
        final_fwd_sol : Array
            Forward solution at final time. Shape: (nstates,)
        final_dqdu : Array
            Gradient dQ/dy at final time. Shape: (nstates,)

        Returns
        -------
        Array
            Adjoint solution at final time λ_N. Shape: (nstates,)
        """
        drdu = self.jacobian(final_fwd_sol)
        return self._bkd.solve(drdu.T, -final_dqdu)

    def adjoint_final_solution(
        self,
        fsol_0: Array,
        asol_1: Array,
        dqdu_0: Array,
        deltat_1: float,
    ) -> Array:
        """
        Compute the adjoint at initial time (final step of backward sweep).

        Solves: M^T·λ_0 = -B_1^T·λ_1 - dQ/dy_0

        Parameters
        ----------
        fsol_0 : Array
            Forward solution at initial time. Shape: (nstates,)
        asol_1 : Array
            Adjoint solution at time step 1. Shape: (nstates,)
        dqdu_0 : Array
            Gradient dQ/dy at initial time. Shape: (nstates,)
        deltat_1 : float
            Time step size for first interval.

        Returns
        -------
        Array
            Adjoint solution at initial time λ_0. Shape: (nstates,)
        """
        drduT_diag = self._residual.mass_matrix(fsol_0.shape[0]).T
        drduT_offdiag = self.adjoint_off_diag_jacobian(fsol_0, deltat_1)
        return self._bkd.solve(drduT_diag, -drduT_offdiag @ asol_1 - dqdu_0)

    def quadrature_samples_weights(self, times: Array) -> Tuple[Array, Array]:
        """
        Compute quadrature rule consistent with Crank-Nicolson.

        Crank-Nicolson uses trapezoidal (linear) quadrature, consistent with
        its 2nd order accuracy.

        Parameters
        ----------
        times : Array
            Time points. Shape: (ntimes,)

        Returns
        -------
        quadx : Array
            Quadrature sample points. Shape: (ntimes,)
        quadw : Array
            Quadrature weights. Shape: (ntimes,)
        """
        from pyapprox.typing.surrogates.basis.piecewisepoly.linear import (
            PiecewiseLinear,
        )
        quadrature = PiecewiseLinear(times, self._bkd)
        return quadrature.quadrature_rule()

    # =========================================================================
    # HVP Methods (conditionally available via dynamic binding)
    # =========================================================================

    def _state_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """
        Compute (d^2R/dy_n^2)·w contracted with adjoint.

        For Crank-Nicolson: d^2R/dy^2 = -(Δt/2)·(d^2f/dy^2)|_{y_n, t_n}
        """
        self._residual.set_time(self._time + self._deltat)
        return -0.5 * self._deltat * self._residual.state_state_hvp(
            fsol_n, adj_state, wvec
        )

    def _state_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """
        Compute (d^2R/dy_n dp)·v contracted with adjoint.

        For Crank-Nicolson: contributions from both y_{n-1} and y_n
        """
        # Contribution from y_{n-1} term
        self._residual.set_time(self._time)
        hvp_nm1 = self._residual.state_param_hvp(fsol_nm1, adj_state, vvec)

        # Contribution from y_n term
        self._residual.set_time(self._time + self._deltat)
        hvp_n = self._residual.state_param_hvp(fsol_n, adj_state, vvec)

        return -0.5 * self._deltat * (hvp_nm1 + hvp_n)

    def _param_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """
        Compute (d^2R/dp dy_n)·w contracted with adjoint.

        For Crank-Nicolson: contributions from both y_{n-1} and y_n
        """
        # Contribution from y_{n-1} term
        self._residual.set_time(self._time)
        hvp_nm1 = self._residual.param_state_hvp(fsol_nm1, adj_state, wvec)

        # Contribution from y_n term
        self._residual.set_time(self._time + self._deltat)
        hvp_n = self._residual.param_state_hvp(fsol_n, adj_state, wvec)

        return -0.5 * self._deltat * (hvp_nm1 + hvp_n)

    def _param_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """
        Compute (d^2R/dp^2)·v contracted with adjoint.

        For Crank-Nicolson: contributions from both y_{n-1} and y_n
        """
        # Contribution from y_{n-1} term
        self._residual.set_time(self._time)
        hvp_nm1 = self._residual.param_param_hvp(fsol_nm1, adj_state, vvec)

        # Contribution from y_n term
        self._residual.set_time(self._time + self._deltat)
        hvp_n = self._residual.param_param_hvp(fsol_n, adj_state, vvec)

        return -0.5 * self._deltat * (hvp_nm1 + hvp_n)
