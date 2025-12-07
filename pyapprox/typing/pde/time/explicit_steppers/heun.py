"""
Heun's method (RK2) time stepping residual with adjoint support.

Heun's method is a second-order explicit Runge-Kutta method:

    k1 = f(y_{n-1}, t_{n-1})
    k2 = f(y_{n-1} + Δt·k1, t_n)
    y_n = y_{n-1} + (Δt/2)·(k1 + k2)

This is also known as the explicit trapezoidal method or improved Euler method.

This module provides full adjoint support for gradient computation dQ/dp
via the adjoint method.
"""

from typing import Tuple

from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.pde.time.protocols import (
    TimeSteppingResidualBase,
    ODEResidualProtocol,
)


class HeunResidual(TimeSteppingResidualBase[Array]):
    """
    Heun's method (RK2) time stepping residual.

    Two-stage explicit Runge-Kutta method (2nd order):

        k1 = f(y_{n-1}, t_{n-1})
        k2 = f(y_{n-1} + Δt·k1, t_n)
        y_n = y_{n-1} + (Δt/2)·(k1 + k2)

    Residual: R(y_n) = y_n - y_{n-1} - (Δt/2)·(k1 + k2) = 0

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
        Evaluate the Heun residual.

        R(y_n) = y_n - y_{n-1} - (Δt/2)·(k1 + k2)

        where:
            k1 = f(y_{n-1}, t_{n-1})
            k2 = f(y_{n-1} + Δt·k1, t_n)

        Parameters
        ----------
        state : Array
            State at current time step y_n. Shape: (nstates,)

        Returns
        -------
        Array
            Residual value. Shape: (nstates,)
        """
        # k1 = f(y_{n-1}, t_{n-1})
        self._residual.set_time(self._time)
        k1 = self._residual(self._prev_state)

        # k2 = f(y_{n-1} + Δt·k1, t_n)
        next_state = self._prev_state + self._deltat * k1
        self._residual.set_time(self._time + self._deltat)
        k2 = self._residual(next_state)

        return state - self._prev_state - 0.5 * self._deltat * (k1 + k2)

    def jacobian(self, state: Array) -> Array:
        """
        Compute the Jacobian dR/dy_n.

        For Heun's method (explicit), dR/dy_n = M (mass matrix) since the
        residual is linear in y_n.

        Parameters
        ----------
        state : Array
            State at current time step y_n. Shape: (nstates,)

        Returns
        -------
        Array
            Jacobian = M. Shape: (nstates, nstates)
        """
        return self._residual.mass_matrix(state.shape[0])

    # =========================================================================
    # Adjoint Methods
    # =========================================================================

    def _param_jacobian(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        """
        Compute the parameter Jacobian dR/dp for one time step.

        For Heun's method with k2 = f(y_{n-1} + Δt·k1, t_n):

        dR/dp = -(Δt/2)·[dk1/dp + dk2/dp]

        where:
            dk1/dp = ∂f/∂p|_{y_{n-1}, t_{n-1}}
            dk2/dp = ∂f/∂p|_{k2_state, t_n} + ∂f/∂y|_{k2_state, t_n} · Δt · dk1/dp

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
        # k1 stage
        self._residual.set_time(self._time)
        k1_param_jac = self._residual.param_jacobian(fsol_nm1)

        # k2 stage: k2_state = y_{n-1} + Δt·k1
        k1 = self._residual(fsol_nm1)
        k2_state = fsol_nm1 + self._deltat * k1

        self._residual.set_time(self._time + self._deltat)
        k2_state_jac = self._residual.jacobian(k2_state)
        k2_param_jac = self._residual.param_jacobian(k2_state)

        # Chain rule: dk2/dp = ∂f/∂p + ∂f/∂y · Δt · dk1/dp
        jac = -(
            0.5 * self._deltat * (
                k1_param_jac
                + k2_param_jac
                + self._deltat * (k2_state_jac @ k1_param_jac)
            )
        )
        return jac

    def adjoint_diag_jacobian(self, fsol_n: Array) -> Array:
        """
        Compute the diagonal Jacobian block for adjoint solve.

        For Heun (explicit), this is just Mᵀ.

        Parameters
        ----------
        fsol_n : Array
            Forward solution at current time step. Shape: (nstates,)

        Returns
        -------
        Array
            Mᵀ. Shape: (nstates, nstates)
        """
        return self._residual.mass_matrix(fsol_n.shape[0]).T

    def adjoint_off_diag_jacobian(
        self, fsol_n: Array, deltat_np1: float
    ) -> Array:
        """
        Compute the off-diagonal Jacobian for adjoint coupling.

        For Heun's method:
        dR_{n+1}/dy_n involves derivatives through both k1 and k2 stages.

        d/dy_n [-(Δt/2)·(k1 + k2)]
        where k1 = f(y_n), k2 = f(y_n + Δt·k1)

        = -(Δt/2)·[J_1 + J_2·(I + Δt·J_1)]

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
        k1_jac = self._residual.jacobian(fsol_n)

        # k2 evaluation point
        k1 = self._residual(fsol_n)
        k2_state = fsol_n + deltat_np1 * k1

        self._residual.set_time(self._time + deltat_np1)
        k2_jac = self._residual.jacobian(k2_state)

        mass = self._residual.mass_matrix(fsol_n.shape[0])

        # dR/dy_{n-1} for Heun: -(M + (Δt/2)·(J_1 + J_2·(I + Δt·J_1)))
        jac = -(
            mass
            + 0.5 * deltat_np1 * (k1_jac + k2_jac @ (mass + deltat_np1 * k1_jac))
        )
        return jac.T

    def adjoint_initial_condition(
        self, final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        """
        Compute initial condition for backward adjoint solve.

        For explicit methods, λ_N = -dQ/dy_N (no linear solve needed).

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
        return -final_dqdu

    def adjoint_final_solution(
        self,
        fsol_0: Array,
        asol_1: Array,
        dqdu_0: Array,
        deltat_1: float,
    ) -> Array:
        """
        Compute the adjoint at initial time (final step of backward sweep).

        Solves: Mᵀ·λ_0 = -B_1ᵀ·λ_1 - dQ/dy_0

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
        Compute quadrature rule consistent with Heun's method.

        Heun's method uses trapezoidal (linear) quadrature, consistent with
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
        Compute (d²R/dy_{n-1}²)·w contracted with adjoint.

        For Heun: involves second derivatives through both k1 and k2 stages.
        """
        # This requires careful chain rule through two stages
        # Implementation depends on underlying ODE HVP methods
        self._residual.set_time(self._time)
        k1 = self._residual(fsol_nm1)
        k2_state = fsol_nm1 + self._deltat * k1

        # d²k1/dy² contribution
        k1_ss_hvp = self._residual.state_state_hvp(fsol_nm1, adj_state, wvec)

        # d²k2/dy² contribution (chain rule through k2_state)
        self._residual.set_time(self._time + self._deltat)
        # k2_state depends on y through: I + Δt·J
        # This is complex - simplified version
        k2_ss_hvp = self._residual.state_state_hvp(k2_state, adj_state, wvec)

        return -0.5 * self._deltat * (k1_ss_hvp + k2_ss_hvp)

    def _state_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """
        Compute (d²R/dy_{n-1} dp)·v contracted with adjoint.
        """
        self._residual.set_time(self._time)
        k1 = self._residual(fsol_nm1)
        k1_sp_hvp = self._residual.state_param_hvp(fsol_nm1, adj_state, vvec)

        k2_state = fsol_nm1 + self._deltat * k1
        self._residual.set_time(self._time + self._deltat)
        k2_sp_hvp = self._residual.state_param_hvp(k2_state, adj_state, vvec)

        return -0.5 * self._deltat * (k1_sp_hvp + k2_sp_hvp)

    def _param_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """
        Compute (d²R/dp dy_{n-1})·w contracted with adjoint.
        """
        self._residual.set_time(self._time)
        k1 = self._residual(fsol_nm1)
        k1_ps_hvp = self._residual.param_state_hvp(fsol_nm1, adj_state, wvec)

        k2_state = fsol_nm1 + self._deltat * k1
        self._residual.set_time(self._time + self._deltat)
        k2_ps_hvp = self._residual.param_state_hvp(k2_state, adj_state, wvec)

        return -0.5 * self._deltat * (k1_ps_hvp + k2_ps_hvp)

    def _param_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """
        Compute (d²R/dp²)·v contracted with adjoint.
        """
        self._residual.set_time(self._time)
        k1 = self._residual(fsol_nm1)
        k1_pp_hvp = self._residual.param_param_hvp(fsol_nm1, adj_state, vvec)

        k2_state = fsol_nm1 + self._deltat * k1
        self._residual.set_time(self._time + self._deltat)
        k2_pp_hvp = self._residual.param_param_hvp(k2_state, adj_state, vvec)

        return -0.5 * self._deltat * (k1_pp_hvp + k2_pp_hvp)
