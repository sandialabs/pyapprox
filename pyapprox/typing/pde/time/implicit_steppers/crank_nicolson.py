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
    # Sensitivity Protocol Methods
    # =========================================================================

    def is_explicit(self) -> bool:
        """Return False since Crank-Nicolson is an implicit scheme."""
        return False

    def has_prev_state_hessian(self) -> bool:
        """
        Return True since R_{n+1} depends on f(y_n) (y_n is y_{n-1} for R_{n+1}).

        Crank-Nicolson: R_n = y_n - y_{n-1} - (Δt/2)·[f(y_{n-1}) + f(y_n)]

        This means R_{n+1} has a term f(y_n), so when computing d²L/dy_n²
        we need to include the contribution from R_{n+1}'s dependence on y_n.
        """
        return True

    def sensitivity_off_diag_jacobian(
        self, fsol_nm1: Array, fsol_n: Array, deltat: float
    ) -> Array:
        """
        Compute dR_n/dy_{n-1} for forward sensitivity propagation.

        For Crank-Nicolson R_n = y_n - y_{n-1} - (Δt/2)·[f(y_{n-1}) + f(y_n)]:
            dR_n/dy_{n-1} = -(M + (Δt/2)·J_{n-1})

        Parameters
        ----------
        fsol_nm1 : Array
            Solution at previous time step y_{n-1}. Shape: (nstates,)
        fsol_n : Array
            Solution at current time step y_n. Shape: (nstates,)
        deltat : float
            Time step size Δt.

        Returns
        -------
        Array
            Off-diagonal Jacobian dR_n/dy_{n-1}. Shape: (nstates, nstates)
        """
        self._residual.set_time(self._time)
        mass = self._residual.mass_matrix(fsol_nm1.shape[0])
        jac = self._residual.jacobian(fsol_nm1)
        return -(mass + 0.5 * deltat * jac)

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
    #
    # For Crank-Nicolson, the residual is:
    #   R(y_n) = y_n - y_{n-1} - (Δt/2)·[f(y_{n-1}, t_{n-1}) + f(y_n, t_n)]
    #
    # Key insight: y_{n-1} is FIXED when computing derivatives w.r.t. y_n.
    # The term f(y_{n-1}) does NOT contribute to d²R/dy_n² since it's independent
    # of the current state y_n.
    #
    # However, when computing d²R/dp², BOTH terms contribute since both
    # f(y_{n-1}, p) and f(y_n, p) depend on parameters p.
    # =========================================================================

    def _setup_derivative_methods(self) -> None:
        """
        Override base class to add Crank-Nicolson specific prev_* methods.

        In addition to the standard HVP methods (state_state_hvp, etc.),
        Crank-Nicolson needs prev_* methods because R_{n+1} depends on y_n
        through the f(y_{n-1}) term.
        """
        # Call parent implementation for standard derivative methods
        super()._setup_derivative_methods()

        # Add Crank-Nicolson specific methods for R_{n+1} contributions
        if hasattr(self._residual, "state_state_hvp"):
            self.prev_state_state_hvp = self._prev_state_state_hvp
            self.prev_state_param_hvp = self._prev_state_param_hvp
            self.prev_param_state_hvp = self._prev_param_state_hvp

    def _state_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """
        Compute (d²R/dy_n²)·w contracted with adjoint.

        For Crank-Nicolson: d²R/dy_n² = -(Δt/2)·(d²f/dy²)|_{y_n, t_n}

        Note: The f(y_{n-1}) term doesn't contribute because y_{n-1} is fixed.
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
        Compute (d²R/dy_n dp)·v contracted with adjoint.

        For Crank-Nicolson:
        dR/dp = -(Δt/2)·[df/dp|_{y_{n-1}} + df/dp|_{y_n}]

        d²R/(dy_n dp) = -(Δt/2)·d²f/(dy dp)|_{y_n, t_n}

        Note: The f(y_{n-1}) term doesn't contribute to d²R/(dy_n dp)
        because y_{n-1} is independent of y_n.
        """
        # Only contribution from y_n term (y_{n-1} is fixed w.r.t. y_n)
        self._residual.set_time(self._time + self._deltat)
        hvp_n = self._residual.state_param_hvp(fsol_n, adj_state, vvec)

        return -0.5 * self._deltat * hvp_n

    def _param_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """
        Compute (d²R/dp dy_n)·w contracted with adjoint.

        For Crank-Nicolson:
        dR/dy_n = I - (Δt/2)·df/dy|_{y_n}

        d²R/(dp dy_n) = -(Δt/2)·d²f/(dp dy)|_{y_n, t_n}

        Note: The f(y_{n-1}) term doesn't contribute because y_{n-1} is fixed.
        """
        # Only contribution from y_n term (y_{n-1} is fixed w.r.t. y_n)
        self._residual.set_time(self._time + self._deltat)
        hvp_n = self._residual.param_state_hvp(fsol_n, adj_state, wvec)

        return -0.5 * self._deltat * hvp_n

    def _param_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """
        Compute (d²R/dp²)·v contracted with adjoint.

        For Crank-Nicolson:
        dR/dp = -(Δt/2)·[df/dp|_{y_{n-1}} + df/dp|_{y_n}]

        d²R/dp² = -(Δt/2)·[d²f/dp²|_{y_{n-1}} + d²f/dp²|_{y_n}]

        Note: BOTH terms contribute because both f(y_{n-1}, p) and f(y_n, p)
        depend on parameters p.
        """
        # Contribution from y_{n-1} term
        self._residual.set_time(self._time)
        hvp_nm1 = self._residual.param_param_hvp(fsol_nm1, adj_state, vvec)

        # Contribution from y_n term
        self._residual.set_time(self._time + self._deltat)
        hvp_n = self._residual.param_param_hvp(fsol_n, adj_state, vvec)

        return -0.5 * self._deltat * (hvp_nm1 + hvp_n)

    # =========================================================================
    # Off-diagonal HVP Methods for Crank-Nicolson
    # =========================================================================
    #
    # For Crank-Nicolson, R_n depends on BOTH y_{n-1} and y_n.
    # When computing ∂²L/∂y_n², we need contributions from:
    # - R_n (via f(y_n)) - computed by state_state_hvp above
    # - R_{n+1} (via f(y_n) where y_n is y_{n-1} for R_{n+1})
    #
    # These "prev" methods compute the R_{n+1} contribution evaluated at y_n.
    # =========================================================================

    def _prev_state_state_hvp(
        self,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """
        Compute (∂²R_{n+1}/∂y_n²)·w contracted with adjoint.

        For Crank-Nicolson R_{n+1}, the y_n term is f(y_n) (the y_{n-1} of R_{n+1}):
        ∂²R_{n+1}/∂y_n² = -(Δt/2)·∂²f/∂y²|_{y_n}

        Parameters
        ----------
        fsol_n : Array
            Solution at time n (acts as y_{n-1} for R_{n+1}). Shape: (nstates,)
        adj_state : Array
            Adjoint at time n+1 (λ_{n+1}). Shape: (nstates,)
        wvec : Array
            Sensitivity at y_n. Shape: (nstates,)

        Returns
        -------
        Array
            Hessian contribution. Shape: (nstates,)
        """
        # Time is set for t_n (because we're evaluating f at y_n)
        return -0.5 * self._deltat * self._residual.state_state_hvp(
            fsol_n, adj_state, wvec
        )

    def _prev_state_param_hvp(
        self,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """
        Compute (∂²R_{n+1}/∂y_n ∂p)·v contracted with adjoint.

        For Crank-Nicolson R_{n+1}, the y_n term is f(y_n, p):
        ∂²R_{n+1}/∂y_n ∂p = -(Δt/2)·∂²f/∂y∂p|_{y_n}

        Parameters
        ----------
        fsol_n : Array
            Solution at time n. Shape: (nstates,)
        adj_state : Array
            Adjoint at time n+1 (λ_{n+1}). Shape: (nstates,)
        vvec : Array
            Direction vector. Shape: (nparams, 1)

        Returns
        -------
        Array
            Mixed Hessian contribution. Shape: (nstates,)
        """
        return -0.5 * self._deltat * self._residual.state_param_hvp(
            fsol_n, adj_state, vvec
        )

    def _prev_param_state_hvp(
        self,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """
        Compute (∂²R_{n+1}/∂p ∂y_n)·w contracted with adjoint.

        Parameters
        ----------
        fsol_n : Array
            Solution at time n. Shape: (nstates,)
        adj_state : Array
            Adjoint at time n+1 (λ_{n+1}). Shape: (nstates,)
        wvec : Array
            Sensitivity at y_n. Shape: (nstates,)

        Returns
        -------
        Array
            Mixed Hessian contribution. Shape: (nparams,)
        """
        return -0.5 * self._deltat * self._residual.param_state_hvp(
            fsol_n, adj_state, wvec
        )
