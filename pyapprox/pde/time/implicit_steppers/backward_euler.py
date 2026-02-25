"""
Backward Euler time stepping residual with adjoint support.

The Backward Euler method is a first-order implicit time integrator:

    M·(y_n - y_{n-1}) - Δt·f(y_n, t_n) = 0

This module provides full adjoint support for gradient computation dQ/dp
via the adjoint method.
"""

from typing import Tuple

from pyapprox.util.backends.protocols import Array
from pyapprox.pde.sparse_utils import solve_maybe_sparse
from pyapprox.pde.time.protocols import (
    TimeSteppingResidualBase,
    ODEResidualProtocol,
)


class BackwardEulerResidual(TimeSteppingResidualBase[Array]):
    """
    Backward Euler time stepping residual.

    Residual: R(y_n) = M·(y_n - y_{n-1}) - Δt·f(y_n, t_n) = 0

    This is a first-order implicit method (A-stable).

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
        Evaluate the Backward Euler residual.

        R(y_n) = M·(y_n - y_{n-1}) - Δt·f(y_n, t_n)

        Parameters
        ----------
        state : Array
            State at current time step y_n. Shape: (nstates,)

        Returns
        -------
        Array
            Residual value. Shape: (nstates,)
        """
        self._residual.set_time(self._time + self._deltat)
        return (
            self._residual.apply_mass_matrix(state - self._prev_state)
            - self._deltat * self._residual(state)
        )

    def jacobian(self, state: Array) -> Array:
        """
        Compute the Jacobian dR/dy_n.

        dR/dy_n = M - Δt·(df/dy)

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
        return self._residual.mass_matrix(
            state.shape[0]
        ) - self._deltat * self._residual.jacobian(state)

    # =========================================================================
    # Sensitivity Protocol Methods
    # =========================================================================

    def is_explicit(self) -> bool:
        """Return False since Backward Euler is an implicit scheme."""
        return False

    def has_prev_state_hessian(self) -> bool:
        """Return False since R_{n+1} does not depend on f(y_n)."""
        return False

    def sensitivity_off_diag_jacobian(
        self, fsol_nm1: Array, fsol_n: Array, deltat: float
    ) -> Array:
        """
        Compute dR_n/dy_{n-1} for forward sensitivity propagation.

        For Backward Euler R_n = y_n - y_{n-1} - Δt·f(y_n):
            dR_n/dy_{n-1} = -M

        The f(y_n) term does not depend on y_{n-1}.

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
        return -self._residual.mass_matrix(fsol_nm1.shape[0])

    # =========================================================================
    # Adjoint Methods
    # =========================================================================

    def _param_jacobian(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        """
        Compute the parameter Jacobian dR/dp for one time step.

        dR/dp = -Δt·(df/dp)|_{y_n, t_n}

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
        self._residual.set_time(self._time + self._deltat)
        return -self._deltat * self._residual.param_jacobian(fsol_n)

    def adjoint_diag_jacobian(self, fsol_n: Array) -> Array:
        """
        Compute the diagonal Jacobian block for adjoint solve.

        (dR/dy_n)ᵀ = (M - Δt·J)ᵀ

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
            - self._deltat * self._residual.jacobian(fsol_n)
        ).T

    def adjoint_off_diag_jacobian(
        self, fsol_n: Array, deltat_np1: float
    ) -> Array:
        """
        Compute the off-diagonal Jacobian for adjoint coupling.

        For Backward Euler: -Mᵀ (couples λ_n to λ_{n+1})

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
        return -self._residual.mass_matrix(fsol_n.shape[0]).T

    def adjoint_initial_condition(
        self, final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        """
        Compute initial condition for backward adjoint solve.

        At final time T, solve: (dR/dy_N)ᵀ·λ_N = -dQ/dy_N

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
        return solve_maybe_sparse(self._bkd, drdu.T, -final_dqdu)

    def _get_quadrature_class(self):
        """Return quadrature class for Backward Euler (right-constant)."""
        from pyapprox.surrogates.affine.univariate.piecewisepoly import (
            PiecewiseConstantRight,
        )
        return PiecewiseConstantRight

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
        Compute (d²R/dy_n²)·w contracted with adjoint.

        For Backward Euler: d²R/dy² = -Δt·(d²f/dy²)
        """
        self._residual.set_time(self._time + self._deltat)
        return -self._deltat * self._residual.state_state_hvp(
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

        For Backward Euler: d²R/dydp = -Δt·(d²f/dydp)
        """
        self._residual.set_time(self._time + self._deltat)
        return -self._deltat * self._residual.state_param_hvp(
            fsol_n, adj_state, vvec
        )

    def _param_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """
        Compute (d²R/dp dy_n)·w contracted with adjoint.

        For Backward Euler: d²R/dpdy = -Δt·(d²f/dpdy)
        """
        self._residual.set_time(self._time + self._deltat)
        return -self._deltat * self._residual.param_state_hvp(
            fsol_n, adj_state, wvec
        )

    def _param_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """
        Compute (d²R/dp²)·v contracted with adjoint.

        For Backward Euler: d²R/dp² = -Δt·(d²f/dp²)
        """
        self._residual.set_time(self._time + self._deltat)
        return -self._deltat * self._residual.param_param_hvp(
            fsol_n, adj_state, vvec
        )
