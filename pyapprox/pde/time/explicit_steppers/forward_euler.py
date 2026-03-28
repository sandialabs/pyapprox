"""
Forward Euler time stepping residual with adjoint support.

The Forward Euler method is a first-order explicit time integrator:

    M·(y_n - y_{n-1}) - Δt·f(y_{n-1}, t_{n-1}) = 0

This module provides full adjoint support for gradient computation dQ/dp
via the adjoint method.
"""

from pyapprox.pde.time.protocols import (
    TimeSteppingResidualBase,
)
from pyapprox.util.backends.protocols import Array


class ForwardEulerResidual(TimeSteppingResidualBase[Array]):
    """
    Forward Euler time stepping residual.

    Residual: R(y_n) = M·(y_n - y_{n-1}) - Δt·f(y_{n-1}, t_{n-1}) = 0

    This is a first-order explicit method. The residual is linear in y_n
    and can be solved without Newton iteration.

    Note: For explicit methods, the "residual" is actually just the update
    step. The __call__ method returns Δt·f(y_{n-1}) for consistency with
    the integrator interface.

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
        Evaluate the Forward Euler residual.

        R(y_n) = M·(y_n - y_{n-1}) - Δt·f(y_{n-1}, t_{n-1})

        Parameters
        ----------
        state : Array
            State at current time step y_n. Shape: (nstates,)

        Returns
        -------
        Array
            Residual R(y_n). Shape: (nstates,)
        """
        self._residual.set_time(self._time)
        return self._residual.apply_mass_matrix(
            state - self._prev_state
        ) - self._deltat * self._residual(self._prev_state)

    def jacobian(self, state: Array) -> Array:
        """
        Compute the Jacobian dR/dy_n.

        For Forward Euler, this is just the mass matrix M since the
        residual R = y_n - y_{n-1} - Δt·f(y_{n-1}) is linear in y_n.

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
    # Sensitivity Protocol Methods
    # =========================================================================

    def is_explicit(self) -> bool:
        """Return True since Forward Euler is an explicit scheme."""
        return True

    def has_prev_state_hessian(self) -> bool:
        """Return False since R_{n+1} does not depend on f(y_n)."""
        return False

    def sensitivity_off_diag_jacobian(
        self, fsol_nm1: Array, fsol_n: Array, deltat: float
    ) -> Array:
        """
        Compute dR_n/dy_{n-1} for forward sensitivity propagation.

        For Forward Euler R_n = y_n - y_{n-1} - Δt·f(y_{n-1}):
            dR_n/dy_{n-1} = -M - Δt·J_{n-1}

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
        return -mass - deltat * jac

    # =========================================================================
    # Adjoint Methods
    # =========================================================================

    def _param_jacobian(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        """
        Compute the parameter Jacobian dR/dp for one time step.

        dR/dp = -Δt·(df/dp)|_{y_{n-1}, t_{n-1}}

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
        return -self._deltat * self._residual.param_jacobian(fsol_nm1)

    def adjoint_diag_jacobian(self, fsol_n: Array) -> Array:
        """
        Compute the diagonal Jacobian block for adjoint solve.

        For Forward Euler (explicit), this is just Mᵀ.

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

    def adjoint_off_diag_jacobian(self, fsol_n: Array, deltat_np1: float) -> Array:
        """
        Compute the off-diagonal Jacobian for adjoint coupling.

        For Forward Euler:
        dR_{n+1}/dy_n = -(M + Δt_{n+1}·J_n)

        So the transpose is: -(M + Δt_{n+1}·J_n)ᵀ

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
            deltat_np1 * self._residual.jacobian(fsol_n)
            + self._residual.mass_matrix(fsol_n.shape[0])
        ).T

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

    def _get_quadrature_class(self) -> type:
        """Return quadrature class for Forward Euler (left-constant)."""
        from pyapprox.surrogates.affine.univariate.piecewisepoly import (
            PiecewiseConstantLeft,
        )

        return PiecewiseConstantLeft

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

        For Forward Euler: d²R/dy² = -Δt·(d²f/dy²)
        Note: Depends on y_{n-1}, not y_n.
        """
        self._residual.set_time(self._time)
        return -self._deltat * self._residual.state_state_hvp(fsol_nm1, adj_state, wvec)

    def _state_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """
        Compute (d²R/dy_{n-1} dp)·v contracted with adjoint.

        For Forward Euler: d²R/dydp = -Δt·(d²f/dydp)
        """
        self._residual.set_time(self._time)
        return -self._deltat * self._residual.state_param_hvp(fsol_nm1, adj_state, vvec)

    def _param_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """
        Compute (d²R/dp dy_{n-1})·w contracted with adjoint.

        For Forward Euler: d²R/dpdy = -Δt·(d²f/dpdy)
        """
        self._residual.set_time(self._time)
        return -self._deltat * self._residual.param_state_hvp(fsol_nm1, adj_state, wvec)

    def _param_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """
        Compute (d²R/dp²)·v contracted with adjoint.

        For Forward Euler: d²R/dp² = -Δt·(d²f/dp²)
        """
        self._residual.set_time(self._time)
        return -self._deltat * self._residual.param_param_hvp(fsol_nm1, adj_state, vvec)
