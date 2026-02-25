"""
Protocols for time stepping residuals.

These protocols define the interface for time discretization schemes
(Forward Euler, Backward Euler, Crank-Nicolson, etc.) at different
capability levels.

Protocol Hierarchy
------------------
TimeSteppingResidualProtocol
    Basic time step R(y_n) = 0.
AdjointEnabledTimeSteppingResidualProtocol
    Adds adjoint methods for gradient computation.
HVPEnabledTimeSteppingResidualProtocol
    Adds HVP methods for Hessian-vector products.
"""

from typing import Protocol, Generic, runtime_checkable, Tuple

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class TimeSteppingResidualProtocol(Protocol, Generic[Array]):
    """
    Protocol for discretized time step residuals.

    This is the framework-level protocol for time stepping schemes like
    Forward Euler, Backward Euler, Crank-Nicolson, etc.

    The residual R(y_n) = 0 represents one time step from y_{n-1} to y_n.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def set_time(self, time: float, deltat: float, prev_state: Array) -> None:
        """
        Set the time stepping context.

        This must be called before __call__ or jacobian.

        Parameters
        ----------
        time : float
            Time at start of step (t_{n-1}).
        deltat : float
            Time step size (t_n - t_{n-1}).
        prev_state : Array
            State at previous time step y_{n-1}. Shape: (nstates,)
        """
        ...

    def __call__(self, state: Array) -> Array:
        """
        Evaluate the time stepping residual R(y_n).

        Parameters
        ----------
        state : Array
            State at current time step y_n. Shape: (nstates,)

        Returns
        -------
        Array
            Residual R(y_n). Shape: (nstates,)
        """
        ...

    def jacobian(self, state: Array) -> Array:
        """
        Compute the Jacobian dR/dy_n.

        Parameters
        ----------
        state : Array
            State at current time step y_n. Shape: (nstates,)

        Returns
        -------
        Array
            Jacobian dR/dy_n. Shape: (nstates, nstates)
        """
        ...

    def linsolve(self, state: Array, residual: Array) -> Array:
        """
        Solve the linear system J dy = residual.

        Parameters
        ----------
        state : Array
            State at current time step y_n. Shape: (nstates,)
        residual : Array
            Right-hand side. Shape: (nstates,)

        Returns
        -------
        Array
            Solution dy. Shape: (nstates,)
        """
        ...


@runtime_checkable
class AdjointEnabledTimeSteppingResidualProtocol(Protocol, Generic[Array]):
    """
    Time stepping residual with adjoint capability for gradient computation.

    Extends TimeSteppingResidualProtocol with methods required for the
    adjoint method to compute dQ/dp.

    Methods for Adjoint Computation
    -------------------------------
    param_jacobian(fsol_nm1, fsol_n)
        Parameter Jacobian dR/dp for one time step.
    adjoint_diag_jacobian(fsol_n)
        Transpose of Jacobian block (dR/dy_n)^T.
    adjoint_off_diag_jacobian(fsol_n, deltat_np1)
        Off-diagonal coupling term.
    adjoint_initial_condition(final_fwd_sol, final_dqdu)
        Initial condition for backward adjoint solve.
    adjoint_final_solution(fsol_0, asol_1, dqdu_0, deltat_1)
        Final step of backward adjoint solve.
    quadrature_samples_weights(times)
        Quadrature rule consistent with time discretization.
    """

    def bkd(self) -> Backend[Array]:
        ...

    def set_time(self, time: float, deltat: float, prev_state: Array) -> None:
        ...

    def __call__(self, state: Array) -> Array:
        ...

    def jacobian(self, state: Array) -> Array:
        ...

    def linsolve(self, state: Array, residual: Array) -> Array:
        ...

    def param_jacobian(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        """
        Compute the parameter Jacobian dR/dp for one time step.

        Parameters
        ----------
        fsol_nm1 : Array
            Forward solution at previous time step. Shape: (nstates,)
        fsol_n : Array
            Forward solution at current time step. Shape: (nstates,)

        Returns
        -------
        Array
            Parameter Jacobian dR/dp. Shape: (nstates, nparams)
        """
        ...

    def adjoint_diag_jacobian(self, fsol_n: Array) -> Array:
        """
        Compute the diagonal Jacobian block for adjoint solve.

        This is the transpose (dR/dy_n)^T evaluated at time t_n.

        Parameters
        ----------
        fsol_n : Array
            Forward solution at current time step. Shape: (nstates,)

        Returns
        -------
        Array
            (dR/dy_n)^T. Shape: (nstates, nstates)
        """
        ...

    def adjoint_off_diag_jacobian(
        self, fsol_n: Array, deltat_np1: float
    ) -> Array:
        """
        Compute the off-diagonal Jacobian for adjoint coupling.

        This couples the adjoint at time n to time n+1.

        Parameters
        ----------
        fsol_n : Array
            Forward solution at current time step. Shape: (nstates,)
        deltat_np1 : float
            Time step size for the next interval.

        Returns
        -------
        Array
            Off-diagonal coupling term. Shape: (nstates, nstates)
        """
        ...

    def adjoint_initial_condition(
        self, final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        """
        Compute the initial condition for backward adjoint solve.

        At final time T, solve (dR/dy_N)^T lambda_N = -dQ/dy_N.

        Parameters
        ----------
        final_fwd_sol : Array
            Forward solution at final time. Shape: (nstates,)
        final_dqdu : Array
            Gradient dQ/dy at final time. Shape: (nstates,)

        Returns
        -------
        Array
            Adjoint solution at final time lambda_N. Shape: (nstates,)
        """
        ...

    def adjoint_final_solution(
        self,
        fsol_0: Array,
        asol_1: Array,
        dqdu_0: Array,
        deltat_1: float,
    ) -> Array:
        """
        Compute the adjoint at initial time (final step of backward sweep).

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
            Adjoint solution at initial time lambda_0. Shape: (nstates,)
        """
        ...

    def quadrature_samples_weights(self, times: Array) -> Tuple[Array, Array]:
        """
        Compute quadrature rule consistent with time discretization.

        Different time steppers use different quadrature rules:
        - Backward Euler: right-constant
        - Forward Euler: left-constant
        - Crank-Nicolson: trapezoidal

        Parameters
        ----------
        times : Array
            Time points. Shape: (ntimes,)

        Returns
        -------
        quadx : Array
            Quadrature sample points. Shape: (nquad,)
        quadw : Array
            Quadrature weights. Shape: (nquad,)
        """
        ...


@runtime_checkable
class HVPEnabledTimeSteppingResidualProtocol(Protocol, Generic[Array]):
    """
    Time stepping residual with HVP capability for second-order adjoints.

    Extends AdjointEnabledTimeSteppingResidualProtocol with four HVP methods
    for the time stepping residual, enabling Hessian-vector product computation.
    """

    # All AdjointEnabledTimeSteppingResidualProtocol methods
    def bkd(self) -> Backend[Array]:
        ...

    def set_time(self, time: float, deltat: float, prev_state: Array) -> None:
        ...

    def __call__(self, state: Array) -> Array:
        ...

    def jacobian(self, state: Array) -> Array:
        ...

    def linsolve(self, state: Array, residual: Array) -> Array:
        ...

    def param_jacobian(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        ...

    def adjoint_diag_jacobian(self, fsol_n: Array) -> Array:
        ...

    def adjoint_off_diag_jacobian(
        self, fsol_n: Array, deltat_np1: float
    ) -> Array:
        ...

    def adjoint_initial_condition(
        self, final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        ...

    def adjoint_final_solution(
        self,
        fsol_0: Array,
        asol_1: Array,
        dqdu_0: Array,
        deltat_1: float,
    ) -> Array:
        ...

    def quadrature_samples_weights(self, times: Array) -> Tuple[Array, Array]:
        ...

    # HVP methods
    def state_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """
        Compute (d^2R/dy_n^2) w contracted with adjoint.

        Parameters
        ----------
        fsol_nm1 : Array
            Forward solution at previous time step. Shape: (nstates,)
        fsol_n : Array
            Forward solution at current time step. Shape: (nstates,)
        adj_state : Array
            Adjoint state. Shape: (nstates,)
        wvec : Array
            Direction vector. Shape: (nstates,)

        Returns
        -------
        Array
            HVP result. Shape: (nstates,)
        """
        ...

    def state_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """
        Compute (d^2R/dy_n dp) v contracted with adjoint.

        Parameters
        ----------
        fsol_nm1 : Array
            Forward solution at previous time step. Shape: (nstates,)
        fsol_n : Array
            Forward solution at current time step. Shape: (nstates,)
        adj_state : Array
            Adjoint state. Shape: (nstates,)
        vvec : Array
            Direction vector. Shape: (nparams,)

        Returns
        -------
        Array
            HVP result. Shape: (nstates,)
        """
        ...

    def param_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """
        Compute (d^2R/dp dy_n) w contracted with adjoint.

        Parameters
        ----------
        fsol_nm1 : Array
            Forward solution at previous time step. Shape: (nstates,)
        fsol_n : Array
            Forward solution at current time step. Shape: (nstates,)
        adj_state : Array
            Adjoint state. Shape: (nstates,)
        wvec : Array
            Direction vector. Shape: (nstates,)

        Returns
        -------
        Array
            HVP result. Shape: (nparams,)
        """
        ...

    def param_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """
        Compute (d^2R/dp^2) v contracted with adjoint.

        Parameters
        ----------
        fsol_nm1 : Array
            Forward solution at previous time step. Shape: (nstates,)
        fsol_n : Array
            Forward solution at current time step. Shape: (nstates,)
        adj_state : Array
            Adjoint state. Shape: (nstates,)
        vvec : Array
            Direction vector. Shape: (nparams,)

        Returns
        -------
        Array
            HVP result. Shape: (nparams,)
        """
        ...
