"""
Unified protocols for time integration with adjoint and HVP support.

This module defines the protocol hierarchy for ODE residuals and time stepping
residuals used in time integration with sensitivity computation.

Protocol Hierarchy
------------------
ODE Residuals (user-defined):
    ODEResidualProtocol
        └── ODEResidualWithParamJacobianProtocol
                └── ODEResidualWithHVPProtocol

Time Stepping Residuals (framework):
    TimeSteppingResidualProtocol
        └── AdjointEnabledTimeSteppingResidualProtocol
                └── HVPEnabledTimeSteppingResidualProtocol

Time Handling
-------------
Time is incorporated via stateful `set_time()` methods rather than explicit
arguments. This keeps the Newton solver generic (just solves R(y)=0).

- ODE residual: `set_time(time)` sets current time
- Time stepping residual: `set_time(time, deltat, prev_state)` sets context
- Integrator calls set_time before each evaluation
"""

from typing import Protocol, Generic, runtime_checkable, Tuple
from abc import ABC, abstractmethod

from pyapprox.typing.util.backends.protocols import Array, Backend


# =============================================================================
# ODE Residual Protocols (User-Defined)
# =============================================================================


@runtime_checkable
class ODEResidualProtocol(Protocol, Generic[Array]):
    """
    Base protocol for ODE residuals: dy/dt = f(y, t; p).

    This is the user-defined ODE system. The residual evaluates f(y, t).

    Methods
    -------
    bkd()
        Get the computational backend.
    __call__(state)
        Evaluate f(y, t) at current time (set via set_time).
    set_time(time)
        Set the current time for evaluation.
    jacobian(state)
        Compute df/dy at current state and time.
    mass_matrix(nstates)
        Return the mass matrix M (identity for standard ODEs).
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def __call__(self, state: Array) -> Array:
        """
        Evaluate the ODE residual f(y, t) at current time.

        Parameters
        ----------
        state : Array
            Current state y. Shape: (nstates,)

        Returns
        -------
        Array
            Residual f(y, t). Shape: (nstates,)
        """
        ...

    def set_time(self, time: float) -> None:
        """
        Set the current time for evaluation.

        Parameters
        ----------
        time : float
            Current time.
        """
        ...

    def jacobian(self, state: Array) -> Array:
        """
        Compute the Jacobian df/dy at current state and time.

        Parameters
        ----------
        state : Array
            Current state y. Shape: (nstates,)

        Returns
        -------
        Array
            Jacobian df/dy. Shape: (nstates, nstates)
        """
        ...

    def mass_matrix(self, nstates: int) -> Array:
        """
        Return the mass matrix M.

        For standard ODEs, this is the identity matrix.
        For DAEs, this may be singular.

        Parameters
        ----------
        nstates : int
            Number of states.

        Returns
        -------
        Array
            Mass matrix. Shape: (nstates, nstates)
        """
        ...


@runtime_checkable
class ODEResidualWithParamJacobianProtocol(Protocol, Generic[Array]):
    """
    ODE residual with parameter sensitivity: dy/dt = f(y, t; p).

    Extends ODEResidualProtocol with parameter Jacobian for adjoint
    sensitivity computations.

    Additional Methods
    ------------------
    nparams()
        Number of parameters.
    set_param(param)
        Set the parameter values.
    param_jacobian(state)
        Compute df/dp at current state and time.
    initial_param_jacobian()
        Jacobian of initial condition with respect to parameters.
    """

    def bkd(self) -> Backend[Array]:
        ...

    def __call__(self, state: Array) -> Array:
        ...

    def set_time(self, time: float) -> None:
        ...

    def jacobian(self, state: Array) -> Array:
        ...

    def mass_matrix(self, nstates: int) -> Array:
        ...

    def nparams(self) -> int:
        """
        Get the number of parameters.

        Returns
        -------
        int
            Number of parameters.
        """
        ...

    def set_param(self, param: Array) -> None:
        """
        Set the parameter values.

        Parameters
        ----------
        param : Array
            Parameter values. Shape: (nparams,)
        """
        ...

    def param_jacobian(self, state: Array) -> Array:
        """
        Compute the Jacobian df/dp at current state and time.

        Parameters
        ----------
        state : Array
            Current state y. Shape: (nstates,)

        Returns
        -------
        Array
            Parameter Jacobian df/dp. Shape: (nstates, nparams)
        """
        ...

    def initial_param_jacobian(self) -> Array:
        """
        Jacobian of initial condition with respect to parameters.

        For y(0) = g(p), this returns dg/dp.
        Returns zeros if initial condition doesn't depend on parameters.

        Returns
        -------
        Array
            dy0/dp. Shape: (nstates, nparams)
        """
        ...


@runtime_checkable
class ODEResidualWithHVPProtocol(Protocol, Generic[Array]):
    """
    ODE residual with HVP support for second-order adjoint computations.

    Extends ODEResidualWithParamJacobianProtocol with four HVP methods
    for computing Hessian-vector products via the adjoint method.

    The HVP methods compute products of second derivatives with vectors,
    contracted with the adjoint state λ for efficiency.

    Additional Methods
    ------------------
    state_state_hvp(state, adj_state, wvec)
        (d²f/dy²)·w contracted with λ
    state_param_hvp(state, adj_state, vvec)
        (d²f/dydp)·v contracted with λ
    param_state_hvp(state, adj_state, wvec)
        (d²f/dpdy)·w contracted with λ
    param_param_hvp(state, adj_state, vvec)
        (d²f/dp²)·v contracted with λ
    """

    def bkd(self) -> Backend[Array]:
        ...

    def __call__(self, state: Array) -> Array:
        ...

    def set_time(self, time: float) -> None:
        ...

    def jacobian(self, state: Array) -> Array:
        ...

    def mass_matrix(self, nstates: int) -> Array:
        ...

    def nparams(self) -> int:
        ...

    def set_param(self, param: Array) -> None:
        ...

    def param_jacobian(self, state: Array) -> Array:
        ...

    def initial_param_jacobian(self) -> Array:
        ...

    def state_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """
        Compute (d²f/dy²)·w contracted with adjoint state.

        Parameters
        ----------
        state : Array
            Current state y. Shape: (nstates,)
        adj_state : Array
            Adjoint state λ. Shape: (nstates,)
        wvec : Array
            Direction vector w. Shape: (nstates,)

        Returns
        -------
        Array
            λᵀ·(d²f/dy²)·w. Shape: (nstates,)
        """
        ...

    def state_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """
        Compute (d²f/dydp)·v contracted with adjoint state.

        Parameters
        ----------
        state : Array
            Current state y. Shape: (nstates,)
        adj_state : Array
            Adjoint state λ. Shape: (nstates,)
        vvec : Array
            Direction vector v. Shape: (nparams,)

        Returns
        -------
        Array
            λᵀ·(d²f/dydp)·v. Shape: (nstates,)
        """
        ...

    def param_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """
        Compute (d²f/dpdy)·w contracted with adjoint state.

        Parameters
        ----------
        state : Array
            Current state y. Shape: (nstates,)
        adj_state : Array
            Adjoint state λ. Shape: (nstates,)
        wvec : Array
            Direction vector w. Shape: (nstates,)

        Returns
        -------
        Array
            λᵀ·(d²f/dpdy)·w. Shape: (nparams,)
        """
        ...

    def param_param_hvp(
        self, state: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """
        Compute (d²f/dp²)·v contracted with adjoint state.

        Parameters
        ----------
        state : Array
            Current state y. Shape: (nstates,)
        adj_state : Array
            Adjoint state λ. Shape: (nstates,)
        vvec : Array
            Direction vector v. Shape: (nparams,)

        Returns
        -------
        Array
            λᵀ·(d²f/dp²)·v. Shape: (nparams,)
        """
        ...


# =============================================================================
# Time Stepping Residual Protocols (Framework)
# =============================================================================


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
        Solve the linear system J·δy = residual.

        Parameters
        ----------
        state : Array
            State at current time step y_n. Shape: (nstates,)
        residual : Array
            Right-hand side. Shape: (nstates,)

        Returns
        -------
        Array
            Solution δy. Shape: (nstates,)
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
        Transpose of Jacobian block (dR/dy_n)ᵀ.
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

        This is the transpose (dR/dy_n)ᵀ evaluated at time t_n.

        Parameters
        ----------
        fsol_n : Array
            Forward solution at current time step. Shape: (nstates,)

        Returns
        -------
        Array
            (dR/dy_n)ᵀ. Shape: (nstates, nstates)
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

        At final time T, solve (dR/dy_N)ᵀ·λ_N = -dQ/dy_N.

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
            Adjoint solution at initial time λ_0. Shape: (nstates,)
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
        Compute (d²R/dy_n²)·w contracted with adjoint.

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
        Compute (d²R/dy_n dp)·v contracted with adjoint.

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
        Compute (d²R/dp dy_n)·w contracted with adjoint.

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
        Compute (d²R/dp²)·v contracted with adjoint.

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


# =============================================================================
# Base Classes
# =============================================================================


class TimeSteppingResidualBase(ABC, Generic[Array]):
    """
    Abstract base class for time stepping residuals.

    Provides common functionality for explicit and implicit time steppers.
    Uses dynamic method binding pattern for optional derivative methods.

    Subclasses must implement:
    - __call__(state)
    - jacobian(state) [for implicit methods]
    """

    def __init__(self, residual: ODEResidualProtocol[Array]):
        """
        Initialize time stepping residual.

        Parameters
        ----------
        residual : ODEResidualProtocol[Array]
            The underlying ODE residual.
        """
        self._residual = residual
        self._bkd = residual.bkd()
        self._setup_derivative_methods()

    @property
    def native_residual(self) -> ODEResidualProtocol[Array]:
        """Get the underlying ODE residual."""
        return self._residual

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def set_time(self, time: float, deltat: float, prev_state: Array) -> None:
        """
        Set the time stepping context.

        Parameters
        ----------
        time : float
            Time at start of step.
        deltat : float
            Time step size.
        prev_state : Array
            State at previous time step.
        """
        self._time = time
        self._deltat = deltat
        self._prev_state = prev_state

    @abstractmethod
    def __call__(self, state: Array) -> Array:
        """Evaluate the time stepping residual."""
        raise NotImplementedError

    def linsolve(self, state: Array, residual: Array) -> Array:
        """Solve the linear system J·δy = residual."""
        return self._bkd.solve(self.jacobian(state), residual)

    def _setup_derivative_methods(self) -> None:
        """
        Conditionally expose derivative methods based on residual capabilities.

        This follows the dynamic binding pattern used throughout pyapprox.typing.
        Methods are only exposed if the underlying ODE residual supports them.
        """
        # Adjoint methods depend on param_jacobian
        if hasattr(self._residual, "param_jacobian"):
            self.param_jacobian = self._param_jacobian

        # HVP methods depend on underlying HVP capability
        if hasattr(self._residual, "state_state_hvp"):
            self.state_state_hvp = self._state_state_hvp
            self.state_param_hvp = self._state_param_hvp
            self.param_state_hvp = self._param_state_hvp
            self.param_param_hvp = self._param_param_hvp

    def _param_jacobian(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        """Default implementation - subclasses should override."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _param_jacobian"
        )

    def _state_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """Default implementation - subclasses should override."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _state_state_hvp"
        )

    def _state_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """Default implementation - subclasses should override."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _state_param_hvp"
        )

    def _param_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """Default implementation - subclasses should override."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _param_state_hvp"
        )

    def _param_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """Default implementation - subclasses should override."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _param_param_hvp"
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  residual={type(self._residual).__name__},\n"
            f"  backend={type(self._bkd).__name__},\n"
            f"  time={getattr(self, '_time', None)},\n"
            f"  deltat={getattr(self, '_deltat', None)},\n"
            ")"
        )
