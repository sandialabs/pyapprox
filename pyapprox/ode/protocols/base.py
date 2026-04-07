"""
Base class for time stepping residuals.

Provides shared implementation for all time steppers (Forward Euler,
Backward Euler, Crank-Nicolson, Heun, etc.).
"""

from abc import ABC, abstractmethod
from typing import Generic, Tuple, cast

from pyapprox.util.linalg.sparse_dispatch import solve_maybe_sparse
from pyapprox.util.backends.protocols import Array, Backend

from .ode_residual import (
    ODEResidualProtocol,
    ODEResidualWithHVPProtocol,
    ODEResidualWithParamJacobianProtocol,
)


class ParamJacobianCapableMixin(Generic[Array]):
    """Mixin for time steppers whose ODE residual supports param_jacobian.

    Provides ``_param_residual`` for typed access to the narrowed residual.
    Only mix in when ``_setup_derivative_methods`` has confirmed capability.
    """

    _residual: ODEResidualProtocol[Array]

    def _param_residual(self) -> ODEResidualWithParamJacobianProtocol[Array]:
        return cast(ODEResidualWithParamJacobianProtocol[Array], self._residual)


class HVPCapableMixin(Generic[Array]):
    """Mixin for time steppers whose ODE residual supports HVP methods.

    Provides ``_hvp_residual`` for typed access to the narrowed residual.
    Only mix in when ``_setup_derivative_methods`` has confirmed capability.
    """

    _residual: ODEResidualProtocol[Array]

    def _hvp_residual(self) -> ODEResidualWithHVPProtocol[Array]:
        return cast(ODEResidualWithHVPProtocol[Array], self._residual)


class TimeSteppingResidualBase(ABC, Generic[Array]):
    """
    Abstract base class for time stepping residuals.

    Provides common functionality for explicit and implicit time steppers.
    Uses dynamic method binding pattern for optional derivative methods.

    Subclasses must implement:
    - __call__(state)
    - jacobian(state) [for implicit methods]
    - _get_quadrature_class()
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

    @abstractmethod
    def jacobian(self, state: Array) -> Array:
        """Compute the Jacobian of the time stepping residual."""
        raise NotImplementedError

    def linsolve(self, state: Array, residual: Array) -> Array:
        """Solve the linear system J dy = residual."""
        return solve_maybe_sparse(self._bkd, self.jacobian(state), residual)

    def _setup_derivative_methods(self) -> None:
        """
        Conditionally expose derivative methods based on residual capabilities.

        This follows the dynamic binding pattern used throughout pyapprox.
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

    # =========================================================================
    # Template Methods (shared implementations for all steppers)
    # =========================================================================

    def quadrature_samples_weights(self, times: Array) -> Tuple[Array, Array]:
        """
        Compute quadrature rule consistent with time discretization.

        Uses template method pattern - subclasses override _get_quadrature_class()
        to specify their quadrature type.

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
        quadrature_class = self._get_quadrature_class()
        quadrature = quadrature_class(times, self._bkd)
        quadx, quadw = quadrature.quadrature_rule()
        # Flatten weights if needed
        if quadw.ndim > 1:
            quadw = quadw[:, 0]
        return quadx, quadw

    @abstractmethod
    def _get_quadrature_class(self) -> type:
        """
        Return the quadrature class for this time stepper.

        Override in subclass to specify the quadrature type.

        Returns
        -------
        type
            Quadrature class (e.g., PiecewiseConstantLeft, PiecewiseLinear)
        """
        raise NotImplementedError

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
