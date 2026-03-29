"""Core time stepper mixin providing shared state and initialization."""

from abc import ABC, abstractmethod
from typing import Any, Generic

from pyapprox.pde.time.protocols.ode_residual import ODEResidualProtocol
from pyapprox.util.backends.protocols import Array, Backend


class CoreStepperMixin(ABC, Generic[Array]):
    """Core mixin providing __init__, set_time, bkd, native_residual.

    All time steppers must include this mixin (rightmost in MRO).
    """

    _residual: ODEResidualProtocol[Array]

    def __init__(self, residual: ODEResidualProtocol[Array], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._residual = residual
        self._bkd: Backend[Array] = residual.bkd()

    @property
    def native_residual(self) -> ODEResidualProtocol[Array]:
        """Get the underlying ODE residual."""
        return self._residual

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def set_time(self, time: float, deltat: float, prev_state: Array) -> None:
        """Set the time stepping context."""
        self._time = time
        self._deltat = deltat
        self._prev_state = prev_state

    @abstractmethod
    def __call__(self, state: Array) -> Array:
        """Evaluate the time stepping residual."""
        ...

    @abstractmethod
    def jacobian(self, state: Array) -> Array:
        """Compute the Jacobian dR/dy_n."""
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  residual={type(self._residual).__name__},\n"
            f"  backend={type(self._bkd).__name__},\n"
            f"  time={getattr(self, '_time', None)},\n"
            f"  deltat={getattr(self, '_deltat', None)},\n"
            ")"
        )
