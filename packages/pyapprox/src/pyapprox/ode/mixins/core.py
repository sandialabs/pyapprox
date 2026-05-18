"""Core time stepper mixin providing shared state and initialization."""

from abc import ABC, abstractmethod
from typing import Any, Generic

from pyapprox.ode.protocols.ode_residual import ODEResidualProtocol
from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Array, Backend


class CoreStepperMixin(ABC, Generic[Array]):
    """Core mixin providing __init__, bind, bkd, native_residual.

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

    def bind(self, ctx: StepContext[Array]) -> None:
        """Bind the step context for Newton-facing methods."""
        self._ctx = ctx

    @abstractmethod
    def __call__(self, state: Array) -> Array:
        """Evaluate the time stepping residual."""
        ...

    @abstractmethod
    def jacobian(self, state: Array) -> Array:
        """Compute the Jacobian dR/dy_n."""
        ...

    def __repr__(self) -> str:
        ctx = getattr(self, '_ctx', None)
        return (
            f"{self.__class__.__name__}(\n"
            f"  residual={type(self._residual).__name__},\n"
            f"  backend={type(self._bkd).__name__},\n"
            f"  t_prev={ctx.t_prev if ctx else None},\n"
            f"  deltat={ctx.deltat if ctx else None},\n"
            ")"
        )
