"""Implicit stepper mixin providing linsolve via newton_jacobian."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

from pyapprox.ode.protocols.ode_residual import (
    ImplicitODEResidualProtocol,
    ODEResidualProtocol,
)
from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Array, Backend


class ImplicitStepperMixin(ABC, Generic[Array]):
    """Mixin for implicit steppers (backward Euler, Crank-Nicolson).

    Provides linsolve via residual's newton_jacobian operator. The
    stepper-specific coefficient (dt for BE, dt/2 for CN) is defined
    once per stepper in _newton_coefficient.
    """

    if TYPE_CHECKING:
        _bkd: Backend[Array]
        _residual: ODEResidualProtocol[Array]
        _ctx: StepContext[Array]

    @property
    def _implicit_residual(self) -> ImplicitODEResidualProtocol[Array]:
        """Narrow the ODE residual to newton_jacobian capable, or raise.

        Capability narrowing is lazy: constructing an implicit stepper
        over a residual without newton_jacobian is legal until the
        first linsolve/jacobian call reaches this accessor.
        """
        residual = self._residual
        if not isinstance(residual, ImplicitODEResidualProtocol):
            raise TypeError(
                f"{type(self).__name__} requires an ODE residual with "
                "newton_jacobian for implicit stepping; got "
                f"{type(residual).__name__}"
            )
        return residual

    @abstractmethod
    def _newton_coefficient(self) -> float:
        """Stepper-specific coefficient for Newton Jacobian.

        BackwardEuler: self._ctx.deltat. CrankNicolson: 0.5 * self._ctx.deltat.
        Internal to the stepper hierarchy, not exposed externally.
        """
        ...

    def linsolve(self, state: Array, residual: Array) -> Array:
        """Solve (M - coeff*J) dy = residual via residual's newton_jacobian."""
        op = self._implicit_residual.newton_jacobian(
            state, self._newton_coefficient()
        )
        return op.solve(residual)
