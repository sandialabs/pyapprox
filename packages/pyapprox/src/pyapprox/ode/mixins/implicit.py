"""Implicit stepper mixin providing linsolve via newton_jacobian."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

from pyapprox.ode.protocols.ode_residual import (
    ImplicitODEResidualProtocol,
)
from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Array, Backend


class ImplicitStepperMixin(ABC, Generic[Array]):
    """Mixin for implicit steppers (backward Euler, Crank-Nicolson).

    Provides linsolve via residual's newton_jacobian operator. The
    stepper-specific coefficient (dt for BE, dt/2 for CN) is defined
    once per stepper in _newton_coefficient.

    Concrete steppers must declare:
        _residual: ImplicitODEResidualProtocol[Array]
    to narrow the base class annotation and enable newton_jacobian access.
    """

    if TYPE_CHECKING:
        _bkd: Backend[Array]
        _residual: ImplicitODEResidualProtocol[Array]
        _ctx: StepContext[Array]

    @abstractmethod
    def _newton_coefficient(self) -> float:
        """Stepper-specific coefficient for Newton Jacobian.

        BackwardEuler: self._ctx.deltat. CrankNicolson: 0.5 * self._ctx.deltat.
        Internal to the stepper hierarchy, not exposed externally.
        """
        ...

    def linsolve(self, state: Array, residual: Array) -> Array:
        """Solve (M - coeff*J) dy = residual via residual's newton_jacobian."""
        op = self._residual.newton_jacobian(state, self._newton_coefficient())
        return op.solve(residual)
