"""Implicit stepper mixin providing linsolve via newton_jacobian."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

from pyapprox.ode.protocols.ode_residual import (
    ImplicitODEResidualProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class ImplicitStepperMixin(ABC, Generic[Array]):
    """Mixin for implicit steppers (backward Euler, Crank-Nicolson).

    Provides linsolve via residual's newton_jacobian operator. The
    stepper-specific coefficient (dt for BE, dt/2 for CN) is defined
    once per stepper in _newton_coefficient.
    """

    if TYPE_CHECKING:
        _bkd: Backend[Array]
        _residual: ImplicitODEResidualProtocol[Array]
        _deltat: float
        _time: float

    @abstractmethod
    def _newton_coefficient(self) -> float:
        """Stepper-specific coefficient for Newton Jacobian.

        BackwardEuler: self._deltat. CrankNicolson: 0.5 * self._deltat.
        Internal to the stepper hierarchy, not exposed externally.
        """
        ...

    def linsolve(self, state: Array, residual: Array) -> Array:
        """Solve (M - coeff*J) dy = residual via residual's newton_jacobian."""
        op = self._residual.newton_jacobian(state, self._newton_coefficient())
        return op.solve(residual)
