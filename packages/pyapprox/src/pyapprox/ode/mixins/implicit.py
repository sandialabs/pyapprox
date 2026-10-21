"""Implicit stepper mixin providing jacobian and linsolve from the
stepper's Newton operator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

from pyapprox.ode.linear_operator import LinearOperatorProtocol
from pyapprox.ode.protocols.ode_residual import ODEResidualProtocol
from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Array, Backend


class ImplicitStepperMixin(ABC, Generic[Array]):
    """Mixin for implicit steppers (backward Euler, Crank-Nicolson,
    implicit midpoint).

    Provides linsolve via the stepper's Newton operator. Each stepper
    states its linearization once, in ``_newton_operator``: the state
    and time at which ``f`` is linearized and the coefficient (backward
    Euler: ``y_n``, ``t_n``, ``dt``; Crank-Nicolson: ``y_n``, ``t_n``,
    ``dt/2``; implicit midpoint: the midpoint state and time, ``dt/2``),
    so ``jacobian`` and ``linsolve`` cannot disagree.
    """

    if TYPE_CHECKING:
        _bkd: Backend[Array]
        _residual: ODEResidualProtocol[Array]
        _ctx: StepContext[Array]

    @abstractmethod
    def _newton_operator(
        self, ctx: StepContext[Array], state: Array
    ) -> LinearOperatorProtocol[Array]:
        """Return ``dR/dy_n`` of the step ``ctx`` at ``state``.

        The context is explicit so the Newton solve (the bound step)
        and the adjoint (any step) share this one linearization.
        Internal to the stepper hierarchy, not exposed externally.
        """
        ...

    def jacobian(self, state: Array) -> Array:
        """Return ``dR/dy_n`` of the current step as a matrix."""
        return self._newton_operator(self._ctx, state).as_matrix()

    def linsolve(self, state: Array, residual: Array) -> Array:
        """Solve ``(dR/dy_n) dy = residual`` with the step's Newton operator."""
        return self._newton_operator(self._ctx, state).solve(residual)
