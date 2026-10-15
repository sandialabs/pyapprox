"""Implicit stepper mixin providing linsolve via newton_jacobian."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

from pyapprox.ode.linear_operator import LinearOperatorProtocol
from pyapprox.ode.protocols.ode_residual import (
    ImplicitODEResidualProtocol,
    ODEResidualProtocol,
)
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
