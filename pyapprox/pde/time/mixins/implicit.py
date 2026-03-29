"""Implicit stepper mixin providing linsolve."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

from pyapprox.pde.sparse_utils import solve_maybe_sparse
from pyapprox.util.backends.protocols import Array, Backend


class ImplicitStepperMixin(ABC, Generic[Array]):
    """Mixin for implicit steppers (backward Euler, Crank-Nicolson).

    Provides linsolve using the Jacobian. Declares jacobian as abstract
    so mypy can verify the call.
    """

    if TYPE_CHECKING:
        _bkd: Backend[Array]

    @abstractmethod
    def jacobian(self, state: Array) -> Array:
        """Compute the Jacobian dR/dy_n (declared here, implemented by concrete class)."""
        ...

    def linsolve(self, state: Array, residual: Array) -> Array:
        """Solve the linear system J dy = residual."""
        return solve_maybe_sparse(self._bkd, self.jacobian(state), residual)
