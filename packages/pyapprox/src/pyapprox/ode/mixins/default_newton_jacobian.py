"""Default newton_jacobian mixin for ODE residuals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

from pyapprox.ode.linear_operator import (
    LinearOperatorProtocol,
    MatrixOperator,
)
from pyapprox.ode.mass_matrix import MassMatrixProtocol
from pyapprox.util.backends.protocols import Array, Backend


class DefaultNewtonJacobianMixin(ABC, Generic[Array]):
    """Default newton_jacobian: builds dense (M - coeff*J) operator.

    For residuals without exploitable structure in the Newton Jacobian.
    Override in subclasses (BlockDiagonalLinearOperator for batched,
    sparse-factored for FEM) for better performance.
    """

    @abstractmethod
    def mass_matrix(self) -> MassMatrixProtocol[Array]: ...

    @abstractmethod
    def jacobian(self, state: Array) -> Array: ...

    if TYPE_CHECKING:
        _bkd: Backend[Array]

    def newton_jacobian(
        self, state: Array, coefficient: float
    ) -> LinearOperatorProtocol[Array]:
        matrix = (
            self.mass_matrix().as_matrix() - coefficient * self.jacobian(state)
        )
        return MatrixOperator(matrix, self._bkd)
