"""Default newton_jacobian mixin for ODE residuals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

from pyapprox.ode.linear_operator import (
    LinearOperatorProtocol,
    create_linear_operator,
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
    def jacobian(self, state: Array, time: float) -> Array: ...

    if TYPE_CHECKING:
        _bkd: Backend[Array]

    def newton_jacobian(
        self, state: Array, time: float, coefficient: float
    ) -> LinearOperatorProtocol[Array]:
        """Return ``M - coefficient * J(state, time)`` as an operator.

        The mass adds itself (``add_to``): an identity or lumped mass
        adds only its diagonal, so it never materializes as a dense
        ``(nstates, nstates)`` matrix and a sparse ``J`` keeps a sparse
        Newton matrix.
        """
        return create_linear_operator(
            self.mass_matrix().add_to(-coefficient * self.jacobian(state, time)),
            self._bkd,
        )
