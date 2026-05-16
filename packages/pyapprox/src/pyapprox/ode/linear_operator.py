"""Linear operator protocol and concrete implementations.

Used by the adjoint solver to represent the diagonal Jacobian block
(dR/dy_n)^T. Explicit steppers return a MassMatrixTransposeOperator
(free for identity mass), while implicit steppers return a
DenseMatrixOperator wrapping the full (M - dt*J)^T.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.ode.mass_matrix import MassMatrixProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.sparse_dispatch import solve_maybe_sparse


@runtime_checkable
class LinearOperatorProtocol(Protocol, Generic[Array]):
    """Protocol for objects that can solve, apply, and expose a matrix."""

    def solve(self, rhs: Array) -> Array: ...

    def apply(self, vec: Array) -> Array: ...

    def as_matrix(self) -> Array: ...


class MassMatrixTransposeOperator(Generic[Array]):
    """M^T as a LinearOperator, where M is a MassMatrix.

    For IdentityMassMatrix, all three methods are no-ops.
    """

    def __init__(self, mass: MassMatrixProtocol[Array]) -> None:
        self._mass = mass

    def solve(self, rhs: Array) -> Array:
        return self._mass.solve_transpose(rhs)

    def apply(self, vec: Array) -> Array:
        return self._mass.apply_transpose(vec)

    def as_matrix(self) -> Array:
        return self._mass.as_matrix().T


class MatrixOperator(Generic[Array]):
    """Wraps a pre-assembled dense or sparse matrix. No caching of factorization."""

    def __init__(self, matrix: Array, bkd: Backend[Array]) -> None:
        self._matrix = matrix
        self._bkd = bkd

    def solve(self, rhs: Array) -> Array:
        return solve_maybe_sparse(self._bkd, self._matrix, rhs)

    def apply(self, vec: Array) -> Array:
        return self._bkd.dot(self._matrix, vec)

    def as_matrix(self) -> Array:
        return self._matrix
