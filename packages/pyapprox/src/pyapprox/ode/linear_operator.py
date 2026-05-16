"""Linear operator protocol and concrete implementations.

Used by the adjoint solver to represent the diagonal Jacobian block
(dR/dy_n)^T, and by implicit steppers for the Newton Jacobian
M - coefficient*J. Explicit steppers return a MassMatrixTransposeOperator
(free for identity mass), while implicit steppers return a MatrixOperator
wrapping the full assembled matrix.
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

    def solve_transpose(self, rhs: Array) -> Array: ...

    def apply_transpose(self, vec: Array) -> Array: ...


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

    def solve_transpose(self, rhs: Array) -> Array:
        return self._mass.solve(rhs)

    def apply_transpose(self, vec: Array) -> Array:
        return self._mass.apply(vec)


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

    def solve_transpose(self, rhs: Array) -> Array:
        return solve_maybe_sparse(self._bkd, self._matrix.T, rhs)

    def apply_transpose(self, vec: Array) -> Array:
        return self._bkd.dot(self._matrix.T, vec)


class TransposeLinearOperator(Generic[Array]):
    """Lazy transpose: swaps solve/solve_transpose, apply/apply_transpose."""

    def __init__(self, op: LinearOperatorProtocol[Array]) -> None:
        self._op = op

    def solve(self, rhs: Array) -> Array:
        return self._op.solve_transpose(rhs)

    def apply(self, vec: Array) -> Array:
        return self._op.apply_transpose(vec)

    def as_matrix(self) -> Array:
        return self._op.as_matrix().T

    def solve_transpose(self, rhs: Array) -> Array:
        return self._op.solve(rhs)

    def apply_transpose(self, vec: Array) -> Array:
        return self._op.apply(vec)
