"""Linear operator protocol and concrete implementations.

Used by the adjoint solver to represent the diagonal Jacobian block
(dR/dy_n)^T, and by implicit steppers for the Newton Jacobian
M - coefficient*J. Explicit steppers return a MassMatrixTransposeOperator
(free for identity mass), while implicit steppers return a MatrixOperator
wrapping the full assembled matrix.
"""

from typing import Any, Generic, Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix, issparse, spmatrix
from scipy.sparse.linalg import SuperLU, splu

from pyapprox.ode.mass_matrix import MassMatrixProtocol
from pyapprox.util.backends.numpy import NumpyBkd
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


class SparseMatrixOperator:
    """Wraps a scipy sparse matrix with a cached LU factorization.

    Unlike MatrixOperator, which re-solves from scratch on every call
    (O(n^3)-ish per solve), the first ``solve``/``solve_transpose``
    triggers a single sparse LU factorization that is reused for all
    subsequent solves in either orientation (``trans="T"`` reuses the
    same factors). Intended for implicit time stepping where one
    Jacobian is solved against many right-hand sides.

    NumPy backend only — deliberately non-generic and concretely
    NDArray-typed: routing a sparse factorization through scipy would
    silently break the torch autograd graph, the same reason
    ``TorchBkd.solve_sparse`` raises. Satisfies
    ``LinearOperatorProtocol[ndarray]``.
    """

    def __init__(self, matrix: spmatrix, bkd: NumpyBkd) -> None:
        if not issparse(matrix):
            raise TypeError(
                "SparseMatrixOperator requires a scipy sparse matrix, "
                f"got {type(matrix).__name__}. Use MatrixOperator for "
                "dense matrices."
            )
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"matrix must be square, got shape {matrix.shape}")
        if not isinstance(bkd, NumpyBkd):
            raise TypeError(
                "SparseMatrixOperator supports only NumpyBkd (sparse "
                "factorization would break the torch autograd graph), "
                f"got {type(bkd).__name__}"
            )
        self._matrix = csc_matrix(matrix) if matrix.format != "csc" else matrix
        self._bkd = bkd
        self._lu: Optional[SuperLU] = None

    def _factorization(self) -> SuperLU:
        if self._lu is None:
            self._lu = splu(self._matrix)
        return self._lu

    def solve(self, rhs: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        return self._bkd.asarray(self._factorization().solve(rhs))

    def apply(self, vec: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        return self._bkd.asarray(self._matrix @ vec)

    def as_matrix(self) -> NDArray[np.floating[Any]]:
        """Materialize the operator as a dense matrix (like the
        block-diagonal operator, only on explicit request)."""
        return self._bkd.asarray(self._matrix.toarray())

    def solve_transpose(
        self, rhs: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        return self._bkd.asarray(self._factorization().solve(rhs, trans="T"))

    def apply_transpose(
        self, vec: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        return self._bkd.asarray(self._matrix.T @ vec)


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


class BlockDiagonalLinearOperator(Generic[Array]):
    """Block-diagonal operator: k blocks of size (m, m).

    Stores blocks as a (k, m, m) array. Uses backend-batched linalg
    routines for solve/apply, giving O(k * m^3) cost instead of
    O((k*m)^3) for a dense operator.
    """

    def __init__(self, blocks: Array, bkd: Backend[Array]) -> None:
        if blocks.ndim != 3:
            raise ValueError(
                f"blocks must be 3D (k, m, m), got shape {blocks.shape}"
            )
        if blocks.shape[1] != blocks.shape[2]:
            raise ValueError(
                f"blocks must be square per block, got shape {blocks.shape}"
            )
        self._blocks = blocks
        self._bkd = bkd
        self._nblocks: int = blocks.shape[0]
        self._block_size: int = blocks.shape[1]

    def solve(self, rhs: Array) -> Array:
        rhs_3d = self._bkd.reshape(rhs, (self._nblocks, self._block_size, 1))
        sol_3d = self._bkd.solve(self._blocks, rhs_3d)
        return self._bkd.reshape(sol_3d, (self._nblocks * self._block_size,))

    def apply(self, vec: Array) -> Array:
        vec_2d = self._bkd.reshape(vec, (self._nblocks, self._block_size))
        result_2d = self._bkd.einsum("kij,kj->ki", self._blocks, vec_2d)
        return self._bkd.reshape(result_2d, (self._nblocks * self._block_size,))

    def solve_transpose(self, rhs: Array) -> Array:
        rhs_3d = self._bkd.reshape(rhs, (self._nblocks, self._block_size, 1))
        blocks_T = self._bkd.transpose(self._blocks, (0, 2, 1))
        sol_3d = self._bkd.solve(blocks_T, rhs_3d)
        return self._bkd.reshape(sol_3d, (self._nblocks * self._block_size,))

    def apply_transpose(self, vec: Array) -> Array:
        vec_2d = self._bkd.reshape(vec, (self._nblocks, self._block_size))
        result_2d = self._bkd.einsum("kji,kj->ki", self._blocks, vec_2d)
        return self._bkd.reshape(result_2d, (self._nblocks * self._block_size,))

    def as_matrix(self) -> Array:
        n = self._nblocks * self._block_size
        full = self._bkd.zeros((n, n))
        for i in range(self._nblocks):
            s = i * self._block_size
            e = s + self._block_size
            full[s:e, s:e] = self._blocks[i]
        return full
