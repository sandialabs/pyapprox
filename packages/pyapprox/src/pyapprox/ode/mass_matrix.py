"""Mass matrix value objects for ODE time stepping.

Encapsulates mass matrix storage, application, and solve so that
steppers never dispatch on sparsity or rebuild factorizations.
"""

from typing import Generic, Optional, Protocol, runtime_checkable

import numpy as np
from scipy.sparse import csc_matrix, issparse, spmatrix
from scipy.sparse.linalg import SuperLU, splu

from pyapprox.util.backends.protocols import Array, Backend


def _find_zero_rows(matrix: Array, bkd: Backend[Array]) -> list[int]:
    """Structural zero-row detection (DAE algebraic DOFs)."""
    row_sums = bkd.sum(bkd.abs(matrix), axis=1)
    return [
        i
        for i in range(matrix.shape[0])
        if bkd.to_float(row_sums[i]) == 0.0
    ]


@runtime_checkable
class MassMatrixProtocol(Protocol, Generic[Array]):
    """Protocol for mass matrix value objects."""

    def apply(self, vec: Array) -> Array: ...

    def solve(self, vec: Array) -> Array: ...

    def apply_transpose(self, vec: Array) -> Array: ...

    def solve_transpose(self, vec: Array) -> Array: ...

    def as_matrix(self) -> Array: ...

    def is_identity(self) -> bool: ...

    def is_singular(self) -> bool:
        """Whether the mass matrix is structurally singular.

        Detection is structural (all-zero rows, the DAE pattern, e.g.
        the Stokes mass [[M_vel, 0], [0, 0]]): True means definitely
        singular; False does not prove nonsingularity for a general
        matrix. Mass-only solves must be guarded when this is True.
        """
        ...

    def zero_rows(self) -> list[int]:
        """Indices of structurally zero rows (algebraic DOFs).

        Empty when the matrix has no zero rows. For DAE masses these
        are the constraint rows (e.g. pressure DOFs for Stokes).
        """
        ...


class IdentityMassMatrix(Generic[Array]):
    """Identity mass matrix — all operations are no-ops."""

    def __init__(self, n: int, bkd: Backend[Array]) -> None:
        self._n = n
        self._bkd = bkd
        self._cached_matrix: Optional[Array] = None

    def apply(self, vec: Array) -> Array:
        return vec

    def solve(self, vec: Array) -> Array:
        return vec

    def apply_transpose(self, vec: Array) -> Array:
        return vec

    def solve_transpose(self, vec: Array) -> Array:
        return vec

    def as_matrix(self) -> Array:
        if self._cached_matrix is None:
            self._cached_matrix = self._bkd.eye(self._n)
        return self._cached_matrix

    def is_identity(self) -> bool:
        return True

    def is_singular(self) -> bool:
        return False

    def zero_rows(self) -> list[int]:
        return []


class ConstantDenseMassMatrix(Generic[Array]):
    """Dense mass matrix with lazily cached compact LU factorization.

    Uses backend-native lu_factor/lu_solve so that PyTorch autograd
    is preserved and no scipy roundtrip is needed.

    Factorization is deferred to the first solve() call so that
    singular DAE masses can be constructed and used for apply()
    (mirroring ConstantSparseMassMatrix).
    """

    def __init__(self, matrix: Array, bkd: Backend[Array]) -> None:
        self._matrix = matrix
        self._bkd = bkd
        self._lu_and_pivots: Optional[tuple[Array, Array]] = None
        self._zero_rows: list[int] = _find_zero_rows(matrix, bkd)

    def _ensure_lu(self) -> tuple[Array, Array]:
        if self._lu_and_pivots is None:
            self._lu_and_pivots = self._bkd.lu_factor(self._matrix)
        return self._lu_and_pivots

    def apply(self, vec: Array) -> Array:
        return self._bkd.dot(self._matrix, vec)

    def solve(self, vec: Array) -> Array:
        lu, pivots = self._ensure_lu()
        return self._bkd.lu_solve(lu, pivots, vec)

    def apply_transpose(self, vec: Array) -> Array:
        return self._bkd.dot(self._matrix.T, vec)

    def solve_transpose(self, vec: Array) -> Array:
        lu, pivots = self._ensure_lu()
        return self._bkd.lu_solve(lu, pivots, vec, adjoint=True)

    def as_matrix(self) -> Array:
        return self._matrix

    def is_identity(self) -> bool:
        return False

    def is_singular(self) -> bool:
        return len(self._zero_rows) > 0

    def zero_rows(self) -> list[int]:
        return list(self._zero_rows)


class ConstantSparseMassMatrix(Generic[Array]):
    """Sparse mass matrix with lazily cached SuperLU factorization.

    Uses scipy sparse LU. Autograd is not preserved — acceptable
    because sparse mass matrices only appear in Galerkin FEM where
    torch autograd is not used.

    Factorization is deferred to first solve() call so that DAE mass
    matrices (e.g. Stokes [M_vel, 0; 0, 0]) can be constructed and
    used for apply() without triggering a singular-factor error.
    """

    def __init__(self, matrix: spmatrix, bkd: Backend[Array]) -> None:
        self._matrix = matrix
        self._csc = csc_matrix(matrix)
        self._bkd = bkd
        self._lu: Optional[SuperLU] = None
        # scipy's sparse row-sum is numpy-native; the result is pure
        # index metadata, so no backend round trip is warranted.
        row_sums = np.asarray(abs(self._csc).sum(axis=1)).ravel()
        self._zero_rows: list[int] = [
            int(i) for i in np.nonzero(row_sums == 0.0)[0]
        ]

    def _ensure_lu(self) -> SuperLU:
        if self._lu is None:
            self._lu = splu(self._csc)
        return self._lu

    def apply(self, vec: Array) -> Array:
        vec_np = self._bkd.to_numpy(vec)
        return self._bkd.asarray(self._matrix @ vec_np)

    def solve(self, vec: Array) -> Array:
        vec_np = self._bkd.to_numpy(vec)
        return self._bkd.asarray(self._ensure_lu().solve(vec_np))

    def apply_transpose(self, vec: Array) -> Array:
        vec_np = self._bkd.to_numpy(vec)
        return self._bkd.asarray(self._matrix.T @ vec_np)

    def solve_transpose(self, vec: Array) -> Array:
        vec_np = self._bkd.to_numpy(vec)
        return self._bkd.asarray(self._ensure_lu().solve(vec_np, trans="T"))

    def as_matrix(self) -> spmatrix:
        return self._matrix

    def is_identity(self) -> bool:
        return False

    def is_singular(self) -> bool:
        return len(self._zero_rows) > 0

    def zero_rows(self) -> list[int]:
        return list(self._zero_rows)


def create_mass_matrix(
    matrix: Array, bkd: Backend[Array]
) -> MassMatrixProtocol[Array]:
    """Create a MassMatrix from a raw array, detecting identity and sparsity."""
    if issparse(matrix):
        return ConstantSparseMassMatrix(matrix, bkd)
    matrix_np = bkd.to_numpy(matrix)
    if np.allclose(matrix_np, np.eye(matrix_np.shape[0])):
        return IdentityMassMatrix(matrix_np.shape[0], bkd)
    return ConstantDenseMassMatrix(matrix, bkd)
