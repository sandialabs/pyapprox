"""Mass matrix value objects for ODE time stepping.

Encapsulates mass matrix storage, application, and solve so that
steppers never dispatch on sparsity or rebuild factorizations.
"""

from typing import Generic, Optional, Protocol, runtime_checkable

from scipy.sparse import issparse, spmatrix
from scipy.sparse.linalg import SuperLU, splu

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class MassMatrixProtocol(Protocol, Generic[Array]):
    """Protocol for mass matrix value objects."""

    def apply(self, vec: Array) -> Array: ...

    def solve(self, vec: Array) -> Array: ...

    def apply_transpose(self, vec: Array) -> Array: ...

    def solve_transpose(self, vec: Array) -> Array: ...

    def as_matrix(self) -> Array: ...

    def is_identity(self) -> bool: ...


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


class ConstantDenseMassMatrix(Generic[Array]):
    """Dense mass matrix with cached compact LU factorization.

    Uses backend-native lu_factor/lu_solve so that PyTorch autograd
    is preserved and no scipy roundtrip is needed.
    """

    def __init__(self, matrix: Array, bkd: Backend[Array]) -> None:
        self._matrix = matrix
        self._bkd = bkd
        self._LU, self._pivots = bkd.lu_factor(matrix)

    def apply(self, vec: Array) -> Array:
        return self._bkd.dot(self._matrix, vec)

    def solve(self, vec: Array) -> Array:
        return self._bkd.lu_solve(self._LU, self._pivots, vec)

    def apply_transpose(self, vec: Array) -> Array:
        return self._bkd.dot(self._matrix.T, vec)

    def solve_transpose(self, vec: Array) -> Array:
        return self._bkd.lu_solve(self._LU, self._pivots, vec, adjoint=True)

    def as_matrix(self) -> Array:
        return self._matrix

    def is_identity(self) -> bool:
        return False


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
        from scipy.sparse import csc_matrix

        self._matrix = matrix
        self._csc = csc_matrix(matrix)
        self._bkd = bkd
        self._lu: Optional[SuperLU] = None

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

    def as_matrix(self) -> Array:
        return self._bkd.asarray(self._matrix.toarray())

    def is_identity(self) -> bool:
        return False


def create_mass_matrix(
    matrix: Array, bkd: Backend[Array]
) -> MassMatrixProtocol[Array]:
    """Create a MassMatrix from a raw array, detecting identity and sparsity."""
    if issparse(matrix):
        return ConstantSparseMassMatrix(matrix, bkd)
    import numpy as np

    matrix_np = bkd.to_numpy(matrix)
    if np.allclose(matrix_np, np.eye(matrix_np.shape[0])):
        return IdentityMassMatrix(matrix_np.shape[0], bkd)
    return ConstantDenseMassMatrix(matrix, bkd)
