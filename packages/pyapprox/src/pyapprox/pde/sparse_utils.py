"""Sparse matrix utilities for boundary condition application.

Provides utilities for applying Dirichlet boundary conditions to
both scipy sparse matrices and dense numpy arrays.
"""

from __future__ import annotations

from typing import Any, Union, overload

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, spmatrix

# Re-export for backward compatibility
from pyapprox.util.linalg.sparse_dispatch import (  # noqa: F401
    solve_maybe_sparse,
    sparse_or_dense_solve,
)


@overload
def apply_dirichlet_rows(
    matrix: NDArray[np.floating[Any]],
    dof_indices: NDArray[np.integer[Any]],
) -> NDArray[np.floating[Any]]: ...


@overload
def apply_dirichlet_rows(
    matrix: spmatrix,
    dof_indices: NDArray[np.integer[Any]],
) -> spmatrix: ...


def apply_dirichlet_rows(
    matrix: Union[spmatrix, NDArray[np.floating[Any]]],
    dof_indices: NDArray[np.integer[Any]],
) -> Union[spmatrix, NDArray[np.floating[Any]]]:
    """Zero rows and set diagonal to 1.0 for Dirichlet DOFs.

    Works for both scipy sparse matrices and dense numpy arrays.
    For sparse matrices, operates on CSR format directly.

    Parameters
    ----------
    matrix : sparse matrix or ndarray
        System matrix to modify. Shape: (n, n).
    dof_indices : array_like
        DOF indices where Dirichlet conditions are applied.

    Returns
    -------
    sparse matrix or ndarray
        Modified matrix (same type as input).
    """
    if isinstance(matrix, spmatrix):
        mat = matrix.tocsr(copy=True)
        nrows = mat.shape[0]
        dofs = np.unique(np.asarray(dof_indices))
        is_dirichlet_row = np.zeros(nrows, dtype=bool)
        is_dirichlet_row[dofs] = True

        # Zero every stored entry of the Dirichlet rows in one masked
        # assignment: the row of each stored entry is expanded from
        # indptr, so the cost is O(nnz) with no per-DOF Python loop.
        entry_rows = np.repeat(np.arange(nrows), np.diff(mat.indptr))
        mat.data[is_dirichlet_row[entry_rows]] = 0.0
        mat.eliminate_zeros()

        # The Dirichlet rows are now structurally empty, so adding the
        # identity restricted to them sets each diagonal to 1.0 whether
        # or not it was in the original sparsity pattern (it is absent
        # when a matrix is assembled without a diagonal entry for some
        # Dirichlet DOFs, e.g. pressure DOFs in some Stokes
        # formulations), and leaves every other row unchanged.
        dirichlet_identity = csr_matrix(
            (np.ones(dofs.shape[0], dtype=mat.dtype), (dofs, dofs)),
            shape=mat.shape,
        )
        return mat + dirichlet_identity
    else:
        mat = matrix.copy()
        mat[dof_indices, :] = 0.0
        mat[dof_indices, dof_indices] = 1.0
        return mat
