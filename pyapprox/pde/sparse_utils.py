"""Sparse matrix utilities for boundary condition application.

Provides utilities for applying Dirichlet boundary conditions to
both scipy sparse matrices and dense numpy arrays.
"""

from __future__ import annotations

import warnings
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse, spmatrix

# Re-export for backward compatibility
from pyapprox.util.linalg.sparse_dispatch import (  # noqa: F401
    solve_maybe_sparse,
    sparse_or_dense_solve,
)


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
    if issparse(matrix):
        mat = matrix.tocsr().copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            for dof in dof_indices:
                mat[dof, :] = 0.0
                mat[dof, dof] = 1.0
        mat.eliminate_zeros()
        return mat
    else:
        mat = matrix.copy()
        for dof in dof_indices:
            mat[dof, :] = 0.0
            mat[dof, dof] = 1.0
        return mat
