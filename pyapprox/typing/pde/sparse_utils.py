"""Sparse matrix utilities shared by galerkin and time integration modules.

Provides dispatch functions that work with both scipy sparse matrices
and dense numpy arrays, avoiding the need to convert sparse matrices
to dense for linear solves and boundary condition application.
"""

import warnings

import numpy as np
from scipy.sparse import issparse, csc_matrix
from scipy.sparse.linalg import spsolve


def galerkin_solve(A, b):
    """Solve A @ x = b, dispatching to spsolve when A is sparse.

    Parameters
    ----------
    A : sparse matrix or ndarray
        System matrix. If sparse, converted to CSC for spsolve.
    b : ndarray
        Right-hand side vector. Shape: (n,).

    Returns
    -------
    ndarray
        Solution vector. Shape: (n,).

    Raises
    ------
    NotImplementedError
        If A is sparse and b has more than one column.
    """
    if issparse(A):
        A_csc = csc_matrix(A) if A.format != "csc" else A
        if b.ndim > 1:
            raise NotImplementedError(
                "Sparse solve with multiple RHS columns is not supported. "
                "Use scipy.sparse.linalg.splu for multi-column solves."
            )
        return spsolve(A_csc, b)
    return np.linalg.solve(A, b)


def apply_dirichlet_rows(matrix, dof_indices):
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
