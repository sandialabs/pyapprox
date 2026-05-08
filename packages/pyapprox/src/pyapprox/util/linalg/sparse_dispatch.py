"""Sparse/dense linear solve dispatch.

Provides dispatch functions that work with both scipy sparse matrices
and dense numpy arrays, avoiding the need to convert sparse matrices
to dense for linear solves.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix, issparse, spmatrix
from scipy.sparse.linalg import spsolve

from pyapprox.util.backends.protocols import Array, Backend


def sparse_or_dense_solve(
    A: Union[spmatrix, NDArray[np.floating[Any]]],
    b: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Solve A @ x = b, dispatching to spsolve when A is sparse.

    Low-level function operating on numpy arrays. For backend-aware
    solving, use ``solve_maybe_sparse`` instead.

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
        result: NDArray[np.floating[Any]] = spsolve(A_csc, b)
        return result
    return np.linalg.solve(A, b)


def solve_maybe_sparse(
    bkd: Backend[Array],
    A: Union[spmatrix, Array],
    b: Array,
) -> Array:
    """Backend-aware solve that handles both sparse and dense matrices.

    If A is a scipy sparse matrix, delegates to ``bkd.solve_sparse(A, b)``.
    If A is dense, delegates to ``bkd.solve(A, b)``.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    A : sparse matrix or Array
        System matrix. Must be 2D with shape (n, n).
    b : Array
        Right-hand side vector. Shape: (n,).

    Returns
    -------
    Array
        Solution vector in backend format. Shape: (n,).

    Raises
    ------
    ValueError
        If A is not 2D or b is not 1D, or dimensions are incompatible.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}")
    if issparse(A):
        return bkd.solve_sparse(A, b)
    return bkd.solve(A, b)
