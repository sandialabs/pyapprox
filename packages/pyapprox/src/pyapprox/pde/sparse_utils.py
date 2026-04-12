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
        indptr = mat.indptr
        data = mat.data
        indices = mat.indices
        dof_set = np.asarray(dof_indices)

        # Identify which Dirichlet DOFs have the diagonal entry
        # already in the sparsity pattern and which need a CSR insert.
        starts = indptr[dof_set]
        ends = indptr[dof_set + 1]

        has_diag_mask = np.zeros(len(dof_set), dtype=bool)
        diag_data_pos = np.empty(len(dof_set), dtype=np.intp)
        needs_insert = []
        for ii, dof in enumerate(dof_set):
            s, e = starts[ii], ends[ii]
            col_slice = indices[s:e]
            dp = np.searchsorted(col_slice, dof)
            if dp < len(col_slice) and col_slice[dp] == dof:
                has_diag_mask[ii] = True
                diag_data_pos[ii] = s + dp
            else:
                needs_insert.append(dof)

        # Zero all data entries in Dirichlet rows in one pass.
        lengths = ends - starts
        total = int(lengths.sum())
        if total > 0:
            flat_idx = np.empty(total, dtype=np.intp)
            pos = 0
            for s, ln in zip(starts, lengths):
                flat_idx[pos:pos + ln] = np.arange(s, s + ln)
                pos += ln
            data[flat_idx] = 0.0

        # Set diagonals that are in the sparsity pattern.
        if has_diag_mask.any():
            data[diag_data_pos[has_diag_mask]] = 1.0

        # Insert diagonals that are missing from the sparsity pattern.
        # This is rare (only when the matrix was assembled without a
        # diagonal entry for certain Dirichlet DOFs, e.g. pressure
        # DOFs in some Stokes formulations).
        if needs_insert:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                for dof in needs_insert:
                    mat[dof, dof] = 1.0

        mat.eliminate_zeros()
        return mat
    else:
        mat = matrix.copy()
        mat[dof_indices, :] = 0.0
        mat[dof_indices, dof_indices] = 1.0
        return mat
