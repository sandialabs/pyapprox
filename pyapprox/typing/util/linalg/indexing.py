"""Indexing utilities for arrays.

This module provides backend-agnostic array indexing functions that work
with both NumPy and PyTorch arrays.
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array


def extract_submatrix(
    mat: Array, row_indices: Array, col_indices: Array
) -> Array:
    """Extract submatrix using row and column indices.

    This is a backend-agnostic replacement for NumPy's np.ix_ indexing pattern.
    Instead of using ``mat[np.ix_(row_indices, col_indices)]``, use
    ``extract_submatrix(mat, row_indices, col_indices)``.

    Parameters
    ----------
    mat : Array
        Input matrix of shape (m, n).
    row_indices : Array
        Row indices to select. Shape (nrows,).
    col_indices : Array
        Column indices to select. Shape (ncols,).

    Returns
    -------
    submat : Array
        Submatrix with shape (len(row_indices), len(col_indices)).

    Examples
    --------
    NumPy equivalent:
        >>> # Instead of: result = mat[np.ix_([0, 2], [1, 3])]
        >>> result = extract_submatrix(mat, np.array([0, 2]), np.array([1, 3]))

    PyTorch equivalent:
        >>> # Works identically with torch tensors
        >>> result = extract_submatrix(mat, torch.tensor([0, 2]), torch.tensor([1, 3]))
    """
    # First select rows, then columns
    # This works for both NumPy and PyTorch
    rows = mat[row_indices, :]
    return rows[:, col_indices]
