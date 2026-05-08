"""Pivoted QR factorization for Fekete point selection.

This module provides pivoted QR factorization, which is used for
one-shot Fekete point selection where we need to select all points
at once.
"""

from typing import Generic, Optional, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class PivotedQRFactorizer(Generic[Array]):
    """Pivoted QR factorization for one-shot point selection.

    Uses column pivoting to select the most linearly independent
    columns/rows, which is useful for Fekete point selection.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>> bkd = NumpyBkd()
    >>> A = bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
    >>> factorizer = PivotedQRFactorizer(bkd)
    >>> Q, R, pivots = factorizer.factorize(A.T, k=2)
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def factorize(
        self, matrix: Array, k: Optional[int] = None
    ) -> Tuple[Array, Array, Array]:
        """Perform pivoted QR factorization.

        Computes A[:, pivots] = Q @ R with column pivoting.

        Parameters
        ----------
        matrix : Array
            Matrix to factorize. Shape: (nrows, ncols)
            For Fekete selection, pass the transpose of the basis matrix
            (i.e., basis_mat.T where basis_mat has shape (nsamples, nterms)).
        k : int, optional
            Number of pivots to return. If None, returns all pivots.

        Returns
        -------
        Q : Array
            Orthogonal matrix. Shape: (nrows, min(nrows, ncols))
        R : Array
            Upper triangular matrix. Shape: (min(nrows, ncols), ncols)
        pivots : Array
            Pivot indices. Shape: (k,) or (ncols,) if k is None.
        """
        # Use scipy's pivoted QR via numpy operations
        import scipy.linalg

        mat_np = self._bkd.to_numpy(matrix)
        Q_np, R_np, pivots_np = scipy.linalg.qr(mat_np, pivoting=True, mode="economic")

        Q = self._bkd.asarray(Q_np)
        R = self._bkd.asarray(R_np)
        pivots = self._bkd.asarray(pivots_np, dtype=self._bkd.int64_dtype())

        if k is not None:
            pivots = pivots[:k]

        return Q, R, pivots

    def select_points(self, basis_matrix: Array, npoints: int) -> Array:
        """Select Fekete points from candidate samples.

        Parameters
        ----------
        basis_matrix : Array
            Basis evaluated at candidate samples. Shape: (nsamples, nterms)
        npoints : int
            Number of points to select.

        Returns
        -------
        Array
            Indices of selected points. Shape: (npoints,)
        """
        # Transpose for QR: we want to select rows of basis_matrix
        # QR with column pivoting on basis_matrix.T selects columns of .T
        # which corresponds to rows of basis_matrix
        _, _, pivots = self.factorize(basis_matrix.T, k=npoints)
        return pivots

    def __repr__(self) -> str:
        return f"PivotedQRFactorizer(bkd={self._bkd.__class__.__name__})"
