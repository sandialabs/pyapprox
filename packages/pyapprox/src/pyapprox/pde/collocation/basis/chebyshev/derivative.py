"""Chebyshev derivative matrix computation.

Provides the barycentric formula for computing derivative matrices
at Chebyshev-Gauss-Lobatto nodes.
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class ChebyshevDerivativeMatrix1D(Generic[Array]):
    """Compute derivative matrix for Chebyshev collocation.

    Uses the barycentric formula for Lagrange interpolation derivatives:

        D[i,j] = (c_i / c_j) / (x_i - x_j)  for i != j
        D[i,i] = -sum_{j != i} D[i,j]       (row sum property)

    where c_i are the barycentric weights:
        c_i = 2 at endpoints (i=0 and i=n-1), 1 elsewhere
        alternating sign: c_i *= (-1)^i

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def compute(self, nodes: Array) -> Array:
        """Compute Chebyshev derivative matrix.

        Parameters
        ----------
        nodes : Array
            Chebyshev-Gauss-Lobatto nodes. Shape: (npts,)

        Returns
        -------
        Array
            Derivative matrix. Shape: (npts, npts)
        """
        bkd = self._bkd
        npts = nodes.shape[0]

        if npts == 1:
            return bkd.zeros((1, 1))

        # Barycentric weights: c_i = 2 at endpoints, alternating sign
        c = bkd.ones((npts,))
        c = bkd.copy(c)  # Ensure writeable
        c[0] = 2.0
        c[npts - 1] = 2.0

        # Alternating sign: multiply by (-1)^i
        # Use where to avoid numpy roundtrip and preserve backend dtype
        idx = bkd.arange(npts)
        signs = bkd.where(idx % 2 == 0, bkd.ones((npts,)), -bkd.ones((npts,)))
        c = c * signs

        # Compute D[i,j] = (c_i/c_j) / (x_i - x_j) for i != j
        # Using outer operations for full vectorization
        X = nodes[:, None] - nodes[None, :]  # (npts, npts) differences
        C = c[:, None] / c[None, :]  # (npts, npts) weight ratios

        # Create mask for diagonal (where X = 0)
        diag_idx = bkd.arange(npts)
        # Avoid division by zero: set diagonal of X to 1 temporarily
        X_safe = bkd.copy(X)
        X_safe[diag_idx, diag_idx] = 1.0

        # Compute off-diagonal elements
        D: Array = C / X_safe

        # Zero out diagonal (will be set via row sum property)
        D[diag_idx, diag_idx] = 0.0

        # Row sum property: D[i,i] = -sum_{j != i} D[i,j]
        row_sums = bkd.sum(D, axis=1)
        D[diag_idx, diag_idx] = -row_sums

        return D
