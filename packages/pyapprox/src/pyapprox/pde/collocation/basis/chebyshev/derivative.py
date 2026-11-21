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
        # (-1)^i
        c = bkd.asarray(
            [(-1.0) ** i * (2.0 if i in (0, npts - 1) else 1.0) for i in range(npts)]
        )

        # Compute D[i,j] = (c_i/c_j) / (x_i - x_j) for i != j
        # Using outer operations for full vectorization
        X = nodes[:, None] - nodes[None, :]  # (npts, npts) differences
        C = c[:, None] / c[None, :]  # (npts, npts) weight ratios

        # Avoid division by zero on the diagonal (where X = 0), then
        # zero it (it is set via the row sum property)
        eye = bkd.eye(npts)
        off_diagonal = C / (X + eye) * (1.0 - eye)

        # Row sum property: D[i,i] = -sum_{j != i} D[i,j]
        D: Array = off_diagonal - bkd.diag(bkd.sum(off_diagonal, axis=1))
        return D
