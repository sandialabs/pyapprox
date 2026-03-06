"""Piecewise polynomial density basis with analytical mass matrices."""

from typing import Generic, Tuple

from pyapprox.surrogates.affine.univariate.piecewisepoly import (
    PiecewiseLinear,
    PiecewiseQuadratic,
)
from pyapprox.util.backends.protocols import Array, Backend


class PiecewiseDensityBasis(Generic[Array]):
    """Density basis wrapping piecewise polynomial basis with analytical mass matrix.

    Supports degree=1 (piecewise linear hat functions) and degree=2
    (piecewise quadratic Lagrange functions).

    Parameters
    ----------
    y_min : float
        Left domain boundary.
    y_max : float
        Right domain boundary.
    nbasis : int
        Number of basis functions (nodes). For degree=2, forced to be odd.
    degree : int
        Polynomial degree (1 or 2).
    bkd : Backend[Array]
        Computational backend.

    Raises
    ------
    ValueError
        If degree is not 1 or 2, or nbasis < 2.
    """

    def __init__(
        self,
        y_min: float,
        y_max: float,
        nbasis: int,
        degree: int,
        bkd: Backend[Array],
    ) -> None:
        if degree not in (1, 2):
            raise ValueError(f"degree must be 1 or 2, got {degree}")
        if nbasis < 2:
            raise ValueError(f"nbasis must be >= 2, got {nbasis}")

        self._bkd = bkd
        self._degree = degree
        self._y_min = y_min
        self._y_max = y_max

        if degree == 2 and nbasis % 2 == 0:
            nbasis = nbasis + 1

        nodes = bkd.linspace(y_min, y_max, nbasis)
        self._nodes = nodes
        self._nbasis = nbasis

        if degree == 1:
            self._poly = PiecewiseLinear(nodes, bkd)
        else:
            self._poly = PiecewiseQuadratic(nodes, bkd)

        self._mass = self._compute_mass_matrix()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nbasis(self) -> int:
        """Return the number of basis functions."""
        return self._nbasis

    def domain(self) -> Tuple[float, float]:
        """Return the domain boundaries."""
        return (self._y_min, self._y_max)

    def evaluate(self, y_values: Array) -> Array:
        """Evaluate all basis functions at given points.

        Parameters
        ----------
        y_values : Array
            Query points. Shape: (1, npts).

        Returns
        -------
        Array
            Basis values. Shape: (nbasis, npts).
        """
        if y_values.ndim == 2:
            y_1d = y_values[0]
        else:
            y_1d = y_values
        # poly(y_1d) returns (npts, nbasis)
        vals = self._poly(y_1d)
        # Transpose to (nbasis, npts)
        return self._bkd.transpose(vals, (1, 0))

    def mass_matrix(self) -> Array:
        """Return the precomputed mass matrix.

        Returns
        -------
        Array
            Mass matrix M_ij = int phi_i(y) phi_j(y) dy.
            Shape: (nbasis, nbasis).
        """
        return self._mass

    def _compute_mass_matrix(self) -> Array:
        """Compute the analytical mass matrix."""
        if self._degree == 1:
            return self._mass_matrix_linear()
        else:
            return self._mass_matrix_quadratic()

    def _mass_matrix_linear(self) -> Array:
        """Analytical mass matrix for piecewise linear (hat) functions.

        For hat functions on nodes y_0, ..., y_{n-1}:
        M[i,i] = (h_{i-1} + h_i) / 3   (interior)
        M[i,i+1] = M[i+1,i] = h_i / 6
        Boundary: M[0,0] = h_0/3, M[n-1,n-1] = h_{n-2}/3
        """
        bkd = self._bkd
        nodes = self._nodes
        n = self._nbasis
        h = nodes[1:] - nodes[:-1]  # (n-1,)

        M = bkd.zeros((n, n))

        # Diagonal entries
        # Interior nodes: M[i,i] = (h[i-1] + h[i]) / 3
        for ii in range(n):
            if ii > 0 and ii < n - 1:
                M[ii, ii] = (h[ii - 1] + h[ii]) / 3.0
            elif ii == 0:
                M[0, 0] = h[0] / 3.0
            else:
                M[n - 1, n - 1] = h[n - 2] / 3.0

        # Off-diagonal entries
        for ii in range(n - 1):
            M[ii, ii + 1] = h[ii] / 6.0
            M[ii + 1, ii] = h[ii] / 6.0

        return M

    def _mass_matrix_quadratic(self) -> Array:
        """Analytical mass matrix for piecewise quadratic (Lagrange) functions.

        Per element [y_{2k}, y_{2k+1}, y_{2k+2}] with h = y_{2k+2} - y_{2k}:
        M_local = (h/30) * [[4, 2, -1],
                             [2, 16, 2],
                             [-1, 2, 4]]
        Accumulated into global matrix.
        """
        bkd = self._bkd
        nodes = self._nodes
        n = self._nbasis
        n_elements = (n - 1) // 2

        M = bkd.zeros((n, n))

        for kk in range(n_elements):
            i0 = 2 * kk
            i1 = i0 + 1
            i2 = i0 + 2
            h = nodes[i2] - nodes[i0]
            c = h / 30.0

            M[i0, i0] = M[i0, i0] + 4.0 * c
            M[i0, i1] = M[i0, i1] + 2.0 * c
            M[i0, i2] = M[i0, i2] - 1.0 * c
            M[i1, i0] = M[i1, i0] + 2.0 * c
            M[i1, i1] = M[i1, i1] + 16.0 * c
            M[i1, i2] = M[i1, i2] + 2.0 * c
            M[i2, i0] = M[i2, i0] - 1.0 * c
            M[i2, i1] = M[i2, i1] + 2.0 * c
            M[i2, i2] = M[i2, i2] + 4.0 * c

        return M

# TODO: Why is __all__ in this file
__all__ = ["PiecewiseDensityBasis"]
