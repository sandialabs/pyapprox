"""Interpolation operators for interface to subdomain mapping.

Provides Lagrange interpolation between interface basis nodes and
subdomain boundary nodes.
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


def lagrange_interpolation_matrix(
    source_pts: Array, target_pts: Array, bkd: Backend[Array]
) -> Array:
    """Build Lagrange interpolation matrix from source to target points.

    Returns matrix M such that values_target = M @ values_source,
    where interpolation is exact for polynomials of degree len(source_pts)-1.

    Parameters
    ----------
    source_pts : Array
        Source interpolation nodes. Shape: (n_source,)
    target_pts : Array
        Target evaluation points. Shape: (n_target,)
    bkd : Backend
        Computational backend.

    Returns
    -------
    Array
        Interpolation matrix. Shape: (n_target, n_source)
    """
    n_source = source_pts.shape[0]
    n_target = target_pts.shape[0]

    interp_matrix = bkd.zeros((n_target, n_source))

    # Lagrange basis: l_j(x) = prod_{k != j} (x - x_k) / (x_j - x_k)
    for j in range(n_source):
        l_j = bkd.ones((n_target,))
        for k in range(n_source):
            if k != j:
                l_j = (
                    l_j * (target_pts - source_pts[k]) / (source_pts[j] - source_pts[k])
                )
        interp_matrix[:, j] = l_j

    return interp_matrix


class InterpolationOperator(Generic[Array]):
    """Interpolation operator between two sets of 1D points.

    Caches the interpolation matrix for efficient repeated use.

    Parameters
    ----------
    source_pts : Array
        Source interpolation nodes. Shape: (n_source,)
    target_pts : Array
        Target evaluation points. Shape: (n_target,)
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        source_pts: Array,
        target_pts: Array,
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._source_pts = source_pts
        self._target_pts = target_pts
        self._n_source = source_pts.shape[0]
        self._n_target = target_pts.shape[0]

        # Build and cache interpolation matrix
        self._matrix = lagrange_interpolation_matrix(source_pts, target_pts, bkd)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def n_source(self) -> int:
        """Return number of source points."""
        return self._n_source

    def n_target(self) -> int:
        """Return number of target points."""
        return self._n_target

    def matrix(self) -> Array:
        """Return the interpolation matrix.

        Returns
        -------
        Array
            Interpolation matrix. Shape: (n_target, n_source)
        """
        return self._matrix

    def apply(self, source_values: Array) -> Array:
        """Interpolate values from source to target points.

        Parameters
        ----------
        source_values : Array
            Values at source points. Shape: (n_source,)

        Returns
        -------
        Array
            Interpolated values at target points. Shape: (n_target,)
        """
        return self._matrix @ source_values

    def __repr__(self) -> str:
        return f"InterpolationOperator({self._n_source} -> {self._n_target} points)"


class RestrictionOperator(Generic[Array]):
    """Restriction operator: find best-fit coefficients from target values.

    Given values at target points, find source coefficients that minimize
    interpolation error. For overdetermined systems (n_target > n_source),
    uses least squares.

    Parameters
    ----------
    source_pts : Array
        Source nodes for the representation. Shape: (n_source,)
    target_pts : Array
        Points where values are given. Shape: (n_target,)
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        source_pts: Array,
        target_pts: Array,
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._source_pts = source_pts
        self._target_pts = target_pts
        self._n_source = source_pts.shape[0]
        self._n_target = target_pts.shape[0]

        # Build interpolation matrix: target = M @ source
        interp_matrix = lagrange_interpolation_matrix(source_pts, target_pts, bkd)

        # Build restriction matrix (pseudo-inverse of interp_matrix)
        # For square matrix: restriction = interp_matrix^(-1)
        # For overdetermined: restriction = (M^T M)^(-1) M^T (least squares)
        if self._n_target == self._n_source:
            self._matrix = bkd.inv(interp_matrix)
        else:
            # Least squares: (M^T M)^(-1) M^T
            MTM = interp_matrix.T @ interp_matrix
            self._matrix = bkd.inv(MTM) @ interp_matrix.T

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def n_source(self) -> int:
        """Return number of source points."""
        return self._n_source

    def n_target(self) -> int:
        """Return number of target points."""
        return self._n_target

    def matrix(self) -> Array:
        """Return the restriction matrix.

        Returns
        -------
        Array
            Restriction matrix. Shape: (n_source, n_target)
        """
        return self._matrix

    def apply(self, target_values: Array) -> Array:
        """Compute source coefficients from target values.

        Parameters
        ----------
        target_values : Array
            Values at target points. Shape: (n_target,)

        Returns
        -------
        Array
            Coefficients at source points. Shape: (n_source,)
        """
        return self._matrix @ target_values

    def __repr__(self) -> str:
        return f"RestrictionOperator({self._n_target} -> {self._n_source} points)"
