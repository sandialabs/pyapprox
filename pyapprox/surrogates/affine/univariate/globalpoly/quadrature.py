"""Quadrature rules for polynomial interpolation and integration.

This module provides quadrature rule classes including:
- GaussQuadratureRule: Gaussian quadrature based on orthonormal polynomials
- GaussLobattoQuadratureRule: Gauss-Lobatto quadrature including endpoints
- ClenshawCurtisQuadratureRule: Clenshaw-Curtis quadrature on [-1, 1]
"""

import math
from typing import Any, Generic, Tuple

from pyapprox.surrogates.affine.univariate.globalpoly.orthopoly_base import (
    OrthonormalPolynomial1D,
)
from pyapprox.util.backends.protocols import Array, Backend


class GaussQuadratureRule(Generic[Array]):
    """Gaussian quadrature rule based on an orthonormal polynomial.

    This class wraps an orthonormal polynomial to provide quadrature rules.
    Points and weights can optionally be cached for repeated use.

    Parameters
    ----------
    poly : OrthonormalPolynomial1D[Array]
        Orthonormal polynomial to use for quadrature.
    store : bool, optional
        If True, cache computed quadrature rules. Default False.
    """

    def __init__(
        self,
        poly: OrthonormalPolynomial1D[Array],
        store: bool = False,
    ):
        self._poly = poly
        self._bkd = poly.bkd()
        self._store = store
        self._cached_rules: dict[str, Any] = {}

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def __call__(self, npoints: int) -> Tuple[Array, Array]:
        """Compute quadrature points and weights.

        Parameters
        ----------
        npoints : int
            Number of quadrature points.

        Returns
        -------
        points : Array
            Quadrature points. Shape: (1, npoints)
        weights : Array
            Quadrature weights. Shape: (npoints, 1)
        """
        if self._store and npoints in self._cached_rules:
            return self._cached_rules[npoints]

        # Ensure polynomial has enough terms
        if self._poly.nterms() < npoints:
            self._poly.set_nterms(npoints)

        points, weights = self._poly.gauss_quadrature_rule(npoints)

        if self._store:
            self._cached_rules[npoints] = (points, weights)

        return points, weights

    def __repr__(self) -> str:
        return f"GaussQuadratureRule(poly={self._poly})"


class GaussLobattoQuadratureRule(Generic[Array]):
    """Gauss-Lobatto quadrature rule including endpoints.

    Only works with Jacobi polynomial families which support
    Gauss-Lobatto quadrature.

    Parameters
    ----------
    poly : JacobiPolynomial1D[Array]
        Jacobi polynomial (or subclass) to use for quadrature.
    store : bool, optional
        If True, cache computed quadrature rules. Default False.
    """

    def __init__(
        self,
        poly: OrthonormalPolynomial1D[Array],
        store: bool = False,
    ) -> None:
        self._poly = poly
        self._bkd = poly.bkd()
        self._store = store
        self._cached_rules: dict[str, Any] = {}

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def __call__(self, npoints: int) -> Tuple[Array, Array]:
        """Compute quadrature points and weights.

        Parameters
        ----------
        npoints : int
            Number of quadrature points (must be >= 3).

        Returns
        -------
        points : Array
            Quadrature points. Shape: (1, npoints)
        weights : Array
            Quadrature weights. Shape: (npoints, 1)
        """
        if self._store and npoints in self._cached_rules:
            return self._cached_rules[npoints]

        # Ensure polynomial has enough terms
        if self._poly.nterms() < npoints:
            self._poly.set_nterms(npoints)

        points, weights = self._poly.gauss_lobatto_quadrature_rule(npoints)

        if self._store:
            self._cached_rules[npoints] = (points, weights)

        return points, weights

    def __repr__(self) -> str:
        return f"GaussLobattoQuadratureRule(poly={self._poly})"


def _is_power_of_two(integer: int) -> bool:
    """Check if an integer is a power of two."""
    return (integer & (integer - 1) == 0) and integer != 0


class ClenshawCurtisQuadratureRule(Generic[Array]):
    """Clenshaw-Curtis quadrature rule on [-1, 1].

    This quadrature rule uses Chebyshev extrema as nodes and is nested,
    meaning points from level l are a subset of points at level l+1.

    The growth rule is:
    - level 0: 1 point (midpoint)
    - level l (l >= 1): 2^l + 1 points

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    store : bool, optional
        If True, cache computed quadrature rules. Default False.
    prob_measure : bool, optional
        If True, weights integrate to 1 (probability measure).
        If False, weights integrate to 2 (Lebesgue measure on [-1, 1]).
        Default True.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        store: bool = False,
        prob_measure: bool = True,
    ):
        self._bkd = bkd
        self._store = store
        self._prob_measure = prob_measure
        self._cached_rules: dict[str, Any] = {}

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def _growth_rule(self, level: int) -> int:
        """Return number of points for a given level."""
        if level == 0:
            return 1
        return 2**level + 1

    def _hierarchical_to_nodal_index(self, level: int, ll: int, ii: int) -> int:
        """Convert hierarchical index to nodal index in quadrature rule."""
        nindices = self._growth_rule(level)
        # mid point
        if ll == 0:
            return nindices // 2
        # boundaries
        if ll == 1:
            if ii == 0:
                return 0
            return nindices - 1
        # higher level points
        return (2 * ii + 1) * 2 ** (level - ll)

    def _poly_indices_to_quad_rule_indices(self, level: int) -> Array:
        """Convert polynomial indices to quadrature rule indices.

        Returns indices for polynomial ordering: midpoint first, then
        boundaries, then fill in remaining points level by level.
        """
        quad_rule_indices = []
        nprevious_hier_indices = 0
        for ll in range(level + 1):
            nhierarchical_indices = self._growth_rule(ll) - nprevious_hier_indices
            for ii in range(nhierarchical_indices):
                quad_index = self._hierarchical_to_nodal_index(level, ll, ii)
                quad_rule_indices.append(quad_index)
            nprevious_hier_indices += nhierarchical_indices
        return self._bkd.asarray(quad_rule_indices, dtype=int)

    def _unordered_pts_wts(self, level: int) -> Tuple[Array, Array]:
        """Compute points and weights in natural (not polynomial) order."""
        nsamples = self._growth_rule(level)

        if level == 0:
            return self._bkd.zeros((1,)), self._bkd.ones((1,))

        # Chebyshev extrema: x_j = -cos(pi * j / (n-1))
        jj = self._bkd.arange(nsamples)
        x = -self._bkd.cos(math.pi * jj / (nsamples - 1.0))

        # Fix numerical precision at boundaries and midpoint using concatenation
        # (avoids in-place assignment which isn't supported in all backends)
        mid = nsamples // 2
        x = self._bkd.concatenate(
            [
                self._bkd.asarray([-1.0]),
                x[1:mid],
                self._bkd.asarray([0.0]),
                x[mid + 1 : -1],
                self._bkd.asarray([1.0]),
            ]
        )

        # Compute weights using Clenshaw-Curtis formula
        wt_factor = 1.0 / 2.0
        boundary_wt = wt_factor / (nsamples * (nsamples - 2.0))

        # Interior weights
        jj = self._bkd.arange(1, nsamples - 1)
        kk = self._bkd.arange(1, (nsamples - 3) // 2 + 1)
        mysum = self._bkd.sum(
            1.0
            / (4.0 * kk[:, None] ** 2 - 1.0)
            * self._bkd.cos(
                2.0 * math.pi * kk[:, None] * jj[None, :] / (nsamples - 1.0)
            ),
            axis=0,
        )
        interior_wts = wt_factor * (
            2.0
            / float(nsamples - 1.0)
            * (
                1.0
                - self._bkd.cos(math.pi * jj) / (nsamples * (nsamples - 2.0))
                - 2.0 * mysum
            )
        )

        # Build weights array using concatenation
        w = self._bkd.concatenate(
            [
                self._bkd.asarray([boundary_wt]),
                interior_wts,
                self._bkd.asarray([boundary_wt]),
            ]
        )

        return x, w

    def _ordered_pts_wts(self, level: int) -> Tuple[Array, Array]:
        """Return points and weights in polynomial ordering.

        The first point is the midpoint, then boundaries, then remaining
        points are filled in level by level.
        """
        x, w = self._unordered_pts_wts(level)
        quad_indices = self._poly_indices_to_quad_rule_indices(level)
        ordered_samples = x[quad_indices]
        ordered_weights = w[quad_indices]
        return ordered_samples, ordered_weights

    def npoints_from_level(self, level: int) -> int:
        """Return number of points for a given level."""
        return self._growth_rule(level)

    def level_from_npoints(self, npoints: int) -> int:
        """Return level for a given number of points.

        Parameters
        ----------
        npoints : int
            Number of quadrature points. Must be 1 or 2^l + 1 for some l >= 1.

        Returns
        -------
        level : int
            The level corresponding to npoints.

        Raises
        ------
        ValueError
            If npoints is not a valid Clenshaw-Curtis point count.
        """
        if npoints == 1:
            return 0
        if not _is_power_of_two(npoints - 1):
            raise ValueError(
                f"npoints must be 1 or 2^l + 1 for some l >= 1, got {npoints}"
            )
        return int(round(math.log(npoints - 1, 2), 0))

    def __call__(self, npoints: int) -> Tuple[Array, Array]:
        """Compute quadrature points and weights.

        Parameters
        ----------
        npoints : int
            Number of quadrature points. Must be 1 or 2^l + 1 for some l >= 1.

        Returns
        -------
        points : Array
            Quadrature points in polynomial ordering. Shape: (1, npoints)
        weights : Array
            Quadrature weights. Shape: (npoints, 1)
        """
        if self._store and npoints in self._cached_rules:
            return self._cached_rules[npoints]

        level = self.level_from_npoints(npoints)
        quad_samples, quad_weights = self._ordered_pts_wts(level)

        if not self._prob_measure:
            # Scale weights to integrate to 2 (Lebesgue measure on [-1, 1])
            quad_weights = quad_weights * 2.0

        # Reshape to standard convention: (1, npoints) and (npoints, 1)
        points = self._bkd.reshape(quad_samples, (1, -1))
        weights = self._bkd.reshape(quad_weights, (-1, 1))

        if self._store:
            self._cached_rules[npoints] = (points, weights)

        return points, weights

    def __repr__(self) -> str:
        return f"ClenshawCurtisQuadratureRule(prob_measure={self._prob_measure})"
