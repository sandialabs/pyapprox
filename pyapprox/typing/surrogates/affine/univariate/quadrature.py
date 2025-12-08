"""Gaussian quadrature rules for orthonormal polynomial families.

This module provides quadrature rule classes that wrap orthonormal
polynomial families to compute Gaussian quadrature points and weights.
"""

from typing import Generic, Tuple, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.univariate.orthopoly_base import (
    OrthonormalPolynomial1D,
)


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
        self._cached_rules: dict = {}

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
        poly,  # JacobiPolynomial1D or subclass
        store: bool = False,
    ):
        self._poly = poly
        self._bkd = poly.bkd()
        self._store = store
        self._cached_rules: dict = {}

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
