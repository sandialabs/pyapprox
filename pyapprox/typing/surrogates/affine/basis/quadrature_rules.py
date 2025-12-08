"""Tensor product quadrature rules for multivariate integration.

This module provides quadrature rules for computing integrals of
multivariate functions using tensor products of univariate rules.
"""

from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Tuple, Callable

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import (
    OrthonormalPolynomial1DProtocol,
)


class QuadratureRule(ABC, Generic[Array]):
    """Abstract base class for quadrature rules.

    A quadrature rule provides points and weights for numerical integration:
        ∫ f(x) dμ(x) ≈ Σ_i w_i f(x_i)
    """

    @abstractmethod
    def __call__(self) -> Tuple[Array, Array]:
        """Return quadrature points and weights.

        Returns
        -------
        points : Array
            Quadrature points. Shape depends on rule dimensionality.
        weights : Array
            Quadrature weights. Shape: (npoints,)
        """
        raise NotImplementedError


class TensorProductQuadratureRule(QuadratureRule[Array], Generic[Array]):
    """Tensor product of univariate quadrature rules.

    Constructs a multivariate quadrature rule from univariate rules
    by taking their tensor product.

    Parameters
    ----------
    bases_1d : List[OrthonormalPolynomial1DProtocol[Array]]
        List of orthonormal polynomial bases providing quadrature rules.
    npoints_1d : List[int]
        Number of quadrature points per dimension.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        bases_1d: List[OrthonormalPolynomial1DProtocol[Array]],
        npoints_1d: List[int],
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._bases_1d = bases_1d
        self._npoints_1d = npoints_1d

        if len(bases_1d) != len(npoints_1d):
            raise ValueError(
                f"Length mismatch: {len(bases_1d)} bases vs {len(npoints_1d)} npoints"
            )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return len(self._bases_1d)

    def npoints(self) -> int:
        """Return the total number of quadrature points."""
        total = 1
        for n in self._npoints_1d:
            total *= n
        return total

    def __call__(self) -> Tuple[Array, Array]:
        """Return tensor product quadrature points and weights.

        Returns
        -------
        points : Array
            Quadrature points. Shape: (nvars, npoints_total)
        weights : Array
            Quadrature weights. Shape: (npoints_total,)
        """
        points_1d = []
        weights_1d = []

        for dd in range(self.nvars()):
            # Ensure basis has enough terms for quadrature
            basis = self._bases_1d[dd]
            if basis.nterms() < self._npoints_1d[dd]:
                basis.set_nterms(self._npoints_1d[dd])
            pts, wts = basis.gauss_quadrature_rule(self._npoints_1d[dd])
            points_1d.append(pts[0, :])  # Remove first dim
            weights_1d.append(self._bkd.flatten(wts))  # Flatten weights

        # Build tensor product grid
        grids = self._bkd.meshgrid(tuple(points_1d), indexing="ij")
        points = self._bkd.stack(
            [self._bkd.flatten(g) for g in grids], axis=0
        )

        # Tensor product of weights
        weights = weights_1d[0]
        for dd in range(1, self.nvars()):
            weights = self._bkd.flatten(
                self._bkd.reshape(weights, (-1, 1)) * weights_1d[dd]
            )

        return points, weights

    def integrate(self, func: Callable[[Array], Array]) -> Array:
        """Integrate a function using this quadrature rule.

        Parameters
        ----------
        func : Callable[[Array], Array]
            Function to integrate. Takes samples (nvars, nsamples) and
            returns values (nsamples, nqoi).

        Returns
        -------
        Array
            Integral values. Shape: (nqoi,)
        """
        points, weights = self()
        values = func(points)  # (npoints, nqoi)
        return self._bkd.dot(weights, values)

    def __repr__(self) -> str:
        return (
            f"TensorProductQuadratureRule(nvars={self.nvars()}, "
            f"npoints_1d={self._npoints_1d})"
        )


class FixedTensorProductQuadratureRule(QuadratureRule[Array], Generic[Array]):
    """Tensor product quadrature with pre-computed points and weights.

    This is a lightweight wrapper for storing pre-computed quadrature
    points and weights.

    Parameters
    ----------
    points : Array
        Quadrature points. Shape: (nvars, npoints)
    weights : Array
        Quadrature weights. Shape: (npoints,)
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        points: Array,
        weights: Array,
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._points = points
        self._weights = weights

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._points.shape[0]

    def npoints(self) -> int:
        """Return the number of quadrature points."""
        return self._points.shape[1]

    def __call__(self) -> Tuple[Array, Array]:
        """Return the stored quadrature points and weights."""
        return self._points, self._weights

    def integrate(self, func: Callable[[Array], Array]) -> Array:
        """Integrate a function using this quadrature rule."""
        values = func(self._points)
        return self._bkd.dot(self._weights, values)

    def __repr__(self) -> str:
        return (
            f"FixedTensorProductQuadratureRule(nvars={self.nvars()}, "
            f"npoints={self.npoints()})"
        )
