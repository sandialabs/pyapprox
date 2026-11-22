"""Chebyshev 1D nodal bases for collocation: Gauss-Lobatto and Gauss.

Each satisfies ``CollocationBasis1DProtocol``: it makes its own nodes and
differentiates on them with its family's closed-form barycentric
weights, so its derivative formula never meets another family's nodes.
Interpolation to other points uses weights computed from the nodes.
"""

from typing import Generic

from pyapprox.polynomials.univariate.chebyshev_barycentric import (
    chebyshev_gauss_lobatto_nodes,
    chebyshev_gauss_lobatto_weights,
    chebyshev_gauss_nodes,
    chebyshev_gauss_weights,
)
from pyapprox.polynomials.univariate.lagrange import (
    lagrange_differentiation_matrix,
    univariate_lagrange_polynomial,
)
from pyapprox.util.backends.protocols import Array, Backend


class ChebyshevGaussLobatto1D(Generic[Array]):
    """Chebyshev-Gauss-Lobatto nodes :math:`-\\cos(\\pi j / (n - 1))`,
    both ends included; at least 2."""

    def nodes(self, npts: int, bkd: Backend[Array]) -> Array:
        """``npts`` nodes from -1 to 1. Shape: ``(npts,)``"""
        return chebyshev_gauss_lobatto_nodes(npts, bkd)

    def derivative_matrix(self, npts: int, bkd: Backend[Array]) -> Array:
        """The derivative of the interpolant on the ``npts`` nodes.
        Shape: ``(npts, npts)``"""
        return lagrange_differentiation_matrix(
            self.nodes(npts, bkd), bkd, chebyshev_gauss_lobatto_weights(npts, bkd)
        )

    def interpolation_matrix(
        self, npts: int, points: Array, bkd: Backend[Array]
    ) -> Array:
        """The interpolant on the ``npts`` nodes at ``points``, shape
        ``(npoints,)``. Shape: ``(npoints, npts)``"""
        return univariate_lagrange_polynomial(self.nodes(npts, bkd), points, bkd)

    def __repr__(self) -> str:
        return "ChebyshevGaussLobatto1D()"


class ChebyshevGauss1D(Generic[Array]):
    """Chebyshev-Gauss nodes :math:`-\\cos(\\pi (2 j + 1) / (2 n))`, the
    roots of :math:`T_n`; at least 1. They lie inside :math:`(-1, 1)`,
    so a direction on them has no boundary. A staggered pressure grid
    uses them."""

    def nodes(self, npts: int, bkd: Backend[Array]) -> Array:
        """``npts`` nodes inside (-1, 1), increasing. Shape: ``(npts,)``"""
        return chebyshev_gauss_nodes(npts, bkd)

    def derivative_matrix(self, npts: int, bkd: Backend[Array]) -> Array:
        """The derivative of the interpolant on the ``npts`` nodes; zero
        for one node. Shape: ``(npts, npts)``"""
        return lagrange_differentiation_matrix(
            self.nodes(npts, bkd), bkd, chebyshev_gauss_weights(npts, bkd)
        )

    def interpolation_matrix(
        self, npts: int, points: Array, bkd: Backend[Array]
    ) -> Array:
        """The interpolant on the ``npts`` nodes at ``points``, shape
        ``(npoints,)``. Shape: ``(npoints, npts)``"""
        return univariate_lagrange_polynomial(self.nodes(npts, bkd), points, bkd)

    def __repr__(self) -> str:
        return "ChebyshevGauss1D()"
