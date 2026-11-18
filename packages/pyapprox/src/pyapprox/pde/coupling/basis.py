r"""Nodal bases on a reference face of the skeleton.

A face of a :math:`d`-dimensional partition has dimension
:math:`m = d - 1`, possibly 0. Its reference element is
:math:`[-1, 1]^m`; quadrature weights average over it (they sum to 1),
so a space scales them by each face's measure.
"""

import itertools
from typing import Generic

from pyapprox.pde.coupling.reference_faces import ReferenceBox
from pyapprox.surrogates.affine.univariate.globalpoly.jacobi import (
    LegendrePolynomial1D,
)
from pyapprox.surrogates.affine.univariate.lagrange import (
    univariate_lagrange_polynomial,
)
from pyapprox.util.backends.protocols import Array, Backend


class TensorLagrangeBasis(Generic[Array]):
    r"""Tensor-product Lagrange basis on :math:`[-1, 1]^m`.

    The nodes are the tensor product of ``npoints`` Gauss–Lobatto points
    per direction (the endpoints when ``npoints`` is 2, the midpoint when
    it is 1), the last direction varying fastest. Lobatto nodes put
    basis functions on the face's closure, which a continuous space
    shares between faces. The quadrature is Gauss–Legendre with
    ``npoints`` points per direction, exact for products of two basis
    functions.

    Parameters
    ----------
    ndim
        The face dimension :math:`m`, at least 0.
    npoints
        Nodes per direction, at least 1.
    bkd
        Computational backend.
    """

    def __init__(self, ndim: int, npoints: int, bkd: Backend[Array]) -> None:
        if ndim < 0 or npoints < 1:
            raise ValueError(
                f"ndim must be >= 0 and npoints >= 1, got {ndim} and {npoints}"
            )
        self._ndim = ndim
        self._npoints = npoints
        self._bkd = bkd
        self._reference = ReferenceBox(ndim)
        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(max(npoints, 3))
        if npoints == 2:
            nodes_1d = bkd.asarray([-1.0, 1.0])
        else:
            nodes_1d = bkd.flatten(poly.gauss_lobatto_quadrature_rule(npoints)[0])
        gauss_x, gauss_w = poly.gauss_quadrature_rule(npoints)
        self._nodes_1d = nodes_1d
        self._nodes = self._tensor_points(nodes_1d)
        self._quad_points = self._tensor_points(bkd.flatten(gauss_x))
        weights = bkd.ones((1,))
        for _ in range(ndim):
            weights = bkd.kron(weights, bkd.flatten(gauss_w))
        self._quad_weights = bkd.reshape(weights, (weights.shape[0], 1))

    def _tensor_points(self, points_1d: Array) -> Array:
        bkd, n = self._bkd, points_1d.shape[0]
        rows = [
            bkd.kron(
                bkd.kron(bkd.ones((n**j,)), points_1d),
                bkd.ones((n ** (self._ndim - 1 - j),)),
            )
            for j in range(self._ndim)
        ]
        if not rows:
            return bkd.zeros((0, 1))
        return bkd.stack(rows, axis=0)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def ndim(self) -> int:
        return self._ndim

    def nterms(self) -> int:
        return int(self._nodes.shape[1])

    def nodes(self) -> Array:
        """The nodes on the reference face. Shape: (ndim, nterms)."""
        return self._nodes

    def reference(self) -> ReferenceBox:
        """The reference face the nodes lie on."""
        return self._reference

    def node_entities(self) -> tuple[int, ...]:
        """The sub-entity of :meth:`reference` each node lies on, in node
        order. Read from integer node indices, not coordinates: per
        direction, the first and last of ``npoints`` nodes are at -1 and
        +1, the others strictly inside."""
        n = self._npoints

        def sign(index: int) -> int:
            if n == 1:
                return 0
            return -1 if index == 0 else 1 if index == n - 1 else 0

        return tuple(
            self._reference.subentity([sign(i) for i in multi])
            for multi in itertools.product(range(n), repeat=self._ndim)
        )

    def evaluate(self, points: Array) -> Array:
        if points.ndim != 2 or points.shape[0] != self._ndim:
            raise ValueError(
                f"points must have shape ({self._ndim}, npoints), "
                f"got {tuple(points.shape)}"
            )
        bkd, npts = self._bkd, points.shape[1]
        values = bkd.ones((npts, 1))
        for j in range(self._ndim):
            factor = univariate_lagrange_polynomial(self._nodes_1d, points[j], bkd)
            values = bkd.reshape(
                values[:, :, None] * factor[:, None, :], (npts, -1)
            )
        return values

    def quadrature(self) -> tuple[Array, Array]:
        return self._quad_points, self._quad_weights
