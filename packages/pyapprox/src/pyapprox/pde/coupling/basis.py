r"""Nodal bases on a reference face of the skeleton.

A face of a :math:`d`-dimensional partition has dimension
:math:`m = d - 1`, possibly 0. Its reference element is
:math:`[-1, 1]^m`; quadrature weights average over it (they sum to 1),
so a space scales them by each face's measure.
"""

import itertools
from typing import Generic

from pyapprox.pde.coupling.reference_faces import ReferenceBox
from pyapprox.polynomials.univariate.globalpoly.jacobi import (
    LegendrePolynomial1D,
)
from pyapprox.polynomials.univariate.lagrange import (
    univariate_lagrange_polynomial,
)
from pyapprox.util.backends.protocols import Array, Backend


def _legendre(npoints: int, bkd: Backend[Array]) -> LegendrePolynomial1D[Array]:
    if npoints < 1:
        raise ValueError(f"npoints must be >= 1, got {npoints}")
    poly = LegendrePolynomial1D(bkd)
    poly.set_nterms(max(npoints, 3))
    return poly


class TensorLagrangeBasis(Generic[Array]):
    r"""Tensor-product Lagrange basis on :math:`[-1, 1]^m`.

    The nodes are the tensor product of the given 1D nodes, the last
    direction varying fastest. The quadrature is Gauss–Legendre with as
    many points per direction as nodes, exact for products of two basis
    functions. :func:`lobatto_lagrange_basis` and
    :func:`gauss_lagrange_basis` build the two node families in use.

    Parameters
    ----------
    ndim
        The face dimension :math:`m`, at least 0.
    nodes_1d
        The nodes per direction, increasing in :math:`[-1, 1]`. Shape:
        (npoints,), at least one.
    closed
        Whether the first and last nodes are :math:`-1` and :math:`+1`,
        which puts basis functions on the face's closure for a
        continuous space to share between faces.
    bkd
        Computational backend.

    Raises
    ------
    ValueError
        If ``ndim`` is negative, ``nodes_1d`` is not a nonempty 1D
        array, or ``closed`` disagrees with its end nodes.
    """

    def __init__(
        self, ndim: int, nodes_1d: Array, closed: bool, bkd: Backend[Array]
    ) -> None:
        if ndim < 0:
            raise ValueError(f"ndim must be >= 0, got {ndim}")
        if nodes_1d.ndim != 1 or nodes_1d.shape[0] < 1:
            raise ValueError(
                f"nodes_1d must have shape (npoints,), got {tuple(nodes_1d.shape)}"
            )
        npoints = int(nodes_1d.shape[0])
        ends = (bkd.to_float(nodes_1d[0]), bkd.to_float(nodes_1d[-1]))
        if closed and (npoints < 2 or abs(ends[0] + 1.0) + abs(ends[1] - 1.0) > 1e-12):
            raise ValueError(f"closed nodes must start at -1 and end at 1, got {ends}")
        if not closed and (ends[0] <= -1.0 or ends[1] >= 1.0):
            raise ValueError(f"open nodes must lie inside (-1, 1), got {ends}")
        self._ndim = ndim
        self._npoints = npoints
        self._closed = closed
        self._bkd = bkd
        self._reference = ReferenceBox(ndim)
        gauss_x, gauss_w = _legendre(npoints, bkd).gauss_quadrature_rule(npoints)
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
        direction of closed nodes, the first and last are at -1 and +1,
        the others strictly inside; open nodes are all inside."""
        n = self._npoints

        def sign(index: int) -> int:
            if not self._closed:
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
            values = bkd.reshape(values[:, :, None] * factor[:, None, :], (npts, -1))
        return values

    def quadrature(self) -> tuple[Array, Array]:
        return self._quad_points, self._quad_weights


def lobatto_lagrange_basis(
    ndim: int, npoints: int, bkd: Backend[Array]
) -> TensorLagrangeBasis[Array]:
    """A :class:`TensorLagrangeBasis` on ``npoints`` Gauss–Lobatto nodes
    per direction: closed (the endpoints when ``npoints`` is 2), except
    the midpoint alone when ``npoints`` is 1."""
    poly = _legendre(npoints, bkd)
    if npoints == 2:
        nodes_1d = bkd.asarray([-1.0, 1.0])
    else:
        nodes_1d = bkd.flatten(poly.gauss_lobatto_quadrature_rule(npoints)[0])
    return TensorLagrangeBasis(ndim, nodes_1d, npoints >= 2, bkd)


def gauss_lagrange_basis(
    ndim: int, npoints: int, bkd: Backend[Array]
) -> TensorLagrangeBasis[Array]:
    """A :class:`TensorLagrangeBasis` on ``npoints`` Gauss–Legendre nodes
    per direction, all inside the face: its nodes are its quadrature
    points, so its mass is diagonal. Suits a facet-discontinuous space,
    whose faces share no nodes."""
    nodes_1d = bkd.flatten(_legendre(npoints, bkd).gauss_quadrature_rule(npoints)[0])
    return TensorLagrangeBasis(ndim, nodes_1d, False, bkd)
