r"""Port traces: a subdomain's trace functions on its port and their
quadrature, the data the projection builders of
:mod:`pyapprox.pde.coupling.ports` read.
"""

from typing import Generic, Sequence

from pyapprox.polynomials.univariate.globalpoly.jacobi import LegendrePolynomial1D
from pyapprox.util.backends.protocols import Array, Backend


class LinearSegmentTrace(Generic[Array]):
    r"""Linear hat trace functions on port nodes joined by straight
    segments: the trace of a P1 or Q1 mesh on a port that is a curve, as
    in a 2D problem. Higher-degree meshes and surface ports need other
    traces satisfying the same protocol.

    Element :math:`e` is the segment from node ``chain[e]`` to node
    ``chain[e + 1]``; its two hats are :math:`(1 - t)/2` and
    :math:`(1 + t)/2` at :math:`t \in [-1, 1]`, integrated by
    ``nquad``-point Gauss–Legendre, exact for polynomials of degree
    :math:`2\,\text{nquad} - 1` along the segment. Weights are in arc
    length.

    Parameters
    ----------
    points
        The physical point of each port node, in port order. Shape:
        (ndim, nnodes).
    chain
        Every port node once, in order along the curve.
    nquad
        Quadrature points per segment, at least 1.
    bkd
        Computational backend.

    Raises
    ------
    ValueError
        If ``points`` is not 2D, ``chain`` is not a permutation of the
        port nodes with at least two entries, ``nquad`` is not positive,
        or a segment has zero length.
    """

    def __init__(
        self, points: Array, chain: Sequence[int], nquad: int, bkd: Backend[Array]
    ) -> None:
        if points.ndim != 2:
            raise ValueError(
                f"points must have shape (ndim, nnodes), got {tuple(points.shape)}"
            )
        nnodes = int(points.shape[1])
        if nnodes < 2 or sorted(chain) != list(range(nnodes)):
            raise ValueError(
                f"chain must order all {nnodes} port nodes (at least two), "
                f"got {tuple(chain)}"
            )
        if nquad < 1:
            raise ValueError(f"nquad must be >= 1, got {nquad}")
        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(max(nquad, 3))
        t, w = poly.gauss_quadrature_rule(nquad)
        t, w = bkd.flatten(t), bkd.flatten(w)
        # Gauss weights of the probability measure on [-1, 1] sum to 1.
        w = w / bkd.sum(w)
        start, end = list(chain[:-1]), list(chain[1:])
        a, b = points[:, start], points[:, end]
        lengths = bkd.sqrt(bkd.sum((b - a) ** 2, axis=0))
        if bkd.to_float(bkd.min(lengths)) <= 0.0:
            raise ValueError("a segment of the chain has zero length")
        s = (t + 1.0) / 2.0
        # (ndim, nelems, nquad), element-major when flattened.
        located = a[:, :, None] + (b - a)[:, :, None] * s[None, None, :]
        self._points = bkd.reshape(located, (int(points.shape[0]), -1))
        self._weights = lengths[:, None] * w[None, :]
        hats = bkd.stack([1.0 - s, s], axis=1)
        self._values = bkd.tile(hats[None, :, :], (len(start), 1, 1))
        self._nodes = bkd.asarray(
            [[i, j] for i, j in zip(start, end)], dtype=bkd.int64_dtype()
        )
        self._nnodes = nnodes
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nnodes(self) -> int:
        return self._nnodes

    def element_nodes(self) -> Array:
        return self._nodes

    def points(self) -> Array:
        return self._points

    def weights(self) -> Array:
        return self._weights

    def values(self) -> Array:
        return self._values
