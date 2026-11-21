r"""Port traces: a subdomain's trace functions on its port and their
quadrature, the data the projection builders of
:mod:`pyapprox.pde.coupling.ports` read.
"""

from typing import Generic, Sequence

from pyapprox.polynomials.univariate.globalpoly.jacobi import LegendrePolynomial1D
from pyapprox.util.backends.protocols import Array, Backend


class LumpedNodalTrace(Generic[Array]):
    r"""Nodal trace functions with lumped quadrature: each port node is an
    element with one trace function, :math:`\phi = 1`, and one quadrature
    point there, weighted by the node's quadrature weight.

    A flux tested against it is :math:`r_n = w_n q(x_n)`. The ends of a
    1D domain are points, of unit weight. A collocation face's nodes carry
    the face quadrature weights at those nodes (Clenshaw–Curtis on
    Chebyshev–Gauss–Lobatto nodes), so the moments are the flux tested
    against the face's nodal interpolants under that quadrature.

    Parameters
    ----------
    points
        The physical point of each port node, in port order. Shape:
        (ndim, nnodes), at least one node.
    weights
        Each node's quadrature weight in the physical measure. Shape:
        (nnodes,).
    bkd
        Computational backend.

    Raises
    ------
    ValueError
        If ``points`` is not 2D with at least one column, or ``weights``
        does not have one entry per node.
    """

    def __init__(self, points: Array, weights: Array, bkd: Backend[Array]) -> None:
        if points.ndim != 2 or points.shape[1] < 1:
            raise ValueError(
                f"points must have shape (ndim, nnodes), got {tuple(points.shape)}"
            )
        n = int(points.shape[1])
        if tuple(weights.shape) != (n,):
            raise ValueError(
                f"weights must have shape ({n},), got {tuple(weights.shape)}"
            )
        self._points, self._nnodes, self._bkd = points, n, bkd
        self._weights = bkd.reshape(weights, (n, 1))
        self._nodes = bkd.asarray([[i] for i in range(n)], dtype=bkd.int64_dtype())

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
        return self._bkd.ones((self._nnodes, 1, 1))


class TabulatedTrace(Generic[Array]):
    r"""Trace functions given by their values at the port's quadrature
    points, as one element holding every port node.

    Suits a spectral subdomain whose trace functions are global on each of
    its sides: the side's Lagrange polynomials, tabulated at a side rule's
    points, the blocks of several sides placed side by side (a function is
    zero on a side it does not belong to). With a rule exact for products
    of two of them, the moments are the flux's nodal interpolant tested
    exactly; with the nodes as the points, the values are 0/1 and the
    moments those of :class:`LumpedNodalTrace`.

    The single element is dense, points times nodes.

    Parameters
    ----------
    points
        The physical quadrature points. Shape: (ndim, npoints).
    weights
        Their weights in the physical measure. Shape: (npoints,).
    values
        Each trace function at each point, columns in port order. Shape:
        (npoints, nnodes), at least one node.
    bkd
        Computational backend.

    Raises
    ------
    ValueError
        If the shapes disagree or there is no node.
    """

    def __init__(
        self, points: Array, weights: Array, values: Array, bkd: Backend[Array]
    ) -> None:
        if values.ndim != 2 or values.shape[1] < 1:
            raise ValueError(
                f"values must have shape (npoints, nnodes), got {tuple(values.shape)}"
            )
        npoints, nnodes = int(values.shape[0]), int(values.shape[1])
        if points.ndim != 2 or points.shape[1] != npoints:
            raise ValueError(
                f"points must have shape (ndim, {npoints}), got {tuple(points.shape)}"
            )
        if tuple(weights.shape) != (npoints,):
            raise ValueError(
                f"weights must have shape ({npoints},), got {tuple(weights.shape)}"
            )
        self._points, self._nnodes, self._bkd = points, nnodes, bkd
        self._weights = bkd.reshape(weights, (1, npoints))
        self._values = bkd.reshape(values, (1, npoints, nnodes))
        self._nodes = bkd.asarray([list(range(nnodes))], dtype=bkd.int64_dtype())

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


class LinearSegmentTrace(Generic[Array]):
    r"""Linear hat trace functions on port nodes joined by straight
    segments: the trace of a P1 or Q1 mesh on a port that is a curve, as
    in a 2D problem. Higher-degree meshes take
    :class:`LagrangeSegmentTrace`; surface ports need other traces
    satisfying the same protocol.

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


class LagrangeSegmentTrace(Generic[Array]):
    r"""Degree-:math:`p` Lagrange trace functions on straight segments: the
    trace of a degree-:math:`p` Lagrange mesh on a port that is a curve,
    as in a 2D problem, e.g. the Q2 edges of a quadrilateral mesh.

    Element :math:`e` lists its :math:`p + 1` port nodes in order along
    the segment, ends first and last, the others equally spaced between
    them. Its trace functions are the Lagrange polynomials on the nodes
    :math:`s_j = j / p` of :math:`s \in [0, 1]`, integrated by
    ``nquad``-point Gauss–Legendre, exact for polynomials of degree
    :math:`2\,\text{nquad} - 1` along the segment. Weights are in arc
    length.

    Parameters
    ----------
    points
        The physical point of each port node, in port order. Shape:
        (ndim, nnodes).
    elements
        Each segment's port nodes in order, all of length :math:`p + 1
        \ge 2`; every port node in at least one.
    nquad
        Quadrature points per segment, at least 1.
    bkd
        Computational backend.

    Raises
    ------
    ValueError
        If ``points`` is not 2D, the elements are empty, differ in length,
        repeat a node within one, or do not name exactly the port nodes;
        ``nquad`` is not positive; a segment has zero length; or an
        interior node is not at its equally spaced point.
    """

    def __init__(
        self,
        points: Array,
        elements: Sequence[Sequence[int]],
        nquad: int,
        bkd: Backend[Array],
    ) -> None:
        if points.ndim != 2:
            raise ValueError(
                f"points must have shape (ndim, nnodes), got {tuple(points.shape)}"
            )
        nnodes = int(points.shape[1])
        lengths_seen = sorted({len(e) for e in elements})
        if len(lengths_seen) != 1 or lengths_seen[0] < 2:
            raise ValueError(
                "elements must be nonempty, all of one length p + 1 >= 2, got "
                f"lengths {lengths_seen}"
            )
        if any(len(set(e)) != len(e) for e in elements):
            raise ValueError("an element repeats a node")
        named = {n for e in elements for n in e}
        if named != set(range(nnodes)):
            raise ValueError(
                f"the elements must name exactly the {nnodes} port nodes, got "
                f"{sorted(named)}"
            )
        if nquad < 1:
            raise ValueError(f"nquad must be >= 1, got {nquad}")
        degree = lengths_seen[0] - 1
        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(max(nquad, 3))
        t, w = poly.gauss_quadrature_rule(nquad)
        t, w = bkd.flatten(t), bkd.flatten(w)
        # Gauss weights of the probability measure on [-1, 1] sum to 1.
        w = w / bkd.sum(w)
        s = (t + 1.0) / 2.0
        table = bkd.asarray([list(e) for e in elements], dtype=bkd.int64_dtype())
        a, b = points[:, table[:, 0]], points[:, table[:, -1]]
        lengths = bkd.sqrt(bkd.sum((b - a) ** 2, axis=0))
        if bkd.to_float(bkd.min(lengths)) <= 0.0:
            raise ValueError("a segment has zero length")
        nodes = [j / degree for j in range(degree + 1)]
        for j in range(1, degree):
            gap = points[:, table[:, j]] - (a + nodes[j] * (b - a))
            relative = bkd.sqrt(bkd.sum(gap**2, axis=0)) / lengths
            if bkd.to_float(bkd.max(relative)) > 1e-10:
                raise ValueError(
                    f"node {j} of an element is not at its equally spaced point"
                )
        # (ndim, nelems, nquad), element-major when flattened.
        located = a[:, :, None] + (b - a)[:, :, None] * s[None, None, :]
        self._points = bkd.reshape(located, (int(points.shape[0]), -1))
        self._weights = lengths[:, None] * w[None, :]
        functions = []
        for j in range(degree + 1):
            value = bkd.ones_like(s)
            for m in range(degree + 1):
                if m != j:
                    value = value * (s - nodes[m]) / (nodes[j] - nodes[m])
            functions.append(value)
        lagrange = bkd.stack(functions, axis=1)
        self._values = bkd.tile(lagrange[None, :, :], (len(elements), 1, 1))
        self._nodes = table
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
