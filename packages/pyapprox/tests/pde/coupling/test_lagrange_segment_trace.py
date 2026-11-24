r"""Degree-p Lagrange traces on straight segments.

Checks, on both backends: degree 1 is :class:`LinearSegmentTrace`; the
degree-2 element mass equals the closed form :math:`\frac{h}{30}
[[4, 2, -1], [2, 16, 2], [-1, 2, 4]]`; the trace functions reproduce a
quadratic from its nodal values at every quadrature point of a tilted
port whose nodes are not in port order. Refusals: mixed element lengths,
a repeated or missing node, nquad of zero, a zero-length segment, an
interior node off its equally spaced point.
"""

import numpy as np
import pytest

from pyapprox.pde.assembly.backend_dense_sink import BackendDenseMatrixAssembly
from pyapprox.pde.coupling.ports import port_trace_mass
from pyapprox.pde.coupling.traces import LagrangeSegmentTrace, LinearSegmentTrace
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend


def test_degree_one_is_the_linear_segment_trace(bkd: Backend[Array]) -> None:
    ys = [0.4, 0.0, 1.0, 0.7]
    points = bkd.stack([bkd.full((4,), 0.5), bkd.asarray(ys)], axis=0)
    chain = [1, 0, 3, 2]
    linear = LinearSegmentTrace(points, chain, 3, bkd)
    elements = [chain[e : e + 2] for e in range(3)]
    lagrange = LagrangeSegmentTrace(points, elements, 3, bkd)
    bkd.assert_allclose(lagrange.points(), linear.points(), rtol=1e-14)
    bkd.assert_allclose(lagrange.weights(), linear.weights(), rtol=1e-14)
    bkd.assert_allclose(lagrange.values(), linear.values(), rtol=1e-14, atol=1e-15)
    bkd.assert_allclose(
        bkd.asarray(lagrange.element_nodes(), dtype=bkd.real_dtype()),
        bkd.asarray(linear.element_nodes(), dtype=bkd.real_dtype()),
    )


def test_quadratic_element_mass_is_the_closed_form(bkd: Backend[Array]) -> None:
    h = 0.3
    points = bkd.asarray([[0.0, 0.5 * h, h], [0.0, 0.0, 0.0]])
    trace = LagrangeSegmentTrace(points, [[0, 1, 2]], 3, bkd)
    mass = port_trace_mass(trace, 1, BackendDenseMatrixAssembly(bkd))
    dense = mass.apply(bkd.eye(3, dtype=bkd.real_dtype()))
    expected = (h / 30.0) * bkd.asarray(
        [[4.0, 2.0, -1.0], [2.0, 16.0, 2.0], [-1.0, 2.0, 4.0]]
    )
    bkd.assert_allclose(dense, expected, rtol=1e-13)


def test_quadratic_trace_reproduces_quadratics(bkd: Backend[Array]) -> None:
    # Two tilted segments, nodes listed out of order along the port.
    s = bkd.asarray([0.5, 1.0, 0.0, 0.25, 0.75])
    points = bkd.stack([0.2 + 0.6 * s, 0.1 + 0.8 * s], axis=0)
    elements = [[2, 3, 0], [0, 4, 1]]
    trace = LagrangeSegmentTrace(points, elements, 3, bkd)

    def g(p: Array) -> Array:
        return 1.0 + 2.0 * p[0] - p[1] ** 2 + 3.0 * p[0] * p[1]

    nodal = g(points)
    nodes = trace.element_nodes()
    by_element = bkd.stack([nodal[nodes[e]] for e in range(2)], axis=0)
    interpolated = bkd.einsum("eqa,ea->eq", trace.values(), by_element)
    expected = bkd.reshape(g(trace.points()), (2, 3))
    bkd.assert_allclose(interpolated, expected, rtol=1e-13)
    # Weights sum to the port's length.
    bkd.assert_allclose(
        bkd.reshape(bkd.sum(trace.weights()), (1,)), bkd.asarray([1.0]), rtol=1e-14
    )


def test_refusals() -> None:
    bkd = NumpyBkd()
    points = np.vstack([np.zeros(3), np.asarray([0.0, 0.5, 1.0])])
    with pytest.raises(ValueError):
        LagrangeSegmentTrace(points[0], [[0, 1, 2]], 2, bkd)
    with pytest.raises(ValueError):
        LagrangeSegmentTrace(points, [[0, 1, 2], [1, 2]], 2, bkd)
    with pytest.raises(ValueError):
        LagrangeSegmentTrace(points, [], 2, bkd)
    with pytest.raises(ValueError):
        LagrangeSegmentTrace(points, [[0, 0, 2]], 2, bkd)
    with pytest.raises(ValueError):
        LagrangeSegmentTrace(points, [[0, 1]], 2, bkd)
    with pytest.raises(ValueError):
        LagrangeSegmentTrace(points, [[0, 1, 2]], 0, bkd)
    with pytest.raises(ValueError):
        LagrangeSegmentTrace(points[:, [0, 1, 0]], [[0, 1, 2]], 2, bkd)
    with pytest.raises(ValueError):
        LagrangeSegmentTrace(points[:, [0, 2, 1]], [[0, 1, 2]], 2, bkd)
