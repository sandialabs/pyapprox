r"""Mortar projection port transfers: a segment port on the interface
x = 1/2 of a 2 x 1 partition whose ends y = 0 and y = 1 are fixed, not
port nodes, read from a degree-4 continuous skeleton.

Checks, for P1 and P2 traces with nodes that do not match the
skeleton's, on both backends through the dense port stack: the test
functions sum to one at every point, as do the trial and end functions
together; the map reproduces a polynomial of the trace degree at the
port nodes (1e-12); a constant flux, given as its moments against the
trial functions, is tested exactly against every skeleton function off
the ends (1e-12), which plain projection onto the interior functions
cannot do; the SciPy stack gives the same map as the dense one.
Refusals: ends that are empty, repeated, out of range, inside an
element, or two in one element; element traces of mismatched shapes;
traces on different points; end points of the wrong shape.
"""

from typing import Any, Generic, Sequence, TypeVar

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.pde.assembly.backend_dense_sink import BackendDenseMatrixAssembly
from pyapprox.pde.assembly.scipy_stack import ScipyAssemblyStack
from pyapprox.pde.coupling.basis import lobatto_lagrange_basis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.geometry import TensorBoxGeometry, skeleton_dof_points
from pyapprox.pde.coupling.ports import (
    PortAssemblyProtocol,
    RefuseCrossPoints,
    port_projection,
    port_trace_coupling,
    port_trace_mass,
    port_trace_mortar_coupling,
    port_trace_pairing,
    subdomain_port_dofs,
)
from pyapprox.pde.coupling.spaces import ContinuousSkeletonSpace, UniformBases
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.pde.coupling.traces import (
    ElementTrace,
    LagrangeSegmentTrace,
    MortarSegmentTraces,
    mortar_segment_traces,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU, DenseFormat
from pyapprox.util.linalg.operators import LinearSolverProtocol
from pyapprox.util.linalg.operators.algebra import Product
from pyapprox.util.linalg.scipy_stack import CSRFormat, ScipySparseLU

Np = NDArray[Any]
P = TypeVar("P")
F = TypeVar("F")
_TOPO = TensorBoxPartition((2, 1))
# Vertices of the port's elements along y; P2 adds the midpoints.
_VERTICES = [0.0, 0.17, 0.4, 0.62, 0.81, 1.0]
_NQUAD = 5


def _port(degree: int) -> tuple[Np, list[list[int]], list[int]]:
    """Port node y-coordinates, elements and ends, nodes listed vertices
    first, then midpoints, so the ends are not the first and last nodes
    of the port order."""
    vertices = list(_VERTICES)
    nvert = len(vertices)
    if degree == 1:
        ys = vertices
        elements = [[e, e + 1] for e in range(nvert - 1)]
    else:
        mids = [0.5 * (a + b) for a, b in zip(vertices[:-1], vertices[1:])]
        ys = vertices + mids
        elements = [[e, nvert + e, e + 1] for e in range(nvert - 1)]
    return np.asarray(ys), elements, [nvert - 1, 0]


class _Setup(Generic[Array]):
    def __init__(self, bkd: Backend[Array], degree: int) -> None:
        bases = UniformBases(lobatto_lagrange_basis(1, 5, bkd))
        self.space = ContinuousSkeletonSpace(
            _TOPO, bases, VertexWeightNumbering(_TOPO, bases)
        )
        self.geometry = TensorBoxGeometry(_TOPO, bkd)
        self.coords = skeleton_dof_points(self.space, self.geometry, bases)
        self.faces = [f for f in _TOPO.subdomain_entities(0) if f in self.space.faces()]
        self.dofs = list(subdomain_port_dofs(_TOPO, self.space, 0))
        ys, self.elements, self.end_nodes = _port(degree)
        self.points = bkd.stack([bkd.full((len(ys),), 0.5), bkd.asarray(ys)], axis=0)
        self.traces: MortarSegmentTraces[Array] = mortar_segment_traces(
            self.points, self.elements, self.end_nodes, _NQUAD, bkd
        )
        self.bkd = bkd

    def local(
        self,
        assembly: PortAssemblyProtocol[P, F, Array, Array],
        solver: LinearSolverProtocol[P, F],
    ) -> Product[P]:
        t = self.traces
        pairing = port_trace_pairing(t.test, t.trial, 1, assembly)
        coupling = port_trace_mortar_coupling(
            self.space,
            t.test,
            t.ends,
            t.end_points,
            self.geometry,
            self.faces,
            self.dofs,
            RefuseCrossPoints(),
            assembly,
        )
        return port_projection(pairing, coupling, solver)

    def dense_local(self) -> Product[Array]:
        assembly: PortAssemblyProtocol[Array, DenseFormat[Array], Array, Array] = (
            BackendDenseMatrixAssembly(self.bkd)
        )
        solver: LinearSolverProtocol[Array, DenseFormat[Array]] = BackendDenseLU(
            self.bkd
        )
        return self.local(assembly, solver)

    def free(self) -> list[int]:
        """The positions in ``dofs`` of skeleton DOFs off the ends."""
        y = [self.bkd.to_float(self.coords[1, d]) for d in self.dofs]
        return [j for j, v in enumerate(y) if 1e-12 < v < 1.0 - 1e-12]


@pytest.mark.parametrize("degree", [1, 2])
def test_test_functions_sum_to_one(bkd: Backend[Array], degree: int) -> None:
    t = _Setup(bkd, degree).traces
    ones = bkd.ones(tuple(t.test.weights().shape))
    bkd.assert_allclose(bkd.sum(t.test.values(), axis=2), ones, rtol=1e-13)
    together = bkd.sum(t.trial.values(), axis=2) + bkd.sum(t.ends.values(), axis=2)
    bkd.assert_allclose(together, ones, rtol=1e-13)


@pytest.mark.parametrize("degree", [1, 2])
def test_reproduces_polynomials_of_the_trace_degree(
    bkd: Backend[Array], degree: int
) -> None:
    setup = _Setup(bkd, degree)

    def g(y: Array) -> Array:
        return 1.0 + 2.0 * y - 3.0 * y**2 if degree == 2 else 1.0 + 2.0 * y

    lam = bkd.reshape(g(setup.coords[1, setup.dofs]), (-1, 1))
    data = setup.dense_local().apply(lam)
    interior = bkd.asarray(setup.traces.interior, dtype=bkd.int64_dtype())
    expected = bkd.reshape(g(setup.points[1, interior]), (-1, 1))
    bkd.assert_allclose(data, expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("degree", [1, 2])
def test_a_constant_flux_is_tested_exactly(bkd: Backend[Array], degree: int) -> None:
    setup = _Setup(bkd, degree)
    trial = setup.traces.trial
    # r_i = int psi_i * 1, the moments a port returns for a unit flux.
    nint = trial.nnodes()
    slot_integrals = bkd.einsum("eq,eqa->ea", trial.weights(), trial.values())
    placement = bkd.asarray(
        [
            [
                [
                    1.0 if bkd.to_int(trial.element_nodes()[e, a]) == i else 0.0
                    for i in range(nint)
                ]
                for a in range(int(trial.element_nodes().shape[1]))
            ]
            for e in range(int(trial.element_nodes().shape[0]))
        ]
    )
    moments = bkd.reshape(bkd.einsum("ea,eai->i", slot_integrals, placement), (-1, 1))
    tested = setup.dense_local().apply_transpose(moments)
    # int phi_j over the port: the full trace's functions sum to one.
    full = LagrangeSegmentTrace(setup.points, setup.elements, _NQUAD, bkd)
    integrals = port_trace_coupling(
        setup.space,
        full,
        setup.geometry,
        setup.faces,
        setup.dofs,
        RefuseCrossPoints(),
        BackendDenseMatrixAssembly(bkd),
    ).apply_transpose(bkd.ones((full.nnodes(), 1)))
    free = bkd.asarray(setup.free(), dtype=bkd.int64_dtype())
    bkd.assert_allclose(tested[free], integrals[free], rtol=1e-12, atol=1e-14)
    # Control: plain projection onto the interior functions alone.
    dense = BackendDenseMatrixAssembly(bkd)
    plain = port_projection(
        port_trace_mass(trial, 1, dense),
        port_trace_coupling(
            setup.space,
            trial,
            setup.geometry,
            setup.faces,
            setup.dofs,
            RefuseCrossPoints(),
            dense,
        ),
        BackendDenseLU(bkd),
    ).apply_transpose(moments)
    miss = bkd.max(bkd.abs(plain[free] - integrals[free]))
    assert bkd.to_float(miss) > 1e-3


def test_scipy_stack_matches_the_dense_one() -> None:
    bkd = NumpyBkd()
    setup = _Setup(bkd, 2)
    stack: PortAssemblyProtocol[Np, CSRFormat, Np, Np] = ScipyAssemblyStack(np.float64)
    scipy_map = setup.local(stack, ScipySparseLU())
    eye = np.eye(len(setup.dofs))
    bkd.assert_allclose(
        np.asarray(scipy_map.apply(eye)),
        setup.dense_local().apply(eye),
        rtol=1e-12,
        atol=1e-13,
    )


def _refused_ends(ends: Sequence[int], elements: Sequence[Sequence[int]]) -> None:
    bkd = NumpyBkd()
    points = np.vstack([np.full(3, 0.5), np.asarray([0.0, 0.5, 1.0])])
    with pytest.raises(ValueError):
        mortar_segment_traces(points, elements, ends, 3, bkd)


def test_refusals() -> None:
    bkd = NumpyBkd()
    chain = [[0, 1], [1, 2]]
    _refused_ends([], chain)
    _refused_ends([0, 0], chain)
    _refused_ends([3], chain)
    _refused_ends([1], chain)
    # One element holding both ends.
    _refused_ends([0, 2], [[0, 1, 2]])
    points = np.vstack([np.full(3, 0.5), np.asarray([0.0, 0.5, 1.0])])
    traces = mortar_segment_traces(points, chain, [0, 2], 3, bkd)
    test = traces.test
    with pytest.raises(ValueError):
        ElementTrace(
            test.points(),
            test.weights(),
            test.values(),
            test.element_nodes()[:1],
            test.nnodes(),
            bkd,
        )
    with pytest.raises(ValueError):
        ElementTrace(
            test.points(), test.weights(), test.values(), test.element_nodes(), 0, bkd
        )
    shifted = ElementTrace(
        test.points() + 0.1,
        test.weights(),
        test.values(),
        test.element_nodes(),
        test.nnodes(),
        bkd,
    )
    dense = BackendDenseMatrixAssembly(bkd)
    with pytest.raises(ValueError):
        port_trace_pairing(test, shifted, 1, dense)
    with pytest.raises(ValueError):
        port_trace_pairing(test, traces.trial, 0, dense)
    setup = _Setup(bkd, 1)
    with pytest.raises(ValueError):
        port_trace_mortar_coupling(
            setup.space,
            setup.traces.test,
            setup.traces.ends,
            setup.traces.end_points[:, :1],
            setup.geometry,
            setup.faces,
            setup.dofs,
            RefuseCrossPoints(),
            dense,
        )
