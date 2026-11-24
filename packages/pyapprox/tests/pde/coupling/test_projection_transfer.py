r"""Projection port transfers :math:`L = M_p^{-1} B` on the interface
x = 1/2 of a 2 x 1 partition, with linear hat traces on port nodes that
do not match the skeleton's.

Checks: L is the identity on coincident nodes and the interpolant when
the trace space contains the skeleton space; :math:`M_p` is SPD with a
condition number bounded in the port size; :math:`L^\top` keeps the sum
of a tested flux (the hats and the skeleton functions both sum to 1);
the composed map equals the dense :math:`M_p^{-1} B`; k = 2 components
act componentwise; the opaque mock stack, reached through its adapter,
and the dense backend stack, on either backend, give the same transfer
as the SciPy stack. The SciPy and mock stacks are NumPy-based, so only
the dense-stack test runs on both backends.
"""

from typing import Any, Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.pde.assembly.backend_dense_sink import BackendDenseMatrixAssembly
from pyapprox.pde.assembly.scipy_stack import ScipyAssemblyStack
from pyapprox.pde.coupling.basis import lobatto_lagrange_basis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.geometry import TensorBoxGeometry, skeleton_dof_points
from pyapprox.pde.coupling.ports import (
    AdaptedPortRows,
    BackendPortRows,
    PortAssemblyProtocol,
    PortMapProtocol,
    PortMatrixTransfer,
    RefuseCrossPoints,
    port_interpolation,
    port_projection,
    port_trace_coupling,
    port_trace_mass,
    subdomain_port_dofs,
)
from pyapprox.pde.coupling.spaces import (
    ComponentSkeletonSpace,
    ContinuousSkeletonSpace,
    UniformBases,
)
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.pde.coupling.traces import LinearSegmentTrace
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU
from pyapprox.util.linalg.operators import SpaceDescriptor
from pyapprox.util.linalg.scipy_stack import CSRFormat, ScipySparseLU
from tests._helpers.coupling import MockPortAdapter
from tests._helpers.linalg.mock_stack import (
    MockAssemblyStack,
    MockFormat,
    MockLU,
    MockVector,
    peek,
    wrap,
)

Np = NDArray[Any]
_BKD = NumpyBkd()
_TOPO = TensorBoxPartition((2, 1))
_GEOMETRY = TensorBoxGeometry(_TOPO, _BKD)
_SCIPY: PortAssemblyProtocol[Np, CSRFormat, Np, Np] = ScipyAssemblyStack(np.float64)


def _space(npoints: int) -> ContinuousSkeletonSpace[Np]:
    bases = UniformBases(lobatto_lagrange_basis(1, npoints, _BKD))
    return ContinuousSkeletonSpace(_TOPO, bases, VertexWeightNumbering(_TOPO, bases))


# The interface's faces; the same for every basis.
_FACES = [f for f in _TOPO.subdomain_entities(0) if f in _space(2).faces()]


def _trace(ys: Sequence[float], nquad: int = 3) -> LinearSegmentTrace[Np]:
    """Port nodes at x = 1/2 and ``ys``, in the order given."""
    y = np.asarray(ys, dtype=float)
    points = np.vstack([np.full(y.shape, 0.5), y])
    return LinearSegmentTrace(points, [int(i) for i in np.argsort(y)], nquad, _BKD)


# Port order deliberately not the order along the interface.
_YS = [0.71, 0.0, 0.4, 1.0, 0.13, 0.9, 0.55]


def _scipy_map(
    space: Any, trace: LinearSegmentTrace[Np], k: int = 1
) -> tuple[Any, Any, PortMapProtocol[Np]]:
    dofs = subdomain_port_dofs(_TOPO, space, 0)
    mass = port_trace_mass(trace, k, _SCIPY)
    coupling = port_trace_coupling(
        space, trace, _GEOMETRY, _FACES, dofs, RefuseCrossPoints(), _SCIPY
    )
    return mass, coupling, port_projection(mass, coupling, ScipySparseLU())


def _dense(local: PortMapProtocol[Np]) -> Np:
    return np.asarray(local.apply(np.eye(local.domain().size)))


def test_identity_on_coincident_nodes() -> None:
    space = _space(2)
    dofs = subdomain_port_dofs(_TOPO, space, 0)
    coords = skeleton_dof_points(
        space, _GEOMETRY, UniformBases(lobatto_lagrange_basis(1, 2, _BKD))
    )
    ys = [float(coords[1, d]) for d in dofs]
    _, _, local = _scipy_map(space, _trace(ys))
    np.testing.assert_allclose(_dense(local), np.eye(len(dofs)), atol=1e-13)


def test_equals_interpolation_when_the_trace_holds_the_skeleton() -> None:
    space = _space(2)
    trace = _trace(_YS)
    dofs = subdomain_port_dofs(_TOPO, space, 0)
    _, _, local = _scipy_map(space, trace)
    points = np.vstack([np.full(len(_YS), 0.5), np.asarray(_YS)])
    interpolant = port_interpolation(
        space, points, _GEOMETRY, _FACES, dofs, RefuseCrossPoints(), _SCIPY
    )
    np.testing.assert_allclose(
        _dense(local), interpolant.format().matrix().toarray(), atol=1e-13
    )


@pytest.mark.parametrize("nsegments", [4, 16, 64])
def test_trace_mass_is_spd_with_bounded_condition(nsegments: int) -> None:
    rng = np.random.default_rng(nsegments)
    # Interior nodes moved by up to h/4: segment lengths within a factor 3.
    ys = np.linspace(0.0, 1.0, nsegments + 1)
    ys[1:-1] += rng.uniform(-0.25, 0.25, nsegments - 1) / nsegments
    matrix = port_trace_mass(_trace(list(ys)), 1, _SCIPY).format().matrix().toarray()
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-15)
    eigs = np.linalg.eigvalsh(matrix)
    assert eigs.min() > 0.0
    # Gershgorin: eigenvalues lie in [h_min / 6, h_max], so the condition
    # number is at most 6 * 3, whatever the port size.
    assert eigs.max() / eigs.min() < 18.0


def test_transpose_keeps_the_sum_of_a_tested_flux() -> None:
    _, _, local = _scipy_map(_space(5), _trace(_YS))
    r = np.random.default_rng(0).normal(size=(len(_YS), 3))
    back = local.apply_transpose(r)
    np.testing.assert_allclose(back.sum(axis=0), r.sum(axis=0), rtol=1e-14, atol=1e-14)


def test_composed_map_equals_dense_inverse_times_coupling() -> None:
    mass, coupling, local = _scipy_map(_space(5), _trace(_YS))
    m = mass.format().matrix().toarray()
    b = coupling.format().matrix().toarray()
    dense = np.linalg.solve(m, b)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(b.shape[1], 2))
    y = rng.normal(size=(b.shape[0], 2))
    np.testing.assert_allclose(local.apply(x), dense @ x, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        local.apply_transpose(y), dense.T @ y, rtol=1e-12, atol=1e-13
    )


def test_components_act_componentwise() -> None:
    scalar = _space(4)
    trace = _trace(_YS)
    _, _, one = _scipy_map(scalar, trace)
    _, _, two = _scipy_map(ComponentSkeletonSpace(scalar, 2), trace, k=2)
    np.testing.assert_allclose(_dense(two), np.kron(_dense(one), np.eye(2)), atol=1e-13)


def test_mock_stack_through_its_adapter_matches_scipy() -> None:
    space = _space(5)
    trace = _trace(_YS)
    dofs = subdomain_port_dofs(_TOPO, space, 0)
    mock: PortAssemblyProtocol[MockVector, MockFormat, Np, Np] = MockAssemblyStack()
    mass = port_trace_mass(trace, 1, mock)
    coupling = port_trace_coupling(
        space, trace, _GEOMETRY, _FACES, dofs, RefuseCrossPoints(), mock
    )
    mock_local = port_projection(mass, coupling, MockLU())
    _, _, scipy_local = _scipy_map(space, trace)

    descriptor = space.descriptor()
    rows = BackendPortRows(descriptor, dofs, _BKD)
    port_rows = SpaceDescriptor("port-rows", len(dofs), "float64")
    mock_transfer = PortMatrixTransfer(
        AdaptedPortRows(rows, MockPortAdapter(), port_rows), mock_local
    )
    scipy_transfer = PortMatrixTransfer(rows, scipy_local)
    rng = np.random.default_rng(2)
    lam = rng.normal(size=(descriptor.size, 2))
    np.testing.assert_allclose(
        peek(mock_transfer.scatter(lam)), scipy_transfer.scatter(lam), rtol=1e-12
    )
    block = rng.normal(size=(len(_YS), 2))
    zeros = np.zeros((descriptor.size, 2))
    np.testing.assert_allclose(
        mock_transfer.gather_add(wrap(block, mock_local.range()), zeros),
        scipy_transfer.gather_add(block, zeros),
        rtol=1e-12,
    )


def test_dense_backend_stack_matches_scipy(bkd: Backend[Array]) -> None:
    topo = TensorBoxPartition((2, 1))
    bases = UniformBases(lobatto_lagrange_basis(1, 5, bkd))
    space = ContinuousSkeletonSpace(topo, bases, VertexWeightNumbering(topo, bases))
    points = bkd.stack([bkd.full((len(_YS),), 0.5), bkd.asarray(_YS)], axis=0)
    trace = LinearSegmentTrace(points, [int(i) for i in np.argsort(_YS)], 3, bkd)
    dofs = subdomain_port_dofs(topo, space, 0)
    dense = BackendDenseMatrixAssembly(bkd)
    mass = port_trace_mass(trace, 1, dense)
    coupling = port_trace_coupling(
        space,
        trace,
        TensorBoxGeometry(topo, bkd),
        _FACES,
        dofs,
        RefuseCrossPoints(),
        dense,
    )
    local = port_projection(mass, coupling, BackendDenseLU(bkd))
    _, _, reference = _scipy_map(_space(5), _trace(_YS))
    bkd.assert_allclose(
        local.apply(bkd.eye(len(dofs), dtype=bkd.real_dtype())),
        bkd.asarray(_dense(reference)),
        rtol=1e-12,
        atol=1e-13,
    )


class _BadTrace:
    def bkd(self) -> NumpyBkd:
        return _BKD

    def nnodes(self) -> int:
        return 3

    def element_nodes(self) -> Np:
        return np.asarray([[0, 1], [1, 2]], dtype=np.int64)

    def points(self) -> Np:
        return np.zeros((2, 4))

    def weights(self) -> Np:
        return np.ones((2, 2))

    def values(self) -> Np:
        return np.ones((2, 3, 2))


def test_refusals() -> None:
    with pytest.raises(ValueError):
        port_trace_mass(_BadTrace(), 1, _SCIPY)
    with pytest.raises(ValueError):
        port_trace_mass(_trace(_YS), 0, _SCIPY)
    with pytest.raises(TypeError):
        port_trace_mass(object(), 1, _SCIPY)  # type: ignore[arg-type]
    rows = BackendPortRows(SpaceDescriptor("skeleton", 4), [0, 1], _BKD)
    with pytest.raises(ValueError):
        AdaptedPortRows(rows, MockPortAdapter(), SpaceDescriptor("port-rows", 3))
    points = np.vstack([np.full(3, 0.5), np.asarray([0.0, 0.5, 1.0])])
    with pytest.raises(ValueError):
        LinearSegmentTrace(points, [0, 1], 2, _BKD)
    with pytest.raises(ValueError):
        LinearSegmentTrace(points, [0, 1, 2], 0, _BKD)
    with pytest.raises(ValueError):
        LinearSegmentTrace(points[:, [0, 0, 1]], [0, 1, 2], 2, _BKD)
