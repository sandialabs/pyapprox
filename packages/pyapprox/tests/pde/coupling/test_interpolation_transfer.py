"""The interpolation port transfer and its cross-point rules.

Checks are derivation-free: a skeleton function that is a polynomial of
the basis degree is reproduced at arbitrary port points; scatter and
gather are transposes; at points that are skeleton nodes the transfer is
the matching one; at a cross point the rules average or refuse; the
torch graph survives through both directions; the SciPy and the opaque
mock port stacks assemble the same map as the dense backend one.
"""

from dataclasses import dataclass
from typing import Any, Generic, Sequence

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from pyapprox.pde.assembly.backend_dense_sink import BackendDenseMatrixAssembly
from pyapprox.pde.assembly.scipy_stack import ScipyAssemblyStack
from pyapprox.pde.coupling.basis import gauss_lagrange_basis, lobatto_lagrange_basis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.geometry import TensorBoxGeometry, skeleton_dof_points
from pyapprox.pde.coupling.ports import (
    AdaptedPortRows,
    AverageIncidentFaces,
    BackendPortRows,
    CrossPointRuleProtocol,
    FaceValuesProtocol,
    MatchingNodalTransfer,
    PortAssemblyProtocol,
    PortMatrixTransfer,
    PortTransferProtocol,
    RefuseCrossPoints,
    port_interpolation,
    subdomain_port_dofs,
)
from pyapprox.pde.coupling.spaces import (
    ComponentSkeletonSpace,
    ContinuousSkeletonSpace,
    FacetSkeletonSpace,
    UniformBases,
)
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.linalg.backend_dense import (
    BackendDenseOperator,
    DenseFormat,
    backend_dtype_name,
)
from pyapprox.util.linalg.operators import ActionOperatorProtocol, SpaceDescriptor
from pyapprox.util.linalg.scipy_stack import CSRFormat
from tests._helpers.coupling import MockPortAdapter
from tests._helpers.linalg.mock_stack import (
    MockAssemblyStack,
    MockFormat,
    MockVector,
    peek,
    wrap,
)

Np = NDArray[Any]
_BOUNDS = [(0.0, 2.0), (-1.0, 1.0), (1.0, 4.0)]
_NPOINTS = 3


def _interpolation(
    space: FaceValuesProtocol[Array],
    points: Array,
    geometry: TensorBoxGeometry[Array],
    faces: Sequence[int],
    dofs: Sequence[int],
    rule: CrossPointRuleProtocol[Array],
) -> ActionOperatorProtocol[Array, DenseFormat[Array]]:
    """The interpolation map assembled densely on the space's backend."""
    assembly = BackendDenseMatrixAssembly(space.bkd())
    return port_interpolation(space, points, geometry, faces, dofs, rule, assembly)


def _matrix(local: ActionOperatorProtocol[Array, DenseFormat[Array]]) -> Array:
    """The stored matrix of a dense port map."""
    return local.format().matrix()


def _poly(bkd: Backend[Array], x: Array) -> Array:
    """A polynomial of degree _NPOINTS - 1 per direction. Shape: (n,)."""
    value = bkd.ones((x.shape[1],))
    for i in range(x.shape[0]):
        value = value * (0.5 + x[i] - 0.25 * x[i] ** (_NPOINTS - 1))
    return value


def _subdomain_faces(
    topo: TensorBoxPartition, faces: Sequence[int], subdomain: int
) -> list[int]:
    return [f for f in topo.subdomain_entities(subdomain) if f in faces]


def _port_points(
    bkd: Backend[Array],
    topo: TensorBoxPartition,
    geometry: TensorBoxGeometry[Array],
    faces: Sequence[int],
) -> Array:
    """Random points strictly inside each of ``faces``."""
    ref = bkd.asarray(
        np.random.default_rng(11).uniform(-0.95, 0.95, (topo.ndim() - 1, 3))
    )
    return bkd.concatenate([geometry.face_points(f, ref) for f in faces], axis=1)


def _continuous(
    bkd: Backend[Array], ncells: tuple[int, ...]
) -> tuple[TensorBoxPartition, ContinuousSkeletonSpace[Array], Array]:
    topo = TensorBoxPartition(ncells, bounds=_BOUNDS[: len(ncells)])
    bases = UniformBases(lobatto_lagrange_basis(len(ncells) - 1, _NPOINTS, bkd))
    space = ContinuousSkeletonSpace(topo, bases, VertexWeightNumbering(topo, bases))
    coords = skeleton_dof_points(space, TensorBoxGeometry(topo, bkd), bases)
    return topo, space, coords


@pytest.mark.parametrize("ncells", [(2, 2), (2, 2, 2)])
@pytest.mark.parametrize("k", [1, 2])
def test_continuous_reproduces_polynomials(
    bkd: Backend[Array], ncells: tuple[int, ...], k: int
) -> None:
    topo, scalar, coords = _continuous(bkd, ncells)
    space = ComponentSkeletonSpace(scalar, k)
    geometry = TensorBoxGeometry(topo, bkd)
    # Component c is (c + 1) * poly, interleaved node-major.
    nodal = _poly(bkd, coords)
    lam = bkd.reshape(bkd.stack([(c + 1.0) * nodal for c in range(k)], axis=1), (-1, 1))
    for s in range(topo.nsubdomains()):
        faces = _subdomain_faces(topo, scalar.faces(), s)
        points = _port_points(bkd, topo, geometry, faces)
        dofs = subdomain_port_dofs(topo, space, s)
        matrix = _interpolation(
            space, points, geometry, faces, dofs, AverageIncidentFaces()
        )
        transfer = PortMatrixTransfer(
            BackendPortRows(space.descriptor(), dofs, bkd), matrix
        )
        assert isinstance(transfer, PortTransferProtocol)
        assert transfer.nport() == k * points.shape[1]
        exact = _poly(bkd, points)
        expected = bkd.reshape(
            bkd.stack([(c + 1.0) * exact for c in range(k)], axis=1), (-1, 1)
        )
        bkd.assert_allclose(transfer.scatter(lam), expected, rtol=1e-12, atol=1e-13)


def test_gauss_facet_reproduces_polynomials(bkd: Backend[Array]) -> None:
    topo = TensorBoxPartition((2, 2), bounds=_BOUNDS[:2])
    basis = gauss_lagrange_basis(1, _NPOINTS, bkd)
    space = FacetSkeletonSpace(topo, UniformBases(basis))
    geometry = TensorBoxGeometry(topo, bkd)
    lam = bkd.zeros((space.descriptor().size, 1))
    for f in space.faces():
        on_face = bkd.asarray(list(space.entity_dofs(f)), dtype=bkd.int64_dtype())
        values = _poly(bkd, geometry.face_points(f, basis.nodes()))
        lam = bkd.index_update(lam, on_face, bkd.reshape(values, (-1, 1)))
    for s in range(topo.nsubdomains()):
        faces = _subdomain_faces(topo, space.faces(), s)
        points = _port_points(bkd, topo, geometry, faces)
        dofs = subdomain_port_dofs(topo, space, s)
        matrix = _interpolation(
            space, points, geometry, faces, dofs, RefuseCrossPoints()
        )
        transfer = PortMatrixTransfer(
            BackendPortRows(space.descriptor(), dofs, bkd), matrix
        )
        bkd.assert_allclose(
            transfer.scatter(lam),
            bkd.reshape(_poly(bkd, points), (-1, 1)),
            rtol=1e-12,
            atol=1e-13,
        )


def test_scatter_and_gather_are_transposes(bkd: Backend[Array]) -> None:
    topo, space, _ = _continuous(bkd, (2, 2, 2))
    geometry = TensorBoxGeometry(topo, bkd)
    faces = _subdomain_faces(topo, space.faces(), 3)
    points = _port_points(bkd, topo, geometry, faces)
    dofs = subdomain_port_dofs(topo, space, 3)
    matrix = _interpolation(
        space, points, geometry, faces, dofs, AverageIncidentFaces()
    )
    transfer = PortMatrixTransfer(
        BackendPortRows(space.descriptor(), dofs, bkd), matrix
    )
    rng = np.random.default_rng(13)
    lam = bkd.asarray(rng.normal(size=(space.descriptor().size, 2)))
    block = bkd.asarray(rng.normal(size=(transfer.nport(), 2)))
    zero = bkd.zeros((space.descriptor().size, 2))
    left = bkd.sum(transfer.scatter(lam) * block)
    right = bkd.sum(lam * transfer.gather_add(block, zero))
    bkd.assert_allclose(bkd.reshape(left, (1,)), bkd.reshape(right, (1,)), rtol=1e-14)


def test_at_skeleton_nodes_equals_matching(bkd: Backend[Array]) -> None:
    topo, space, coords = _continuous(bkd, (3, 2))
    geometry = TensorBoxGeometry(topo, bkd)
    s = 2
    faces = _subdomain_faces(topo, space.faces(), s)
    dofs = subdomain_port_dofs(topo, space, s)
    points = coords[:, list(dofs)]
    matrix = _interpolation(
        space, points, geometry, faces, dofs, AverageIncidentFaces()
    )
    descriptor = space.descriptor()
    transfer = PortMatrixTransfer(BackendPortRows(descriptor, dofs, bkd), matrix)
    matching = MatchingNodalTransfer(BackendPortRows(descriptor, dofs, bkd))
    lam = bkd.asarray(np.random.default_rng(17).normal(size=(descriptor.size, 1)))
    bkd.assert_allclose(transfer.scatter(lam), matching.scatter(lam), atol=1e-13)


@dataclass(frozen=True)
class _Center(Generic[Array]):
    """The 2x2 partition's center as a port node of subdomain 0."""

    geometry: TensorBoxGeometry[Array]
    point: Array
    # Subdomain 0's faces, the faces through the center, its port DOFs.
    faces: list[int]
    incident: list[int]
    dofs: list[int]


def _center(
    bkd: Backend[Array],
    space: ContinuousSkeletonSpace[Array] | FacetSkeletonSpace[Array],
) -> _Center[Array]:
    topo = TensorBoxPartition((2, 2), bounds=_BOUNDS[:2])
    geometry = TensorBoxGeometry(topo, bkd)
    point = bkd.asarray([[1.0], [0.0]])
    faces = _subdomain_faces(topo, space.faces(), 0)
    incident = [fp.face for fp in geometry.locate(point, faces)[0]]
    dofs = list(subdomain_port_dofs(topo, space, 0))
    return _Center(geometry, point, faces, incident, dofs)


def test_cross_point_rules(bkd: Backend[Array]) -> None:
    _, continuous, _ = _continuous(bkd, (2, 2))
    c = _center(bkd, continuous)
    assert len(c.incident) == 2
    average = _matrix(
        _interpolation(
            continuous, c.point, c.geometry, c.faces, c.dofs, AverageIncidentFaces()
        )
    )
    for face in c.incident:
        single = _matrix(
            _interpolation(
                continuous, c.point, c.geometry, [face], c.dofs, AverageIncidentFaces()
            )
        )
        bkd.assert_allclose(average, single, atol=1e-14)
    with pytest.raises(ValueError, match="several faces"):
        _interpolation(
            continuous, c.point, c.geometry, c.faces, c.dofs, RefuseCrossPoints()
        )

    topo = TensorBoxPartition((2, 2), bounds=_BOUNDS[:2])
    facet = FacetSkeletonSpace(
        topo, UniformBases(lobatto_lagrange_basis(1, _NPOINTS, bkd))
    )
    c = _center(bkd, facet)
    rows = [
        _matrix(
            _interpolation(
                facet, c.point, c.geometry, [face], c.dofs, RefuseCrossPoints()
            )
        )
        for face in c.incident
    ]
    average = _matrix(
        _interpolation(
            facet, c.point, c.geometry, c.faces, c.dofs, AverageIncidentFaces()
        )
    )
    bkd.assert_allclose(average, 0.5 * (rows[0] + rows[1]), atol=1e-14)
    assert bkd.to_float(bkd.max(bkd.abs(rows[0] - rows[1]))) > 0.5


def test_rejects_bad_input(bkd: Backend[Array]) -> None:
    topo, space, _ = _continuous(bkd, (2, 2))
    c = _center(bkd, space)
    dofs = c.dofs
    rule = AverageIncidentFaces[Array]()
    dense = BackendDenseMatrixAssembly(bkd)
    # Inside subdomain 0, on none of its faces.
    inside = bkd.asarray([[0.5], [-0.5]])
    with pytest.raises(ValueError, match="none of the faces"):
        port_interpolation(space, inside, c.geometry, c.faces, dofs, rule, dense)
    with pytest.raises(ValueError, match="at least one"):
        port_interpolation(
            space, bkd.zeros((2, 0)), c.geometry, c.faces, dofs, rule, dense
        )
    with pytest.raises(ValueError, match="outside"):
        port_interpolation(space, c.point, c.geometry, c.faces, dofs[1:], rule, dense)
    with pytest.raises(TypeError, match="FaceValuesProtocol"):
        port_interpolation(
            object(),  # type: ignore[arg-type]
            c.point,
            c.geometry,
            c.faces,
            dofs,
            rule,
            dense,
        )
    with pytest.raises(TypeError, match="CrossPointRuleProtocol"):
        port_interpolation(
            space,
            c.point,
            c.geometry,
            c.faces,
            dofs,
            object(),  # type: ignore[arg-type]
            dense,
        )
    with pytest.raises(TypeError, match="PointLocatorProtocol"):
        port_interpolation(
            space,
            c.point,
            object(),  # type: ignore[arg-type]
            c.faces,
            dofs,
            rule,
            dense,
        )
    with pytest.raises(TypeError, match="PortAssemblyProtocol"):
        port_interpolation(
            space,
            c.point,
            c.geometry,
            c.faces,
            dofs,
            rule,
            object(),  # type: ignore[arg-type]
        )
    rows = BackendPortRows(space.descriptor(), dofs, bkd)
    wide = _Blocked([bkd.zeros((2, len(dofs) + 1))], bkd)
    with pytest.raises(ValueError, match="domain has size"):
        PortMatrixTransfer(rows, wide)
    with pytest.raises(TypeError, match="PortRowsProtocol"):
        PortMatrixTransfer(object(), wide)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="PortMapProtocol"):
        PortMatrixTransfer(rows, bkd.zeros((2, len(dofs))))  # type: ignore[arg-type]


class _Blocked(Generic[Array]):
    """A port map stored as diagonal blocks, never as one matrix: a
    stand-in for a face-blocked map."""

    def __init__(self, blocks: Sequence[Array], bkd: Backend[Array]) -> None:
        self._blocks, self._bkd = list(blocks), bkd
        self._rows = [int(b.shape[0]) for b in blocks]
        self._cols = [int(b.shape[1]) for b in blocks]

    def domain(self) -> SpaceDescriptor:
        return SpaceDescriptor("port-rows", sum(self._cols))

    def range(self) -> SpaceDescriptor:
        return SpaceDescriptor("port", sum(self._rows))

    def _split(self, x: Array, sizes: Sequence[int]) -> list[Array]:
        starts = [sum(sizes[:i]) for i in range(len(sizes))]
        return [x[s : s + n] for s, n in zip(starts, sizes)]

    def apply(self, x: Array) -> Array:
        parts = self._split(x, self._cols)
        out = [b @ p for b, p in zip(self._blocks, parts)]
        return self._bkd.concatenate(out, axis=0)

    def apply_transpose(self, y: Array) -> Array:
        parts = self._split(y, self._rows)
        out = [b.T @ p for b, p in zip(self._blocks, parts)]
        return self._bkd.concatenate(out, axis=0)


def test_any_port_map_gives_the_same_transfer(bkd: Backend[Array]) -> None:
    # A two-block map agrees with the same map stored dense.
    topo, space, _ = _continuous(bkd, (2, 2))
    dofs = subdomain_port_dofs(topo, space, 0)
    n = len(dofs)
    rng = np.random.default_rng(19)
    top = bkd.asarray(rng.normal(size=(3, 2)))
    bottom = bkd.asarray(rng.normal(size=(4, n - 2)))
    dense = bkd.concatenate(
        [
            bkd.concatenate([top, bkd.zeros((3, n - 2))], axis=1),
            bkd.concatenate([bkd.zeros((4, 2)), bottom], axis=1),
        ],
        axis=0,
    )
    descriptor = space.descriptor()
    blocked = PortMatrixTransfer(
        BackendPortRows(descriptor, dofs, bkd), _Blocked([top, bottom], bkd)
    )
    stored = PortMatrixTransfer(
        BackendPortRows(descriptor, dofs, bkd),
        BackendDenseOperator(
            dense,
            SpaceDescriptor("port-rows", n, backend_dtype_name(bkd)),
            bkd,
            range=SpaceDescriptor("port", 7, backend_dtype_name(bkd)),
        ),
    )
    assert blocked.nport() == stored.nport() == 7
    lam = bkd.asarray(rng.normal(size=(descriptor.size, 2)))
    block = bkd.asarray(rng.normal(size=(7, 2)))
    bkd.assert_allclose(blocked.scatter(lam), stored.scatter(lam), rtol=1e-14)
    bkd.assert_allclose(
        blocked.gather_add(block, lam), stored.gather_add(block, lam), rtol=1e-14
    )


def test_transfer_keeps_the_torch_graph(torch_bkd: TorchBkd) -> None:
    bkd = torch_bkd
    _, space, _ = _continuous(bkd, (2, 2))
    c = _center(bkd, space)
    matrix = _interpolation(
        space, c.point, c.geometry, c.faces, c.dofs, AverageIncidentFaces()
    )
    transfer = PortMatrixTransfer(
        BackendPortRows(space.descriptor(), c.dofs, bkd), matrix
    )
    n = space.descriptor().size
    lam = torch.linspace(0.0, 1.0, n, dtype=torch.float64).reshape(n, 1)
    lam.requires_grad_()
    block = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    out = transfer.gather_add(block * transfer.scatter(lam), lam)
    dlam, dblock = torch.autograd.grad(out.sum(), [lam, block])
    # d/dlam [lam + E^T (b E lam)] summed = 1 + b E^T E 1-vector column sums.
    e_full = transfer.gather_add(torch.eye(1, dtype=torch.float64), torch.zeros(n, 1))
    expected = 1.0 + 2.0 * e_full * float(e_full.sum())
    bkd.assert_allclose(dlam, expected, rtol=1e-12)
    bkd.assert_allclose(
        dblock.reshape(1), (transfer.scatter(lam) * e_full.sum()).detach().reshape(1)
    )


def test_assembly_keeps_the_graph_through_the_port_nodes(torch_bkd: TorchBkd) -> None:
    # The interpolant reproduces _poly, so its derivative along the face
    # at a port node is the polynomial's.
    bkd = torch_bkd
    topo, space, coords = _continuous(bkd, (2, 2))
    geometry = TensorBoxGeometry(topo, bkd)
    faces = _subdomain_faces(topo, space.faces(), 0)
    dofs = subdomain_port_dofs(topo, space, 0)
    y = torch.tensor([[-0.3]], dtype=torch.float64, requires_grad=True)
    point = torch.cat([torch.ones((1, 1), dtype=torch.float64), y], dim=0)
    matrix = _interpolation(space, point, geometry, faces, dofs, RefuseCrossPoints())
    lam = bkd.reshape(_poly(bkd, coords), (-1, 1))
    value = PortMatrixTransfer(
        BackendPortRows(space.descriptor(), dofs, bkd), matrix
    ).scatter(lam)
    (dy,) = torch.autograd.grad(value.sum(), [y])
    # d/dy of (0.5 + x - x^2/4)(0.5 + y - y^2/4) at x = 1.
    bkd.assert_allclose(dy, 1.25 * (1.0 - 0.5 * y.detach()), rtol=1e-12)


def test_port_stacks_assemble_the_same_map() -> None:
    bkd = NumpyBkd()
    topo, scalar, _ = _continuous(bkd, (2, 2))
    space = ComponentSkeletonSpace(scalar, 2)
    geometry = TensorBoxGeometry(topo, bkd)
    faces = _subdomain_faces(topo, scalar.faces(), 0)
    points = _port_points(bkd, topo, geometry, faces)
    dofs = list(subdomain_port_dofs(topo, space, 0))
    rule = AverageIncidentFaces[Np]()
    dense = _matrix(_interpolation(space, points, geometry, faces, dofs, rule))

    scipy: PortAssemblyProtocol[Np, CSRFormat, Np, Np] = ScipyAssemblyStack(np.float64)
    sparse = port_interpolation(space, points, geometry, faces, dofs, rule, scipy)
    np.testing.assert_allclose(
        sparse.format().matrix().toarray(), dense, rtol=1e-14, atol=1e-15
    )

    mock: PortAssemblyProtocol[MockVector, MockFormat, Np, Np] = MockAssemblyStack()
    mock_local = port_interpolation(space, points, geometry, faces, dofs, rule, mock)
    descriptor = space.descriptor()
    rows = BackendPortRows(descriptor, dofs, bkd)
    port_rows = SpaceDescriptor("port-rows", len(dofs), "float64")
    mock_transfer = PortMatrixTransfer(
        AdaptedPortRows(rows, MockPortAdapter(), port_rows), mock_local
    )
    rng = np.random.default_rng(23)
    lam = rng.normal(size=(descriptor.size, 2))
    np.testing.assert_allclose(peek(mock_transfer.scatter(lam)), dense @ lam[dofs])
    block = rng.normal(size=(dense.shape[0], 2))
    expected = np.zeros((descriptor.size, 2))
    expected[dofs] = dense.T @ block
    np.testing.assert_allclose(
        mock_transfer.gather_add(wrap(block, mock_local.range()), np.zeros_like(lam)),
        expected,
        rtol=1e-12,
        atol=1e-14,
    )
