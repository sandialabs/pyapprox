"""The interpolation port transfer and its cross-point rules.

Checks are derivation-free: a skeleton function that is a polynomial of
the basis degree is reproduced at arbitrary port points; scatter and
gather are transposes; at points that are skeleton nodes the transfer is
the matching one; at a cross point the rules average or refuse; the
torch graph survives through both directions.
"""

from typing import Generic, Sequence

import numpy as np
import pytest
import torch

from pyapprox.pde.coupling.basis import gauss_lagrange_basis, lobatto_lagrange_basis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.geometry import (
    FacePoint,
    TensorBoxGeometry,
    skeleton_dof_points,
)
from pyapprox.pde.coupling.ports import (
    AverageIncidentFaces,
    BackendPortRows,
    MatchingNodalTransfer,
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
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.linalg.backend_dense import BackendDenseOperator, backend_dtype_name
from pyapprox.util.linalg.operators import SpaceDescriptor

_BOUNDS = [(0.0, 2.0), (-1.0, 1.0), (1.0, 4.0)]
_NPOINTS = 3


def _matrix(local: BackendDenseOperator[Array]) -> Array:
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
        matrix = port_interpolation(
            space, geometry.locate(points, faces), dofs, AverageIncidentFaces()
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
        matrix = port_interpolation(
            space, geometry.locate(points, faces), dofs, RefuseCrossPoints()
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
    matrix = port_interpolation(
        space, geometry.locate(points, faces), dofs, AverageIncidentFaces()
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
    matrix = port_interpolation(
        space, geometry.locate(points, faces), dofs, AverageIncidentFaces()
    )
    descriptor = space.descriptor()
    transfer = PortMatrixTransfer(BackendPortRows(descriptor, dofs, bkd), matrix)
    matching = MatchingNodalTransfer(BackendPortRows(descriptor, dofs, bkd))
    lam = bkd.asarray(np.random.default_rng(17).normal(size=(descriptor.size, 1)))
    bkd.assert_allclose(transfer.scatter(lam), matching.scatter(lam), atol=1e-13)


def _center_rows(
    bkd: Backend[Array],
    space: ContinuousSkeletonSpace[Array] | FacetSkeletonSpace[Array],
) -> tuple[list[FacePoint[Array]], list[int], list[int]]:
    """The 2x2 partition's center located on subdomain 0's faces, those
    faces and subdomain 0's port DOFs."""
    topo = TensorBoxPartition((2, 2), bounds=_BOUNDS[:2])
    faces = _subdomain_faces(topo, space.faces(), 0)
    center = bkd.asarray([[1.0], [0.0]])
    located = TensorBoxGeometry(topo, bkd).locate(center, faces)[0]
    return list(located), faces, list(subdomain_port_dofs(topo, space, 0))


def test_cross_point_rules(bkd: Backend[Array]) -> None:
    _, continuous, _ = _continuous(bkd, (2, 2))
    located, _, dofs = _center_rows(bkd, continuous)
    assert len(located) == 2
    average = _matrix(
        port_interpolation(continuous, [located], dofs, AverageIncidentFaces())
    )
    for fp in located:
        single = _matrix(
            port_interpolation(continuous, [[fp]], dofs, AverageIncidentFaces())
        )
        bkd.assert_allclose(average, single, atol=1e-14)
    with pytest.raises(ValueError, match="several faces"):
        port_interpolation(continuous, [located], dofs, RefuseCrossPoints())

    topo = TensorBoxPartition((2, 2), bounds=_BOUNDS[:2])
    facet = FacetSkeletonSpace(
        topo, UniformBases(lobatto_lagrange_basis(1, _NPOINTS, bkd))
    )
    located, _, dofs = _center_rows(bkd, facet)
    rows = [
        _matrix(port_interpolation(facet, [[fp]], dofs, RefuseCrossPoints()))
        for fp in located
    ]
    average = _matrix(
        port_interpolation(facet, [located], dofs, AverageIncidentFaces())
    )
    bkd.assert_allclose(average, 0.5 * (rows[0] + rows[1]), atol=1e-14)
    assert bkd.to_float(bkd.max(bkd.abs(rows[0] - rows[1]))) > 0.5


def test_rejects_bad_input(bkd: Backend[Array]) -> None:
    topo, space, _ = _continuous(bkd, (2, 2))
    located, _, dofs = _center_rows(bkd, space)
    rule = AverageIncidentFaces[Array]()
    with pytest.raises(ValueError, match="no face"):
        port_interpolation(space, [[]], dofs, rule)
    with pytest.raises(ValueError, match="at least one"):
        port_interpolation(space, [], dofs, rule)
    with pytest.raises(ValueError, match="outside"):
        port_interpolation(space, [located], dofs[1:], rule)
    with pytest.raises(TypeError):
        port_interpolation(object(), [located], dofs, rule)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        port_interpolation(space, [located], dofs, object())  # type: ignore[arg-type]
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
    located, _, dofs = _center_rows(bkd, space)
    matrix = port_interpolation(space, [located], dofs, AverageIncidentFaces())
    transfer = PortMatrixTransfer(
        BackendPortRows(space.descriptor(), dofs, bkd), matrix
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
