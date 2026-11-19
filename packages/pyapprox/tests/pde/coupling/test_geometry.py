"""Face geometry and the points of skeleton DOFs.

Checks are derivation-free: a face's reference corners land on its
extent's corners; every DOF point lies on its entity's extent, the
points are distinct, and points of DOFs on the physical boundary are on
it; faces disagreeing about a DOF's point are refused.
"""

from typing import Generic, Sequence

import pytest

from pyapprox.pde.coupling.basis import lobatto_lagrange_basis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.geometry import (
    FaceGeometryProtocol,
    TensorBoxGeometry,
    skeleton_dof_points,
)
from pyapprox.pde.coupling.spaces import ContinuousSkeletonSpace, UniformBases
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import SpaceDescriptor

_BOUNDS = [(0.0, 2.0), (-1.0, 1.0), (1.0, 4.0)]


def test_face_corners_land_on_extent(bkd: Backend[Array]) -> None:
    topo = TensorBoxPartition((2, 3), bounds=_BOUNDS[:2])
    geometry = TensorBoxGeometry(topo, bkd)
    assert isinstance(geometry, FaceGeometryProtocol)
    corners = bkd.asarray([[-1.0, 1.0]])
    for face in range(topo.nentities()):
        if topo.entity_dim(face) != 1:
            continue
        (xlo, xup), (ylo, yup) = topo.entity_bounds(face)
        expected = bkd.asarray([[xlo, xup], [ylo, yup]])
        bkd.assert_allclose(geometry.face_points(face, corners), expected, atol=1e-14)


@pytest.mark.parametrize(
    "ncells,npoints", [((3,), 1), ((2, 2), 4), ((3, 2), 3), ((2, 2, 2), 3)]
)
def test_dof_points(bkd: Backend[Array], ncells: tuple[int, ...], npoints: int) -> None:
    d = len(ncells)
    topo = TensorBoxPartition(ncells, bounds=_BOUNDS[:d])
    bases = UniformBases(lobatto_lagrange_basis(d - 1, npoints, bkd))
    space = ContinuousSkeletonSpace(topo, bases, VertexWeightNumbering(topo, bases))
    points = skeleton_dof_points(space, TensorBoxGeometry(topo, bkd), bases)
    size = space.descriptor().size
    assert tuple(points.shape) == (d, size)
    coords = [[bkd.to_float(points[i, k]) for i in range(d)] for k in range(size)]
    assert len({tuple(round(x, 10) for x in c) for c in coords}) == size
    for entity in range(topo.nentities()):
        extent = topo.entity_bounds(entity)
        for dof in space.entity_dofs(entity):
            for x, (lo, up) in zip(coords[dof], extent):
                assert lo - 1e-12 <= x <= up + 1e-12
            on_wall = any(
                abs(x - lo) < 1e-12 or abs(x - up) < 1e-12
                for x, (lo, up) in zip(coords[dof], _BOUNDS)
            )
            assert on_wall == topo.on_boundary(entity)


class _Shifted(Generic[Array]):
    """A box geometry that moves each face by its id: faces sharing a
    DOF disagree about its point."""

    def __init__(self, inner: TensorBoxGeometry[Array]) -> None:
        self._inner = inner

    def bkd(self) -> Backend[Array]:
        return self._inner.bkd()

    def ndim(self) -> int:
        return self._inner.ndim()

    def face_points(self, face: int, points: Array) -> Array:
        return self._inner.face_points(face, points) + float(face)


class _Missing:
    """A space with a DOF no face reads."""

    def descriptor(self) -> SpaceDescriptor:
        return SpaceDescriptor("skeleton", 2)

    def faces(self) -> Sequence[int]:
        return (0,)

    def face_dofs(self, face: int) -> tuple[int, ...]:
        return (0,)


def test_rejects_bad_input(bkd: Backend[Array]) -> None:
    topo = TensorBoxPartition((2, 2))
    basis = lobatto_lagrange_basis(1, 2, bkd)
    bases = UniformBases(basis)
    space = ContinuousSkeletonSpace(topo, bases, VertexWeightNumbering(topo, bases))
    geometry = TensorBoxGeometry(topo, bkd)
    with pytest.raises(ValueError, match="different points"):
        skeleton_dof_points(space, _Shifted(geometry), bases)
    line = TensorBoxPartition((2,))
    with pytest.raises(ValueError, match="no face"):
        skeleton_dof_points(
            _Missing(),
            TensorBoxGeometry(line, bkd),
            UniformBases(lobatto_lagrange_basis(0, 1, bkd)),
        )
    vertex = next(e for e in range(topo.nentities()) if topo.entity_dim(e) == 0)
    with pytest.raises(ValueError, match="not a face"):
        geometry.face_points(vertex, basis.nodes())
    with pytest.raises(ValueError, match="shape"):
        geometry.face_points(space.faces()[0], bkd.zeros((2, 1)))
    with pytest.raises(TypeError):
        skeleton_dof_points(object(), geometry, bases)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        skeleton_dof_points(space, object(), bases)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        skeleton_dof_points(space, geometry, object())  # type: ignore[arg-type]
