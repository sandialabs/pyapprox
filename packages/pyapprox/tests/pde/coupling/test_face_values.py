"""Skeleton functions on a face at given points, and locating physical
points on faces, both backends.

Checks are derivation-free: a nodal basis's values at its nodes are the
gluing (identity on the face's DOFs); a skeleton function is single
valued at a cross point on a continuous space and not on a facet one; a
polynomial of the basis degree is reproduced at arbitrary points;
component values are the scalar ones times unit vectors; ``locate``
inverts ``face_points`` and finds every face meeting at a cross point.
"""

import numpy as np
import pytest

from pyapprox.pde.coupling.basis import gauss_lagrange_basis, lobatto_lagrange_basis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.geometry import TensorBoxGeometry
from pyapprox.pde.coupling.spaces import (
    ComponentSkeletonSpace,
    ContinuousSkeletonSpace,
    FacetSkeletonSpace,
    UniformBases,
)
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.util.backends.protocols import Array, Backend

_BOUNDS = [(0.0, 2.0), (-1.0, 1.0), (1.0, 4.0)]


def _continuous(
    bkd: Backend[Array], ncells: tuple[int, ...], npoints: int
) -> tuple[TensorBoxPartition, ContinuousSkeletonSpace[Array]]:
    topo = TensorBoxPartition(ncells, bounds=_BOUNDS[: len(ncells)])
    bases = UniformBases(lobatto_lagrange_basis(len(ncells) - 1, npoints, bkd))
    space = ContinuousSkeletonSpace(topo, bases, VertexWeightNumbering(topo, bases))
    return topo, space


@pytest.mark.parametrize("ncells,npoints", [((2, 2), 3), ((2, 2, 2), 3)])
def test_values_at_nodes_are_the_gluing(
    bkd: Backend[Array], ncells: tuple[int, ...], npoints: int
) -> None:
    _, space = _continuous(bkd, ncells, npoints)
    nodes = lobatto_lagrange_basis(len(ncells) - 1, npoints, bkd).nodes()
    for face in space.faces():
        fv = space.face_values(face, nodes)
        assert fv.dofs == space.face_dofs(face)
        assert tuple(fv.values.shape) == (nodes.shape[1], len(fv.dofs), 1)
        bkd.assert_allclose(fv.values[:, :, 0], bkd.eye(len(fv.dofs)), atol=1e-12)


def _values_at_center(
    bkd: Backend[Array],
    space: ContinuousSkeletonSpace[Array] | FacetSkeletonSpace[Array],
    topo: TensorBoxPartition,
    coeffs: Array,
) -> list[float]:
    """A skeleton function's value at the partition's center, read
    through each face meeting there."""
    center = bkd.asarray([[(lo + up) / 2.0] for lo, up in _BOUNDS[: topo.ndim()]])
    located = TensorBoxGeometry(topo, bkd).locate(center, space.faces())[0]
    values = []
    for fp in located:
        fv = space.face_values(fp.face, fp.point)
        dofs = bkd.asarray(list(fv.dofs), dtype=bkd.int64_dtype())
        values.append(bkd.to_float(bkd.sum(fv.values[0, :, 0] * coeffs[dofs])))
    return values


@pytest.mark.parametrize("ncells", [(2, 2), (2, 2, 2)])
def test_cross_point_single_valued_only_when_continuous(
    bkd: Backend[Array], ncells: tuple[int, ...]
) -> None:
    rng = np.random.default_rng(3)
    topo, continuous = _continuous(bkd, ncells, 3)
    nfaces = 4 if len(ncells) == 2 else 12
    coeffs = bkd.asarray(rng.normal(size=continuous.descriptor().size))
    values = _values_at_center(bkd, continuous, topo, coeffs)
    assert len(values) == nfaces
    assert max(values) - min(values) < 1e-12
    facet = FacetSkeletonSpace(
        topo, UniformBases(lobatto_lagrange_basis(len(ncells) - 1, 3, bkd))
    )
    coeffs = bkd.asarray(rng.normal(size=facet.descriptor().size))
    values = _values_at_center(bkd, facet, topo, coeffs)
    assert len(values) == nfaces
    assert max(values) - min(values) > 1e-3


@pytest.mark.parametrize("ndim,npoints", [(2, 3), (3, 2), (3, 4)])
def test_gauss_facet_reproduces_polynomials(
    bkd: Backend[Array], ndim: int, npoints: int
) -> None:
    # Interpolate x^a y^b (a, b < npoints) at the Gauss nodes; evaluating
    # the interpolant anywhere on the face gives the polynomial back.
    topo = TensorBoxPartition((2,) * ndim, bounds=_BOUNDS[:ndim])
    basis = gauss_lagrange_basis(ndim - 1, npoints, bkd)
    space = FacetSkeletonSpace(topo, UniformBases(basis))

    def poly(x: Array) -> Array:
        value = bkd.ones((x.shape[1],))
        for i in range(x.shape[0]):
            value = value * x[i] ** (npoints - 1 - i % npoints)
        return value

    points = bkd.asarray(np.random.default_rng(5).uniform(-1, 1, (ndim - 1, 9)))
    for face in space.faces():
        assert space.face_values(face, points).dofs == space.entity_dofs(face)
        coeffs = poly(basis.nodes())
        values = space.face_values(face, points).values[:, :, 0]
        bkd.assert_allclose(bkd.dot(values, coeffs), poly(points), rtol=1e-12)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_component_values_are_scalar_times_unit_vectors(
    bkd: Backend[Array], k: int
) -> None:
    _, scalar = _continuous(bkd, (2, 2), 3)
    space = ComponentSkeletonSpace(scalar, k)
    assert space.faces() == scalar.faces()
    points = bkd.asarray(np.random.default_rng(7).uniform(-1, 1, (1, 5)))
    for face in space.faces():
        inner = scalar.face_values(face, points)
        fv = space.face_values(face, points)
        assert fv.dofs == tuple(k * d + c for d in inner.dofs for c in range(k))
        assert tuple(fv.values.shape) == (5, k * len(inner.dofs), k)
        for c in range(k):
            for cc in range(k):
                expected = inner.values[:, :, 0] * float(c == cc)
                bkd.assert_allclose(fv.values[:, c::k, cc], expected, atol=1e-15)


@pytest.mark.parametrize("ncells", [(3, 2), (2, 2, 2)])
def test_locate_inverts_face_points(
    bkd: Backend[Array], ncells: tuple[int, ...]
) -> None:
    topo, space = _continuous(bkd, ncells, 2)
    geometry = TensorBoxGeometry(topo, bkd)
    d = len(ncells)
    ref = bkd.asarray(np.random.default_rng(9).uniform(-0.9, 0.9, (d - 1, 4)))
    for face in space.faces():
        located = geometry.locate(geometry.face_points(face, ref), space.faces())
        assert len(located) == 4
        for j, found in enumerate(located):
            assert [fp.face for fp in found] == [face]
            bkd.assert_allclose(found[0].point, ref[:, j : j + 1], atol=1e-14)


def test_locate_refuses(bkd: Backend[Array]) -> None:
    topo, space = _continuous(bkd, (2, 2), 2)
    geometry = TensorBoxGeometry(topo, bkd)
    with pytest.raises(ValueError, match="none of the faces"):
        geometry.locate(bkd.asarray([[0.5], [0.5]]), space.faces())
    with pytest.raises(ValueError, match="shape"):
        geometry.locate(bkd.zeros((1, 2)), space.faces())
    vertex = next(e for e in range(topo.nentities()) if topo.entity_dim(e) == 0)
    with pytest.raises(ValueError, match="not a face"):
        geometry.locate(bkd.zeros((2, 1)), [vertex])
    with pytest.raises(KeyError):
        space.face_values(vertex, bkd.zeros((1, 1)))
