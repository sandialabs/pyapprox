"""Reference-face bases and the facet-discontinuous skeleton space.

Checks are derivation-free: the nodal (Kronecker) property, partition
of unity, the known mass of linear hat functions, and the total
skeleton measure recovered as 1^T M 1.
"""

from typing import Generic, Sequence

import numpy as np
import pytest

from pyapprox.pde.coupling.assembly import DenseMassAssembler
from pyapprox.pde.coupling.basis import (
    TensorLagrangeBasis,
    gauss_lagrange_basis,
    lobatto_lagrange_basis,
)
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.reference_faces import ReferenceBox
from pyapprox.pde.coupling.spaces import (
    ClosureNumbering,
    ContinuousSkeletonSpace,
    FacetSkeletonSpace,
    SkeletonBasisProtocol,
    UniformBases,
)
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.util.backends.protocols import Array, Backend


@pytest.mark.parametrize("ndim,npoints", [(0, 1), (1, 1), (1, 2), (1, 4), (2, 3)])
def test_basis_is_nodal_partition_of_unity(
    bkd: Backend[Array], ndim: int, npoints: int
) -> None:
    basis = lobatto_lagrange_basis(ndim, npoints, bkd)
    assert isinstance(basis, SkeletonBasisProtocol)
    assert basis.nterms() == npoints**ndim
    nodes = basis.nodes()
    assert tuple(nodes.shape) == (ndim, basis.nterms())
    bkd.assert_allclose(
        basis.evaluate(nodes), bkd.eye(basis.nterms()), rtol=1e-12, atol=1e-12
    )
    points = bkd.asarray(np.random.default_rng(1).uniform(-1, 1, (ndim, 7)))
    bkd.assert_allclose(
        bkd.sum(basis.evaluate(points), axis=1), bkd.ones((7,)), rtol=1e-12
    )
    _, weights = basis.quadrature()
    bkd.assert_allclose(bkd.sum(weights, axis=0), bkd.ones((1,)), rtol=1e-12)


def test_linear_hat_mass(bkd: Backend[Array]) -> None:
    # Hats on [-1, 1], averaged over the interval: [[1/3, 1/6], [1/6, 1/3]].
    basis = lobatto_lagrange_basis(1, 2, bkd)
    points, weights = basis.quadrature()
    phi = basis.evaluate(points)
    expected = bkd.asarray([[1 / 3, 1 / 6], [1 / 6, 1 / 3]])
    bkd.assert_allclose(bkd.dot(phi.T, weights * phi), expected, rtol=1e-12)


def test_basis_rejects_bad_input(bkd: Backend[Array]) -> None:
    with pytest.raises(ValueError):
        lobatto_lagrange_basis(-1, 2, bkd)
    with pytest.raises(ValueError):
        lobatto_lagrange_basis(1, 0, bkd)
    with pytest.raises(ValueError):
        gauss_lagrange_basis(1, 0, bkd)
    with pytest.raises(ValueError):
        lobatto_lagrange_basis(2, 2, bkd).evaluate(bkd.zeros((1, 3)))
    with pytest.raises(ValueError):
        TensorLagrangeBasis(1, bkd.zeros((0,)), False, bkd)
    with pytest.raises(ValueError):
        TensorLagrangeBasis(1, bkd.asarray([-0.5, 0.5]), True, bkd)
    with pytest.raises(ValueError):
        TensorLagrangeBasis(1, bkd.asarray([-1.0, 1.0]), False, bkd)


@pytest.mark.parametrize("ndim,npoints", [(0, 1), (1, 1), (1, 3), (2, 2), (3, 2)])
def test_gauss_basis_is_open_and_nodal_at_its_quadrature(
    bkd: Backend[Array], ndim: int, npoints: int
) -> None:
    basis = gauss_lagrange_basis(ndim, npoints, bkd)
    assert isinstance(basis, SkeletonBasisProtocol)
    assert basis.nterms() == npoints**ndim
    bkd.assert_allclose(
        basis.evaluate(basis.nodes()), bkd.eye(basis.nterms()), atol=1e-12
    )
    points, _ = basis.quadrature()
    bkd.assert_allclose(points, basis.nodes(), atol=1e-14)
    interior = basis.reference().subentity([0] * ndim)
    assert basis.node_entities() == (interior,) * basis.nterms()


@pytest.mark.parametrize("ncells,npoints", [((3,), 2), ((2, 2), 3), ((2, 2, 2), 2)])
def test_facet_mass_on_gauss_nodes_is_diagonal(
    bkd: Backend[Array], ncells: tuple[int, ...], npoints: int
) -> None:
    # Nodes at the quadrature points make Phi = I there, so each face's
    # block is |f| diag(w).
    bounds = [(0.0, 2.0)] + [(0.0, 1.0)] * (len(ncells) - 1)
    topo = TensorBoxPartition(ncells, bounds=bounds)
    basis = gauss_lagrange_basis(topo.ndim() - 1, npoints, bkd)
    space = FacetSkeletonSpace(topo, UniformBases(basis))
    matrix = DenseMassAssembler(bkd).assemble(space).format().matrix()
    _, weights = basis.quadrature()
    diagonal = []
    for e in range(topo.nentities()):
        if space.entity_dofs(e):
            diagonal.append(topo.entity_measure(e) * bkd.flatten(weights))
    bkd.assert_allclose(matrix, bkd.diag(bkd.concatenate(diagonal)), atol=1e-14)


@pytest.mark.parametrize(
    "ncells,npoints", [((2,), 1), ((4,), 1), ((2, 2), 3), ((3, 2), 2), ((2, 2, 2), 2)]
)
def test_facet_space_dofs_and_mass(
    bkd: Backend[Array], ncells: tuple[int, ...], npoints: int
) -> None:
    bounds = [(0.0, 2.0)] + [(0.0, 1.0)] * (len(ncells) - 1)
    topo = TensorBoxPartition(ncells, bounds=bounds)
    d = topo.ndim()
    space = FacetSkeletonSpace(
        topo, UniformBases(lobatto_lagrange_basis(d - 1, npoints, bkd))
    )
    faces = [e for e in range(topo.nentities()) if topo.entity_dim(e) == d - 1]
    nterms = npoints ** (d - 1)
    size = space.descriptor().size
    assert size == len(faces) * nterms
    blocks = [space.entity_dofs(e) for e in range(topo.nentities())]
    assert sorted(dof for block in blocks for dof in block) == list(range(size))
    for e, block in enumerate(blocks):
        assert len(block) == (nterms if e in faces else 0)

    mass = DenseMassAssembler(bkd).assemble(space)
    matrix = mass.format().matrix()
    bkd.assert_allclose(matrix, matrix.T, rtol=1e-12)
    eigenvalues, _ = bkd.eigh(matrix)
    assert bkd.to_float(bkd.min(eigenvalues)) > 0.0
    ones = bkd.ones((space.descriptor().size, 1))
    total = bkd.sum(ones * mass.apply(ones))
    expected = sum(topo.entity_measure(f) for f in faces)
    bkd.assert_allclose(bkd.reshape(total, (1,)), bkd.asarray([expected]), rtol=1e-12)


def _node_coordinates(
    topo: TensorBoxPartition, face: int, nodes: list[list[float]]
) -> list[tuple[float, ...]]:
    """Physical coordinates of reference nodes (one list per node) on
    ``face``, rounded so coincident points compare equal."""
    coords = []
    for ref in nodes:
        it = iter(ref)
        point = []
        for lo, up in topo.entity_bounds(face):
            r = next(it) if up > lo else -1.0
            point.append(round(lo + (r + 1.0) * (up - lo) / 2.0, 10))
        coords.append(tuple(point))
    return coords


_CONTINUOUS_CASES = [
    ((3,), 3),
    ((2, 2), 2),
    ((3, 2), 3),
    ((2, 2, 2), 3),
    ((2, 3, 2), 2),
]


@pytest.mark.parametrize("ncells,npoints", _CONTINUOUS_CASES)
def test_continuous_space_is_single_valued(
    bkd: Backend[Array], ncells: tuple[int, ...], npoints: int
) -> None:
    topo = TensorBoxPartition(ncells, bounds=[(0.0, 2.0)] * len(ncells))
    d = topo.ndim()
    basis = lobatto_lagrange_basis(d - 1, npoints, bkd)
    bases = UniformBases(basis)
    space = ContinuousSkeletonSpace(topo, bases, VertexWeightNumbering(topo, bases))
    # Lobatto nodes: (npoints - 2)^m strictly inside an m-dimensional entity.
    expected = sum((npoints - 2) ** topo.entity_dim(e) for e in range(topo.nentities()))
    assert space.descriptor().size == expected
    nodes = [
        [bkd.to_float(basis.nodes()[i, j]) for i in range(d - 1)]
        for j in range(basis.nterms())
    ]
    point_of: dict[int, tuple[float, ...]] = {}
    for face in range(topo.nentities()):
        if topo.entity_dim(face) != d - 1 or topo.on_boundary(face):
            continue
        for dof, point in zip(
            space.face_dofs(face), _node_coordinates(topo, face, nodes)
        ):
            # A DOF is one point, and distinct DOFs are distinct points.
            assert point_of.setdefault(dof, point) == point
    assert len(set(point_of.values())) == len(point_of) == expected


@pytest.mark.parametrize("ncells,npoints", _CONTINUOUS_CASES)
def test_continuous_mass_integrates_linears(
    bkd: Backend[Array], ncells: tuple[int, ...], npoints: int
) -> None:
    topo = TensorBoxPartition(ncells)
    d = topo.ndim()
    basis = lobatto_lagrange_basis(d - 1, npoints, bkd)
    bases = UniformBases(basis)
    space = ContinuousSkeletonSpace(topo, bases, VertexWeightNumbering(topo, bases))
    mass = DenseMassAssembler(bkd).assemble(space)
    matrix = mass.format().matrix()
    bkd.assert_allclose(matrix, matrix.T, rtol=1e-12)
    eigenvalues, _ = bkd.eigh(matrix)
    assert bkd.to_float(bkd.min(eigenvalues)) > 0.0
    # f(x) = 1 + sum_j (j + 1) x_j at each DOF; the integral of a linear
    # function over a face is its measure times f at the centroid.
    nodes = [
        [bkd.to_float(basis.nodes()[i, j]) for i in range(d - 1)]
        for j in range(basis.nterms())
    ]
    values = [0.0] * space.descriptor().size
    integral = 0.0
    for face in range(topo.nentities()):
        if topo.entity_dim(face) != d - 1 or topo.on_boundary(face):
            continue
        for dof, point in zip(
            space.face_dofs(face), _node_coordinates(topo, face, nodes)
        ):
            values[dof] = 1.0 + sum((j + 1) * x for j, x in enumerate(point))
        centroid = [(lo + up) / 2 for lo, up in topo.entity_bounds(face)]
        f_centroid = 1.0 + sum((j + 1) * x for j, x in enumerate(centroid))
        integral += topo.entity_measure(face) * f_centroid
    f = bkd.reshape(bkd.asarray(values), (len(values), 1))
    ones = bkd.ones((space.descriptor().size, 1))
    total = bkd.sum(ones * mass.apply(f))
    bkd.assert_allclose(bkd.reshape(total, (1,)), bkd.asarray([integral]), rtol=1e-12)


class _Anisotropic(Generic[Array]):
    """A 2-by-3 grid of nodes on the reference square, claiming 6 terms:
    edges along its two reference directions carry 0 and 1 nodes, so
    faces oriented differently disagree on a shared edge."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._inner = lobatto_lagrange_basis(2, 2, bkd)
        self._bkd = bkd
        self._reference = ReferenceBox(2)
        self._grid = [(a, b) for a in (-1, 1) for b in (-1, 0, 1)]

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def ndim(self) -> int:
        return 2

    def nterms(self) -> int:
        return 6

    def evaluate(self, points: Array) -> Array:
        return self._inner.evaluate(points)

    def quadrature(self) -> tuple[Array, Array]:
        return self._inner.quadrature()

    def reference(self) -> ReferenceBox:
        return self._reference

    def nodes(self) -> Array:
        return self._bkd.asarray(
            [[float(a) for a, _ in self._grid], [float(b) for _, b in self._grid]]
        )

    def node_entities(self) -> tuple[int, ...]:
        return tuple(self._reference.subentity(node) for node in self._grid)


def test_continuous_space_rejects_nonconforming(bkd: Backend[Array]) -> None:
    topo = TensorBoxPartition((2, 2, 2))
    anisotropic = UniformBases(_Anisotropic(bkd))
    with pytest.raises(ValueError, match="disagree"):
        ContinuousSkeletonSpace(
            topo, anisotropic, VertexWeightNumbering(topo, anisotropic)
        )
    square = TensorBoxPartition((2, 2))
    with pytest.raises(ValueError, match="dimension"):
        ContinuousSkeletonSpace(
            square, anisotropic, VertexWeightNumbering(square, anisotropic)
        )
    bases = UniformBases(lobatto_lagrange_basis(1, 2, bkd))
    space = ContinuousSkeletonSpace(square, bases, VertexWeightNumbering(square, bases))
    with pytest.raises(KeyError):
        space.face_dofs(space.descriptor().size + 100)
    with pytest.raises(IndexError):
        space.entity_dofs(-1)
    with pytest.raises(TypeError, match="ClosureNumberingProtocol"):
        ContinuousSkeletonSpace(square, bases, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="FaceBasesProtocol"):
        ContinuousSkeletonSpace(
            square,
            object(),  # type: ignore[arg-type]
            VertexWeightNumbering(square, bases),
        )


class _PerFace(Generic[Array]):
    """``npoints`` Lobatto points on the faces listed, ``default`` on the
    others."""

    def __init__(
        self, ndim: int, default: int, chosen: dict[int, int], bkd: Backend[Array]
    ) -> None:
        self._default = lobatto_lagrange_basis(ndim, default, bkd)
        self._chosen = {
            f: lobatto_lagrange_basis(ndim, n, bkd) for f, n in chosen.items()
        }

    def face_basis(self, face: int) -> TensorLagrangeBasis[Array]:
        return self._chosen.get(face, self._default)


def test_faces_may_differ_where_they_conform(bkd: Backend[Array]) -> None:
    # In 2D faces share only vertices, one node each: one face of degree
    # 3 beside faces of degree 1 conforms, adding its two interior nodes.
    square = TensorBoxPartition((2, 2))
    uniform = UniformBases(lobatto_lagrange_basis(1, 2, bkd))
    base = ContinuousSkeletonSpace(
        square, uniform, VertexWeightNumbering(square, uniform)
    )
    face = base.faces()[0]
    mixed = _PerFace(1, 2, {face: 4}, bkd)
    space = ContinuousSkeletonSpace(square, mixed, VertexWeightNumbering(square, mixed))
    assert space.descriptor().size == base.descriptor().size + 2
    assert len(space.face_dofs(face)) == 4
    ones = bkd.ones((space.descriptor().size, 1))
    total = bkd.sum(ones * DenseMassAssembler(bkd).assemble(space).apply(ones))
    expected = sum(square.entity_measure(f) for f in space.faces())
    bkd.assert_allclose(bkd.reshape(total, (1,)), bkd.asarray([expected]), rtol=1e-12)
    # In 3D faces share edges: degrees 2 and 1 put 1 and 0 nodes on one.
    cube = TensorBoxPartition((2, 2, 2))
    clash = _PerFace(2, 2, {0: 3}, bkd)
    with pytest.raises(ValueError, match="disagree"):
        ContinuousSkeletonSpace(cube, clash, VertexWeightNumbering(cube, clash))


class _FixedNumbering:
    """A numbering that is not key-based: it returns a given answer."""

    def __init__(self, answer: ClosureNumbering) -> None:
        self._answer = answer

    def number(self, faces: Sequence[int]) -> ClosureNumbering:
        return self._answer


def test_continuous_space_takes_any_numbering(bkd: Backend[Array]) -> None:
    # Three subdomains on a line: faces 0 and 1 are points, one node each.
    topo = TensorBoxPartition((3,))
    basis = UniformBases(lobatto_lagrange_basis(0, 1, bkd))
    reversed_ranks = ClosureNumbering((((0, 0),), ((1, 0),)), {1: 1, 0: 1})
    space = ContinuousSkeletonSpace(topo, basis, _FixedNumbering(reversed_ranks))
    assert space.descriptor().size == 2
    assert (space.face_dofs(0), space.face_dofs(1)) == ((0,), (1,))
    for bad, message in (
        (ClosureNumbering((((0, 0),),), {0: 1}), "nodes for the faces"),
        (ClosureNumbering((((0, 0),), ((1, 1),)), {0: 1, 1: 1}), "outside"),
        (ClosureNumbering((((0, 0),), ((0, 0),)), {0: 1, 1: 1}), "no face node"),
    ):
        with pytest.raises(ValueError, match=message):
            ContinuousSkeletonSpace(topo, basis, _FixedNumbering(bad))


def test_facet_space_rejects_mismatch(bkd: Backend[Array]) -> None:
    topo = TensorBoxPartition((2, 2))
    bases = UniformBases(lobatto_lagrange_basis(1, 2, bkd))
    with pytest.raises(ValueError):
        FacetSkeletonSpace(topo, UniformBases(lobatto_lagrange_basis(2, 2, bkd)))
    with pytest.raises(TypeError):
        FacetSkeletonSpace(object(), bases)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no skeleton faces"):
        FacetSkeletonSpace(TensorBoxPartition((1, 1)), bases)
    space = FacetSkeletonSpace(topo, bases)
    with pytest.raises(IndexError):
        space.entity_dofs(topo.nentities())
    # A face of higher degree gets a bigger block.
    mixed = FacetSkeletonSpace(topo, _PerFace(1, 2, {0: 3}, bkd))
    assert len(mixed.entity_dofs(0)) == 3
    assert mixed.descriptor().size == space.descriptor().size + 1
