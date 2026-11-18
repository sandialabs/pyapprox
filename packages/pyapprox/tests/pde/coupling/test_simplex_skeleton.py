r"""A continuous skeleton on triangular faces: three tetrahedra around a
common edge AB, so three skeleton triangles (A, B, C_i) share AB.

The triangles list their vertices in different orders, so their local
edge coordinates run opposite ways along AB; cubic nodes put two nodes
inside each edge, which only a correct matching pairs up. Checks are
derivation-free: the DOF count; every face places each DOF at the same
physical point (:func:`skeleton_dof_points` refuses otherwise); distinct
DOFs are distinct points; the mass integrates 1 and a linear function
exactly; bases that put different nodes on AB, and a box basis on a
triangle, are refused.
"""

import itertools
import math
from typing import Generic, Sequence

import numpy as np
import pytest

from pyapprox.pde.coupling.assembly import DenseMassAssembler
from pyapprox.pde.coupling.basis import TensorLagrangeBasis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.geometry import skeleton_dof_points
from pyapprox.pde.coupling.reference_faces import ReferenceSimplex
from pyapprox.pde.coupling.spaces import (
    ClosureNumbering,
    ContinuousSkeletonSpace,
    UniformBases,
)
from pyapprox.util.backends.protocols import Array, Backend

_TRIANGLE = ReferenceSimplex(2)
# The exponents (a, b) of the monomials x^a y^b of degree at most 3.
_EXPONENTS = [(a, total - a) for total in range(4) for a in range(total + 1)]


class _CubicTriangle(Generic[Array]):
    """Cubic Lagrange basis on the reference triangle: nodes at the
    vertices, at fractions ``s`` and ``1 - s`` along each edge (from its
    lower-numbered vertex), and at the centroid."""

    def __init__(self, bkd: Backend[Array], s: float = 1.0 / 3.0) -> None:
        self._bkd = bkd
        corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        nodes = list(corners)
        entities = [0, 1, 2]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            (xa, ya), (xb, yb) = corners[a], corners[b]
            for t in (s, 1.0 - s):
                nodes.append((xa + t * (xb - xa), ya + t * (yb - ya)))
                entities.append(_TRIANGLE.subentity((a, b)))
        nodes.append((1.0 / 3.0, 1.0 / 3.0))
        entities.append(_TRIANGLE.subentity((0, 1, 2)))
        self._nodes = bkd.asarray([[x for x, _ in nodes], [y for _, y in nodes]])
        self._entities = tuple(entities)
        self._coef = bkd.inv(self._monomials(self._nodes))
        # Collapsed Gauss rule, exact to degree 7; weights average to 1.
        u, wu = np.polynomial.legendre.leggauss(4)
        u, wu = (u + 1.0) / 2.0, wu / 2.0
        xs = [ui for ui in u for _ in u]
        ys = [vj * (1.0 - ui) for ui in u for vj in u]
        ws = [2.0 * (1.0 - ui) * wi * wj for ui, wi in zip(u, wu) for wj in wu]
        self._quad = (
            bkd.asarray([xs, ys]),
            bkd.reshape(bkd.asarray(ws), (len(ws), 1)),
        )

    def _monomials(self, points: Array) -> Array:
        return self._bkd.stack(
            [points[0] ** a * points[1] ** b for a, b in _EXPONENTS], axis=1
        )

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def ndim(self) -> int:
        return 2

    def nterms(self) -> int:
        return len(self._entities)

    def evaluate(self, points: Array) -> Array:
        return self._bkd.dot(self._monomials(points), self._coef)

    def quadrature(self) -> tuple[Array, Array]:
        return self._quad

    def reference(self) -> ReferenceSimplex:
        return _TRIANGLE

    def nodes(self) -> Array:
        return self._nodes

    def node_entities(self) -> tuple[int, ...]:
        return self._entities


# A and B are the common edge's ends, C0..C2 around it.
_POINTS = {
    "A": (0.0, 0.0, 0.0),
    "B": (0.0, 0.0, 1.0),
    **{
        f"C{i}": (math.cos(2 * math.pi * i / 3), math.sin(2 * math.pi * i / 3), 0.5)
        for i in range(3)
    },
}
# Local vertex order differs between faces, so edge AB runs both ways.
_FACES = (("A", "B", "C0"), ("C1", "B", "A"), ("B", "C2", "A"))


class _AroundEdge:
    """The skeleton of three tetrahedra (A, B, C_i, C_{i+1}): the faces
    (A, B, C_i) and their edges and vertices. Only the faces and AB are
    inside the union; everything else is on its boundary."""

    def __init__(self) -> None:
        sets = {frozenset(face) for face in _FACES}
        for face in _FACES:
            for size in (1, 2):
                sets.update(
                    frozenset(sub) for sub in itertools.combinations(face, size)
                )
        self._entities = sorted(sets, key=lambda s: (-len(s), sorted(s)))
        self._ids = {s: k for k, s in enumerate(self._entities)}

    def ndim(self) -> int:
        return 3

    def nentities(self) -> int:
        return len(self._entities)

    def entity_dim(self, entity: int) -> int:
        return len(self._entities[entity]) - 1

    def entity_measure(self, entity: int) -> float:
        points = [np.asarray(_POINTS[v]) for v in sorted(self._entities[entity])]
        if len(points) == 3:
            cross = np.cross(points[1] - points[0], points[2] - points[0])
            return float(np.linalg.norm(cross) / 2.0)
        if len(points) == 2:
            return float(np.linalg.norm(points[1] - points[0]))
        return 1.0

    def on_boundary(self, entity: int) -> bool:
        labels = self._entities[entity]
        return len(labels) < 3 and labels != frozenset(("A", "B"))

    def face_reference(self, face: int) -> ReferenceSimplex:
        return _TRIANGLE

    def face_closure(self, face: int) -> tuple[int, ...]:
        local = self.face_labels(face)
        return tuple(
            self._ids[frozenset(local[v] for v in _TRIANGLE.subentity_vertices(k))]
            for k in range(_TRIANGLE.nsubentities())
        )

    def face_labels(self, face: int) -> tuple[str, ...]:
        """``face``'s vertices in its local order."""
        return next(f for f in _FACES if frozenset(f) == self._entities[face])

    def entity_id(self, labels: tuple[str, ...]) -> int:
        return self._ids[frozenset(labels)]


class _AffineFaces(Generic[Array]):
    """Each triangle as the affine image of the reference triangle, its
    local vertices 0, 1, 2 at its listed points."""

    def __init__(self, topology: _AroundEdge, bkd: Backend[Array]) -> None:
        self._topology, self._bkd = topology, bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def ndim(self) -> int:
        return 3

    def face_points(self, face: int, points: Array) -> Array:
        p0, p1, p2 = (_POINTS[v] for v in self._topology.face_labels(face))
        bkd = self._bkd
        origin = bkd.reshape(bkd.asarray(p0), (3, 1))
        jacobian = bkd.asarray(
            [[b - a for a, b in zip(p0, p1)], [c - a for a, c in zip(p0, p2)]]
        ).T
        return origin + bkd.dot(jacobian, points)


def _space(
    bkd: Backend[Array], bases: UniformBases[_CubicTriangle[Array]]
) -> tuple[_AroundEdge, ContinuousSkeletonSpace[Array]]:
    topo = _AroundEdge()
    numbering = VertexWeightNumbering(topo, bases)
    return topo, ContinuousSkeletonSpace(topo, bases, numbering)


def test_shared_edge_nodes_match(bkd: Backend[Array]) -> None:
    bases = UniformBases(_CubicTriangle(bkd))
    topo, space = _space(bkd, bases)
    # 5 vertices, 7 edges with 2 nodes each, 3 face centroids.
    assert space.descriptor().size == 5 + 7 * 2 + 3
    assert len(space.entity_dofs(topo.entity_id(("A", "B")))) == 2
    points = skeleton_dof_points(space, _AffineFaces(topo, bkd), bases)
    size = space.descriptor().size
    coords = {
        tuple(round(bkd.to_float(points[i, k]), 10) for i in range(3))
        for k in range(size)
    }
    assert len(coords) == size


def test_mass_integrates_linears(bkd: Backend[Array]) -> None:
    bases = UniformBases(_CubicTriangle(bkd))
    topo, space = _space(bkd, bases)
    mass = DenseMassAssembler(bkd).assemble(space)
    points = skeleton_dof_points(space, _AffineFaces(topo, bkd), bases)
    # f(x) = 1 + x + 2y + 3z at each DOF; its integral over a triangle is
    # the area times f at the centroid.
    f = 1.0 + points[0] + 2.0 * points[1] + 3.0 * points[2]
    f = bkd.reshape(f, (f.shape[0], 1))
    ones = space.vector_space().zeros() + 1.0
    integral, area = 0.0, 0.0
    for face in space.faces():
        centroid = np.mean([_POINTS[v] for v in topo.face_labels(face)], axis=0)
        measure = topo.entity_measure(face)
        area += measure
        integral += measure * (1.0 + centroid @ np.asarray([1.0, 2.0, 3.0]))
    for values, expected in ((ones, area), (f, integral)):
        total = bkd.sum(ones * mass.apply(values))
        bkd.assert_allclose(
            bkd.reshape(total, (1,)), bkd.asarray([expected]), rtol=1e-12
        )


class _LocalOrder(Generic[Array]):
    """A wrong numbering: nodes on an entity ranked by each face's own
    node order, ignoring orientation."""

    def __init__(self, topo: _AroundEdge, basis: _CubicTriangle[Array]) -> None:
        self._topo, self._basis = topo, basis

    def number(self, faces: Sequence[int]) -> ClosureNumbering:
        local, counts = [], {}
        for face in faces:
            closure = self._topo.face_closure(face)
            seen: dict[int, int] = {}
            nodes = []
            for sub in self._basis.node_entities():
                entity = closure[sub]
                nodes.append((entity, seen.setdefault(entity, 0)))
                seen[entity] += 1
            counts.update(seen)
            local.append(tuple(nodes))
        return ClosureNumbering(tuple(local), counts)


def test_orientation_blind_numbering_is_caught(bkd: Backend[Array]) -> None:
    basis = _CubicTriangle(bkd)
    bases = UniformBases(basis)
    topo = _AroundEdge()
    space = ContinuousSkeletonSpace(topo, bases, _LocalOrder(topo, basis))
    with pytest.raises(ValueError, match="different points"):
        skeleton_dof_points(space, _AffineFaces(topo, bkd), bases)


class _Two(Generic[Array]):
    """One basis on face ``chosen``, another on the rest."""

    def __init__(
        self,
        chosen: int,
        special: _CubicTriangle[Array],
        other: _CubicTriangle[Array],
    ) -> None:
        self._chosen, self._special, self._other = chosen, special, other

    def face_basis(self, face: int) -> _CubicTriangle[Array]:
        return self._special if face == self._chosen else self._other


def test_nonconforming_faces_are_refused(bkd: Backend[Array]) -> None:
    topo = _AroundEdge()
    face = topo.entity_id(_FACES[1])
    # Nodes at 0.3 and 0.7 along AB on one face, at 1/3 and 2/3 on others.
    shifted = _Two(face, _CubicTriangle(bkd, 0.3), _CubicTriangle(bkd))
    with pytest.raises(ValueError, match="disagree on the nodes"):
        ContinuousSkeletonSpace(topo, shifted, VertexWeightNumbering(topo, shifted))
    boxes = UniformBases(TensorLagrangeBasis(2, 2, bkd))
    with pytest.raises(ValueError, match="shape"):
        ContinuousSkeletonSpace(topo, boxes, VertexWeightNumbering(topo, boxes))
