r"""Skeleton spaces: the discrete interface unknowns :math:`\Lambda_h`.

A skeleton space numbers the unknowns on the skeleton, says which
belong to each skeleton entity, and supplies the interface mass
:math:`M_{ij} = \int_\Gamma \phi_i \phi_j \, ds` as small dense blocks,
one per face, with their global DOFs. It forms no operator: an
assembler of the vector stack does
(:mod:`~pyapprox.pde.coupling.assembly`), so the space does not decide
how skeleton vectors and operators are stored.
"""

from dataclasses import dataclass
from typing import Generic, Mapping, Protocol, Sequence, TypeVar, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendVectorSpace,
    backend_dtype_name,
)
from pyapprox.util.linalg.operators import SpaceDescriptor

B = TypeVar("B")


@runtime_checkable
class SkeletonBasisProtocol(Protocol, Generic[Array]):
    """A basis on a reference face, known by evaluation and quadrature.

    Nothing assumes a tensor grid: an implementation may be nodal,
    modal or a sparse grid.
    """

    def bkd(self) -> Backend[Array]:
        """The computational backend."""
        ...

    def ndim(self) -> int:
        """The dimension of the reference face, at least 0."""
        ...

    def nterms(self) -> int:
        """The number of basis functions."""
        ...

    def evaluate(self, points: Array) -> Array:
        """The basis at ``points`` of shape (ndim, npoints).

        Shape: (npoints, nterms).
        """
        ...

    def quadrature(self) -> tuple[Array, Array]:
        """Points, shape (ndim, nquad), and weights averaging over the
        reference face, shape (nquad, 1), summing to 1."""
        ...


@runtime_checkable
class FaceBasesProtocol(Protocol[Array]):
    """The basis on each skeleton face: degree, node family and face shape
    may differ between faces."""

    def face_basis(self, face: int) -> SkeletonBasisProtocol[Array]:
        """The basis on ``face``'s reference face."""
        ...


class UniformBases(Generic[B]):
    """The same basis on every face.

    Parameters
    ----------
    basis
        The basis, of the face dimension.
    """

    def __init__(self, basis: B) -> None:
        self._basis = basis

    def face_basis(self, face: int) -> B:
        return self._basis


def _face_bases(
    bases: FaceBasesProtocol[Array], faces: Sequence[int], face_dim: int
) -> Backend[Array]:
    """Check each face's basis; the backend they share."""
    if not isinstance(bases, FaceBasesProtocol):
        raise TypeError(
            f"bases must satisfy FaceBasesProtocol, got {type(bases).__name__}"
        )
    if not faces:
        raise ValueError("the partition has no skeleton faces")
    for face in faces:
        basis = bases.face_basis(face)
        if not isinstance(basis, SkeletonBasisProtocol):
            raise TypeError(
                f"face {face}'s basis must satisfy SkeletonBasisProtocol, got "
                f"{type(basis).__name__}"
            )
        if basis.ndim() != face_dim:
            raise ValueError(
                f"face {face}'s basis dimension {basis.ndim()} differs from the "
                f"face dimension {face_dim}"
            )
    return bases.face_basis(faces[0]).bkd()


@dataclass(frozen=True)
class MassBlock(Generic[Array]):
    r"""One face's contribution to the interface mass: the dense block
    :math:`|f|\, \Phi_f^\top W_f \Phi_f` on the global DOFs ``dofs`` of
    the face's nodes, in node order. The mass is the sum of the blocks,
    each added at its rows and columns; an assembler of the vector stack
    forms it."""

    dofs: tuple[int, ...]
    matrix: Array


def _face_blocks(
    face_dofs: Mapping[int, Sequence[int]],
    bases: FaceBasesProtocol[Array],
    measures: Mapping[int, float],
) -> tuple[MassBlock[Array], ...]:
    blocks = []
    for face, dofs in face_dofs.items():
        basis = bases.face_basis(face)
        bkd = basis.bkd()
        points, weights = basis.quadrature()
        phi = basis.evaluate(points)
        block = measures[face] * bkd.dot(phi.T, weights * phi)
        blocks.append(MassBlock(tuple(dofs), block))
    return tuple(blocks)


@runtime_checkable
class FacetTopologyProtocol(Protocol):
    """What a facet space needs of a partition: its faces and their size."""

    def ndim(self) -> int: ...

    def nentities(self) -> int: ...

    def entity_dim(self, entity: int) -> int: ...

    def entity_measure(self, entity: int) -> float: ...


class FacetSkeletonSpace(Generic[Array]):
    """Unknowns on skeleton faces only, discontinuous across their closure.

    Each face (an entity of dimension ``ndim() - 1``) carries its own
    basis's unknowns; lower-dimensional entities carry none. DOFs are
    numbered face by face, in entity order.

    Parameters
    ----------
    topology
        The partition.
    bases
        The basis on each face, of dimension ``topology.ndim() - 1``.

    Raises
    ------
    TypeError
        If an argument or a face's basis does not satisfy its protocol.
    ValueError
        If a face's basis has the wrong dimension, or there are no faces.
    """

    def __init__(
        self, topology: FacetTopologyProtocol, bases: FaceBasesProtocol[Array]
    ) -> None:
        if not isinstance(topology, FacetTopologyProtocol):
            raise TypeError(
                "topology must satisfy FacetTopologyProtocol, got "
                f"{type(topology).__name__}"
            )
        face_dim = topology.ndim() - 1
        faces = [
            e for e in range(topology.nentities()) if topology.entity_dim(e) == face_dim
        ]
        self._bkd = _face_bases(bases, faces, face_dim)
        self._bases = bases
        self._entity_dofs: list[tuple[int, ...]] = [
            () for _ in range(topology.nentities())
        ]
        size = 0
        for face in faces:
            nterms = bases.face_basis(face).nterms()
            self._entity_dofs[face] = tuple(range(size, size + nterms))
            size += nterms
        self._face_dofs = {face: self._entity_dofs[face] for face in faces}
        self._measures = {face: topology.entity_measure(face) for face in faces}
        self._descriptor = SpaceDescriptor(
            "skeleton", size, backend_dtype_name(self._bkd)
        )
        self._space = BackendVectorSpace(self._descriptor, self._bkd)

    def descriptor(self) -> SpaceDescriptor:
        return self._descriptor

    def vector_space(self) -> BackendVectorSpace[Array]:
        """The stack's space of skeleton vectors."""
        return self._space

    def entity_dofs(self, entity: int) -> tuple[int, ...]:
        """The DOFs on ``entity``: its face's block, empty below faces."""
        if not 0 <= entity < len(self._entity_dofs):
            raise IndexError(
                f"entity {entity} is out of range [0, {len(self._entity_dofs)})"
            )
        return self._entity_dofs[entity]

    def mass_blocks(self) -> tuple[MassBlock[Array], ...]:
        r"""The interface mass by face; the blocks do not overlap, so the
        mass is block diagonal, each block the face measure times
        :math:`\Phi^\top W \Phi` on the reference face."""
        return _face_blocks(self._face_dofs, self._bases, self._measures)


def _check_numbering(
    face_nodes: Sequence[Sequence[tuple[int, int]]],
    counts: Mapping[int, int],
    nterms: Sequence[int],
) -> None:
    if [len(nodes) for nodes in face_nodes] != list(nterms):
        raise ValueError(
            f"the numbering must give {list(nterms)} nodes for the faces, got "
            f"{[len(nodes) for nodes in face_nodes]}"
        )
    read = {(e, r) for nodes in face_nodes for e, r in nodes}
    if any(e not in counts or not 0 <= r < counts[e] for e, r in read):
        raise ValueError("the numbering gives a rank outside its entity's count")
    if len(read) != sum(counts.values()):
        raise ValueError("the numbering counts a DOF no face node reads")


@dataclass(frozen=True)
class ClosureNumbering:
    """Which DOF each face node reads, before DOFs get global numbers.

    Attributes
    ----------
    face_nodes
        Per face, in the order the faces were given, per basis node:
        ``(entity, rank)``, the skeleton entity the node lies on and its
        rank among that entity's nodes.
    entity_counts
        The number of nodes on each entity that carries any.
    """

    face_nodes: tuple[tuple[tuple[int, int], ...], ...]
    entity_counts: Mapping[int, int]


@runtime_checkable
class ClosureNumberingProtocol(Protocol):
    """Decides which nodes of the faces' closures are the same unknown.

    The whole procedure is the implementation's: what identifies a node
    (keys, orientation tables, integer lattices), and how shared nodes
    are found.
    """

    def number(self, faces: Sequence[int]) -> ClosureNumbering:
        """Each node of each of ``faces`` as (entity, rank)."""
        ...


@runtime_checkable
class ContinuousTopologyProtocol(FacetTopologyProtocol, Protocol):
    """What a continuous space needs of a partition: faces, their size,
    and which entities lie on the physical boundary."""

    def on_boundary(self, entity: int) -> bool: ...


class ContinuousSkeletonSpace(Generic[Array]):
    r"""Nodal unknowns continuous across the skeleton's faces.

    Each node of each skeleton face lies on one entity of the face's
    closure (the face, an edge, a vertex, or a piece on the physical
    boundary); every face touching that entity reads the same DOF, so
    closure values are single-valued. Which face nodes are the same
    unknown is the injected numbering's decision; the space numbers DOFs
    entity by entity, nodes on an entity by the numbering's ranks. The
    mass is :math:`G^\top M_{\text{broken}} G`, where :math:`G` copies
    DOFs to face-by-face nodal values.

    Parameters
    ----------
    topology
        The partition.
    bases
        The nodal basis on each face, of dimension ``topology.ndim() - 1``.
    numbering
        Identifies the faces' closure nodes, for the same topology and
        bases (e.g. :class:`~pyapprox.pde.coupling.closure_numbering.
        VertexWeightNumbering`).

    Raises
    ------
    TypeError
        If an argument or a face's basis does not satisfy its protocol.
    ValueError
        If a face's basis has the wrong dimension, there are no faces, or
        the numbering's answer does not fit the faces and bases: a face
        or node missing, a rank out of range, or a DOF no node reads.
    """

    def __init__(
        self,
        topology: ContinuousTopologyProtocol,
        bases: FaceBasesProtocol[Array],
        numbering: ClosureNumberingProtocol,
    ) -> None:
        if not isinstance(topology, ContinuousTopologyProtocol):
            raise TypeError(
                "topology must satisfy ContinuousTopologyProtocol, got "
                f"{type(topology).__name__}"
            )
        if not isinstance(numbering, ClosureNumberingProtocol):
            raise TypeError(
                "numbering must satisfy ClosureNumberingProtocol, got "
                f"{type(numbering).__name__}"
            )
        face_dim = topology.ndim() - 1
        faces = [
            e
            for e in range(topology.nentities())
            if topology.entity_dim(e) == face_dim and not topology.on_boundary(e)
        ]
        self._bkd = _face_bases(bases, faces, face_dim)
        self._bases = bases
        numbered = numbering.number(faces)
        local, counts = numbered.face_nodes, dict(numbered.entity_counts)
        nterms = [bases.face_basis(face).nterms() for face in faces]
        _check_numbering(local, counts, nterms)
        offsets, size = {}, 0
        for entity in sorted(counts):
            offsets[entity] = size
            size += counts[entity]
        self._entity_dofs = [
            tuple(range(offsets[e], offsets[e] + counts[e])) if e in counts else ()
            for e in range(topology.nentities())
        ]
        self._face_dofs = {
            face: tuple(offsets[p] + r for p, r in nodes)
            for face, nodes in zip(faces, local)
        }
        self._measures = {face: topology.entity_measure(face) for face in faces}
        self._descriptor = SpaceDescriptor(
            "skeleton", size, backend_dtype_name(self._bkd)
        )
        self._space = BackendVectorSpace(self._descriptor, self._bkd)

    def descriptor(self) -> SpaceDescriptor:
        return self._descriptor

    def vector_space(self) -> BackendVectorSpace[Array]:
        """The stack's space of skeleton vectors."""
        return self._space

    def entity_dofs(self, entity: int) -> tuple[int, ...]:
        """The DOFs on ``entity`` itself (not its closure)."""
        if not 0 <= entity < len(self._entity_dofs):
            raise IndexError(
                f"entity {entity} is out of range [0, {len(self._entity_dofs)})"
            )
        return self._entity_dofs[entity]

    def faces(self) -> tuple[int, ...]:
        """The skeleton faces, increasing: the entities that carry nodes
        of their own, faces of dimension ``ndim - 1`` off the boundary."""
        return tuple(self._face_dofs)

    def face_dofs(self, face: int) -> tuple[int, ...]:
        """The DOF of each of ``face``'s basis nodes, in node order."""
        if face not in self._face_dofs:
            raise KeyError(f"entity {face} is not a skeleton face")
        return self._face_dofs[face]

    def mass_blocks(self) -> tuple[MassBlock[Array], ...]:
        r"""The interface mass :math:`G^\top M_{\text{broken}} G` by face:
        faces sharing a closure entity overlap on its DOFs, where their
        blocks add."""
        return _face_blocks(self._face_dofs, self._bases, self._measures)
