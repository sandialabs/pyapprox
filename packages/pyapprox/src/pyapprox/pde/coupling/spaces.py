r"""Skeleton spaces: the discrete interface unknowns :math:`\Lambda_h`.

A skeleton space numbers the unknowns on the skeleton, says which
belong to each skeleton entity, and supplies the interface mass
:math:`M_{ij} = \int_\Gamma \phi_i \phi_j \, ds` as an operator. Its
vectors are those of a linear-algebra stack; nothing here indexes them.
"""

from dataclasses import dataclass
from typing import Generic, Mapping, Protocol, Sequence, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseOperator,
    BackendVectorSpace,
    backend_dtype_name,
)
from pyapprox.util.linalg.operators import OperatorProperties, SpaceDescriptor


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
class FacetTopologyProtocol(Protocol):
    """What a facet space needs of a partition: its faces and their size."""

    def ndim(self) -> int: ...

    def nentities(self) -> int: ...

    def entity_dim(self, entity: int) -> int: ...

    def entity_measure(self, entity: int) -> float: ...


class FacetSkeletonSpace(Generic[Array]):
    """Unknowns on skeleton faces only, discontinuous across their closure.

    Every face (an entity of dimension ``ndim() - 1``) carries the same
    reference basis; lower-dimensional entities carry no unknowns. DOFs
    are numbered face by face, in entity order.

    Parameters
    ----------
    topology
        The partition.
    basis
        The basis on the reference face, of dimension
        ``topology.ndim() - 1``.
    """

    def __init__(
        self, topology: FacetTopologyProtocol, basis: SkeletonBasisProtocol[Array]
    ) -> None:
        if not isinstance(topology, FacetTopologyProtocol):
            raise TypeError(
                "topology must satisfy FacetTopologyProtocol, got "
                f"{type(topology).__name__}"
            )
        if not isinstance(basis, SkeletonBasisProtocol):
            raise TypeError(
                "basis must satisfy SkeletonBasisProtocol, got "
                f"{type(basis).__name__}"
            )
        face_dim = topology.ndim() - 1
        if basis.ndim() != face_dim:
            raise ValueError(
                f"basis dimension {basis.ndim()} differs from the face "
                f"dimension {face_dim}"
            )
        self._bkd = basis.bkd()
        self._basis = basis
        faces = [
            e for e in range(topology.nentities()) if topology.entity_dim(e) == face_dim
        ]
        nterms = basis.nterms()
        self._entity_dofs: list[tuple[int, ...]] = [
            () for _ in range(topology.nentities())
        ]
        for k, face in enumerate(faces):
            self._entity_dofs[face] = tuple(range(k * nterms, (k + 1) * nterms))
        self._measures = [topology.entity_measure(face) for face in faces]
        self._descriptor = SpaceDescriptor(
            "skeleton", len(faces) * nterms, backend_dtype_name(self._bkd)
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

    def mass(self) -> BackendDenseOperator[Array]:
        r"""The interface mass, block diagonal over faces: each block is
        the face measure times :math:`\Phi^\top W \Phi` on the reference
        face."""
        matrix = _broken_mass(self._basis, self._measures)
        props = OperatorProperties(symmetric=True, spd=True)
        return BackendDenseOperator(
            matrix, self._descriptor, self._bkd, properties=props
        )


def _broken_mass(basis: SkeletonBasisProtocol[Array], measures: list[float]) -> Array:
    """The mass of face-by-face nodal values: diag(measures) kron the
    reference mass."""
    bkd = basis.bkd()
    points, weights = basis.quadrature()
    phi = basis.evaluate(points)
    reference = bkd.dot(phi.T, weights * phi)
    return bkd.kron(bkd.diag(bkd.asarray(measures)), reference)


def _check_numbering(
    face_nodes: Sequence[Sequence[tuple[int, int]]],
    counts: Mapping[int, int],
    nfaces: int,
    nterms: int,
) -> None:
    if len(face_nodes) != nfaces or any(len(n) != nterms for n in face_nodes):
        raise ValueError(
            f"the numbering must give {nterms} nodes for each of {nfaces} faces"
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
    basis
        The basis on every face, of dimension ``topology.ndim() - 1``.
    numbering
        Identifies the faces' closure nodes, for the same topology and
        basis (e.g. :class:`~pyapprox.pde.coupling.closure_numbering.
        VertexWeightNumbering`).

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    ValueError
        If the basis dimension is not the face dimension, or the
        numbering's answer does not fit the faces and basis: a face or
        node missing, a rank out of range, or a DOF no node reads.
    """

    def __init__(
        self,
        topology: ContinuousTopologyProtocol,
        basis: SkeletonBasisProtocol[Array],
        numbering: ClosureNumberingProtocol,
    ) -> None:
        if not isinstance(topology, ContinuousTopologyProtocol):
            raise TypeError(
                "topology must satisfy ContinuousTopologyProtocol, got "
                f"{type(topology).__name__}"
            )
        if not isinstance(basis, SkeletonBasisProtocol):
            raise TypeError(
                "basis must satisfy SkeletonBasisProtocol, got "
                f"{type(basis).__name__}"
            )
        if not isinstance(numbering, ClosureNumberingProtocol):
            raise TypeError(
                "numbering must satisfy ClosureNumberingProtocol, got "
                f"{type(numbering).__name__}"
            )
        face_dim = topology.ndim() - 1
        if basis.ndim() != face_dim:
            raise ValueError(
                f"basis dimension {basis.ndim()} differs from the face "
                f"dimension {face_dim}"
            )
        self._bkd = basis.bkd()
        self._basis = basis
        faces = [
            e
            for e in range(topology.nentities())
            if topology.entity_dim(e) == face_dim and not topology.on_boundary(e)
        ]
        numbered = numbering.number(faces)
        local, counts = numbered.face_nodes, dict(numbered.entity_counts)
        _check_numbering(local, counts, len(faces), basis.nterms())
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
        self._measures = [topology.entity_measure(face) for face in faces]
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

    def face_dofs(self, face: int) -> tuple[int, ...]:
        """The DOF of each of ``face``'s basis nodes, in node order."""
        if face not in self._face_dofs:
            raise KeyError(f"entity {face} is not a skeleton face")
        return self._face_dofs[face]

    def mass(self) -> BackendDenseOperator[Array]:
        r"""The interface mass :math:`G^\top M_{\text{broken}} G`."""
        bkd, size = self._bkd, self._descriptor.size
        rows = [
            [1.0 if dof == col else 0.0 for col in range(size)]
            for dofs in self._face_dofs.values()
            for dof in dofs
        ]
        gather = bkd.asarray(rows, dtype=bkd.default_dtype())
        broken = _broken_mass(self._basis, self._measures)
        matrix = bkd.dot(gather.T, bkd.dot(broken, gather))
        props = OperatorProperties(symmetric=True, spd=True)
        return BackendDenseOperator(matrix, self._descriptor, bkd, properties=props)
