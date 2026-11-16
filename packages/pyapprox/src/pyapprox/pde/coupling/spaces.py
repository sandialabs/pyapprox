r"""Skeleton spaces: the discrete interface unknowns :math:`\Lambda_h`.

A skeleton space numbers the unknowns on the skeleton, says which
belong to each skeleton entity, and supplies the interface mass
:math:`M_{ij} = \int_\Gamma \phi_i \phi_j \, ds` as an operator. Its
vectors are those of a linear-algebra stack; nothing here indexes them.
"""

import itertools
from typing import Generic, Protocol, Sequence, runtime_checkable

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


@runtime_checkable
class ClosureNodalBasisProtocol(SkeletonBasisProtocol[Array], Protocol[Array]):
    """A nodal basis that says which reference sub-entity each node is on."""

    def node_signs(self) -> tuple[tuple[int, ...], ...]:
        """Per node, per reference direction: -1 or +1 on the lower or
        upper side of [-1, 1], 0 strictly inside."""
        ...


@runtime_checkable
class ClosureTopologyProtocol(FacetTopologyProtocol, Protocol):
    """What a continuous space needs: faces, and their closure pieces."""

    def on_boundary(self, entity: int) -> bool: ...

    def closure_entity(self, entity: int, signs: Sequence[int]) -> int: ...


class ContinuousSkeletonSpace(Generic[Array]):
    r"""Nodal unknowns continuous across the skeleton's faces.

    Each node of each skeleton face lies on one entity of the face's
    closure (the face, an edge, a vertex, or a piece on the physical
    boundary); every face touching that entity reads the same DOF, so
    closure values are single-valued. DOFs are numbered entity by
    entity, nodes on an entity in the order each face lists them. That
    order agrees between faces because every face lists them by
    increasing reference coordinate, in the same axis order. The mass
    is :math:`G^\top M_{\text{broken}} G`, where :math:`G` copies DOFs
    to face-by-face nodal values.

    Parameters
    ----------
    topology
        The partition.
    basis
        A nodal basis on the reference face, of dimension
        ``topology.ndim() - 1``.

    Raises
    ------
    ValueError
        If two faces give a shared entity different numbers of nodes: the
        basis is not conforming.
    """

    def __init__(
        self,
        topology: ClosureTopologyProtocol,
        basis: ClosureNodalBasisProtocol[Array],
    ) -> None:
        if not isinstance(topology, ClosureTopologyProtocol):
            raise TypeError(
                "topology must satisfy ClosureTopologyProtocol, got "
                f"{type(topology).__name__}"
            )
        if not isinstance(basis, ClosureNodalBasisProtocol):
            raise TypeError(
                "basis must satisfy ClosureNodalBasisProtocol, got "
                f"{type(basis).__name__}"
            )
        face_dim = topology.ndim() - 1
        if basis.ndim() != face_dim:
            raise ValueError(
                f"basis dimension {basis.ndim()} differs from the face "
                f"dimension {face_dim}"
            )
        if len(basis.node_signs()) != basis.nterms():
            raise ValueError(
                f"basis gives {len(basis.node_signs())} node signs for "
                f"{basis.nterms()} basis functions"
            )
        self._bkd = basis.bkd()
        self._basis = basis
        faces = [
            e
            for e in range(topology.nentities())
            if topology.entity_dim(e) == face_dim and not topology.on_boundary(e)
        ]
        # Each face node as (entity, rank among the face's nodes on it).
        local: list[list[tuple[int, int]]] = []
        counts: dict[int, int] = {}
        for face in faces:
            # Every closure piece starts at 0, so a face with no node on
            # a piece another face has nodes on is caught.
            seen = {
                topology.closure_entity(face, signs): 0
                for signs in itertools.product((-1, 0, 1), repeat=face_dim)
            }
            nodes = []
            for signs in basis.node_signs():
                piece = topology.closure_entity(face, signs)
                nodes.append((piece, seen[piece]))
                seen[piece] += 1
            for piece, count in seen.items():
                if counts.setdefault(piece, count) != count:
                    raise ValueError(
                        f"faces disagree on the number of nodes on entity "
                        f"{piece}: {counts[piece]} and {count}"
                    )
            local.append(nodes)
        offsets, size = {}, 0
        for piece in sorted(counts):
            offsets[piece] = size
            size += counts[piece]
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
