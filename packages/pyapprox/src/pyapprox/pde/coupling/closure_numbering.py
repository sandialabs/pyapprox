r"""Numbering the nodes on skeleton faces' closures by vertex weights.

A continuous skeleton space asks a numbering which DOF each face node
reads (:class:`~pyapprox.pde.coupling.spaces.ClosureNumberingProtocol`).
:class:`VertexWeightNumbering` answers it for nodal bases on faces known
by their sub-entities. A node's sub-entity and the face's closure give
the skeleton entity it lies on; its weights on that sub-entity's
vertices, attached to the vertex entities the closure maps them to, give
a key every face computes the same way whatever its orientation. The
first face reaching an entity fixes its nodes' order (sorted keys); a
later face reads each node's rank by finding its key there.
"""

from typing import Generic, Protocol, Sequence, runtime_checkable

from pyapprox.pde.coupling.spaces import ClosureNumbering
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class ReferenceFaceProtocol(Protocol):
    """A reference face shape known by its sub-entities: vertices,
    edges, ..., the face itself, numbered from 0."""

    def ndim(self) -> int:
        """The dimension of the face."""
        ...

    def nsubentities(self) -> int:
        """The number of sub-entities, the face itself included."""
        ...

    def subentity_dim(self, subentity: int) -> int:
        """The dimension of ``subentity``."""
        ...

    def subentity_vertices(self, subentity: int) -> tuple[int, ...]:
        """The vertices (sub-entities of dimension 0) of ``subentity``;
        a vertex's are itself."""
        ...

    def vertex_weights(
        self, subentity: int, point: Sequence[float]
    ) -> tuple[float, ...]:
        """The weights of ``point``, which lies on ``subentity``, on
        :meth:`subentity_vertices`, in that order; they sum to 1."""
        ...


@runtime_checkable
class FaceClosureProtocol(Protocol):
    """What the numbering asks of a partition: each face's shape and the
    entity each of its sub-entities is."""

    def face_reference(self, face: int) -> ReferenceFaceProtocol:
        """The reference shape of ``face``."""
        ...

    def face_closure(self, face: int) -> tuple[int, ...]:
        """The entity each sub-entity of :meth:`face_reference` is."""
        ...


@runtime_checkable
class PlacedNodesProtocol(Protocol[Array]):
    """What the numbering asks of a nodal basis: where its nodes are on
    which reference face, and the sub-entity each lies on."""

    def bkd(self) -> Backend[Array]: ...

    def reference(self) -> ReferenceFaceProtocol:
        """The reference face the nodes lie on."""
        ...

    def nodes(self) -> Array:
        """The nodes on the reference face. Shape: (ndim, nterms)."""
        ...

    def node_entities(self) -> tuple[int, ...]:
        """The sub-entity of :meth:`reference` each node lies on."""
        ...


@runtime_checkable
class PlacedBasesProtocol(Protocol[Array]):
    """What the numbering asks for each face: its nodal basis."""

    def face_basis(self, face: int) -> PlacedNodesProtocol[Array]:
        """The basis on ``face``'s reference face."""
        ...


# Weights on a node's vertices, as (vertex entity, weight) pairs sorted by
# entity: the same from every face that has the node in its closure.
_NodeKey = tuple[tuple[int, float], ...]


class VertexWeightNumbering(Generic[Array]):
    """Numbers face nodes by their weights on the vertex entities.

    Satisfies :class:`~pyapprox.pde.coupling.spaces.
    ClosureNumberingProtocol`. Keys match within ``tol``; the lookup is
    quadratic in the nodes per entity, once at setup.

    Parameters
    ----------
    topology
        Each face's shape and closure.
    bases
        Each face's nodes; faces may differ, provided neighbors put the
        same nodes on the entities they share.
    tol
        The largest weight difference two faces' keys of one node may
        have.

    Raises
    ------
    TypeError
        If ``topology`` or ``bases`` does not satisfy its protocol.
    """

    def __init__(
        self,
        topology: FaceClosureProtocol,
        bases: PlacedBasesProtocol[Array],
        tol: float = 1e-10,
    ) -> None:
        if not isinstance(topology, FaceClosureProtocol):
            raise TypeError(
                "topology must satisfy FaceClosureProtocol, got "
                f"{type(topology).__name__}"
            )
        if not isinstance(bases, PlacedBasesProtocol):
            raise TypeError(
                "bases must satisfy PlacedBasesProtocol, got "
                f"{type(bases).__name__}"
            )
        self._topology, self._bases, self._tol = topology, bases, tol

    def number(self, faces: Sequence[int]) -> ClosureNumbering:
        """Each node of each of ``faces`` as (entity, rank).

        Raises
        ------
        TypeError
            If a face's basis does not satisfy its protocol.
        ValueError
            If a face's shape is not its basis's, or two faces put
            different nodes on a shared entity.
        """
        topology = self._topology
        # Node positions are fixed structure, read once per basis. Each
        # entry keeps its basis alive, so no later basis reuses its id.
        placed: dict[
            int,
            tuple[list[list[float]], tuple[int, ...], PlacedNodesProtocol[Array]],
        ] = {}
        keys: dict[int, list[_NodeKey]] = {}
        local = []
        for face in faces:
            basis = self._bases.face_basis(face)
            if not isinstance(basis, PlacedNodesProtocol):
                raise TypeError(
                    f"face {face}'s basis must satisfy PlacedNodesProtocol, got "
                    f"{type(basis).__name__}"
                )
            reference = basis.reference()
            if id(basis) not in placed:
                bkd, nodes = basis.bkd(), basis.nodes()
                on = basis.node_entities()
                placed[id(basis)] = (
                    [
                        [bkd.to_float(nodes[i, j]) for i in range(reference.ndim())]
                        for j in range(len(on))
                    ],
                    on,
                    basis,
                )
            points, on, _ = placed[id(basis)]
            if topology.face_reference(face) != reference:
                raise ValueError(
                    f"face {face} has shape {topology.face_reference(face)}, "
                    f"the basis {reference}"
                )
            closure = topology.face_closure(face)
            # Every closure entity is listed, so a face with no node on
            # an entity another face has nodes on is caught.
            mine: dict[int, list[tuple[_NodeKey, int]]] = {e: [] for e in closure}
            for j, sub in enumerate(on):
                weights = reference.vertex_weights(sub, points[j])
                vertices = [closure[v] for v in reference.subentity_vertices(sub)]
                mine[closure[sub]].append((tuple(sorted(zip(vertices, weights))), j))
            rank = [0] * len(on)
            for entity, found in mine.items():
                found.sort()
                known = keys.setdefault(entity, [key for key, _ in found])
                if len(known) != len(found):
                    raise ValueError(
                        f"faces disagree on the number of nodes on entity "
                        f"{entity}: {len(known)} and {len(found)}"
                    )
                for key, j in found:
                    matches = [r for r, k in enumerate(known) if self._same(k, key)]
                    if len(matches) != 1:
                        raise ValueError(
                            f"faces disagree on the nodes of entity {entity}"
                        )
                    rank[j] = matches[0]
            local.append(tuple((closure[sub], rank[j]) for j, sub in enumerate(on)))
        counts = {entity: len(found) for entity, found in keys.items()}
        return ClosureNumbering(tuple(local), counts)

    def _same(self, a: _NodeKey, b: _NodeKey) -> bool:
        return [e for e, _ in a] == [e for e, _ in b] and all(
            abs(wa - wb) <= self._tol for (_, wa), (_, wb) in zip(a, b)
        )
