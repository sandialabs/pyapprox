r"""Skeleton spaces: the discrete interface unknowns :math:`\Lambda_h`.

A skeleton space numbers the unknowns on the skeleton, says which
belong to each skeleton entity, and supplies the interface mass
:math:`M_{ij} = \int_\Gamma \phi_i \phi_j \, ds` as an operator. Its
vectors are those of a linear-algebra stack; nothing here indexes them.
"""

from typing import Generic, Protocol, runtime_checkable

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
        bkd = self._bkd
        points, weights = self._basis.quadrature()
        phi = self._basis.evaluate(points)
        reference = bkd.dot(phi.T, weights * phi)
        matrix = bkd.kron(bkd.diag(bkd.asarray(self._measures)), reference)
        props = OperatorProperties(symmetric=True, spd=True)
        return BackendDenseOperator(matrix, self._descriptor, bkd, properties=props)
