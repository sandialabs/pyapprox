r"""Where skeleton faces are: maps from reference faces to physical space.

Geometry is separate from the topology and the space. A partition says
which entities exist and how they meet, with plain integers; a space
says which unknowns there are; a geometry says where a face's reference
points land. Curved or mesh-derived faces are new geometries, with the
same topology and space.
"""

from dataclasses import dataclass
from typing import Generic, Protocol, Sequence, TypeVar, runtime_checkable

from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.util.backends.protocols import Array, ArrayProtocol, Backend
from pyapprox.util.linalg.operators import SpaceDescriptor

A_co = TypeVar("A_co", bound=ArrayProtocol, covariant=True)


@runtime_checkable
class FaceGeometryProtocol(Protocol[Array]):
    """Maps a skeleton face's reference points to physical points."""

    def bkd(self) -> Backend[Array]: ...

    def ndim(self) -> int:
        """The dimension of physical space."""
        ...

    def face_points(self, face: int, points: Array) -> Array:
        """The physical points of reference ``points`` of ``face``,
        shape (ndim - 1, npoints). Shape: (ndim, npoints)."""
        ...


@dataclass(frozen=True)
class FacePoint(Generic[Array]):
    """A physical point located on a skeleton face: the face and the
    point's reference coordinates on it, shape (ndim - 1, 1)."""

    face: int
    point: Array


class TensorBoxGeometry(Generic[Array]):
    r"""The affine maps of a :class:`~pyapprox.pde.coupling.topology.
    TensorBoxPartition`'s faces: reference coordinate :math:`x` along a
    spanned direction lands at :math:`l + (x + 1)(u - l)/2`, in the
    order of the partition's reference boxes (increasing axis).

    Parameters
    ----------
    partition
        The partition, read for its entity extents.
    bkd
        The backend of the points.
    """

    def __init__(self, partition: TensorBoxPartition, bkd: Backend[Array]) -> None:
        self._partition, self._bkd = partition, bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def ndim(self) -> int:
        return self._partition.ndim()

    def face_points(self, face: int, points: Array) -> Array:
        """See :class:`FaceGeometryProtocol`.

        Raises
        ------
        ValueError
            If ``face`` is not a face or ``points`` has the wrong shape.
        """
        bkd, partition = self._bkd, self._partition
        if partition.entity_dim(face) != partition.ndim() - 1:
            raise ValueError(f"entity {face} is not a face")
        if points.ndim != 2 or points.shape[0] != partition.ndim() - 1:
            raise ValueError(
                f"points must have shape ({partition.ndim() - 1}, npoints), "
                f"got {tuple(points.shape)}"
            )
        npoints, spanned, rows = int(points.shape[1]), 0, []
        for lo, up in partition.entity_bounds(face):
            if up > lo:
                rows.append(lo + (points[spanned] + 1.0) * ((up - lo) / 2.0))
                spanned += 1
            else:
                rows.append(bkd.full((npoints,), lo))
        return bkd.stack(rows, axis=0)

    def locate(
        self, points: Array, faces: Sequence[int], tol: float = 1e-12
    ) -> tuple[tuple[FacePoint[Array], ...], ...]:
        r"""Which of ``faces`` each physical point lies on, and where on
        their reference faces: the inverse of :meth:`face_points`.

        A point on the closure of several faces (an edge or vertex of the
        skeleton) lies on each of them. The reference coordinate along a
        spanned direction is :math:`2(x - l)/(u - l) - 1`, computed from
        the point's array entries, so it keeps their computation graph.

        Parameters
        ----------
        points
            Physical points. Shape: (ndim, npoints).
        faces
            The faces to search, e.g. a skeleton space's faces.
        tol
            How far outside a face, in physical distance per direction, a
            point may lie and still count as on it.

        Returns
        -------
        tuple[tuple[FacePoint, ...], ...]
            Per point, its faces in the order of ``faces``.

        Raises
        ------
        ValueError
            If ``points`` has the wrong shape, an entry of ``faces`` is not
            a face, or a point lies on none of ``faces``.
        """
        bkd, partition = self._bkd, self._partition
        ndim = partition.ndim()
        if points.ndim != 2 or points.shape[0] != ndim:
            raise ValueError(
                f"points must have shape ({ndim}, npoints), got {tuple(points.shape)}"
            )
        for face in faces:
            if partition.entity_dim(face) != ndim - 1:
                raise ValueError(f"entity {face} is not a face")
        bounds = [partition.entity_bounds(face) for face in faces]
        located = []
        for j in range(int(points.shape[1])):
            coords = [bkd.to_float(points[i, j]) for i in range(ndim)]
            found = []
            for face, extent in zip(faces, bounds):
                if not all(
                    lo - tol <= x <= up + tol for x, (lo, up) in zip(coords, extent)
                ):
                    continue
                reference = [
                    2.0 * (points[i, j] - lo) / (up - lo) - 1.0
                    for i, (lo, up) in enumerate(extent)
                    if up > lo
                ]
                point = (
                    bkd.reshape(bkd.stack(reference, axis=0), (ndim - 1, 1))
                    if reference
                    else bkd.zeros((0, 1))
                )
                found.append(FacePoint(face, point))
            if not found:
                raise ValueError(f"point {j} at {coords} lies on none of the faces")
            located.append(tuple(found))
        return tuple(located)


@runtime_checkable
class FaceDofsProtocol(Protocol):
    """What :func:`skeleton_dof_points` asks of a skeleton space."""

    def descriptor(self) -> SpaceDescriptor:
        """The space's descriptor; its size is the number of DOFs."""
        ...

    def faces(self) -> Sequence[int]:
        """The faces carrying nodes."""
        ...

    def face_dofs(self, face: int) -> tuple[int, ...]:
        """The DOF each of ``face``'s nodes reads, in node order."""
        ...


@runtime_checkable
class NodalBasisProtocol(Protocol[A_co]):
    """What :func:`skeleton_dof_points` asks of a face's basis."""

    def nodes(self) -> A_co:
        """The nodes on the reference face. Shape: (ndim - 1, nterms)."""
        ...


@runtime_checkable
class NodalBasesProtocol(Protocol[A_co]):
    """What :func:`skeleton_dof_points` asks for each face: its basis."""

    def face_basis(self, face: int) -> NodalBasisProtocol[A_co]: ...


def skeleton_dof_points(
    space: FaceDofsProtocol,
    geometry: FaceGeometryProtocol[Array],
    bases: NodalBasesProtocol[Array],
    tol: float = 1e-12,
) -> Array:
    r"""The physical point of each skeleton DOF, shape (ndim, ndofs):
    each face's basis nodes through its map.

    Every face reading a DOF must put it at the same point; a numbering
    or geometry that disagrees is refused, which checks the two against
    each other without a reference solution.

    Raises
    ------
    TypeError
        If ``space``, ``geometry`` or ``bases`` does not satisfy its
        protocol.
    ValueError
        If faces place one DOF at points farther apart than ``tol``, or
        some DOF is on no face.
    """
    if not isinstance(bases, NodalBasesProtocol):
        raise TypeError(
            f"bases must satisfy NodalBasesProtocol, got {type(bases).__name__}"
        )
    if not isinstance(space, FaceDofsProtocol):
        raise TypeError(
            f"space must satisfy FaceDofsProtocol, got {type(space).__name__}"
        )
    if not isinstance(geometry, FaceGeometryProtocol):
        raise TypeError(
            f"geometry must satisfy FaceGeometryProtocol, got {type(geometry).__name__}"
        )
    bkd = geometry.bkd()
    columns: dict[int, Array] = {}
    for face in space.faces():
        placed = geometry.face_points(face, bases.face_basis(face).nodes())
        for j, dof in enumerate(space.face_dofs(face)):
            point = placed[:, j]
            if dof not in columns:
                columns[dof] = point
            elif bkd.to_float(bkd.max(bkd.abs(columns[dof] - point))) > tol:
                raise ValueError(f"faces place DOF {dof} at different points")
    size = space.descriptor().size
    if sorted(columns) != list(range(size)):
        raise ValueError("some skeleton DOF lies on no face")
    return bkd.stack([columns[d] for d in range(size)], axis=1)
