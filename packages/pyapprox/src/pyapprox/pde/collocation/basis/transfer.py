"""Transfer of nodal values between two tensor meshes of one geometry.

A staggered discretization keeps fields on different nodes of the same
mapped reference cube (velocity on Gauss-Lobatto nodes, pressure on
interior Gauss nodes); a term coupling them reads one field at the
other's nodes through the source mesh's interpolant.
"""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.pde.collocation.protocols.basis_1d import CollocationBasis1DProtocol
from pyapprox.util.backends.protocols import Array, Backend


class TransferMapProtocol(Protocol, Generic[Array]):
    """A mesh's map of the reference cube onto the domain, as a transfer
    reads it: two meshes must hold the same map object."""

    def map_to_physical(self, reference_pts: Array) -> Array:
        """Shape: (d, npts) to (ncoords, npts)"""
        ...


@runtime_checkable
class TransferMeshProtocol(Protocol, Generic[Array]):
    """What a transfer reads from a mesh: its tensor nodes in the
    reference cube, first direction fastest, the 1D basis that made each
    direction's nodes, and its map."""

    def npts_per_dim(self) -> Tuple[int, ...]: ...

    def reference_nodes_1d(self, dim: int) -> Array:
        """Reference nodes of direction ``dim``. Shape: (npts_dim,)"""
        ...

    def basis_1d(self, dim: int) -> CollocationBasis1DProtocol[Array]:
        """The 1D basis of direction ``dim``."""
        ...

    def transform(self) -> TransferMapProtocol[Array]:
        """The map of the reference cube onto the domain; only its
        identity is read."""
        ...


def grid_transfer_matrix(
    source: TransferMeshProtocol[Array],
    target: TransferMeshProtocol[Array],
    bkd: Backend[Array],
) -> Array:
    """The interpolant of nodal values on ``source`` at ``target``'s nodes.

    Built direction by direction in the reference cube: each factor is the
    source direction's 1D interpolation at the target direction's nodes,
    and the matrix is their Kronecker product, last direction outermost.
    It is exact on polynomials of degree below the source's node count in
    each reference direction, so on an affine map exact on physical
    polynomials of those degrees.

    Parameters
    ----------
    source : TransferMeshProtocol
        The mesh the values live on.
    target : TransferMeshProtocol
        The mesh they are wanted on.
    bkd : Backend
        The backend of the matrix.

    Returns
    -------
    Array
        Shape: (target npts, source npts)

    Raises
    ------
    TypeError
        If a mesh does not satisfy ``TransferMeshProtocol``.
    ValueError
        If the meshes have different numbers of directions or different
        maps (the same reference point would be different physical points).
    """
    for name, mesh in (("source", source), ("target", target)):
        if not isinstance(mesh, TransferMeshProtocol):
            raise TypeError(
                f"{name} must satisfy TransferMeshProtocol, got {type(mesh).__name__}"
            )
    sizes = source.npts_per_dim()
    ndirections = len(sizes)
    if len(target.npts_per_dim()) != ndirections:
        raise ValueError(
            f"the source mesh has {ndirections} directions, the target "
            f"{len(target.npts_per_dim())}"
        )
    if source.transform() is not target.transform():
        raise ValueError(
            "the meshes must share one map, so a reference point is one "
            "physical point on both"
        )
    factors = [
        source.basis_1d(d).interpolation_matrix(
            sizes[d], target.reference_nodes_1d(d), bkd
        )
        for d in range(ndirections)
    ]
    # Node index i_0 + n_0 i_1 + ...: the last direction is the outer factor.
    matrix = factors[-1]
    for factor in factors[-2::-1]:
        matrix = bkd.kron(matrix, factor)
    return matrix
