"""Transformed tensor-product meshes for spectral collocation.

:class:`TransformedMesh` places tensor-product nodes on the reference cube
:math:`[-1, 1]^d`, any number of directions, and maps them through a
transform, keeping the Jacobians and gradient factors the derivative
matrices need. ``TransformedMesh1D``, ``TransformedMesh2D`` and
``TransformedMesh3D`` are the same mesh built from per-direction counts.

These classes implement ``MeshWithTransformProtocol`` from
``protocols.mesh``.
"""

import math
from typing import Generic, List, Optional, Sequence, Tuple

from pyapprox.pde.collocation.basis.chebyshev.nodal import ChebyshevGaussLobatto1D
from pyapprox.pde.collocation.mesh.base import (
    compute_boundary_indices,
    compute_cartesian_product,
)
from pyapprox.pde.collocation.mesh.transforms.affine import (
    AffineTransform,
    AffineTransform1D,
    AffineTransform2D,
    AffineTransform3D,
)
from pyapprox.pde.collocation.protocols.basis_1d import CollocationBasis1DProtocol
from pyapprox.pde.collocation.protocols.mesh import TransformProtocol
from pyapprox.util.backends.protocols import Array, Backend


class TransformedMesh(Generic[Array]):
    """Tensor-product mesh of the reference cube mapped by a transform.

    Each direction has a 1D basis, which gives its nodes and which the
    tensor-product basis on the mesh differentiates with. Nodes are in
    tensor order, the first direction fastest. Boundary ``2 d`` is the
    lower end of reference direction ``d`` and ``2 d + 1`` the upper; it
    exists only when that direction's nodes reach the end
    (Chebyshev-Gauss nodes do not).

    Implements ``MeshWithTransformProtocol``.

    Parameters
    ----------
    npts_per_dim : Sequence[int]
        Nodes per direction, at least 1 each (a basis may need more).
    bkd : Backend
        Computational backend.
    transform : TransformProtocol, optional
        The map from the reference cube, with as many directions. Default:
        the reference cube itself.
    bases : Sequence[CollocationBasis1DProtocol], optional
        One 1D basis per direction. Default: Chebyshev-Gauss-Lobatto in
        every direction.

    Raises
    ------
    ValueError
        If ``npts_per_dim`` is empty or has a count below 1, a basis
        refuses its count, or the transform or the bases do not match its
        length.
    TypeError
        If a basis does not satisfy ``CollocationBasis1DProtocol``.
    """

    def __init__(
        self,
        npts_per_dim: Sequence[int],
        bkd: Backend[Array],
        transform: Optional[TransformProtocol[Array]] = None,
        bases: Optional[Sequence[CollocationBasis1DProtocol[Array]]] = None,
    ) -> None:
        counts = tuple(int(n) for n in npts_per_dim)
        if len(counts) == 0:
            raise ValueError("a mesh needs at least one direction")
        if min(counts) < 1:
            raise ValueError(f"every direction needs at least 1 node, got {counts}")
        ndim = len(counts)
        if transform is None:
            transform = AffineTransform([(-1.0, 1.0)] * ndim, bkd)
        if transform.ndim() != ndim:
            raise ValueError(
                f"the transform has {transform.ndim()} directions, the mesh {ndim}"
            )
        chosen: Tuple[CollocationBasis1DProtocol[Array], ...] = (
            tuple(ChebyshevGaussLobatto1D[Array]() for _ in counts)
            if bases is None
            else tuple(bases)
        )
        if len(chosen) != ndim:
            raise ValueError(f"{len(chosen)} 1D bases for {ndim} directions")
        for basis in chosen:
            if not isinstance(basis, CollocationBasis1DProtocol):
                raise TypeError(
                    "1D bases must satisfy CollocationBasis1DProtocol, got "
                    f"{type(basis).__name__}"
                )
        self._bkd = bkd
        self._npts_per_dim = counts
        self._transform: TransformProtocol[Array] = transform
        self._bases = chosen
        self._reference_nodes: List[Array] = [
            basis.nodes(count, bkd) for basis, count in zip(chosen, counts)
        ]
        reference = compute_cartesian_product(self._reference_nodes, bkd)
        self._reference_points = reference
        self._points = transform.map_to_physical(reference)
        self._gradient_factors = transform.gradient_factors(reference)
        self._jacobian_matrix = transform.jacobian_matrix(reference)
        self._jacobian_determinant = transform.jacobian_determinant(reference)
        self._boundary_indices = compute_boundary_indices(counts, bkd)
        # Boundary 2 d + s exists when direction d's nodes reach end s.
        self._has_boundary = tuple(
            bkd.to_float(nodes[-1 if side else 0]) == (1.0 if side else -1.0)
            for nodes in self._reference_nodes
            for side in (0, 1)
        )

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return the number of directions."""
        return len(self._npts_per_dim)

    def npts(self) -> int:
        """Return the total number of nodes."""
        return math.prod(self._npts_per_dim)

    def npts_per_dim(self) -> Tuple[int, ...]:
        """Return the number of nodes in each direction."""
        return self._npts_per_dim

    def nboundaries(self) -> int:
        """Return the number of boundaries, two per direction."""
        return 2 * self.ndim()

    def points(self) -> Array:
        """Return physical coordinates. Shape: ``(ndim, npts)``."""
        return self._points

    def reference_points(self) -> Array:
        """Return reference coordinates. Shape: ``(ndim, npts)``."""
        return self._reference_points

    def reference_nodes_1d(self, dim: int) -> Array:
        """Return the reference nodes of direction ``dim``. Shape:
        ``(npts_dim,)``."""
        if not 0 <= dim < self.ndim():
            raise ValueError(f"dim must be in [0, {self.ndim()}), got {dim}")
        return self._reference_nodes[dim]

    def basis_1d(self, dim: int) -> CollocationBasis1DProtocol[Array]:
        """Return the 1D basis of direction ``dim``, which gave its nodes."""
        if not 0 <= dim < self.ndim():
            raise ValueError(f"dim must be in [0, {self.ndim()}), got {dim}")
        return self._bases[dim]

    def boundary_indices(self, boundary_id: int) -> Array:
        """Return the indices of the nodes on boundary ``boundary_id``."""
        self._check_boundary(boundary_id)
        return self._boundary_indices[boundary_id]

    def transform(self) -> TransformProtocol[Array]:
        """Return the coordinate transform."""
        return self._transform

    def jacobian_matrix(self) -> Array:
        """Return Jacobian matrices. Shape: ``(npts, ndim, ndim)``."""
        return self._jacobian_matrix

    def jacobian_determinant(self) -> Array:
        """Return Jacobian determinants. Shape: ``(npts,)``."""
        return self._jacobian_determinant

    def gradient_factors(self) -> Array:
        """Return gradient factors. Shape: ``(npts, ndim, ndim)``."""
        return self._gradient_factors

    def boundary_normals(self, boundary_id: int) -> Array:
        """Return outward unit normals at the nodes of a boundary.

        Boundary ``2 d + s`` (``s = 0`` lower, ``1`` upper) has reference
        normal :math:`(2 s - 1) e_d`, mapped to :math:`J^{-T} e_d`, which
        is row ``d`` of :math:`J^{-1}` (the gradient of the reference
        coordinate :math:`\\xi_d`), then normalized.

        Returns
        -------
        Array
            Unit normals in physical coordinates. Shape:
            ``(nboundary_pts, ndim)``.
        """
        self._check_boundary(boundary_id)
        direction, side = divmod(boundary_id, 2)
        jacobians = self._jacobian_matrix[self._boundary_indices[boundary_id]]
        gradients = self._bkd.inv(jacobians)[:, direction, :]
        lengths = self._bkd.sqrt(self._bkd.sum(gradients**2, axis=1))
        return (2 * side - 1) * gradients / lengths[:, None]

    def _check_boundary(self, boundary_id: int) -> None:
        if not 0 <= boundary_id < self.nboundaries():
            raise ValueError(
                f"boundary_id must be in [0, {self.nboundaries()}), got {boundary_id}"
            )
        if not self._has_boundary[boundary_id]:
            direction, side = divmod(boundary_id, 2)
            raise ValueError(
                f"boundary {boundary_id} does not exist: the nodes of direction "
                f"{direction} do not reach its {'upper' if side else 'lower'} end"
            )


class TransformedMesh1D(TransformedMesh[Array]):
    """:class:`TransformedMesh` of an interval.

    Parameters
    ----------
    npts : int
        Number of nodes.
    bkd : Backend
        Computational backend.
    transform : TransformProtocol, optional
        Coordinate transform. Default: ``AffineTransform1D((-1.0, 1.0),
        bkd)``.
    """

    def __init__(
        self,
        npts: int,
        bkd: Backend[Array],
        transform: Optional[TransformProtocol[Array]] = None,
    ) -> None:
        super().__init__(
            (npts,),
            bkd,
            AffineTransform1D((-1.0, 1.0), bkd) if transform is None else transform,
        )


class TransformedMesh2D(TransformedMesh[Array]):
    """:class:`TransformedMesh` in two directions.

    Parameters
    ----------
    npts_x, npts_y : int
        Nodes in the first and second direction.
    bkd : Backend
        Computational backend.
    transform : TransformProtocol, optional
        Coordinate transform. Default: ``AffineTransform2D((-1.0, 1.0,
        -1.0, 1.0), bkd)``.
    """

    def __init__(
        self,
        npts_x: int,
        npts_y: int,
        bkd: Backend[Array],
        transform: Optional[TransformProtocol[Array]] = None,
    ) -> None:
        super().__init__(
            (npts_x, npts_y),
            bkd,
            (
                AffineTransform2D((-1.0, 1.0, -1.0, 1.0), bkd)
                if transform is None
                else transform
            ),
        )


class TransformedMesh3D(TransformedMesh[Array]):
    """:class:`TransformedMesh` in three directions.

    Parameters
    ----------
    npts_x, npts_y, npts_z : int
        Nodes in each direction.
    bkd : Backend
        Computational backend.
    transform : TransformProtocol, optional
        Coordinate transform. Default: ``AffineTransform3D((-1.0, 1.0,
        -1.0, 1.0, -1.0, 1.0), bkd)``.
    """

    def __init__(
        self,
        npts_x: int,
        npts_y: int,
        npts_z: int,
        bkd: Backend[Array],
        transform: Optional[TransformProtocol[Array]] = None,
    ) -> None:
        super().__init__(
            (npts_x, npts_y, npts_z),
            bkd,
            (
                AffineTransform3D((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), bkd)
                if transform is None
                else transform
            ),
        )
