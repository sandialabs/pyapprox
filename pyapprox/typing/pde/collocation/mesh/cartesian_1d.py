"""1D Cartesian mesh implementation.

Provides mesh construction for 1D domains with tensor-product structure.
"""

from typing import Generic, Tuple, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.mesh.base import (
    MeshData,
    MeshDataWithTransform,
    compute_boundary_indices_1d,
)
from pyapprox.typing.pde.collocation.mesh.transforms.affine import (
    AffineTransform1D,
)
from pyapprox.typing.pde.collocation.protocols.mesh import TransformProtocol


class CartesianMesh1D(Generic[Array]):
    """1D Cartesian mesh with optional coordinate transform.

    Constructs a mesh on a 1D interval using specified node distribution
    in reference space [-1, 1], optionally transformed to physical coordinates.

    Parameters
    ----------
    reference_pts_1d : Array
        1D array of node positions in reference space [-1, 1].
        Shape: (npts,)
    bkd : Backend
        Computational backend.
    transform : Optional[TransformProtocol]
        Coordinate transformation. If None, uses identity mapping
        (reference = physical).

    Attributes
    ----------
    _data : MeshData or MeshDataWithTransform
        Underlying mesh data storage.
    """

    def __init__(
        self,
        reference_pts_1d: Array,
        bkd: Backend[Array],
        transform: Optional[TransformProtocol[Array]] = None,
    ):
        self._bkd = bkd
        self._transform = transform
        npts = reference_pts_1d.shape[0]
        self._npts_per_dim = (npts,)

        # Reference points: shape (1, npts)
        self._reference_points = bkd.reshape(reference_pts_1d, (1, npts))

        # Compute physical points
        if transform is not None:
            self._points = transform.map_to_physical(self._reference_points)
        else:
            self._points = bkd.copy(self._reference_points)

        # Compute boundary indices
        self._boundary_indices = compute_boundary_indices_1d(npts, bkd)

        # Store transform data if available
        if transform is not None:
            jac_mat = transform.jacobian_matrix(self._reference_points)
            jac_det = transform.jacobian_determinant(self._reference_points)
            self._data = MeshDataWithTransform(
                points=self._points,
                reference_points=self._reference_points,
                npts_per_dim=self._npts_per_dim,
                boundary_indices=self._boundary_indices,
                jacobian_matrices=jac_mat,
                jacobian_determinants=jac_det,
            )
        else:
            self._data = MeshData(
                points=self._points,
                npts_per_dim=self._npts_per_dim,
                boundary_indices=self._boundary_indices,
            )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return 1

    def npts(self) -> int:
        """Return the total number of mesh points."""
        return self._npts_per_dim[0]

    def npts_per_dim(self) -> Tuple[int, ...]:
        """Return the number of points in each dimension."""
        return self._npts_per_dim

    def points(self) -> Array:
        """Return mesh coordinates in physical space.

        Returns
        -------
        Array
            Physical coordinates. Shape: (1, npts)
        """
        return self._points

    def reference_points(self) -> Array:
        """Return mesh coordinates in reference space.

        Returns
        -------
        Array
            Reference coordinates. Shape: (1, npts)
        """
        return self._reference_points

    def boundary_indices(self, boundary_id: int) -> Array:
        """Return indices of points on specified boundary.

        Parameters
        ----------
        boundary_id : int
            Boundary identifier: 0=left, 1=right

        Returns
        -------
        Array
            Integer array of point indices on the boundary.
        """
        if boundary_id < 0 or boundary_id >= 2:
            raise ValueError(
                f"Invalid boundary_id {boundary_id} for 1D mesh. "
                "Must be 0 (left) or 1 (right)."
            )
        return self._boundary_indices[boundary_id]

    def nboundaries(self) -> int:
        """Return the number of boundaries (2 for 1D)."""
        return 2

    def transform(self) -> Optional[TransformProtocol[Array]]:
        """Return the coordinate transformation, if any."""
        return self._transform

    def jacobian_matrix(self) -> Array:
        """Return Jacobian matrices at mesh points.

        Returns
        -------
        Array
            Jacobian matrices. Shape: (npts, 1, 1)
        """
        if isinstance(self._data, MeshDataWithTransform):
            return self._data.jacobian_matrices
        # Identity transform: Jacobian is 1
        npts = self.npts()
        return self._bkd.ones((npts, 1, 1))

    def jacobian_determinant(self) -> Array:
        """Return Jacobian determinants at mesh points.

        Returns
        -------
        Array
            Jacobian determinants. Shape: (npts,)
        """
        if isinstance(self._data, MeshDataWithTransform):
            return self._data.jacobian_determinants
        # Identity transform: determinant is 1
        return self._bkd.ones((self.npts(),))

    @property
    def data(self):
        """Return underlying mesh data structure."""
        return self._data


def create_uniform_mesh_1d(
    npts: int,
    physical_bounds: Tuple[float, float],
    bkd: Backend[Array],
) -> CartesianMesh1D[Array]:
    """Create a 1D mesh with uniformly spaced points.

    Parameters
    ----------
    npts : int
        Number of mesh points.
    physical_bounds : Tuple[float, float]
        Physical domain bounds (a, b).
    bkd : Backend
        Computational backend.

    Returns
    -------
    CartesianMesh1D
        Mesh with uniform spacing in physical coordinates.
    """
    # Uniform points in reference space [-1, 1]
    reference_pts = bkd.linspace(-1.0, 1.0, npts)
    transform = AffineTransform1D(physical_bounds, bkd)
    return CartesianMesh1D(reference_pts, bkd, transform)
