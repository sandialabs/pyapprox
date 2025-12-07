"""2D Cartesian mesh implementation.

Provides mesh construction for 2D domains with tensor-product structure.
"""

from typing import Generic, Tuple, Optional, List

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.mesh.base import (
    MeshData,
    MeshDataWithTransform,
    compute_cartesian_product,
    compute_boundary_indices_2d,
)
from pyapprox.typing.pde.collocation.mesh.transforms.affine import (
    AffineTransform2D,
)
from pyapprox.typing.pde.collocation.protocols.mesh import TransformProtocol


class CartesianMesh2D(Generic[Array]):
    """2D Cartesian mesh with optional coordinate transform.

    Constructs a mesh on a 2D rectangle using tensor product of 1D node
    distributions in reference space [-1, 1]^2, optionally transformed
    to physical coordinates.

    Parameters
    ----------
    reference_pts_1d : List[Array]
        List of two 1D arrays of node positions in reference space.
        [x_nodes, y_nodes] where each has shape (npts_i,)
    bkd : Backend
        Computational backend.
    transform : Optional[TransformProtocol]
        Coordinate transformation. If None, uses identity mapping.
    """

    def __init__(
        self,
        reference_pts_1d: List[Array],
        bkd: Backend[Array],
        transform: Optional[TransformProtocol[Array]] = None,
    ):
        if len(reference_pts_1d) != 2:
            raise ValueError(
                f"Expected 2 arrays for 2D mesh, got {len(reference_pts_1d)}"
            )

        self._bkd = bkd
        self._transform = transform
        npts_x = reference_pts_1d[0].shape[0]
        npts_y = reference_pts_1d[1].shape[0]
        self._npts_per_dim = (npts_x, npts_y)

        # Reference points via Cartesian product: shape (2, npts_x * npts_y)
        self._reference_points = compute_cartesian_product(
            reference_pts_1d, bkd
        )

        # Compute physical points
        if transform is not None:
            self._points = transform.map_to_physical(self._reference_points)
        else:
            self._points = bkd.copy(self._reference_points)

        # Compute boundary indices
        self._boundary_indices = compute_boundary_indices_2d(
            npts_x, npts_y, bkd
        )

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
        return 2

    def npts(self) -> int:
        """Return the total number of mesh points."""
        return self._npts_per_dim[0] * self._npts_per_dim[1]

    def npts_per_dim(self) -> Tuple[int, ...]:
        """Return the number of points in each dimension."""
        return self._npts_per_dim

    def points(self) -> Array:
        """Return mesh coordinates in physical space.

        Returns
        -------
        Array
            Physical coordinates. Shape: (2, npts)
        """
        return self._points

    def reference_points(self) -> Array:
        """Return mesh coordinates in reference space.

        Returns
        -------
        Array
            Reference coordinates. Shape: (2, npts)
        """
        return self._reference_points

    def boundary_indices(self, boundary_id: int) -> Array:
        """Return indices of points on specified boundary.

        Parameters
        ----------
        boundary_id : int
            Boundary identifier: 0=left, 1=right, 2=bottom, 3=top

        Returns
        -------
        Array
            Integer array of point indices on the boundary.
        """
        if boundary_id < 0 or boundary_id >= 4:
            raise ValueError(
                f"Invalid boundary_id {boundary_id} for 2D mesh. "
                "Must be 0 (left), 1 (right), 2 (bottom), or 3 (top)."
            )
        return self._boundary_indices[boundary_id]

    def nboundaries(self) -> int:
        """Return the number of boundaries (4 for 2D)."""
        return 4

    def transform(self) -> Optional[TransformProtocol[Array]]:
        """Return the coordinate transformation, if any."""
        return self._transform

    def jacobian_matrix(self) -> Array:
        """Return Jacobian matrices at mesh points.

        Returns
        -------
        Array
            Jacobian matrices. Shape: (npts, 2, 2)
        """
        if isinstance(self._data, MeshDataWithTransform):
            return self._data.jacobian_matrices
        # Identity transform: Jacobian is identity
        npts = self.npts()
        jac = self._bkd.zeros((npts, 2, 2))
        jac[:, 0, 0] = 1.0
        jac[:, 1, 1] = 1.0
        return jac

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


def create_uniform_mesh_2d(
    npts: Tuple[int, int],
    physical_bounds: Tuple[float, float, float, float],
    bkd: Backend[Array],
) -> CartesianMesh2D[Array]:
    """Create a 2D mesh with uniformly spaced points.

    Parameters
    ----------
    npts : Tuple[int, int]
        Number of mesh points (npts_x, npts_y).
    physical_bounds : Tuple[float, float, float, float]
        Physical domain bounds (ax, bx, ay, by).
    bkd : Backend
        Computational backend.

    Returns
    -------
    CartesianMesh2D
        Mesh with uniform spacing in physical coordinates.
    """
    npts_x, npts_y = npts
    # Uniform points in reference space [-1, 1]
    reference_pts = [
        bkd.linspace(-1.0, 1.0, npts_x),
        bkd.linspace(-1.0, 1.0, npts_y),
    ]
    transform = AffineTransform2D(physical_bounds, bkd)
    return CartesianMesh2D(reference_pts, bkd, transform)
