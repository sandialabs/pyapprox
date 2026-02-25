"""Mesh protocols for spectral collocation methods.

Defines interfaces for computational meshes supporting 1D, 2D, and 3D domains.
"""

from typing import Protocol, Generic, runtime_checkable, Tuple

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class MeshProtocol(Protocol, Generic[Array]):
    """Protocol for computational meshes.

    A mesh represents a discretization of a spatial domain using
    collocation points. Supports 1D, 2D, and 3D domains.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def ndim(self) -> int:
        """Return the number of spatial dimensions (1, 2, or 3)."""
        ...

    def npts(self) -> int:
        """Return the total number of mesh points."""
        ...

    def npts_per_dim(self) -> Tuple[int, ...]:
        """Return the number of points in each dimension.

        Returns
        -------
        Tuple[int, ...]
            Tuple of length ndim with point counts.
            E.g., (10,) for 1D, (10, 15) for 2D.
        """
        ...

    def points(self) -> Array:
        """Return mesh coordinates in physical space.

        Returns
        -------
        Array
            Coordinates of shape (ndim, npts).
        """
        ...

    def boundary_indices(self, boundary_id: int) -> Array:
        """Return indices of points on specified boundary.

        Parameters
        ----------
        boundary_id : int
            Boundary identifier:
            - 1D: 0=left, 1=right
            - 2D: 0=left, 1=right, 2=bottom, 3=top
            - 3D: 0=left, 1=right, 2=bottom, 3=top, 4=front, 5=back

        Returns
        -------
        Array
            Integer array of point indices on the boundary.
        """
        ...

    def nboundaries(self) -> int:
        """Return the number of boundaries.

        Returns
        -------
        int
            Number of boundaries (2 for 1D, 4 for 2D, 6 for 3D).
        """
        ...


@runtime_checkable
class TransformProtocol(Protocol, Generic[Array]):
    """Protocol for coordinate transformations.

    Maps between reference coordinates (typically [-1, 1]^d) and
    physical coordinates.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        ...

    def map_to_physical(self, reference_pts: Array) -> Array:
        """Map from reference to physical coordinates.

        Parameters
        ----------
        reference_pts : Array
            Points in reference space. Shape: (ndim, npts)

        Returns
        -------
        Array
            Points in physical space. Shape: (ndim, npts)
        """
        ...

    def map_to_reference(self, physical_pts: Array) -> Array:
        """Map from physical to reference coordinates.

        Parameters
        ----------
        physical_pts : Array
            Points in physical space. Shape: (ndim, npts)

        Returns
        -------
        Array
            Points in reference space. Shape: (ndim, npts)
        """
        ...

    def jacobian_matrix(self, reference_pts: Array) -> Array:
        """Compute Jacobian matrix of the transformation.

        The Jacobian J[i,j] = dx_i/dxi_j where x is physical and xi is reference.

        Parameters
        ----------
        reference_pts : Array
            Points in reference space. Shape: (ndim, npts)

        Returns
        -------
        Array
            Jacobian matrices. Shape: (npts, ndim, ndim)
        """
        ...

    def jacobian_determinant(self, reference_pts: Array) -> Array:
        """Compute determinant of Jacobian matrix.

        Parameters
        ----------
        reference_pts : Array
            Points in reference space. Shape: (ndim, npts)

        Returns
        -------
        Array
            Jacobian determinants. Shape: (npts,)
        """
        ...


@runtime_checkable
class MeshWithTransformProtocol(Protocol, Generic[Array]):
    """Protocol for mesh with coordinate transformation support.

    Extends MeshProtocol with access to reference coordinates and
    transformation Jacobians.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        ...

    def npts(self) -> int:
        """Return the total number of mesh points."""
        ...

    def npts_per_dim(self) -> Tuple[int, ...]:
        """Return the number of points in each dimension."""
        ...

    def points(self) -> Array:
        """Return mesh coordinates in physical space. Shape: (ndim, npts)."""
        ...

    def boundary_indices(self, boundary_id: int) -> Array:
        """Return indices of points on specified boundary."""
        ...

    def nboundaries(self) -> int:
        """Return the number of boundaries."""
        ...

    def reference_points(self) -> Array:
        """Return mesh coordinates in reference space.

        Returns
        -------
        Array
            Reference coordinates. Shape: (ndim, npts)
        """
        ...

    def transform(self) -> TransformProtocol[Array]:
        """Return the coordinate transformation."""
        ...

    def jacobian_matrix(self) -> Array:
        """Return Jacobian matrices at mesh points.

        Returns
        -------
        Array
            Jacobian matrices. Shape: (npts, ndim, ndim)
        """
        ...

    def jacobian_determinant(self) -> Array:
        """Return Jacobian determinants at mesh points.

        Returns
        -------
        Array
            Jacobian determinants. Shape: (npts,)
        """
        ...

    def gradient_factors(self) -> Array:
        """Return gradient factors for converting reference to physical derivatives.

        The gradient factors G are used to compute physical derivatives from
        reference derivatives:
            d/dx_phys[d] = sum_j G[:, d, j] * d/d_xi_ref[j]

        For Cartesian (untransformed) meshes, this is the identity matrix.
        For affine transforms, it's diagonal with 1/scale entries.
        For curvilinear transforms, it's the inverse of the metric tensor.

        Returns
        -------
        Array
            Gradient factors. Shape: (npts, ndim, ndim)
        """
        ...

    def reference_nodes_1d(self, dim: int) -> Array:
        """Return 1D reference nodes for specified dimension.

        For tensor product meshes, the full reference coordinates are the
        Cartesian product of 1D reference nodes. This method returns the
        1D nodes for a single dimension.

        Parameters
        ----------
        dim : int
            Dimension index (0 for x, 1 for y, 2 for z).

        Returns
        -------
        Array
            1D reference nodes. Shape: (npts_dim,)
        """
        ...
