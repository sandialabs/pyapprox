"""Transformed mesh classes for spectral collocation.

Provides mesh classes that combine tensor product structure with
coordinate transformations, computing gradient factors for derivative
matrix scaling.

These classes implement MeshWithTransformProtocol from protocols.mesh.
"""

from typing import Generic, Optional, Tuple

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.mesh.base import (
    compute_boundary_indices_1d,
    compute_boundary_indices_2d,
    compute_boundary_indices_3d,
    compute_cartesian_product,
)
from pyapprox.typing.pde.collocation.protocols.mesh import TransformProtocol


def _chebyshev_gauss_lobatto_points(npts: int, bkd: Backend[Array]) -> Array:
    """Generate Chebyshev-Gauss-Lobatto points on [-1, 1].

    Parameters
    ----------
    npts : int
        Number of points.
    bkd : Backend
        Computational backend.

    Returns
    -------
    Array
        Points in increasing order from -1 to 1. Shape: (npts,)
    """
    j = bkd.arange(npts)
    # cos(pi * j / (npts - 1)) gives 1 to -1, negate for -1 to 1
    return -bkd.cos(np.pi * j / (npts - 1))


class TransformedMesh1D(Generic[Array]):
    """1D mesh with coordinate transformation.

    Generates Chebyshev points on reference interval [-1, 1] and maps them
    to physical coordinates via the provided transform.

    Implements MeshWithTransformProtocol.

    Parameters
    ----------
    npts : int
        Number of mesh points.
    bkd : Backend
        Computational backend.
    transform : TransformProtocol, optional
        Coordinate transform. If None, uses identity (no scaling).
    """

    def __init__(
        self,
        npts: int,
        bkd: Backend[Array],
        transform: Optional[TransformProtocol[Array]] = None,
    ):
        self._bkd = bkd
        self._npts = npts
        self._transform = transform

        # Generate Chebyshev-Gauss-Lobatto points on [-1, 1]
        self._reference_nodes_1d_arr = _chebyshev_gauss_lobatto_points(npts, bkd)

        # Reference nodes as 2D array (1, npts)
        ref_nodes_2d = self._reference_nodes_1d_arr.reshape(1, -1)
        self._reference_points = ref_nodes_2d

        # Compute physical nodes, gradient factors, and Jacobians
        if transform is not None:
            self._points = transform.map_to_physical(ref_nodes_2d)
            self._gradient_factors = transform.gradient_factors(ref_nodes_2d)
            self._jacobian_matrix = transform.jacobian_matrix(ref_nodes_2d)
            self._jacobian_determinant = transform.jacobian_determinant(ref_nodes_2d)
        else:
            # Identity transform
            self._points = ref_nodes_2d
            self._gradient_factors = bkd.ones((npts, 1, 1))
            self._jacobian_matrix = bkd.ones((npts, 1, 1))
            self._jacobian_determinant = bkd.ones((npts,))

        # Compute boundary indices
        self._boundary_indices = compute_boundary_indices_1d(npts, bkd)

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return number of spatial dimensions."""
        return 1

    def npts(self) -> int:
        """Return total number of mesh points."""
        return self._npts

    def npts_per_dim(self) -> Tuple[int, ...]:
        """Return number of points in each dimension."""
        return (self._npts,)

    def nboundaries(self) -> int:
        """Return number of boundaries."""
        return 2

    def points(self) -> Array:
        """Return physical coordinates. Shape: (1, npts)"""
        return self._points

    def reference_points(self) -> Array:
        """Return reference coordinates. Shape: (1, npts)"""
        return self._reference_points

    def reference_nodes_1d(self, dim: int) -> Array:
        """Return 1D reference nodes for dimension. Shape: (npts,)"""
        if dim != 0:
            raise ValueError(f"dim must be 0 for 1D mesh, got {dim}")
        return self._reference_nodes_1d_arr

    def boundary_indices(self, boundary_id: int) -> Array:
        """Return indices of boundary points."""
        return self._boundary_indices[boundary_id]

    def transform(self) -> TransformProtocol[Array]:
        """Return the coordinate transform."""
        return self._transform

    def jacobian_matrix(self) -> Array:
        """Return Jacobian matrices. Shape: (npts, 1, 1)"""
        return self._jacobian_matrix

    def jacobian_determinant(self) -> Array:
        """Return Jacobian determinants. Shape: (npts,)"""
        return self._jacobian_determinant

    def gradient_factors(self) -> Array:
        """Return gradient factors. Shape: (npts, 1, 1)"""
        return self._gradient_factors


class TransformedMesh2D(Generic[Array]):
    """2D mesh with coordinate transformation.

    Generates tensor product Chebyshev points on reference square [-1, 1]^2
    and maps them to physical coordinates via the provided transform.

    Implements MeshWithTransformProtocol.

    Parameters
    ----------
    npts_x : int
        Number of points in x (first) direction.
    npts_y : int
        Number of points in y (second) direction.
    bkd : Backend
        Computational backend.
    transform : TransformProtocol, optional
        Coordinate transform. If None, uses identity.
    """

    def __init__(
        self,
        npts_x: int,
        npts_y: int,
        bkd: Backend[Array],
        transform: Optional[TransformProtocol[Array]] = None,
    ):
        self._bkd = bkd
        self._npts_x = npts_x
        self._npts_y = npts_y
        self._npts = npts_x * npts_y
        self._transform = transform

        # Generate Chebyshev points for each dimension
        self._reference_nodes_x = _chebyshev_gauss_lobatto_points(npts_x, bkd)
        self._reference_nodes_y = _chebyshev_gauss_lobatto_points(npts_y, bkd)

        # Tensor product (first dim varies fastest for Kronecker ordering)
        ref_nodes_2d = compute_cartesian_product(
            [self._reference_nodes_x, self._reference_nodes_y], bkd
        )
        self._reference_points = ref_nodes_2d

        # Compute physical nodes, gradient factors, and Jacobians
        if transform is not None:
            self._points = transform.map_to_physical(ref_nodes_2d)
            self._gradient_factors = transform.gradient_factors(ref_nodes_2d)
            self._jacobian_matrix = transform.jacobian_matrix(ref_nodes_2d)
            self._jacobian_determinant = transform.jacobian_determinant(ref_nodes_2d)
        else:
            self._points = ref_nodes_2d
            self._gradient_factors = bkd.zeros((self._npts, 2, 2))
            self._gradient_factors[:, 0, 0] = 1.0
            self._gradient_factors[:, 1, 1] = 1.0
            self._jacobian_matrix = bkd.zeros((self._npts, 2, 2))
            self._jacobian_matrix[:, 0, 0] = 1.0
            self._jacobian_matrix[:, 1, 1] = 1.0
            self._jacobian_determinant = bkd.ones((self._npts,))

        # Compute boundary indices
        self._boundary_indices = compute_boundary_indices_2d(npts_x, npts_y, bkd)

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return number of spatial dimensions."""
        return 2

    def npts(self) -> int:
        """Return total number of mesh points."""
        return self._npts

    def npts_per_dim(self) -> Tuple[int, ...]:
        """Return number of points in each dimension."""
        return (self._npts_x, self._npts_y)

    def nboundaries(self) -> int:
        """Return number of boundaries."""
        return 4

    def points(self) -> Array:
        """Return physical coordinates. Shape: (2, npts)"""
        return self._points

    def reference_points(self) -> Array:
        """Return reference coordinates. Shape: (2, npts)"""
        return self._reference_points

    def reference_nodes_1d(self, dim: int) -> Array:
        """Return 1D reference nodes for dimension. Shape: (npts_dim,)"""
        if dim == 0:
            return self._reference_nodes_x
        elif dim == 1:
            return self._reference_nodes_y
        raise ValueError(f"dim must be 0 or 1 for 2D mesh, got {dim}")

    def boundary_indices(self, boundary_id: int) -> Array:
        """Return indices of boundary points."""
        return self._boundary_indices[boundary_id]

    def transform(self) -> TransformProtocol[Array]:
        """Return the coordinate transform."""
        return self._transform

    def jacobian_matrix(self) -> Array:
        """Return Jacobian matrices. Shape: (npts, 2, 2)"""
        return self._jacobian_matrix

    def jacobian_determinant(self) -> Array:
        """Return Jacobian determinants. Shape: (npts,)"""
        return self._jacobian_determinant

    def gradient_factors(self) -> Array:
        """Return gradient factors. Shape: (npts, 2, 2)"""
        return self._gradient_factors


class TransformedMesh3D(Generic[Array]):
    """3D mesh with coordinate transformation.

    Generates tensor product Chebyshev points on reference cube [-1, 1]^3
    and maps them to physical coordinates via the provided transform.

    Implements MeshWithTransformProtocol.

    Parameters
    ----------
    npts_x, npts_y, npts_z : int
        Number of points in each direction.
    bkd : Backend
        Computational backend.
    transform : TransformProtocol, optional
        Coordinate transform. If None, uses identity.
    """

    def __init__(
        self,
        npts_x: int,
        npts_y: int,
        npts_z: int,
        bkd: Backend[Array],
        transform: Optional[TransformProtocol[Array]] = None,
    ):
        self._bkd = bkd
        self._npts_x = npts_x
        self._npts_y = npts_y
        self._npts_z = npts_z
        self._npts = npts_x * npts_y * npts_z
        self._transform = transform

        # Generate Chebyshev points for each dimension
        self._reference_nodes_x = _chebyshev_gauss_lobatto_points(npts_x, bkd)
        self._reference_nodes_y = _chebyshev_gauss_lobatto_points(npts_y, bkd)
        self._reference_nodes_z = _chebyshev_gauss_lobatto_points(npts_z, bkd)

        # Tensor product
        ref_nodes_3d = compute_cartesian_product(
            [
                self._reference_nodes_x,
                self._reference_nodes_y,
                self._reference_nodes_z,
            ],
            bkd,
        )
        self._reference_points = ref_nodes_3d

        # Compute physical nodes, gradient factors, and Jacobians
        if transform is not None:
            self._points = transform.map_to_physical(ref_nodes_3d)
            self._gradient_factors = transform.gradient_factors(ref_nodes_3d)
            self._jacobian_matrix = transform.jacobian_matrix(ref_nodes_3d)
            self._jacobian_determinant = transform.jacobian_determinant(ref_nodes_3d)
        else:
            self._points = ref_nodes_3d
            self._gradient_factors = bkd.zeros((self._npts, 3, 3))
            self._gradient_factors[:, 0, 0] = 1.0
            self._gradient_factors[:, 1, 1] = 1.0
            self._gradient_factors[:, 2, 2] = 1.0
            self._jacobian_matrix = bkd.zeros((self._npts, 3, 3))
            self._jacobian_matrix[:, 0, 0] = 1.0
            self._jacobian_matrix[:, 1, 1] = 1.0
            self._jacobian_matrix[:, 2, 2] = 1.0
            self._jacobian_determinant = bkd.ones((self._npts,))

        # Compute boundary indices
        self._boundary_indices = compute_boundary_indices_3d(
            npts_x, npts_y, npts_z, bkd
        )

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return number of spatial dimensions."""
        return 3

    def npts(self) -> int:
        """Return total number of mesh points."""
        return self._npts

    def npts_per_dim(self) -> Tuple[int, ...]:
        """Return number of points in each dimension."""
        return (self._npts_x, self._npts_y, self._npts_z)

    def nboundaries(self) -> int:
        """Return number of boundaries."""
        return 6

    def points(self) -> Array:
        """Return physical coordinates. Shape: (3, npts)"""
        return self._points

    def reference_points(self) -> Array:
        """Return reference coordinates. Shape: (3, npts)"""
        return self._reference_points

    def reference_nodes_1d(self, dim: int) -> Array:
        """Return 1D reference nodes for dimension. Shape: (npts_dim,)"""
        if dim == 0:
            return self._reference_nodes_x
        elif dim == 1:
            return self._reference_nodes_y
        elif dim == 2:
            return self._reference_nodes_z
        raise ValueError(f"dim must be 0, 1, or 2 for 3D mesh, got {dim}")

    def boundary_indices(self, boundary_id: int) -> Array:
        """Return indices of boundary points."""
        return self._boundary_indices[boundary_id]

    def transform(self) -> TransformProtocol[Array]:
        """Return the coordinate transform."""
        return self._transform

    def jacobian_matrix(self) -> Array:
        """Return Jacobian matrices. Shape: (npts, 3, 3)"""
        return self._jacobian_matrix

    def jacobian_determinant(self) -> Array:
        """Return Jacobian determinants. Shape: (npts,)"""
        return self._jacobian_determinant

    def gradient_factors(self) -> Array:
        """Return gradient factors. Shape: (npts, 3, 3)"""
        return self._gradient_factors
