"""Base mesh data structures for spectral collocation methods.

Provides pure data structures (dataclasses) for mesh representation,
separating data from algorithms for C++ portability.
"""

from dataclasses import dataclass
from typing import Generic, List, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.cartesian import cartesian_product_samples


@dataclass
class MeshData(Generic[Array]):
    """Pure data structure for mesh storage.

    This dataclass stores mesh data without algorithms, making it
    easy to serialize and port to C++.

    Attributes
    ----------
    points : Array
        Physical coordinates of mesh points. Shape: (ndim, npts)
    npts_per_dim : Tuple[int, ...]
        Number of points in each dimension.
    boundary_indices : List[Array]
        List of index arrays, one per boundary.
        - 1D: [left_indices, right_indices]
        - 2D: [left, right, bottom, top]
        - 3D: [left, right, bottom, top, front, back]
    """

    points: Array
    npts_per_dim: Tuple[int, ...]
    boundary_indices: List[Array]

    @property
    def ndim(self) -> int:
        """Return number of spatial dimensions."""
        return len(self.npts_per_dim)

    @property
    def npts(self) -> int:
        """Return total number of mesh points."""
        return self.points.shape[1]

    @property
    def nboundaries(self) -> int:
        """Return number of boundaries."""
        return len(self.boundary_indices)


@dataclass
class MeshDataWithTransform(Generic[Array]):
    """Mesh data with coordinate transformation information.

    Extends MeshData with reference coordinates and transformation
    Jacobians for non-affine mappings.

    Attributes
    ----------
    points : Array
        Physical coordinates. Shape: (ndim, npts)
    reference_points : Array
        Reference coordinates (typically [-1, 1]^d). Shape: (ndim, npts)
    npts_per_dim : Tuple[int, ...]
        Number of points in each dimension.
    boundary_indices : List[Array]
        Index arrays for each boundary.
    jacobian_matrices : Array
        Jacobian matrices at each point. Shape: (npts, ndim, ndim)
    jacobian_determinants : Array
        Jacobian determinants at each point. Shape: (npts,)
    """

    points: Array
    reference_points: Array
    npts_per_dim: Tuple[int, ...]
    boundary_indices: List[Array]
    jacobian_matrices: Array
    jacobian_determinants: Array

    @property
    def ndim(self) -> int:
        """Return number of spatial dimensions."""
        return len(self.npts_per_dim)

    @property
    def npts(self) -> int:
        """Return total number of mesh points."""
        return self.points.shape[1]

    @property
    def nboundaries(self) -> int:
        """Return number of boundaries."""
        return len(self.boundary_indices)


def compute_cartesian_product(arrays_1d: List[Array], bkd: Backend[Array]) -> Array:
    """Compute Cartesian product of 1D point arrays.

    Parameters
    ----------
    arrays_1d : List[Array]
        List of 1D arrays, one per dimension. Each has shape (npts_i,)
    bkd : Backend
        Computational backend.

    Returns
    -------
    Array
        Cartesian product. Shape: (ndim, prod(npts_i))
        Points are ordered with first dimension varying fastest.

    Notes
    -----
    This function uses first_dim_fastest=True ordering which matches
    Kronecker product conventions used in spectral methods.
    """
    return cartesian_product_samples(arrays_1d, bkd, first_dim_fastest=True)


def compute_boundary_indices_1d(npts: int, bkd: Backend[Array]) -> List[Array]:
    """Compute boundary indices for 1D mesh.

    Parameters
    ----------
    npts : int
        Number of mesh points.
    bkd : Backend
        Computational backend.

    Returns
    -------
    List[Array]
        [left_indices, right_indices] where left is index 0, right is index npts-1.
    """
    left = bkd.asarray([0], dtype=bkd.int64_dtype())
    right = bkd.asarray([npts - 1], dtype=bkd.int64_dtype())
    return [left, right]


def compute_boundary_indices_2d(
    npts_x: int, npts_y: int, bkd: Backend[Array]
) -> List[Array]:
    """Compute boundary indices for 2D mesh.

    Assumes tensor product ordering with x varying fastest.

    Parameters
    ----------
    npts_x : int
        Number of points in x direction.
    npts_y : int
        Number of points in y direction.
    bkd : Backend
        Computational backend.

    Returns
    -------
    List[Array]
        [left, right, bottom, top] boundary indices.
    """
    # Left boundary: x = 0, all y -> indices 0, npts_x, 2*npts_x, ...
    left = bkd.asarray([j * npts_x for j in range(npts_y)], dtype=bkd.int64_dtype())

    # Right boundary: x = npts_x-1, all y
    right = bkd.asarray(
        [j * npts_x + (npts_x - 1) for j in range(npts_y)],
        dtype=bkd.int64_dtype(),
    )

    # Bottom boundary: y = 0, all x -> indices 0, 1, ..., npts_x-1
    bottom = bkd.asarray(list(range(npts_x)), dtype=bkd.int64_dtype())

    # Top boundary: y = npts_y-1, all x
    top = bkd.asarray(
        [(npts_y - 1) * npts_x + i for i in range(npts_x)],
        dtype=bkd.int64_dtype(),
    )

    return [left, right, bottom, top]


def compute_boundary_indices_3d(
    npts_x: int, npts_y: int, npts_z: int, bkd: Backend[Array]
) -> List[Array]:
    """Compute boundary indices for 3D mesh.

    Assumes tensor product ordering with x varying fastest, then y, then z.

    Parameters
    ----------
    npts_x, npts_y, npts_z : int
        Number of points in each direction.
    bkd : Backend
        Computational backend.

    Returns
    -------
    List[Array]
        [left, right, bottom, top, front, back] boundary indices.
    """
    npts_xy = npts_x * npts_y

    # Left boundary: x = 0
    left_indices = []
    for k in range(npts_z):
        for j in range(npts_y):
            left_indices.append(k * npts_xy + j * npts_x)
    left = bkd.asarray(left_indices, dtype=bkd.int64_dtype())

    # Right boundary: x = npts_x - 1
    right_indices = []
    for k in range(npts_z):
        for j in range(npts_y):
            right_indices.append(k * npts_xy + j * npts_x + (npts_x - 1))
    right = bkd.asarray(right_indices, dtype=bkd.int64_dtype())

    # Bottom boundary: y = 0
    bottom_indices = []
    for k in range(npts_z):
        for i in range(npts_x):
            bottom_indices.append(k * npts_xy + i)
    bottom = bkd.asarray(bottom_indices, dtype=bkd.int64_dtype())

    # Top boundary: y = npts_y - 1
    top_indices = []
    for k in range(npts_z):
        for i in range(npts_x):
            top_indices.append(k * npts_xy + (npts_y - 1) * npts_x + i)
    top = bkd.asarray(top_indices, dtype=bkd.int64_dtype())

    # Front boundary: z = 0
    front_indices = []
    for j in range(npts_y):
        for i in range(npts_x):
            front_indices.append(j * npts_x + i)
    front = bkd.asarray(front_indices, dtype=bkd.int64_dtype())

    # Back boundary: z = npts_z - 1
    back_indices = []
    for j in range(npts_y):
        for i in range(npts_x):
            back_indices.append((npts_z - 1) * npts_xy + j * npts_x + i)
    back = bkd.asarray(back_indices, dtype=bkd.int64_dtype())

    return [left, right, bottom, top, front, back]
