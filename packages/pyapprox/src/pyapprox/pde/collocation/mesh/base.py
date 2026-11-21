"""Base mesh data structures for spectral collocation methods.

Provides pure data structures (dataclasses) for mesh representation,
separating data from algorithms for C++ portability.
"""

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Generic, List, Sequence, Tuple

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.cartesian import cartesian_product_samples


class BoundarySide(IntEnum):
    """The boundary ids of a tensor-product mesh, by the side of the
    reference hypercube :math:`[-1, 1]^d` they lie on.

    A mesh of dimension :math:`d` has the first :math:`2d`: an interval
    ``LEFT`` and ``RIGHT``; a rectangle also ``BOTTOM`` and ``TOP``; a box
    also ``FRONT`` and ``BACK``. The names are of the reference sides,
    so under a curved transform (polar, say) ``LEFT`` is the side mapped
    from reference :math:`x = -1`, wherever that lies physically.
    """

    LEFT = 0
    """Reference :math:`x = -1`."""
    RIGHT = 1
    """Reference :math:`x = 1`."""
    BOTTOM = 2
    """Reference :math:`y = -1`."""
    TOP = 3
    """Reference :math:`y = 1`."""
    FRONT = 4
    """Reference :math:`z = -1`."""
    BACK = 5
    """Reference :math:`z = 1`."""


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


def compute_boundary_indices(
    npts_per_dim: Sequence[int], bkd: Backend[Array]
) -> List[Array]:
    """Boundary node indices of a tensor-product mesh in any dimension.

    The nodes are in tensor order, the first direction fastest. Boundary
    ``2 d`` is the first node along direction ``d``, boundary ``2 d + 1``
    the last; each lists its nodes in increasing index, which is the
    tensor order of the remaining directions.

    Parameters
    ----------
    npts_per_dim : Sequence[int]
        Nodes per direction.
    bkd : Backend
        Computational backend.

    Returns
    -------
    List[Array]
        ``2 len(npts_per_dim)`` integer arrays.
    """
    npts = math.prod(npts_per_dim)
    flat = np.arange(npts)
    indices: List[Array] = []
    stride = 1
    for count in npts_per_dim:
        position = (flat // stride) % count
        for end in (0, count - 1):
            indices.append(
                bkd.asarray(flat[position == end].tolist(), dtype=bkd.int64_dtype())
            )
        stride *= count
    return indices


def compute_boundary_indices_1d(npts: int, bkd: Backend[Array]) -> List[Array]:
    """Boundary indices of a 1D mesh: ``[left, right]``, the first and
    last node."""
    return compute_boundary_indices((npts,), bkd)


def compute_boundary_indices_2d(
    npts_x: int, npts_y: int, bkd: Backend[Array]
) -> List[Array]:
    """Boundary indices of a 2D mesh, x fastest: ``[left, right, bottom,
    top]``."""
    return compute_boundary_indices((npts_x, npts_y), bkd)


def compute_boundary_indices_3d(
    npts_x: int, npts_y: int, npts_z: int, bkd: Backend[Array]
) -> List[Array]:
    """Boundary indices of a 3D mesh, x fastest, then y, then z: x = -1,
    x = +1, y = -1, y = +1, z = -1, z = +1 (:class:`BoundarySide` ``LEFT``
    to ``BACK``)."""
    return compute_boundary_indices((npts_x, npts_y, npts_z), bkd)
