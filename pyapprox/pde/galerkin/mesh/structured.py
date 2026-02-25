"""Structured mesh implementations for Galerkin finite element methods.

Wraps scikit-fem mesh objects with backend abstraction.
"""

from typing import Generic, List, Tuple

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend

# Import skfem for mesh construction
try:
    from skfem import (
        MeshHex,
        MeshLine,
        MeshQuad,
        MeshTet,  # noqa: F401
        MeshTri,  # noqa: F401
    )
except ImportError:
    raise ImportError(
        "scikit-fem is required for Galerkin module. "
        "Install with: pip install scikit-fem"
    )


class StructuredMesh1D(Generic[Array]):
    """Structured 1D mesh (line elements).

    Wraps skfem.MeshLine with backend abstraction.

    Parameters
    ----------
    nx : int
        Number of elements in x direction.
    bounds : Tuple[float, float]
        Domain bounds (xmin, xmax).
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    >>> mesh.nnodes()
    11
    """

    def __init__(
        self,
        nx: int,
        bounds: Tuple[float, float],
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._nx = nx
        self._bounds_tuple = bounds

        # Create skfem mesh with named boundaries
        xmin, xmax = bounds
        nodes = np.linspace(xmin, xmax, nx + 1)
        self._skfem_mesh = MeshLine(nodes).with_boundaries(
            {
                "left": lambda x: np.abs(x[0] - xmin) < 1e-12,
                "right": lambda x: np.abs(x[0] - xmax) < 1e-12,
            }
        )

        # Cache nodes as backend array
        self._nodes = bkd.asarray(self._skfem_mesh.p.astype(np.float64))
        self._bounds = bkd.asarray(np.array([[xmin, xmax]], dtype=np.float64))

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return spatial dimension."""
        return 1

    def nelements(self) -> int:
        """Return total number of mesh elements."""
        return self._nx

    def nnodes(self) -> int:
        """Return total number of mesh nodes."""
        return self._nx + 1

    def nodes(self) -> Array:
        """Return mesh node coordinates.

        Returns
        -------
        Array
            Node coordinates. Shape: (1, nnodes)
        """
        return self._nodes

    def elements(self) -> Array:
        """Return element connectivity.

        Returns
        -------
        Array
            Element-to-node connectivity. Shape: (2, nelements)
        """
        return self._bkd.asarray(self._skfem_mesh.t.astype(np.int64))

    def skfem_mesh(self) -> MeshLine:
        """Return the underlying skfem mesh object."""
        return self._skfem_mesh

    def boundary_nodes(self, boundary_id: str) -> Array:
        """Return node indices on a named boundary.

        Parameters
        ----------
        boundary_id : str
            Boundary identifier: "left" or "right".

        Returns
        -------
        Array
            Node indices on the boundary. Shape: (1,)
        """
        if boundary_id == "left":
            return self._bkd.asarray(np.array([0], dtype=np.int64))
        elif boundary_id == "right":
            return self._bkd.asarray(np.array([self._nx], dtype=np.int64))
        else:
            raise ValueError(
                f"Unknown boundary_id '{boundary_id}'. Valid options: 'left', 'right'"
            )

    def shape(self) -> Tuple[int]:
        """Return grid shape (number of nodes in each dimension)."""
        return (self._nx + 1,)

    def bounds(self) -> Array:
        """Return domain bounds.

        Returns
        -------
        Array
            Bounds. Shape: (1, 2)
        """
        return self._bounds

    def __repr__(self) -> str:
        return f"StructuredMesh1D(nx={self._nx}, bounds={self._bounds_tuple})"


class StructuredMesh2D(Generic[Array]):
    """Structured 2D mesh (quadrilateral or triangular elements).

    Wraps skfem.MeshQuad or MeshTri with backend abstraction.

    Parameters
    ----------
    nx : int
        Number of elements in x direction.
    ny : int
        Number of elements in y direction.
    bounds : List[Tuple[float, float]]
        Domain bounds [[xmin, xmax], [ymin, ymax]].
    bkd : Backend[Array]
        Computational backend.
    element_type : str, default="quad"
        Element type: "quad" for quadrilaterals, "tri" for triangles.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> mesh = StructuredMesh2D(
    ...     nx=10, ny=10,
    ...     bounds=[[0.0, 1.0], [0.0, 1.0]],
    ...     bkd=bkd
    ... )
    >>> mesh.nnodes()
    121
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        bounds: List[Tuple[float, float]],
        bkd: Backend[Array],
        element_type: str = "quad",
    ):
        self._bkd = bkd
        self._nx = nx
        self._ny = ny
        self._bounds_list = bounds
        self._element_type = element_type

        # Create skfem mesh
        xmin, xmax = bounds[0]
        ymin, ymax = bounds[1]

        # Define boundary functions
        boundary_defs = {
            "left": lambda x: np.abs(x[0] - xmin) < 1e-12,
            "right": lambda x: np.abs(x[0] - xmax) < 1e-12,
            "bottom": lambda x: np.abs(x[1] - ymin) < 1e-12,
            "top": lambda x: np.abs(x[1] - ymax) < 1e-12,
        }

        if element_type == "quad":
            self._skfem_mesh = MeshQuad.init_tensor(
                np.linspace(xmin, xmax, nx + 1),
                np.linspace(ymin, ymax, ny + 1),
            ).with_boundaries(boundary_defs)
        elif element_type == "tri":
            quad_mesh = MeshQuad.init_tensor(
                np.linspace(xmin, xmax, nx + 1),
                np.linspace(ymin, ymax, ny + 1),
            )
            self._skfem_mesh = quad_mesh.to_meshtri().with_boundaries(boundary_defs)
        else:
            raise ValueError(
                f"Unknown element_type '{element_type}'. Valid options: 'quad', 'tri'"
            )

        # Cache nodes as backend array
        self._nodes = bkd.asarray(self._skfem_mesh.p.astype(np.float64))
        self._bounds = bkd.asarray(
            np.array([[xmin, xmax], [ymin, ymax]], dtype=np.float64)
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return spatial dimension."""
        return 2

    def nelements(self) -> int:
        """Return total number of mesh elements."""
        return self._skfem_mesh.nelements

    def nnodes(self) -> int:
        """Return total number of mesh nodes."""
        return self._skfem_mesh.nvertices

    def nodes(self) -> Array:
        """Return mesh node coordinates.

        Returns
        -------
        Array
            Node coordinates. Shape: (2, nnodes)
        """
        return self._nodes

    def elements(self) -> Array:
        """Return element connectivity.

        Returns
        -------
        Array
            Element-to-node connectivity.
            Shape: (4, nelements) for quad, (3, nelements) for tri.
        """
        return self._bkd.asarray(self._skfem_mesh.t.astype(np.int64))

    def skfem_mesh(self):
        """Return the underlying skfem mesh object."""
        return self._skfem_mesh

    def boundary_nodes(self, boundary_id: str) -> Array:
        """Return node indices on a named boundary.

        Parameters
        ----------
        boundary_id : str
            Boundary identifier: "left", "right", "bottom", "top".

        Returns
        -------
        Array
            Node indices on the boundary. Shape: (nboundary_nodes,)
        """
        nodes_np = self._bkd.to_numpy(self._nodes)
        xmin, xmax = self._bounds_list[0]
        ymin, ymax = self._bounds_list[1]

        tol = 1e-12

        if boundary_id == "left":
            mask = np.abs(nodes_np[0, :] - xmin) < tol
        elif boundary_id == "right":
            mask = np.abs(nodes_np[0, :] - xmax) < tol
        elif boundary_id == "bottom":
            mask = np.abs(nodes_np[1, :] - ymin) < tol
        elif boundary_id == "top":
            mask = np.abs(nodes_np[1, :] - ymax) < tol
        else:
            raise ValueError(
                f"Unknown boundary_id '{boundary_id}'. "
                "Valid options: 'left', 'right', 'bottom', 'top'"
            )

        return self._bkd.asarray(np.where(mask)[0].astype(np.int64))

    def shape(self) -> Tuple[int, int]:
        """Return grid shape (number of nodes in each dimension)."""
        return (self._nx + 1, self._ny + 1)

    def bounds(self) -> Array:
        """Return domain bounds.

        Returns
        -------
        Array
            Bounds. Shape: (2, 2)
        """
        return self._bounds

    def __repr__(self) -> str:
        return (
            f"StructuredMesh2D(nx={self._nx}, ny={self._ny}, "
            f"bounds={self._bounds_list}, element_type='{self._element_type}')"
        )


class StructuredMesh3D(Generic[Array]):
    """Structured 3D mesh (hexahedral or tetrahedral elements).

    Wraps skfem.MeshHex or MeshTet with backend abstraction.

    Parameters
    ----------
    nx : int
        Number of elements in x direction.
    ny : int
        Number of elements in y direction.
    nz : int
        Number of elements in z direction.
    bounds : List[Tuple[float, float]]
        Domain bounds [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
    bkd : Backend[Array]
        Computational backend.
    element_type : str, default="hex"
        Element type: "hex" for hexahedra, "tet" for tetrahedra.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> mesh = StructuredMesh3D(
    ...     nx=5, ny=5, nz=5,
    ...     bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    ...     bkd=bkd
    ... )
    >>> mesh.nnodes()
    216
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        bounds: List[Tuple[float, float]],
        bkd: Backend[Array],
        element_type: str = "hex",
    ):
        self._bkd = bkd
        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._bounds_list = bounds
        self._element_type = element_type

        # Create skfem mesh
        xmin, xmax = bounds[0]
        ymin, ymax = bounds[1]
        zmin, zmax = bounds[2]

        # Define boundary functions
        boundary_defs = {
            "left": lambda x: np.abs(x[0] - xmin) < 1e-12,
            "right": lambda x: np.abs(x[0] - xmax) < 1e-12,
            "bottom": lambda x: np.abs(x[1] - ymin) < 1e-12,
            "top": lambda x: np.abs(x[1] - ymax) < 1e-12,
            "front": lambda x: np.abs(x[2] - zmin) < 1e-12,
            "back": lambda x: np.abs(x[2] - zmax) < 1e-12,
        }

        if element_type == "hex":
            self._skfem_mesh = MeshHex.init_tensor(
                np.linspace(xmin, xmax, nx + 1),
                np.linspace(ymin, ymax, ny + 1),
                np.linspace(zmin, zmax, nz + 1),
            ).with_boundaries(boundary_defs)
        elif element_type == "tet":
            hex_mesh = MeshHex.init_tensor(
                np.linspace(xmin, xmax, nx + 1),
                np.linspace(ymin, ymax, ny + 1),
                np.linspace(zmin, zmax, nz + 1),
            )
            self._skfem_mesh = hex_mesh.to_meshtet().with_boundaries(boundary_defs)
        else:
            raise ValueError(
                f"Unknown element_type '{element_type}'. Valid options: 'hex', 'tet'"
            )

        # Cache nodes as backend array
        self._nodes = bkd.asarray(self._skfem_mesh.p.astype(np.float64))
        self._bounds = bkd.asarray(
            np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]], dtype=np.float64)
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ndim(self) -> int:
        """Return spatial dimension."""
        return 3

    def nelements(self) -> int:
        """Return total number of mesh elements."""
        return self._skfem_mesh.nelements

    def nnodes(self) -> int:
        """Return total number of mesh nodes."""
        return self._skfem_mesh.nvertices

    def nodes(self) -> Array:
        """Return mesh node coordinates.

        Returns
        -------
        Array
            Node coordinates. Shape: (3, nnodes)
        """
        return self._nodes

    def elements(self) -> Array:
        """Return element connectivity.

        Returns
        -------
        Array
            Element-to-node connectivity.
            Shape: (8, nelements) for hex, (4, nelements) for tet.
        """
        return self._bkd.asarray(self._skfem_mesh.t.astype(np.int64))

    def skfem_mesh(self):
        """Return the underlying skfem mesh object."""
        return self._skfem_mesh

    def boundary_nodes(self, boundary_id: str) -> Array:
        """Return node indices on a named boundary.

        Parameters
        ----------
        boundary_id : str
            Boundary identifier: "left", "right", "bottom", "top",
            "front", "back".

        Returns
        -------
        Array
            Node indices on the boundary. Shape: (nboundary_nodes,)
        """
        nodes_np = self._bkd.to_numpy(self._nodes)
        xmin, xmax = self._bounds_list[0]
        ymin, ymax = self._bounds_list[1]
        zmin, zmax = self._bounds_list[2]

        tol = 1e-12

        if boundary_id == "left":
            mask = np.abs(nodes_np[0, :] - xmin) < tol
        elif boundary_id == "right":
            mask = np.abs(nodes_np[0, :] - xmax) < tol
        elif boundary_id == "bottom":
            mask = np.abs(nodes_np[1, :] - ymin) < tol
        elif boundary_id == "top":
            mask = np.abs(nodes_np[1, :] - ymax) < tol
        elif boundary_id == "front":
            mask = np.abs(nodes_np[2, :] - zmin) < tol
        elif boundary_id == "back":
            mask = np.abs(nodes_np[2, :] - zmax) < tol
        else:
            raise ValueError(
                f"Unknown boundary_id '{boundary_id}'. "
                "Valid options: 'left', 'right', 'bottom', 'top', "
                "'front', 'back'"
            )

        return self._bkd.asarray(np.where(mask)[0].astype(np.int64))

    def shape(self) -> Tuple[int, int, int]:
        """Return grid shape (number of nodes in each dimension)."""
        return (self._nx + 1, self._ny + 1, self._nz + 1)

    def bounds(self) -> Array:
        """Return domain bounds.

        Returns
        -------
        Array
            Bounds. Shape: (3, 2)
        """
        return self._bounds

    def __repr__(self) -> str:
        return (
            f"StructuredMesh3D(nx={self._nx}, ny={self._ny}, nz={self._nz}, "
            f"bounds={self._bounds_list}, element_type='{self._element_type}')"
        )
