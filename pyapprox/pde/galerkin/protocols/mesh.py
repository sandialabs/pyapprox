"""Mesh protocols for Galerkin finite element methods.

Defines interfaces for finite element meshes that wrap scikit-fem (skfem).
"""

from typing import Protocol, Generic, runtime_checkable, Any

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class GalerkinMeshProtocol(Protocol, Generic[Array]):
    """Protocol for finite element meshes.

    This protocol wraps skfem mesh objects, providing a backend-agnostic
    interface while maintaining access to the underlying skfem mesh.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def ndim(self) -> int:
        """Return spatial dimension (1, 2, or 3)."""
        ...

    def nelements(self) -> int:
        """Return total number of mesh elements."""
        ...

    def nnodes(self) -> int:
        """Return total number of mesh nodes."""
        ...

    def nodes(self) -> Array:
        """Return mesh node coordinates.

        Returns
        -------
        Array
            Node coordinates. Shape: (ndim, nnodes)
        """
        ...

    def elements(self) -> Array:
        """Return element connectivity.

        Returns
        -------
        Array
            Element-to-node connectivity. Shape varies by element type.
        """
        ...

    def skfem_mesh(self) -> Any:
        """Return the underlying skfem mesh object.

        Returns
        -------
        skfem.Mesh
            The scikit-fem mesh object.
        """
        ...

    def boundary_nodes(self, boundary_id: str) -> Array:
        """Return node indices on a named boundary.

        Parameters
        ----------
        boundary_id : str
            Boundary identifier (e.g., "left", "right", "top", "bottom").

        Returns
        -------
        Array
            Node indices on the boundary. Shape: (nboundary_nodes,)
        """
        ...


@runtime_checkable
class StructuredMeshProtocol(Protocol, Generic[Array]):
    """Protocol for structured finite element meshes.

    Extends GalerkinMeshProtocol with structured grid information.
    """

    # --- All GalerkinMeshProtocol methods ---
    def bkd(self) -> Backend[Array]: ...
    def ndim(self) -> int: ...
    def nelements(self) -> int: ...
    def nnodes(self) -> int: ...
    def nodes(self) -> Array: ...
    def elements(self) -> Array: ...
    def skfem_mesh(self) -> Any: ...
    def boundary_nodes(self, boundary_id: str) -> Array: ...

    # --- Structured mesh methods ---
    def shape(self) -> tuple:
        """Return grid shape (number of nodes in each dimension).

        Returns
        -------
        tuple
            Shape tuple, e.g., (nx,) for 1D, (nx, ny) for 2D.
        """
        ...

    def bounds(self) -> Array:
        """Return domain bounds.

        Returns
        -------
        Array
            Bounds. Shape: (ndim, 2) where [:, 0] are lower bounds
            and [:, 1] are upper bounds.
        """
        ...
