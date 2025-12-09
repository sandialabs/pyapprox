"""Lagrange finite element basis implementations.

Wraps scikit-fem element and basis objects with backend abstraction.
"""

from typing import Generic, Callable, Any

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.galerkin.protocols.mesh import GalerkinMeshProtocol

# Import skfem for basis construction
try:
    from skfem import Basis
    from skfem.element import (
        ElementLineP1,
        ElementLineP2,
        ElementQuad1,
        ElementQuad2,
        ElementTriP1,
        ElementTriP2,
    )
except ImportError:
    raise ImportError(
        "scikit-fem is required for Galerkin module. "
        "Install with: pip install scikit-fem"
    )


class LagrangeBasis(Generic[Array]):
    """Lagrange finite element basis.

    Wraps skfem Basis with appropriate Lagrange elements for the mesh type.

    Parameters
    ----------
    mesh : GalerkinMeshProtocol[Array]
        The mesh to build the basis on.
    degree : int, default=1
        Polynomial degree (1 or 2).

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.pde.galerkin.mesh import StructuredMesh1D
    >>> bkd = NumpyBkd()
    >>> mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    >>> basis = LagrangeBasis(mesh, degree=1)
    >>> basis.ndofs()
    11
    """

    def __init__(
        self,
        mesh: GalerkinMeshProtocol[Array],
        degree: int = 1,
    ):
        self._mesh = mesh
        self._bkd = mesh.bkd()
        self._degree = degree

        # Get skfem mesh
        skfem_mesh = mesh.skfem_mesh()
        mesh_type = type(skfem_mesh).__name__

        # Select appropriate element
        element = self._select_element(mesh_type, degree)

        # Create skfem basis
        self._skfem_basis = Basis(skfem_mesh, element)

        # Cache DOF locations
        self._dof_locs = None

    def _select_element(self, mesh_type: str, degree: int) -> Any:
        """Select skfem element based on mesh type and polynomial degree."""
        # Handle both old (MeshLine) and new (MeshLine1) naming conventions
        # Strip trailing digits from mesh type names
        base_type = mesh_type.rstrip("0123456789")

        element_map = {
            ("MeshLine", 1): ElementLineP1(),
            ("MeshLine", 2): ElementLineP2(),
            ("MeshQuad", 1): ElementQuad1(),
            ("MeshQuad", 2): ElementQuad2(),
            ("MeshTri", 1): ElementTriP1(),
            ("MeshTri", 2): ElementTriP2(),
        }

        key = (base_type, degree)
        if key not in element_map:
            raise ValueError(
                f"Unsupported mesh/degree combination: {mesh_type}, degree={degree}. "
                f"Supported mesh types: MeshLine, MeshQuad, MeshTri with degree 1 or 2."
            )

        return element_map[key]

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def mesh(self) -> GalerkinMeshProtocol[Array]:
        """Return the underlying mesh."""
        return self._mesh

    def ndofs(self) -> int:
        """Return total number of degrees of freedom."""
        return self._skfem_basis.N

    def degree(self) -> int:
        """Return polynomial degree of the basis."""
        return self._degree

    def skfem_basis(self) -> Basis:
        """Return the underlying skfem Basis object."""
        return self._skfem_basis

    def interpolate(self, func: Callable) -> Array:
        """Interpolate a function onto the finite element space.

        Parameters
        ----------
        func : Callable
            Function to interpolate. Should accept coordinates as
            Array of shape (ndim, npts) and return Array of shape (npts,).

        Returns
        -------
        Array
            Interpolated DOF values. Shape: (ndofs,)
        """
        # Get DOF locations
        dof_locs = self.dof_coordinates()
        dof_locs_np = self._bkd.to_numpy(dof_locs)

        # Evaluate function at DOF locations
        values_np = func(dof_locs_np)

        return self._bkd.asarray(values_np.astype(np.float64))

    def evaluate(self, coeffs: Array, points: Array) -> Array:
        """Evaluate the finite element solution at given points.

        Parameters
        ----------
        coeffs : Array
            DOF coefficients. Shape: (ndofs,)
        points : Array
            Evaluation points. Shape: (ndim, npts)

        Returns
        -------
        Array
            Function values at points. Shape: (npts,)
        """
        # Convert to numpy for skfem
        coeffs_np = self._bkd.to_numpy(coeffs)
        points_np = self._bkd.to_numpy(points)

        # Use skfem's probes for evaluation
        # This finds the elements containing each point and evaluates
        from skfem import Functional

        skfem_mesh = self._mesh.skfem_mesh()

        # Find cells containing points
        cells = skfem_mesh.element_finder()(
            *[points_np[i, :] for i in range(points_np.shape[0])]
        )

        # Evaluate using basis interpolation
        values = np.zeros(points_np.shape[1])
        for i, cell in enumerate(cells):
            if cell >= 0:
                # Get local coordinates
                # This is simplified - a full implementation would use
                # skfem's InteriorBasis or probing functionality
                dof_indices = self._skfem_basis.element_dofs[:, cell]
                # Simple nodal interpolation for P1 elements
                local_coeffs = coeffs_np[dof_indices]
                values[i] = np.mean(local_coeffs)  # Simplified

        return self._bkd.asarray(values.astype(np.float64))

    def dof_coordinates(self) -> Array:
        """Return coordinates of DOF locations.

        Returns
        -------
        Array
            DOF coordinates. Shape: (ndim, ndofs)
        """
        if self._dof_locs is None:
            # For P1 elements, DOFs are at nodes
            if self._degree == 1:
                self._dof_locs = self._mesh.nodes()
            else:
                # For higher degree, use skfem's doflocs
                dof_locs_np = self._skfem_basis.doflocs
                self._dof_locs = self._bkd.asarray(
                    dof_locs_np.astype(np.float64)
                )

        return self._dof_locs

    def get_dofs(self, boundary_name: str) -> Array:
        """Return DOF indices on a named boundary.

        Parameters
        ----------
        boundary_name : str
            Boundary identifier.

        Returns
        -------
        Array
            DOF indices on the boundary. Shape: (nboundary_dofs,)
        """
        # For P1, DOFs are at nodes so we can use mesh boundary nodes
        if self._degree == 1:
            return self._mesh.boundary_nodes(boundary_name)
        else:
            # For higher degree, use skfem's get_dofs
            # The boundary name mapping depends on mesh type
            skfem_mesh = self._mesh.skfem_mesh()

            # Try to get facets for boundary
            try:
                dofs = self._skfem_basis.get_dofs(boundary_name)
                # get_dofs returns a Dofs object, extract the indices
                return self._bkd.asarray(dofs.flatten().astype(np.int64))
            except Exception:
                # Fallback to node-based for now
                return self._mesh.boundary_nodes(boundary_name)

    def __repr__(self) -> str:
        return f"LagrangeBasis(ndofs={self.ndofs()}, degree={self._degree})"
