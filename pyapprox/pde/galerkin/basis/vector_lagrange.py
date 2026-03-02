"""Vector Lagrange finite element basis implementations.

Wraps scikit-fem ElementVector for multi-component problems like elasticity.
"""

from typing import Any, Callable, Generic

import numpy as np

from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
from pyapprox.pde.galerkin.protocols.mesh import GalerkinMeshProtocol
from pyapprox.util.backends.protocols import Array, Backend

# Import skfem for basis construction
try:
    from skfem import Basis, ElementVector
    from skfem.element import (
        ElementHex1,
        ElementHex2,
        ElementLineP1,
        ElementLineP2,
        ElementQuad1,
        ElementQuad2,
        ElementTetP1,
        ElementTetP2,
        ElementTriP1,
        ElementTriP2,
    )
except ImportError:
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency(
        "skfem", feature_name="Galerkin module", extra_name="fem"
    )


class VectorLagrangeBasis(Generic[Array]):
    """Vector Lagrange finite element basis.

    For multi-component vector fields (e.g., displacement in elasticity).
    Wraps skfem Basis with ElementVector for vector-valued problems.

    Parameters
    ----------
    mesh : GalerkinMeshProtocol[Array]
        The mesh to build the basis on.
    degree : int, default=1
        Polynomial degree (1 or 2).

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.pde.galerkin.mesh import StructuredMesh2D
    >>> bkd = NumpyBkd()
    >>> mesh = StructuredMesh2D(
    ...     nx=5, ny=5, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd
    ... )
    >>> basis = VectorLagrangeBasis(mesh, degree=1)
    >>> basis.ndofs()  # 2 components * 36 nodes
    72
    """

    def __init__(
        self,
        mesh: GalerkinMeshProtocol[Array],
        degree: int = 1,
    ):
        self._mesh = mesh
        self._bkd = mesh.bkd()
        self._degree = degree
        self._ndim = mesh.ndim()

        # Get skfem mesh
        skfem_mesh = mesh.skfem_mesh()
        mesh_type = type(skfem_mesh).__name__

        # Select appropriate scalar element
        scalar_element = self._select_element(mesh_type, degree)

        # Create vector element (ndim components)
        vector_element = ElementVector(scalar_element)

        # Create skfem basis
        self._skfem_basis = Basis(skfem_mesh, vector_element)

        # Also store scalar basis for component access
        self._scalar_basis = LagrangeBasis(mesh, degree)

        # Cache DOF locations
        self._dof_locs = None

    def _select_element(self, mesh_type: str, degree: int) -> Any:
        """Select skfem scalar element based on mesh type and polynomial degree."""
        # Handle both old (MeshLine) and new (MeshLine1) naming conventions
        base_type = mesh_type.rstrip("0123456789")

        element_map = {
            ("MeshLine", 1): ElementLineP1(),
            ("MeshLine", 2): ElementLineP2(),
            ("MeshQuad", 1): ElementQuad1(),
            ("MeshQuad", 2): ElementQuad2(),
            ("MeshTri", 1): ElementTriP1(),
            ("MeshTri", 2): ElementTriP2(),
            ("MeshHex", 1): ElementHex1(),
            ("MeshHex", 2): ElementHex2(),
            ("MeshTet", 1): ElementTetP1(),
            ("MeshTet", 2): ElementTetP2(),
        }

        key = (base_type, degree)
        if key not in element_map:
            raise ValueError(
                f"Unsupported mesh/degree combination: {mesh_type}, degree={degree}. "
                f"Supported mesh types: MeshLine, MeshQuad, MeshTri, MeshHex, MeshTet "
                f"with degree 1 or 2."
            )

        return element_map[key]

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def mesh(self) -> GalerkinMeshProtocol[Array]:
        """Return the underlying mesh."""
        return self._mesh

    def ndofs(self) -> int:
        """Return total number of degrees of freedom (all components)."""
        return self._skfem_basis.N

    def ncomponents(self) -> int:
        """Return number of vector components (= spatial dimension)."""
        return self._ndim

    def ndofs_per_component(self) -> int:
        """Return number of DOFs per vector component."""
        return self._scalar_basis.ndofs()

    def degree(self) -> int:
        """Return polynomial degree of the basis."""
        return self._degree

    def skfem_basis(self) -> Basis:
        """Return the underlying skfem Basis object (vector-valued)."""
        return self._skfem_basis

    def scalar_basis(self) -> LagrangeBasis:
        """Return the scalar Lagrange basis for components."""
        return self._scalar_basis

    def interpolate(self, func: Callable) -> Array:
        """Interpolate a vector function onto the finite element space.

        Parameters
        ----------
        func : Callable
            Function to interpolate. Should accept coordinates as
            Array of shape (ndim, npts) and return Array of shape (ndim, npts).

        Returns
        -------
        Array
            Interpolated DOF values. Shape: (ndofs,)
            DOFs are interleaved by node: [u_x_0, u_y_0, u_x_1, u_y_1, ...]
            (following skfem's ElementVector DOF ordering).
        """
        # Get DOF locations (from scalar basis, which has nnodes locations)
        dof_locs = self._bkd.to_numpy(self._scalar_basis.dof_coordinates())

        # Evaluate function at DOF locations
        values_np = func(dof_locs)  # Shape: (ndim, nnodes)

        # Interleave: [u_x_0, u_y_0, u_z_0, u_x_1, u_y_1, u_z_1, ...]
        # This matches skfem's ElementVector DOF ordering
        dof_values = values_np.T.flatten()  # Transpose then flatten gives interleaved

        return self._bkd.asarray(dof_values.astype(np.float64))

    def dof_coordinates(self) -> Array:
        """Return coordinates of DOF locations.

        For vector elements, DOF coordinates are repeated for each component.

        Returns
        -------
        Array
            DOF coordinates. Shape: (ndim, ndofs)
        """
        if self._dof_locs is None:
            dof_locs_np = self._skfem_basis.doflocs
            self._dof_locs = self._bkd.asarray(dof_locs_np.astype(np.float64))

        return self._dof_locs

    def get_dofs(self, boundary_name: str) -> Array:
        """Return all DOF indices on a named boundary.

        For vector elements, returns DOFs for ALL components on the boundary.

        Parameters
        ----------
        boundary_name : str
            Name of the boundary (e.g., "left", "right", "bottom", "top").

        Returns
        -------
        Array
            DOF indices on this boundary (all components).
        """
        ndim = self._ndim
        if ndim == 1:
            # In 1D there is only one component — skip logic would skip it,
            # so get all boundary DOFs directly.
            combined = np.asarray(self._skfem_basis.get_dofs(boundary_name)).flatten()
        else:
            all_dofs = []
            dofnames = self._skfem_basis.get_dofs().obj.element.dofnames
            for idx in range(ndim):
                skip = dofnames[ndim - idx - 1]
                component_dofs = self._skfem_basis.get_dofs(boundary_name, skip=skip)
                all_dofs.append(np.asarray(component_dofs).flatten())
            combined = np.concatenate(all_dofs)
        combined.sort()
        return self._bkd.asarray(combined)

    def get_component_dofs(self, component: int) -> Array:
        """Return DOF indices for a specific component.

        Parameters
        ----------
        component : int
            Component index (0 for x, 1 for y, 2 for z).

        Returns
        -------
        Array
            DOF indices for this component.
        """
        # DOFs are interleaved: [u_x_0, u_y_0, u_z_0, u_x_1, u_y_1, u_z_1, ...]
        # So component i is at indices i, i+ndim, i+2*ndim, ...
        n_per_comp = self.ndofs_per_component()
        indices = np.arange(n_per_comp, dtype=np.int64) * self._ndim + component
        return self._bkd.asarray(indices)

    def __repr__(self) -> str:
        return (
            f"VectorLagrangeBasis(ndofs={self.ndofs()}, "
            f"ncomponents={self.ncomponents()}, degree={self._degree})"
        )
