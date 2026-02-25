"""Basis protocols for Galerkin finite element methods.

Defines interfaces for finite element bases that wrap scikit-fem (skfem).
"""

from typing import Protocol, Generic, runtime_checkable, Any, Callable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.galerkin.protocols.mesh import GalerkinMeshProtocol


@runtime_checkable
class GalerkinBasisProtocol(Protocol, Generic[Array]):
    """Protocol for finite element basis.

    This protocol wraps skfem Basis objects, providing a backend-agnostic
    interface while maintaining access to the underlying skfem basis.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def mesh(self) -> GalerkinMeshProtocol[Array]:
        """Return the underlying mesh."""
        ...

    def ndofs(self) -> int:
        """Return total number of degrees of freedom."""
        ...

    def degree(self) -> int:
        """Return polynomial degree of the basis."""
        ...

    def skfem_basis(self) -> Any:
        """Return the underlying skfem Basis object.

        Returns
        -------
        skfem.Basis
            The scikit-fem basis object.
        """
        ...

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
        ...

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
        ...

    def dof_coordinates(self) -> Array:
        """Return coordinates of DOF locations.

        Returns
        -------
        Array
            DOF coordinates. Shape: (ndim, ndofs)
        """
        ...


@runtime_checkable
class VectorBasisProtocol(Protocol, Generic[Array]):
    """Protocol for vector-valued finite element basis.

    For multi-component problems (e.g., Stokes with velocity components).
    """

    def bkd(self) -> Backend[Array]: ...
    def mesh(self) -> GalerkinMeshProtocol[Array]: ...
    def ndofs(self) -> int: ...
    def skfem_basis(self) -> Any: ...

    def ncomponents(self) -> int:
        """Return number of vector components."""
        ...

    def component_basis(self, component: int) -> GalerkinBasisProtocol[Array]:
        """Return basis for a single component.

        Parameters
        ----------
        component : int
            Component index.

        Returns
        -------
        GalerkinBasisProtocol
            Scalar basis for the component.
        """
        ...
