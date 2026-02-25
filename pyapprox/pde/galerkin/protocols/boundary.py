"""Boundary condition protocols for Galerkin finite element methods.

Defines interfaces for boundary conditions that modify PDE residuals
and Jacobians in the finite element context.
"""

from typing import Protocol, Generic, runtime_checkable, Tuple

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class BoundaryConditionProtocol(Protocol, Generic[Array]):
    """Protocol for boundary conditions in Galerkin FEM.

    Boundary conditions modify the residual and Jacobian at boundary
    DOFs to enforce constraints.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def boundary_dofs(self) -> Array:
        """Return indices of DOFs on this boundary.

        Returns
        -------
        Array
            Integer DOF indices. Shape: (nboundary_dofs,)
        """
        ...

    def apply_to_residual(
        self, residual: Array, state: Array, time: float
    ) -> Array:
        """Apply boundary condition to residual.

        Modifies rows of residual corresponding to boundary DOFs.

        Parameters
        ----------
        residual : Array
            Residual vector. Shape: (nstates,)
        state : Array
            Current solution. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified residual. Shape: (nstates,)
        """
        ...

    def apply_to_jacobian(
        self, jacobian: Array, state: Array, time: float
    ) -> Array:
        """Apply boundary condition to Jacobian.

        Modifies rows of Jacobian corresponding to boundary DOFs.

        Parameters
        ----------
        jacobian : Array
            Jacobian matrix. Shape: (nstates, nstates)
        state : Array
            Current solution. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified Jacobian. Shape: (nstates, nstates)
        """
        ...


@runtime_checkable
class DirichletBCProtocol(Protocol, Generic[Array]):
    """Protocol for Dirichlet boundary conditions.

    Enforces u = g(x, t) on the boundary DOFs.
    """

    def bkd(self) -> Backend[Array]: ...
    def boundary_dofs(self) -> Array: ...
    def apply_to_residual(
        self, residual: Array, state: Array, time: float
    ) -> Array: ...
    def apply_to_jacobian(
        self, jacobian: Array, state: Array, time: float
    ) -> Array: ...

    def boundary_values(self, time: float) -> Array:
        """Return Dirichlet boundary values.

        Parameters
        ----------
        time : float
            Current time.

        Returns
        -------
        Array
            Boundary values. Shape: (nboundary_dofs,)
        """
        ...


@runtime_checkable
class NeumannBCProtocol(Protocol, Generic[Array]):
    """Protocol for Neumann boundary conditions.

    Enforces flux . n = g(x, t) on the boundary.

    In weak form, this contributes to the load vector via
    boundary integral: integral_{Gamma} g * phi ds
    """

    def bkd(self) -> Backend[Array]: ...
    def boundary_dofs(self) -> Array: ...

    def flux_values(self, time: float) -> Array:
        """Return Neumann flux values.

        Parameters
        ----------
        time : float
            Current time.

        Returns
        -------
        Array
            Flux values. Shape: (nboundary_dofs,)
        """
        ...

    def apply_to_load(self, load: Array, time: float) -> Array:
        """Apply Neumann BC contribution to load vector.

        Parameters
        ----------
        load : Array
            Load vector. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified load vector. Shape: (nstates,)
        """
        ...


@runtime_checkable
class RobinBCProtocol(Protocol, Generic[Array]):
    """Protocol for Robin boundary conditions.

    Enforces alpha * u + beta * (flux . n) = g on the boundary.
    """

    def bkd(self) -> Backend[Array]: ...
    def boundary_dofs(self) -> Array: ...
    def apply_to_residual(
        self, residual: Array, state: Array, time: float
    ) -> Array: ...
    def apply_to_jacobian(
        self, jacobian: Array, state: Array, time: float
    ) -> Array: ...

    def alpha(self) -> float:
        """Return coefficient for u term."""
        ...

    def boundary_values(self, time: float) -> Array:
        """Return Robin boundary values g.

        Returns
        -------
        Array
            Boundary values. Shape: (nboundary_dofs,)
        """
        ...

    def apply_to_stiffness(self, stiffness: Array, time: float) -> Array:
        """Apply Robin BC contribution to stiffness matrix.

        Adds: alpha * integral_{Gamma} u * phi ds

        Parameters
        ----------
        stiffness : Array
            Stiffness matrix. Shape: (nstates, nstates)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified stiffness matrix.
        """
        ...

    def apply_to_load(self, load: Array, time: float) -> Array:
        """Apply Robin BC contribution to load vector.

        Adds: integral_{Gamma} g * phi ds

        Parameters
        ----------
        load : Array
            Load vector. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified load vector.
        """
        ...
