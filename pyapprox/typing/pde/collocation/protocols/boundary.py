"""Boundary condition protocols for spectral collocation methods.

Defines interfaces for boundary conditions that modify PDE residuals
and Jacobians.
"""

from typing import Protocol, Generic, runtime_checkable, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class BoundaryConditionProtocol(Protocol, Generic[Array]):
    """Protocol for boundary conditions.

    Boundary conditions modify the residual and Jacobian at boundary
    points to enforce constraints.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def boundary_indices(self) -> Array:
        """Return indices of mesh points on this boundary.

        Returns
        -------
        Array
            Integer indices. Shape: (nboundary_pts,)
        """
        ...

    def apply_to_residual(
        self, residual: Array, state: Array, time: float
    ) -> Array:
        """Apply boundary condition to residual.

        Modifies rows of residual corresponding to boundary points.

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

        Modifies rows of Jacobian corresponding to boundary points.

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
class BoundaryConditionWithParamJacobianProtocol(Protocol, Generic[Array]):
    """Protocol for boundary conditions with parameter sensitivity.

    Extends BoundaryConditionProtocol with parameter Jacobian for
    adjoint sensitivity analysis.
    """

    def bkd(self) -> Backend[Array]: ...
    def boundary_indices(self) -> Array: ...
    def apply_to_residual(
        self, residual: Array, state: Array, time: float
    ) -> Array: ...
    def apply_to_jacobian(
        self, jacobian: Array, state: Array, time: float
    ) -> Array: ...

    def apply_to_param_jacobian(
        self, param_jacobian: Array, state: Array, time: float
    ) -> Array:
        """Apply boundary condition to parameter Jacobian.

        Parameters
        ----------
        param_jacobian : Array
            Parameter Jacobian. Shape: (nstates, nparams)
        state : Array
            Current solution. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified parameter Jacobian. Shape: (nstates, nparams)
        """
        ...


@runtime_checkable
class DirichletBCProtocol(Protocol, Generic[Array]):
    """Protocol for Dirichlet boundary conditions.

    Enforces u = g(x, t) on the boundary.
    """

    def bkd(self) -> Backend[Array]: ...
    def boundary_indices(self) -> Array: ...
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
            Boundary values. Shape: (nboundary_pts,)
        """
        ...


@runtime_checkable
class RobinBCProtocol(Protocol, Generic[Array]):
    """Protocol for Robin boundary conditions.

    Enforces alpha * u + beta * (flux . n) = g on the boundary.
    """

    def bkd(self) -> Backend[Array]: ...
    def boundary_indices(self) -> Array: ...
    def apply_to_residual(
        self, residual: Array, state: Array, time: float
    ) -> Array: ...
    def apply_to_jacobian(
        self, jacobian: Array, state: Array, time: float
    ) -> Array: ...

    def alpha(self) -> Array:
        """Return coefficient for u term.

        Returns
        -------
        Array
            Coefficient. Shape: (nboundary_pts,) or scalar.
        """
        ...

    def beta(self) -> Array:
        """Return coefficient for flux term.

        Returns
        -------
        Array
            Coefficient. Shape: (nboundary_pts,) or scalar.
        """
        ...

    def boundary_values(self, time: float) -> Array:
        """Return Robin boundary values g.

        Returns
        -------
        Array
            Boundary values. Shape: (nboundary_pts,)
        """
        ...
