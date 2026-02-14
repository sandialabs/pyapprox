"""Boundary condition protocols for spectral collocation methods.

Defines interfaces for boundary conditions that modify PDE residuals
and Jacobians.
"""

from dataclasses import dataclass
from typing import Protocol, Generic, List, runtime_checkable, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend


@dataclass(frozen=True)
class BCDofClassification:
    """Classification of boundary DOFs for adjoint operations.

    Produced by the physics/solver layer, consumed by the time
    integration layer. The time integration layer uses these index
    sets without knowing solver internals.

    Attributes
    ----------
    essential : list[int]
        DOFs where the solution is prescribed (e.g., Dirichlet).
        The adjoint variable is zero at these DOFs (lambda[b] = 0)
        when differentiating w.r.t. PDE parameters. When differentiating
        w.r.t. BC parameters, lambda[b] = -dq/dy[b] acts as the
        Lagrange multiplier for the constraint.
        Always a subset of row_replaced.
    row_replaced : list[int]
        DOFs where the solver replaced the PDE residual row with a BC
        equation. At these DOFs: B_n[b,:] = 0 (no dependence on
        previous time step) and dR_n/dp[b,:] = 0 (assuming BCs are
        independent of PDE parameters).

        For collocation: all BC DOFs (both Dirichlet and Robin).
        For Galerkin FEM: only strongly-enforced Dirichlet DOFs.
        Natural BCs in Galerkin are assembled into the weak form
        without row replacement.
    """
    essential: list
    row_replaced: list

    def __post_init__(self):
        if not set(self.essential) <= set(self.row_replaced):
            raise ValueError(
                "essential must be a subset of row_replaced: "
                "every prescribed-value BC must replace its residual row"
            )


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

    def is_essential(self) -> bool:
        """Return True if this BC directly constrains DOF values.

        Essential BCs (Dirichlet-like) prescribe the solution value at
        boundary DOFs. In the adjoint equation, the adjoint variable at
        essential BC DOFs is zero: lambda[bc] = 0.

        Natural BCs (Robin, Neumann) couple boundary DOFs through
        derivative operators. The adjoint variable at natural BC DOFs
        evolves freely according to the adjoint equation.

        Returns
        -------
        bool
            True for essential BCs (Dirichlet, periodic value-matching),
            False for natural BCs (Robin, Neumann).
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
        self, param_jacobian: Array, state: Array, time: float,
        physical_sensitivities=None,
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


@runtime_checkable
class NormalOperatorProtocol(Protocol, Generic[Array]):
    """Protocol for computing a normal-direction term at boundary points.

    Encapsulates the computation of the normal term in Robin/Neumann BCs.
    Two conventions:
    - Gradient: grad(u) . n  (uses derivative matrices + normals)
    - Flux: flux(u) . n  (uses physics flux provider + normals)
    """

    def __call__(self, state: Array) -> Array:
        """Compute the normal operation on the given state.

        Parameters
        ----------
        state : Array
            Full solution vector. Shape: (npts,)

        Returns
        -------
        Array
            Result at boundary points. Shape: (nboundary_pts,)
        """
        ...

    def jacobian(self, state: Array) -> Array:
        """Return the Jacobian of the normal operation w.r.t. state.

        Parameters
        ----------
        state : Array
            Full solution vector. Shape: (npts,)

        Returns
        -------
        Array
            Jacobian rows at boundary points. Shape: (nboundary_pts, npts)
        """
        ...


@runtime_checkable
class FluxProviderProtocol(Protocol, Generic[Array]):
    """Protocol for physics that can provide a flux vector field.

    Flux is the total conservative flux (diffusive + advective).
    For pure diffusion: flux_d = -D * du/dx_d
    For ADR: flux_d = -D * du/dx_d + v_d * u

    Note: No time parameter. Currently flux depends only on state,
    matching the existing compute_interface_flux convention. If
    time-dependent coefficients (D(t), v(t)) are needed, add a
    time parameter to both methods.
    """

    def compute_flux(self, state: Array) -> List[Array]:
        """Compute flux vector field at all mesh points.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (npts,)

        Returns
        -------
        List[Array]
            Flux components, one per spatial dimension.
            Each shape: (npts,)
        """
        ...

    def compute_flux_jacobian(self, state: Array) -> List[Array]:
        """Compute Jacobian of each flux component w.r.t. state.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (npts,)

        Returns
        -------
        List[Array]
            Jacobian of each flux component.
            Each shape: (npts, npts)
        """
        ...
