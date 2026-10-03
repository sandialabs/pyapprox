"""Physics protocols for Galerkin finite element methods.

Defines the core protocol for PDE physics:
- GalerkinPhysicsProtocol - basic residual, Jacobian, mass matrix

Parameter sensitivity (param_jacobian, HVP) is handled by the separate
ParameterizationProtocol layer, not embedded in physics.

The key difference from collocation is that Galerkin uses weak formulation
with mass matrices: M*du/dt = F(u,t) instead of du/dt = f(u,t).
"""

from typing import (
    Generic,
    List,
    Protocol,
    Tuple,
    runtime_checkable,
)

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.boundary import ConstraintSetProtocol, WeakFormBCProtocol
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinTransientSystemProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class GalerkinPhysicsProtocol(Protocol, Generic[Array]):
    """Protocol for Galerkin PDE physics (Level 1).

    Defines weak form discretization of a PDE system.
    This is the minimum interface for forward solve.

    The Galerkin formulation produces:
        M * du/dt = F(u, t)
    where M is the mass matrix from the weak form.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nstates(self) -> int:
        """Return total number of DOFs."""
        ...

    def mass_matrix(self) -> Array:
        """Return the mass matrix from weak form.

        For Galerkin FEM, this is typically:
            M_ij = integral(phi_i * phi_j)

        Returns
        -------
        Array
            Mass matrix. Shape: (nstates, nstates)
        """
        ...

    def mass_solve(self, rhs: Array) -> Array:
        """Solve M * x = rhs for x.

        This method can be overridden to exploit structure in the mass matrix.
        For example, with a lumped (diagonal) mass matrix, this becomes a
        simple element-wise division.

        Parameters
        ----------
        rhs : Array
            Right-hand side vector. Shape: (nstates,) or (nstates, ncols)

        Returns
        -------
        Array
            Solution x = M^{-1} * rhs. Same shape as rhs.
        """
        ...

    def residual(self, state: Array, time: float) -> Array:
        """Compute residual F(u, t) with Dirichlet BCs applied.

        For transient problems: M * du/dt = residual(u, t)
        For steady problems: solve residual(u) = 0

        Parameters
        ----------
        state : Array
            Solution state (DOF coefficients). Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual. Shape: (nstates,)
        """
        ...

    def spatial_residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual without Dirichlet enforcement.

        Returns F = b - K*u (or equivalent) with Robin/Neumann BC
        contributions but no Dirichlet row replacement.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Spatial residual. Shape: (nstates,)
        """
        ...

    def spatial_jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian dF/du without Dirichlet enforcement.

        Returns the Jacobian with Robin/Neumann BC contributions but
        no Dirichlet row replacement.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian dF/du. Shape: (nstates, nstates)
        """
        ...

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian dF/du with Dirichlet BCs applied.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nstates, nstates)
        """
        ...

    def system(self) -> GalerkinTransientSystemProtocol[Array]:
        """Return the composed system (spatial operator, constraints,
        mass) that adapters and models consume."""
        ...

    def constraint_set(self) -> ConstraintSetProtocol[Array]:
        """Return the essential-constraint set for this physics.

        Aggregates all essential (Dirichlet) BCs into a single cached
        constraint object used for row replacement.
        """
        ...

    def weak_form_bcs(self) -> List[WeakFormBCProtocol[Array]]:
        """Return the natural (Neumann/Robin) BCs, in list order."""
        ...

    def dirichlet_dof_info(self, time: float) -> Tuple[Array, Array]:
        """Return Dirichlet DOF indices and their exact values.

        Parameters
        ----------
        time : float
            Current time.

        Returns
        -------
        Tuple[Array, Array]
            (dof_indices, dof_values) — shapes (ndirichlet,) each.
        """
        ...


@runtime_checkable
class GalerkinInteriorOperatorProtocol(Protocol, Generic[Array]):
    """The interior part of the spatial operator, ``F_Omega``.

    A physics' own mathematics: the weak form over the domain, with no
    boundary condition in it. The spatial operator is composed as
    ``F = F_Omega + F_Gamma``, where the natural-BC part ``F_Gamma`` is
    added once, by ``pyapprox.pde.boundary.NaturalBCOperator``, and never
    by the physics itself.

    Anything implementing this protocol (a physics, or an operator such
    as a prior's precision) gets the composed ``F`` from
    ``ComposedSpatialOperator``, with no base class required.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nstates(self) -> int:
        """Return the number of states."""
        ...

    def interior_residual(self, state: Array, time: float) -> Array:
        """Compute ``F_Omega(u, t)``. Shape: (nstates,)."""
        ...

    def interior_jacobian(self, state: Array, time: float) -> Array:
        """Compute ``dF_Omega/du``. Shape: (nstates, nstates)."""
        ...

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        """Return the optional second state derivatives of ``F_Omega``.

        ``StateDerivatives.linear`` for an interior linear in u;
        ``StateDerivatives.none()`` when the curvature is not supplied.
        """
        ...
