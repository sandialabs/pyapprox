"""Physics protocols for Galerkin finite element methods.

Defines the core protocol for PDE physics:
- GalerkinPhysicsProtocol - basic residual, Jacobian, mass matrix

Parameter sensitivity (param_jacobian, HVP) is handled by the separate
ParameterizationProtocol layer, not embedded in physics.

The key difference from collocation is that Galerkin uses weak formulation
with mass matrices: M*du/dt = F(u,t) instead of du/dt = f(u,t).
"""

from typing import (  # noqa: F401
    Any,
    Generic,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
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

    def basis(self) -> Any:
        """Return the finite element basis."""
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

    def apply_boundary_conditions(
        self,
        residual: Optional[Array],
        jacobian: Optional[Array],
        state: Array,
        time: float = 0.0,
    ) -> Tuple[Optional[Array], Optional[Array]]:
        """Apply boundary conditions to residual and Jacobian.

        Applies in correct order: Robin first, then Dirichlet.

        Parameters
        ----------
        residual : Array or None
            Residual vector. Shape: (nstates,). None to skip.
        jacobian : Array or None
            Jacobian matrix. Shape: (nstates, nstates). None to skip.
        state : Array
            Current state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Tuple[Optional[Array], Optional[Array]]
            Modified (residual, jacobian).
        """
        ...
