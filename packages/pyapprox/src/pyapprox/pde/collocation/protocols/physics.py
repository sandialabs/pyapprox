"""Physics protocols for spectral collocation methods.

Defines the protocols for PDE physics:
- CollocationInteriorOperatorProtocol - the interior operator, with no
  boundary condition in it, and what it is declared to depend on
- PhysicsProtocol - basic residual and Jacobian
- PhysicsWithStateStateHVPProtocol - adds the state-state HVP contraction

Parameter sensitivity (param_jacobian, parameter HVPs) is owned by the
separate ParameterizationProtocol layer via its ParamDerivatives bundle,
not embedded in physics.
"""

from typing import Generic, List, Protocol, Tuple, runtime_checkable

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.boundary import BCDofClassification
from pyapprox.pde.collocation.protocols.basis import BasisProtocol
from pyapprox.pde.collocation.protocols.boundary import (
    BoundaryConditionProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class CollocationInteriorOperatorProtocol(Protocol, Generic[Array]):
    r"""The interior operator :math:`f(u, t)` of a collocation physics.

    The PDE's spatial operator collocated at every node, boundary nodes
    included, with no boundary condition in it: the boundary rows are
    treated by the system that holds the physics, never by the physics
    itself. A transient problem reads :math:`M \dot u = f(u, t)`.

    The members are those of the Galerkin interior operator, so one
    consumer can read either. What the operator depends on is declared,
    not inferred: the declarations decide whether its Jacobian is
    reassembled at each state, at each time or never.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nstates(self) -> int:
        """Return the number of states."""
        ...

    def interior_residual(self, state: Array, time: float) -> Array:
        """Compute :math:`f(u, t)`. Shape: (nstates,)."""
        ...

    def interior_jacobian(self, state: Array, time: float) -> Array:
        r"""Compute :math:`\partial f/\partial u`. Shape: (nstates, nstates)."""
        ...

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        """Return the optional second state derivatives of :math:`f`.

        ``StateDerivatives.linear`` for an operator affine in the state;
        ``StateDerivatives.none()`` when the curvature is not supplied.
        """
        ...

    def interior_is_time_invariant(self) -> bool:
        """Whether :math:`f(u, t)` is DECLARED independent of :math:`t`:
        every coefficient and forcing it holds is declared
        time-independent."""
        ...

    def interior_jacobian_is_time_invariant(self) -> bool:
        r"""Whether :math:`\partial f/\partial u` at a fixed state is
        DECLARED independent of :math:`t`. Weaker than
        ``interior_is_time_invariant``: a forcing does not enter the
        Jacobian."""
        ...


@runtime_checkable
class PhysicsProtocol(Protocol, Generic[Array]):
    """Protocol for PDE physics (Level 1).

    Defines spatial discretization of a PDE system with state Jacobian.
    This is the minimum interface for forward solve.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def basis(self) -> BasisProtocol[Array]:
        """Return the collocation basis."""
        ...

    def nstates(self) -> int:
        """Return total number of states (ncomponents * npts)."""
        ...

    def ncomponents(self) -> int:
        """Return number of solution components.

        E.g., 1 for scalar PDE, 2-3 for coupled systems.
        """
        ...

    def residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual f(u, t).

        For steady problems: solve residual(u) = 0
        For transient problems: du/dt = residual(u, t)

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual. Shape: (nstates,)
        """
        ...

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian df/du.

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

    def boundary_conditions(self) -> List[BoundaryConditionProtocol[Array]]:
        """Return list of boundary conditions."""
        ...

    def bc_dof_classification(self) -> BCDofClassification:
        """Classify boundary DOFs for adjoint operations.

        Returns
        -------
        BCDofClassification
            Classification with essential and row_replaced index lists
            (essential is a subset of row_replaced).
        """
        ...

    def apply_boundary_conditions(
        self, residual: Array, jacobian: Array, state: Array, time: float
    ) -> Tuple[Array, Array]:
        """Apply boundary conditions to residual and Jacobian.

        Modifies rows corresponding to boundary points to enforce BCs.

        Parameters
        ----------
        residual : Array
            Residual vector. Shape: (nstates,)
        jacobian : Array
            Jacobian matrix. Shape: (nstates, nstates)
        state : Array
            Current state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Tuple[Array, Array]
            Modified (residual, jacobian).
        """
        ...

    def mass_matrix(self) -> Array:
        """Return mass matrix for time integration.

        For standard ODEs, this is the identity matrix.
        For DAEs, this may be singular.

        Returns
        -------
        Array
            Mass matrix. Shape: (nstates, nstates)
        """
        ...

    def apply_bc_to_mass(self, mass: Array) -> Array:
        """Apply BC enforcement to the mass matrix for the adjoint at
        the initial time (identity rows/columns at essential DOFs).

        Parameters
        ----------
        mass : Array
            The mass matrix. Shape: (nstates, nstates)

        Returns
        -------
        Array
            Modified mass matrix.
        """
        ...

    def apply_mass_matrix(self, vec: Array) -> Array:
        """Apply mass matrix to a vector.

        Default is identity (returns vec unchanged). Overridden for
        non-identity mass matrices (e.g., split physics).

        Parameters
        ----------
        vec : Array
            Vector to multiply. Shape: (nstates,)

        Returns
        -------
        Array
            M @ vec. Shape: (nstates,)
        """
        ...


@runtime_checkable
class PhysicsWithStateStateHVPProtocol(PhysicsProtocol[Array], Protocol):
    """Physics additionally providing the state-state HVP contraction.

    Required by the HVP-tier ODE-residual adapter: parameterizations own
    the parameter-facing second derivatives (via their ParamDerivatives
    bundle), but lambda^T (d^2f/dy^2) w depends only on the physics.
    """

    def state_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array, time: float
    ) -> Array:
        """Compute lambda^T * (d^2f/du^2) * w.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        adj_state : Array
            Adjoint variable (Lagrange multiplier). Shape: (nstates,)
        wvec : Array
            Direction vector. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            HVP result. Shape: (nstates,)
        """
        ...
