"""Physics protocols for spectral collocation methods.

Defines the protocols for PDE physics:
- PhysicsProtocol - basic residual and Jacobian
- PhysicsWithStateStateHVPProtocol - adds the state-state HVP contraction

Parameter sensitivity (param_jacobian, parameter HVPs) is owned by the
separate ParameterizationProtocol layer via its ParamDerivatives bundle,
not embedded in physics.
"""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.pde.collocation.protocols.basis import BasisProtocol
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.util.backends.protocols import Array, Backend


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

    def apply_boundary_conditions(
        self, residual: Array, jacobian: Array, state: Array
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


@runtime_checkable
class ParameterizationProtocol(Protocol, Generic[Array]):
    """Minimal interface for physics parameterizations.

    Maps a parameter vector to physics inputs. Implementations live in
    pde.parameterizations; this protocol is defined here so collocation
    can depend on the interface without importing the implementation
    module. Optional derivative capability is expressed through the
    :class:`ParamDerivatives` bundle — absence of a capability is a
    ``None`` field, never a missing attribute.
    """

    def nparams(self) -> int: ...

    def apply(self, physics: object, params_1d: Array) -> None: ...

    def param_derivatives(self) -> ParamDerivatives[Array]: ...
